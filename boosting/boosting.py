from __future__ import annotations

from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from typing import Optional


def score(clf, X, y):
    return roc_auc_score(y == 1, clf.predict_proba(X)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int | None = 0,
        subsample: float | int = 1.0,
        bagging_temperature: float | int = 1.0,
        bootstrap_type: str | None = "Bernoulli",
        rsm: float | int = 1.0,
        quantization_type: str | None = None,
        nbins: int = 255,
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.early_stopping_rounds: int | None = early_stopping_rounds

        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins

        self.models: list = []
        self.gammas: list = []

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.feature_importances_ = None

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def quantize_features(self, X):
        if self.quantization_type is None:
            return X

        X_quantized = np.copy(X)
        for i in range(X.shape[1]):
            feature = X[:, i]
            if self.quantization_type == "Uniform":
                min_val, max_val = np.min(feature), np.max(feature)
                bins = np.linspace(min_val, max_val, self.nbins + 1)
                X_quantized[:, i] = np.digitize(feature, bins, right=False)
            elif self.quantization_type == "Quantile":
                bins = np.quantile(feature, q=np.linspace(0, 1, self.nbins + 1))
                X_quantized[:, i] = np.digitize(feature, bins, right=False)

        return X_quantized

    def select_random_features(self, X):
        n_features = X.shape[1]
        if isinstance(self.rsm, float):
            n_selected = int(n_features * self.rsm)
        else:
            n_selected = min(self.rsm, n_features)

        selected_indices = np.random.choice(n_features, size=n_selected, replace=False)
        return X[:, selected_indices], selected_indices

    def create_bootstrap_sample(self, X, y):
        n_samples = len(y)
        if self.bootstrap_type == "Bernoulli":
            if isinstance(self.subsample, float):
                indices = np.random.choice(
                    n_samples, size=int(n_samples * self.subsample), replace=False
                )
            else:
                indices = np.random.choice(n_samples, size=self.subsample, replace=False)
            return X[indices], y[indices], indices
        elif self.bootstrap_type == "Bayesian":
            weights = (-np.log(np.random.uniform(size=n_samples))) ** self.bagging_temperature
            return X, y, weights
        else:
            return X, y, np.arange(n_samples)

    def partial_fit(self, X, y, train_predictions):
        X = self.quantize_features(X)
        X, selected_indices = self.select_random_features(X)

        if self.bootstrap_type == "Bayesian":
            X, y, weights = self.create_bootstrap_sample(X, y)
            train_predictions_subset = train_predictions[weights.nonzero()[0]]
            residuals = y - self.sigmoid(train_predictions_subset)
            model = self.base_model_class(**self.base_model_params)
            model.fit(X, residuals, sample_weight=weights)
        else:
            X, y, indices = self.create_bootstrap_sample(X, y)
            train_predictions_subset = train_predictions[indices]
            residuals = y - self.sigmoid(train_predictions_subset)
            model = self.base_model_class(**self.base_model_params)
            model.fit(X, residuals)

        new_predictions = model.predict(X)
        gamma = self.find_optimal_gamma(y, train_predictions_subset, new_predictions)
        train_predictions_subset += self.learning_rate * gamma * new_predictions

        if self.bootstrap_type == "Bayesian":
            train_predictions[weights.nonzero()[0]] = train_predictions_subset
        else:
            train_predictions[indices] = train_predictions_subset

        self.models.append((model, selected_indices))
        self.gammas.append(gamma)
        return train_predictions

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        train_predictions = np.zeros(y_train.shape[0])
        best_val_loss = np.inf
        no_improvement_rounds = 0

        for i in range(self.n_estimators):
            train_predictions = self.partial_fit(X_train, y_train, train_predictions)
            train_loss = self.loss_fn(y_train, train_predictions)
            train_roc_auc = score(self, X_train, y_train)
            self.history['train_loss'].append(train_loss)
            self.history['train_roc_auc'].append(train_roc_auc)

            if X_val is not None and y_val is not None:
                val_predictions = self.predict_proba(X_val)[:, 1]
                val_loss = self.loss_fn(y_val, val_predictions)
                val_roc_auc = score(self, X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_roc_auc'].append(val_roc_auc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_rounds = 0
                else:
                    no_improvement_rounds += 1

                if self.early_stopping_rounds and no_improvement_rounds >= self.early_stopping_rounds:
                    print(f"Early stopping at iteration {i + 1}")
                    break

        self.compute_feature_importances(X_train.shape[1])
        if plot:
            self.plot_history(X_train, y_train)

    def predict_proba(self, X):
        predictions = np.zeros(X.shape[0])
        for (model, selected_indices), gamma in zip(self.models, self.gammas):
            X_subset = X[:, selected_indices]
            predictions += self.learning_rate * gamma * model.predict(X_subset)
        probabilities = self.sigmoid(predictions)
        return np.column_stack((1 - probabilities, probabilities))

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def compute_feature_importances(self, n_features):
        importances = np.zeros(n_features)
        for (model, selected_indices) in self.models:
            model_importances = model.feature_importances_
            for i, idx in enumerate(selected_indices):
                importances[idx] += model_importances[i]
        self.feature_importances_ = importances / importances.sum()

    def score(self, X, y):
        return score(self, X, y)

    def plot_history(self, X, y):
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss History')
        plt.show()

