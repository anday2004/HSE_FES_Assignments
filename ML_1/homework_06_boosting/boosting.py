from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

from typing import Optional
from math import floor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])

def quantize_features(X, quantization_type, nbins):
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X_quantized = X.copy()
    for i in range(X.shape[1]):
        if quantization_type == 'Uniform':
            min_val, max_val = np.min(X[:, i]), np.max(X[:, i])
            bins = np.linspace(min_val, max_val, nbins + 1)
        elif quantization_type == 'Quantile':
            bins = np.quantile(X[:, i], np.linspace(0, 1, nbins + 1))
        X_quantized[:, i] = np.digitize(X[:, i], bins) - 1
    return X_quantized


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = None,
        subsample: float | int = 0.3,
        bagging_temperature: float | int = 1.0,
        bootstrap_type: str = None,
        goss: bool | None = False,
        goss_k: float | int = 0.2,
        rsm: float | int = 1.0,
        quantization_type: str | None = None,
        nbins: int = 255,
        dart: bool = False,
        dropout_rate: float = 0.05
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.train_predictions = None
        self.val_predictions = None
        self.early_stopping_rounds: int = early_stopping_rounds
        self.subsample: float = subsample
        self.bagging_temperature: float = bagging_temperature
        self.bootstrap_type: str = bootstrap_type
        self.goss: bool = goss
        self.goss_k: float = goss_k
        self.rsm: float = rsm
        self.quantization_type: str = quantization_type
        self.nbins: int = nbins
        self.dart = dart
        self.dropout_rate = dropout_rate
        
        
    def partial_fit(self, X, y):
        sample_weights = None
        
        if self.bootstrap_type == 'Bernoulli':
            sample_mask = np.random.rand(X.shape[0]) < self.subsample
            X_sample, y_sample = X[sample_mask], y[sample_mask]
            train_preds_sample = self.train_predictions[sample_mask]
        elif self.bootstrap_type == 'Bayesian':
            weights = (-np.log(np.random.uniform(0, 1, size=X.shape[0]))) ** self.bagging_temperature
            weights /= weights.sum()
            X_sample, y_sample, train_preds_sample = X, y, self.train_predictions
            sample_weights = weights
        else:
            X_sample, y_sample = X, y
            train_preds_sample = self.train_predictions

        shift = -self.loss_derivative(y_sample, train_preds_sample)

        if self.goss:
            n_samples = X_sample.shape[0]
            k = int(n_samples * self.goss_k)
            big_grad_indices = np.argsort(np.abs(shift))[-k:]

            small_grad_indices = np.setdiff1d(np.arange(n_samples), big_grad_indices)
            small_sample_size = int(len(small_grad_indices) * self.subsample)
            small_grad_indices = np.random.choice(
                small_grad_indices, size=small_sample_size, replace=False
            )

            selected_indices = np.concatenate([big_grad_indices, small_grad_indices])

            small_grad_factor = len(small_grad_indices) / small_sample_size
            goss_weights = np.ones_like(shift)
            goss_weights[small_grad_indices] *= small_grad_factor

            X_sample, y_sample = X_sample[selected_indices], y_sample[selected_indices]
            shift = shift[selected_indices]
            if self.bootstrap_type == 'Bayesian':
                sample_weights = sample_weights[selected_indices] * goss_weights[selected_indices]
            else:
                sample_weights = goss_weights[selected_indices]

        model = self.base_model_class(**self.base_model_params)
        if sample_weights is not None:
            model.fit(X_sample, shift, sample_weight=sample_weights)
        else:
            model.fit(X_sample, shift)

        model_predictions = model.predict(X)
        if self.dart and len(self.models) > 0:
            if int(len(self.models) * self.dropout_rate) >= 1:
                k = floor(int(len(self.models) * self.dropout_rate))
            else:
                k = 0            
            drop_indices = np.random.choice(len(self.models), size=k, replace=False)
            
            scaling_factor = len(self.models) / (len(self.models) - k)
            for i in drop_indices:
                self.gammas[i] *= scaling_factor

            model_predictions *= scaling_factor
        
        self.gammas.append(self.find_optimal_gamma(y, self.train_predictions, model_predictions))
        self.models.append(model)

    
    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        if self.quantization_type is not None:
            X_train = quantize_features(X_train, self.quantization_type, self.nbins)
            if X_val is not None:
                X_val = quantize_features(X_val, self.quantization_type, self.nbins)
        
        self.train_predictions = np.zeros(y_train.shape[0])
        best_valid_loss = 1234567890
        curr_valid_roc_auc = -1
        no_improve_rounds = 0
        self.val_predictions = np.zeros(y_val.shape[0])

        for _ in range(self.n_estimators):
            self.partial_fit(X_train, y_train)

            self.train_predictions += self.models[-1].predict(X_train) * self.learning_rate * self.gammas[-1]
            train_loss = self.loss_fn(self.train_predictions, y_train)
            train_roc_auc = roc_auc_score(y_train, self.train_predictions)
            self.history['train_loss'].append(train_loss)
            self.history['train_roc_auc'].append(train_roc_auc)

            if X_val is not None and y_val is not None:
                self.val_predictions += self.models[-1].predict(X_val) * self.learning_rate * self.gammas[-1]
                val_loss = self.loss_fn(self.val_predictions, y_val)
                val_roc_auc = roc_auc_score(y_val, self.val_predictions)
                self.history['val_loss'].append(val_loss)
                self.history['val_roc_auc'].append(val_roc_auc)
            else:
                val_loss = None
                
            if self.early_stopping_rounds is not None:
                
                if val_roc_auc > curr_valid_roc_auc:
                    curr_valid_roc_auc = val_roc_auc
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                    curr_valid_roc_auc = val_roc_auc

                if no_improve_rounds == self.early_stopping_rounds:
                    break

        if plot:
            self.plot_history(X_val if X_val is not None else X_train, 
                              y_val if y_val is not None else y_train)
            
        self.compute_feature_importances(X_train)


    def predict_proba(self, x):
        if self.quantization_type:
            x = quantize_features(x, self.quantization_type, self.nbins)
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += model.predict(x) * self.learning_rate * gamma

        predictions = self.sigmoid(predictions)
        probs = np.zeros([x.shape[0], 2])
        probs[:, 0], probs[:, 1] = 1 - predictions, predictions
        return probs


    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, loss='loss', data_type='train'):
        history_key = f'{data_type}_{loss}'

        values = self.history.get(history_key, [])

        if not values:
            print(f"No values to plot for {history_key}")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(values, label=f'{data_type.capitalize()} {loss.capitalize()}', color='blue' if data_type == 'train' else 'red')
        plt.xlabel('Number of Estimators')
        plt.ylabel(loss.capitalize())
        plt.title(f'{loss.capitalize()} vs Number of Estimators ({data_type.capitalize()})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        
    def compute_feature_importances(self, X_train):
        n_features = X_train.shape[1]

        feature_importances = np.zeros(n_features)

        for gamma, model in zip(self.gammas, self.models):
            if hasattr(model, 'feature_importances_'):
                feature_importances += model.feature_importances_ * gamma

        feature_importances /= feature_importances.sum()
        self.feature_importances_ = feature_importances
        return self.feature_importances_
