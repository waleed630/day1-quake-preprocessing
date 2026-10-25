"""Part 2, Task B: an sklearn-style estimator and model factories."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from .transform import build_preprocessor


class GDLinearRegression(BaseEstimator, RegressorMixin):
    """Port your vectorized notebook version here.

    fit(X, y) must set: coef_ (array), intercept_ (float), loss_history_ (list
    of the loss at every iteration) and return self.
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m = X.shape[0]
        w, b = np.zeros(X.shape[1]), 0.0
        self.loss_history_ = []
        for _ in range(self.n_iter):
            err = X @ w + b - y
            self.loss_history_.append(float((err ** 2).sum() / (2 * m)))
            grad_w, grad_b = X.T @ err / m, err.mean()
            w = w - self.lr * grad_w
            b = b - self.lr * grad_b
        self.coef_, self.intercept_ = w, float(b)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def make_regressor(kind: str = "linear", degree: int = 2, alpha: float = 1.0):
    """Return Pipeline([('prep', build_preprocessor(...)), ('model', estimator)]).

    kinds: 'dummy' (mean baseline), 'linear', 'ridge', 'poly' (Ridge on
    polynomial features of the numeric columns). Unknown kind: raise ValueError.
    First update transform.build_preprocessor to accept degree=1 and add
    PolynomialFeatures(degree, include_bias=False) to the numeric branch
    when degree > 1.
    """
    estimators = {
        "dummy": lambda: DummyRegressor(strategy="mean"),
        "linear": LinearRegression,
        "ridge": lambda: Ridge(alpha=alpha),
        "poly": lambda: Ridge(alpha=alpha),
    }
    if kind not in estimators:
        raise ValueError(f"Unknown regressor {kind!r}; choose from {sorted(estimators)}")
    prep = build_preprocessor(degree=degree if kind == "poly" else 1)
    return Pipeline([("prep", prep), ("model", estimators[kind]())])


def make_classifier(kind: str = "logistic", C: float = 1.0, class_weight=None):
    """Return Pipeline([('prep', ...), ('model', estimator)]).

    kinds: 'dummy' (majority class, still needs predict_proba) and 'logistic'.
    """
    estimators = {
        "dummy": lambda: DummyClassifier(strategy="most_frequent"),
        "logistic": lambda: LogisticRegression(C=C, class_weight=class_weight, max_iter=1000),
    }
    if kind not in estimators:
        raise ValueError(f"Unknown classifier {kind!r}; choose from {sorted(estimators)}")
    return Pipeline([("prep", build_preprocessor()), ("model", estimators[kind]())])
