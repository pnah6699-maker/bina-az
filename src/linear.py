"""
linear.py — BONUS (Problem 8): Ridge regression from scratch via the normal
equations  w = (X^T X + lambda I)^{-1} X^T y.

The intercept is not penalised: we centre X and y with TRAIN means, solve for
w on the centred data, then b = mean(y) - mean(X) . w.
"""

from __future__ import annotations

import numpy as np


class RidgeRegression:
    def __init__(self, lambda_: float = 1.0):
        self.lambda_ = lambda_

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.x_mean_, self.y_mean_ = X.mean(axis=0), y.mean()
        Xc, yc = X - self.x_mean_, y - self.y_mean_
        A = Xc.T @ Xc + self.lambda_ * np.eye(X.shape[1])
        # solve() instead of an explicit inverse: same maths, numerically safer.
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = float(self.y_mean_ - self.x_mean_ @ self.coef_)
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_
