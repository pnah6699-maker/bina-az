"""
ensemble.py — BONUS (Problem 9): a random forest built from OUR OWN
DecisionTree (bootstrap rows + random feature subset at every split).

Regression: average of tree predictions.
Classification: average of tree leaf-probabilities (soft voting).
"""

from __future__ import annotations

import numpy as np

from .decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 50,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features="sqrt",
        criterion: str | None = None,
        bootstrap: bool = True,
        random_state: int = 42,
    ):
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        self.trees_ = []
        if self.task == "classification":
            self.classes_ = np.unique(y)
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            tree = DecisionTree(
                task=self.task, criterion=self.criterion, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).fit(X[idx], y[idx])
            self.trees_.append(tree)
        imps = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        self.feature_importances_ = imps / imps.sum()
        return self

    def _tree_outputs(self, X):
        if self.task == "regression":
            return np.array([t.predict(X) for t in self.trees_])
        # Align each tree's class columns to the forest's classes.
        out = np.zeros((len(self.trees_), len(X), len(self.classes_)))
        for k, t in enumerate(self.trees_):
            cols = np.searchsorted(self.classes_, t.classes_)
            out[k][:, cols] = t.predict_proba(X)
        return out

    def staged_predict(self, X):
        """Prediction after 1, 2, ..., n_estimators trees (for the curve)."""
        outs = self._tree_outputs(X)
        cum = np.cumsum(outs, axis=0) / np.arange(1, len(outs) + 1).reshape(-1, *[1] * (outs.ndim - 1))
        for c in cum:
            yield c if self.task == "regression" else self.classes_[np.argmax(c, axis=1)]

    def predict_proba(self, X):
        return self._tree_outputs(X).mean(axis=0)

    def predict(self, X):
        if self.task == "regression":
            return self._tree_outputs(X).mean(axis=0)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
