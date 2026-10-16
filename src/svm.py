"""
Linear SVM implemented from scratch using Pegasos and NumPy.

Also includes Random Fourier Features (RFF) for non-linear classification.

Pegasos trains the SVM using mini-batch gradient updates.
RFF allows the linear SVM to learn non-linear patterns.

Scikit-learn is not used in these implementations.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 32,
        lr_schedule: str = "pegasos",
        eta0: float = 0.1,
        project: bool = True,
        fit_intercept: bool = True,
        track_objective: bool = True,
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.eta0 = eta0
        self.project = project
        self.fit_intercept = fit_intercept
        self.track_objective = track_objective
        self.random_state = random_state
        self.w = None
        self.b = 0.0

    def get_params(self) -> dict:
        return {k: getattr(self, k) for k in (
            "lambda_", "epochs", "batch_size", "lr_schedule", "eta0",
            "project", "fit_intercept", "random_state")}

    def _to_pm1(self, y) -> np.ndarray:
        """Map two class labels to {-1, +1}; remember the mapping."""
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("PegasosSVM is a binary classifier")
        return np.where(y == self.classes_[1], 1.0, -1.0)

    def _augment(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = np.hstack([X, np.ones((len(X), 1))])
        return X

    def _eta(self, t: int) -> float:
        if self.lr_schedule == "pegasos":
            return 1.0 / (self.lambda_ * t)
        if self.lr_schedule == "invsqrt":
            return self.eta0 / np.sqrt(t)
        if self.lr_schedule == "constant":
            return self.eta0
        raise ValueError(f"unknown lr_schedule {self.lr_schedule!r}")

    def objective(self, X, y_pm) -> float:
        """J(w,b) on (already augmented) X with +/-1 labels."""
        margins = y_pm * (X @ self._wb)
        hinge = np.maximum(0.0, 1.0 - margins).mean()
        return 0.5 * self.lambda_ * float(self._wb @ self._wb) + float(hinge)
    
    def fit(self, X, y):
        Xa = self._augment(X)
        y_pm = self._to_pm1(y)
        n, d = Xa.shape
        k = n if self.batch_size is None else min(self.batch_size, n)
        rng = np.random.default_rng(self.random_state)
        self._wb = np.zeros(d)
        radius = 1.0 / np.sqrt(self.lambda_)

        self.objective_history_ = []
        if self.track_objective:
            self.objective_history_.append(self.objective(Xa, y_pm))

        t = 0
        for _ in range(self.epochs):
            perm = rng.permutation(n)          
            for start in range(0, n, k):
                t += 1
                idx = perm[start:start + k]
                Xb, yb = Xa[idx], y_pm[idx]
                eta = self._eta(t)
                active = yb * (Xb @ self._wb) < 1.0          
                grad_hinge = (yb[active, None] * Xb[active]).sum(axis=0) / len(idx)
                self._wb *= (1.0 - eta * self.lambda_)
                self._wb += eta * grad_hinge
                if self.project:
                    norm = np.linalg.norm(self._wb)
                    if norm > radius:
                        self._wb *= radius / norm
            if self.track_objective:
                self.objective_history_.append(self.objective(Xa, y_pm))

        self.n_steps_ = t
        if self.fit_intercept:
            self.w, self.b = self._wb[:-1].copy(), float(self._wb[-1])
        else:
            self.w, self.b = self._wb.copy(), 0.0
        return self

    
    def decision_function(self, X) -> np.ndarray:
        """Signed score  w . x + b  for each row."""
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    
    @property
    def w_norm(self) -> float:
        return float(np.linalg.norm(self.w))

    @property
    def margin_width(self) -> float:
        """Geometric width of the margin band, 2 / ||w||."""
        return 2.0 / self.w_norm if self.w_norm > 0 else np.inf

    def support_vector_mask(self, X, y) -> np.ndarray:
        """Training points on or inside the margin: y_i (w . x_i + b) <= 1."""
        y_pm = np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)
        return y_pm * self.decision_function(X) <= 1.0


class RandomFourierFeatures:
    """Explicit map whose inner product approximates the RBF kernel."""

    def __init__(self, gamma: float = 0.05, n_components: int = 500, random_state: int = 42):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X):
        d = np.asarray(X).shape[1]
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(d, self.n_components))
        self.c_ = rng.uniform(0.0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X) -> np.ndarray:
        return np.sqrt(2.0 / self.n_components) * np.cos(np.asarray(X, dtype=float) @ self.W_ + self.c_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RFFPegasosSVM:
    """Non-linear SVM: RandomFourierFeatures -> linear PegasosSVM."""

    def __init__(self, gamma: float = 0.05, n_components: int = 500, **svm_kwargs):
        self.gamma = gamma
        self.n_components = n_components
        self.svm_kwargs = svm_kwargs
        rs = svm_kwargs.get("random_state", 42)
        self.rff = RandomFourierFeatures(gamma, n_components, random_state=rs)
        self.svm = PegasosSVM(**svm_kwargs)

    def get_params(self) -> dict:
        return {"gamma": self.gamma, "n_components": self.n_components, **self.svm.get_params()}

    def fit(self, X, y):
        self.svm.fit(self.rff.fit_transform(X), y)
        self.classes_ = self.svm.classes_
        self.objective_history_ = self.svm.objective_history_
        return self

    def decision_function(self, X):
        return self.svm.decision_function(self.rff.transform(X))

    def predict(self, X):
        return self.svm.predict(self.rff.transform(X))
