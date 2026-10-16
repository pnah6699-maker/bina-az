"""Tests for the from-scratch Pegasos SVM and the RFF extension."""

import numpy as np
from sklearn.linear_model import SGDClassifier

from src.svm import PegasosSVM, RandomFourierFeatures, RFFPegasosSVM


def _blobs(n=400, gap=2.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-gap / 2, 1, (n // 2, 2)), rng.normal(gap / 2, 1, (n // 2, 2))])
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    return X, y


def test_separable_data_is_classified_perfectly():
    X, y = _blobs(gap=8.0)
    m = PegasosSVM(lambda_=1e-3, epochs=30).fit(X, y)
    assert (m.predict(X) == y).mean() == 1.0


def test_objective_decreases():
    X, y = _blobs()
    m = PegasosSVM(lambda_=1e-2, epochs=30).fit(X, y)
    h = np.array(m.objective_history_)
    assert h[-1] < h[0]
    assert h[-1] <= h[1:].min() + 0.01  # converged near the best value seen


def test_full_batch_objective_is_monotone_with_small_constant_step():
    X, y = _blobs()
    m = PegasosSVM(lambda_=1e-2, epochs=50, batch_size=None, lr_schedule="constant",
                   eta0=0.05, project=False).fit(X, y)
    assert np.all(np.diff(m.objective_history_) <= 1e-9)


def test_label_mapping_roundtrip():
    X, y = _blobs(gap=6.0)
    labels = np.where(y == 1, "premium", "standard")
    m = PegasosSVM(lambda_=1e-3, epochs=10).fit(X, labels)
    assert set(m.predict(X)) <= {"premium", "standard"}
    assert (m.predict(X) == labels).mean() > 0.95


def test_larger_lambda_gives_wider_margin():
    X, y = _blobs()
    widths = [PegasosSVM(lambda_=l, epochs=30).fit(X, y).margin_width for l in (1e-3, 1e-1, 1.0)]
    assert widths[0] < widths[1] < widths[2]


def test_support_vectors_definition():
    X, y = _blobs()
    m = PegasosSVM(lambda_=1e-2, epochs=20).fit(X, y)
    sv = m.support_vector_mask(X, y)
    y_pm = np.where(y == 1, 1, -1)
    np.testing.assert_array_equal(sv, y_pm * m.decision_function(X) <= 1)
    assert 0 < sv.sum() < len(y)


def test_agrees_with_sklearn_hinge_sgd():
    X, y = _blobs(gap=2.5)
    ours = PegasosSVM(lambda_=1e-2, epochs=50, batch_size=1).fit(X, y)
    sk = SGDClassifier(loss="hinge", alpha=1e-2, max_iter=50, tol=None, random_state=0).fit(X, y)
    assert (ours.predict(X) == sk.predict(X)).mean() > 0.97
    cos = ours.w @ sk.coef_[0] / (np.linalg.norm(ours.w) * np.linalg.norm(sk.coef_[0]))
    assert cos > 0.98


def test_rff_approximates_rbf_kernel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    gamma = 0.3
    Z = RandomFourierFeatures(gamma, 20000, random_state=0).fit_transform(X)
    K = np.exp(-gamma * ((X[:, None] - X[None]) ** 2).sum(-1))
    assert np.abs(Z @ Z.T - K).max() < 0.05


def test_rff_svm_solves_xor_that_linear_cannot():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (600, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    lin = PegasosSVM(lambda_=1e-3, epochs=30).fit(X, y)
    rff = RFFPegasosSVM(gamma=2.0, n_components=300, lambda_=1e-4, epochs=40).fit(X, y)
    assert (lin.predict(X) == y).mean() < 0.7
    assert (rff.predict(X) == y).mean() > 0.9


def test_deterministic_given_seed():
    X, y = _blobs()
    a = PegasosSVM(epochs=5, random_state=7).fit(X, y)
    b = PegasosSVM(epochs=5, random_state=7).fit(X, y)
    np.testing.assert_array_equal(a.w, b.w)
