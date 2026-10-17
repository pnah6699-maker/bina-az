"""Correctness tests for the decision tree, using scikit-learn as a reference."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.decision_tree import DecisionTree


def _sk_structure(sk):
    t = sk.tree_
    feats = np.where(t.feature < 0, -1, t.feature)
    thr = np.where(t.feature < 0, np.nan, t.threshold)
    return feats, thr


def _our_structure(tree):
    return tree._feat, tree._thr


# A tiny, fully-determined example: every split has a unique best gain.
X_TOY = np.array([
    [1.0, 7.0], [2.0, 3.0], [3.0, 8.0], [4.0, 1.0],
    [5.0, 6.0], [6.0, 2.0], [7.0, 9.0], [8.0, 4.0],
])
Y_CLS = np.array([0, 0, 0, 1, 1, 1, 1, 0])
Y_REG = np.array([1.0, 1.2, 0.9, 3.0, 3.1, 2.8, 5.0, 4.0])


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_toy_classification_matches_sklearn_splits(criterion):
    ours = DecisionTree("classification", criterion).fit(X_TOY, Y_CLS)
    sk = DecisionTreeClassifier(criterion=criterion, random_state=0).fit(X_TOY, Y_CLS)
    f1, t1 = _our_structure(ours)
    f2, t2 = _sk_structure(sk)
    assert np.array_equal(f1, f2)
    np.testing.assert_allclose(t1, t2, equal_nan=True)
    np.testing.assert_array_equal(ours.predict(X_TOY), sk.predict(X_TOY))
    np.testing.assert_allclose(ours.predict_proba(X_TOY), sk.predict_proba(X_TOY))


def test_toy_regression_matches_sklearn_splits():
    ours = DecisionTree("regression", "mse", max_depth=2).fit(X_TOY, Y_REG)
    sk = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_TOY, Y_REG)
    f1, t1 = _our_structure(ours)
    f2, t2 = _sk_structure(sk)
    assert np.array_equal(f1, f2)
    np.testing.assert_allclose(t1, t2, equal_nan=True)
    np.testing.assert_allclose(ours.predict(X_TOY), sk.predict(X_TOY))


@pytest.mark.parametrize("seed", range(5))
def test_random_continuous_data_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] * 2 + np.sin(3 * X[:, 1]) + 0.1 * rng.normal(size=200)
    ours = DecisionTree("regression", max_depth=5, min_samples_leaf=3).fit(X, y)
    sk = DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, random_state=0).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), sk.predict(X), atol=1e-10)
    # Classification gains are ratios of small integer counts, so exact ties
    # between different splits happen; sklearn breaks them via a random
    # feature permutation, we take the first. Hence agreement, not identity.
    yc = (y > np.median(y)).astype(int)
    ours = DecisionTree("classification", "gini", max_depth=4).fit(X, yc)
    sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, yc)
    assert (ours.predict(X) == sk.predict(X)).mean() >= 0.97
    assert ours.root.feature == sk.tree_.feature[0]


def test_impurities_by_hand():
    t = DecisionTree("classification", "gini").fit(X_TOY, Y_CLS)
    assert t._impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    t = DecisionTree("classification", "entropy").fit(X_TOY, Y_CLS)
    assert t._impurity(np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert t._impurity(np.array([1, 1, 1])) == pytest.approx(0.0)
    t = DecisionTree("regression").fit(X_TOY, Y_REG)
    assert t._impurity(np.array([1.0, 3.0])) == pytest.approx(1.0)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + X[:, 1] ** 2 + 0.3 * rng.normal(size=n)
    return X, y


@pytest.mark.parametrize("param,small,large", [
    ("max_depth", 2, 8),
    ("min_samples_split", 200, 2),
    ("min_samples_leaf", 50, 1),
    ("min_impurity_decrease", 0.01, 0.0),
])
def test_every_hyperparameter_changes_the_tree(param, small, large):
    X, y = _data()
    a = DecisionTree("regression", **{param: small}).fit(X, y)
    b = DecisionTree("regression", **{param: large}).fit(X, y)
    assert a.n_leaves_ < b.n_leaves_


def test_min_samples_leaf_respected():
    X, y = _data()
    t = DecisionTree("regression", min_samples_leaf=17).fit(X, y)
    leaves = t.apply(X)
    assert np.bincount(leaves)[np.unique(leaves)].min() >= 17


def test_max_depth_respected():
    X, y = _data()
    assert DecisionTree("regression", max_depth=3).fit(X, y).depth_ <= 3


def test_vectorised_predict_equals_recursive_traversal():
    X, y = _data()
    t = DecisionTree("regression", max_depth=6).fit(X, y)
    slow = np.array([t._predict_one(x, t.root) for x in X])
    np.testing.assert_allclose(t.predict(X), slow)


def test_predict_proba_rows_sum_to_one_and_labels_preserved():
    X, y = _data()
    labels = np.where(y > 0, "premium", "standard")
    t = DecisionTree("classification", "entropy", max_depth=4).fit(X, labels)
    p = t.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert set(t.predict(X)) <= {"premium", "standard"}


def test_feature_importances_match_sklearn():
    X, y = _data()
    ours = DecisionTree("regression", max_depth=4).fit(X, y)
    sk = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    np.testing.assert_allclose(ours.feature_importances_, sk.feature_importances_, atol=1e-10)


def test_pure_node_is_a_leaf():
    t = DecisionTree("classification").fit(np.arange(10).reshape(-1, 1), np.ones(10, int))
    assert t.root.is_leaf() and t.n_leaves_ == 1
