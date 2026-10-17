"""Our NumPy metrics vs scikit-learn's (sanity check only)."""

import numpy as np
import pytest
from sklearn import metrics as skm

from src import evaluate as ev


@pytest.fixture
def cls_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = y + rng.normal(0, 1, 500)
    s[:50] = np.round(s[:50])  # create ties
    return y, s, (s > 0.5).astype(int)


def test_regression_metrics():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=100), rng.normal(size=100)
    assert ev.rmse(a, b) == pytest.approx(np.sqrt(skm.mean_squared_error(a, b)))
    assert ev.mae(a, b) == pytest.approx(skm.mean_absolute_error(a, b))
    assert ev.r2(a, b) == pytest.approx(skm.r2_score(a, b))


def test_confusion_and_prf(cls_data):
    y, _, p = cls_data
    np.testing.assert_array_equal(ev.confusion_matrix(y, p), skm.confusion_matrix(y, p))
    pr, rc, f1 = ev.precision_recall_f1(y, p)
    assert pr == pytest.approx(skm.precision_score(y, p))
    assert rc == pytest.approx(skm.recall_score(y, p))
    assert f1 == pytest.approx(skm.f1_score(y, p))


def test_auc_and_ap(cls_data):
    y, s, _ = cls_data
    assert ev.roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))
    assert ev.average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s))
    fpr, tpr = ev.roc_curve(y, s)
    assert np.trapezoid(tpr, fpr) == pytest.approx(skm.roc_auc_score(y, s))
