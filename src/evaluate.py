"""Evaluation metrics and model comparison helpers implemented in NumPy."""

from __future__ import annotations

import time

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)))


def regression_report(log_true, log_pred) -> dict:
    """Metrics on log(price) and, after exp(), on price in AZN."""
    p_true, p_pred = np.exp(log_true), np.exp(log_pred)
    return {
        "rmse_log": rmse(log_true, log_pred),
        "mae_log": mae(log_true, log_pred),
        "r2_log": r2(log_true, log_pred),
        "rmse_azn": rmse(p_true, p_pred),
        "mae_azn": mae(p_true, p_pred),
        "r2_azn": r2(p_true, p_pred),
        "mape": mape(p_true, p_pred),
    }


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """2x2 matrix [[TN, FP], [FN, TP]] for labels {0, 1}."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def precision_recall_f1(y_true, y_pred):
    """(precision, recall, f1) for the positive (premium = 1) class."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def _average_ranks(x) -> np.ndarray:
    """1-based ranks with ties sharing their average rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true, scores) -> float:
    """Rank-based (Mann-Whitney U) AUC; ties count 1/2."""
    y_true = np.asarray(y_true).astype(int)
    n_pos, n_neg = y_true.sum(), (1 - y_true).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(y_true, scores):
    """(fpr, tpr) arrays, one point per distinct score threshold."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    s, y = np.asarray(scores, float)[order], y_true[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (distinct + 1) - tps
    return np.r_[0.0, fps / fps[-1]], np.r_[0.0, tps / tps[-1]]


def average_precision(y_true, scores) -> float:
    """PR-AUC as average precision: sum_k (R_k - R_{k-1}) P_k."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    s, y = np.asarray(scores, float)[order], y_true[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[distinct]
    precision = tps / (distinct + 1)
    recall = tps / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def classification_report(y_true, y_pred, scores) -> dict:
    p, r, f = precision_recall_f1(y_true, y_pred)
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "precision": p, "recall": r, "f1": f,
        "roc_auc": roc_auc(y_true, scores),
        "pr_auc": average_precision(y_true, scores),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0


def summarize(fold_metrics: list[dict]) -> dict:
    """{metric: (mean, std)} across CV folds (numeric metrics only)."""
    keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float))]
    return {k: (float(np.mean([m[k] for m in fold_metrics])),
                float(np.std([m[k] for m in fold_metrics], ddof=1)))
            for k in keys}


def compare(name_to_metrics: dict, metrics: list[str], title: str = "") -> str:
    """
    Pretty table: rows = models, cols = metrics. Values may be floats or
    (mean, std) tuples. Returns the string (and prints it).
    """
    w = max(len(n) for n in name_to_metrics) + 2
    head = f"{'model':<{w}}" + "".join(f"{m:>20s}" for m in metrics)
    lines = [title, head, "-" * len(head)] if title else [head, "-" * len(head)]
    for name, res in name_to_metrics.items():
        row = f"{name:<{w}}"
        for m in metrics:
            v = res.get(m, float("nan"))
            if isinstance(v, (tuple, list)):
                row += f"{v[0]:>12.4f} ± {v[1]:<5.4f}"
            else:
                row += f"{v:>20.4f}"
        lines.append(row)
    out = "\n".join(lines)
    print(out)
    return out
