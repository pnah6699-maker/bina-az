"""
CART decision tree implemented in NumPy.

Supports both tasks:
  - classification (criterion = "gini" | "entropy")  -> price tier
  - regression     (criterion = "mse")               -> (log) price

Impurities (N samples in a node, p_k = class-k fraction, ybar = node mean):
    Gini     G   = 1 - sum_k p_k^2
    Entropy  H   = - sum_k p_k log2 p_k
    MSE      I   = (1/N) sum_i (y_i - ybar)^2
    Gain     IG  = I(parent) - sum_c (N_c / N) I(child_c)

Split search
------------
Every feature is sorted once at the root; each child inherits its rows in
sorted order by filtering the parent's order (no re-sorting, O(N d) per
level). For each feature we then sweep all N-1 cut positions in a single
vectorised pass using cumulative sums:
  * classification: cumulative class counts  -> left/right p_k at every cut
  * regression    : cumulative sum(y), sum(y^2) -> left/right SSE at every cut
Only cuts between two different feature values are valid, and the threshold
is the midpoint of the two neighbouring values (same convention as sklearn).
A sample goes left iff x[feature] <= threshold.

Tie-breaking: when several (feature, threshold) pairs have exactly the same
gain we keep the first one found (lowest feature index among the features
scanned, then lowest threshold). sklearn instead visits features in a
random permutation, which is the main source of structural differences.
"""

from __future__ import annotations

import numpy as np


def _binary_entropy(p):
    """H(p) = -p log2 p - (1-p) log2 (1-p), with 0 log 0 := 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nan_to_num(p * np.log2(p)) - np.nan_to_num((1 - p) * np.log2(1 - p))


class Node:
    """One node of the tree (internal node or leaf)."""

    __slots__ = ("feature", "threshold", "left", "right", "value",
                 "n_samples", "impurity", "depth")

    def __init__(self, value, n_samples, impurity, depth):
        self.feature: int | None = None      # split feature index (internal)
        self.threshold: float | None = None  # split threshold (internal)
        self.left: Node | None = None
        self.right: Node | None = None
        self.value = value                   # class-probability vector or mean
        self.n_samples = n_samples
        self.impurity = impurity
        self.depth = depth

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    """
    Parameters mirror sklearn's DecisionTreeClassifier / Regressor.

    max_depth              : stop splitting at this depth (None = unlimited)
    min_samples_split      : a node needs at least this many samples to split
    min_samples_leaf       : every child must keep at least this many samples
    min_impurity_decrease  : split only if  N_t/N * (I_t - weighted child I)
                             >= this value (sklearn's definition)
    max_features           : features examined per split (None = all,
                             "sqrt", "log2", int or float fraction). Used by
                             the random-forest bonus; with None the tree is
                             fully deterministic.
    """

    def __init__(
        self,
        task: str = "classification",
        criterion: str | None = None,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features=None,
        random_state: int = 42,
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if criterion is None:
            criterion = "gini" if task == "classification" else "mse"
        valid = {"classification": ("gini", "entropy"), "regression": ("mse",)}
        if criterion not in valid[task]:
            raise ValueError(f"criterion {criterion!r} invalid for {task}")
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.root: Node | None = None

    def get_params(self) -> dict:
        return {k: getattr(self, k) for k in (
            "task", "criterion", "max_depth", "min_samples_split",
            "min_samples_leaf", "min_impurity_decrease", "max_features",
            "random_state")}

    def _n_features_to_try(self, d: int) -> int:
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return min(d, int(mf))

    def _impurity_from_counts(self, counts: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Vectorised Gini / entropy from class counts (..., K) and totals (...)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            p = counts / n[..., None]
            if self.criterion == "gini":
                imp = 1.0 - np.sum(p * p, axis=-1)
            else:  # entropy, with 0 log 0 := 0
                logp = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
                imp = -np.sum(p * logp, axis=-1)
        return np.where(n > 0, imp, 0.0)

    def _impurity(self, y) -> float:
        """Impurity of a single node (y = encoded labels or targets)."""
        y = np.asarray(y)
        if len(y) == 0:
            return 0.0
        if self.task == "regression":
            return float(np.mean((y - y.mean()) ** 2))
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        return float(self._impurity_from_counts(counts, np.array(len(y), dtype=float)))

    def _leaf_value(self, y):
        if self.task == "regression":
            return float(np.mean(y))
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        return counts / counts.sum()

    def _best_split(self, X, y, order_all, parent_imp: float):
        """
        Return (feature, threshold, weighted_gain, child_impurities) of the
        best split, or None if no valid split exists.
        `weighted_gain` = N_t/N_total * IG, i.e. sklearn's impurity decrease.
        """
        n, d = X.shape
        msl = self.min_samples_leaf
        if n < 2 * msl:
            return None

        feats = np.arange(d)
        k = self._n_features_to_try(d)
        if k < d:
            feats = np.sort(self._rng.choice(d, size=k, replace=False))

        order = order_all[:, feats] if k < d else order_all        # (n, k)
        xs = np.take_along_axis(X[:, feats] if k < d else X, order, axis=0)

        # Cut after position i (0-based) -> left has i+1 samples.
        n_left = np.arange(1, n, dtype=float)                      # (n-1,)
        n_right = n - n_left
        valid = xs[1:] > xs[:-1]                                   # distinct values
        size_ok = (n_left >= msl) & (n_right >= msl)
        valid &= size_ok[:, None]
        if not valid.any():
            return None

        if self.task == "regression":
            yc = y - y.mean()                  # centre: avoids cancellation in SSE
            ys = yc[order]                                         # (n, k)
            cs = np.cumsum(ys, axis=0)[:-1]
            cs2 = np.cumsum(ys * ys, axis=0)[:-1]
            tot, tot2 = yc.sum(), (yc * yc).sum()
            sse_l = cs2 - cs * cs / n_left[:, None]
            sse_r = (tot2 - cs2) - (tot - cs) ** 2 / n_right[:, None]
            imp_l = sse_l / n_left[:, None]
            imp_r = sse_r / n_right[:, None]
            child = (sse_l + sse_r) / n
        elif self.n_classes_ == 2:
            # Binary fast path: only the positive-class count is needed.
            c1 = np.cumsum(y[order], axis=0)[:-1].astype(float)    # (n-1, k)
            t1 = float(y.sum())
            pl = c1 / n_left[:, None]
            pr = (t1 - c1) / n_right[:, None]
            if self.criterion == "gini":
                imp_l, imp_r = 2 * pl * (1 - pl), 2 * pr * (1 - pr)
            else:
                imp_l, imp_r = _binary_entropy(pl), _binary_entropy(pr)
            child = (n_left[:, None] * imp_l + n_right[:, None] * imp_r) / n
        else:
            onehot = np.eye(self.n_classes_)[y]                    # (n, K)
            counts = np.cumsum(onehot[order], axis=0)[:-1]         # (n-1, k, K)
            total = onehot.sum(axis=0)
            imp_l = self._impurity_from_counts(counts, np.broadcast_to(n_left[:, None], counts.shape[:2]))
            imp_r = self._impurity_from_counts(total - counts, np.broadcast_to(n_right[:, None], counts.shape[:2]))
            child = (n_left[:, None] * imp_l + n_right[:, None] * imp_r) / n

        gain = np.where(valid, parent_imp - child, -np.inf)
        # argmax over the transposed array -> feature-major order, so ties are
        # broken by lowest feature index, then lowest threshold.
        flat = int(np.argmax(gain.T))
        j, i = divmod(flat, n - 1)
        best = gain[i, j]
        if not np.isfinite(best):
            return None
        thr = (xs[i, j] + xs[i + 1, j]) / 2.0
        if thr == xs[i + 1, j]:          # float midpoint rounding up
            thr = xs[i, j]
        weighted = n / self._n_total * best
        return int(feats[j]), float(thr), float(weighted), (float(imp_l[i, j]), float(imp_r[i, j]))

    @staticmethod
    def _partition_order(order, mask):
        """
        Split the per-feature sorted index matrix between the two children
        without re-sorting: keep each column's rows that go left (right), in
        the same order, then renumber them to the child's local indices. O(n d).
        """
        n_l = int(mask.sum())
        new_idx = np.where(mask, np.cumsum(mask) - 1, np.cumsum(~mask) - 1)
        goes_left = mask[order].T                                    # (d, n)
        oT = order.T
        left = new_idx[oT[goes_left].reshape(order.shape[1], n_l)].T
        right = new_idx[oT[~goes_left].reshape(order.shape[1], len(mask) - n_l)].T
        return np.ascontiguousarray(left), np.ascontiguousarray(right)

    def _grow(self, X, y, order, depth: int, impurity: float) -> Node:
        node = Node(self._leaf_value(y), len(y), impurity, depth)
        self.n_nodes_ += 1
        self.depth_ = max(self.depth_, depth)

        # Stopping rules.
        if (self.max_depth is not None and depth >= self.max_depth) \
                or len(y) < self.min_samples_split or impurity <= 1e-12:
            self.n_leaves_ += 1
            return node

        split = self._best_split(X, y, order, impurity)
        if split is None or split[2] < self.min_impurity_decrease or split[2] <= 0:
            self.n_leaves_ += 1
            return node

        f, thr, wgain, (imp_l, imp_r) = split
        node.feature, node.threshold = f, thr
        self._importances[f] += wgain
        mask = X[:, f] <= thr
        o_l, o_r = self._partition_order(order, mask)
        node.left = self._grow(X[mask], y[mask], o_l, depth + 1, imp_l)
        node.right = self._grow(X[~mask], y[~mask], o_r, depth + 1, imp_r)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "classification":
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
        else:
            y = y.astype(float)
        self.n_features_in_ = X.shape[1]
        self._n_total = len(y)
        self._rng = np.random.default_rng(self.random_state)
        self._importances = np.zeros(X.shape[1])
        self.n_nodes_ = self.n_leaves_ = self.depth_ = 0
        order = np.argsort(X, axis=0, kind="stable")               # pre-sort once
        self.root = self._grow(X, y, order, depth=0, impurity=self._impurity(y))
        s = self._importances.sum()
        self.feature_importances_ = self._importances / s if s > 0 else self._importances
        self._flatten()
        return self

    def _flatten(self):
        """Store the tree as parallel arrays for fast vectorised prediction."""
        feats, thrs, lefts, rights, vals = [], [], [], [], []

        def visit(node: Node) -> int:
            idx = len(feats)
            feats.append(-1 if node.is_leaf() else node.feature)
            thrs.append(np.nan if node.is_leaf() else node.threshold)
            lefts.append(-1); rights.append(-1); vals.append(node.value)
            if not node.is_leaf():
                lefts[idx] = visit(node.left)
                rights[idx] = visit(node.right)
            return idx

        visit(self.root)
        self._feat = np.array(feats)
        self._thr = np.array(thrs)
        self._left = np.array(lefts)
        self._right = np.array(rights)
        self._val = np.array(vals)

    def _predict_one(self, x, node: Node):
        """Reference (slow) traversal of a single sample, used in tests."""
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def apply(self, X) -> np.ndarray:
        """Index (in the flattened arrays) of the leaf each sample lands in."""
        X = np.asarray(X, dtype=float)
        idx = np.zeros(len(X), dtype=int)
        active = self._feat[idx] >= 0
        while active.any():
            a = np.flatnonzero(active)
            node = idx[a]
            go_left = X[a, self._feat[node]] <= self._thr[node]
            idx[a] = np.where(go_left, self._left[node], self._right[node])
            active[a] = self._feat[idx[a]] >= 0
        return idx

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities = class frequencies in the leaf."""
        if self.task != "classification":
            raise AttributeError("predict_proba is only for classification")
        return self._val[self.apply(X)]

    def predict(self, X) -> np.ndarray:
        leaf_vals = self._val[self.apply(X)]
        if self.task == "regression":
            return leaf_vals.astype(float)
        return self.classes_[np.argmax(leaf_vals, axis=1)]

    def export_text(self, feature_names=None, max_depth: int = 3) -> str:
        """Human-readable dump of the top of the tree."""
        names = feature_names or [f"x{i}" for i in range(self.n_features_in_)]
        lines: list[str] = []

        def rec(node: Node, indent: str):
            if node.is_leaf() or node.depth >= max_depth:
                v = node.value
                v = f"{v:.3f}" if self.task == "regression" else np.round(v, 3).tolist()
                lines.append(f"{indent}leaf: value={v} n={node.n_samples}")
                return
            lines.append(f"{indent}{names[node.feature]} <= {node.threshold:.4g}  (n={node.n_samples})")
            rec(node.left, indent + "|   ")
            lines.append(f"{indent}{names[node.feature]} >  {node.threshold:.4g}")
            rec(node.right, indent + "|   ")

        rec(self.root, "")
        return "\n".join(lines)
