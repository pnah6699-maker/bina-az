"""
experiments.py — Problems 4-6 (+ bonuses 8-10): tuning sweeps, final model
selection on VALIDATION only, benchmark vs scikit-learn with 5-fold CV,
test-set evaluation, and error analysis. Called by run_all.py.

Protocol (no test-set peeking):
  * hyper-parameters are chosen on the validation split only;
  * final models are fit on TRAIN and scored once on TEST;
  * uncertainty = 5-fold CV on train+val ("dev"); inside each fold the
    featurizer, tier threshold and scaler are re-fit on the fold's train part.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import data_prep as dp
from . import evaluate as ev
from .decision_tree import DecisionTree
from .ensemble import RandomForest
from .linear import RidgeRegression
from .plots import C1, C2, C3, C4, C5, INK2, plt, savefig
from .svm import PegasosSVM, RFFPegasosSVM

SEED = dp.SEED


# =============================================================================
# Data container
# =============================================================================
@dataclass
class Split:
    """Arrays for one (fit-part, eval-part) pair, all statistics from fit-part."""
    names: list
    X_a: np.ndarray; X_b: np.ndarray      # raw features [for tree]
    Z_a: np.ndarray; Z_b: np.ndarray      # standardized (fit-part mean/std)  [for SVM and Ridge]
    y_a: np.ndarray; y_b: np.ndarray      # log(price)   [Task A]
    t_a: np.ndarray; t_b: np.ndarray      # tier labels  [Task B]
    thr_log: float                        # tier threshold (log AZN), from fit-part  [traind median]


def make_split(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Split:
    names, X_a, X_b, _ = dp.make_features(df_a, df_b)
    y_a, y_b = dp.make_target(df_a), dp.make_target(df_b)
    t_a, thr = dp.make_tier_label(y_a)            # threshold from fit-part ONLY
    t_b, _ = dp.make_tier_label(y_b, thr)
    Z_a, Z_b = dp.standardize(X_a, X_b)
    return Split(names, X_a, X_b, Z_a, Z_b, y_a, y_b, t_a, t_b, thr)


def _scores(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


# =============================================================================
# Problem 4(a): tree tuning
# =============================================================================
DEPTHS = [2, 4, 6, 8, 10, 12, 14, 17, 20, 25]
LEAVES = [1, 5, 20, 50]


# for every model 
def tune_tree_regression(S: Split, quick=False):
    depths = DEPTHS[::2] if quick else DEPTHS
    leaves = [1, 20] if quick else LEAVES
    rows = []
    for msl in leaves:
        for d in depths:
            with ev.Timer() as t:
                m = DecisionTree("regression", "mse", max_depth=d, min_samples_leaf=msl).fit(S.X_a, S.y_a)
            rows.append(dict(max_depth=d, min_samples_leaf=msl,
                             train_rmse=ev.rmse(S.y_a, m.predict(S.X_a)),
                             val_rmse=ev.rmse(S.y_b, m.predict(S.X_b)),
                             n_leaves=m.n_leaves_, fit_s=t.elapsed))
    res = pd.DataFrame(rows)
    best = res.loc[res["val_rmse"].idxmin()]
    cfg = dict(max_depth=int(best.max_depth), min_samples_leaf=int(best.min_samples_leaf))

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    r1 = res[res.min_samples_leaf == leaves[0]]
    ax[0].plot(r1.max_depth, r1.train_rmse, "o--", color=C1, label="train")
    ax[0].plot(r1.max_depth, r1.val_rmse, "o-", color=C2, label="validation")
    ax[0].set(xlabel="max_depth", ylabel="RMSE on log(price)",
              title=f"Bias–variance (min_samples_leaf={leaves[0]})")
    ax[0].legend()
    for msl, c in zip(leaves, [C1, C2, C3, C4]):
        r = res[res.min_samples_leaf == msl]
        ax[1].plot(r.max_depth, r.val_rmse, "o-", color=c, label=f"min_samples_leaf={msl}")
    ax[1].scatter([cfg["max_depth"]], [best.val_rmse], s=120, facecolors="none",
                  edgecolors="black", zorder=5, label="selected")
    ax[1].set(xlabel="max_depth", ylabel="validation RMSE", title="Task A: second hyper-parameter")
    ax[1].legend(fontsize=7)
    savefig(fig, "tree_regression_tuning")
    return res, cfg


def tune_tree_classification(S: Split, quick=False):
    depths = DEPTHS[::2] if quick else DEPTHS
    leaves = [1, 20]
    rows = []
    for crit in ["gini", "entropy"]:
        for msl in leaves:
            for d in depths:
                with ev.Timer() as t:
                    m = DecisionTree("classification", crit, max_depth=d, min_samples_leaf=msl).fit(S.X_a, S.t_a)
                _, _, f_tr = ev.precision_recall_f1(S.t_a, m.predict(S.X_a))
                _, _, f_va = ev.precision_recall_f1(S.t_b, m.predict(S.X_b))
                rows.append(dict(criterion=crit, max_depth=d, min_samples_leaf=msl,
                                 train_f1=f_tr, val_f1=f_va,
                                 val_auc=ev.roc_auc(S.t_b, m.predict_proba(S.X_b)[:, 1]),
                                 n_leaves=m.n_leaves_, fit_s=t.elapsed))
    res = pd.DataFrame(rows)
    best = res.loc[res["val_f1"].idxmax()]
    cfg = dict(criterion=best.criterion, max_depth=int(best.max_depth),
               min_samples_leaf=int(best.min_samples_leaf))

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for crit, c in [("gini", C1), ("entropy", C2)]:
        r = res[(res.criterion == crit) & (res.min_samples_leaf == 1)]
        ax[0].plot(r.max_depth, r.train_f1, "o--", color=c, alpha=0.7, label=f"{crit} train")
        ax[0].plot(r.max_depth, r.val_f1, "o-", color=c, label=f"{crit} val")
    ax[0].set(xlabel="max_depth", ylabel="F1 (premium)", title="Task B: Gini vs entropy (leaf=1)")
    ax[0].legend(fontsize=7)
    for (crit, msl), c, ls in [(("gini", 1), C1, "-"), (("gini", 20), C1, ":"),
                               (("entropy", 1), C2, "-"), (("entropy", 20), C2, ":")]:
        r = res[(res.criterion == crit) & (res.min_samples_leaf == msl)]
        ax[1].plot(r.max_depth, r.val_f1, "o" + ls, color=c, label=f"{crit}, leaf={msl}")
    ax[1].scatter([cfg["max_depth"]], [best.val_f1], s=120, facecolors="none",
                  edgecolors="black", zorder=5, label="selected")
    ax[1].set(xlabel="max_depth", ylabel="validation F1", title="Task B: min_samples_leaf")
    ax[1].legend(fontsize=7)
    savefig(fig, "tree_classification_tuning")
    return res, cfg


def tree_learning_curve(S: Split, cfg: dict):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(S.y_a))
    rows = []
    for frac in [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
        idx = perm[: int(frac * len(perm))]
        m = DecisionTree("regression", **cfg).fit(S.X_a[idx], S.y_a[idx])
        rows.append(dict(n=len(idx), train=ev.rmse(S.y_a[idx], m.predict(S.X_a[idx])),
                         val=ev.rmse(S.y_b, m.predict(S.X_b))))
    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(res.n, res.train, "o--", color=C1, label="train")
    ax.plot(res.n, res.val, "o-", color=C2, label="validation")
    ax.set(xscale="log", xlabel="training rows", ylabel="RMSE on log(price)",
           title="Learning curve, selected tree")
    ax.legend()
    savefig(fig, "tree_learning_curve")
    return res


# =============================================================================
# Problem 4(b) / 3(b-d): SVM tuning
# =============================================================================
LAMBDAS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]


def tune_svm(S: Split, quick=False, epochs=30):
    lambdas = LAMBDAS[::2] if quick else LAMBDAS
    rows, hist = [], {}
    for lam in lambdas:
        with ev.Timer() as t:
            m = PegasosSVM(lambda_=lam, epochs=epochs, random_state=SEED).fit(S.Z_a, S.t_a)
        _, _, f = ev.precision_recall_f1(S.t_b, m.predict(S.Z_b))
        rows.append(dict(lambda_=lam, C=1 / (lam * len(S.t_a)), val_f1=f,
                         val_auc=ev.roc_auc(S.t_b, m.decision_function(S.Z_b)),
                         train_obj=m.objective_history_[-1], w_norm=m.w_norm,
                         margin=m.margin_width,
                         n_sv=int(m.support_vector_mask(S.Z_a, S.t_a).sum()), fit_s=t.elapsed))
        hist[lam] = m.objective_history_
    res = pd.DataFrame(rows)
    # **Best F1; ties within 0.002 broken toward the larger lambda (wider margin)**.
    top = res["val_f1"].max()
    best = res[res["val_f1"] >= top - 0.002].sort_values("lambda_", ascending=False).iloc[0]
    lam_best = float(best.lambda_)

    # Margin vs validation metric: two panels (never a dual axis).
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax[0].plot(res.lambda_, res.val_f1, "o-", color=C1, label="F1")
    ax[0].plot(res.lambda_, res.val_auc, "s-", color=C2, label="ROC-AUC")
    ax[0].axvline(lam_best, color=INK2, lw=0.8, ls=":")
    ax[0].set(xscale="log", xlabel="λ", ylabel="validation score", title="Validation metric vs λ")
    ax[0].legend()
    ax[1].plot(res.lambda_, res.margin, "o-", color=C1)
    ax[1].axvline(lam_best, color=INK2, lw=0.8, ls=":")
    ax[1].set(xscale="log", yscale="log", xlabel="λ", ylabel="2 / ||w||", title="Margin width vs λ")
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
    ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].plot(res.lambda_, res.n_sv / len(S.t_a), "o-", color=C1)
    ax[2].axvline(lam_best, color=INK2, lw=0.8, ls=":")
    ax[2].set(xscale="log", xlabel="λ", ylabel="fraction of train rows",
              title="Support vectors (y·f(x) ≤ 1)")
    savefig(fig, "svm_lambda_sweep")

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    show = [l for l in lambdas if l in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)] or lambdas
    for l, c in zip(show, [C1, C2, C3, C4, C5]):
        ax.plot(np.arange(len(hist[l])), hist[l], color=c, label=f"λ={l:g}")
    ax.set(yscale="log", xlabel="epoch", ylabel="J(w, b) on train",
           title="Pegasos objective convergence")
    ax.legend(fontsize=7)
    savefig(fig, "svm_objective")
    return res, lam_best


def svm_schedule_epochs_study(S: Split, lam: float, quick=False):
    """Problem 3(b): epochs and learning-rate schedule each change the solution."""
    rows = []
    sched = [("pegasos", None, "η=1/(λt)"), ("invsqrt", 0.1, "η=0.1/√t"),
             ("constant", 0.01, "η=0.01")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for (s, eta0, label), c in zip(sched, [C1, C2, C3]):
        m = PegasosSVM(lambda_=lam, epochs=30, lr_schedule=s, eta0=eta0 or 0.1,
                       project=(s == "pegasos"), random_state=SEED).fit(S.Z_a, S.t_a)
        ax[0].plot(m.objective_history_, color=c, label=label)
        _, _, f = ev.precision_recall_f1(S.t_b, m.predict(S.Z_b))
        rows.append(dict(study="schedule", setting=label, epochs=30, val_f1=f,
                         w_norm=m.w_norm, margin=m.margin_width,
                         train_obj=m.objective_history_[-1]))
    ax[0].set(yscale="log", xlabel="epoch", ylabel="J(w, b)", title=f"Learning-rate schedule (λ={lam:g})")
    ax[0].legend()
    ep_list = [1, 2, 5, 10, 20] if quick else [1, 2, 3, 5, 10, 20, 40]
    ep_rows = []
    for ep in ep_list:
        m = PegasosSVM(lambda_=lam, epochs=ep, random_state=SEED, track_objective=False).fit(S.Z_a, S.t_a)
        _, _, f = ev.precision_recall_f1(S.t_b, m.predict(S.Z_b))
        ep_rows.append(dict(study="epochs", setting=f"{ep} epochs", epochs=ep, val_f1=f,
                            w_norm=m.w_norm, margin=m.margin_width))
    e = pd.DataFrame(ep_rows)
    ax[1].plot(e.epochs, e.margin, "o-", color=C1)
    ax[1].set(xscale="log", xlabel="epochs", ylabel="2 / ||w||", title="Margin width vs epochs")
    savefig(fig, "svm_schedule_epochs")
    return pd.concat([pd.DataFrame(rows), e], ignore_index=True)


def tune_rff(S: Split, quick=False):
    gammas = [0.003, 0.01, 0.03] if quick else [0.001, 0.003, 0.01, 0.03, 0.1]
    lams = [1e-3, 1e-4]
    rows = []
    for g in gammas:
        for lam in lams:
            with ev.Timer() as t:
                m = RFFPegasosSVM(gamma=g, n_components=1000, lambda_=lam, epochs=30,
                                  random_state=SEED).fit(S.Z_a, S.t_a)
            _, _, f = ev.precision_recall_f1(S.t_b, m.predict(S.Z_b))
            rows.append(dict(gamma=g, lambda_=lam, val_f1=f,
                             val_auc=ev.roc_auc(S.t_b, m.decision_function(S.Z_b)),
                             train_obj=m.objective_history_[-1], fit_s=t.elapsed))
    res = pd.DataFrame(rows)
    best = res.loc[res["val_f1"].idxmax()]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for lam, c in zip(lams, [C1, C2]):
        r = res[res.lambda_ == lam]
        ax.plot(r.gamma, r.val_f1, "o-", color=c, label=f"λ={lam:g}")
    ax.set(xscale="log", xlabel="RBF bandwidth γ", ylabel="validation F1",
           title="RFF Pegasos (D=1000)")
    ax.legend()
    savefig(fig, "svm_rff_gamma")
    return res, dict(gamma=float(best.gamma), lambda_=float(best.lambda_))


def svm_boundary_demo(S: Split, names: list):
    """2-D illustration of how λ moves the boundary/margin (Problem 3(b,c))."""
    fi = [names.index("log_area"), names.index("dist_center_km")]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(S.t_a), 600, replace=False)
    X2, y2 = S.Z_a[idx][:, fi], S.t_a[idx]
    gx, gy = np.meshgrid(np.linspace(-2.5, 3, 250), np.linspace(-1.5, 3.5, 250))
    G = np.c_[gx.ravel(), gy.ravel()]
    configs = [("linear", 1e-3), ("linear", 1e-1), ("linear", 1.0), ("rff", 1e-3)]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3), sharey=True)
    out = []
    for ax, (kind, lam) in zip(axes, configs):
        if kind == "linear":
            m = PegasosSVM(lambda_=lam, epochs=100, random_state=SEED).fit(X2, y2)
            sv = m.support_vector_mask(X2, y2)
            title = f"linear, λ={lam:g}\n||w||={m.w_norm:.2f}, margin={m.margin_width:.2f}"
            out.append(dict(lambda_=lam, w_norm=m.w_norm, margin=m.margin_width, n_sv=int(sv.sum())))
        else:
            m = RFFPegasosSVM(gamma=1.0, n_components=300, lambda_=lam, epochs=100,
                              random_state=SEED).fit(X2, y2)
            y_pm = np.where(y2 == 1, 1, -1)
            sv = y_pm * m.decision_function(X2) <= 1
            title = f"RFF (γ=1), λ={lam:g}"
        f = m.decision_function(G).reshape(gx.shape)
        ax.contourf(gx, gy, f > 0, levels=[-0.5, 0.5, 1.5], colors=["#fbe3d8", "#dbe8f8"], alpha=0.7)
        ax.contour(gx, gy, f, levels=[-1, 0, 1], colors=[INK2, "black", INK2],
                   linestyles=["--", "-", "--"], linewidths=[0.8, 1.4, 0.8])
        ax.scatter(*X2[y2 == 0].T, s=6, color=C2, linewidths=0, label="standard")
        ax.scatter(*X2[y2 == 1].T, s=6, color=C1, linewidths=0, label="premium")
        ax.scatter(*X2[sv].T, s=18, facecolors="none", edgecolors="black", linewidths=0.4,
                   label="support vectors")
        ax.set(title=title, xlabel="log area (z)")
        ax.title.set_fontsize(7.5)
        ax.grid(False)
    axes[0].set_ylabel("distance to centre (z)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, markerscale=1.5,
               bbox_to_anchor=(0.5, -0.06))
    savefig(fig, "svm_boundary_2d")
    return pd.DataFrame(out)


# =============================================================================
# Problem 5: model zoo (ours + matched sklearn baselines)
# =============================================================================
def task_a_zoo(cfg: dict, with_rf=False):
    t, r = cfg["tree_a"], cfg["ridge_lambda"]
    zoo = {
        "Mean baseline": ("raw", lambda: _MeanModel()),
        "Ours: decision tree": ("raw", lambda: DecisionTree("regression", "mse", **t)),
        "sklearn: DecisionTreeRegressor": ("raw", lambda: DecisionTreeRegressor(random_state=SEED, **t)),
        "Ours: Ridge (normal eq., bonus)": ("scaled", lambda: RidgeRegression(r)),
        "sklearn: Ridge (Pipeline)": ("raw", lambda: make_pipeline(StandardScaler(), Ridge(alpha=r))),
    }
    if with_rf:
        rf = cfg["rf"]
        zoo["Ours: random forest (bonus)"] = ("raw", lambda: RandomForest("regression", random_state=SEED, **rf))
        zoo["sklearn: RandomForestRegressor"] = ("raw", lambda: RandomForestRegressor(
            n_estimators=rf["n_estimators"], max_features=rf["max_features"],
            min_samples_leaf=rf["min_samples_leaf"], max_depth=rf["max_depth"],
            random_state=SEED, n_jobs=1))
    return zoo


def task_b_zoo(cfg: dict, n_train: int, with_svc=False):
    t, lam, rff = cfg["tree_b"], cfg["svm_lambda"], cfg["rff"]
    ep = cfg["svm_epochs"]
    zoo = {
        "Majority baseline": ("raw", lambda: _MajorityModel()),
        "Ours: decision tree": ("raw", lambda: DecisionTree("classification", **t)),
        "sklearn: DecisionTreeClassifier": ("raw", lambda: DecisionTreeClassifier(random_state=SEED, **t)),
        "Ours: Pegasos SVM (linear)": ("scaled", lambda: PegasosSVM(lambda_=lam, epochs=ep, random_state=SEED)),
        "sklearn: SGDClassifier hinge": ("raw", lambda: make_pipeline(
            StandardScaler(), SGDClassifier(loss="hinge", alpha=lam, max_iter=ep, tol=None, random_state=SEED))),
        "sklearn: LinearSVC": ("raw", lambda: make_pipeline(
            StandardScaler(), LinearSVC(C=1.0 / (lam * n_train), max_iter=5000, random_state=SEED))),
        "Ours: RFF Pegasos SVM": ("scaled", lambda: RFFPegasosSVM(
            gamma=rff["gamma"], n_components=1000, lambda_=rff["lambda_"], epochs=ep, random_state=SEED)),
        "sklearn: RBFSampler + SGD hinge": ("raw", lambda: make_pipeline(
            StandardScaler(), RBFSampler(gamma=rff["gamma"], n_components=1000, random_state=SEED),
            SGDClassifier(loss="hinge", alpha=rff["lambda_"], max_iter=ep, tol=None, random_state=SEED))),
    }
    if with_svc:
        zoo["sklearn: SVC rbf (10k subsample)"] = ("raw-sub", lambda: make_pipeline(
            StandardScaler(), SVC(kernel="rbf", gamma=rff["gamma"], C=1.0 / (rff["lambda_"] * 10_000))))
    return zoo


class _MeanModel:
    def fit(self, X, y):
        self.m = float(np.mean(y)); return self

    def predict(self, X):
        return np.full(len(X), self.m)


class _MajorityModel:
    def fit(self, X, y):
        vals, c = np.unique(y, return_counts=True); self.v = vals[np.argmax(c)]; return self

    def predict(self, X):
        return np.full(len(X), self.v)

    def predict_proba(self, X):
        p = np.zeros((len(X), 2)); p[:, int(self.v)] = 1.0; return p


def _run_model(kind, factory, S: Split, task: str):
    Xa, Xb = (S.Z_a, S.Z_b) if kind == "scaled" else (S.X_a, S.X_b)
    ya = S.y_a if task == "A" else S.t_a
    if kind == "raw-sub":
        sub = np.random.default_rng(SEED).choice(len(ya), min(10_000, len(ya)), replace=False)
        Xa, ya = Xa[sub], ya[sub]
    model = factory()
    with ev.Timer() as tf:
        model.fit(Xa, ya)
    with ev.Timer() as tp:
        pred = model.predict(Xb)
        score = _scores(model, Xb) if task == "B" else None
    if task == "A":
        met = ev.regression_report(S.y_b, pred)
    else:
        met = ev.classification_report(S.t_b, pred, score)
    met["fit_s"], met["predict_s"] = tf.elapsed, tp.elapsed
    return model, pred, score, met


def evaluate_on_test(S: Split, cfg: dict):
    """Final configs fit on TRAIN, scored once on TEST."""
    out = {"A": {}, "B": {}}
    models, preds = {"A": {}, "B": {}}, {"A": {}, "B": {}}
    for name, (kind, fac) in task_a_zoo(cfg, with_rf=True).items():
        m, p, _, met = _run_model(kind, fac, S, "A")
        out["A"][name], models["A"][name], preds["A"][name] = met, m, (p, None)
        print(f"    [A] {name:<36s} RMSE(log)={met['rmse_log']:.4f}  fit={met['fit_s']:.2f}s")
    for name, (kind, fac) in task_b_zoo(cfg, len(S.t_a), with_svc=True).items():
        m, p, s, met = _run_model(kind, fac, S, "B")
        out["B"][name], models["B"][name], preds["B"][name] = met, m, (p, s)
        print(f"    [B] {name:<36s} F1={met['f1']:.4f}  AUC={met['roc_auc']:.4f}  fit={met['fit_s']:.2f}s")
    return out, models, preds


def bootstrap_ci(y_true, y_pred, metric, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), (n, len(y_true)))
    vals = np.array([metric(y_true[i], y_pred[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cross_validate(df_dev: pd.DataFrame, cfg: dict, n_splits=5):
    """5-fold CV on train+val; everything re-fit inside each fold."""
    y = dp.make_target(df_dev)
    folds = {"A": {}, "B": {}}
    for k, (a, b) in enumerate(dp.stratified_kfold(y, n_splits, seed=SEED)):
        S = make_split(df_dev.iloc[a], df_dev.iloc[b])
        for name, (kind, fac) in task_a_zoo(cfg).items():
            folds["A"].setdefault(name, []).append(_run_model(kind, fac, S, "A")[3])
        for name, (kind, fac) in task_b_zoo(cfg, len(S.t_a)).items():
            folds["B"].setdefault(name, []).append(_run_model(kind, fac, S, "B")[3])
        print(f"    fold {k + 1}/{n_splits} done")
    return {t: {n: ev.summarize(v) for n, v in d.items()} for t, d in folds.items()}


def tree_vs_sklearn(S: Split, models: dict, cfg: dict) -> pd.DataFrame:
    rows = []
    for task, ours_n, sk_n in [("A", "Ours: decision tree", "sklearn: DecisionTreeRegressor"),
                               ("B", "Ours: decision tree", "sklearn: DecisionTreeClassifier")]:
        o, s = models[task][ours_n], models[task][sk_n]
        po, ps = o.predict(S.X_b), s.predict(S.X_b)
        same = np.isclose(po, ps).mean() if task == "A" else (po == ps).mean()
        rows.append(dict(task=task, ours_leaves=o.n_leaves_, sk_leaves=int(s.get_n_leaves()),
                         ours_depth=o.depth_, sk_depth=int(s.get_depth()),
                         root_feature_ours=S.names[o.root.feature],
                         root_feature_sk=S.names[s.tree_.feature[0]],
                         root_thr_ours=o.root.threshold, root_thr_sk=float(s.tree_.threshold[0]),
                         identical_test_predictions=float(same)))
    return pd.DataFrame(rows)


# =============================================================================
# Bonus 8 / 9
# =============================================================================
def tune_ridge(S: Split):
    rows = []
    for lam in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        m = RidgeRegression(lam).fit(S.Z_a, S.y_a)
        rows.append(dict(lambda_=lam, val_rmse=ev.rmse(S.y_b, m.predict(S.Z_b))))
    res = pd.DataFrame(rows)
    lam = float(res.loc[res.val_rmse.idxmin(), "lambda_"])
    sk = Ridge(alpha=lam).fit(S.Z_a, S.y_a)
    ours = RidgeRegression(lam).fit(S.Z_a, S.y_a)
    res.attrs["max_coef_diff_vs_sklearn"] = float(np.abs(sk.coef_ - ours.coef_).max())
    return res, lam


def random_forest_curve(rf: RandomForest, S_val: Split, S_test: Split):
    """Error vs number of OWN trees (val and test), Problem 9. `rf` is fit on TRAIN."""
    val = [ev.rmse(S_val.y_b, p) for p in rf.staged_predict(S_val.X_b)]
    test = [ev.rmse(S_test.y_b, p) for p in rf.staged_predict(S_test.X_b)]
    k = np.arange(1, len(val) + 1)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(k, val, color=C1, label="validation")
    ax.plot(k, test, color=C2, label="test")
    ax.set(xlabel="number of trees (our DecisionTree)", ylabel="RMSE on log(price)",
           title="Bonus: our random forest")
    ax.legend()
    savefig(fig, "bonus_random_forest")
    return pd.DataFrame(dict(n_trees=k, val_rmse=val, test_rmse=test))


# =============================================================================
# Problem 6: error analysis + bonus 10 (clusters)
# =============================================================================
def feature_importance_figs(names, tree_a, tree_b, svm, top=15):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for a, tree, title in [(ax[0], tree_a, "Tree, Task A (price)"), (ax[1], tree_b, "Tree, Task B (tier)")]:
        imp = tree.feature_importances_
        o = np.argsort(imp)[::-1][:top][::-1]
        a.barh([names[i] for i in o], imp[o], color=C1, height=0.7)
        a.set(xlabel="impurity-decrease importance", title=title)
        a.grid(axis="y", visible=False)
        a.tick_params(axis="y", labelsize=7)
    savefig(fig, "feature_importance")

    w = svm.w
    o = np.argsort(np.abs(w))[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.barh([names[i] for i in o], w[o], color=[C1 if w[i] > 0 else C2 for i in o], height=0.7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set(xlabel="weight (standardized features)", title="Pegasos SVM: largest |w|\nblue → premium, orange → standard")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7)
    savefig(fig, "svm_weights")
    imp_a = pd.Series(tree_a.feature_importances_, index=names).sort_values(ascending=False)
    imp_b = pd.Series(tree_b.feature_importances_, index=names).sort_values(ascending=False)
    wser = pd.Series(w, index=names).sort_values(key=np.abs, ascending=False)
    return imp_a, imp_b, wser


def classification_figs(S: Split, preds: dict, names: list):
    show = ["Ours: decision tree", "Ours: Pegasos SVM (linear)", "Ours: RFF Pegasos SVM"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for a, n in zip(ax, show):
        cm = ev.confusion_matrix(S.t_b, preds[n][0])
        a.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["std", "prem"], yticklabels=["std", "prem"],
              xlabel="predicted", title=n.replace("Ours: ", ""))
        a.grid(False)
    ax[0].set_ylabel("true")
    savefig(fig, "confusion_matrices")

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for n, c in zip(show + ["sklearn: DecisionTreeClassifier"], [C1, C2, C3, C4]):
        fpr, tpr = ev.roc_curve(S.t_b, preds[n][1])
        ax.plot(fpr, tpr, color=c, lw=1.5, label=f"{n.replace('Ours: ', '')} ({ev.roc_auc(S.t_b, preds[n][1]):.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title="ROC, Task B (test)")
    ax.legend(fontsize=6.5, loc="lower right")
    savefig(fig, "roc_curves")

    # Error rate vs distance of the true price from the tier threshold.
    dist = np.abs(S.y_b - S.thr_log)
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 5]
    lab = ["<5%", "5–10%", "10–20%", "20–30%", "30–50%", ">50%"]
    b = np.digitize(dist, bins[1:-1])
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    rows = []
    for n, c, off in zip(show, [C1, C2, C3], [-0.27, 0, 0.27]):
        err = [(preds[n][0][b == k] != S.t_b[b == k]).mean() for k in range(len(lab))]
        ax.bar(np.arange(len(lab)) + off, err, width=0.25, color=c, label=n.replace("Ours: ", ""))
        rows.append(pd.Series(err, index=lab, name=n))
    ax.set_xticks(np.arange(len(lab)), lab, fontsize=7)
    ax.set(xlabel="|log price − log threshold|  (≈ % away from median)", ylabel="error rate",
           title="Task B errors sit at the boundary")
    ax.legend(fontsize=7)
    ax.grid(axis="x", visible=False)
    savefig(fig, "tier_error_vs_distance")
    return pd.DataFrame(rows)


def regression_error_analysis(df_tr, df_te, S: Split, pred_log: np.ndarray):
    res = pred_log - S.y_b                       # signed error in log space
    ape = np.abs(np.exp(pred_log) - np.exp(S.y_b)) / np.exp(S.y_b)
    te = df_te.copy()
    te["pred_price"] = np.exp(pred_log)
    te["log_err"] = res
    te["ape"] = ape

    # (1) by true-price decile
    dec = pd.qcut(te["price"], 10, labels=False)
    by_dec = te.groupby(dec).agg(price_med=("price", "median"), bias=("log_err", "mean"),
                                 mape=("ape", "mean")).reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax[0].bar(np.arange(1, 11), by_dec.bias, color=[C2 if v < 0 else C1 for v in by_dec.bias], width=0.7)
    ax[0].axhline(0, color=INK2, lw=0.8)
    ax[0].set(xlabel="true-price decile (1 = cheapest)", ylabel="mean(pred − true) in log",
              title="Bias: shrinkage toward the middle")
    ax[0].grid(axis="x", visible=False)
    ax[1].bar(np.arange(1, 11), by_dec.mape * 100, color=C1, width=0.7)
    ax[1].set(xlabel="true-price decile", ylabel="MAPE (%)", title="Relative error by price level")
    ax[1].grid(axis="x", visible=False)
    savefig(fig, "error_by_price_decile")

    # (2) sparse locations: train frequency of the micro-location vs error
    cnt = df_tr["location"].value_counts()
    te["loc_train_n"] = te["location"].map(cnt).fillna(0)
    by_loc = te.groupby("location").agg(n_test=("ape", "size"), mape=("ape", "mean"),
                                        n_train=("loc_train_n", "first")).reset_index()
    by_loc = by_loc[by_loc.n_test >= 5]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.scatter(by_loc.n_train + 1, by_loc.mape * 100, s=np.clip(by_loc.n_test, 8, 80),
               color=C1, alpha=0.6, linewidths=0)
    ax.set(xscale="log", xlabel="train listings in micro-location (+1)", ylabel="test MAPE (%)",
           title="Sparse locations are harder")
    savefig(fig, "error_by_location_frequency")

    # (3) category / raion tables
    by_cat = te.groupby("category").agg(n=("ape", "size"), mape=("ape", "mean"),
                                        bias=("log_err", "mean"))
    by_raion = te.groupby(te["raion"].fillna("unknown")).agg(
        n=("ape", "size"), mape=("ape", "mean"), bias=("log_err", "mean")).sort_values("mape")
    worst = te.sort_values("ape", ascending=False).head(10)[
        ["category", "raion", "location", "area_m2", "rooms", "floor", "price", "pred_price", "ape"]]
    sparse_thr = 150
    sparse_mape = float(te.loc[te.loc_train_n < sparse_thr, "ape"].mean())
    dense_mape = float(te.loc[te.loc_train_n >= sparse_thr, "ape"].mean())
    return dict(by_decile=by_dec, by_category=by_cat, by_raion=by_raion, worst=worst,
                sparse_mape=sparse_mape, dense_mape=dense_mape,
                corr_loc=float(np.corrcoef(np.log1p(by_loc.n_train), by_loc.mape)[0, 1]))


def clusters_bonus(df_tr, df_te, S: Split, pred_log, tier_pred_svm):
    """Bonus 10: k-means on (lat, lng) + PCA, used for error analysis."""
    km = KMeans(n_clusters=8, n_init=10, random_state=SEED).fit(df_tr[["lat", "lng"]].to_numpy())
    c_tr = km.labels_
    c_te = km.predict(df_te[["lat", "lng"]].to_numpy())
    rows = []
    for k in range(8):
        m = c_te == k
        rows.append(dict(cluster=k, n_train=int((c_tr == k).sum()), n_test=int(m.sum()),
                         median_price=float(np.median(df_te["price"].to_numpy()[m])),
                         tree_rmse_log=ev.rmse(S.y_b[m], pred_log[m]),
                         svm_err=float((tier_pred_svm[m] != S.t_b[m]).mean()),
                         lat=float(km.cluster_centers_[k, 0]), lng=float(km.cluster_centers_[k, 1])))
    tab = pd.DataFrame(rows).sort_values("median_price").reset_index(drop=True)

    pca = PCA(n_components=2, random_state=SEED).fit(S.Z_a)
    P = pca.transform(S.Z_b)
    fig, ax = plt.subplots(2, 1, figsize=(3.6, 5.6))
    pal = [C1, C2, C3, "#eda100", "#e87ba4", "#008300", C4, C5]
    d = df_tr.reset_index(drop=True)
    msk = d["lat"].between(40.25, 40.62) & d["lng"].between(49.6, 50.35)
    for k in range(8):
        mm = msk & (c_tr == k)
        ax[0].scatter(d.loc[mm, "lng"], d.loc[mm, "lat"], s=1, color=pal[k], linewidths=0)
        lat_c, lng_c = km.cluster_centers_[k]
        # Only label clusters inside the Baku/Absheron view; labels of remote
        # clusters (other regions) would stretch the saved figure's bounding box.
        if 40.25 <= lat_c <= 40.62 and 49.6 <= lng_c <= 50.35:
            ax[0].text(lng_c, lat_c, str(k), fontsize=8, weight="bold", ha="center",
                       va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=pal[k], lw=1))
    ax[0].set(xlim=(49.6, 50.35), ylim=(40.25, 40.62))
    ax[0].set(xlabel="longitude", ylabel="latitude", title="k-means (k=8) on lat/lng (train)")
    ax[0].set_aspect(1 / np.cos(np.radians(40.4)))
    ax[0].grid(False)
    for lab, c, name in [(0, C2, "standard"), (1, C1, "premium")]:
        mm = S.t_b == lab
        ax[1].scatter(P[mm, 0], P[mm, 1], s=1, color=c, alpha=0.4, linewidths=0, label=name)
    ax[1].set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
              ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", title="PCA of test rows, by tier",
              xlim=np.percentile(P[:, 0], [0.5, 99.5]), ylim=np.percentile(P[:, 1], [0.5, 99.5]))
    ax[1].legend(markerscale=6)
    savefig(fig, "bonus_clusters_pca")
    # recode cluster ids for the table to match the figure labels
    return tab, pca.explained_variance_ratio_[:2].tolist()
