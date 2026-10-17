"""
run_all.py — ONE command reproduces every headline number and figure.

    python -m src.run_all            # full run (~10-15 min on a laptop)
    python -m src.run_all --quick    # coarser sweeps, for smoke-testing

Outputs
    report/figures/*.pdf|png   every figure in the report

    report/tables/*.tex        LaTeX tables \\input by report/report.tex
    results/metrics.json       every number (test + CV mean/std + configs)
    results/*.csv              sweep tables, error-analysis tables
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import data_prep as dp
from . import evaluate as ev
from . import experiments as ex
from .eda import run_eda

RES_DIR = "results"
TAB_DIR = "report/tables"


def _save_csv(df: pd.DataFrame, name: str, index=False):
    os.makedirs(RES_DIR, exist_ok=True)
    df.to_csv(os.path.join(RES_DIR, name), index=index, encoding="utf-8")


def _tex(name: str, body: str):
    os.makedirs(TAB_DIR, exist_ok=True)
    with open(os.path.join(TAB_DIR, name), "w", encoding="utf-8") as f:
        f.write(body)


def _esc(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _pm(cv: dict, k: str, fmt="{:.3f}") -> str:
    if cv is None or k not in cv:
        return "---"
    m, s = cv[k]
    return f"{fmt.format(m)}$\\pm${fmt.format(s)}"


def _short(name: str) -> str:
    return (name.replace("sklearn: ", "sk: ").replace(" (normal eq., bonus)", "$^{b}$")
            .replace(" (bonus)", "$^{b}$").replace(" (Pipeline)", "")
            .replace(" (10k subsample)", " (10k sub.)"))


# =============================================================================
def main(quick: bool = False) -> None:
    t0 = time.time()
    np.random.seed(dp.SEED)
    metrics: dict = {"seed": dp.SEED, "quick": quick}

    # 1. data -------------------------------------------------------------
    print("[1] Loading + cleaning", dp.DATA_PATH)
    df = dp.load_clean(verbose=True)
    log = df.attrs["cleaning_log"]
    i_tr, i_va, i_te = dp.split_indices(df["price"])
    df_tr, df_va, df_te = df.iloc[i_tr], df.iloc[i_va], df.iloc[i_te]
    df_dev = pd.concat([df_tr, df_va])
    S_val = ex.make_split(df_tr, df_va)     # train -> validation (tuning)
    S_test = ex.make_split(df_tr, df_te)    # train -> test (final, once)
    thr_azn = float(np.exp(S_val.thr_log))
    metrics["data"] = {
        "cleaning_log": log, "n_rows": len(df), "n_features": len(S_val.names),
        "n_train": len(i_tr), "n_val": len(i_va), "n_test": len(i_te),
        "tier_threshold_azn": thr_azn,
        "premium_share": {"train": float(S_val.t_a.mean()), "val": float(S_val.t_b.mean()),
                          "test": float(S_test.t_b.mean())},
        "leakage_columns": dp.LEAKAGE_COLUMNS,
    }
    print(f"    rows={len(df):,}  features={len(S_val.names)}  "
          f"train/val/test={len(i_tr):,}/{len(i_va):,}/{len(i_te):,}  "
          f"tier threshold={thr_azn:,.0f} AZN")

    print("[2] EDA figures")
    metrics["eda"] = run_eda(df, thr_azn)

    # 2. tuning on VALIDATION ---------------------------------------------
    print("[3] Tuning the decision tree (validation only)")
    res_ta, cfg_ta = ex.tune_tree_regression(S_val, quick)
    res_tb, cfg_tb = ex.tune_tree_classification(S_val, quick)
    lc = ex.tree_learning_curve(S_val, cfg_ta)
    _save_csv(res_ta, "tune_tree_regression.csv"); _save_csv(res_tb, "tune_tree_classification.csv")
    _save_csv(lc, "tree_learning_curve.csv")
    print(f"    Task A tree: {cfg_ta}   Task B tree: {cfg_tb}")

    print("[4] Tuning the Pegasos SVM (validation only)")
    res_svm, lam = ex.tune_svm(S_val, quick)
    res_sched = ex.svm_schedule_epochs_study(S_val, lam, quick)
    res_rff, cfg_rff = ex.tune_rff(S_val, quick)
    res_2d = ex.svm_boundary_demo(S_val, S_val.names)
    for d, n in [(res_svm, "tune_svm_lambda.csv"), (res_sched, "svm_schedule_epochs.csv"),
                 (res_rff, "tune_svm_rff.csv"), (res_2d, "svm_boundary_2d.csv")]:
        _save_csv(d, n)
    print(f"    SVM lambda={lam:g}   RFF={cfg_rff}")

    res_ridge, lam_ridge = ex.tune_ridge(S_val)
    _save_csv(res_ridge, "tune_ridge.csv")

    cfg = {
        "tree_a": cfg_ta, "tree_b": cfg_tb, "svm_lambda": lam, "svm_epochs": 30,
        "rff": cfg_rff, "ridge_lambda": lam_ridge,
        "rf": {"n_estimators": 10 if quick else 30, "max_features": 0.33,
               "min_samples_leaf": 2, "max_depth": None},
    }
    metrics["selected_configs"] = cfg

    # 3. final fit on TRAIN, evaluate once on TEST --------------------------
    print("[5] Final models: fit on train, score on test")
    test, models, preds = ex.evaluate_on_test(S_test, cfg)
    metrics["test"] = test
    tree_a = models["A"]["Ours: decision tree"]
    tree_b = models["B"]["Ours: decision tree"]
    svm = models["B"]["Ours: Pegasos SVM (linear)"]
    yA = preds["A"]["Ours: decision tree"][0]
    metrics["test_bootstrap_ci95"] = {
        "tree_rmse_log": ex.bootstrap_ci(S_test.y_b, yA, ev.rmse),
        "tree_f1": ex.bootstrap_ci(S_test.t_b, preds["B"]["Ours: decision tree"][0],
                                   lambda a, b: ev.precision_recall_f1(a, b)[2]),
        "svm_f1": ex.bootstrap_ci(S_test.t_b, preds["B"]["Ours: Pegasos SVM (linear)"][0],
                                  lambda a, b: ev.precision_recall_f1(a, b)[2]),
    }
    sv_mask = svm.support_vector_mask(S_test.Z_a, S_test.t_a)
    metrics["svm_margin"] = {"lambda": lam, "w_norm": svm.w_norm, "margin_width": svm.margin_width,
                             "bias": svm.b, "n_support_vectors": int(sv_mask.sum()),
                             "sv_fraction": float(sv_mask.mean()),
                             "final_objective": svm.objective_history_[-1],
                             "objective_epoch0": svm.objective_history_[0],
                             "objective_epoch1": svm.objective_history_[1]}
    _save_csv(pd.DataFrame({"train_row": np.flatnonzero(sv_mask)}), "svm_support_vector_indices.csv")
    tvs = ex.tree_vs_sklearn(S_test, models, cfg)
    _save_csv(tvs, "tree_vs_sklearn.csv")
    metrics["tree_vs_sklearn"] = tvs.to_dict(orient="records")
    metrics["ridge_max_coef_diff_vs_sklearn"] = res_ridge.attrs["max_coef_diff_vs_sklearn"]

    print("[6] Bonus: random-forest curve")
    rf_curve = ex.random_forest_curve(models["A"]["Ours: random forest (bonus)"], S_val, S_test)
    _save_csv(rf_curve, "random_forest_curve.csv")

    # 4. cross-validated uncertainty ---------------------------------------
    print("[7] 5-fold CV on train+val (uncertainty)")
    cv = ex.cross_validate(df_dev, cfg, n_splits=3 if quick else 5)
    metrics["cv"] = cv

    # 5. analysis -----------------------------------------------------------
    print("[8] Error analysis, importances, clusters")
    imp_a, imp_b, wser = ex.feature_importance_figs(S_test.names, tree_a, tree_b, svm)
    _save_csv(imp_a.rename("importance").reset_index(), "importance_tree_a.csv")
    _save_csv(imp_b.rename("importance").reset_index(), "importance_tree_b.csv")
    _save_csv(wser.rename("weight").reset_index(), "svm_weights.csv")
    tier_err = ex.classification_figs(S_test, preds["B"], S_test.names)
    _save_csv(tier_err, "tier_error_vs_distance.csv", index=True)
    ea = ex.regression_error_analysis(df_tr, df_te, S_test, yA)
    for k in ["by_decile", "by_category", "by_raion", "worst"]:
        _save_csv(ea[k], f"error_{k}.csv", index=k != "by_decile")
    clus, pca_var = ex.clusters_bonus(df_tr, df_te, S_test, yA, preds["B"]["Ours: Pegasos SVM (linear)"][0])
    _save_csv(clus, "bonus_clusters.csv")
    metrics["analysis"] = {
        "top_importance_a": imp_a.head(10).to_dict(), "top_importance_b": imp_b.head(10).to_dict(),
        "top_svm_weights": wser.head(10).to_dict(),
        "decile_bias": ea["by_decile"]["bias"].tolist(), "decile_mape": ea["by_decile"]["mape"].tolist(),
        "category": ea["by_category"].to_dict(orient="index"),
        "sparse_location_mape": ea["sparse_mape"], "dense_location_mape": ea["dense_mape"],
        "corr_logtraincount_mape": ea["corr_loc"],
        "tier_error_vs_distance": tier_err.to_dict(orient="index"),
        "clusters": clus.to_dict(orient="records"), "pca_var": pca_var,
    }

    # 6. tables + summary ---------------------------------------------------
    write_tables(metrics, cfg, res_svm)
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=float)

    print("\n" + "=" * 100)
    print(f"FINAL COMPARISON  (test set; CV = 5-fold mean ± std on train+val)   seed={dp.SEED}")
    print("=" * 100)
    rows_a = {n: {**m, **{f"cv_{k}": v for k, v in cv["A"].get(n, {}).items()}} for n, m in test["A"].items()}
    ev.compare(rows_a, ["rmse_log", "mae_log", "r2_log", "mape", "cv_rmse_log", "fit_s"],
               "Task A — regression on log(price)")
    print()
    rows_b = {n: {**m, **{f"cv_{k}": v for k, v in cv["B"].get(n, {}).items()}} for n, m in test["B"].items()}
    ev.compare(rows_b, ["precision", "recall", "f1", "roc_auc", "cv_f1", "fit_s"],
               "Task B — premium vs standard tier")
    print(f"\nSVM: ||w||={svm.w_norm:.3f}  margin 2/||w||={svm.margin_width:.3f}  "
          f"support vectors={int(sv_mask.sum()):,} ({sv_mask.mean():.1%} of train)")
    print(f"Done in {time.time() - t0:.0f}s. Figures -> report/figures, tables -> report/tables, "
          f"numbers -> results/metrics.json")


def write_tables(M: dict, cfg: dict, res_svm: pd.DataFrame):
    d = M["data"]
    # Cleaning log
    rows = "\n".join(f"    {_esc(s).replace('²', '$^2$')} & {n:,} \\\\" for s, n in d["cleaning_log"])
    _tex("tab_cleaning.tex", "\\begin{tabular}{@{}p{0.70\\linewidth}r@{}}\n\\toprule\n"
         "Step & Rows left \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n")
    # Data summary
    ps = d["premium_share"]
    _tex("tab_data.tex", f"""\\begin{{tabular}}{{@{{}}lr@{{}}}}
\\toprule
Property & Value \\\\
\\midrule
Rows (after cleaning) & {d['n_rows']:,} \\\\
Features used & {d['n_features']} \\\\
Train / Val / Test & {d['n_train']:,} / {d['n_val']:,} / {d['n_test']:,} \\\\
Tier threshold (train median) & {d['tier_threshold_azn']:,.0f} AZN \\\\
Premium share train/val/test (\\%) & {100*ps['train']:.1f} / {100*ps['val']:.1f} / {100*ps['test']:.1f} \\\\
\\bottomrule
\\end{{tabular}}
""")
    # Results A
    T, C = M["test"], M["cv"]
    lines = []
    for n, m in T["A"].items():
        lines.append(f"{_short(n)} & {m['rmse_log']:.3f} & {m['mae_log']:.3f} & {m['r2_log'] + 0.0 if abs(m['r2_log']) > 5e-4 else 0.0:.3f} & "
                     f"{100*m['mape']:.1f} & {_pm(C['A'].get(n), 'rmse_log')} & {m['fit_s']:.2f} \\\\")
    _tex("tab_results_a.tex", "\\begin{tabular}{@{}lcccccr@{}}\n\\toprule\n"
         "Model & RMSE & MAE & $R^2$ & MAPE\\% & CV RMSE & fit s \\\\\n\\midrule\n"
         + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    # Results B
    lines = []
    for n, m in T["B"].items():
        lines.append(f"{_short(n)} & {m['precision']:.3f} & {m['recall']:.3f} & {m['f1']:.3f} & "
                     f"{m['roc_auc']:.3f} & {_pm(C['B'].get(n), 'f1')} & {m['fit_s']:.2f} \\\\")
    _tex("tab_results_b.tex", "\\begin{tabular}{@{}lcccccr@{}}\n\\toprule\n"
         "Model & Prec. & Rec. & F1 & AUC & CV F1 & fit s \\\\\n\\midrule\n"
         + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    # Headline
    a_t, a_s, b_t, b_s = (T["A"]["Ours: decision tree"], T["A"]["sklearn: DecisionTreeRegressor"],
                          T["B"]["Ours: decision tree"], T["B"]["sklearn: DecisionTreeClassifier"])
    sv, sg = T["B"]["Ours: Pegasos SVM (linear)"], T["B"]["sklearn: SGDClassifier hinge"]
    rf = T["B"]["Ours: RFF Pegasos SVM"]
    _tex("tab_headline.tex", f"""\\begin{{tabular}}{{@{{}}lcc@{{}}}}
\\toprule
Model & Task A RMSE (log) & Task B F1 \\\\
\\midrule
Ours: decision tree & {a_t['rmse_log']:.3f} ({_pm(C['A']['Ours: decision tree'], 'rmse_log')}) & {b_t['f1']:.3f} ({_pm(C['B']['Ours: decision tree'], 'f1')}) \\\\
Ours: SVM (Pegasos) & n/a & {sv['f1']:.3f} ({_pm(C['B']['Ours: Pegasos SVM (linear)'], 'f1')}) \\\\
Ours: SVM + RFF & n/a & {rf['f1']:.3f} ({_pm(C['B']['Ours: RFF Pegasos SVM'], 'f1')}) \\\\
sk: decision tree & {a_s['rmse_log']:.3f} ({_pm(C['A']['sklearn: DecisionTreeRegressor'], 'rmse_log')}) & {b_s['f1']:.3f} ({_pm(C['B']['sklearn: DecisionTreeClassifier'], 'f1')}) \\\\
sk: SGD hinge & n/a & {sg['f1']:.3f} ({_pm(C['B']['sklearn: SGDClassifier hinge'], 'f1')}) \\\\
\\bottomrule
\\end{{tabular}}
""")
    # SVM lambda sweep
    lines = [f"{r.lambda_:g} & {r.val_f1:.3f} & {r.val_auc:.3f} & {r.w_norm:.2f} & {r.margin:.3f} & "
             f"{100*r.n_sv/ (M['data']['n_train']):.1f} \\\\" for r in res_svm.itertuples()]
    _tex("tab_svm_lambda.tex", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
         "$\\lambda$ & val F1 & val AUC & $\\|\\mathbf{w}\\|$ & $2/\\|\\mathbf{w}\\|$ & SV \\% \\\\\n\\midrule\n"
         + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    # Tree vs sklearn
    lines = []
    for r in M["tree_vs_sklearn"]:
        lines.append(f"{r['task']} & {r['ours_leaves']} / {r['sk_leaves']} & {r['ours_depth']} / {r['sk_depth']} & "
                     f"{_esc(r['root_feature_ours'])}$\\le${r['root_thr_ours']:.4g} / "
                     f"{_esc(r['root_feature_sk'])}$\\le${r['root_thr_sk']:.4g} & "
                     f"{100*r['identical_test_predictions']:.1f} \\\\")
    _tex("tab_tree_vs_sklearn.tex", "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
         "Task & leaves & depth & root split (ours / sklearn) & same pred.\\% \\\\\n\\midrule\n"
         + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarser sweeps for a fast smoke test")
    main(quick=ap.parse_args().quick)
