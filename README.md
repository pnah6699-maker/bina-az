# bina.az price & price-tier — decision tree and Pegasos SVM from scratch

MLE-AI-201 Machine Learning — Final Project (V3).

We implement a **CART decision tree** (Gini / entropy / MSE) and a
**soft-margin SVM trained with Pegasos** (plus random Fourier features for an
RBF-like non-linear version) **in NumPy only**, apply them to ~57k cleaned
Baku real-estate listings from bina.az, and benchmark them against
scikit-learn under matched settings.

| Task | Target | Our models | Metrics |
|------|--------|------------|---------|
| A — regression | `log(price)` | decision tree (MSE) | RMSE, MAE, R² (log and AZN), MAPE |
| B — classification | premium (price > **train** median) vs standard | decision tree (Gini/entropy), Pegasos SVM, RFF-Pegasos SVM | precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix |

Bonuses: Ridge regression via the normal equations (P8), a random forest of
our own trees (P9), k-means / PCA geospatial analysis (P10).

## Headline results (test set, 8,620 listings; ± = 5-fold CV std on train+val)

| Model | Task A RMSE on log(price) ↓ | Task A R² | Task B F1 ↑ | Task B ROC-AUC |
|---|---|---|---|---|
| **Ours: decision tree** | **0.232** (0.232 ± 0.004) | 0.864 | **0.921** (0.926 ± 0.004) | 0.917 |
| sklearn DecisionTree (matched) | 0.232 (0.233 ± 0.004) | 0.864 | 0.921 (0.926 ± 0.004) | 0.916 |
| **Ours: Pegasos SVM (linear)** | — | — | **0.900** (0.907 ± 0.003) | 0.965 |
| sklearn SGDClassifier hinge (matched) | — | — | 0.900 (0.908 ± 0.003) | 0.965 |
| **Ours: RFF Pegasos SVM** | — | — | 0.900 (0.906 ± 0.003) | 0.967 |
| Ours: random forest (bonus) | 0.185 | 0.914 | — | — |
| Mean / majority baseline | 0.629 | 0.000 | 0.000 | 0.500 |

Tier threshold = 209,000 AZN (train median). Task A in AZN: tree MAE ≈ 44.6k AZN, MAPE 15.4%.

Full tables: `report/tables/*.tex`, every number: `results/metrics.json`.

## Reproduce everything (one command)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py        # or place the CSV by hand, see data/README.md
python -m src.run_all                  # ~15 min: all numbers, figures and LaTeX tables
python -m pytest -q                    # 40 unit tests (incl. tree == sklearn on toy data)
```

`python -m src.run_all --quick` runs coarser sweeps for a fast smoke test.
All randomness is seeded (`SEED = 42` in `src/data_prep.py`), so reruns
reproduce the same numbers.

Build the report / slides (any TeX distribution; uses only standard packages):

```bash
cd report && pdflatex report.tex && pdflatex report.tex
cd presentation && pdflatex slides.tex && pdflatex slides.tex
```

## Repo layout
```
.
├── README.md
├── requirements.txt          # pinned versions (Python 3.14)
├── contribution_report.md    # who did what
├── data/README.md            # how to obtain the dataset (CSV is NOT committed)
├── scripts/download_data.py  # optional kagglehub download
├── src/
│   ├── data_prep.py          # loading, cleaning, leakage removal, features, split, tier label, scaling
│   ├── decision_tree.py      # OUR DecisionTree (Gini / entropy / MSE, 4 stopping rules, importances)
│   ├── svm.py                # OUR PegasosSVM + RandomFourierFeatures + RFFPegasosSVM
│   ├── evaluate.py           # OUR metrics (RMSE/MAE/R², P/R/F1, ROC-AUC, AP, confusion) + tables
│   ├── linear.py             # bonus: Ridge via normal equations
│   ├── ensemble.py           # bonus: random forest of our own trees
│   ├── eda.py                # EDA figures
│   ├── experiments.py        # tuning sweeps, sklearn benchmark, CV, error analysis
│   ├── plots.py              # figure style
│   └── run_all.py            # ONE entry point
├── tests/                    # pytest suite
├── report/                   # IEEE-style report (report.tex, figures/, tables/)
└── presentation/             # slides.tex -> slides.pdf
```

## Key design decisions
- **Leakage:** `unit_price` (price per m²) and `total_price` (= price) are
  dropped first; identifiers, owner names, raw address, free text, scrape
  metadata and `views` (accumulated after listing) are dropped too.
- **De-duplication:** 36k rows are repeat scrapes of the same listing URL →
  keep the latest; a further ~750 re-posts with identical
  price/area/rooms/floor/coordinates are removed.
- **Scope:** the three residential categories (new build, old stock,
  house/villa). Land, commercial, office and garage listings are a different
  market with different units (land is priced per *sot*).
- **No test peeking:** the featurizer (medians, category levels), tier
  threshold and standardisation are fit on the training part only (and re-fit
  inside every CV fold); hyper-parameters are chosen on validation only; the
  test set is scored once.
- **Split:** 70/15/15, stratified on price deciles, so all three splits have
  the same price distribution and the same ~50/50 tier balance.
- **Pegasos:** mini-batch (32) sub-gradient steps, η_t = 1/(λt), projection
  onto ‖w‖ ≤ 1/√λ, bias folded in as a constant feature, objective tracked
  every epoch.
- **Non-linearity:** random Fourier features (D = 1000) + the same linear
  Pegasos. Chosen over kernelised Pegasos because n ≈ 40k makes O(n) kernel
  evaluations per step impractical.
