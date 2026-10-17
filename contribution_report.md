# Contribution Report

Repository: <https://github.com/pnah6699-maker/bina-az> (tag `v1.0-final`)

Everything below is taken from the Git history of `main`: commit authors, the
files each person added, and lines of code. Every member can explain the files
listed against their name.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Budagov, Farhad        | [@budagovfn](https://github.com/budagovfn) | Decision tree, evaluation metrics and their tests; pytest config; final report (`src/decision_tree.py`, `src/evaluate.py`, `tests/test_decision_tree.py`, `tests/test_evaluate.py`, `tests/conftest.py`, `report/`) |
| Atlukhanov, Murad      | [@atlukhanovmurad2-ui](https://github.com/atlukhanovmurad2-ui) | Data cleaning, features, EDA, plotting style, dataset download (`src/data_prep.py`, `src/eda.py`, `src/plots.py`, `scripts/download_data.py`, `tests/test_data_prep.py`) |
| Guluyev, Panah         | [@pnah6699-maker](https://github.com/pnah6699-maker) | Repository setup, Pegasos SVM + RFF and tests, random forest, pipeline entry point, dependencies (`README.md`, `.gitignore`, `LICENSE`, `src/svm.py`, `tests/test_svm.py`, `src/ensemble.py`, `src/run_all.py`, `requirements.txt`) |
| Hasanov, Mikayil       | [@mickael044](https://github.com/mickael044) | Experiments (tuning, sklearn benchmark, CV, error analysis), Ridge regression, package setup, data docs (`src/experiments.py`, `src/linear.py`, `src/__init__.py`, `tests/__init__.py`, `data/README.md`) |

## Per-member summary
> Drafted from the commit history. **Each member: read your paragraph, correct
> it, and add anything that the history does not show** (debugging, reviews,
> meetings, slides).

**Budagov, Farhad —** Implemented the CART decision tree in
`src/decision_tree.py` (Gini/entropy/MSE criteria, four stopping rules,
`predict_proba`, feature importances) and the NumPy metrics in
`src/evaluate.py` (RMSE/MAE/R², confusion matrix, precision/recall/F1,
ROC-AUC, PR-AUC). Wrote `tests/test_decision_tree.py`, including the check
that the tree reproduces scikit-learn's splits on a fully determined toy
example, plus `tests/test_evaluate.py` and the pytest configuration
(`tests/conftest.py`). Added the final IEEE report (`report/report.tex`, the
generated tables and the compiled PDF). Commits: `4cee884`, `0c666d2`,
`f2f192d`, `956f711`, `8fe3528`, `f7d188d`, `8420f35`.

**Atlukhanov, Murad —** Built the data pipeline in `src/data_prep.py`:
de-duplication, leakage-column removal, parsing of area/floor/district,
fixed outlier rules, the train-only `Featurizer`, the stratified split, the
tier label and standardisation. Added the EDA figures (`src/eda.py`), the
shared plotting style (`src/plots.py`), the Kaggle download helper
(`scripts/download_data.py`) and `tests/test_data_prep.py`. Also shortened the
module docstring of `src/svm.py` (commit `7efdab4`). Commits: `8dc3048`,
`eb71cf5`, `7efdab4`, `c285ac6`, `64d4219`.

**Guluyev, Panah —** Set up the repository (`README.md`, `.gitignore`, MIT
`LICENSE`). Implemented the linear soft-margin SVM trained with Pegasos
(mini-batch sub-gradient steps, η = 1/(λt), projection, margin and
support-vector reporting) and the random-Fourier-feature extension in
`src/svm.py`, with `tests/test_svm.py`. Wrote the random forest built from our
own trees (`src/ensemble.py`, bonus P9), the single entry point
`src/run_all.py`, and the pinned `requirements.txt`. Commits: `87e3b3d`,
`8ad6ab1`, `4180baf`, `b7b3f99`, `d239626`, `e655b6d`.

**Hasanov, Mikayil —** Wrote `src/experiments.py`: validation-only
hyper-parameter sweeps for the tree and SVM, matched scikit-learn baselines,
5-fold cross-validation, the error analysis and the k-means/PCA bonus. Added
Ridge regression via the normal equations (`src/linear.py`, bonus P8), the
package `__init__.py` files and `data/README.md`, and fixed an accidental
submodule entry on `main`. Commits: `27a0568`, `9e14689`, `5ac2e81`,
`5c67c0e`, `98d3b0a`, `b180d17`, `22d5df0`.

## AI assistance (applies to the whole team)
AI coding assistants (Claude Code, Anthropic, ChatGPT) were used to draft the structure of the code(its pipeline). The idea of tests, figures and the first version of the report text were proposed by assistants. 

## Rough effort split
Measured from Git: lines added to the code folders (`src/`, `tests/`,
`scripts/`) by each author, excluding merge commits. The report, README and
other documents are not counted.

| Member            | Commits | Code lines added | Approx. % |
|-------------------|---------|------------------|-----------|
| Budagov, Farhad   | 7       | 694              | 26        |
| Atlukhanov, Murad | 5       | 622              | 23        |
| Guluyev, Panah    | 6       | 667              | 25        |
| Hasanov, Mikayil  | 7       | 688              | 26        |
| **Total**         | 25      | 2,671            | 100       |

The split is close to even. Farhad additionally committed the report, and
Panah the repository setup files and dependencies.
