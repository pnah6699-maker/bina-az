"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

ALL data wrangling lives here so the rest of the code just calls these
functions. Every random operation is seeded with SEED.

Pipeline:

df  = load_raw()                    # raw Kaggle dump
df  = clean(df)                       # types, units, dedup, outliers, leakage
idx = split_indices(df["price"])     # stratified train /val / test indices
fz  = Featurizer().fit(df_train)    # learns medians /category levels on TRAIN
X   = fz.transform(df_any)         # numeric design matrix
y_tier, thr = make_tier_label(y_train)          # threshold from TRAIN only
X_tr_s, X_val_s, ... = standardize(X_tr, X_val, ...)  # TRAIN mean/std only
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd


SEED = 42
DATA_PATH = os.environ.get("BINAAZ_DATA", "data/house_sale.csv")

# Columns that LEAK the target (arithmetic functions of price). Dropped first.
#   unit_price : "3 450 AZN/m²"= price/area->price per m²
#   total_price : identical to price in 100 762/100 775 rows(price restated)
LEAKAGE_COLUMNS: list[str] = ["unit_price", "total_price"]

# Identifiers / free text / scrape metadata: no predictive meaning at listing
# time, or personal data we should not model on.
ID_AND_META_COLUMNS: list[str] = [
    "id_x", "id_y", "estate_id", "estate_details_id_x", "estate_details_id_y",
    "rel_url", "estate_rel_url_x", "estate_rel_url_y", "estate_rel_url",
    "img_url", "owner_name", "shop_name", "shop_title", "address",
    "description", "attributes", "city_when", "day_x", "hour_x", "day_y",
    "hour_y", "updated", "datetime_scrape_x", "datetime_scrape_y",
    "currency_x", "currency_y",
    # views is accumulated AFTER the listing is posted(and is partly driven
    # by price itself)-> not available at prediction time.
    "views",
    # Duplicates of the Azerbaijani detail columns (kept those instead).
    "repair", "bill_of_sale", "mortgage",
    # 99.8 % missing.
    "Binanın növü",
]


RENAME = {
    "Sahə": "area_m2",
    "Otaq sayı": "rooms",
    "Mərtəbə": "floor_raw",
    "Torpaq sahəsi": "land_raw",
    "Kateqoriya": "category",
    "Təmir": "repaired",
    "Çıxarış": "bill_of_sale",
    "İpoteka": "mortgage",
    "location": "location",
    "city": "city",
    "lat": "lat",
    "lng": "lng",
    "owner_title": "owner_title",
    "products_label": "products_label",
    "vip": "vip",
    "featured": "featured",
    "extra_info": "extra_info",
}

CATEGORY_EN = {
    "Yeni tikili": "new_build",
    "Köhnə tikili": "old_build",
    "Həyət evi/Bağ evi": "house_villa",
}

TO_AZN = {"AZN": 1.0, "USD": 1.70, "EUR": 1.85}

# Fixed, domain-based outlier rules (NOT data-driven percentiles, so no
# statistic is computed on the test rows).
PRICE_MIN, PRICE_MAX = 10_000, 5_000_000        # AZN
AREA_MIN, AREA_MAX = 15, 1_000                  # m²
PPM2_MIN, PPM2_MAX = 300, 15_000                # AZN / m², data-entry sanity only
ROOMS_MAX = 10
FLOORS_MAX = 40
AZ_BBOX = (38.3, 41.95, 44.7, 50.7)             # lat_min, lat_max, lng_min, lng_max

# City centre (Fountain Square, Baku) for a distance feature.
CENTER_LAT, CENTER_LNG = 40.3712, 49.8370

SOT_TO_M2 = 100.0


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV (UTF-8, Azerbaijani headers). Return it unmodified."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. See data/README.md for how to get it."
        )
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def _parse_area(s: pd.Series) -> pd.Series:
    """'145 m²' -> 145.0 ; '4.5 sot' -> 450.0 (1 sot = 100 m²)."""
    s = s.astype("string").str.strip()
    num = pd.to_numeric(s.str.extract(r"([\d.,]+)")[0].str.replace(",", ".", regex=False),
                        errors="coerce")
    is_sot = s.str.contains("sot", na=False)
    return num.where(~is_sot, num * SOT_TO_M2).astype(float)


def _parse_floor(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'7 / 9' -> (7, 9)."""
    parts = s.astype("string").str.extract(r"(-?\d+)\s*/\s*(\d+)")
    return (pd.to_numeric(parts[0], errors="coerce").astype(float),
            pd.to_numeric(parts[1], errors="coerce").astype(float))


def _parse_raion(s: pd.Series) -> pd.Series:
    """extra_info 'Neftçilər m.* Nizami r.* 8-ci kilometr q.' -> 'Nizami r.'."""
    return s.astype("string").str.extract(r"([^*]+?\sr\.)")[0].str.strip()


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Clean the raw frame. Every rule is logged in the returned frame's
    `.attrs["cleaning_log"]` so the report can quote exact row counts.
      1. currency -> AZN,
      2. de-duplicate (listing scraped several times; re-posted listings),
      3. keep the three residential categories,
      4. parse strings to numbers (area, floor, land area),
      5. fixed domain outlier rules,
      6. drop leakage / id / metadata columns, translate names.
    """
    log: list[tuple[str, int]] = [("raw rows", len(df))]
    df = df.copy()

    rate = df["currency_x"].map(TO_AZN)
    df = df[rate.notna()].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce") * rate[rate.notna()]
    df = df[df["price"].notna()]
    log.append(("known currency, numeric price", len(df)))


    df = (df.sort_values("datetime_scrape_x")
            .drop_duplicates("estate_rel_url_x", keep="last"))
    log.append(("unique listing URL (latest scrape)", len(df)))

    df = df[df["Kateqoriya"].isin(CATEGORY_EN)].copy()
    log.append(("residential categories", len(df)))

    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS + ID_AND_META_COLUMNS if c in df])
    df = df.rename(columns=RENAME)

    df["category"] = df["category"].map(CATEGORY_EN)
    df["area_m2"] = _parse_area(df["area_m2"])
    df["land_m2"] = _parse_area(df["land_raw"])
    df["floor"], df["total_floors"] = _parse_floor(df["floor_raw"])
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    df["raion"] = _parse_raion(df["extra_info"])
    df["has_metro_tag"] = df["extra_info"].astype("string").str.contains(r"\sm\.", na=False).astype(int)
    for col, yes in [("repaired", "var"), ("bill_of_sale", "var"), ("mortgage", "var")]:
        v = df[col].map({"var": 1.0, "yoxdur": 0.0})
        df[col] = v.fillna(0.0) if col == "mortgage" else v
    df["is_agent"] = (df["owner_title"] == "vasitəçi (agent)").astype(int)
    df["is_complex"] = (df["products_label"] == "Kompleks").astype(int)
    df["is_vip"] = df["vip"].notna().astype(int)
    df["is_featured"] = df["featured"].notna().astype(int)
    df["is_baku"] = (df["city"] == "bakı").astype(int)
    df = df.drop(columns=["floor_raw", "land_raw", "owner_title", "products_label",
                          "vip", "featured", "extra_info"])

    key = ["price", "area_m2", "rooms", "floor", "lat", "lng", "category"]
    df = df.drop_duplicates(subset=key, keep="last")
    log.append(("no exact re-posts (same price/area/rooms/floor/coords)", len(df)))


    df = df[df["price"].between(PRICE_MIN, PRICE_MAX)]
    log.append((f"price in [{PRICE_MIN:,}, {PRICE_MAX:,}] AZN", len(df)))
    df = df[df["area_m2"].between(AREA_MIN, AREA_MAX)]
    log.append((f"area in [{AREA_MIN}, {AREA_MAX}] m²", len(df)))
    ppm2 = df["price"] / df["area_m2"]  
    df = df[ppm2.between(PPM2_MIN, PPM2_MAX)]
    log.append((f"price/m² in [{PPM2_MIN}, {PPM2_MAX:,}] (entry errors)", len(df)))
    lat0, lat1, lng0, lng1 = AZ_BBOX
    df = df[df["lat"].between(lat0, lat1) & df["lng"].between(lng0, lng1)]
    log.append(("coordinates inside Azerbaijan", len(df)))

    df.loc[df["rooms"] > ROOMS_MAX, "rooms"] = np.nan
    bad_floor = ((df["total_floors"] > FLOORS_MAX) | (df["floor"] > df["total_floors"])
                 | (df["floor"] < 0))
    df.loc[bad_floor, ["floor", "total_floors"]] = np.nan
   
    df.loc[df["category"] != "house_villa", "land_m2"] = 0.0

    df = df.reset_index(drop=True)
    df.attrs["cleaning_log"] = log
    if verbose:
        for step, n in log:
            print(f"  {step:<55s} {n:>7,d}")
    return df


NUMERIC_FEATURES = [
    "area_m2", "log_area", "rooms", "area_per_room", "floor", "total_floors",
    "floor_ratio", "is_first_floor", "is_top_floor", "land_m2", "lat", "lng",
    "dist_center_km",
]
BINARY_FEATURES = [
    "repaired", "bill_of_sale", "mortgage", "is_agent", "is_complex", "is_vip",
    "is_featured", "is_baku", "has_metro_tag",
]


def _derived(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["area_m2"] = df["area_m2"]
    out["log_area"] = np.log(df["area_m2"])
    out["rooms"] = df["rooms"]
    out["area_per_room"] = df["area_m2"] / df["rooms"]
    out["floor"] = df["floor"]
    out["total_floors"] = df["total_floors"]
    out["floor_ratio"] = df["floor"] / df["total_floors"]
    out["is_first_floor"] = (df["floor"] == 1).astype(float).where(df["floor"].notna())
    out["is_top_floor"] = (df["floor"] == df["total_floors"]).astype(float).where(df["floor"].notna())
    out["land_m2"] = df["land_m2"]
    out["lat"] = df["lat"]
    out["lng"] = df["lng"]
    out["dist_center_km"] = haversine_km(df["lat"], df["lng"], CENTER_LAT, CENTER_LNG)
    for c in BINARY_FEATURES:
        out[c] = df[c].astype(float)
    return out


class Featurizer:
    """
    Turns a cleaned frame into a numeric matrix. All learned state
    (imputation medians, category levels) comes from the frame passed to
    `fit` — always the TRAINING split — so nothing leaks from val/test.

    Encoding choice: one-hot. Trees split on single indicator columns
    easily and the SVM needs a numeric, scale-comparable input; ordinal codes
    would impose a fake order on districts. Rare districts (< min_count train
    rows) are pooled into "other" to avoid hundreds of near-empty columns.
    """

    def __init__(self, min_count: int = 150):
        self.min_count = min_count

    def fit(self, df: pd.DataFrame) -> "Featurizer":
        base = _derived(df)
        self.medians_ = base.median(numeric_only=True)
        self.missing_cols_ = [c for c in ["rooms", "floor", "total_floors", "repaired"]
                              if base[c].isna().any()]
        self.categories_ = sorted(df["category"].dropna().unique())
        vc = df["raion"].value_counts()
        self.raions_ = sorted(vc[vc >= self.min_count].index)
        vc = df["location"].value_counts()
        self.locations_ = sorted(vc[vc >= self.min_count].index)
        self.feature_names_ = (
            list(base.columns)
            + [f"missing_{c}" for c in self.missing_cols_]
            + [f"cat={c}" for c in self.categories_]
            + [f"raion={r}" for r in self.raions_] + ["raion=other"]
            + [f"loc={l}" for l in self.locations_] + ["loc=other"]
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        base = _derived(df)
        flags = {f"missing_{c}": base[c].isna().astype(float) for c in self.missing_cols_}

        base = base.fillna(self.medians_)
        parts = [base, pd.DataFrame(flags, index=df.index)]
        parts.append(pd.DataFrame({f"cat={c}": (df["category"] == c).astype(float)
                                   for c in self.categories_}, index=df.index))
        r = df["raion"].where(df["raion"].isin(self.raions_), "other")
        parts.append(pd.DataFrame({f"raion={v}": (r == v).astype(float)
                                   for v in self.raions_ + ["other"]}, index=df.index))
        l = df["location"].where(df["location"].isin(self.locations_), "other")
        parts.append(pd.DataFrame({f"loc={v}": (l == v).astype(float)
                                   for v in self.locations_ + ["other"]}, index=df.index))
        X = pd.concat(parts, axis=1)[self.feature_names_]
        return X.to_numpy(dtype=float)


def make_features(df_train: pd.DataFrame, *others: pd.DataFrame):
    """
    Fit the featurizer on TRAIN, transform every split.
    Returns (feature_names, X_train, X_other_1, ...) and the fitted Featurizer
    as the last element.
    """
    fz = Featurizer().fit(df_train)
    return (fz.feature_names_, fz.transform(df_train),
            *[fz.transform(o) for o in others], fz)


def make_target(df: pd.DataFrame) -> np.ndarray:
    """Task A target: log(price). Prices are heavy-tailed (skew ~ 4)."""
    return np.log(df["price"].to_numpy(dtype=float))



def make_tier_label(y_price, threshold: float | None = None):
    """
    Task B label: premium = 1 (price > threshold), standard = 0.
    Call with threshold=None on the TRAINING prices only -> threshold is the
    training median; pass that threshold explicitly for val/test.
    Works on raw or log prices (median is invariant to the monotone log).
    """
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    return (y_price > threshold).astype(int), threshold


def _price_bins(y, n_bins: int = 10) -> np.ndarray:
    """Rank-based price deciles used ONLY to stratify the split assignment."""
    ranks = np.argsort(np.argsort(y, kind="stable"), kind="stable")
    return (ranks * n_bins // len(y)).astype(int)


def split_indices(y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic stratified split. Strata = price deciles, so every split
    has the same price distribution and therefore (after thresholding at the
    TRAIN median) the same ~50/50 tier balance -> stratified for Task B.
    Returns (idx_train, idx_val, idx_test).
    """
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    bins = _price_bins(y)
    tr, va, te = [], [], []
    for b in np.unique(bins):
        idx = rng.permutation(np.flatnonzero(bins == b))
        n_te = int(round(test_size * len(idx)))
        n_va = int(round(val_size * len(idx)))
        te.append(idx[:n_te]); va.append(idx[n_te:n_te + n_va]); tr.append(idx[n_te + n_va:])
    return tuple(np.sort(np.concatenate(p)) for p in (tr, va, te))


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """Array version of split_indices: (X_tr, y_tr, X_val, y_val, X_te, y_te)."""
    i_tr, i_va, i_te = split_indices(y, val_size, test_size, seed)
    X, y = np.asarray(X), np.asarray(y)
    return X[i_tr], y[i_tr], X[i_va], y[i_va], X[i_te], y[i_te]


def stratified_kfold(y, n_splits: int = 5, seed: int = SEED):
    """Yield (train_idx, test_idx) folds stratified on price deciles."""
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    bins = _price_bins(y)
    fold_of = np.empty(len(y), dtype=int)
    for b in np.unique(bins):
        idx = rng.permutation(np.flatnonzero(bins == b))
        fold_of[idx] = np.arange(len(idx)) % n_splits
    for k in range(n_splits):
        yield np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)


def standardize(X_tr, *others):
    """
    z-score with TRAIN mean/std only; constant columns get std = 1.
    Returns the scaled arrays in the order given.
    """
    X_tr = np.asarray(X_tr, dtype=float)
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    out = [(X_tr - mu) / sd] + [(np.asarray(o, dtype=float) - mu) / sd for o in others]
    return out if others else out[0]


def load_clean(path: str = DATA_PATH, verbose: bool = False) -> pd.DataFrame:
    return clean(load_raw(path), verbose=verbose)
