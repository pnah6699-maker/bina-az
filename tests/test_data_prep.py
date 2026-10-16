"""Tests for parsing, leakage-free splitting and scaling."""

import numpy as np
import pandas as pd

from src import data_prep as dp


def test_parse_area_units():
    s = pd.Series(["145 m²", "4.5 sot", "60 m²", None])
    np.testing.assert_allclose(dp._parse_area(s), [145, 450, 60, np.nan])


def test_parse_floor():
    f, tot = dp._parse_floor(pd.Series(["7 / 9", "1/5", None]))
    np.testing.assert_allclose(f, [7, 1, np.nan])
    np.testing.assert_allclose(tot, [9, 5, np.nan])


def test_parse_raion():
    s = pd.Series(["Neftçilər m.* Nizami r.* 8-ci kilometr q.", "Səbail r."])
    assert dp._parse_raion(s).tolist() == ["Nizami r.", "Səbail r."]


def test_split_is_deterministic_disjoint_and_complete():
    y = np.random.default_rng(0).lognormal(12, 0.5, 1000)
    a = dp.split_indices(y)
    b = dp.split_indices(y)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    allidx = np.concatenate(a)
    assert len(allidx) == len(np.unique(allidx)) == 1000
    assert abs(len(a[2]) - 150) <= 10


def test_tier_threshold_comes_from_train_only():
    y_tr = np.array([1.0, 2.0, 3.0, 4.0])
    y_te = np.array([100.0, 0.0])
    lab_tr, thr = dp.make_tier_label(y_tr)
    lab_te, thr2 = dp.make_tier_label(y_te, thr)
    assert thr == thr2 == 2.5
    assert lab_tr.tolist() == [0, 0, 1, 1] and lab_te.tolist() == [1, 0]


def test_standardize_uses_train_stats():
    tr = np.array([[0.0, 5.0], [2.0, 5.0]])
    te = np.array([[4.0, 7.0]])
    s_tr, s_te = dp.standardize(tr, te)
    np.testing.assert_allclose(s_tr.mean(0), 0)
    np.testing.assert_allclose(s_te, [[3.0, 2.0]])  # const col: std := 1


def test_kfold_partitions():
    y = np.random.default_rng(1).normal(size=503)
    seen = np.concatenate([te for _, te in dp.stratified_kfold(y, 5)])
    assert sorted(seen.tolist()) == list(range(503))


def test_leakage_columns_are_dropped():
    assert set(dp.LEAKAGE_COLUMNS) == {"unit_price", "total_price"}
    raw = pd.DataFrame({
        "id_x": ["a"], "estate_rel_url_x": ["/items/1"], "datetime_scrape_x": ["2024"],
        "price": [100000.0], "currency_x": ["AZN"], "unit_price": ["1 000 AZN/m²"],
        "total_price": [100000.0], "Kateqoriya": ["Yeni tikili"], "Sahə": ["100 m²"],
        "Torpaq sahəsi": [None], "Mərtəbə": ["3 / 9"], "Otaq sayı": [3.0],
        "Təmir": ["var"], "Çıxarış": ["var"], "İpoteka": [None], "location": ["X"],
        "city": ["bakı"], "lat": [40.4], "lng": [49.85], "owner_title": ["mülkiyyətçi"],
        "products_label": [None], "vip": [None], "featured": [None],
        "extra_info": ["Nəsimi r."],
    })
    out = dp.clean(raw)
    assert len(out) == 1
    assert not {"unit_price", "total_price"} & set(out.columns)
    names, X, fz = dp.make_features(out)
    assert not any("price" in n for n in names)
