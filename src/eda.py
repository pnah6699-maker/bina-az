"""eda.py — Problem 1(b): exploratory figures on the CLEANED data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .plots import C1, C2, C3, INK2, SEQ_CMAP, plt, savefig

CAT_LABEL = {"new_build": "New build", "old_build": "Old stock", "house_villa": "House / villa"}
CAT_COLOR = {"new_build": C1, "old_build": C2, "house_villa": C3}


def run_eda(df: pd.DataFrame, tier_threshold_azn: float) -> dict:
    price = df["price"].to_numpy()
    out = {
        "n_rows": int(len(df)),
        "price_median": float(np.median(price)),
        "price_mean": float(price.mean()),
        "price_skew": float(pd.Series(price).skew()),
        "logprice_skew": float(pd.Series(np.log(price)).skew()),
        "category_counts": df["category"].value_counts().to_dict(),
    }

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax[0].hist(price / 1e3, bins=120, range=(0, 1500), color=C1, edgecolor="white", linewidth=0.3)
    ax[0].set(xlabel="price (thousand AZN, clipped at 1.5M)", ylabel="listings",
              title=f"Price (skew {out['price_skew']:.1f})")
    ax[1].hist(np.log(price), bins=80, color=C1, edgecolor="white", linewidth=0.3)
    ax[1].axvline(np.log(tier_threshold_azn), color=C2, lw=1.5, ls="--")
    ax[1].text(np.log(tier_threshold_azn) + 0.08, ax[1].get_ylim()[1] * 0.9,
               "tier threshold\n(train median)", color=INK2, fontsize=7, va="top")
    ax[1].set(xlabel="log(price)", title=f"log(price) (skew {out['logprice_skew']:.2f})")
    savefig(fig, "eda_price_distribution")

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    rng = np.random.default_rng(0)
    for cat in ["new_build", "old_build", "house_villa"]:
        d = df[df["category"] == cat]
        d = d.iloc[rng.permutation(len(d))[:3000]]
        ax.scatter(d["area_m2"], d["price"], s=3, alpha=0.35, color=CAT_COLOR[cat],
                   label=CAT_LABEL[cat], linewidths=0)
    ax.set(xscale="log", yscale="log", xlabel="area (m²)", ylabel="price (AZN)",
           title="Area vs price (3k sample / category)")
    ax.legend(markerscale=4, loc="upper left")
    savefig(fig, "eda_area_vs_price")

    
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    rooms = df["rooms"].clip(upper=7)
    groups = [np.log(df.loc[rooms == r, "price"]) for r in range(1, 8)]
    bp = ax.boxplot(groups, tick_labels=["1", "2", "3", "4", "5", "6", "7+"], showfliers=False,
                    patch_artist=True, medianprops={"color": C2, "linewidth": 1.5})
    for b in bp["boxes"]:
        b.set(facecolor="#dbe8f8", edgecolor=C1)
    ax.set(xlabel="rooms", ylabel="log(price)", title="Price by number of rooms")
    savefig(fig, "eda_rooms")

    d = df[df["lat"].between(40.25, 40.62) & df["lng"].between(49.6, 50.35)]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sc = ax.scatter(d["lng"], d["lat"], c=np.log(d["price"] / d["area_m2"]), s=1.2,
                    cmap=SEQ_CMAP, vmin=np.log(600), vmax=np.log(5000), linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    ticks = [700, 1000, 2000, 3000, 5000]
    cb.set_ticks(np.log(ticks)); cb.set_ticklabels([str(t) for t in ticks])
    cb.set_label("price per m² (AZN) — EDA only, never a feature")
    ax.set(xlabel="longitude", ylabel="latitude", title="Location effect: Baku / Absheron")
    ax.set_aspect(1 / np.cos(np.radians(40.4)))
    ax.grid(False)
    savefig(fig, "eda_location")

    top = df["raion"].value_counts().index[:12]
    med = (df[df["raion"].isin(top)].groupby("raion")["price"].median()
           .sort_values() / 1e3)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.barh([r.replace(" r.", "") for r in med.index], med.values, color=C1, height=0.7)
    ax.set(xlabel="median price (thousand AZN)", title="Median price by district")
    ax.grid(axis="y", visible=False)
    savefig(fig, "eda_district")
    out["district_median_price"] = med.to_dict()
    return out
