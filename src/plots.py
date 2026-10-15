"""plots.py — shared matplotlib style + save helper (figures -> report/figures/)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = "report/figures"

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # blue, orange, aqua
C4, C5 = "#4a3aa7", "#e34948"                  # violet, red
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_CMAP = "Blues"                             

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def savefig(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    plt.close(fig)
    return path
