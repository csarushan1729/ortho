"""Cumulative net return and rolling 60-month rank IC, raw vs beta-residual."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_figures(panel: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, label in (("raw", "Raw 12-1"), ("resid", "Beta-residual 12-1")):
        net = panel[f"net_{name}"].dropna()
        ax[0].plot(net.index, np.log1p(net).cumsum(), label=label, lw=1.4)
        roll = panel[f"ic_{name}"].rolling(60, min_periods=60).mean()
        ax[1].plot(roll.index, roll, label=label, lw=1.4)
    ax[0].set_ylabel("Cumulative net log return")
    ax[0].set_title("49 French industries, 12-1 quintile long-short, 10 bps one-way (illustrative)")
    ax[0].legend()
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("Rolling 60-month mean rank IC")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "french_books.png", dpi=140)
    plt.close(fig)
