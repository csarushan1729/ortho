"""Rank IC, long-short books, and uncertainty. No look-ahead in the callers."""

from __future__ import annotations

import numpy as np


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties. NaNs stay NaN."""
    out = np.full(a.shape, np.nan)
    mask = np.isfinite(a)
    vals = a[mask]
    if vals.size == 0:
        return out
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(vals.size, dtype=float)
    sorted_vals = vals[order]
    i = 0
    while i < vals.size:
        j = i
        while j + 1 < vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    out[mask] = ranks
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return np.nan
    return float(np.corrcoef(rankdata(x[m]), rankdata(y[m]))[0, 1])


def ols_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of y on [1, x]. NaN where either input is NaN, and NaN everywhere if fewer than 8 valid pairs."""
    out = np.full(y.shape, np.nan)
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 8:
        return out
    yy = y[m]
    xx = x[m]
    X = np.column_stack([np.ones(yy.size), xx])
    try:
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    except np.linalg.LinAlgError:
        return out
    out[m] = yy - X @ beta
    return out


def ols_residual2(y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    out = np.full(y.shape, np.nan)
    m = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    if m.sum() < 10:
        return out
    yy = y[m]
    X = np.column_stack([np.ones(yy.size), x1[m], x2[m]])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    out[m] = yy - X @ beta
    return out


def quintile_weights(signal: np.ndarray) -> np.ndarray | None:
    m = np.isfinite(signal)
    idx = np.flatnonzero(m)
    if idx.size < 15:
        return None
    order = idx[np.argsort(signal[idx], kind="mergesort")]
    q = max(2, idx.size // 5)
    w = np.zeros(signal.shape, dtype=float)
    w[order[:q]] = -1.0 / q
    w[order[-q:]] = 1.0 / q
    return w


def nw_mean_tstat(x: np.ndarray, lags: int = 4) -> tuple[float, float]:
    """Mean and Newey-West t-stat of a 1-d series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < lags + 5:
        return np.nan, np.nan
    mu = float(x.mean())
    e = x - mu
    g0 = float(np.dot(e, e) / n)
    var = g0
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = float(np.dot(e[lag:], e[:-lag]) / n)
        var += 2.0 * w * g
    var = max(var, 0.0)
    se = np.sqrt(var / n)
    t = mu / se if se > 0 else np.nan
    return mu, float(t)


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 12 or np.std(r, ddof=1) == 0:
        return np.nan
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(12.0))


def sharpe_se(ann_sharpe: float, n_months: int) -> float:
    """Approx SE of an annualized Sharpe. Lo (2002) iid form, scaled by sqrt(12)."""
    if n_months < 12 or not np.isfinite(ann_sharpe):
        return np.nan
    # Monthly Sharpe sr_m = SR / sqrt(12). Var(sr_m) ≈ (1 + 0.5 sr_m^2) / T.
    sr_m = ann_sharpe / np.sqrt(12.0)
    var_m = (1.0 + 0.5 * sr_m**2) / n_months
    return float(np.sqrt(var_m) * np.sqrt(12.0))


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def one_way_turnover(w: np.ndarray, prev: np.ndarray | None) -> float:
    if prev is None:
        return float(0.5 * np.abs(w).sum())
    return float(0.5 * np.abs(w - prev).sum())


def paired_sharpe_diff_bootstrap(
    a: np.ndarray, b: np.ndarray, block: int = 12, n_boot: int = 4000, seed: int = 1
) -> tuple[float, float, float, float]:
    """Annualized Sharpe(b) - Sharpe(a) with a paired circular block bootstrap.

    The two return series are resampled with the same block indices, so their
    correlation is preserved. Returns (observed diff, 2.5%, 97.5%, P(diff <= 0)).
    Comparing a gap to each book's own SE ignores that correlation; this does not.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = a.size
    if n < 3 * block:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    offs = np.arange(block)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=nb)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        diffs[i] = sharpe(b[idx]) - sharpe(a[idx])
    obs = sharpe(b) - sharpe(a)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(obs), float(lo), float(hi), float((diffs <= 0).mean())
