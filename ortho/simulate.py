"""Monte Carlo power of a pre-committed keep/reject rule.

The panel is planted. It is not a backtest. The question is: at what latent
signal strength does the rule keep a true residual effect, and how often does
it keep pure noise?

Contamination is large on purpose. The size factor is strong and nearly
unforecastable, so it adds noise to a 12-1 score without predicting next month's
residual return. Neutralizing on the observable size characteristic and on a
*rolling estimated* beta (not the population beta) is what the researcher does.

Latent state is an AR(1) updated within month t and enters that month's return, so
it is predictable from earlier months only through its persistence (0.65). The 12-1
score used for holding month t is the compounded return from t-12 through t-2.

`size_vol` (monthly vol of the size factor) is the contamination dial. The headline
power gap between raw and residual books depends on it, so power_study takes it as an
argument and the runner reports a sensitivity table instead of a single point. Note the
size loading is exactly -size_z, and size_z is what the researcher residualizes on, so
size is the easy case by construction.
"""

from __future__ import annotations

import numpy as np

from ortho.metrics import ols_residual2, one_way_turnover, quintile_weights, sharpe, spearman

N = 80
T = 240
INDUSTRIES = 8
BURN = 48  # need 36m beta + 12m momentum


def _gauss(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal(n)


def one_path(seed: int, signal: float, cost_bps: float, size_vol: float = 0.07) -> dict:
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.55, 1.7, size=N)
    log_size = rng.normal(0, 1.0, size=N)
    size_z = (log_size - log_size.mean()) / log_size.std()
    industry = np.arange(N) % INDUSTRIES
    # SMB-style: small names load positively on the size factor.
    size_load = -size_z

    kappa = (signal / 100.0) * 0.04
    rets = np.zeros((T, N))
    latent = np.zeros(N)
    size_fac = 0.0
    for t in range(T):
        mkt = rng.normal(0.005, 0.045)
        # Strong, almost unpredictable size shock. It pollutes 12-1 but does not
        # forecast next month, so raw momentum's IC should fall relative to residual.
        size_fac = 0.05 * size_fac + rng.normal(0.0, size_vol)
        ind = rng.normal(0.0, 0.03, size=INDUSTRIES)
        latent = 0.65 * latent + rng.normal(0.0, 1.0, size=N)
        idio = rng.normal(0.0, 0.08, size=N)
        rets[t] = mkt * beta + size_fac * size_load + ind[industry] + kappa * latent + idio

    # Estimated beta through t-1, 36-month window. Not the true beta.
    est_beta = np.full((T, N), np.nan)
    for t in range(36, T):
        x = rets[t - 36 : t].mean(axis=1)  # cross-sectional mean as market proxy
        x = x - x.mean()
        var = np.dot(x, x)
        if var <= 0:
            continue
        window = rets[t - 36 : t]
        # cov(r_i, mkt) / var
        est_beta[t] = (window * x[:, None]).sum(axis=0) / var

    raw_s = np.full((T, N), np.nan)
    resid_s = np.full((T, N), np.nan)
    gross_r = np.full(T, np.nan)
    # precompute compounded 11-month returns ending at each month
    gross = 1.0 + rets
    log_g = np.log(np.clip(gross, 1e-8, None))
    csum = np.cumsum(log_g, axis=0)
    for t in range(13, T):
        # window t-12 .. t-2 inclusive
        end = t - 2
        start = t - 12
        prev = csum[start - 1] if start - 1 >= 0 else 0.0
        raw_s[t] = np.exp(csum[end] - prev) - 1.0
        resid_s[t] = ols_residual2(raw_s[t], est_beta[t], size_z)
        # industry demean the residual
        demeaned = resid_s[t].copy()
        for k in range(INDUSTRIES):
            m = industry == k
            if np.isfinite(demeaned[m]).sum() >= 3:
                demeaned[m] = demeaned[m] - np.nanmean(demeaned[m])
        resid_s[t] = demeaned

    out = {}
    for name, sig in (("raw", raw_s), ("resid", resid_s)):
        ics = []
        nets = []
        prev_w = None
        for t in range(BURN, T - 1):
            ics.append(spearman(sig[t], rets[t]))  # holding month is t; signal indexed at t
            w = quintile_weights(sig[t])
            if w is None:
                continue
            g = float(np.dot(w, rets[t]))
            to = one_way_turnover(w, prev_w)
            prev_w = w
            nets.append(g - to * (cost_bps / 10000.0))
        ics_a = np.asarray(ics, dtype=float)
        nets_a = np.asarray(nets, dtype=float)
        mu = float(np.nanmean(ics_a))
        sd = float(np.nanstd(ics_a, ddof=1))
        n = int(np.isfinite(ics_a).sum())
        tstat = mu / (sd / np.sqrt(n)) if sd > 0 and n > 5 else np.nan
        sh = sharpe(nets_a)
        # Same pre-committed rule as the write-up. Sharpe SE on 16y is ~0.25, so a 0.40
        # cutoff is still inside a noisy region; the power study is about that rule.
        keep = bool(np.isfinite(tstat) and abs(tstat) >= 2.0 and np.isfinite(sh) and sh >= 0.40)
        out[name] = {"mean_ic": mu, "t": float(tstat), "net_sharpe": float(sh) if np.isfinite(sh) else np.nan, "keep": keep}
    return out


def power_study(
    signals: list[int], n_seeds: int = 200, cost_bps: float = 12.0, base_seed: int = 1000, size_vol: float = 0.07
) -> list[dict]:
    rows = []
    for sig in signals:
        raw_k = resid_k = 0
        raw_sh, resid_sh = [], []
        for i in range(n_seeds):
            path = one_path(base_seed + sig * 10007 + i, sig, cost_bps, size_vol)
            raw_k += int(path["raw"]["keep"])
            resid_k += int(path["resid"]["keep"])
            raw_sh.append(path["raw"]["net_sharpe"])
            resid_sh.append(path["resid"]["net_sharpe"])
        rows.append(
            {
                "signal": sig,
                "size_vol": size_vol,
                "n": n_seeds,
                "raw_keep": raw_k / n_seeds,
                "resid_keep": resid_k / n_seeds,
                "raw_median_sharpe": float(np.nanmedian(raw_sh)),
                "resid_median_sharpe": float(np.nanmedian(resid_sh)),
                "resid_beats_raw": float(np.mean(np.asarray(resid_sh) > np.asarray(raw_sh))),
            }
        )
    return rows
