"""12-1 momentum on Ken French 49 industry portfolios.

Holding month t uses the compounded return from t-12 through t-2 (skip t-1).
Beta is a 36-month OLS of excess return on Mkt-RF using only months through t-1.
The cross-section is residualized on that estimated beta. Oracle betas are not used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ortho.metrics import (
    bootstrap_mean_ci,
    nw_mean_tstat,
    ols_residual,
    one_way_turnover,
    paired_sharpe_diff_bootstrap,
    quintile_weights,
    sharpe,
    sharpe_se,
    spearman,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _monthly_block(path: Path, header_marker: str) -> pd.DataFrame:
    lines = path.read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if header_marker in line)
    header = lines[start + 1]
    rows = []
    for line in lines[start + 2 :]:
        if not line.strip():
            break
        rows.append(line)
    from io import StringIO

    df = pd.read_csv(StringIO("\n".join([header] + rows)))
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = df["date"].astype(str).str.strip()
    df = df[df["date"].str.fullmatch(r"\d{6}")].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date").sort_index()


def load_industries() -> pd.DataFrame:
    df = _monthly_block(DATA / "49_Industry_Portfolios.csv", "Average Value Weighted Returns -- Monthly")
    df.columns = [c.strip() for c in df.columns]
    # French missing code. Returns are in percent.
    df = df.where(df > -99.0) / 100.0
    return df


def load_factors() -> pd.DataFrame:
    """Monthly Mkt-RF, SMB, HML, RF as decimals. The block header is the ",Mkt-RF,..." line itself."""
    lines = (DATA / "F-F_Research_Data_Factors.csv").read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(",Mkt-RF"))
    rows = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        rows.append(line)
    from io import StringIO

    df = pd.read_csv(StringIO("\n".join([lines[start]] + rows)))
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = df["date"].astype(str).str.strip()
    df = df[df["date"].str.fullmatch(r"\d{6}")].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("date").sort_index()
    df.columns = [c.strip() for c in df.columns]
    return df / 100.0


def momentum_12_1(ret: pd.DataFrame) -> pd.DataFrame:
    """Signal for holding month t: compound of returns t-12..t-2. Skips one month, not two."""
    # log-compound via shift. At index t, window is positions t-12 .. t-2 inclusive.
    gross = 1.0 + ret
    # product over an 11-month window ending at t-2: log sum of shifted series.
    log_g = np.log(gross.where(gross > 0))
    # rolling 11 ending at t, then shift by 2 so the window ends at t-2 when the label is t.
    comp = np.exp(log_g.rolling(11, min_periods=11).sum().shift(2)) - 1.0
    return comp


def rolling_beta(excess: pd.DataFrame, mkt: pd.Series, window: int = 36, min_obs: int = 24) -> pd.DataFrame:
    """Beta known at the start of month t uses excess returns through t-1."""
    mkt = mkt.reindex(excess.index)
    betas = pd.DataFrame(index=excess.index, columns=excess.columns, dtype=float)
    x = mkt.to_numpy()
    y = excess.to_numpy()
    n, k = y.shape
    out = np.full((n, k), np.nan)
    for t in range(window, n):
        xs = x[t - window : t]
        ys = y[t - window : t, :]
        for j in range(k):
            col = ys[:, j]
            m = np.isfinite(xs) & np.isfinite(col)
            if m.sum() < min_obs:
                continue
            xx = xs[m]
            yy = col[m]
            var = np.dot(xx - xx.mean(), xx - xx.mean())
            if var <= 0:
                continue
            cov = np.dot(xx - xx.mean(), yy - yy.mean())
            out[t, j] = cov / var
    betas.iloc[:, :] = out
    return betas


def run_books(cost_bps: float = 10.0, start: str = "1970-01-01") -> dict:
    ind = load_industries()
    fac = load_factors()
    common = ind.index.intersection(fac.index)
    ind = ind.loc[common]
    fac = fac.loc[common]
    # Drop industries that are missing more than 5% of months from 1970 on.
    sample = ind.loc[ind.index >= start]
    keep = sample.columns[sample.isna().mean() < 0.05]
    ind = ind[keep]
    excess = ind.sub(fac["RF"], axis=0)
    signal = momentum_12_1(ind)
    beta = rolling_beta(excess, fac["Mkt-RF"])

    dates = ind.index[ind.index >= pd.Timestamp(start)]
    records = []
    prev = {"raw": None, "resid": None}
    for dt in dates:
        sig = signal.loc[dt].to_numpy(dtype=float)
        b = beta.loc[dt].to_numpy(dtype=float)
        r = ind.loc[dt].to_numpy(dtype=float)
        resid = ols_residual(sig, b)
        row = {"date": dt}
        for name, s in (("raw", sig), ("resid", resid)):
            ic = spearman(s, r)
            w = quintile_weights(s)
            if w is None:
                row[f"ic_{name}"] = ic
                continue
            gross = float(np.nansum(w * np.where(np.isfinite(r), r, 0.0)))
            # Do not trade a name whose return is missing.
            if not np.isfinite(r[w != 0]).all():
                row[f"ic_{name}"] = ic
                continue
            to = one_way_turnover(w, prev[name])
            net = gross - to * (cost_bps / 10000.0)
            prev[name] = w
            row[f"ic_{name}"] = ic
            row[f"gross_{name}"] = gross
            row[f"net_{name}"] = net
            row[f"to_{name}"] = to
        records.append(row)
    panel = pd.DataFrame(records).set_index("date")

    def pack(name: str) -> dict:
        ic = panel[f"ic_{name}"].dropna()
        net = panel[f"net_{name}"].dropna()
        mu, tstat = nw_mean_tstat(ic.to_numpy(), lags=4)
        lo, hi = bootstrap_mean_ci(ic.to_numpy(), n_boot=2000, seed=7 if name == "raw" else 11)
        sh = sharpe(net.to_numpy())
        return {
            "name": name,
            "months": int(ic.shape[0]),
            "mean_ic": mu,
            "ic_t_nw": tstat,
            "ic_ci95": (lo, hi),
            "net_sharpe": sh,
            "sharpe_se": sharpe_se(sh, int(net.shape[0])),
            "mean_turnover": float(panel[f"to_{name}"].mean()),
            "gross_sharpe": sharpe(panel[f"gross_{name}"].dropna().to_numpy()),
        }

    d_ic = (panel["ic_resid"] - panel["ic_raw"]).dropna()
    ic_diff, ic_diff_t = nw_mean_tstat(d_ic.to_numpy(), lags=4)
    sr_diff, sr_lo, sr_hi, sr_p = paired_sharpe_diff_bootstrap(
        panel["net_raw"].to_numpy(), panel["net_resid"].to_numpy()
    )
    paired = {
        "ic_diff": ic_diff,
        "ic_diff_t_nw": ic_diff_t,
        "sharpe_diff": sr_diff,
        "sharpe_diff_ci95": (sr_lo, sr_hi),
        "p_sharpe_diff_le_0": sr_p,
        "corr_net": float(panel["net_raw"].corr(panel["net_resid"])),
        "corr_ic": float(panel["ic_raw"].corr(panel["ic_resid"])),
    }

    return {
        "paired": paired,
        "subperiods": subperiod_table(panel),
        "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "n_industries": int(ind.shape[1]),
        "cost_bps": cost_bps,
        "raw": pack("raw"),
        "resid": pack("resid"),
        "panel": panel,
    }


SUBPERIODS = (("1970-1999", "1970-01-01", "1999-12-31"), ("2000-2026", "2000-01-01", "2026-12-31"), ("2010-2026", "2010-01-01", "2026-12-31"))


def subperiod_table(panel: pd.DataFrame) -> list[dict]:
    """Descriptive subsample split, fixed in advance. Not a search over break dates."""
    rows = []
    for label, a, b in SUBPERIODS:
        q = panel.loc[a:b]
        row = {"period": label, "months": int(q["ic_raw"].notna().sum())}
        for name in ("raw", "resid"):
            mu, t = nw_mean_tstat(q[f"ic_{name}"].dropna().to_numpy(), lags=4)
            row[name] = {
                "mean_ic": mu,
                "ic_t_nw": t,
                "net_sharpe": sharpe(q[f"net_{name}"].dropna().to_numpy()),
            }
        rows.append(row)
    return rows


def format_report(result: dict) -> str:
    lines = []
    lines.append(
        f"Ken French 49 industries, value-weighted, {result['start']} to {result['end']} "
        f"({result['n_industries']} industries with <5% missing)."
    )
    lines.append(
        "Signal is compounded 12-1 (months t-12..t-2), held in month t. "
        "Beta is 36-month OLS through t-1, not the population beta. "
        f"Cost is one-way turnover times {result['cost_bps']:.0f} bps, illustrative only: "
        "these are portfolios, not a tradable stock book."
    )
    lines.append("")
    for key in ("raw", "resid"):
        b = result[key]
        lo, hi = b["ic_ci95"]
        label = "Raw 12-1" if key == "raw" else "Beta-residual 12-1"
        lines.append(
            f"{label}: mean rank IC {b['mean_ic']:.4f} "
            f"(NW t {b['ic_t_nw']:.2f}, bootstrap 95% CI [{lo:.4f}, {hi:.4f}]), "
            f"gross Sharpe {b['gross_sharpe']:.2f}, "
            f"net Sharpe {b['net_sharpe']:.2f} (SE ≈ {b['sharpe_se']:.2f}), "
            f"mean one-way turnover {b['mean_turnover']:.2f}, n={b['months']}."
        )
    pr = result["paired"]
    lo, hi = pr["sharpe_diff_ci95"]
    lines.append("")
    lines.append(
        f"Paired comparison (resid minus raw; net-return corr {pr['corr_net']:.2f}, IC corr {pr['corr_ic']:.2f}): "
        f"IC diff {pr['ic_diff']:+.4f} (NW t {pr['ic_diff_t_nw']:.2f}); "
        f"net Sharpe diff {pr['sharpe_diff']:+.3f}, paired 12-month block-bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}], "
        f"P(diff<=0) = {pr['p_sharpe_diff_le_0']:.2f}."
    )
    lines.append("")
    lines.append("Subperiods (fixed split, descriptive):")
    for row in result["subperiods"]:
        r, e = row["raw"], row["resid"]
        lines.append(
            f"  {row['period']} (n={row['months']}): raw IC {r['mean_ic']:.3f} (t {r['ic_t_nw']:.1f}), net Sharpe {r['net_sharpe']:.2f} | "
            f"resid IC {e['mean_ic']:.3f} (t {e['ic_t_nw']:.1f}), net Sharpe {e['net_sharpe']:.2f}"
        )
    lines.append("")
    lines.append(
        "Two pre-committed books, not a factor zoo. A CI that covers zero means the mean IC "
        "is not distinguishable from noise at 95%, even if a single Sharpe looks large. "
        "Industry portfolios have less idiosyncratic noise than stocks, so ICs here run higher "
        "than a stock-level study and should not be pasted onto a stock-trading resume line."
    )
    return "\n".join(lines)
