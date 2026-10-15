import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from ortho.french import momentum_12_1, rolling_beta
from ortho.metrics import (
    bootstrap_mean_ci,
    nw_mean_tstat,
    one_way_turnover,
    paired_sharpe_diff_bootstrap,
    quintile_weights,
    spearman,
)
from ortho.simulate import one_path


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 6, 200).astype(float)
    y = rng.normal(size=200)
    assert abs(spearman(x, y) - stats.spearmanr(x, y)[0]) < 1e-12


def _returns(n=60, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame(rng.normal(0.01, 0.05, (n, k)), index=idx, columns=list("abcde"))


def test_momentum_window_is_t12_to_t2():
    r = _returns()
    sig = momentum_12_1(r)
    t = 30
    expected = (1 + r.iloc[t - 12 : t - 1]).prod() - 1  # rows t-12 .. t-2 inclusive
    assert np.allclose(sig.iloc[t].to_numpy(), expected.to_numpy())


def test_momentum_ignores_months_t_minus_1_and_later():
    r = _returns()
    t = 30
    base = momentum_12_1(r).iloc[t].to_numpy()
    r2 = r.copy()
    r2.iloc[t - 1 :] += 0.5  # shock t-1, t and everything after
    assert np.allclose(momentum_12_1(r2).iloc[t].to_numpy(), base)
    r3 = r.copy()
    r3.iloc[t - 2] += 0.5  # month t-2 is inside the window and must matter
    assert not np.allclose(momentum_12_1(r3).iloc[t].to_numpy(), base)


def test_rolling_beta_uses_only_data_through_t_minus_1():
    r = _returns(n=80)
    mkt = pd.Series(np.random.default_rng(1).normal(0, 0.04, 80), index=r.index)
    t = 60
    base = rolling_beta(r, mkt).iloc[t].to_numpy()
    r2, m2 = r.copy(), mkt.copy()
    r2.iloc[t:] += 1.0
    m2.iloc[t:] += 1.0
    assert np.allclose(rolling_beta(r2, m2).iloc[t].to_numpy(), base)
    r3 = r.copy()
    r3.iloc[t - 1] += 1.0
    assert not np.allclose(rolling_beta(r3, mkt).iloc[t].to_numpy(), base)


def test_rolling_beta_recovers_known_beta():
    rng = np.random.default_rng(2)
    n = 120
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    mkt = pd.Series(rng.normal(0, 0.04, n), index=idx)
    ex = pd.DataFrame({"a": 1.3 * mkt + rng.normal(0, 0.005, n)}, index=idx)
    assert abs(rolling_beta(ex, mkt).iloc[-1, 0] - 1.3) < 0.1


def test_quintile_weights_are_dollar_neutral():
    w = quintile_weights(np.random.default_rng(3).normal(size=49))
    assert abs(w[w > 0].sum() - 1) < 1e-12 and abs(w[w < 0].sum() + 1) < 1e-12


def test_first_month_turnover_is_one_way_half_of_gross():
    w = quintile_weights(np.random.default_rng(4).normal(size=49))
    assert abs(one_way_turnover(w, None) - 1.0) < 1e-12


def test_newey_west_close_to_plain_t_for_iid():
    x = np.random.default_rng(5).normal(0.05, 0.1, 2000)
    mu, t = nw_mean_tstat(x, lags=4)
    plain = x.mean() / (x.std(ddof=1) / np.sqrt(x.size))
    assert abs(t - plain) / plain < 0.1


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(6).normal(0.05, 0.1, 300)
    lo, hi = bootstrap_mean_ci(x, n_boot=500, seed=1)
    assert lo < x.mean() < hi


def test_paired_bootstrap_identical_series_has_zero_diff():
    r = np.random.default_rng(7).normal(0.01, 0.04, 240)
    obs, lo, hi, _ = paired_sharpe_diff_bootstrap(r, r, n_boot=200)
    assert obs == 0 and lo == 0 and hi == 0


def test_simulation_smoke_and_determinism():
    a = one_path(1, 50, 12.0)
    b = one_path(1, 50, 12.0)
    assert a == b and set(a) == {"raw", "resid"}
