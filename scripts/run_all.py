"""Run the French study, the Monte Carlo power tables, and the figures. Writes results/.

    python scripts/run_all.py            # full run (200 seeds, ~10 min on one core)
    python scripts/run_all.py --quick    # 40 seeds, for a smoke test
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ortho.french import format_report, run_books
from ortho.simulate import power_study

SIGNALS = [0, 25, 50, 75]
SIZE_VOLS = [0.01, 0.03, 0.07]  # 0.07 is the original, deliberately heavy contamination


def power_table(rows: list[dict]) -> list[str]:
    lines = ["signal  raw_keep  resid_keep  median_net_sharpe_raw  median_net_sharpe_resid  P(resid>raw)"]
    for r in rows:
        lines.append(
            f"{r['signal']:6d}  {r['raw_keep']:8.1%}  {r['resid_keep']:10.1%}  "
            f"{r['raw_median_sharpe']:21.2f}  {r['resid_median_sharpe']:24.2f}  {r['resid_beats_raw']:12.1%}"
        )
    return lines


def main() -> None:
    quick = "--quick" in sys.argv
    n_seeds = 40 if quick else 200
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    result = run_books(cost_bps=10.0)
    text = format_report(result)
    result["panel"].to_csv(out / "french_monthly.csv")
    print(text)

    lines = [
        f"Simulation power study. {n_seeds} seeds, 80 names, 240 months, cost 12 bps one-way.",
        "Keep rule: |IC t| >= 2 and net Sharpe >= 0.40. Betas are 36-month estimates.",
        "signal=0 is the false-keep rate. Higher signal is power, not a backtest.",
        "size_vol is the monthly vol of the size factor (contamination dial). SMB's realized monthly vol is roughly 3%.",
    ]
    for vol in SIZE_VOLS:
        rows = power_study(SIGNALS, n_seeds=n_seeds, cost_bps=12.0, size_vol=vol)
        lines += ["", f"size_vol = {vol:.2f}"] + power_table(rows)
    power_text = "\n".join(lines)
    print()
    print(power_text)
    if not quick:
        (out / "REPORT.txt").write_text(text + "\n\n" + power_text + "\n")
        from scripts.make_figures import make_figures

        make_figures(result["panel"], out)


if __name__ == "__main__":
    main()
