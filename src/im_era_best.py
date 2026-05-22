"""IM-era apples-to-apples 1-day strategy backtest.

Runs the 180-spec grid (3 families x 5 thresholds x 4 futures x 3 US signals)
restricted to 2022-07-22+, then picks the best per-futures spec.
multi_day strategies have been removed from the framework.
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REPORT_DIR, CN_FUTURES
from src.strategy import run_grid, backtest, StrategySpec

IM_LISTING = "2022-07-22"


def find_best_per_future(summary: pd.DataFrame, min_trades: int = 30) -> pd.DataFrame:
    """For each cn_fut, return the spec with highest Sharpe (subject to min trades)."""
    d = summary[summary["n_trades"] >= min_trades].copy()
    d = d.dropna(subset=["sharpe_ann"])
    rows = []
    for cn_fut in CN_FUTURES:
        sub = d[d["cn_fut"] == cn_fut].sort_values("sharpe_ann", ascending=False)
        if len(sub):
            rows.append(sub.iloc[0])
    return pd.DataFrame(rows)


def show_top_per_future(summary: pd.DataFrame, min_trades: int = 30, top_k: int = 5):
    d = summary[summary["n_trades"] >= min_trades].dropna(subset=["sharpe_ann"]).copy()
    cols = ["family", "us_code", "threshold", "n_trades",
            "hit_rate", "mean_ret_bps", "total_ret_pct", "sharpe_ann",
            "max_dd_pct", "t_stat"]
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.3f}".format)
    for cn_fut in CN_FUTURES:
        sub = d[d["cn_fut"] == cn_fut].sort_values("sharpe_ann", ascending=False).head(top_k)
        print(f"\n=== {cn_fut} — top {top_k} (IM-era, 1-day only, n_trades>={min_trades}) ===")
        print(sub[cols].to_string(index=False))


def build_trade_log(best_spec: dict) -> pd.DataFrame:
    spec = StrategySpec(
        name=f"best_{best_spec['cn_fut']}",
        us_code=best_spec["us_code"], cn_fut=best_spec["cn_fut"],
        family=best_spec["family"], threshold=float(best_spec["threshold"]),
        sample_start=IM_LISTING,
    )
    trades, _ = backtest(spec)
    return trades


if __name__ == "__main__":
    print(f"Running IM-era 1-day grid (sample_start={IM_LISTING})...")
    summary, _ = run_grid(sample_start=IM_LISTING, out_suffix="_im_era_1day")
    print(f"Total 1-day specs: {len(summary)}")

    show_top_per_future(summary, min_trades=30, top_k=5)

    best = find_best_per_future(summary)
    print("\n\n===== BEST 1-day strategy per futures (IM-era) =====")
    cols = ["cn_fut", "family", "us_code", "threshold", "n_trades",
            "hit_rate", "mean_ret_bps", "total_ret_pct", "sharpe_ann",
            "max_dd_pct", "t_stat"]
    print(best[cols].to_string(index=False))
    best.to_csv(REPORT_DIR / "im_era_best_per_future_1day.csv", index=False)

    for _, row in best.iterrows():
        log = build_trade_log(row.to_dict())
        out = REPORT_DIR / f"trades_im_era_1day_{row['cn_fut']}.csv"
        log.to_csv(out, index=False)
        print(f"\nTrade log {row['cn_fut']} ({len(log)} trades) -> {out.name}")
