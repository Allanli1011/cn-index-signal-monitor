"""Trading strategy logic + backtester (1-day intraday only).

Three strategy families on each (US-signal index, A-share index futures) pair:

  A. naive_short  — Hypothesis-literal: short A-share futures at OPEN when US
                    prior-day close-to-close return <= -threshold; close at A-share close.
  B. bounce_long  — Contrarian intraday: long futures at OPEN when US <= -threshold;
                    close at A-share close. Captures the gap-fade / oversold bounce.
  C. fade_short   — Contrarian intraday: short futures at OPEN when US >= +threshold;
                    close at A-share close.

All trades are intraday: enter at A-share open T, exit at A-share close T.
Costs: round-trip frictions in basis points, applied to each completed trade.
"""
from __future__ import annotations
from pathlib import Path
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import US_ASSETS, CN_FUTURES, REPORT_DIR
from src.alignment import align

REPORT_DIR.mkdir(parents=True, exist_ok=True)
TRADING_DAYS = 244  # avg CN trading days per year

FAMILIES = ("naive_short", "bounce_long", "fade_short")


@dataclass
class StrategySpec:
    name: str
    us_code: str
    cn_fut: str          # IF/IC/IM/IH
    family: str          # naive_short | bounce_long | fade_short
    threshold: float     # absolute US return threshold (e.g. 0.02 = 2%)
    friction_bps: float = 6.0  # round-trip cost in bps
    sample_start: str | None = None  # optional date filter (YYYY-MM-DD)


def _signal(df: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    """Boolean series: True on days the strategy enters at A-share open."""
    us = df["us_ret_cc"]
    if spec.family in ("naive_short", "bounce_long"):
        return (us <= -spec.threshold).fillna(False)
    if spec.family == "fade_short":
        return (us >= spec.threshold).fillna(False)
    raise ValueError(spec.family)


def _side(spec: StrategySpec) -> int:
    return {"naive_short": -1, "bounce_long": +1, "fade_short": -1}[spec.family]


def _trade_returns(df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Per-entry trade-level returns: enter at open T, exit at close T."""
    sig = _signal(df, spec)
    side = _side(spec)
    out_rows = []
    for idx in np.where(sig.values)[0]:
        entry = df["cn_open"].iloc[idx]
        exit_ = df["cn_close"].iloc[idx]
        if pd.isna(entry) or pd.isna(exit_):
            continue
        gross = side * (exit_ / entry - 1)
        net = gross - spec.friction_bps / 10000
        out_rows.append({
            "entry_date": df["cn_date"].iloc[idx],
            "exit_date": df["cn_date"].iloc[idx],
            "entry_price": float(entry), "exit_price": float(exit_),
            "us_ret_signal": float(df["us_ret_cc"].iloc[idx]),
            "gross_ret": float(gross), "net_ret": float(net),
        })
    return pd.DataFrame(out_rows)


def _stats(trades: pd.DataFrame, total_days: int) -> dict:
    if len(trades) == 0:
        return {"n_trades": 0}
    nets = trades["net_ret"].values
    wealth = np.cumprod(1 + nets)
    max_dd = float(((wealth / np.maximum.accumulate(wealth)) - 1).min())
    n = len(nets)
    mean = float(nets.mean())
    std = float(nets.std(ddof=1)) if n > 1 else 0.0
    trades_per_year = n / (total_days / TRADING_DAYS) if total_days else 0
    sharpe = (mean / std * np.sqrt(trades_per_year)) if std > 0 else np.nan
    return {
        "n_trades": int(n),
        "trades_per_year": float(trades_per_year),
        "hit_rate": float((nets > 0).mean()),
        "mean_ret_bps": float(mean * 10000),
        "median_ret_bps": float(np.median(nets) * 10000),
        "std_ret_bps": float(std * 10000),
        "best_ret_bps": float(nets.max() * 10000),
        "worst_ret_bps": float(nets.min() * 10000),
        "total_ret_pct": float((wealth[-1] - 1) * 100),
        "max_dd_pct": float(max_dd * 100),
        "sharpe_ann": float(sharpe) if not np.isnan(sharpe) else None,
        "t_stat": float(mean / (std / np.sqrt(n))) if std > 0 else None,
    }


def backtest(spec: StrategySpec) -> tuple[pd.DataFrame, dict]:
    df = align(spec.us_code, spec.cn_fut, "cn_future")
    if spec.sample_start:
        df = df[df["cn_date"] >= pd.Timestamp(spec.sample_start)].reset_index(drop=True)
    trades = _trade_returns(df, spec)
    total_days = (df["cn_date"].max() - df["cn_date"].min()).days if len(df) else 0
    s = _stats(trades, total_days)
    s.update({
        "name": spec.name, "family": spec.family,
        "us_code": spec.us_code, "cn_fut": spec.cn_fut,
        "threshold": spec.threshold,
        "sample_start": spec.sample_start or "all",
    })
    return trades, s


# ---- experiment grid ----
THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.025]


def run_grid(sample_start: str | None = None, out_suffix: str = "") -> tuple[pd.DataFrame, dict]:
    rows = []
    trade_logs = {}
    for us_code in US_ASSETS:
        for cn_fut in CN_FUTURES:
            for thr in THRESHOLDS:
                for family in FAMILIES:
                    spec = StrategySpec(
                        name=f"{family}_{us_code}_{cn_fut}_thr{thr:.3f}",
                        us_code=us_code, cn_fut=cn_fut,
                        family=family, threshold=thr,
                        sample_start=sample_start,
                    )
                    trades, stats = backtest(spec)
                    rows.append(stats)
                    if stats.get("n_trades", 0) > 0:
                        trade_logs[spec.name] = trades
    summary = pd.DataFrame(rows)
    summary.to_csv(REPORT_DIR / f"backtest_summary{out_suffix}.csv", index=False)
    return summary, trade_logs


def print_top(summary: pd.DataFrame, n_show: int = 20):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.3f}".format)
    keep = ["family", "us_code", "cn_fut", "threshold",
            "n_trades", "hit_rate", "mean_ret_bps", "total_ret_pct",
            "sharpe_ann", "max_dd_pct", "t_stat"]
    for family in summary["family"].unique():
        d = summary[summary["family"] == family].copy()
        d = d[d["n_trades"] >= 30]
        if len(d) == 0:
            print(f"\n## {family}: insufficient trades")
            continue
        d = d.sort_values("sharpe_ann", ascending=False)
        print(f"\n## {family} — top by Sharpe")
        print(d[keep].head(n_show).to_string(index=False))


if __name__ == "__main__":
    summary, logs = run_grid()
    print_top(summary, n_show=12)
    print(f"\nSaved: {REPORT_DIR / 'backtest_summary.csv'}")
