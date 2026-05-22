"""Backtest visualizations + robustness checks.

Produces:
  - Equity curves for top strategies (per family)
  - Annual return decomposition
  - Combined long/short strategy (bounce_long + fade_short on same pair)
  - Comparison: literal hypothesis (naive_short) vs profitable opposite (fade_short)
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REPORT_DIR
from src.strategy import StrategySpec, backtest

mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110


def annual_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()
    t = trades.copy()
    t["year"] = pd.to_datetime(t["entry_date"]).dt.year
    g = t.groupby("year").agg(
        n_trades=("net_ret", "size"),
        hit_rate=("net_ret", lambda x: float((x > 0).mean())),
        mean_bps=("net_ret", lambda x: float(x.mean()) * 10000),
        total_pct=("net_ret", lambda x: float((np.prod(1 + x) - 1) * 100)),
    )
    return g


def equity_curve(trades: pd.DataFrame) -> pd.Series:
    if len(trades) == 0:
        return pd.Series(dtype=float)
    t = trades.copy().sort_values("exit_date")
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    t["wealth"] = (1 + t["net_ret"]).cumprod()
    return t.set_index("exit_date")["wealth"]


def combined_long_short(us_code: str, cn_fut: str,
                        down_thr: float, up_thr: float,
                        friction_bps: float = 6.0) -> tuple[pd.DataFrame, dict]:
    """Run bounce_long (US<=-down_thr) and fade_short (US>=up_thr) on the same pair,
    combine into a single trade log."""
    spec_long = StrategySpec(name="combo_long", us_code=us_code, cn_fut=cn_fut,
                              family="bounce_long", threshold=down_thr, friction_bps=friction_bps)
    spec_short = StrategySpec(name="combo_short", us_code=us_code, cn_fut=cn_fut,
                               family="fade_short", threshold=up_thr, friction_bps=friction_bps)
    t_long, _ = backtest(spec_long)
    t_short, _ = backtest(spec_short)
    t_long["leg"] = "long"
    t_short["leg"] = "short"
    combined = pd.concat([t_long, t_short], ignore_index=True).sort_values("entry_date")
    # Drop any same-day double signals (extreme cases). The two filters are mutually exclusive
    # in direction so this should not occur, but check.
    nets = combined["net_ret"].values
    if len(nets) == 0:
        return combined, {"n_trades": 0}
    wealth = np.cumprod(1 + nets)
    dd = (wealth / np.maximum.accumulate(wealth)) - 1
    span_yrs = (pd.to_datetime(combined["entry_date"]).max() -
                pd.to_datetime(combined["entry_date"]).min()).days / 365.25
    sharpe = (nets.mean() / nets.std(ddof=1)) * np.sqrt(len(nets) / max(span_yrs, 1e-6)) if nets.std() > 0 else np.nan
    stats = {
        "n_trades": int(len(nets)),
        "hit_rate": float((nets > 0).mean()),
        "mean_ret_bps": float(nets.mean() * 10000),
        "total_ret_pct": float((wealth[-1] - 1) * 100),
        "max_dd_pct": float(dd.min() * 100),
        "sharpe_ann": float(sharpe),
        "t_stat": float(nets.mean() / (nets.std(ddof=1) / np.sqrt(len(nets)))),
        "span_yrs": float(span_yrs),
    }
    return combined, stats


def plot_top_equity_curves():
    """Plot equity curves of the headline strategies in one figure."""
    specs = [
        ("Hypothesis-literal (FAIL)", StrategySpec("A", "NDX", "IH", "naive_short", 0.02), "#d62728"),
        ("Bounce long after US drop", StrategySpec("B", "SPX", "IC", "bounce_long", 0.015), "#1f77b4"),
        ("Fade short after US rally ⭐", StrategySpec("C", "NDX", "IH", "fade_short", 0.020), "#2ca02c"),
        ("Fade short SPX/IF 1.5%",   StrategySpec("D", "SPX", "IF", "fade_short", 0.015), "#9467bd"),
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    for label, spec, color in specs:
        trades, stats = backtest(spec)
        eq = equity_curve(trades)
        if len(eq) == 0:
            continue
        ax.plot(eq.index, eq.values, label=f"{label}  (n={stats['n_trades']}, Sharpe={stats['sharpe_ann']:.2f})",
                color=color, linewidth=1.6)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Equity curves (1 contract, compounded gross PnL, 6 bps friction per trade)")
    ax.set_ylabel("Wealth multiple")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = REPORT_DIR / "equity_curves_top.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_annual_robustness():
    """Annual return bars for the top fade_short strategy + literal hypothesis short."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    for ax, spec, title in [
        (axes[0], StrategySpec("A", "NDX", "IH", "naive_short", 0.02),
                  "Hypothesis-literal: Short IH after NDX ≤ -2% (annual return %)"),
        (axes[1], StrategySpec("C", "NDX", "IH", "fade_short", 0.02),
                  "Best: Short IH after NDX ≥ +2% (annual return %)"),
    ]:
        trades, _ = backtest(spec)
        ann = annual_stats(trades)
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ann["total_pct"]]
        ax.bar(ann.index.astype(str), ann["total_pct"], color=colors, alpha=0.85)
        for i, (yr, row) in enumerate(ann.iterrows()):
            ax.text(i, row["total_pct"] + (0.3 if row["total_pct"] >= 0 else -0.3),
                    f"n={int(row['n_trades'])}", ha="center",
                    va="bottom" if row["total_pct"] >= 0 else "top", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Return %")
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = REPORT_DIR / "annual_robustness.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def run_combined():
    """Test combined bounce_long + fade_short on a few promising pairs."""
    results = []
    pairs = [
        ("NDX", "IH", 0.015, 0.020),
        ("NDX", "IF", 0.015, 0.020),
        ("SPX", "IF", 0.015, 0.015),
        ("SPX", "IH", 0.015, 0.015),
        ("SPX", "IC", 0.015, 0.015),
        ("NDX", "IC", 0.015, 0.020),
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    for us, cnf, dt, ut in pairs:
        combined, stats = combined_long_short(us, cnf, dt, ut)
        results.append({"us": us, "cn_fut": cnf, "down_thr": dt, "up_thr": ut, **stats})
        if stats["n_trades"] > 0:
            eq = equity_curve(combined)
            ax.plot(eq.index, eq.values,
                    label=f"{us}+{cnf}  Sharpe={stats['sharpe_ann']:.2f}  hit={stats['hit_rate']:.0%}  n={stats['n_trades']}",
                    linewidth=1.4)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Combined strategy: bounce_long (US down) + fade_short (US up)")
    ax.set_ylabel("Wealth multiple")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    eqpath = REPORT_DIR / "combined_strategy_equity.png"
    fig.savefig(eqpath, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)
    print("\n=== Combined long+short strategies ===")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(REPORT_DIR / "combined_strategy_summary.csv", index=False)
    return eqpath, df


if __name__ == "__main__":
    print("Plotting top equity curves...")
    print("  ->", plot_top_equity_curves())
    print("Plotting annual robustness...")
    print("  ->", plot_annual_robustness())
    print("Plotting combined strategy...")
    eq, summary = run_combined()
    print("  ->", eq)
