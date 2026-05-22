"""Optimized portfolio: drop IF, merge IC+IM into a 'mid-cap fade' leg, IH stays.

Structure:
  Leg 1 — "Mid-cap fade short": short 0.5 lot IC + 0.5 lot IM at A-share open
          when NDX prior-day close-to-close >= +1%; exit at close.
          (Capital weight 50% of book, split 50/50 between IC and IM legs.)

  Leg 2 — "Large-cap naive short": short IH at A-share open when NDX <= -2%;
          exit at close. (Capital weight 50% of book.)

Final per-future weights: IF=0, IC=25%, IM=25%, IH=50%.

Compares against the original equal-weight (25% × 4) portfolio for context.
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
from src.portfolio_backtest import (
    load_best_specs, daily_panel, portfolio_stats,
    per_strategy_stats, IM_LISTING,
)

mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
TRADING_DAYS = 244


# ---- Weight schemes to compare ----
WEIGHT_SCHEMES = {
    "EqualWeight_4": {"IF": 0.25, "IC": 0.25, "IM": 0.25, "IH": 0.25},
    "Optimized_BC":  {"IF": 0.00, "IC": 0.25, "IM": 0.25, "IH": 0.50},
}


def vec_weights(panel_cols: list[str], scheme: dict[str, float]) -> np.ndarray:
    return np.array([scheme.get(c, 0.0) for c in panel_cols])


def portfolio_equity(panel: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    daily_ret = (panel.values * weights).sum(axis=1)
    return pd.Series(np.cumprod(1 + daily_ret), index=panel.index)


def portfolio_drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def stats_from_daily(daily_ret: np.ndarray, span_yrs: float) -> dict:
    wealth = np.cumprod(1 + daily_ret)
    dd = wealth / np.maximum.accumulate(wealth) - 1
    mean = daily_ret.mean()
    std = daily_ret.std(ddof=1)
    sharpe = (mean / std) * np.sqrt(TRADING_DAYS) if std > 0 else np.nan
    n_active = int((daily_ret != 0).sum())
    return {
        "total_ret_pct": float((wealth[-1] - 1) * 100),
        "ann_ret_pct": float(((wealth[-1]) ** (1/span_yrs) - 1) * 100) if span_yrs > 0 else 0,
        "sharpe_ann": float(sharpe),
        "max_dd_pct": float(dd.min() * 100),
        "daily_std_bps": float(std * 10000),
        "daily_mean_bps": float(mean * 10000),
        "n_active_days": n_active,
        "hit_rate_active": float((daily_ret[daily_ret != 0] > 0).mean()) if n_active else np.nan,
    }


def compare_schemes(panel: pd.DataFrame, schemes: dict) -> pd.DataFrame:
    cols = list(panel.columns)
    span_yrs = len(panel) / TRADING_DAYS
    rows = []
    for name, scheme in schemes.items():
        w = vec_weights(cols, scheme)
        ret = (panel.values * w).sum(axis=1)
        s = stats_from_daily(ret, span_yrs)
        s["scheme"] = name
        s["weights"] = " ".join([f"{c}={scheme.get(c, 0):.0%}" for c in cols])
        rows.append(s)
    df = pd.DataFrame(rows)
    return df


def plot_comparison(panel: pd.DataFrame, schemes: dict):
    cols = list(panel.columns)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2.5, 1]})
    colors = {"EqualWeight_4": "#1f77b4", "Optimized_BC": "#d62728"}
    for name, scheme in schemes.items():
        w = vec_weights(cols, scheme)
        eq = portfolio_equity(panel, w)
        dd = portfolio_drawdown(eq) * 100
        weights_str = ", ".join([f"{c} {scheme.get(c, 0):.0%}" for c in cols if scheme.get(c, 0) > 0])
        s = stats_from_daily((panel.values * w).sum(axis=1), len(panel) / TRADING_DAYS)
        ax1.plot(eq.index, eq.values,
                 label=f"{name}  ({weights_str})  Sharpe={s['sharpe_ann']:.2f}  DD={s['max_dd_pct']:.1f}%",
                 color=colors.get(name, "black"), linewidth=1.8)
        ax2.plot(dd.index, dd.values, color=colors.get(name, "black"), linewidth=1.2, alpha=0.8)
        ax2.fill_between(dd.index, dd.values, 0, color=colors.get(name, "black"), alpha=0.15)

    # Plot individual sub-strategy equity for reference (thin lines)
    for col in cols:
        eq = (1 + panel[col]).cumprod()
        ax1.plot(eq.index, eq.values, label=f"{col} (single)", linewidth=0.8, alpha=0.5)

    ax1.axhline(1, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title("Portfolio equity — Equal-weight 4 strategies vs Optimized (B+C)")
    ax1.set_ylabel("Wealth multiple")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_title("Portfolio drawdown")
    ax2.set_ylabel("Drawdown %")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = REPORT_DIR / "portfolio_optimized_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_monthly_returns(panel: pd.DataFrame, weights: np.ndarray, title: str, filename: str):
    """Monthly return bars for the optimized portfolio."""
    daily_ret = pd.Series((panel.values * weights).sum(axis=1), index=panel.index)
    daily_ret.index = pd.to_datetime(daily_ret.index)
    monthly = (1 + daily_ret).resample("ME").prod() - 1
    yearly = (1 + daily_ret).resample("YE").prod() - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7))
    colors_m = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly]
    ax1.bar(monthly.index, monthly.values * 100, color=colors_m, alpha=0.85, width=20)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title(f"{title} — monthly returns (%)")
    ax1.set_ylabel("Return %")
    ax1.grid(alpha=0.3, axis="y")

    colors_y = ["#2ca02c" if v >= 0 else "#d62728" for v in yearly]
    ax2.bar([d.year for d in yearly.index], yearly.values * 100, color=colors_y, alpha=0.85)
    for i, (yr, v) in enumerate(zip([d.year for d in yearly.index], yearly.values)):
        ax2.text(yr, v*100 + (0.3 if v >= 0 else -0.3), f"{v*100:+.1f}%",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title(f"{title} — yearly returns (%)")
    ax2.set_ylabel("Return %")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = REPORT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("Building daily panel for all 4 strategies (then we re-weight)...")
    specs = load_best_specs()
    panel, _ = daily_panel(specs, sample_start=IM_LISTING)
    cols = list(panel.columns)
    print(f"Panel shape: {panel.shape}, columns: {cols}")

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)

    print("\n=== Weight schemes ===")
    for name, scheme in WEIGHT_SCHEMES.items():
        ws = ", ".join([f"{k}={v:.0%}" for k, v in scheme.items()])
        print(f"  {name}: {ws}")

    print("\n=== Side-by-side comparison ===")
    compare = compare_schemes(panel, WEIGHT_SCHEMES)
    show_cols = ["scheme", "total_ret_pct", "ann_ret_pct", "sharpe_ann",
                 "max_dd_pct", "daily_std_bps", "n_active_days", "hit_rate_active", "weights"]
    print(compare[show_cols].to_string(index=False))
    compare.to_csv(REPORT_DIR / "portfolio_schemes_comparison.csv", index=False)

    print("\nPlotting equity + drawdown comparison...")
    print("  ->", plot_comparison(panel, WEIGHT_SCHEMES))

    # Monthly / yearly for the optimized scheme
    w_opt = vec_weights(cols, WEIGHT_SCHEMES["Optimized_BC"])
    print("Plotting monthly/yearly breakdown of optimized portfolio...")
    print("  ->", plot_monthly_returns(panel, w_opt, "Optimized B+C portfolio",
                                        "portfolio_optimized_monthly.png"))
