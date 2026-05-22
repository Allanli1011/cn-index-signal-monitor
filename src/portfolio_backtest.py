"""Portfolio backtest: combine the 4 best per-futures 1-day strategies into one book.

Capital allocation: equal-weight (each strategy = 25% of book capital).
Daily portfolio return = mean of the 4 sub-strategy daily returns.
Sub-strategy daily return = trade net_ret on signal days, 0 otherwise.

Outputs:
  - Per-strategy and portfolio stats
  - Equity curve PNG (4 sub-strategies + portfolio)
  - Drawdown curve PNG
  - Strategy correlation matrix
  - Daily-level CSV with all 4 strategies' returns and portfolio
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
from src.data_loader import load

mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

IM_LISTING = "2022-07-22"
TRADING_DAYS = 244


def load_best_specs() -> list[StrategySpec]:
    """The 4 best 1-day strategies in IM era from im_era_best_per_future_1day.csv."""
    best = pd.read_csv(REPORT_DIR / "im_era_best_per_future_1day.csv")
    specs = []
    for _, r in best.iterrows():
        specs.append(StrategySpec(
            name=f"{r['cn_fut']}_{r['family']}_{r['us_code']}_{r['threshold']:.3f}",
            us_code=r["us_code"], cn_fut=r["cn_fut"],
            family=r["family"], threshold=float(r["threshold"]),
            sample_start=IM_LISTING,
        ))
    return specs


def _full_cn_trading_calendar(sample_start: str) -> pd.DatetimeIndex:
    """Get all A-share trading dates in the sample window via any CN futures' data."""
    # IF/IC/IH have data from 2017-01-17; IM from 2022-07-22. Use IF as reference since
    # we're starting from IM listing anyway.
    df = load("cn_future", "IF")
    df = df[df["date"] >= pd.Timestamp(sample_start)]
    return pd.DatetimeIndex(df["date"].sort_values().values)


def daily_panel(specs: list[StrategySpec], sample_start: str) -> tuple[pd.DataFrame, dict]:
    """Build a daily DataFrame indexed by ALL A-share trading days in the sample window.
    Columns = each strategy's daily net return. Non-signal days are 0."""
    calendar = _full_cn_trading_calendar(sample_start)
    cols = {}
    trade_logs = {}
    for spec in specs:
        trades, _ = backtest(spec)
        trade_logs[spec.cn_fut] = trades
        if len(trades) == 0:
            cols[spec.cn_fut] = pd.Series(0.0, index=calendar, name=spec.cn_fut)
            continue
        s = pd.Series(trades["net_ret"].values,
                       index=pd.to_datetime(trades["entry_date"]).values,
                       name=spec.cn_fut)
        s = s.groupby(level=0).sum()
        # Reindex onto full trading calendar, fill non-signal days with 0
        s = s.reindex(calendar).fillna(0.0)
        cols[spec.cn_fut] = s

    panel = pd.concat(cols, axis=1).sort_index()
    panel.index.name = "date"
    return panel, trade_logs


def portfolio_stats(panel: pd.DataFrame, weights: list[float] | None = None) -> dict:
    """Compute equity curve, Sharpe, drawdown, etc. for an equal-weight portfolio."""
    cols = list(panel.columns)
    if weights is None:
        weights = [1.0 / len(cols)] * len(cols)
    w = np.array(weights)

    daily_ret = (panel.values * w).sum(axis=1)
    wealth = np.cumprod(1 + daily_ret)
    dd = wealth / np.maximum.accumulate(wealth) - 1

    # Annualization: use trading-day count
    n_days = len(daily_ret)
    span_yrs = n_days / TRADING_DAYS
    mean = daily_ret.mean()
    std = daily_ret.std(ddof=1)
    sharpe = (mean / std) * np.sqrt(TRADING_DAYS) if std > 0 else np.nan

    n_active_days = int((daily_ret != 0).sum())
    return {
        "n_calendar_days": n_days,
        "n_active_days": n_active_days,
        "span_yrs": float(span_yrs),
        "daily_mean_bps": float(mean * 10000),
        "daily_std_bps": float(std * 10000),
        "ann_return_pct": float(((wealth[-1]) ** (1/span_yrs) - 1) * 100) if span_yrs > 0 else 0,
        "total_return_pct": float((wealth[-1] - 1) * 100),
        "sharpe_ann": float(sharpe),
        "max_dd_pct": float(dd.min() * 100),
        "hit_rate_active": float((daily_ret[daily_ret != 0] > 0).mean()) if n_active_days else np.nan,
    }


def per_strategy_stats(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in panel.columns:
        s = panel[col]
        active = s[s != 0]
        wealth = (1 + s).cumprod()
        dd = wealth / wealth.cummax() - 1
        span_yrs = len(s) / TRADING_DAYS
        rows.append({
            "strategy": col,
            "n_signal_days": int(len(active)),
            "hit_rate": float((active > 0).mean()) if len(active) else np.nan,
            "mean_bps_per_trade": float(active.mean() * 10000) if len(active) else 0,
            "total_return_pct": float((wealth.iloc[-1] - 1) * 100),
            "ann_return_pct": float(((wealth.iloc[-1]) ** (1/span_yrs) - 1) * 100) if span_yrs > 0 else 0,
            "sharpe_ann": float((s.mean() / s.std(ddof=1)) * np.sqrt(TRADING_DAYS)) if s.std() > 0 else np.nan,
            "max_dd_pct": float(dd.min() * 100),
        })
    return pd.DataFrame(rows)


def plot_equity_curves(panel: pd.DataFrame, weights: list[float]):
    w = np.array(weights)
    port_daily = (panel.values * w).sum(axis=1)
    port_wealth = pd.Series(np.cumprod(1 + port_daily), index=panel.index, name="Portfolio")

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for col in panel.columns:
        eq = (1 + panel[col]).cumprod()
        ax.plot(eq.index, eq.values, label=f"{col} (single)", linewidth=1.0, alpha=0.7)
    ax.plot(port_wealth.index, port_wealth.values,
            label=f"Portfolio (equal-weight 25% × 4)", linewidth=2.2, color="black")
    ax.axhline(1, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("IM-era (2022-07-22+) — 4 best 1-day strategies + equal-weight portfolio")
    ax.set_ylabel("Wealth multiple (1 = starting capital)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = REPORT_DIR / "portfolio_equity_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_drawdown(panel: pd.DataFrame, weights: list[float]):
    w = np.array(weights)
    port_daily = (panel.values * w).sum(axis=1)
    port_wealth = pd.Series(np.cumprod(1 + port_daily), index=panel.index)
    dd = (port_wealth / port_wealth.cummax() - 1) * 100

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    ax.plot(dd.index, dd.values, color="#d62728", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Portfolio drawdown (max DD = {dd.min():.1f}%)")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = REPORT_DIR / "portfolio_drawdown.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def compute_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    """Correlation of daily returns (including zero days) and of trade-day returns only."""
    daily_corr = panel.corr()
    # Trade-day-only: only when both strategies were active (rare for IH vs others)
    active_mask = (panel != 0)
    print("\nActive-overlap counts between strategy pairs:")
    cols = list(panel.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            n_both = int((active_mask[c1] & active_mask[c2]).sum())
            print(f"  {c1} & {c2}: {n_both} overlapping signal days")
    return daily_corr


if __name__ == "__main__":
    specs = load_best_specs()
    print(f"Loaded {len(specs)} best strategies:")
    for s in specs:
        print(f"  {s.cn_fut}: {s.family}, US={s.us_code}, thr={s.threshold}")

    panel, trade_logs = daily_panel(specs, sample_start=IM_LISTING)
    print(f"\nDaily panel shape: {panel.shape}")
    print(f"Date range: {panel.index.min().date()} to {panel.index.max().date()}")

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)

    print("\n=== Per-strategy stats (computed from daily panel) ===")
    per = per_strategy_stats(panel)
    print(per.to_string(index=False))

    print("\n=== Portfolio stats (equal weight, 25% each) ===")
    port = portfolio_stats(panel)
    for k, v in port.items():
        print(f"  {k:25s}: {v}")

    print("\n=== Daily return correlation ===")
    corr = compute_correlation(panel)
    print(corr.round(3).to_string())

    # Save outputs
    panel.to_csv(REPORT_DIR / "portfolio_daily_panel.csv")
    per.to_csv(REPORT_DIR / "portfolio_per_strategy.csv", index=False)
    pd.DataFrame([port]).to_csv(REPORT_DIR / "portfolio_summary.csv", index=False)
    corr.to_csv(REPORT_DIR / "portfolio_correlation.csv")

    weights = [0.25] * 4
    print("\nPlotting equity curve...")
    print("  ->", plot_equity_curves(panel, weights))
    print("Plotting drawdown...")
    print("  ->", plot_drawdown(panel, weights))
