"""Rolling Sharpe monitor + decay alert.

Reads realized_pnl_log.csv, computes per-strategy rolling Sharpe and cumulative
PnL, compares to historical backtest Sharpe, and generates:
  - rolling_sharpe.png        — chart per strategy
  - cumulative_pnl.png        — chart per strategy
  - weekly_review.md          — markdown report (for GitHub Issue)
  - weekly_review.json        — marker with alert flags

Alert rules:
  - Last 10 trades net PnL < 0 → "RECENT_LOSS_STREAK"
  - Realized Sharpe over last 30 trades < 0 → "SHARPE_NEGATIVE"
  - Realized Sharpe < 50% of historical → "SHARPE_DECAY"
"""
from __future__ import annotations
import os
import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parents[1]
PNL_LOG = REPO_ROOT / "realized_pnl_log.csv"
REVIEW_MD = REPO_ROOT / "weekly_review.md"
REVIEW_MARKER = REPO_ROOT / "weekly_review.json"
BEIJING_TZ = timezone(timedelta(hours=8))

mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

# Historical Sharpe from the freeze (final_best_per_future analysis)
HISTORICAL = {
    "IH_fade_short": {"sharpe": 0.76, "years": 10, "pos_year_pct": 80},
    "IF_fade_short": {"sharpe": 0.68, "years": 10, "pos_year_pct": 70},
    "IC_fade_short": {"sharpe": 0.49, "years": 9,  "pos_year_pct": 56},
    "IM_fade_short": {"sharpe": 0.74, "years": 5,  "pos_year_pct": 80},
}

# Assumed trades-per-year for annualization
TRADES_PER_YEAR = {"IH": 14, "IF": 14, "IC": 14, "IM": 40}

ROLL_WINDOW = 30           # rolling Sharpe window (trades)
LOSS_STREAK_WINDOW = 10    # alert if last N trades cumulative < 0
DECAY_RATIO = 0.5          # alert if realized < 50% of historical


def annualised_sharpe(rets: np.ndarray, trades_per_year: int) -> float:
    if len(rets) < 2:
        return float("nan")
    sd = rets.std(ddof=1)
    if sd == 0:
        return float("nan")
    return (rets.mean() / sd) * np.sqrt(trades_per_year)


def per_strategy_stats(df: pd.DataFrame, strat_name: str) -> dict:
    sub = df[df["strategy"] == strat_name].copy()
    if len(sub) == 0:
        return {"strategy": strat_name, "n_trades": 0}
    sub["net_ret"] = sub["net_ret_pct"] / 100
    rets = sub["net_ret"].values
    cn_fut = sub["futures"].iloc[0]
    tpy = TRADES_PER_YEAR.get(cn_fut, 15)

    realised_sharpe = annualised_sharpe(rets, tpy)
    realised_sharpe_30 = annualised_sharpe(rets[-ROLL_WINDOW:], tpy) if len(rets) >= 5 else float("nan")
    cum_pnl_cny = float(sub["net_pnl_cny"].sum())
    last10_pnl_cny = float(sub["net_pnl_cny"].tail(LOSS_STREAK_WINDOW).sum())

    hit = float((rets > 0).mean())
    mean_bps = float(rets.mean() * 10000)
    worst_pnl = float(sub["net_pnl_cny"].min())
    best_pnl = float(sub["net_pnl_cny"].max())

    historical_sharpe = HISTORICAL.get(strat_name, {}).get("sharpe", float("nan"))

    alerts = []
    if last10_pnl_cny < 0 and len(rets) >= LOSS_STREAK_WINDOW:
        alerts.append("RECENT_LOSS_STREAK")
    if not np.isnan(realised_sharpe_30) and realised_sharpe_30 < 0 and len(rets) >= ROLL_WINDOW:
        alerts.append("SHARPE_NEGATIVE")
    if (not np.isnan(realised_sharpe) and not np.isnan(historical_sharpe)
            and historical_sharpe > 0
            and realised_sharpe < DECAY_RATIO * historical_sharpe
            and len(rets) >= ROLL_WINDOW):
        alerts.append("SHARPE_DECAY")

    return {
        "strategy": strat_name, "futures": cn_fut,
        "n_trades": int(len(rets)),
        "hit_rate": round(hit, 3),
        "mean_ret_bps": round(mean_bps, 2),
        "realised_sharpe": round(realised_sharpe, 3) if not np.isnan(realised_sharpe) else None,
        "rolling_sharpe_30": round(realised_sharpe_30, 3) if not np.isnan(realised_sharpe_30) else None,
        "historical_sharpe": historical_sharpe,
        "cum_pnl_cny": int(cum_pnl_cny),
        "last10_pnl_cny": int(last10_pnl_cny),
        "worst_pnl_cny": int(worst_pnl),
        "best_pnl_cny": int(best_pnl),
        "alerts": alerts,
    }


def plot_rolling_sharpe(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    axes = axes.flatten()
    for i, strat in enumerate(HISTORICAL.keys()):
        ax = axes[i]
        sub = df[df["strategy"] == strat].copy()
        if len(sub) == 0:
            ax.set_title(f"{strat}: no trades yet")
            ax.text(0.5, 0.5, "no trades yet", ha="center", va="center", transform=ax.transAxes)
            continue
        sub = sub.sort_values("run_date_bj").reset_index(drop=True)
        sub["net_ret"] = sub["net_ret_pct"] / 100
        cn_fut = sub["futures"].iloc[0]
        tpy = TRADES_PER_YEAR.get(cn_fut, 15)
        # Rolling Sharpe
        rolls = []
        for k in range(len(sub)):
            start = max(0, k - ROLL_WINDOW + 1)
            window = sub["net_ret"].iloc[start:k+1].values
            rolls.append(annualised_sharpe(window, tpy) if len(window) >= 5 else np.nan)
        sub["rolling_sharpe"] = rolls
        x = pd.to_datetime(sub["run_date_bj"])
        ax.plot(x, sub["rolling_sharpe"], linewidth=1.5, color="#1f77b4", label="rolling 30")
        ax.axhline(0, color="gray", linewidth=0.5)
        hist = HISTORICAL[strat]["sharpe"]
        ax.axhline(hist, color="#2ca02c", linewidth=1, linestyle="--", label=f"historical={hist:.2f}")
        ax.set_title(f"{strat} ({cn_fut})  n={len(sub)}")
        ax.set_ylabel("Annualised Sharpe")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Rolling Sharpe (window={ROLL_WINDOW} trades)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_pnl(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    for i, strat in enumerate(HISTORICAL.keys()):
        ax = axes[i]
        sub = df[df["strategy"] == strat].copy()
        if len(sub) == 0:
            ax.set_title(f"{strat}: no trades yet")
            continue
        sub = sub.sort_values("run_date_bj").reset_index(drop=True)
        cum = sub["net_pnl_cny"].cumsum()
        x = pd.to_datetime(sub["run_date_bj"])
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in sub["net_pnl_cny"]]
        ax.bar(x, sub["net_pnl_cny"], width=1.5, color=colors, alpha=0.6, label="per-trade")
        ax2 = ax.twinx()
        ax2.plot(x, cum, color="#1f77b4", linewidth=1.8, label="cumulative")
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_title(f"{strat}  n={len(sub)}  total={cum.iloc[-1]:+,.0f} CNY")
        ax.set_ylabel("Per-trade PnL")
        ax2.set_ylabel("Cumulative PnL")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Cumulative realized PnL per strategy (net of friction)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def format_md_report(stats_rows: list[dict], today) -> str:
    md = [f"## Weekly Review — {today.isoformat()}\n"]
    md.append(f"_Window: rolling Sharpe over last {ROLL_WINDOW} trades; "
              f"loss-streak window {LOSS_STREAK_WINDOW} trades; "
              f"decay alert if realised < {DECAY_RATIO:.0%} of historical_\n")

    has_alerts = any(s["alerts"] for s in stats_rows)
    if has_alerts:
        md.append("### 🚨 Alerts")
        for s in stats_rows:
            if s["alerts"]:
                md.append(f"- **{s['strategy']}**: {', '.join(s['alerts'])}")
        md.append("")
    else:
        md.append("### ✅ No alerts\n")

    md.append("### Performance summary")
    md.append("| Strategy | Trades | Hit % | Mean bps | Realised Sharpe | Rolling Sharpe | Historical | Cum PnL | Last10 PnL | Alerts |")
    md.append("|----------|--------|-------|----------|----------------|---------------|------------|---------|-----------|--------|")
    for s in stats_rows:
        if s["n_trades"] == 0:
            md.append(f"| {s['strategy']} | 0 | — | — | — | — | "
                      f"{HISTORICAL.get(s['strategy'], {}).get('sharpe', '—')} | — | — | — |")
            continue
        md.append(f"| {s['strategy']} | {s['n_trades']} | {s['hit_rate']*100:.1f}% | "
                  f"{s['mean_ret_bps']:+.1f} | "
                  f"{s.get('realised_sharpe') if s.get('realised_sharpe') is not None else '—'} | "
                  f"{s.get('rolling_sharpe_30') if s.get('rolling_sharpe_30') is not None else '—'} | "
                  f"{s['historical_sharpe']} | "
                  f"{s['cum_pnl_cny']:+,} | {s['last10_pnl_cny']:+,} | "
                  f"{', '.join(s['alerts']) if s['alerts'] else '—'} |")
    md.append("")
    md.append("_Charts: rolling_sharpe.png + cumulative_pnl.png in repo_")
    return "\n".join(md)


def write_github_outputs(md: str, marker: dict):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"has_alerts={'true' if marker['has_alerts'] else 'false'}\n")
            f.write(f"alert_count={marker['alert_count']}\n")
            f.write(f"run_date={marker['run_date']}\n")


def main():
    today = datetime.now(BEIJING_TZ).date()
    print(f"[info] Beijing today: {today.isoformat()}")

    if not PNL_LOG.exists():
        print(f"[warn] {PNL_LOG.name} does not exist — no realized trades yet.")
        marker = {"run_date": today.isoformat(), "has_alerts": False,
                  "alert_count": 0, "message": "no realised trades yet"}
        REVIEW_MARKER.write_text(json.dumps(marker, indent=2))
        return 0

    df = pd.read_csv(PNL_LOG)
    print(f"[info] Loaded {len(df)} realised trades from log")

    stats_rows = [per_strategy_stats(df, strat) for strat in HISTORICAL.keys()]
    md_report = format_md_report(stats_rows, today)
    print("\n" + md_report)
    REVIEW_MD.write_text(md_report, encoding="utf-8")

    if len(df) > 0:
        plot_rolling_sharpe(df, REPO_ROOT / "rolling_sharpe.png")
        plot_cumulative_pnl(df, REPO_ROOT / "cumulative_pnl.png")
        print("[info] Charts saved: rolling_sharpe.png, cumulative_pnl.png")

    has_alerts = any(s["alerts"] for s in stats_rows)
    alert_count = sum(len(s["alerts"]) for s in stats_rows)
    marker = {
        "run_date": today.isoformat(),
        "has_alerts": has_alerts,
        "alert_count": alert_count,
        "alerts_by_strategy": {s["strategy"]: s["alerts"] for s in stats_rows if s["alerts"]},
        "summary": [
            {k: v for k, v in s.items() if k != "alerts"} for s in stats_rows
        ],
    }
    REVIEW_MARKER.write_text(json.dumps(marker, indent=2, ensure_ascii=False))
    write_github_outputs(md_report, marker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
