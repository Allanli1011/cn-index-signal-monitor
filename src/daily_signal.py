"""Daily A-share index futures signal generator.

Runs before A-share open (~07:00 Beijing time). Fetches the latest US session
data, evaluates each of the 4 strategies, and produces a report.

Strategies (frozen from the stability analysis):
  - IH: fade_short, signal NDX >= +2%   (full-history Sharpe 0.76, 80% pos years)
  - IF: fade_short, signal NDX >= +2%   (full-history Sharpe 0.68, 70% pos years)
  - IC: fade_short, signal SPX >= +1.5% (full-history Sharpe 0.49, 56% pos years)
  - IM: fade_short, signal NDX >= +1%   (IM-era Sharpe 0.74, 5 yrs only)

All 4 strategies share the same operation: short A-share futures at OPEN T,
close at A-share close T (intraday only, no overnight risk).

Designed to run via GitHub Actions cron.
"""
from __future__ import annotations
import os
import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import akshare as ak

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = REPO_ROOT / "signals_log.csv"
MARKER_FILE = REPO_ROOT / "latest_signal.json"
BEIJING_TZ = timezone(timedelta(hours=8))
DEFAULT_CAPITAL_CNY = 1_000_000  # 100 万

# ---- Strategy registry (frozen) ----
STRATEGIES = [
    {
        "name": "IH_fade_short",
        "futures": "IH",
        "underlying": "上证50",
        "us_signal": "NDX",
        "threshold_pct": 2.0,
        "direction": "short_on_us_up",
        "multiplier": 300,
        "margin_pct": 0.15,
        "rank": 1,
        "history_sharpe": 0.76,
        "positive_years_pct": 80,
        "notes": "最稳; 10年里8年正收益; 大盘金融蓝筹反应最强",
    },
    {
        "name": "IF_fade_short",
        "futures": "IF",
        "underlying": "沪深300",
        "us_signal": "NDX",
        "threshold_pct": 2.0,
        "direction": "short_on_us_up",
        "multiplier": 300,
        "margin_pct": 0.15,
        "rank": 2,
        "history_sharpe": 0.68,
        "positive_years_pct": 70,
        "notes": "次稳; 与 IH 同信号但反应弱;",
    },
    {
        "name": "IC_fade_short",
        "futures": "IC",
        "underlying": "中证500",
        "us_signal": "SPX",
        "threshold_pct": 1.5,
        "direction": "short_on_us_up",
        "multiplier": 200,
        "margin_pct": 0.17,
        "rank": 3,
        "history_sharpe": 0.49,
        "positive_years_pct": 56,
        "notes": "用 SPX 阈值低; 2021-2023 连亏需警惕",
    },
    {
        "name": "IM_fade_short",
        "futures": "IM",
        "underlying": "中证1000",
        "us_signal": "NDX",
        "threshold_pct": 1.0,
        "direction": "short_on_us_up",
        "multiplier": 200,
        "margin_pct": 0.18,
        "rank": 4,
        "history_sharpe": 0.74,
        "positive_years_pct": 80,
        "notes": "样本仅5年; 2024年贡献全部收益; 高波动",
    },
]

US_SYMBOLS = {
    "SPX": ".INX",     # S&P 500 via Sina
    "NDX": ".IXIC",    # Nasdaq Composite via Sina
}


# ---- Data fetchers ----
def fetch_us_recent() -> dict[str, dict]:
    """Return {us_code: {date, close, ret_cc, age_days}} for SPX & NDX."""
    today_bj = datetime.now(BEIJING_TZ).date()
    out = {}
    for code, symbol in US_SYMBOLS.items():
        df = ak.index_us_stock_sina(symbol=symbol)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        df["ret_cc"] = df["close"].astype(float).pct_change()
        last = df.iloc[-1]
        age = (today_bj - last["date"]).days
        out[code] = {
            "date": last["date"].isoformat(),
            "close": float(last["close"]),
            "ret_cc": float(last["ret_cc"]),
            "ret_pct": float(last["ret_cc"]) * 100,
            "age_days": int(age),
        }
    return out


def get_cn_trading_dates() -> set:
    """Get all CN trading dates via AKShare's calendar."""
    try:
        cal = ak.tool_trade_date_hist_sina()
        return set(pd.to_datetime(cal["trade_date"]).dt.date)
    except Exception as e:
        print(f"[warn] Failed to fetch CN trading calendar: {e}")
        return set()


def is_cn_trading_day(date) -> tuple[bool, str]:
    """Return (is_trading, reason)."""
    if date.weekday() >= 5:
        return False, f"weekend ({date.strftime('%A')})"
    cal = get_cn_trading_dates()
    if cal and date not in cal:
        return False, "in CN holiday calendar"
    return True, "ok"


# ---- Signal evaluation ----
def suggest_contracts(strat: dict, capital: float) -> tuple[int, float]:
    """Suggest how many contracts to trade for given capital (equal-budget across 4)."""
    per_strategy_budget = capital / 4
    # Approximate notional per contract using a typical mid-range price
    typical_prices = {"IH": 2700, "IF": 4200, "IC": 6300, "IM": 7000}
    price = typical_prices[strat["futures"]]
    notional = price * strat["multiplier"]
    margin_per_contract = notional * strat["margin_pct"]
    n_contracts = max(1, int(per_strategy_budget / margin_per_contract))
    return n_contracts, margin_per_contract


def evaluate_signals(us_returns: dict, capital: float) -> list[dict]:
    rows = []
    for s in STRATEGIES:
        us = us_returns[s["us_signal"]]
        us_ret_pct = us["ret_pct"]
        thr_pct = s["threshold_pct"]

        if s["direction"] == "short_on_us_up":
            triggered = us_ret_pct >= thr_pct
        elif s["direction"] == "short_on_us_down":
            triggered = us_ret_pct <= -thr_pct
        else:
            triggered = False

        n_contracts, margin_per = suggest_contracts(s, capital)
        rows.append({
            "rank": s["rank"],
            "strategy": s["name"],
            "futures": s["futures"],
            "underlying": s["underlying"],
            "us_signal": s["us_signal"],
            "us_date": us["date"],
            "us_close": us["close"],
            "us_ret_pct": round(us_ret_pct, 3),
            "threshold_pct": thr_pct,
            "triggered": bool(triggered),
            "action": "SHORT_AT_OPEN" if triggered else "NO_TRADE",
            "n_contracts_suggested": n_contracts if triggered else 0,
            "margin_per_contract_cny": int(margin_per),
            "notes": s["notes"],
        })
    return rows


# ---- Report formatting ----
def format_report(signals: list[dict], us_returns: dict,
                  run_date_bj, capital: float) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"A-Share Index Futures Signal Report")
    lines.append(f"Run date (Beijing): {run_date_bj.isoformat()}")
    lines.append(f"Capital basis: {capital:,.0f} CNY")
    lines.append("=" * 70)

    lines.append("\n--- Latest US session ---")
    for code, info in us_returns.items():
        warn = " ⚠️ STALE (>5d old)" if info["age_days"] > 5 else ""
        lines.append(f"  {code:4s} @ {info['date']}  close={info['close']:>10.2f}  "
                     f"ret_cc={info['ret_pct']:+6.2f}%  age={info['age_days']}d{warn}")

    triggered = [s for s in signals if s["triggered"]]
    lines.append(f"\n--- Signal summary: {len(triggered)} / 4 triggered ---")
    if not triggered:
        lines.append("  ✗ NO SIGNALS TODAY — stay flat")
    else:
        for s in triggered:
            lines.append(f"  🔴 {s['futures']} ({s['underlying']}): "
                         f"{s['action']}  qty={s['n_contracts_suggested']} 手  "
                         f"trigger={s['us_signal']}={s['us_ret_pct']:+.2f}% ≥ {s['threshold_pct']}%")

    lines.append("\n--- Full strategy table ---")
    df = pd.DataFrame(signals).drop(columns=["notes"])
    lines.append(df.to_string(index=False))

    if triggered:
        lines.append("\n--- Execution plan ---")
        total_margin = sum(s["n_contracts_suggested"] * s["margin_per_contract_cny"]
                            for s in triggered)
        lines.append(f"  Total margin required: {total_margin:,.0f} CNY "
                     f"({100*total_margin/capital:.1f}% of capital)")
        lines.append(f"  Entry: A-share open (09:30 Beijing) via 集合竞价")
        lines.append(f"  Exit:  A-share close (15:00 Beijing) 集合竞价平今")
        lines.append(f"  Stop:  设硬止损 = 开仓价 × 1.005 (亏 0.5% 强平)")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def format_markdown_report(signals: list[dict], us_returns: dict,
                            run_date_bj, capital: float) -> str:
    """Markdown-formatted report for GitHub Issue body."""
    triggered = [s for s in signals if s["triggered"]]
    md = []
    md.append(f"## A-Share Index Futures Signal — {run_date_bj.isoformat()}\n")
    md.append(f"**Capital basis:** {capital:,.0f} CNY\n")

    md.append("### Latest US session")
    md.append("| Index | Date | Close | Return | Age |")
    md.append("|-------|------|-------|--------|-----|")
    for code, info in us_returns.items():
        stale = " ⚠️" if info["age_days"] > 5 else ""
        md.append(f"| {code} | {info['date']} | {info['close']:.2f} | "
                  f"{info['ret_pct']:+.2f}% | {info['age_days']}d{stale} |")
    md.append("")

    md.append(f"### Signals triggered: **{len(triggered)} / 4**\n")
    if not triggered:
        md.append("**No signal today.** Stay flat.\n")
    else:
        md.append("| Futures | Underlying | Action | Qty | Signal | Trigger |")
        md.append("|---------|------------|--------|-----|--------|---------|")
        for s in triggered:
            md.append(f"| **{s['futures']}** | {s['underlying']} | "
                      f"`{s['action']}` | {s['n_contracts_suggested']} 手 | "
                      f"{s['us_signal']}={s['us_ret_pct']:+.2f}% | "
                      f"≥ {s['threshold_pct']}% |")
        md.append("")

        total_margin = sum(s["n_contracts_suggested"] * s["margin_per_contract_cny"]
                            for s in triggered)
        md.append(f"**Total margin required:** {total_margin:,.0f} CNY "
                  f"({100*total_margin/capital:.1f}% of {capital:,.0f})\n")

        md.append("### Execution plan")
        md.append("- **Entry:** 09:15-09:25 集合竞价挂卖单开空")
        md.append("- **Exit:** 14:57+ 集合竞价挂买单平今")
        md.append("- **Stop:** 硬止损 = 开仓价 × 1.005 (-0.5%)\n")

    md.append("### Full strategy table")
    md.append("| Rank | Strategy | Futures | US sig | US ret | Thr | Trig | Notes |")
    md.append("|------|----------|---------|--------|--------|-----|------|-------|")
    for s in signals:
        check = "✅" if s["triggered"] else "❌"
        md.append(f"| {s['rank']} | {s['strategy']} | {s['futures']} | "
                  f"{s['us_signal']} | {s['us_ret_pct']:+.2f}% | "
                  f"{s['threshold_pct']}% | {check} | {s['notes']} |")

    return "\n".join(md)


# ---- IO helpers ----
def append_log(rows: list[dict], log_file: Path, run_date_bj):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "run_date_bj", run_date_bj.isoformat())
    if log_file.exists():
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, index=False)


def write_github_outputs(report: str, md_report: str, marker: dict):
    """Write to GitHub Actions step summary & output."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md_report + "\n")

    # GITHUB_OUTPUT for cross-step variables
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"triggered_count={marker['triggered_count']}\n")
            f.write(f"triggered_list={','.join(marker['triggered_strategies'])}\n")
            f.write(f"run_date={marker['run_date']}\n")

    # Plain-text report saved for issue body fallback
    (REPO_ROOT / "signal_output.txt").write_text(report, encoding="utf-8")
    (REPO_ROOT / "signal_output.md").write_text(md_report, encoding="utf-8")


# ---- Main ----
def main():
    capital = float(os.environ.get("CAPITAL_CNY", DEFAULT_CAPITAL_CNY))
    now_bj = datetime.now(BEIJING_TZ)
    today = now_bj.date()

    print(f"[info] Beijing now: {now_bj.isoformat(timespec='seconds')}")
    print(f"[info] Capital basis: {capital:,.0f} CNY\n")

    is_trading, reason = is_cn_trading_day(today)
    if not is_trading:
        msg = f"[skip] {today} not a CN trading day ({reason})"
        print(msg)
        # Still write marker so action knows
        MARKER_FILE.write_text(json.dumps({
            "run_date": today.isoformat(),
            "skipped": True,
            "reason": reason,
            "triggered_count": 0,
            "triggered_strategies": [],
        }, indent=2))
        return 0

    print("[info] Fetching latest US data via AKShare...")
    us_returns = fetch_us_recent()
    for code, info in us_returns.items():
        print(f"  {code}: {info['date']} close={info['close']:.2f} "
              f"ret={info['ret_pct']:+.2f}% age={info['age_days']}d")

    signals = evaluate_signals(us_returns, capital)
    text_report = format_report(signals, us_returns, today, capital)
    md_report = format_markdown_report(signals, us_returns, today, capital)
    print("\n" + text_report)

    triggered = [s for s in signals if s["triggered"]]
    # Build per-trigger details that the post-market PnL job will need
    triggered_details = []
    for s in triggered:
        # find the matching frozen STRATEGY meta for multiplier/friction
        meta = next((m for m in STRATEGIES if m["name"] == s["strategy"]), None)
        triggered_details.append({
            "strategy": s["strategy"],
            "futures": s["futures"],
            "underlying": s["underlying"],
            "us_signal": s["us_signal"],
            "us_ret_pct": s["us_ret_pct"],
            "threshold_pct": s["threshold_pct"],
            "action": s["action"],
            "side": -1,  # all fade_short
            "n_contracts": s["n_contracts_suggested"],
            "multiplier": meta["multiplier"] if meta else None,
            "margin_pct": meta["margin_pct"] if meta else None,
            "margin_per_contract_cny": s["margin_per_contract_cny"],
            "friction_bps": 6.0,
        })

    marker = {
        "run_date": today.isoformat(),
        "skipped": False,
        "triggered_count": len(triggered),
        "triggered_strategies": [s["strategy"] for s in triggered],
        "triggered_futures": [s["futures"] for s in triggered],
        "triggered_details": triggered_details,
        "us_returns": {k: round(v["ret_pct"], 3) for k, v in us_returns.items()},
    }
    MARKER_FILE.write_text(json.dumps(marker, indent=2, ensure_ascii=False))
    append_log(signals, LOG_FILE, today)
    write_github_outputs(text_report, md_report, marker)

    print(f"\n[info] Log appended to {LOG_FILE.name}")
    print(f"[info] Marker written to {MARKER_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
