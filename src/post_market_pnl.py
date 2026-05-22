"""Post-market PnL recorder.

Runs after A-share close (~16:30 Beijing = 08:30 UTC). Reads the morning's
latest_signal.json marker; for each triggered strategy fetches today's futures
open & close from AKShare, computes realized PnL net of friction, appends to
realized_pnl_log.csv, and posts a comment on the morning's GitHub Issue.
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
LOG_FILE = REPO_ROOT / "realized_pnl_log.csv"
MARKER_FILE = REPO_ROOT / "latest_signal.json"
PNL_MARKER_FILE = REPO_ROOT / "latest_pnl.json"
BEIJING_TZ = timezone(timedelta(hours=8))

FUTURES_SYMBOLS = {"IF": "IF0", "IC": "IC0", "IM": "IM0", "IH": "IH0"}


def fetch_futures_today(cn_fut: str, target_date) -> dict | None:
    """Fetch latest futures data and return today's open/close, or None if not yet available."""
    df = ak.futures_zh_daily_sina(symbol=FUTURES_SYMBOLS[cn_fut])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    row = df[df["date"] == target_date]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return {
        "date": r["date"].isoformat(),
        "open": float(r["open"]),
        "high": float(r["high"]),
        "low": float(r["low"]),
        "close": float(r["close"]),
        "volume": int(r["volume"]) if pd.notna(r["volume"]) else 0,
    }


def compute_pnl(detail: dict, ohlc: dict) -> dict:
    """Compute realized PnL for one triggered strategy.
    side = -1 (short). entry = open, exit = close.
    """
    side = detail["side"]
    open_p = ohlc["open"]
    close_p = ohlc["close"]
    high_p = ohlc["high"]
    low_p = ohlc["low"]
    mult = detail["multiplier"]
    n = detail["n_contracts"]
    friction_bps = detail["friction_bps"]

    gross_ret = side * (close_p / open_p - 1)
    net_ret = gross_ret - friction_bps / 10000

    notional_per_contract = open_p * mult
    gross_pnl_cny = gross_ret * notional_per_contract * n
    friction_cny = (friction_bps / 10000) * notional_per_contract * n
    net_pnl_cny = gross_pnl_cny - friction_cny

    # Max favorable / adverse excursion intraday (informational)
    if side == -1:
        mfe = (open_p - low_p) / open_p   # best price reached for shorts (low)
        mae = (high_p - open_p) / open_p  # worst (high)
    else:
        mfe = (high_p - open_p) / open_p
        mae = (open_p - low_p) / open_p

    return {
        "entry_open": round(open_p, 2),
        "exit_close": round(close_p, 2),
        "intraday_high": round(high_p, 2),
        "intraday_low": round(low_p, 2),
        "gross_ret_pct": round(gross_ret * 100, 4),
        "net_ret_pct": round(net_ret * 100, 4),
        "gross_pnl_cny": int(gross_pnl_cny),
        "net_pnl_cny": int(net_pnl_cny),
        "friction_cny": int(friction_cny),
        "n_contracts": n,
        "notional_per_contract_cny": int(notional_per_contract),
        "mfe_pct": round(mfe * 100, 4),
        "mae_pct": round(mae * 100, 4),
    }


def append_log(rows: list[dict], log_file: Path):
    df = pd.DataFrame(rows)
    if log_file.exists():
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, index=False)


def format_text_report(records: list[dict], target_date) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"Post-Market PnL Report — {target_date.isoformat()}")
    lines.append("=" * 70)
    if not records:
        lines.append("\nNo triggered strategies for this date — nothing to record.")
        return "\n".join(lines)

    total_net_pnl = sum(r["net_pnl_cny"] for r in records)
    total_friction = sum(r["friction_cny"] for r in records)
    winners = sum(1 for r in records if r["net_pnl_cny"] > 0)

    lines.append(f"\nTrades closed: {len(records)} | Winners: {winners} | "
                 f"Net PnL: {total_net_pnl:+,} CNY | Friction paid: {total_friction:,} CNY")
    lines.append("")
    for r in records:
        sign = "✅" if r["net_pnl_cny"] > 0 else "❌"
        lines.append(f"  {sign} {r['futures']:3s} ({r['strategy']:18s})  "
                     f"open={r['entry_open']:>8.2f}  close={r['exit_close']:>8.2f}  "
                     f"ret={r['net_ret_pct']:+6.3f}%  PnL={r['net_pnl_cny']:+,} CNY  "
                     f"(qty={r['n_contracts']})")
    lines.append("")
    lines.append("Intraday MFE/MAE (max favorable / adverse move):")
    for r in records:
        lines.append(f"  {r['futures']:3s}  MFE={r['mfe_pct']:+5.2f}%  MAE={r['mae_pct']:+5.2f}%")
    return "\n".join(lines)


def format_md_report(records: list[dict], target_date) -> str:
    md = []
    md.append(f"## Post-Market PnL — {target_date.isoformat()}\n")
    if not records:
        md.append("_No triggered strategies for this date — nothing to record._")
        return "\n".join(md)

    total_net = sum(r["net_pnl_cny"] for r in records)
    winners = sum(1 for r in records if r["net_pnl_cny"] > 0)
    sign = "🟢" if total_net > 0 else "🔴" if total_net < 0 else "⚪"
    md.append(f"**{sign} Net PnL: {total_net:+,} CNY  |  "
              f"Trades: {len(records)}  |  Winners: {winners}/{len(records)}**\n")

    md.append("| Futures | Strategy | Open | Close | Net Ret | PnL (CNY) | Qty | MFE | MAE |")
    md.append("|---------|----------|------|-------|---------|-----------|-----|-----|-----|")
    for r in records:
        emoji = "✅" if r["net_pnl_cny"] > 0 else "❌"
        md.append(f"| {emoji} **{r['futures']}** | {r['strategy']} | "
                  f"{r['entry_open']:.2f} | {r['exit_close']:.2f} | "
                  f"{r['net_ret_pct']:+.3f}% | {r['net_pnl_cny']:+,} | "
                  f"{r['n_contracts']} | {r['mfe_pct']:+.2f}% | {r['mae_pct']:+.2f}% |")
    md.append("")
    md.append(f"_Friction paid: {sum(r['friction_cny'] for r in records):,} CNY  |  "
              f"Total notional: {sum(r['notional_per_contract_cny']*r['n_contracts'] for r in records):,} CNY_")
    return "\n".join(md)


def is_cn_trading_day(date) -> tuple[bool, str]:
    if date.weekday() >= 5:
        return False, "weekend"
    try:
        cal = ak.tool_trade_date_hist_sina()
        cal_set = set(pd.to_datetime(cal["trade_date"]).dt.date)
        if date not in cal_set:
            return False, "in CN holiday calendar"
    except Exception:
        pass
    return True, "ok"


def write_github_outputs(text: str, md: str, marker: dict):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"trades_count={marker['trades_count']}\n")
            f.write(f"net_pnl_cny={marker['net_pnl_cny']}\n")
            f.write(f"run_date={marker['run_date']}\n")
            f.write(f"signal_issue_title={marker.get('signal_issue_title', '')}\n")
    (REPO_ROOT / "pnl_output.txt").write_text(text, encoding="utf-8")
    (REPO_ROOT / "pnl_output.md").write_text(md, encoding="utf-8")


def main():
    now_bj = datetime.now(BEIJING_TZ)
    today = now_bj.date()
    print(f"[info] Beijing now: {now_bj.isoformat(timespec='seconds')}")

    is_trading, reason = is_cn_trading_day(today)
    if not is_trading:
        print(f"[skip] {today} not a CN trading day ({reason})")
        PNL_MARKER_FILE.write_text(json.dumps({
            "run_date": today.isoformat(), "skipped": True, "reason": reason,
            "trades_count": 0, "net_pnl_cny": 0,
        }, indent=2))
        return 0

    # Read morning marker
    if not MARKER_FILE.exists():
        print(f"[error] {MARKER_FILE} not found — has the morning signal run?")
        return 1
    marker = json.loads(MARKER_FILE.read_text(encoding="utf-8"))
    morning_date = marker.get("run_date")
    if morning_date != today.isoformat():
        print(f"[warn] morning marker date={morning_date} != today={today.isoformat()}")
        print("[info] Morning signal may not have run yet today; aborting to avoid stale data.")
        return 0

    triggered = marker.get("triggered_details", [])
    if not triggered:
        print(f"[skip] No triggered strategies this morning — nothing to record.")
        PNL_MARKER_FILE.write_text(json.dumps({
            "run_date": today.isoformat(), "skipped": False,
            "trades_count": 0, "net_pnl_cny": 0,
            "message": "no signals triggered",
        }, indent=2))
        return 0

    # Fetch futures data and compute realized PnL for each trigger
    print(f"[info] Computing realized PnL for {len(triggered)} triggered trades...")
    records = []
    for det in triggered:
        cn_fut = det["futures"]
        ohlc = fetch_futures_today(cn_fut, today)
        if ohlc is None:
            print(f"[warn] No futures data yet for {cn_fut} on {today}; skipping.")
            continue
        pnl = compute_pnl(det, ohlc)
        rec = {
            "run_date_bj": today.isoformat(),
            "strategy": det["strategy"],
            "futures": cn_fut,
            "underlying": det["underlying"],
            "us_signal": det["us_signal"],
            "us_ret_pct": det["us_ret_pct"],
            "threshold_pct": det["threshold_pct"],
            "side": det["side"],
            "friction_bps": det["friction_bps"],
            **pnl,
        }
        records.append(rec)
        print(f"  {cn_fut}: net_ret={rec['net_ret_pct']:+.3f}%  PnL={rec['net_pnl_cny']:+,}")

    if not records:
        print("[skip] No futures data available yet; PnL recording will retry on next run.")
        return 0

    # Persist
    append_log(records, LOG_FILE)
    text = format_text_report(records, today)
    md = format_md_report(records, today)
    print("\n" + text)

    total_net = sum(r["net_pnl_cny"] for r in records)
    pnl_marker = {
        "run_date": today.isoformat(),
        "skipped": False,
        "trades_count": len(records),
        "net_pnl_cny": int(total_net),
        "winners": sum(1 for r in records if r["net_pnl_cny"] > 0),
        "futures": [r["futures"] for r in records],
        "signal_issue_title": f"[Signal {today.isoformat()}] "
                                + "+".join(marker.get("triggered_futures", []))
                                + f" 触发 ({marker.get('triggered_count', 0)}/4)",
    }
    PNL_MARKER_FILE.write_text(json.dumps(pnl_marker, indent=2, ensure_ascii=False))
    write_github_outputs(text, md, pnl_marker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
