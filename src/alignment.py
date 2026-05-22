"""Time-zone aware alignment of US signals onto A-share trading days.

Logic
-----
US market closes at 16:00 ET, which is ~04:00-05:00 Beijing time the *next*
calendar day. So a US trading day t's close-to-close return is the freshest
signal available for any A-share trading session on Beijing date >= t+1.

For each A-share date d, we attach the US return whose "availability date"
(US date + 1 day) is the latest <= d. This handles:
  - Beijing Tuesday -> US Monday return (1 calendar day shift)
  - Beijing Monday  -> US Friday return (signal availability date = Saturday,
    which is <= Monday so it gets picked up via merge_asof backward)
  - US holidays (Memorial Day etc.): A-share day reuses the prior US signal
    until the next US session produces a new one. We flag staleness in days.
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import US_ASSETS, CN_INDEX_ASSETS, CN_FUTURES, PROCESSED_DIR
from src.data_loader import load

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _compute_returns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add close-to-close and (if open exists) open-to-close return columns."""
    out = df[["date", "open", "high", "low", "close"]].copy()
    out[f"{prefix}_ret_cc"] = out["close"].pct_change()
    out[f"{prefix}_ret_oc"] = out["close"] / out["open"] - 1
    out[f"{prefix}_ret_co_next_open"] = out["open"].shift(-1) / out["close"] - 1  # close→next open
    return out


def build_us_signal(code: str) -> pd.DataFrame:
    """Load a US asset and produce a signal table keyed on the Beijing date it becomes
    actionable (= US date + 1 calendar day).
    Columns: signal_avail_date, us_code, us_date, us_ret_cc, us_ret_oc, us_close
    """
    df = load("us", code)
    df = _compute_returns(df, "us").rename(columns={"date": "us_date", "close": "us_close"})
    df["signal_avail_date"] = df["us_date"] + pd.Timedelta(days=1)
    df["us_code"] = code
    return df[["signal_avail_date", "us_code", "us_date", "us_close", "us_ret_cc", "us_ret_oc"]]


def build_cn_table(code: str, kind: str) -> pd.DataFrame:
    """Load a CN asset (index or future) and compute its own returns.
    kind in {'cn_index', 'cn_future'}.
    """
    df = load(kind, code)
    df = _compute_returns(df, "cn").rename(columns={"date": "cn_date", "open": "cn_open",
                                                     "high": "cn_high", "low": "cn_low",
                                                     "close": "cn_close"})
    df["cn_code"] = code
    df["cn_kind"] = kind
    return df


def align(us_code: str, cn_code: str, cn_kind: str) -> pd.DataFrame:
    """Return a per-A-share-day frame with the appropriate US signal joined on."""
    us = build_us_signal(us_code).sort_values("signal_avail_date").reset_index(drop=True)
    cn = build_cn_table(cn_code, cn_kind).sort_values("cn_date").reset_index(drop=True)

    merged = pd.merge_asof(
        cn,
        us,
        left_on="cn_date",
        right_on="signal_avail_date",
        direction="backward",
        allow_exact_matches=True,
    )
    # Staleness: how many calendar days between US date and CN date
    merged["signal_age_days"] = (merged["cn_date"] - merged["us_date"]).dt.days
    return merged


def align_all() -> dict[tuple[str, str], pd.DataFrame]:
    """Build the full panel of (us_code, cn_code) aligned tables."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    cn_specs = (
        [(code, "cn_index") for code in CN_INDEX_ASSETS]
        + [(code, "cn_future") for code in CN_FUTURES]
    )
    for us_code in US_ASSETS:
        for cn_code, cn_kind in cn_specs:
            key = (us_code, f"{cn_kind}:{cn_code}")
            df = align(us_code, cn_code, cn_kind)
            out[key] = df
            path = PROCESSED_DIR / f"aligned_{us_code}_{cn_kind}_{cn_code}.parquet"
            df.to_parquet(path, index=False)
    return out


if __name__ == "__main__":
    tables = align_all()
    print(f"Built {len(tables)} aligned tables.")
    # Sanity: peek at one
    df = tables[("SPX", "cn_index:HS300")]
    print("\n[SPX vs HS300] head:")
    print(df[["cn_date", "us_date", "signal_age_days", "us_ret_cc", "cn_ret_cc", "cn_ret_oc"]].head(8))
    print("\n[SPX vs HS300] signal_age distribution:")
    print(df["signal_age_days"].value_counts().sort_index())
    print("\n[SPX vs HS300] NaN counts:")
    print(df[["us_ret_cc", "cn_ret_cc", "cn_ret_oc"]].isna().sum())
