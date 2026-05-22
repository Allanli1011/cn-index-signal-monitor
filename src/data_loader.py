"""Pull and cache market data via AKShare.

All data cached as parquet under data/raw/. Each call returns a tidy DataFrame
with at least: date (datetime64), open, high, low, close, volume.
"""
from __future__ import annotations
from pathlib import Path
import sys
import time
import pandas as pd
import akshare as ak

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    RAW_DIR, START_DATE, END_DATE,
    US_ASSETS, CN_INDEX_ASSETS, CN_FUTURES,
)

RAW_DIR.mkdir(parents=True, exist_ok=True)


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to date-indexed DataFrame with lowercase OHLCV columns."""
    df = df.copy()
    # AKShare functions sometimes return Chinese column names; map common variants
    rename_map = {
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount",
        "持仓量": "open_interest", "结算价": "settle",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    keep = ["date", "open", "high", "low", "close", "volume"]
    extras = [c for c in ["amount", "open_interest", "settle"] if c in df.columns]
    df = df[keep + extras].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"] + extras:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_us(spec: dict) -> pd.DataFrame:
    if spec["loader"] == "index_us_stock_sina":
        df = ak.index_us_stock_sina(symbol=spec["symbol"])
    elif spec["loader"] == "stock_us_daily":
        df = ak.stock_us_daily(symbol=spec["symbol"])
    else:
        raise ValueError(f"Unknown US loader: {spec['loader']}")
    return _standardize(df)


def _fetch_cn_index(spec: dict) -> pd.DataFrame:
    df = ak.stock_zh_index_daily(symbol=spec["symbol"])
    return _standardize(df)


def _fetch_cn_future(spec: dict) -> pd.DataFrame:
    df = ak.futures_zh_daily_sina(symbol=spec["symbol"])
    # futures col: date open high low close volume hold settle
    df = df.rename(columns={"hold": "open_interest"})
    return _standardize(df)


def fetch_one(asset_class: str, code: str) -> pd.DataFrame:
    """asset_class in {'us', 'cn_index', 'cn_future'}; code is the dict key."""
    if asset_class == "us":
        return _fetch_us(US_ASSETS[code])
    if asset_class == "cn_index":
        return _fetch_cn_index(CN_INDEX_ASSETS[code])
    if asset_class == "cn_future":
        return _fetch_cn_future(CN_FUTURES[code])
    raise ValueError(asset_class)


def cache_path(asset_class: str, code: str) -> Path:
    return RAW_DIR / f"{asset_class}_{code}.parquet"


def load(asset_class: str, code: str, refresh: bool = False,
         trim: bool = True) -> pd.DataFrame:
    """Load from local parquet cache, or fetch + cache. Trims to configured window by default."""
    path = cache_path(asset_class, code)
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        df = fetch_one(asset_class, code)
        df.to_parquet(path, index=False)
    if trim:
        df = df[(df["date"] >= pd.Timestamp(START_DATE))
                & (df["date"] <= pd.Timestamp(END_DATE))].reset_index(drop=True)
    return df


def pull_all(refresh: bool = False, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Pull every configured asset; return dict keyed by '<class>_<code>'."""
    out: dict[str, pd.DataFrame] = {}
    plan = (
        [("us", code) for code in US_ASSETS]
        + [("cn_index", code) for code in CN_INDEX_ASSETS]
        + [("cn_future", code) for code in CN_FUTURES]
    )
    for cls, code in plan:
        key = f"{cls}_{code}"
        if verbose:
            print(f"[{key}] fetching...", flush=True)
        t0 = time.time()
        df = load(cls, code, refresh=refresh)
        # Trim to the configured window for downstream convenience
        df = df[(df["date"] >= pd.Timestamp(START_DATE)) & (df["date"] <= pd.Timestamp(END_DATE))]
        out[key] = df
        if verbose:
            first = df["date"].min().strftime("%Y-%m-%d") if len(df) else "—"
            last  = df["date"].max().strftime("%Y-%m-%d") if len(df) else "—"
            print(f"  rows={len(df):>5} range=[{first} .. {last}] elapsed={time.time()-t0:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    pull_all(refresh="--refresh" in sys.argv)
