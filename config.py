"""Central configuration: tickers, date range, paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

START_DATE = "2015-01-01"
END_DATE = "2026-05-22"

# AKShare symbol -> friendly name & loader spec
# loader: function name to call; symbol: arg passed to it
US_ASSETS = {
    "SPX": {"loader": "index_us_stock_sina", "symbol": ".INX",  "name": "S&P 500"},
    "NDX": {"loader": "index_us_stock_sina", "symbol": ".IXIC", "name": "Nasdaq Composite"},
    # No direct Russell 2000 index in AKShare; IWM ETF tracks it (TER ~0.19%)
    "RUT": {"loader": "stock_us_daily",      "symbol": "IWM",   "name": "Russell 2000 (IWM ETF proxy)"},
}

CN_INDEX_ASSETS = {
    "HS300":   {"loader": "stock_zh_index_daily", "symbol": "sh000300", "name": "沪深300"},
    "ZZ500":   {"loader": "stock_zh_index_daily", "symbol": "sh000905", "name": "中证500"},
    "ZZ1000":  {"loader": "stock_zh_index_daily", "symbol": "sh000852", "name": "中证1000"},
    "SSE50":   {"loader": "stock_zh_index_daily", "symbol": "sh000016", "name": "上证50"},
}

CN_FUTURES = {
    "IF": {"symbol": "IF0", "name": "沪深300期货", "underlying": "HS300",
           "multiplier": 300, "margin_pct": 0.12},
    "IC": {"symbol": "IC0", "name": "中证500期货", "underlying": "ZZ500",
           "multiplier": 200, "margin_pct": 0.14},
    "IM": {"symbol": "IM0", "name": "中证1000期货", "underlying": "ZZ1000",
           "multiplier": 200, "margin_pct": 0.15},
    "IH": {"symbol": "IH0", "name": "上证50期货",  "underlying": "SSE50",
           "multiplier": 300, "margin_pct": 0.12},
}

# Mapping between underlying index and which CN trading day to assign US signal
# US day t close (~04:00 Beijing day t+1) → CN day t+1 trading session
# So we shift US returns by +1 trading day to align with next CN day
