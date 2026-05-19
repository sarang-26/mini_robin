"""
Watchlist and dashboard configuration.
"""

WATCHLIST = [
    {"symbol": "AAPL",  "name": "Apple Inc.",            "sector": "Technology"},
    {"symbol": "MSFT",  "name": "Microsoft Corp.",        "sector": "Technology"},
    {"symbol": "NVDA",  "name": "NVIDIA Corp.",           "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.",          "sector": "Technology"},
    {"symbol": "META",  "name": "Meta Platforms",         "sector": "Technology"},
    {"symbol": "AMZN",  "name": "Amazon.com Inc.",        "sector": "Consumer"},
    {"symbol": "TSLA",  "name": "Tesla Inc.",             "sector": "Automotive"},
    {"symbol": "JPM",   "name": "JPMorgan Chase",         "sector": "Finance"},
    {"symbol": "GS",    "name": "Goldman Sachs",          "sector": "Finance"},
    {"symbol": "XOM",   "name": "Exxon Mobil",            "sector": "Energy"},
    {"symbol": "SPY",   "name": "S&P 500 ETF",            "sector": "Index"},
    {"symbol": "QQQ",   "name": "Nasdaq-100 ETF",         "sector": "Index"},
]

SYMBOLS = [s["symbol"] for s in WATCHLIST]

SIGNAL_THRESHOLDS = {
    "strong_buy":  70,
    "buy":         55,
    "hold_upper":  55,
    "hold_lower":  45,
    "sell":        45,
    "strong_sell": 30,
}

INDICATOR_WEIGHTS = {
    "vwap":    20,
    "ema":     20,
    "macd":    20,
    "rsi":     15,
    "bb":      10,
    "volume":  15,
}

# Market hours in US/Eastern
MARKET_OPEN_HOUR  = 9
MARKET_OPEN_MIN   = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MIN  = 0

REFRESH_INTERVAL_SECONDS = 60

# Intraday interval: 1m, 2m, 5m, 15m, 30m, 60m
INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD   = "1d"
DAILY_PERIOD      = "6mo"
