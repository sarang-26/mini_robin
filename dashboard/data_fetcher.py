"""
Stock data fetching via yfinance.
Handles intraday + daily data, market hours detection.
"""

import datetime
import pytz
import yfinance as yf
import pandas as pd
from config import (
    INTRADAY_INTERVAL, INTRADAY_PERIOD, DAILY_PERIOD,
    MARKET_OPEN_HOUR, MARKET_OPEN_MIN, MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN,
)

ET = pytz.timezone("America/New_York")


def market_status() -> dict:
    now_et = datetime.datetime.now(ET)
    weekday = now_et.weekday()  # 0=Mon … 4=Fri
    is_weekend = weekday >= 5

    open_time  = now_et.replace(hour=MARKET_OPEN_HOUR,  minute=MARKET_OPEN_MIN,  second=0, microsecond=0)
    close_time = now_et.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0, microsecond=0)
    pre_open   = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
    after_close = now_et.replace(hour=20, minute=0, second=0, microsecond=0)

    if is_weekend:
        status = "WEEKEND"
        color  = "#6c757d"
    elif now_et < pre_open or now_et >= after_close:
        status = "CLOSED"
        color  = "#dc3545"
    elif pre_open <= now_et < open_time:
        status = "PRE-MARKET"
        color  = "#fd7e14"
    elif open_time <= now_et < close_time:
        status = "OPEN"
        color  = "#28a745"
    else:
        status = "AFTER-HOURS"
        color  = "#6f42c1"

    is_open = status == "OPEN"

    if is_open:
        delta = close_time - now_et
        countdown_label = f"Closes in {_fmt_delta(delta)}"
    elif status == "PRE-MARKET":
        delta = open_time - now_et
        countdown_label = f"Opens in {_fmt_delta(delta)}"
    elif status in ("CLOSED", "WEEKEND"):
        # next Monday (or tomorrow if weekday)
        days_ahead = (7 - weekday) % 7 if weekday >= 5 else 1
        next_open = (now_et + datetime.timedelta(days=days_ahead)).replace(
            hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0, microsecond=0
        )
        delta = next_open - now_et
        countdown_label = f"Opens in {_fmt_delta(delta)}"
    else:
        countdown_label = ""

    return {
        "status": status,
        "color":  color,
        "is_open": is_open,
        "now_et": now_et,
        "countdown": countdown_label,
    }


def _fmt_delta(td: datetime.timedelta) -> str:
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def fetch_intraday(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL,
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = _flatten_cols(df)
        df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(ET)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_daily(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=DAILY_PERIOD, interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df = _flatten_cols(df)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_quote(symbol: str) -> dict:
    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.fast_info
        price  = float(info.last_price or 0)
        prev   = float(info.previous_close or price)
        change      = price - prev
        change_pct  = (change / prev * 100) if prev else 0
        volume      = int(info.last_volume or 0)
        avg_volume  = int(getattr(info, "three_month_average_volume", 0) or 0)
        return {
            "symbol":      symbol,
            "price":       price,
            "prev_close":  prev,
            "change":      change,
            "change_pct":  change_pct,
            "volume":      volume,
            "avg_volume":  avg_volume,
            "day_high":    float(getattr(info, "day_high", price) or price),
            "day_low":     float(getattr(info, "day_low",  price) or price),
            "year_high":   float(getattr(info, "year_high", price) or price),
            "year_low":    float(getattr(info, "year_low",  price) or price),
        }
    except Exception:
        return {"symbol": symbol, "price": 0, "change_pct": 0, "volume": 0}


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df
