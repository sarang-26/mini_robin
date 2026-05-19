"""
Bloomberg-grade technical indicators implemented with numpy/pandas.

Indicators:
  - VWAP          (Volume Weighted Average Price)
  - EMA           (Exponential Moving Average — 9, 21, 50)
  - SMA           (Simple Moving Average — 50, 200)
  - RSI           (Relative Strength Index — 14)
  - MACD          (12, 26, 9)
  - Bollinger Bands (20, 2σ)
  - ATR           (Average True Range — 14)
  - OBV           (On-Balance Volume)
  - Stochastic    (%K %D — 14, 3)
"""

import numpy as np
import pandas as pd


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every indicator and return an enriched DataFrame."""
    if df.empty or len(df) < 2:
        return df
    df = df.copy()
    df = _vwap(df)
    df = _ema(df, 9)
    df = _ema(df, 21)
    df = _ema(df, 50)
    df = _sma(df, 50)
    df = _sma(df, 200)
    df = _rsi(df, 14)
    df = _macd(df, 12, 26, 9)
    df = _bollinger(df, 20, 2)
    df = _atr(df, 14)
    df = _obv(df)
    df = _stochastic(df, 14, 3)
    return df


# ── VWAP ──────────────────────────────────────────────────────────────────────

def _vwap(df: pd.DataFrame) -> pd.DataFrame:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol     = df["Volume"].replace(0, np.nan)
    cum_tv  = (typical * vol).cumsum()
    cum_v   = vol.cumsum()
    df["VWAP"] = cum_tv / cum_v
    return df


# ── Moving Averages ───────────────────────────────────────────────────────────

def _ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()
    return df


def _sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()
    return df


# ── RSI ───────────────────────────────────────────────────────────────────────

def _rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# ── MACD ──────────────────────────────────────────────────────────────────────

def _macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow,   adjust=False).mean()
    df["MACD"]        = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    return df


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def _bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
    mid         = df["Close"].rolling(window=period).mean()
    std         = df["Close"].rolling(window=period).std()
    df["BB_mid"]   = mid
    df["BB_upper"] = mid + std_dev * std
    df["BB_lower"] = mid - std_dev * std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / mid  # normalized width
    df["BB_pct"]   = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    return df


# ── ATR ───────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(com=period - 1, adjust=False).mean()
    return df


# ── OBV ───────────────────────────────────────────────────────────────────────

def _obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (df["Volume"] * direction).cumsum()
    return df


# ── Stochastic ────────────────────────────────────────────────────────────────

def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    low_min  = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    denom    = (high_max - low_min).replace(0, np.nan)
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / denom
    df["Stoch_D"] = df["Stoch_K"].rolling(window=d_period).mean()
    return df


# ── Key levels ────────────────────────────────────────────────────────────────

def support_resistance(df: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    """Simple pivot-based support/resistance from the last N bars."""
    if df.empty or len(df) < lookback:
        return (float("nan"), float("nan"))
    window = df.tail(lookback)
    support    = float(window["Low"].min())
    resistance = float(window["High"].max())
    return support, resistance
