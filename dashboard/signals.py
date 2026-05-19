"""
Bloomberg-style composite signal engine.

Scoring model (100 pts total):
  VWAP  20 pts  — price positioning vs intraday VWAP
  EMA   20 pts  — short/medium EMA crossover trend
  MACD  20 pts  — MACD vs signal + histogram direction
  RSI   15 pts  — momentum + extremes
  BB    10 pts  — band position + squeeze/expansion
  Volume 15 pts — relative volume confirmation

Thresholds → STRONG BUY ≥70 | BUY 55-69 | HOLD 45-54 | SELL 30-44 | STRONG SELL <30
"""

import numpy as np
import pandas as pd


def score_signal(df: pd.DataFrame, quote: dict) -> dict:
    """
    Compute composite signal score and per-indicator breakdown.
    Returns a dict with keys: score, label, color, breakdown, details.
    """
    if df.empty or len(df) < 26:
        return _empty_signal()

    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row

    breakdown = {}

    # ── VWAP (20 pts) ─────────────────────────────────────────────────────────
    vwap_score = 0
    vwap_note  = "N/A"
    if not _nan(row.get("VWAP")):
        pct_from_vwap = (row["Close"] - row["VWAP"]) / row["VWAP"] * 100
        if pct_from_vwap > 1.0:
            vwap_score = 20
            vwap_note  = f"Price {pct_from_vwap:.1f}% above VWAP → bullish"
        elif pct_from_vwap > 0:
            vwap_score = 15
            vwap_note  = f"Price {pct_from_vwap:.2f}% above VWAP → mildly bullish"
        elif pct_from_vwap > -1.0:
            vwap_score = 8
            vwap_note  = f"Price {abs(pct_from_vwap):.2f}% below VWAP → mildly bearish"
        else:
            vwap_score = 2
            vwap_note  = f"Price {abs(pct_from_vwap):.1f}% below VWAP → bearish"
    breakdown["VWAP"] = {"score": vwap_score, "max": 20, "note": vwap_note}

    # ── EMA Cross (20 pts) ────────────────────────────────────────────────────
    ema_score = 0
    ema_note  = "N/A"
    if not (_nan(row.get("EMA_9")) or _nan(row.get("EMA_21")) or _nan(row.get("EMA_50"))):
        if row["EMA_9"] > row["EMA_21"] > row["EMA_50"]:
            ema_score = 20
            ema_note  = "EMA 9 > EMA 21 > EMA 50 — strong uptrend"
        elif row["EMA_9"] > row["EMA_21"]:
            ema_score = 14
            ema_note  = "EMA 9 > EMA 21 — short-term uptrend"
        elif row["EMA_9"] < row["EMA_21"] < row["EMA_50"]:
            ema_score = 2
            ema_note  = "EMA 9 < EMA 21 < EMA 50 — strong downtrend"
        elif row["EMA_9"] < row["EMA_21"]:
            ema_score = 7
            ema_note  = "EMA 9 < EMA 21 — short-term downtrend"
        else:
            ema_score = 10
            ema_note  = "EMAs converging — neutral"
    breakdown["EMA Cross"] = {"score": ema_score, "max": 20, "note": ema_note}

    # ── MACD (20 pts) ─────────────────────────────────────────────────────────
    macd_score = 0
    macd_note  = "N/A"
    if not (_nan(row.get("MACD")) or _nan(row.get("MACD_signal")) or _nan(row.get("MACD_hist"))):
        macd_above  = row["MACD"] > row["MACD_signal"]
        hist_rising = row["MACD_hist"] > prev.get("MACD_hist", row["MACD_hist"])
        hist_pos    = row["MACD_hist"] > 0

        if macd_above and hist_pos and hist_rising:
            macd_score = 20
            macd_note  = "MACD above signal, histogram positive & growing — bullish momentum"
        elif macd_above and hist_rising:
            macd_score = 15
            macd_note  = "MACD above signal, histogram growing — building momentum"
        elif macd_above:
            macd_score = 12
            macd_note  = "MACD above signal — mild bullish bias"
        elif not macd_above and not hist_pos and not hist_rising:
            macd_score = 2
            macd_note  = "MACD below signal, histogram negative & falling — bearish"
        else:
            macd_score = 7
            macd_note  = "MACD below signal — mild bearish bias"
    breakdown["MACD"] = {"score": macd_score, "max": 20, "note": macd_note}

    # ── RSI (15 pts) ──────────────────────────────────────────────────────────
    rsi_score = 0
    rsi_note  = "N/A"
    if not _nan(row.get("RSI")):
        rsi = row["RSI"]
        if rsi < 30:
            rsi_score = 13
            rsi_note  = f"RSI {rsi:.1f} — oversold, potential reversal buy"
        elif rsi < 40:
            rsi_score = 15
            rsi_note  = f"RSI {rsi:.1f} — recovering from oversold, bullish"
        elif rsi <= 60:
            rsi_score = 11
            rsi_note  = f"RSI {rsi:.1f} — healthy momentum range"
        elif rsi <= 70:
            rsi_score = 7
            rsi_note  = f"RSI {rsi:.1f} — approaching overbought"
        else:
            rsi_score = 2
            rsi_note  = f"RSI {rsi:.1f} — overbought, watch for reversal"
    breakdown["RSI"] = {"score": rsi_score, "max": 15, "note": rsi_note}

    # ── Bollinger Bands (10 pts) ──────────────────────────────────────────────
    bb_score = 0
    bb_note  = "N/A"
    if not (_nan(row.get("BB_pct")) or _nan(row.get("BB_width"))):
        bb_pct   = row["BB_pct"]
        bb_width = row["BB_width"]
        squeeze  = bb_width < df["BB_width"].quantile(0.20) if len(df) > 20 else False

        if bb_pct < 0.1:
            bb_score = 9
            bb_note  = f"Price near lower band ({bb_pct:.0%}) — potential bounce"
        elif bb_pct < 0.35:
            bb_score = 7
            bb_note  = f"Price in lower half ({bb_pct:.0%}) of BB range"
        elif bb_pct > 0.9:
            bb_score = 2
            bb_note  = f"Price near upper band ({bb_pct:.0%}) — overbought territory"
        elif bb_pct > 0.65:
            bb_score = 5
            bb_note  = f"Price in upper half ({bb_pct:.0%}) of BB range"
        else:
            bb_score = 6
            bb_note  = f"Price mid-range ({bb_pct:.0%}) within bands"

        if squeeze:
            bb_score = min(bb_score + 2, 10)
            bb_note += " + BB squeeze (breakout pending)"
    breakdown["Bollinger"] = {"score": bb_score, "max": 10, "note": bb_note}

    # ── Volume (15 pts) ───────────────────────────────────────────────────────
    vol_score = 0
    vol_note  = "N/A"
    avg_vol   = quote.get("avg_volume", 0)
    cur_vol   = quote.get("volume", 0)
    if avg_vol and avg_vol > 0:
        vol_ratio = cur_vol / avg_vol
        up_day    = quote.get("change_pct", 0) >= 0
        if vol_ratio >= 1.5 and up_day:
            vol_score = 15
            vol_note  = f"Volume {vol_ratio:.1f}x avg on up-day — strong confirmation"
        elif vol_ratio >= 1.2 and up_day:
            vol_score = 11
            vol_note  = f"Volume {vol_ratio:.1f}x avg on up-day — moderate confirmation"
        elif vol_ratio >= 1.0:
            vol_score = 8
            vol_note  = f"Volume at {vol_ratio:.1f}x avg — normal activity"
        elif vol_ratio >= 0.7:
            vol_score = 5
            vol_note  = f"Volume {vol_ratio:.1f}x avg — below average activity"
        else:
            vol_score = 2
            vol_note  = f"Volume {vol_ratio:.1f}x avg — very low, move unconfirmed"
    breakdown["Volume"] = {"score": vol_score, "max": 15, "note": vol_note}

    # ── Composite ─────────────────────────────────────────────────────────────
    total = sum(v["score"] for v in breakdown.values())
    label, color, emoji = _classify(total)

    return {
        "score":     total,
        "label":     label,
        "color":     color,
        "emoji":     emoji,
        "breakdown": breakdown,
        "rsi":       float(row.get("RSI", float("nan"))),
        "macd":      float(row.get("MACD", float("nan"))),
        "macd_hist": float(row.get("MACD_hist", float("nan"))),
        "vwap":      float(row.get("VWAP", float("nan"))),
        "atr":       float(row.get("ATR", float("nan"))),
        "bb_pct":    float(row.get("BB_pct", float("nan"))),
        "stoch_k":   float(row.get("Stoch_K", float("nan"))),
    }


def _classify(score: int) -> tuple[str, str, str]:
    if score >= 70:
        return "STRONG BUY",  "#00e676", "🚀"
    if score >= 55:
        return "BUY",         "#69f0ae", "📈"
    if score >= 45:
        return "HOLD",        "#ffd740", "⏸"
    if score >= 30:
        return "SELL",        "#ff6d6d", "📉"
    return     "STRONG SELL", "#f44336", "🔻"


def _nan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and np.isnan(v))
    except Exception:
        return True


def _empty_signal() -> dict:
    return {
        "score": 50, "label": "HOLD", "color": "#ffd740", "emoji": "⏸",
        "breakdown": {}, "rsi": float("nan"), "macd": float("nan"),
        "macd_hist": float("nan"), "vwap": float("nan"), "atr": float("nan"),
        "bb_pct": float("nan"), "stoch_k": float("nan"),
    }
