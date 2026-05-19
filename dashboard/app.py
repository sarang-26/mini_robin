"""
Stock Trading Dashboard — Bloomberg-style insights.
Run with:  streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import WATCHLIST, SYMBOLS, REFRESH_INTERVAL_SECONDS
from data_fetcher import market_status, fetch_intraday, fetch_daily, fetch_quote
from indicators import compute_all, support_resistance
from signals import score_signal

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }

  .market-banner {
    display: flex; align-items: center; justify-content: space-between;
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 10px 20px; margin-bottom: 12px;
  }
  .market-status { font-size: 1.1rem; font-weight: 700; letter-spacing: 2px; }
  .market-time   { font-size: 0.85rem; color: #8b949e; }
  .countdown     { font-size: 0.9rem; color: #58a6ff; }

  .stock-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px 14px; margin-bottom: 2px;
  }
  .signal-badge {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1px;
  }
  .score-bar-wrap { background: #21262d; border-radius: 4px; height: 8px; }
  .score-bar      { height: 8px; border-radius: 4px; }
  .indicator-row {
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid #21262d; padding: 5px 0; font-size: 0.82rem;
  }
  .ind-label { color: #8b949e; min-width: 110px; }
  .ind-val   { font-weight: 600; color: #e6edf3; min-width: 60px; text-align: right; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_vol(v: int) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}K"
    return str(v)


def render_chart(df: pd.DataFrame, symbol: str, title: str):
    if df.empty:
        st.warning("No data available for this timeframe.")
        return

    has_vwap = "VWAP" in df.columns and not df["VWAP"].isna().all()
    has_bb   = "BB_upper" in df.columns and not df["BB_upper"].isna().all()
    has_macd = "MACD" in df.columns and not df["MACD"].isna().all()
    has_rsi  = "RSI" in df.columns and not df["RSI"].isna().all()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.18, 0.16, 0.16],
        subplot_titles=[f"{symbol} — {title}", "Volume", "MACD", "RSI (14)"],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#00e676", decreasing_line_color="#f44336",
        increasing_fillcolor="#00e676",  decreasing_fillcolor="#f44336",
    ), row=1, col=1)

    # VWAP
    if has_vwap:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color="#ff9800", width=1.8, dash="dot"),
        ), row=1, col=1)

    # EMA lines
    for period, lc in [(9, "#40c4ff"), (21, "#7c4dff"), (50, "#ff4081")]:
        col_name = f"EMA_{period}"
        if col_name in df.columns and not df[col_name].isna().all():
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=f"EMA {period}",
                line=dict(color=lc, width=1.0),
            ), row=1, col=1)

    # Bollinger Bands
    if has_bb:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="#4fc3f7", width=0.8, dash="dash"), showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="#4fc3f7", width=0.8, dash="dash"),
            fill="tonexty", fillcolor="rgba(79,195,247,0.05)", showlegend=False,
        ), row=1, col=1)

    # Volume bars
    vol_colors = [
        "#00e676" if float(c) >= float(o) else "#f44336"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_colors, showlegend=False,
    ), row=2, col=1)

    # MACD
    if has_macd:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#40c4ff", width=1.2),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_signal"], name="MACD Signal",
            line=dict(color="#ff9800", width=1.0),
        ), row=3, col=1)
        hist_col = ["#00e676" if v >= 0 else "#f44336" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_hist"], name="Hist",
            marker_color=hist_col, showlegend=False,
        ), row=3, col=1)
        fig.add_hline(y=0, line_color="#30363d", line_width=1, row=3, col=1)

    # RSI
    if has_rsi:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#e040fb", width=1.5),
        ), row=4, col=1)
        for level, lc in [(70, "#f44336"), (50, "#8b949e"), (30, "#00e676")]:
            fig.add_hline(y=level, line_dash="dot", line_color=lc,
                          line_width=1, row=4, col=1)

    fig.update_layout(
        height=800,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d", borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#21262d", showgrid=True, zeroline=False, row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", showgrid=True, zeroline=False, row=i, col=1)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def get_intraday(symbol: str):
    df = fetch_intraday(symbol)
    return compute_all(df)

@st.cache_data(ttl=300, show_spinner=False)
def get_daily(symbol: str):
    df = fetch_daily(symbol)
    return compute_all(df)

@st.cache_data(ttl=60, show_spinner=False)
def get_quote(symbol: str):
    return fetch_quote(symbol)

@st.cache_data(ttl=60, show_spinner=False)
def get_signal(symbol: str):
    q  = get_quote(symbol)
    df = get_intraday(symbol)
    if df.empty or len(df) < 5:
        df = get_daily(symbol)
    return score_signal(df, q)


# ── Session state ─────────────────────────────────────────────────────────────
if "selected" not in st.session_state:
    st.session_state.selected = SYMBOLS[0]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    sym_options = {f"{s['symbol']} — {s['name']}": s["symbol"] for s in WATCHLIST}
    chosen_label = st.selectbox(
        "Focus stock",
        list(sym_options.keys()),
        index=list(sym_options.values()).index(st.session_state.selected),
    )
    st.session_state.selected = sym_options[chosen_label]

    st.markdown("---")
    auto_refresh = st.toggle("Auto-refresh (60s)", value=True)
    if st.button("↺  Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Signal Key**")
    for txt, col in [
        ("🚀 STRONG BUY  ≥ 70", "#00e676"),
        ("📈 BUY          55–69", "#69f0ae"),
        ("⏸  HOLD         45–54", "#ffd740"),
        ("📉 SELL          30–44", "#ff6d6d"),
        ("🔻 STRONG SELL  < 30",  "#f44336"),
    ]:
        st.markdown(f'<span style="color:{col};font-size:.82rem;font-weight:600;">{txt}</span>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Monitored Stocks**")
    for s in WATCHLIST:
        st.caption(f"`{s['symbol']}` {s['name']}")

    st.markdown("---")
    st.caption("Data via Yahoo Finance · Not financial advice")


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET STATUS BANNER
# ═══════════════════════════════════════════════════════════════════════════════
mkt = market_status()
st.markdown(f"""
<div class="market-banner">
  <div>
    <span class="market-status" style="color:{mkt['color']};">● {mkt['status']}</span>
    <span class="market-time"> &nbsp;NYSE / NASDAQ</span>
  </div>
  <div class="countdown">{mkt['countdown']}</div>
  <div class="market-time">{mkt['now_et'].strftime('%A %b %d, %Y  %I:%M:%S %p ET')}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("## 📊 Stock Trading Intelligence Dashboard")
st.caption("Bloomberg-grade signals · VWAP · EMA Cross · MACD · RSI · Bollinger Bands · Volume Confirmation · Stochastic")


# ═══════════════════════════════════════════════════════════════════════════════
# WATCHLIST GRID
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### Watchlist Overview")

cols_per_row = 4
for row_idx in range(math.ceil(len(WATCHLIST) / cols_per_row)):
    cols = st.columns(cols_per_row)
    for col_idx, col in enumerate(cols):
        entry_idx = row_idx * cols_per_row + col_idx
        if entry_idx >= len(WATCHLIST):
            break
        entry = WATCHLIST[entry_idx]
        sym   = entry["symbol"]
        with col:
            q   = get_quote(sym)
            sig = get_signal(sym)

            price   = q.get("price", 0)
            chg_pct = q.get("change_pct", 0)
            chg     = q.get("change", 0)
            score   = sig["score"]
            label   = sig["label"]
            color   = sig["color"]
            emoji   = sig["emoji"]

            chg_color = "#00e676" if chg_pct >= 0 else "#f44336"
            arrow     = "▲" if chg_pct >= 0 else "▼"
            border    = "#58a6ff" if sym == st.session_state.selected else "#30363d"

            st.markdown(f"""
<div class="stock-card" style="border-color:{border};">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="font-size:.95rem;font-weight:700;color:#e6edf3;">{sym}</span>
    <span class="signal-badge" style="background:{color}22;color:{color};">{emoji} {label}</span>
  </div>
  <div style="font-size:1.25rem;font-weight:700;color:#e6edf3;">${price:,.2f}</div>
  <div style="font-size:.80rem;color:{chg_color};">{arrow} ${abs(chg):.2f} ({chg_pct:+.2f}%)</div>
  <div style="margin-top:6px;">
    <div style="display:flex;justify-content:space-between;font-size:.70rem;color:#8b949e;margin-bottom:2px;">
      <span>Score</span>
      <span style="color:{color};font-weight:700;">{score}/100</span>
    </div>
    <div class="score-bar-wrap">
      <div class="score-bar" style="width:{score}%;background:{color};"></div>
    </div>
  </div>
  <div style="font-size:.70rem;color:#6e7681;margin-top:3px;">{entry['sector']}</div>
</div>
""", unsafe_allow_html=True)
            if st.button(f"Analyse {sym}", key=f"btn_{sym}", use_container_width=True):
                st.session_state.selected = sym
                st.rerun()


st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# DEEP ANALYSIS PANEL
# ═══════════════════════════════════════════════════════════════════════════════
selected = st.session_state.selected
entry    = next(s for s in WATCHLIST if s["symbol"] == selected)
st.markdown(f"### Deep Analysis — {selected}  ·  {entry['name']}")

with st.spinner(f"Loading {selected}…"):
    q        = get_quote(selected)
    df_intra = get_intraday(selected)
    df_daily = get_daily(selected)
    sig      = get_signal(selected)

price    = q.get("price", 0)
chg_pct  = q.get("change_pct", 0)
chg      = q.get("change", 0)
volume   = q.get("volume", 0)
avg_vol  = q.get("avg_volume", 0)
day_high = q.get("day_high", price)
day_low  = q.get("day_low", price)
yr_high  = q.get("year_high", price)
yr_low   = q.get("year_low", price)
atr_val  = sig.get("atr", float("nan"))

# Metrics row
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Price",       f"${price:,.2f}",   f"{chg_pct:+.2f}%")
m2.metric("Day High",    f"${day_high:,.2f}")
m3.metric("Day Low",     f"${day_low:,.2f}")
m4.metric("52W High",    f"${yr_high:,.2f}")
m5.metric("Volume",      fmt_vol(volume),    f"{volume/avg_vol:.1f}x avg" if avg_vol else "")
m6.metric("ATR (14)",    f"${atr_val:.2f}" if not math.isnan(atr_val) else "—")

st.markdown("")

# Signal hero + breakdown
sc1, sc2 = st.columns([1, 2])
with sc1:
    score = sig["score"]
    label = sig["label"]
    color = sig["color"]
    emoji = sig["emoji"]
    st.markdown(f"""
<div style="background:#161b22;border:2px solid {color};border-radius:12px;
            padding:22px;text-align:center;margin-top:4px;">
  <div style="font-size:2.8rem;">{emoji}</div>
  <div style="font-size:1.9rem;font-weight:900;color:{color};letter-spacing:2px;margin:4px 0;">{label}</div>
  <div style="font-size:.85rem;color:#8b949e;">Composite Signal Score</div>
  <div style="font-size:2.8rem;font-weight:900;color:{color};">{score}
    <span style="font-size:1rem;color:#8b949e;">/100</span>
  </div>
  <div style="background:#21262d;border-radius:6px;height:12px;margin-top:10px;">
    <div style="width:{score}%;height:12px;border-radius:6px;background:{color};"></div>
  </div>
</div>
""", unsafe_allow_html=True)

with sc2:
    st.markdown("**Signal Breakdown (Bloomberg Strategy)**")
    for ind_name, data in sig.get("breakdown", {}).items():
        s   = data["score"]
        mx  = data["max"]
        note = data["note"]
        pct  = int(s / mx * 100) if mx else 0
        bar_c = color if pct >= 60 else ("#ffd740" if pct >= 40 else "#f44336")
        st.markdown(f"""
<div style="margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">
    <span style="color:#8b949e;font-size:.82rem;min-width:100px;">{ind_name}</span>
    <div style="flex:1;background:#21262d;border-radius:3px;height:7px;">
      <div style="width:{pct}%;height:7px;border-radius:3px;background:{bar_c};"></div>
    </div>
    <span style="color:{bar_c};font-size:.80rem;font-weight:700;min-width:40px;text-align:right;">{s}/{mx}</span>
  </div>
  <div style="font-size:.70rem;color:#6e7681;padding-left:108px;">{note}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ── Charts ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([f"📉 Intraday (5m)", "📅 Daily (6 months)"])
with tab1:
    render_chart(df_intra, selected, "Intraday 5-min")
with tab2:
    render_chart(df_daily, selected, "Daily")


# ── Indicator table ───────────────────────────────────────────────────────────
st.markdown("### Technical Indicator Snapshot")

df_ref = df_intra if (not df_intra.empty) else df_daily
if not df_ref.empty:
    row = df_ref.iloc[-1]

    def _v(col_name, fmt=".2f"):
        val = row.get(col_name)
        try:
            fval = float(val)
            if math.isnan(fval):
                return "—"
            return f"{fval:{fmt}}"
        except Exception:
            return "—"

    indicators = [
        ("VWAP",         _v("VWAP"),         "Intraday anchor — price vs VWAP = directional bias"),
        ("EMA 9",        _v("EMA_9"),         "Fast EMA — short-term trend direction"),
        ("EMA 21",       _v("EMA_21"),        "Medium EMA — 9>21 = bullish crossover"),
        ("EMA 50",       _v("EMA_50"),        "Intermediate EMA — 21>50 = confirmed uptrend"),
        ("SMA 50",       _v("SMA_50"),        "50-day SMA — institutional support/resistance"),
        ("SMA 200",      _v("SMA_200"),       "200-day SMA — long-term trend anchor"),
        ("RSI (14)",     _v("RSI"),           "< 30 oversold · 30–70 healthy · > 70 overbought"),
        ("MACD",         _v("MACD"),          "MACD line — trend momentum"),
        ("MACD Signal",  _v("MACD_signal"),   "Signal line — crossover triggers entries"),
        ("MACD Hist",    _v("MACD_hist"),     "Histogram — growing = accelerating momentum"),
        ("BB Upper",     _v("BB_upper"),      "Upper Bollinger Band (20, 2σ)"),
        ("BB Mid",       _v("BB_mid"),        "Middle band (20-period SMA)"),
        ("BB Lower",     _v("BB_lower"),      "Lower Bollinger Band — bounce zone"),
        ("BB %B",        _v("BB_pct"),        "0 = at lower band · 1 = at upper band"),
        ("BB Width",     _v("BB_width"),      "Narrow = squeeze → breakout pending"),
        ("ATR (14)",     _v("ATR"),           "Volatility — use for stop-loss sizing"),
        ("Stoch %K",     _v("Stoch_K"),       "< 20 oversold · > 80 overbought"),
        ("Stoch %D",     _v("Stoch_D"),       "Smoothed %K — signal line"),
        ("OBV",          _v("OBV", ".0f"),    "On-Balance Volume — rising = accumulation"),
    ]

    col_a, col_b = st.columns(2)
    half = math.ceil(len(indicators) / 2)
    for col_w, chunk in [(col_a, indicators[:half]), (col_b, indicators[half:])]:
        with col_w:
            for ind_nm, val, desc in chunk:
                try:
                    fv = float(val)
                    if "RSI" in ind_nm and not math.isnan(fv):
                        vc = "#69f0ae" if fv < 30 else ("#f44336" if fv > 70 else "#e6edf3")
                    elif "Stoch" in ind_nm and not math.isnan(fv):
                        vc = "#69f0ae" if fv < 20 else ("#f44336" if fv > 80 else "#e6edf3")
                    else:
                        vc = "#e6edf3"
                except Exception:
                    vc = "#e6edf3"

                st.markdown(f"""
<div class="indicator-row">
  <span class="ind-label">{ind_nm}</span>
  <span class="ind-val" style="color:{vc};">{val}</span>
  <span style="color:#6e7681;font-size:.68rem;flex:1;text-align:right;padding-left:8px;">{desc}</span>
</div>""", unsafe_allow_html=True)


# ── Support / Resistance ──────────────────────────────────────────────────────
st.markdown("### Key Price Levels")
support, resistance = support_resistance(df_ref)
l1, l2, l3, l4 = st.columns(4)
l1.metric("Support (20-bar)",    f"${support:.2f}"    if not math.isnan(support)    else "—")
l2.metric("Resistance (20-bar)", f"${resistance:.2f}" if not math.isnan(resistance) else "—")
if not math.isnan(atr_val) and price:
    l3.metric("Stop Loss (1× ATR)", f"${price - atr_val:.2f}")
    l4.metric("Profit Target (2× ATR)", f"${price + 2 * atr_val:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET BREADTH TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Market Breadth — All Signals at a Glance")

rows_data = []
for s_info in WATCHLIST:
    sym2  = s_info["symbol"]
    q2    = get_quote(sym2)
    sig2  = get_signal(sym2)
    rsi2  = sig2.get("rsi", float("nan"))
    rows_data.append({
        "Symbol":  sym2,
        "Sector":  s_info["sector"],
        "Price":   f"${q2.get('price', 0):,.2f}",
        "Change":  f"{q2.get('change_pct', 0):+.2f}%",
        "Score":   sig2["score"],
        "Signal":  sig2["label"],
        "RSI":     f"{rsi2:.1f}" if not math.isnan(rsi2) else "—",
        "Volume":  f"{q2.get('volume',0)/q2.get('avg_volume',1):.1f}x" if q2.get("avg_volume") else "—",
    })

bdf = pd.DataFrame(rows_data)

def _sig_style(val):
    cmap = {
        "STRONG BUY":  "color:#00e676;font-weight:700",
        "BUY":         "color:#69f0ae;font-weight:700",
        "HOLD":        "color:#ffd740",
        "SELL":        "color:#ff6d6d",
        "STRONG SELL": "color:#f44336;font-weight:700",
    }
    return cmap.get(val, "")

def _chg_style(val):
    try:
        return "color:#00e676" if float(val.replace("%","").replace("+","")) >= 0 else "color:#f44336"
    except Exception:
        return ""

styled = (
    bdf.style
    .map(_sig_style, subset=["Signal"])
    .map(_chg_style, subset=["Change"])
    .background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100)
    .set_properties(**{
        "background-color": "#161b22",
        "color": "#e6edf3",
        "border": "1px solid #30363d",
        "font-size": "0.82rem",
    })
)
st.dataframe(styled, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
if auto_refresh:
    if mkt["is_open"]:
        st.caption(f"Auto-refreshing every {REFRESH_INTERVAL_SECONDS}s during market hours.")
        time.sleep(REFRESH_INTERVAL_SECONDS)
        st.cache_data.clear()
        st.rerun()
    else:
        st.info(f"⏰ Market is **{mkt['status']}** — auto-refresh paused.  {mkt['countdown']}")
