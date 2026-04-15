import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import sys
import os
import math

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AXS import (
    fetch_stock as fetch_stock_asx,
    quant_score as quant_asx,
    game_theory_score as gt_asx,
    get_fundamentals as get_fund_asx,
    ASX_STOCKS,
)
from NSE_NIFTY50 import (
    fetch_stock as fetch_stock_nse,
    quant_score as quant_nse,
    game_theory_score as gt_nse,
    NIFTY50_STOCKS,
)

# ─── Try importing get_fundamentals from NSE (may not exist) ────────────────
try:
    from NSE_NIFTY50 import get_fundamentals as get_fund_nse
except ImportError:
    get_fund_nse = None

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantDesk — Stock Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Score colour helpers */
    .score-high { color: #1a7a2e; font-weight: 700; font-size: 1.3rem; }
    .score-low  { color: #b91c1c; font-weight: 700; font-size: 1.3rem; }
    .score-mid  { color: #b45309; font-weight: 700; font-size: 1.3rem; }

    /* Tighten metric labels */
    [data-testid="stMetricLabel"] p { font-size: 0.75rem; }

    /* Tab font */
    button[data-baseweb="tab"] { font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Helper: format large numbers ───────────────────────────────────────────
def _fmt_large(v):
    if v is None:
        return "N/A"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(v) >= 1e12:
        return f"{v/1e12:.2f} T"
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f} B"
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f} M"
    return f"{v:,.2f}"

def _fmt_pct(v):
    if v is None:
        return "N/A"
    try:
        return f"{float(v)*100:.2f}%"
    except (TypeError, ValueError):
        return str(v)

def _fmt_num(v, decimals=2):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)

def _safe_float(v, default=0.0):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default

# ─── Compute w52h / drawdown from history (missing from quant_score return) ─
def _compute_52w(hist: pd.DataFrame, current_price: float):
    close = hist["Close"].dropna()
    w52h = float(close.rolling(min(252, len(close))).max().iloc[-1])
    w52l = float(close.rolling(min(252, len(close))).min().iloc[-1])
    dd   = (current_price - w52h) / (w52h + 1e-9) * 100  # negative value
    return w52h, w52l, dd

# ─── Title ───────────────────────────────────────────────────────────────────
st.markdown("# 📊 QuantDesk — Stock Analysis Dashboard")
st.markdown("**Real-time technical & game theory analysis for ASX & NSE stocks**")
st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    market = st.radio("Select Market", ["ASX", "NSE"], horizontal=True)

    if market == "ASX":
        stocks_dict   = ASX_STOCKS
        currency_sym  = "A$"
        placeholder   = "e.g., CBA, ANZ, BHP"
    else:
        stocks_dict   = NIFTY50_STOCKS
        currency_sym  = "₹"
        placeholder   = "e.g., RELIANCE, TCS, INFY"

    ticker = st.text_input("Enter stock ticker:", placeholder=placeholder).upper().strip()

    period = st.selectbox(
        "Analysis period:",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2,
    )

    st.divider()
    st.info(f"📈 {market} Market | {len(stocks_dict)} stocks available")

    with st.expander("ℹ️ How to use"):
        st.markdown(
            """
- Type a **ticker symbol** above (e.g. `CBA` for ASX, `RELIANCE` for NSE).
- Select the **analysis period** for historical data.
- Scores are **0–100**: 60+ = bullish, 40– = bearish.
- **Not financial advice** — do your own research.
"""
        )

# ─── Main content ────────────────────────────────────────────────────────────
if not ticker:
    st.info("👈 Enter a stock ticker in the sidebar to begin analysis.")
    st.stop()

# ── Fetch & analyse ──────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker}** …"):
    try:
        if market == "ASX":
            stock, full_ticker = fetch_stock_asx(ticker)
            quant_fn  = quant_asx
            gt_fn     = gt_asx
            fund_fn   = get_fund_asx
        else:
            stock, full_ticker = fetch_stock_nse(ticker)
            quant_fn  = quant_nse
            gt_fn     = gt_nse
            fund_fn   = get_fund_nse  # may be None

        hist = stock.history(period=period)

        if hist.empty:
            st.error(
                f"❌ No data found for **{ticker}** on {market}. "
                "Check the ticker symbol and try again."
            )
            st.stop()

    except Exception as e:
        st.error(f"❌ Could not fetch data for **{ticker}**: {e}")
        with st.expander("Debug info"):
            st.code(str(e))
        st.stop()

# ── Core values ──────────────────────────────────────────────────────────────
current_price = _safe_float(hist["Close"].iloc[-1])
prev_price    = _safe_float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
day_change    = ((current_price - prev_price) / (prev_price + 1e-9)) * 100

# ── Scores ───────────────────────────────────────────────────────────────────
if len(hist) < 30:
    st.warning("⚠️ Not enough history for a full analysis. Try a longer period.")
    quant, gt = {}, {}
else:
    quant = quant_fn(hist)
    gt    = gt_fn(hist, quant, {})

# ── 52-week levels (computed here — not in quant_score return dict) ──────────
w52h, w52l, dd = _compute_52w(hist, current_price)

# ─── Hero section ────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    company_name = stocks_dict.get(ticker, full_ticker)
    st.markdown(f"### {full_ticker}")
    st.caption(company_name)
with col_b:
    st.markdown(f"#### {currency_sym}{current_price:.2f}")
    arrow = "🟢" if day_change >= 0 else "🔴"
    st.markdown(f"{arrow} **{day_change:+.2f}%** (1 day)")
with col_c:
    st.caption(f"As of {datetime.now().strftime('%d %b %Y %H:%M')}")
    st.caption(f"Period: {period}")

st.divider()

# ─── Key metrics row ─────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    qc = _safe_float(quant.get("── Quant Composite ──", 50), 50)
    css = "score-high" if qc >= 65 else ("score-low" if qc <= 40 else "score-mid")
    st.markdown("**QUANT SCORE**")
    st.markdown(f"<span class='{css}'>{qc:.1f}/100</span>", unsafe_allow_html=True)

with m2:
    gc = _safe_float(gt.get("── GT Composite ──", 50), 50)
    css = "score-high" if gc >= 65 else ("score-low" if gc <= 40 else "score-mid")
    st.markdown("**GAME THEORY**")
    st.markdown(f"<span class='{css}'>{gc:.1f}/100</span>", unsafe_allow_html=True)

with m3:
    rsi_val = _safe_float(quant.get("_rsi", 50), 50)
    delta_rsi = "Oversold" if rsi_val < 30 else ("Overbought" if rsi_val > 70 else "Neutral")
    st.metric("RSI (14)", f"{rsi_val:.1f}", delta_rsi)

with m4:
    rv30_val = _safe_float(quant.get("_rv30", 30), 30)
    delta_vol = "High" if rv30_val > 50 else ("Low" if rv30_val < 20 else "Normal")
    st.metric("30d Volatility", f"{rv30_val:.1f}%", delta_vol)

st.divider()

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🎯 Quant", "🎮 Game Theory", "🏢 Fundamentals", "📈 Technicals"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Overview — Quick Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Price History (last 100 bars)**")
        chart_df = pd.DataFrame({"Close": hist["Close"].tail(100)})
        st.line_chart(chart_df)

    with col2:
        st.markdown("**Volume (last 100 bars)**")
        vol_df = pd.DataFrame({"Volume": hist["Volume"].tail(100)})
        st.bar_chart(vol_df)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📈 Technical Signals**")
        signals = [
            ("RSI Score",        quant.get("RSI Score", 50)),
            ("Momentum Score",   quant.get("Momentum Score", 50)),
            ("Trend Strength",   quant.get("Trend Strength Score", 50)),
            ("Mean-Reversion",   quant.get("Mean-Reversion Score", 50)),
            ("Volume Surge",     quant.get("Volume Surge Score", 50)),
            ("Volatility Score", quant.get("Volatility Score", 50)),
        ]
        for name, val in signals:
            v = _safe_float(val, 50)
            delta_str = "🟢 Bullish" if v >= 60 else ("🔴 Bearish" if v <= 40 else "🟡 Neutral")
            st.metric(name, f"{v:.1f}/100", delta_str)

    with col2:
        st.markdown("**Moving Averages**")
        emas = {
            "EMA 20":  quant.get("EMA 20",  current_price),
            "EMA 50":  quant.get("EMA 50",  current_price),
            "EMA 100": quant.get("EMA 100", current_price),
            "EMA 200": quant.get("EMA 200", current_price),
        }
        for ema_name, ema_val in emas.items():
            v = _safe_float(ema_val, current_price)
            diff_pct = (current_price - v) / (v + 1e-9) * 100
            delta_str = f"{diff_pct:+.2f}% vs price"
            st.metric(ema_name, f"{currency_sym}{v:.2f}", delta_str)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUANT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🎯 Quantitative Analysis")
    st.info(
        "Measures technical momentum, volatility, and mean-reversion patterns. "
        "Higher scores (60+) indicate bullish momentum."
    )

    score_names  = ["Momentum", "Mean-Reversion", "Volatility", "Volume Surge", "RSI", "Trend"]
    score_keys   = [
        "Momentum Score", "Mean-Reversion Score", "Volatility Score",
        "Volume Surge Score", "RSI Score", "Trend Strength Score",
    ]
    score_values = [_safe_float(quant.get(k, 0)) for k in score_keys]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Score Breakdown**")
        scores_df = pd.DataFrame({"Score": score_values}, index=score_names)
        st.bar_chart(scores_df)

    with col2:
        st.markdown("**Individual Scores**")
        for name, val in zip(score_names, score_values):
            delta_str = "🟢 Bullish" if val >= 60 else ("🔴 Bearish" if val <= 40 else "🟡 Neutral")
            st.metric(name, f"{val:.1f}/100", delta_str)

    st.markdown("---")
    st.markdown("**Raw Helpers (used in Game Theory layer)**")
    c1, c2, c3 = st.columns(3)
    c1.metric("RSI",       f"{_safe_float(quant.get('_rsi', 50)):.1f}")
    c2.metric("30d Vol %", f"{_safe_float(quant.get('_rv30', 0)):.1f}%")
    c3.metric("Vol Ratio", f"{_safe_float(quant.get('_vol_ratio', 1)):.2f}x")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GAME THEORY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎮 Game Theory Analysis")
    st.info(
        "Models a 3-player game: **Institutional** vs **Retail** vs **Market Makers**. "
        "Uses Nash Equilibrium and Prisoner's Dilemma frameworks."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Nash Equilibrium — Player Signals**")
        st.metric("Institutional Signal",  gt.get("NE — Inst. Signal",     "—"))
        st.metric("Retail Signal",         gt.get("NE — Retail Signal",    "—"))
        st.metric("Market Maker Signal",   gt.get("NE — Market Maker",     "—"))
        st.metric("Dominant Strategy",     gt.get("NE — Dominant Strategy","—"))
        ne_score = _safe_float(gt.get("NE Score", 50), 50)
        st.metric("NE Score", f"{ne_score:.1f}/100")

    with col2:
        st.markdown("**Prisoner's Dilemma**")
        st.metric("Rational Choice", gt.get("PD — Rational Choice", "—"))
        pd_score = _safe_float(gt.get("PD Score", 50), 50)
        st.metric("PD Score",        f"{pd_score:.1f}/100")

        st.markdown("**Payoff Matrix**")
        c1, c2 = st.columns(2)
        c1.metric("Reward (R)",        f"{_safe_float(gt.get('PD — Payoff R (Reward)',  1)):.2f}")
        c1.metric("Sucker Loss (S)",   f"{_safe_float(gt.get('PD — Payoff S (Sucker)',  1)):.2f}")
        c2.metric("Temptation (T)",    f"{_safe_float(gt.get('PD — Payoff T (Tempt.)',  1)):.2f}")
        c2.metric("Punishment (P)",    f"{_safe_float(gt.get('PD — Payoff P (Punish)',  1)):.2f}")

    st.markdown("---")
    gt_composite = _safe_float(gt.get("── GT Composite ──", 50), 50)
    bar = int(gt_composite)
    css = "score-high" if gt_composite >= 65 else ("score-low" if gt_composite <= 40 else "score-mid")
    st.markdown(
        f"**GT Composite:** <span class='{css}'>{gt_composite:.1f}/100</span>",
        unsafe_allow_html=True,
    )
    st.progress(bar)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FUNDAMENTALS  (BUG FIX: was only showing computed price data)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🏢 Fundamentals — Company Health")

    # ── Price-derived levels (always available) ──────────────────────────────
    st.markdown("**52-Week Levels & Key Price Data**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price",      f"{currency_sym}{current_price:.2f}")
    c2.metric("52-Week High",       f"{currency_sym}{w52h:.2f}")
    c3.metric("52-Week Low",        f"{currency_sym}{w52l:.2f}")
    c4.metric("Drawdown from 52W High", f"{dd:.1f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Day High",  f"{currency_sym}{_safe_float(hist['High'].iloc[-1]):.2f}")
    c2.metric("Day Low",   f"{currency_sym}{_safe_float(hist['Low'].iloc[-1]):.2f}")
    vwap_val = _safe_float(quant.get("VWAP", current_price), current_price)
    c3.metric("VWAP",      f"{currency_sym}{vwap_val:.2f}")
    c4.metric("Day Change",f"{day_change:+.2f}%")

    st.markdown("---")

    # ── Live fundamentals from yfinance ─────────────────────────────────────
    st.markdown("**Company Fundamentals** *(from yfinance)*")
    with st.spinner("Loading fundamentals…"):
        try:
            if fund_fn is not None:
                fund = fund_fn(stock)
            else:
                # Fallback: read stock.info directly
                info = stock.info
                fund = info if info else None

            if fund:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Valuation**")
                    st.metric("Market Cap",    _fmt_large(fund.get("Market Cap") or fund.get("marketCap")))
                    st.metric("P/E (TTM)",     _fmt_num(fund.get("P/E (TTM)") or fund.get("trailingPE")))
                    st.metric("Forward P/E",   _fmt_num(fund.get("Forward P/E") or fund.get("forwardPE")))
                    st.metric("P/B Ratio",     _fmt_num(fund.get("P/B Ratio") or fund.get("priceToBook")))
                    st.metric("EV/EBITDA",     _fmt_num(fund.get("EV/EBITDA") or fund.get("enterpriseToEbitda")))

                with col2:
                    st.markdown("**Profitability**")
                    st.metric("Revenue (TTM)",    _fmt_large(fund.get("Revenue (TTM)") or fund.get("totalRevenue")))
                    st.metric("Gross Margin",     _fmt_pct(fund.get("Gross Margin") or fund.get("grossMargins")))
                    st.metric("Net Margin",       _fmt_pct(fund.get("Net Margin") or fund.get("profitMargins")))
                    st.metric("ROE",              _fmt_pct(fund.get("ROE") or fund.get("returnOnEquity")))
                    st.metric("ROA",              _fmt_pct(fund.get("ROA") or fund.get("returnOnAssets")))

                with col3:
                    st.markdown("**Dividends & Balance Sheet**")
                    st.metric("Dividend Yield",   _fmt_pct(fund.get("Dividend Yield") or fund.get("dividendYield")))
                    st.metric("Payout Ratio",     _fmt_pct(fund.get("Payout Ratio") or fund.get("payoutRatio")))
                    st.metric("Debt/Equity",      _fmt_num(fund.get("Debt/Equity") or fund.get("debtToEquity")))
                    st.metric("Current Ratio",    _fmt_num(fund.get("Current Ratio") or fund.get("currentRatio")))
                    rec = (fund.get("Recommendation") or fund.get("recommendationKey") or "N/A")
                    st.metric("Analyst Rec.",     str(rec).upper())
            else:
                st.warning("Fundamentals not available for this ticker (yfinance returned no info).")
        except Exception as e:
            st.warning(f"Could not load fundamentals: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TECHNICALS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📈 Technicals — Indicators & Chart Patterns")
    st.info("RSI, Bollinger Bands, MACD signals derived from price history.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**RSI (14-day)**")
        rsi_v = _safe_float(quant.get("_rsi", 50), 50)
        if rsi_v < 30:
            st.error(f"🔴 **Oversold** — RSI {rsi_v:.1f}  (potential bounce)")
        elif rsi_v > 70:
            st.warning(f"🟠 **Overbought** — RSI {rsi_v:.1f}  (potential pullback)")
        else:
            st.success(f"🟢 **Neutral** — RSI {rsi_v:.1f}")
        st.progress(int(min(max(rsi_v, 0), 100)))

        st.markdown("**Bollinger Band Estimate**")
        close_series = hist["Close"].dropna()
        bb_mid   = float(close_series.rolling(20).mean().iloc[-1])
        bb_std   = float(close_series.rolling(20).std().iloc[-1])
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        c1, c2, c3 = st.columns(3)
        c1.metric("Upper Band", f"{currency_sym}{bb_upper:.2f}")
        c2.metric("Mid (SMA20)", f"{currency_sym}{bb_mid:.2f}")
        c3.metric("Lower Band", f"{currency_sym}{bb_lower:.2f}")
        if current_price > bb_upper:
            st.warning("Price is **above** upper Bollinger Band — potentially overbought.")
        elif current_price < bb_lower:
            st.info("Price is **below** lower Bollinger Band — potentially oversold.")
        else:
            st.success("Price is **within** Bollinger Bands — normal range.")

    with col2:
        st.markdown("**Volatility**")
        rv_v = _safe_float(quant.get("_rv30", 30), 30)
        if rv_v > 50:
            st.error(f"🔴 **High Volatility** — {rv_v:.1f}% annualised")
        elif rv_v < 20:
            st.success(f"🟢 **Low Volatility** — {rv_v:.1f}% annualised")
        else:
            st.info(f"🟡 **Normal Volatility** — {rv_v:.1f}% annualised")
        st.progress(int(min(rv_v, 100)))

        st.markdown("**EMA Trend Alignment**")
        ema_vals = {
            "EMA 20":  _safe_float(quant.get("EMA 20",  current_price), current_price),
            "EMA 50":  _safe_float(quant.get("EMA 50",  current_price), current_price),
            "EMA 100": _safe_float(quant.get("EMA 100", current_price), current_price),
            "EMA 200": _safe_float(quant.get("EMA 200", current_price), current_price),
        }
        all_above = all(current_price > v for v in ema_vals.values())
        all_below = all(current_price < v for v in ema_vals.values())
        if all_above:
            st.success("✅ Price is **above all EMAs** — strong uptrend.")
        elif all_below:
            st.error("❌ Price is **below all EMAs** — strong downtrend.")
        else:
            st.warning("⚠️ Price is **mixed vs EMAs** — consolidation / transition.")

        for ema_name, ema_val in ema_vals.items():
            above = "↑ above" if current_price > ema_val else "↓ below"
            st.metric(
                ema_name,
                f"{currency_sym}{ema_val:.2f}",
                f"Price {above} by {abs(current_price - ema_val):.2f}",
            )

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** QuantDesk is for informational purposes only and does not "
    "constitute financial advice. All scores are model-derived estimates. "
    "Past performance does not predict future results."
)
