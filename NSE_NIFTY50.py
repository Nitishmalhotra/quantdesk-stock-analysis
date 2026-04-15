"""
NSE Nifty 50 Stock Analyzer
============================
Search NSE Nifty 50 stocks and view fundamental + technical analysis.

Requirements:
    pip install yfinance pandas numpy requests tabulate colorama ta

Usage:
    python nse_nifty50_analyzer.py
    python nse_nifty50_analyzer.py --ticker RELIANCE    # Direct lookup
    python nse_nifty50_analyzer.py --search bank        # Search by keyword
"""

import argparse
import sys
from datetime import datetime, timedelta
import webbrowser
import os

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    from colorama import Fore, Style, init
    import ta
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "yfinance", "pandas", "numpy", "tabulate", "colorama", "ta", "-q"
    ])
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    from colorama import Fore, Style, init
    import ta

init(autoreset=True)
# ══════════════════════════════════════════════════════════════════════════════
# DEEP SCANNER ENGINE  —  Quant Analysis + Game Theory
# ══════════════════════════════════════════════════════════════════════════════
#
#  Step 1 – SCREEN  : Find stocks down ≥40% from 52W-high with EMA-10 uptrend
#  Step 2 – SCORE   :
#     A. Quantitative Analysis  — momentum, volatility, mean-reversion signals
#     B. Game Theory            — Nash Equilibrium, institutional vs retail
#                                 signal, prisoner's dilemma payoff matrix
#
# This block is designed to be embedded into any exchange analyser that
# already defines:
#   STOCK_DICT   : dict[ticker_str -> company_name_str]
#   fetch_stock  : (ticker: str) -> (yf.Ticker, str)
#   CURRENCY_SYM : str  e.g. "A$" or "₹"
#
# Public entry-points:
#   run_deep_scanner(stock_dict, fetch_fn, currency_sym, period, threshold)
# ══════════════════════════════════════════════════════════════════════════════

import math
import numpy as np
import pandas as pd

# ── A. Quantitative Scoring ────────────────────────────────────────────────────

def _safe(v, default=np.nan):
    """Safety wrapper to convert values to float, returning default if invalid."""
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def quant_score(hist: pd.DataFrame) -> dict:
    """
    Score a stock on six quantitative dimensions (each 0–100).
    
    METHODOLOGY:
    1. Momentum Score (20% weight) — Multi-horizon returns (5d, 21d, 63d) mapped 0-100
    2. Mean-Reversion Score (25% weight) — Z-score from 20d MA; oversold→high score
    3. Volatility Score (10% weight) — 30-day realized volatility; lower vol → safer
    4. Volume Surge Score (15% weight) — 5d/20d volume ratio; >2x → institutional interest
    5. RSI Score (15% weight) — RSI<30 oversold (score↑), RSI>70 overbought (score↓)
    6. Trend Strength (15% weight) — EMA-10 slope; positive slope → uptrend
    
    COMPOSITE: Weighted average of all 6 scores (0-100).
    INTERPRETATION: Score 60+ = bullish, 40- = bearish, 40-60 = neutral
    
    Returns a dict with individual scores + composite + raw helpers for game theory.
    """
    df   = hist.copy().dropna(subset=["Close"])
    if len(df) < 30:
        return {}

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── 1. Momentum Score ──────────────────────────────────────────────────
    # Multi-horizon return percentiles mapped 0-100
    # ret_5:  5-day return (short-term momentum)
    # ret_21: 21-day (1-month) return (medium-term momentum)
    # ret_63: 63-day (3-month) return (long-term momentum)
    # Weights: short 30%, medium 40%, long 30%
    ret_5   = _safe(close.pct_change(5).iloc[-1]  * 100)
    ret_21  = _safe(close.pct_change(21).iloc[-1] * 100)
    ret_63  = _safe(close.pct_change(63).iloc[-1] * 100)
    # Normalise: each ±20% maps to 0-100; cap at extremes
    def norm_ret(r, cap=20):
        return max(0.0, min(100.0, 50 + r / cap * 50))
    mom_score = (norm_ret(ret_5) * 0.3 +
                 norm_ret(ret_21) * 0.4 +
                 norm_ret(ret_63) * 0.3)

    # ── 2. Mean-Reversion Score ────────────────────────────────────────────
    # Distance from 20d mean in σ units (z-score), inverted for mean-reversion logic
    # When price is far below MA (negative z-score) → high score (buying opportunity)
    # When price is far above MA (positive z-score) → low score (selling opportunity)
    mu  = close.rolling(20).mean().iloc[-1]       # 20-day moving average
    sig = close.rolling(20).std().iloc[-1]        # 20-day standard deviation
    z   = (close.iloc[-1] - mu) / (sig + 1e-9)    # Z-score: how many σ from mean
    # z < -2 → score ~100 (deeply oversold, prime for reversion)
    # z > +2 → score ~0   (overbought, expect pullback)
    # z = 0  → score = 50 (neutral)
    mr_score = max(0.0, min(100.0, 50 - z * 25))

    # ── 3. Volatility / Risk Score ─────────────────────────────────────────
    # Lower 30-day realized volatility = safer entry price, less whipsaw risk
    # Higher volatility = riskier but potentially higher reward
    # Calculation: 30-day daily returns std dev, annualized (multiply by √252 trading days)
    daily_ret = close.pct_change().dropna()
    rv_30 = _safe(daily_ret.tail(30).std() * math.sqrt(252) * 100)  # annualised %
    # Mapping: ≤20% vol → score 90 (safe), ≥80% vol → score 10 (risky)
    # 20%-80% vol → linearly interpolated between 90 and 10
    vol_score = max(10.0, min(90.0, 90 - (rv_30 - 20) / 60 * 80)) if rv_30 == rv_30 else 50.0

    # ── 4. Volume Surge Score ──────────────────────────────────────────────
    # Institutional smart money often signals via volume spikes
    # If recent volume >> average volume → likely institutional accumulation/distribution
    vol_5    = _safe(vol.tail(5).mean())            # Last 5 days avg volume
    vol_20   = _safe(vol.tail(20).mean())           # Last 20 days avg volume
    ratio    = vol_5 / (vol_20 + 1e-9)              # 5d/20d ratio (1.0 = normal)
    # ratio 1.0 → score 50 (neutral)
    # ratio 2.0+ → score 83+ (strong volume surge, institutional activity)
    vs_score = max(0.0, min(100.0, 50 + (ratio - 1) * 33))

    # ── 5. RSI Score ───────────────────────────────────────────────────────
    # RSI = Relative Strength Index (momentum oscillator, 0-100)
    # Calculation: RS = avg_gains / avg_losses over 14 periods
    delta = close.diff()                              # Daily price changes
    gain  = delta.clip(lower=0).rolling(14).mean()   # Avg gains over 14 days
    loss  = (-delta.clip(upper=0)).rolling(14).mean() # Avg losses over 14 days
    rs    = gain / (loss + 1e-9)                     # RS = gains / losses
    rsi   = 100 - 100 / (1 + rs.iloc[-1])            # RSI = 100 - (100/(1+RS))
    rsi   = _safe(rsi, 50)
    # RSI <30 = oversold (potential bounce/reversal BUY signal) → score 90
    # RSI >70 = overbought (potential pullback/reversal SELL signal) → score 10
    # RSI =50 = neutral momentum → score 50
    rsi_score = max(0.0, min(100.0, 50 + (50 - rsi) * 80 / 50))

    # ── 6. Trend Strength (EMA10 slope) ───────────────────────────────────
    # EMA-10 is fast exponential moving average (follows price closely)
    # Slope measures 5-day change: positive = uptrend, negative = downtrend
    ema10 = close.ewm(span=10, adjust=False).mean()  # 10-day EMA
    slope = _safe((ema10.iloc[-1] - ema10.iloc[-5]) / (ema10.iloc[-5] + 1e-9) * 100)  # % change over 5d
    # slope +1% in 5d → score 70 (strong uptrend)
    # slope 0% in 5d → score 50 (flat/consolidating)
    # slope -1% in 5d → score 30 (downtrend)
    trend_score = max(0.0, min(100.0, 50 + slope * 20))

    # ── 7. Moving Averages & VWAP ────────────────────────────────────────
    # EMAs are exponential moving averages (more weight on recent prices)
    # EMA-20: short-term trend (days), used for entry/exit timing
    # EMA-50: medium-term trend (weeks), support/resistance level
    # EMA-100: long-term trend (months)
    # EMA-200: very long-term trend (year); classic "above 200 EMA = uptrend"
    ema_20  = _safe(close.ewm(span=20,  adjust=False).mean().iloc[-1])
    ema_50  = _safe(close.ewm(span=50,  adjust=False).mean().iloc[-1])
    ema_100 = _safe(close.ewm(span=100, adjust=False).mean().iloc[-1])
    ema_200 = _safe(close.ewm(span=200, adjust=False).mean().iloc[-1])
    
    # VWAP (Volume-Weighted Average Price) = smart money's average entry price
    # Price > VWAP = institutions underwater, likely to dump (resistance)
    # Price < VWAP = institutions profitable, likely to accumulate more (support)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = _safe((typical_price * vol).sum() / vol.sum())

    composite = (
        mom_score   * 0.20 +
        mr_score    * 0.25 +
        vol_score   * 0.10 +
        vs_score    * 0.15 +
        rsi_score   * 0.15 +
        trend_score * 0.15
    )

    return {
        "Momentum Score":      round(mom_score,   1),
        "Mean-Reversion Score":round(mr_score,    1),
        "Volatility Score":    round(vol_score,   1),
        "Volume Surge Score":  round(vs_score,    1),
        "RSI Score":           round(rsi_score,   1),
        "Trend Strength Score":round(trend_score, 1),
        "── Quant Composite ──":round(composite,  1),
        # Moving Averages & VWAP
        "EMA 20":              round(ema_20,   2),
        "EMA 50":              round(ema_50,   2),
        "EMA 100":             round(ema_100,  2),
        "EMA 200":             round(ema_200,  2),
        "VWAP":                round(vwap,     2),
        # raw helpers for game-theory layer
        "_rsi":   rsi,
        "_rv30":  rv_30,
        "_vol_ratio": ratio,
        "_ret_21": ret_21,
        "_z": z,
    }


# ── B. Game Theory Scoring ─────────────────────────────────────────────────────

def game_theory_score(hist: pd.DataFrame, quant: dict, info: dict) -> dict:
    """
    Model market dynamics using Game Theory constructs.
    
    PHILOSOPHY: Markets are 3-player games between:
      • INSTITUTIONAL investors (large blocks, move on fundamentals & mean-reversion)
      • RETAIL investors (small traders, chase momentum & RSI extremes)
      • MARKET MAKERS (provide liquidity, profit from bid-ask spreads)
    
    Each player has dominant strategies derived from observable market signals:
      • Institutional BUY signal: Price down >10% + Volume spike 40%+ above average
      • Retail BUY signal: RSI < 35 (oversold, momentum chasers buy dips)
      • Market Maker signal: Inferred from bid-ask spread (high → SELL, low → BUY)
    
    THREE SCORING MODELS:
      1. Nash Equilibrium: When all 3 players align on same strategy → strong conviction
      2. Prisoner's Dilemma: Payoff matrix showing optimal strategy for each player
         - R = reward (price appreciates if everyone holds)
         - T = temptation (best payoff for one trader if others hold)
         - S = sucker (worst payoff: you buy, others dump)
         - P = punishment (everyone sells, price crashes)
      3. Institutional vs Retail Divergence: When they disagree, contrarian opportunity
    
    Returns game theory composite score (0-100) + detailed breakdown.
    """
    if not quant:
        return {}

    df    = hist.copy().dropna(subset=["Close"])
    close = df["Close"]
    vol   = df["Volume"]

    # ── B1. Nash Equilibrium Player-Behaviour Model ────────────────────────
    #
    # Players: Retail, Institutional, Market Maker
    # Strategy spaces: BUY | HOLD | SELL
    #
    # We infer each player's dominant strategy from observable signals:
    #   • Institutional : volume spikes on price drops (accumulation signal)
    #   • Retail        : RSI extremes, price momentum chasing
    #   • Market Maker  : bid-ask spread proxy via ATR / price
    #
    rsi      = quant.get("_rsi",  50)
    rv30     = quant.get("_rv30", 30)
    vr       = quant.get("_vol_ratio", 1.0)   # 5d/20d volume ratio
    ret_21   = quant.get("_ret_21", 0.0)
    z        = quant.get("_z", 0.0)           # Z-score from 20d MA

    # Institutional signal  ─ based on mean reversion + volume
    # More responsive: price weakness + volume OR deep oversold + reversal signal
    price_weak   = ret_21 < -5           # price down >5% (more sensitive)
    vol_spiked   = vr > 1.3              # volume 30% above average (more sensitive)
    deeply_oversold = z < -1.5           # price far below 20d MA
    inst_buy     = (price_weak and vol_spiked) or deeply_oversold
    inst_signal  = "BUY" if inst_buy else ("SELL" if (ret_21 > 8 and vr < 0.85) or z > 1.5 else "HOLD")

    # Retail signal ─ momentum chasers / RSI-driven, more responsive
    retail_signal = "BUY" if rsi < 40 else ("SELL" if rsi > 65 else "HOLD")

    # Market Maker signal ─ spread inference via high-low range, more responsive
    hl_ratio = _safe(((df["High"] - df["Low"]) / (df["Close"] + 1e-9)).tail(5).mean())
    mm_signal = "SELL" if hl_ratio > 0.035 else ("BUY" if hl_ratio < 0.018 else "HOLD")

    # Nash Equilibrium: if all three players reach the same strategy,
    # that IS the Nash Equilibrium (no player gains by unilaterally deviating)
    strategies = [inst_signal, retail_signal, mm_signal]
    ne_votes   = {s: strategies.count(s) for s in ("BUY","HOLD","SELL")}
    ne_dominant = max(ne_votes, key=ne_votes.get)
    ne_strength = ne_votes[ne_dominant] / 3 * 100   # 33 | 67 | 100

    # Score: BUY equilibrium = high score, SELL = low
    ne_score = (
        70 + ne_strength * 0.3 if ne_dominant == "BUY"  else
        50                     if ne_dominant == "HOLD" else
        30 - ne_strength * 0.3
    )
    ne_score = max(0.0, min(100.0, ne_score))

    # ── B2. Prisoner's Dilemma Payoff Matrix ──────────────────────────────
    #
    # Each investor faces: COOPERATE (hold/buy) vs DEFECT (sell)
    #
    # Payoff matrix (row=you, col=market):
    #              Market COOPERATES   Market DEFECTS
    #  You BUY          +R                  -S
    #  You SELL         +T                  -P
    #
    # R = reward (price appreciates if everyone holds)
    # T = temptation (sell before others — greatest gain)
    # S = sucker's loss (buy while market dumps)
    # P = punishment (everyone sells, price crashes)
    #
    # We estimate R, T, S, P from historical volatility and drawdown
    w52_high   = _safe(close.rolling(252).max().iloc[-1], close.max())
    current    = _safe(close.iloc[-1])
    drawdown   = (current - w52_high) / (w52_high + 1e-9) * 100  # negative

    R = max(1.0, -drawdown * 0.5)          # recovery potential
    T = max(1.0, rv30 * 0.4)               # temptation ~ volatility
    S = max(1.0, -drawdown * 0.8)          # sucker loss ~ depth of drop
    P = max(1.0, rv30 * 0.3)               # panic loss ~ vol

    # Dominant strategy: if T > R → rational to sell (defect)
    # Nash Equilibrium of PD → both defect (T > R)
    # But if S > T → cooperation dominates
    pd_cooperate = S > T                   # True → BUY / HOLD is rational
    pd_label     = "COOPERATE (BUY/HOLD)" if pd_cooperate else "DEFECT (SELL)"
    pd_score     = (60 + (S - T) / (S + T + 1e-9) * 40) if pd_cooperate else \
                   (40 - (T - S) / (S + T + 1e-9) * 40)
    pd_score     = max(0.0, min(100.0, pd_score))

    # ── B3. Institutional vs Retail Divergence Signal ─────────────────────
    #
    # Large blocks (institutional) tend to move on fundamentals;
    # retail on recency/momentum. Divergence = opportunity.
    #
    inst_score_num  = 80 if inst_signal == "BUY"  else (50 if inst_signal == "HOLD" else 20)
    retail_score_num= 80 if retail_signal == "BUY" else (50 if retail_signal == "HOLD" else 20)
    divergence      = abs(inst_score_num - retail_score_num)

    # Large divergence (inst buys, retail sells) → contrarian opportunity
    div_score = 50 + divergence * 0.5 if inst_signal == "BUY" and retail_signal == "SELL" \
           else 50 - divergence * 0.3 if inst_signal == "SELL" and retail_signal == "BUY" \
           else 50.0
    div_score = max(0.0, min(100.0, div_score))

    gt_composite = (
        ne_score  * 0.40 +
        pd_score  * 0.35 +
        div_score * 0.25
    )

    return {
        # Nash Equilibrium
        "NE — Inst. Signal":      inst_signal,
        "NE — Retail Signal":     retail_signal,
        "NE — Market Maker":      mm_signal,
        "NE — Dominant Strategy": ne_dominant,
        "NE — Equilibrium Strength": f"{ne_strength:.0f}%",
        "NE Score":               round(ne_score, 1),
        # Prisoner's Dilemma
        "PD — Payoff R (Reward)": round(R, 2),
        "PD — Payoff T (Tempt.)": round(T, 2),
        "PD — Payoff S (Sucker)": round(S, 2),
        "PD — Payoff P (Punish)": round(P, 2),
        "PD — Rational Choice":   pd_label,
        "PD Score":               round(pd_score, 1),
        # Inst vs Retail
        "Inst vs Retail Divergence": f"{divergence:.0f} pts",
        "Divergence Signal":         inst_signal + " (inst) / " + retail_signal + " (retail)",
        "Divergence Score":       round(div_score, 1),
        # Composite
        "── GT Composite ──":     round(gt_composite, 1),
    }


# ── Scanner + Screener ─────────────────────────────────────────────────────────

def _ema10_uptrend(close: pd.Series, lookback: int = 5) -> bool:
    """Return True if EMA-10 has been rising for the last `lookback` days."""
    if len(close) < 15:
        return False
    ema10 = close.ewm(span=10, adjust=False).mean()
    return float(ema10.iloc[-1]) > float(ema10.iloc[-lookback])


def _drawdown_from_high(close: pd.Series) -> float:
    """Return % drawdown of latest close from 52-week high."""
    w52 = close.rolling(min(252, len(close))).max().iloc[-1]
    return (close.iloc[-1] - w52) / (w52 + 1e-9) * 100  # negative


def print_scanner_header(exchange_name: str, threshold: float):
    from colorama import Fore, Style
    print(f"\n{Fore.CYAN}╔{'═'*68}╗")
    print(f"║  {'🔬  DEEP SCANNER  —  ' + exchange_name:^66}  ║")
    print(f"║  {f'Stocks ≥{threshold:.0f}% below 52W-high  +  EMA-10 Uptrend':^66}  ║")
    print(f"╚{'═'*68}╝{Style.RESET_ALL}")


def print_quant_block(q: dict, currency_sym: str):
    from colorama import Fore, Style
    from tabulate import tabulate

    print(f"\n{Fore.YELLOW}  ━━  A. QUANTITATIVE ANALYSIS  ━━{Style.RESET_ALL}")
    rows = []
    for k, v in q.items():
        if k.startswith("_"):
            continue
        # Scores (0-100 scale)
        if any(keyword in k for keyword in ["Score", "Composite"]):
            score = float(v)
            col   = Fore.GREEN if score >= 60 else (Fore.RED if score < 40 else Fore.YELLOW)
            rows.append([f"  {k}", f"{col}{v}/100{Style.RESET_ALL}", _bar(score)])
        # EMAs and VWAP (currency-formatted)
        elif any(keyword in k for keyword in ["EMA", "VWAP"]):
            val = float(v) if isinstance(v, (int, float)) else v
            rows.append([f"  {k}", f"{Fore.CYAN}{currency_sym}{val:.2f}{Style.RESET_ALL}", ""])
    print(tabulate(rows, tablefmt="plain"))


def print_gt_block(g: dict):
    from colorama import Fore, Style
    from tabulate import tabulate

    print(f"\n{Fore.YELLOW}  ━━  B. GAME THEORY ANALYSIS  ━━{Style.RESET_ALL}")
    rows = []
    for k, v in g.items():
        if "Composite" in k:
            score = float(v)
            col   = Fore.GREEN if score >= 60 else (Fore.RED if score < 40 else Fore.YELLOW)
            rows.append([f"  {k}", f"{col}{v}/100{Style.RESET_ALL}", _bar(score)])
        elif isinstance(v, (int, float)):
            score = float(v)
            col   = Fore.GREEN if score >= 60 else (Fore.RED if score < 40 else Fore.YELLOW)
            rows.append([f"  {k}", f"{col}{v}/100{Style.RESET_ALL}", _bar(score)])
        else:
            # String fields: colour BUY/SELL
            sv = str(v)
            col = (Fore.GREEN if "BUY" in sv.upper() or "COOPERATE" in sv.upper()
                   else Fore.RED if "SELL" in sv.upper() or "DEFECT" in sv.upper()
                   else Fore.WHITE)
            rows.append([f"  {k}", f"{col}{sv}{Style.RESET_ALL}", ""])
    print(tabulate(rows, tablefmt="plain"))


def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for a 0-100 score."""
    filled = int(round(score / 100 * width))
    from colorama import Fore, Style
    col = Fore.GREEN if score >= 60 else (Fore.RED if score < 40 else Fore.YELLOW)
    return col + "█" * filled + "░" * (width - filled) + Style.RESET_ALL


def final_deep_verdict(q: dict, g: dict) -> str:
    from colorama import Fore, Style
    qc = q.get("── Quant Composite ──", 50)
    gc = g.get("── GT Composite ──", 50)
    total = qc * 0.55 + gc * 0.45
    col   = Fore.GREEN if total >= 65 else (Fore.RED if total < 40 else Fore.YELLOW)
    label = ("STRONG BUY" if total >= 75 else
             "BUY"        if total >= 60 else
             "HOLD"       if total >= 45 else
             "SELL"       if total >= 30 else
             "STRONG SELL")
    return (f"\n{Fore.CYAN}  {'─'*60}")    + \
           (f"\n  Deep Score:  {col}▶  {label}  ({total:.1f}/100)  ◀{Style.RESET_ALL}") + \
           (f"\n  Quant={qc}/100   Game-Theory={gc}/100") + \
           (f"\n{Fore.WHITE}  ⚠  Not financial advice. Use as one input only.{Style.RESET_ALL}\n")


# ── Main entry point ───────────────────────────────────────────────────────────

def run_deep_scanner(stock_dict: dict, fetch_fn, currency_sym: str,
                     period: str = "1y", threshold: float = 40.0,
                     exchange_name: str = "Exchange"):
    """
    Screen all stocks in `stock_dict` for:
      1. Down ≥ threshold% from 52W high
      2. EMA-10 uptrend (last 5 days)

    Then score each matched stock with Quant + Game Theory.
    """
    import yfinance as yf
    from colorama import Fore, Style
    from tabulate import tabulate

    print_scanner_header(exchange_name, threshold)
    print(f"\n  {Fore.WHITE}Scanning {len(stock_dict)} stocks ...{Style.RESET_ALL}\n")

    matches = []

    for ticker, name in stock_dict.items():
        try:
            stock, full_ticker = fetch_fn(ticker)
            hist = stock.history(period=period)
            if hist.empty or len(hist) < 30:
                continue
            close = hist["Close"].dropna()
            dd    = _drawdown_from_high(close)
            if dd <= -threshold and _ema10_uptrend(close):
                matches.append((ticker, name, full_ticker, hist, stock, dd))
        except Exception:
            continue

    if not matches:
        print(f"  {Fore.YELLOW}No stocks matched the screen criteria.{Style.RESET_ALL}")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    summary_rows = []
    for (ticker, name, ft, hist, stock, dd) in matches:
        c = float(hist["Close"].iloc[-1])
        summary_rows.append([
            f"{Fore.YELLOW}{ticker}{Style.RESET_ALL}",
            name[:32],
            f"{Fore.WHITE}{currency_sym}{c:.2f}{Style.RESET_ALL}",
            f"{Fore.RED}{dd:.1f}%{Style.RESET_ALL}",
            f"{Fore.GREEN}✔{Style.RESET_ALL}",
        ])

    print(f"  {Fore.GREEN}Found {len(matches)} stock(s) matching screen:{Style.RESET_ALL}\n")
    print(tabulate(summary_rows,
                   headers=["Ticker","Company","Price","Drawdown","EMA10↑"],
                   tablefmt="rounded_outline"))

    # ── Deep scoring per match ─────────────────────────────────────────────
    for (ticker, name, ft, hist, stock, dd) in matches:
        c      = float(hist["Close"].iloc[-1])
        w52h   = float(hist["Close"].rolling(min(252,len(hist))).max().iloc[-1])
        print(f"\n{Fore.CYAN}{'═'*68}")
        print(f"  🔬  DEEP ANALYSIS:  {ft}  —  {name}")
        print(f"  Price: {currency_sym}{c:.4f}   52W-High: {currency_sym}{w52h:.4f}"
              f"   Drawdown: {Fore.RED}{dd:.1f}%{Style.RESET_ALL}   EMA-10: {Fore.GREEN}↑ Uptrend{Style.RESET_ALL}")
        print(f"{'═'*68}{Style.RESET_ALL}")

        # Quant
        q = quant_score(hist)
        if q:
            print_quant_block(q, currency_sym)
        else:
            print(f"  {Fore.YELLOW}Insufficient data for quant scoring.{Style.RESET_ALL}")
            continue

        # Game Theory
        info = {}
        try:
            info = stock.info or {}
        except Exception:
            pass
        g = game_theory_score(hist, q, info)
        if g:
            print_gt_block(g)

        # Final verdict
        print(final_deep_verdict(q, g))



# ── Nifty 50 stocks database ──────────────────────────────────────────────────
NIFTY50_STOCKS = {
    # Financial Services
    "RELIANCE":  "Reliance Industries Limited",
    "ASIANPAINT": "Asian Paints (India) Limited",
    "HDFCBANK":  "HDFC Bank Limited",
    "ICICIBANK": "ICICI Bank Limited",
    "AXISBANK":  "Axis Bank Limited",
    "KOTAKBANK": "Kotak Mahindra Bank Limited",
    "LICT":      "Life Insurance Corporation of India",
    "SBILIFE":   "SBI Life Insurance Company Limited",
    "HDFCLIFE":  "HDFC Life Insurance Company Limited",
    # IT & Software
    "TCS":       "Tata Consultancy Services Limited",
    "INFY":      "Infosys Limited",
    "WIPRO":     "Wipro Limited",
    "TECHM":     "Tech Mahindra Limited",
    "LTTS":      "L&T Technology Services Limited",
    "MPHASIS":   "Mphasis Limited",
    # Automobiles
    "MARUTI":    "Maruti Suzuki India Limited",
    "BAJAJFINSV": "Bajaj Finserv Limited",
    "BAJAJ-AUTO": "Bajaj Auto Limited",
    "TATAMOTORS": "Tata Motors Limited",
    "EICHERMOT": "Eicher Motors Limited",
    "HEROMOTOCO": "Hero MotoCorp Limited",
    # Construction & Materials
    "JSWSTEEL":  "JSW Steel Limited",
    "TATASTEEL": "Tata Steel Limited",
    "LT":        "Larsen & Toubro Limited",
    "ADANIPORTS": "Adani Ports and Special Economic Zone Limited",
    "ADANIGREEN": "Adani Green Energy Limited",
    # Healthcare & Pharma
    "SUNPHARMA": "Sun Pharmaceutical Industries Limited",
    "DRREDDY":   "Dr. Reddy's Laboratories Limited",
    "CIPLA":     "Cipla Limited",
    "DIVISLAB":  "Divi's Laboratories Limited",
    "LUPIN":     "Lupin Limited",
    # Consumer Staples & Discretionary
    "HINDUNILVR": "Hindustan Unilever Limited",
    "ITC":       "ITC Limited",
    "NESTLEIND": "Nestlé India Limited",
    "BRITANNIA": "Britannia Industries Limited",
    "MARICO":    "Marico Limited",
    # Oil & Gas, Energy
    "ONGC":      "Oil and Natural Gas Corporation Limited",
    "NTPC":      "NTPC Limited",
    "POWERGRID": "Power Grid Corporation of India Limited",
    # Finance & Investment
    "HDFCAMC":   "HDFC Asset Management Company Limited",
    "PIDILITIND": "Pidilite Industries Limited",
    "BPCL":      "Bharat Petroleum Corporation Limited",
    # Metals & Mining
    "HINDALCO":  "Hindalco Industries Limited",
    "SBIN":      "State Bank of India",
    # Telecom
    "BHARTIARTL": "Bharti Airtel Limited",
    # Consumer
    "TATACONSUM": "Tata Consumer Products Limited",
    # Realty
    "DLF":       "DLF Limited",
    # Multi-sector
    "M&M":       "Mahindra & Mahindra Limited",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def colour(val, good_positive=True):
    """Return green/red coloured string based on value sign."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"{Fore.WHITE}N/A{Style.RESET_ALL}"
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return str(val)
    color = Fore.GREEN if (fval >= 0) == good_positive else Fore.RED
    return f"{color}{val}{Style.RESET_ALL}"

def fmt(val, decimals=2, suffix=""):
    """Format a number nicely."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        return f"{float(val):,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)

def fmt_large(val):
    """Format large numbers as Cr/L (Crore/Lakh)."""
    if val is None:
        return "N/A"
    try:
        val = float(val)
        if val >= 1e10:
            return f"₹{val/1e7:.2f}Cr"
        if val >= 1e7:
            return f"₹{val/1e7:.2f}Cr"
        if val >= 1e5:
            return f"₹{val/1e5:.2f}L"
        return f"₹{val:,.0f}"
    except (TypeError, ValueError):
        return "N/A"

def pct_change_str(current, prev):
    if current is None or prev is None or prev == 0:
        return "N/A"
    pct = (current - prev) / abs(prev) * 100
    sign = "+" if pct >= 0 else ""
    col = Fore.GREEN if pct >= 0 else Fore.RED
    return f"{col}{sign}{pct:.2f}%{Style.RESET_ALL}"

# ── Search ────────────────────────────────────────────────────────────────────

def search_stocks(query: str):
    """Search Nifty 50 stocks by ticker or company name keyword."""
    query = query.upper().strip()
    results = []
    for ticker, name in NIFTY50_STOCKS.items():
        if query in ticker or query in name.upper():
            results.append((ticker, name))
    return results

# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_stock(ticker: str):
    """Fetch yfinance Ticker for an NSE stock (appends .NS if needed)."""
    nse_ticker = ticker.upper()
    if not nse_ticker.endswith(".NS"):
        nse_ticker += ".NS"
    stock = yf.Ticker(nse_ticker)
    return stock, nse_ticker

# ── Fundamentals ──────────────────────────────────────────────────────────────

def get_fundamentals(stock):
    info = stock.info
    if not info or "symbol" not in info:
        return None

    data = {
        "Company":          info.get("longName", "N/A"),
        "Sector":           info.get("sector", "N/A"),
        "Industry":         info.get("industry", "N/A"),
        "Exchange":         info.get("exchange", "N/A"),
        "Currency":         info.get("currency", "INR"),
        # Price
        "Current Price":    info.get("currentPrice") or info.get("regularMarketPrice"),
        "52W High":         info.get("fiftyTwoWeekHigh"),
        "52W Low":          info.get("fiftyTwoWeekLow"),
        "Avg Volume (10d)": info.get("averageVolume10days"),
        # Valuation
        "Market Cap":       info.get("marketCap"),
        "Enterprise Value": info.get("enterpriseValue"),
        "P/E (TTM)":        info.get("trailingPE"),
        "Forward P/E":      info.get("forwardPE"),
        "PEG Ratio":        info.get("pegRatio"),
        "P/B Ratio":        info.get("priceToBook"),
        "P/S Ratio":        info.get("priceToSalesTrailing12Months"),
        "EV/EBITDA":        info.get("enterpriseToEbitda"),
        "EV/Revenue":       info.get("enterpriseToRevenue"),
        # Profitability
        "Revenue (TTM)":    info.get("totalRevenue"),
        "Gross Margin":     info.get("grossMargins"),
        "Operating Margin": info.get("operatingMargins"),
        "Net Margin":       info.get("profitMargins"),
        "ROE":              info.get("returnOnEquity"),
        "ROA":              info.get("returnOnAssets"),
        "EBITDA":           info.get("ebitda"),
        # Dividends
        "Dividend Yield":   info.get("dividendYield"),
        "Dividend Rate":    info.get("dividendRate"),
        "Payout Ratio":     info.get("payoutRatio"),
        "Ex-Div Date":      info.get("exDividendDate"),
        # Balance Sheet
        "Total Debt":       info.get("totalDebt"),
        "Total Cash":       info.get("totalCash"),
        "Debt/Equity":      info.get("debtToEquity"),
        "Current Ratio":    info.get("currentRatio"),
        "Quick Ratio":      info.get("quickRatio"),
        # Growth
        "Revenue Growth":   info.get("revenueGrowth"),
        "Earnings Growth":  info.get("earningsGrowth"),
        "EPS (TTM)":        info.get("trailingEps"),
        "Forward EPS":      info.get("forwardEps"),
        # Analyst
        "Target Price":     info.get("targetMeanPrice"),
        "Recommendation":   info.get("recommendationKey", "N/A").upper(),
        "# Analysts":       info.get("numberOfAnalystOpinions"),
        "Beta":             info.get("beta"),
    }
    return data

def print_fundamentals(data, ticker):
    print(f"\n{Fore.CYAN}{'═'*60}")
    print(f"  📊  FUNDAMENTALS — {ticker}")
    print(f"{'═'*60}{Style.RESET_ALL}")

    sections = {
        "🏢  Company Info": [
            "Company", "Sector", "Industry", "Exchange", "Currency"
        ],
        "💰  Price & Volume": [
            "Current Price", "52W High", "52W Low", "Avg Volume (10d)"
        ],
        "📈  Valuation": [
            "Market Cap", "Enterprise Value", "P/E (TTM)", "Forward P/E",
            "PEG Ratio", "P/B Ratio", "P/S Ratio", "EV/EBITDA", "EV/Revenue"
        ],
        "💹  Profitability": [
            "Revenue (TTM)", "Gross Margin", "Operating Margin", "Net Margin",
            "ROE", "ROA", "EBITDA"
        ],
        "💸  Dividends": [
            "Dividend Yield", "Dividend Rate", "Payout Ratio", "Ex-Div Date"
        ],
        "🏦  Balance Sheet": [
            "Total Debt", "Total Cash", "Debt/Equity", "Current Ratio", "Quick Ratio"
        ],
        "📉  Growth & EPS": [
            "Revenue Growth", "Earnings Growth", "EPS (TTM)", "Forward EPS"
        ],
        "🎯  Analyst View": [
            "Target Price", "Recommendation", "# Analysts", "Beta"
        ],
    }

    for section, keys in sections.items():
        rows = []
        for k in keys:
            v = data.get(k)
            # Format based on key type
            if k in ("Market Cap", "Enterprise Value", "Revenue (TTM)", "EBITDA",
                     "Total Debt", "Total Cash"):
                display = fmt_large(v)
            elif k in ("Gross Margin", "Operating Margin", "Net Margin",
                       "ROE", "ROA", "Dividend Yield", "Payout Ratio",
                       "Revenue Growth", "Earnings Growth"):
                display = f"{float(v)*100:.2f}%" if v is not None else "N/A"
            elif k in ("Current Price", "52W High", "52W Low", "Target Price",
                       "Dividend Rate", "EPS (TTM)", "Forward EPS"):
                display = f"₹{fmt(v)}" if v is not None else "N/A"
            elif k == "Ex-Div Date" and v:
                try:
                    display = datetime.fromtimestamp(v).strftime("%d %b %Y")
                except Exception:
                    display = str(v)
            elif k == "Avg Volume (10d)" and v:
                display = f"{int(v):,}"
            elif k == "Recommendation":
                rec = str(v)
                color = (Fore.GREEN if "BUY" in rec else
                         Fore.RED if "SELL" in rec else Fore.YELLOW)
                display = f"{color}{rec}{Style.RESET_ALL}"
            else:
                display = fmt(v) if v is not None else "N/A"

            rows.append([f"  {k}", display])

        print(f"\n{Fore.YELLOW}{section}{Style.RESET_ALL}")
        print(tabulate(rows, tablefmt="plain"))

# ── Technicals ────────────────────────────────────────────────────────────────

def compute_technicals(hist: pd.DataFrame):
    """Compute a wide range of technical indicators using 'ta' library."""
    df = hist.copy()
    if df.empty or len(df) < 20:
        return None, df

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Trend ──
    df["SMA_20"]  = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"]  = ta.trend.sma_indicator(close, window=50)
    df["SMA_200"] = ta.trend.sma_indicator(close, window=200)
    df["EMA_12"]  = ta.trend.ema_indicator(close, window=12)
    df["EMA_26"]  = ta.trend.ema_indicator(close, window=26)

    macd = ta.trend.MACD(close)
    df["MACD"]        = macd.macd()
    df["MACD_Signal"]  = macd.macd_signal()
    df["MACD_Hist"]   = macd.macd_diff()

    # ADX requires sufficient data; wrap in try-except
    try:
        adx = ta.trend.ADXIndicator(high, low, close)
        df["ADX"] = adx.adx()
    except (IndexError, ValueError):
        df["ADX"] = np.nan

    # ── Momentum ──
    df["RSI_14"]    = ta.momentum.RSIIndicator(close, window=14).rsi()
    
    # Stochastic and Williams R require sufficient data
    try:
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        df["Stoch_K"]   = stoch.stoch()
        df["Stoch_D"]   = stoch.stoch_signal()
    except (IndexError, ValueError):
        df["Stoch_K"]   = np.nan
        df["Stoch_D"]   = np.nan
    
    try:
        df["Williams_R"]= ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
    except (IndexError, ValueError):
        df["Williams_R"] = np.nan
    
    df["ROC"]       = ta.momentum.ROCIndicator(close, window=12).roc()
    
    try:
        df["MFI"]       = ta.volume.MFIIndicator(high, low, close, vol).money_flow_index()
    except (IndexError, ValueError):
        df["MFI"] = np.nan

    # ── Volatility ──
    bb = ta.volatility.BollingerBands(close, window=20)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Mid"]    = bb.bollinger_mavg()
    df["BB_Lower"]  = bb.bollinger_lband()
    df["BB_Width"]  = bb.bollinger_wband()
    df["ATR"]       = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

    # ── Volume ──
    df["OBV"]       = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["VWAP"]      = (close * vol).cumsum() / vol.cumsum()

    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    c      = float(latest["Close"])

    summary = {
        # Trend
        "SMA 20":         latest["SMA_20"],
        "SMA 50":         latest["SMA_50"],
        "SMA 200":        latest["SMA_200"],
        "EMA 12":         latest["EMA_12"],
        "EMA 26":         latest["EMA_26"],
        "MACD":           latest["MACD"],
        "MACD Signal":    latest["MACD_Signal"],
        "MACD Histogram": latest["MACD_Hist"],
        "ADX":            latest["ADX"],
        # Momentum
        "RSI (14)":       latest["RSI_14"],
        "Stoch %K":       latest["Stoch_K"],
        "Stoch %D":       latest["Stoch_D"],
        "Williams %R":    latest["Williams_R"],
        "ROC (12)":       latest["ROC"],
        "MFI (14)":       latest["MFI"],
        # Volatility
        "BB Upper":       latest["BB_Upper"],
        "BB Mid":         latest["BB_Mid"],
        "BB Lower":       latest["BB_Lower"],
        "BB Width %":     latest["BB_Width"],
        "ATR":            latest["ATR"],
        # Volume
        "OBV":            latest["OBV"],
        "VWAP":           latest["VWAP"],
        # Price vs MAs
        "Price vs SMA20": c - float(latest["SMA_20"]) if not np.isnan(latest["SMA_20"]) else None,
        "Price vs SMA50": c - float(latest["SMA_50"]) if not np.isnan(latest["SMA_50"]) else None,
        "Price vs SMA200":c - float(latest["SMA_200"]) if not np.isnan(latest["SMA_200"]) else None,
    }
    return summary, df

def signal_label(key, val):
    """Return a simple signal string for key indicators."""
    if val is None or (isinstance(val, float) and np.isnan(float(val))):
        return ""
    val = float(val)
    signals = {
        "RSI (14)":    ("🔴 Overbought", "🟢 Oversold", 70, 30),
        "Stoch %K":    ("🔴 Overbought", "🟢 Oversold", 80, 20),
        "MFI (14)":    ("🔴 Overbought", "🟢 Oversold", 80, 20),
        "Williams %R": ("🟢 Oversold", "🔴 Overbought", -20, -80),
        "ADX":         ("💪 Strong Trend", "😴 Weak Trend", 25, None),
    }
    if key in signals:
        up_label, down_label, up_thresh, down_thresh = signals[key]
        if val >= up_thresh:
            return up_label
        if down_thresh is not None and val <= down_thresh:
            return down_label
    if key in ("Price vs SMA20", "Price vs SMA50", "Price vs SMA200"):
        return "🟢 Above MA" if val > 0 else "🔴 Below MA"
    if key == "MACD Histogram":
        return "🟢 Bullish" if val > 0 else "🔴 Bearish"
    return ""

def print_technicals(summary, ticker, current_price):
    print(f"\n{Fore.CYAN}{'═'*60}")
    print(f"  📉  TECHNICALS — {ticker}")
    print(f"{'═'*60}{Style.RESET_ALL}")

    sections = {
        "📏  Moving Averages": [
            "SMA 20", "SMA 50", "SMA 200", "EMA 12", "EMA 26", "VWAP",
            "Price vs SMA20", "Price vs SMA50", "Price vs SMA200"
        ],
        "⚡  MACD": [
            "MACD", "MACD Signal", "MACD Histogram"
        ],
        "🌀  Momentum Oscillators": [
            "RSI (14)", "Stoch %K", "Stoch %D",
            "Williams %R", "ROC (12)", "MFI (14)", "ADX"
        ],
        "📊  Bollinger Bands & Volatility": [
            "BB Upper", "BB Mid", "BB Lower", "BB Width %", "ATR"
        ],
        "🔊  Volume": [
            "OBV"
        ],
    }

    for section, keys in sections.items():
        rows = []
        for k in keys:
            v = summary.get(k)
            if v is None or (isinstance(v, float) and np.isnan(float(v))):
                display = "N/A"
            else:
                fv = float(v)
                if k in ("Price vs SMA20","Price vs SMA50","Price vs SMA200"):
                    sign = "+" if fv >= 0 else ""
                    col  = Fore.GREEN if fv >= 0 else Fore.RED
                    display = f"{col}{sign}₹{fv:.2f}{Style.RESET_ALL}"
                elif k in ("BB Upper","BB Mid","BB Lower","SMA 20","SMA 50",
                           "SMA 200","EMA 12","EMA 26","VWAP","ATR"):
                    display = f"₹{fv:.2f}"
                elif k == "OBV":
                    display = f"{fv:,.0f}"
                elif k == "BB Width %":
                    display = f"{fv:.2f}%"
                else:
                    display = f"{fv:.2f}"
            sig = signal_label(k, v)
            rows.append([f"  {k}", display, sig])

        print(f"\n{Fore.YELLOW}{section}{Style.RESET_ALL}")
        print(tabulate(rows, headers=["Indicator", "Value", "Signal"],
                       tablefmt="plain", colalign=("left","right","left")))

# ── Price History Chart (ASCII) ───────────────────────────────────────────────

def ascii_chart(hist, width=55, height=12):
    """Simple ASCII price chart."""
    closes = hist["Close"].dropna().tolist()
    if len(closes) < 5:
        return
    closes = closes[-width:]
    mn, mx = min(closes), max(closes)
    rng = mx - mn or 1

    print(f"\n{Fore.CYAN}  📈  Price Chart (last {len(closes)} days){Style.RESET_ALL}")
    print(f"  High: ₹{mx:.2f}  Low: ₹{mn:.2f}")

    grid = [[" "] * len(closes) for _ in range(height)]
    for i, c in enumerate(closes):
        row = int((c - mn) / rng * (height - 1))
        row = height - 1 - row
        grid[row][i] = "•"

    for i, row in enumerate(grid):
        price = mx - i * (rng / (height - 1))
        label = f"₹{price:7.2f} │"
        line  = "".join(row)
        col   = Fore.GREEN if i < height // 2 else Fore.RED
        print(f"  {Fore.WHITE}{label}{col}{line}{Style.RESET_ALL}")
    print(f"  {'─'*8}┴{'─'*len(closes)}")
    print(f"  {'◄ Older':>{8+len(closes)//2}}{'Newer ►':>{len(closes)//2}}")

# ── Overall Signal Summary ─────────────────────────────────────────────────────

def overall_signal(fund, tech):
    """Generate a simple buy/sell/hold summary."""
    signals = []
    if tech:
        rsi = tech.get("RSI (14)")
        if rsi:
            if rsi < 30: signals.append(("BUY",  "RSI oversold"))
            elif rsi > 70: signals.append(("SELL","RSI overbought"))

        macd_h = tech.get("MACD Histogram")
        if macd_h:
            signals.append(("BUY" if macd_h > 0 else "SELL", "MACD histogram"))

        vs_sma50 = tech.get("Price vs SMA50")
        if vs_sma50:
            signals.append(("BUY" if vs_sma50 > 0 else "SELL", "Price vs SMA50"))

        vs_sma200 = tech.get("Price vs SMA200")
        if vs_sma200:
            signals.append(("BUY" if vs_sma200 > 0 else "SELL", "Price vs SMA200"))

    if fund:
        pe = fund.get("P/E (TTM)")
        if pe and not np.isnan(float(pe if pe else np.nan)):
            signals.append(("BUY" if float(pe) < 25 else "SELL", f"P/E={float(pe):.1f}"))

        rec = str(fund.get("Recommendation",""))
        if "BUY" in rec: signals.append(("BUY", "Analyst rec"))
        elif "SELL" in rec: signals.append(("SELL","Analyst rec"))

    buys  = sum(1 for s, _ in signals if s == "BUY")
    sells = sum(1 for s, _ in signals if s == "SELL")
    total = len(signals)

    if total == 0:
        verdict = "HOLD"
        col = Fore.YELLOW
    elif buys / total >= 0.6:
        verdict = "BUY"
        col = Fore.GREEN
    elif sells / total >= 0.6:
        verdict = "SELL"
        col = Fore.RED
    else:
        verdict = "HOLD"
        col = Fore.YELLOW

    print(f"\n{Fore.CYAN}{'═'*60}")
    print(f"  🎯  OVERALL SIGNAL SUMMARY")
    print(f"{'═'*60}{Style.RESET_ALL}")
    print(f"\n  Verdict:  {col}▶  {verdict}  ◀{Style.RESET_ALL}  "
          f"({buys} bullish / {sells} bearish signals from {total} checks)")
    print()
    for s, reason in signals:
        col2 = Fore.GREEN if s == "BUY" else Fore.RED
        print(f"    {col2}{'✔' if s=='BUY' else '✘'}  {s:<4}{Style.RESET_ALL}  ← {reason}")
    print(f"\n{Fore.WHITE}  ⚠  This is not financial advice. Do your own research.{Style.RESET_ALL}\n")

# ── Main analysis flow ────────────────────────────────────────────────────────

def analyse(ticker: str, period: str = "1y"):
    print(f"\n{Fore.CYAN}  🔍  Fetching data for {ticker.upper()} ...{Style.RESET_ALL}")
    stock, nse_ticker = fetch_stock(ticker)

    # History
    hist = stock.history(period=period)
    if hist.empty:
        print(f"{Fore.RED}  ✘  No data found for {nse_ticker}. "
              f"Check the ticker or try adding .NS manually.{Style.RESET_ALL}")
        return

    current_price = hist["Close"].iloc[-1]
    prev_price    = hist["Close"].iloc[-2] if len(hist) > 1 else current_price
    day_change    = pct_change_str(current_price, prev_price)

    print(f"\n{Fore.WHITE}  {'─'*58}")
    print(f"  {Fore.YELLOW}{nse_ticker}{Style.RESET_ALL}   "
          f"Last: {Fore.WHITE}₹{current_price:.2f}{Style.RESET_ALL}   "
          f"Day: {day_change}   "
          f"Period: {period}")
    print(f"  {'─'*58}{Style.RESET_ALL}")

    # ASCII Chart
    ascii_chart(hist)

    # Fundamentals
    fund = get_fundamentals(stock)
    if fund:
        print_fundamentals(fund, nse_ticker)
    else:
        print(f"{Fore.YELLOW}  ⚠  Fundamental data unavailable.{Style.RESET_ALL}")

    # Technicals
    tech_summary, df = compute_technicals(hist)
    if tech_summary:
        print_technicals(tech_summary, nse_ticker, current_price)
    else:
        print(f"{Fore.YELLOW}  ⚠  Not enough history for technicals (need 20+ days).{Style.RESET_ALL}")

    # Overall Signal
    overall_signal(fund, tech_summary)

# ── Interactive search ────────────────────────────────────────────────────────

def interactive_search():
    print(f"\n{Fore.CYAN}╔{'═'*58}╗")
    print(f"║  {'NSE NIFTY 50 STOCK ANALYSER':^56}  ║")
    print(f"║  {'Fundamentals + Technicals + Deep Scanner':^56}  ║")
    print(f"╚{'═'*58}╝{Style.RESET_ALL}")

    while True:
        print(f"\n{Fore.WHITE}Options:")
        print(f"  {Fore.YELLOW}[1]{Style.RESET_ALL} Search stocks by name/keyword")
        print(f"  {Fore.YELLOW}[2]{Style.RESET_ALL} Analyse a specific ticker")
        print(f"  {Fore.YELLOW}[3]{Style.RESET_ALL} View Nifty 50 stocks")
        print(f"  {Fore.YELLOW}[4]{Style.RESET_ALL} 🔬 Deep Scanner (40%+ drawdown + EMA10 uptrend)")
        print(f"  {Fore.YELLOW}[q]{Style.RESET_ALL} Quit")

        choice = input(f"\n{Fore.CYAN}  ▶  Enter choice: {Style.RESET_ALL}").strip().lower()

        if choice == "q":
            print(f"\n{Fore.GREEN}  👋  Goodbye!{Style.RESET_ALL}\n")
            break

        elif choice == "1":
            query = input(f"  Search keyword: ").strip()
            results = search_stocks(query)
            if not results:
                print(f"{Fore.RED}  No stocks found for '{query}'.{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}  Results:{Style.RESET_ALL}")
                rows = [(t, n) for t, n in results]
                print(tabulate(rows, headers=["Ticker", "Company"], tablefmt="rounded_outline"))

                if len(results) == 1:
                    pick = results[0][0]
                elif len(results) > 1:
                    pick = input(f"\n  Enter ticker to analyse (or Enter to skip): ").strip().upper()
                else:
                    continue
                if pick:
                    period = input(f"  Period (1mo/3mo/6mo/1y/2y/5y) [default 1y]: ").strip() or "1y"
                    analyse(pick, period)

        elif choice == "2":
            ticker = input(f"  Enter NSE ticker (e.g. RELIANCE, TCS, INFY): ").strip().upper()
            if ticker:
                period = input(f"  Period (1mo/3mo/6mo/1y/2y/5y) [default 1y]: ").strip() or "1y"
                analyse(ticker, period)

        elif choice == "3":
            rows = [(t, n) for t, n in list(NIFTY50_STOCKS.items())]
            print(f"\n{Fore.YELLOW}  Nifty 50 Stocks:{Style.RESET_ALL}")
            print(tabulate(rows, headers=["Ticker", "Company"], tablefmt="rounded_outline"))


        elif choice == "4":
            thr_str = input(f"  Drawdown threshold % [default 40]: ").strip()
            try:
                thr = float(thr_str) if thr_str else 40.0
            except ValueError:
                print(f"{Fore.YELLOW}  Invalid threshold, using default 40%{Style.RESET_ALL}")
                thr = 40.0
            period = input(f"  History period (1y/2y) [default 1y]: ").strip() or "1y"
            run_deep_scanner(
                NIFTY50_STOCKS, fetch_stock, "₹",
                period=period, threshold=thr,
                exchange_name="NSE NIFTY 50"
            )

        else:
            print(f"{Fore.RED}  Invalid choice.{Style.RESET_ALL}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Nifty 50 Stock Analyser — Fundamentals & Technicals")
    parser.add_argument("--ticker", "-t", help="NSE ticker symbol (e.g. RELIANCE, TCS)")
    parser.add_argument("--search", "-s", help="Search stocks by keyword")
    parser.add_argument("--scan", action="store_true", help="Run deep scanner on all Nifty50 stocks")
    parser.add_argument("--threshold", default=40.0, type=float, help="Drawdown threshold %% (default 40)")
    parser.add_argument("--period", "-p", default="1y",
                        help="History period: 1mo/3mo/6mo/1y/2y/5y (default: 1y)")
    parser.add_argument("--dashboard", action="store_true", help="Open QuantDesk dashboard in browser")
    args = parser.parse_args()

    if args.dashboard:
        dashboard_path = os.path.join(os.path.dirname(__file__), 'quantdesk_dashboard.html')
        if os.path.exists(dashboard_path):
            webbrowser.open(f'file://{os.path.abspath(dashboard_path)}')
            print(f"{Fore.GREEN}✓  Dashboard opened in browser{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✘  Dashboard file not found: {dashboard_path}{Style.RESET_ALL}")
    elif args.scan:
        run_deep_scanner(NIFTY50_STOCKS, fetch_stock, "₹",
                         period=args.period, threshold=args.threshold,
                         exchange_name="NSE NIFTY 50")
    elif args.ticker:
        analyse(args.ticker, args.period)
    elif args.search:
        results = search_stocks(args.search)
        if results:
            print(tabulate(results, headers=["Ticker","Company"], tablefmt="rounded_outline"))
        else:
            print(f"No results for '{args.search}'")
    else:
        interactive_search()