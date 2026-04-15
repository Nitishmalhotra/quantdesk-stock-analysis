# QuantDesk — Stock Analysis Dashboard

A professional-grade stock analyzer for **ASX (Australian)** and **NSE (Indian)** markets with real-time technical analysis, fundamental metrics, and AI-powered trading signals.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-Personal%20Use-orange)

---

## 🎯 Quick Start

### For Non-Technical Users (Web Dashboard)
1. **Open the dashboard**: Click the link shared with you → Opens in browser instantly
2. **Search stocks**: Type ticker symbol (e.g., `CBA`, `RELIANCE`)
3. **View analysis**: See technical charts, valuations, and AI trading signals
4. **No software needed** — just an internet browser!

### For Technical Users (Run Locally)

#### Prerequisites
```bash
python --version  # Ensure Python 3.8+
```

#### Installation (One-time Setup)
```bash
# Clone/download the project
cd stock-dashboard

# Install dependencies
pip install yfinance pandas numpy requests tabulate colorama ta
```

#### Usage

**ASX Stock Analyzer:**
```bash
# Analyze a specific stock
python AXS.py --ticker CBA --period 6mo

# Search by keyword
python AXS.py --search bank

# Run deep scanner (find beaten-down stocks with uptrend)
python AXS.py --scan --threshold 40

# Open interactive dashboard
python AXS.py --dashboard

# Interactive menu
python AXS.py
```

**NSE Stock Analyzer:**
```bash
# Analyze a specific stock
python NSE_NIFTY50.py --ticker RELIANCE --period 6mo

# Search by keyword
python NSE_NIFTY50.py --search energy

# Run deep scanner
python NSE_NIFTY50.py --scan

# Open dashboard
python NSE_NIFTY50.py --dashboard
```

---

## 📊 Features

### Dashboard (Web Interface)
✅ **Real-time stock data** — Live prices, volumes, market cap  
✅ **Interactive charts** — Price movement, MACD, RSI, volume  
✅ **Fundamentals** — P/E ratio, dividends, earnings, debt levels  
✅ **Technicals** — Moving averages, Bollinger bands, ATR, VWAP  
✅ **AI Trading Signals** — BUY/SELL/HOLD recommendations  
✅ **Multi-market** — Switch between ASX (🇦🇺) and NSE (🇮🇳) instantly  
✅ **Deep Scanner** — Find stocks down 40%+ with recovery signals  
✅ **Game Theory Analysis** — Institutional vs retail divergence signals  

### Command-Line Interface
✅ **Fast analysis** — Terminal-based for power users  
✅ **Color-coded output** — Easy to read, color-highlighted signals  
✅ **Batch processing** — Scan 50+ stocks at once  
✅ **Custom thresholds** — Adjust scan parameters (drawdown %, period)  
✅ **Search function** — Find stocks by company name or sector  

---

## 🔬 Technical Methodology

### 6-Dimensional Scoring System
Each stock is rated 0-100 on six independent dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Momentum** | 20% | 5-day, 21-day, 63-day returns |
| **Mean-Reversion** | 25% | Z-score from 20d moving average |
| **Volatility** | 10% | 30-day realized volatility (annualized) |
| **Volume Surge** | 15% | 5d/20d average volume ratio |
| **RSI Score** | 15% | Relative Strength Index (14-period) |
| **Trend** | 15% | EMA-10 slope over 5 days |

**Interpretation:**
- **60-100**: Bullish signals (buy opportunity)
- **40-60**: Neutral (hold/gather more data)
- **0-40**: Bearish signals (sell/avoid)

### Game Theory Analysis
Models market behavior as a 3-player game:
- **Institutional traders** (smart money, large positions)
- **Retail traders** (momentum chasers, social traders)
- **Market makers** (spreads, order flow)

Detects **Nash Equilibrium** (when all players agree) and **Divergence** (conflicting signals = volatility opportunity).

### Indicators

**Moving Averages:**
- EMA-20 (short-term entry/exit)
- EMA-50 (weekly support/resistance)
- EMA-100 (monthly trend)
- EMA-200 (yearly uptrend — classic "above 200 EMA = bullish")

**Momentum:**
- RSI (oversold/overbought)
- MACD (trend direction)
- Stochastic (reversal signals)

**Volatility:**
- Bollinger Bands
- ATR (Average True Range)
- VWAP (institutional entry price)

---

## 📥 Input Parameters

### Period
How much historical data to analyze:
- `1mo` — 1 month
- `3mo` — 3 months
- `6mo` — 6 months
- `1y` — 1 year (default)
- `2y` — 2 years
- `5y` — 5 years

### Threshold (% Drawdown)
For scanner mode — find stocks down X% from 52-week high:
```bash
python AXS.py --scan --threshold 40    # Find stocks down 40%+
python AXS.py --scan --threshold 50    # More aggressive (down 50%+)
```

---

## 📈 Example Analysis Output

### Terminal (ASX.py)
```
  ──────────────────────────────────────────────────────────
  CBA.AX   Last: A$183.52   Day: +0.17%   Period: 6mo
  ──────────────────────────────────────────────────────────

  📊  FUNDAMENTALS
  Company       Commonwealth Bank of Australia
  Current Price A$183.52
  52W High      A$192.00
  P/E Ratio     29.50
  Market Cap    $306.87B

  📉  TECHNICALS
  RSI (14)      68.09        🔴 Overbought
  MACD          Bullish      🟢 Uptrend
  VWAP          A$164.30     🟢 Price above VWAP
  EMA 200       A$164.05     🟢 Above 200 EMA

  🎯  OVERALL SIGNAL SUMMARY
  Verdict: ▶ HOLD ◀ (2 bullish / 2 bearish signals)
```

### Dashboard (Web)
- Interactive charts (drag to zoom)
- Color-coded verdict badges
- Side-by-side ASX/NSE comparison
- Real-time data updates

---

## 🚀 Getting Help

### Common Issues

**"No data found for ticker"**
- Ensure ticker is correct (e.g., `CBA` not `CBA.AX`)
- Try alternative spelling
- Check if company still trades

**"Insufficient data for quant scoring"**
- Some stocks have <30 trading days of history
- Try longer period (e.g., `--period 1y` instead of `--period 1mo`)

**"Dashboard won't open"**
- Ensure `quantdesk_dashboard.html` is in the same folder
- Try opening HTML file directly in browser
- Check browser console (F12) for errors

---

## 📚 Resources

**Data Source:** [Yahoo Finance](https://finance.yahoo.com) (via yfinance)  
**Technical Indicators:** [TA-Lib](https://github.com/bukosabino/ta)  
**Charts:** [Chart.js](https://www.chartjs.org/)  

---

## ⚠️ Disclaimer

**This is NOT financial advice.** Past performance does not predict future results.

- Use this tool as **one input** among many
- Always do your own research
- Consider consulting a financial advisor before trading
- Trading carries risk of loss
- Only invest money you can afford to lose

---

## 🛠️ Technical Details

### Architecture
- **Frontend**: HTML5 + CSS3 + JavaScript (Chart.js)
- **Backend**: Python 3.8+ (yfinance, pandas, numpy)
- **Data**: Real-time Yahoo Finance API
- **Deployment**: GitHub Pages (web) + Local Python (CLI)

### Files
```
stock-dashboard/
├── AXS.py                       # ASX (Australian) analyzer
├── NSE_NIFTY50.py               # NSE (Indian) analyzer
├── quantdesk_dashboard.html     # Web dashboard
└── README.md                    # This file
```

### Requirements
```
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.23.0
ta>=0.9.0
tabulate>=0.9.0
colorama>=0.4.5
```

---

## 💡 Tips for Best Results

1. **Use longer periods** (6mo-1y) for more reliable signals
2. **Combine multiple indicators** — no single indicator is foolproof
3. **Check fundamentals + technicals** — don't rely on just one
4. **Act on divergence** — when institutional & retail signals disagree = volatility opportunity
5. **Wait for EMA confirmations** — when price crosses EMA-20/50, that's a strong signal

---

## 🔄 Updates & Improvements

Planned features:
- [ ] API endpoint for live data integration
- [ ] Mobile-responsive dashboard
- [ ] Alert notifications (Discord, email)
- [ ] Backtest historical performance
- [ ] Export reports to PDF
- [ ] Multi-timeframe analysis

---

## 📧 Contact & Questions

For issues, suggestions, or questions:
1. Check this README first
2. Review the code comments (heavily documented)
3. Try running with `--help` flag

---

**Happy investing! 📈**

*Made with ❤️ for stock analysis*
