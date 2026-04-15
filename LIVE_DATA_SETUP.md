# QuantDesk Dashboard — Live Data Integration

## ✅ Enable Live Data (Real-Time Stock Analysis)

By default, the dashboard shows **demo data** with hardcoded values.

To connect it to **real data from your Python analyzers**, follow these steps:

### Step 1: Start the API Server

Open a terminal in the `StocksCheck` folder and run:

```bash
python api_server.py
```

You should see:
```
============================================================
  QuantDesk API Server Running
  http://localhost:8000
============================================================

  Endpoints:
  • GET /health
  • GET /api/stock?ticker=CBA&period=1y&market=ASX
  • GET /api/scan?threshold=40&market=ASX
  • GET /api/stocks/list?market=ASX

  Press Ctrl+C to stop.
```

**Optional:** Change the port:
```bash
python api_server.py --port 3000
```

Then update the HTML (find `API_BASE = 'http://localhost:8000'` and change `8000` to `3000`).

### Step 2: Open the Dashboard

You have two options:

**Option A: Local file (if API is running locally)**
```bash
# Run from AXS.py
python AXS.py --dashboard

# Or run from NSE_NIFTY50.py
python NSE_NIFTY50.py --dashboard
```

**Option B: Web server (recommended)**
```bash
# In StocksCheck folder, serve the HTML:
python -m http.server 8080

# Then visit: http://localhost:8080/quantdesk_dashboard.html
```

### Step 3: Try a Stock Lookup

1. Leave the API server running (Terminal 1)
2. Open the dashboard in browser (Terminal 2)
3. Type a ticker (e.g., `CBA` for ASX or `RELIANCE` for NSE)
4. Click **Analyse**
5. You should see **real data** instead of demo data!

---

## 🔌 API Endpoints Reference

### Get Stock Analysis
```bash
# ASX stock
curl "http://localhost:8000/api/stock?ticker=CBA&period=1y&market=ASX"

# NSE stock
curl "http://localhost:8000/api/stock?ticker=RELIANCE&period=1y&market=NSE"

# Responses include:
# - Real-time price & daily change
# - Quant scores (6 dimensions + composite)
# - Game theory analysis (Nash equilibrium, PD scores, divergence)
# - 30-day price history
```

### Run Deep Scanner
```bash
# Find ASX stocks down 40%+ from 52-week high
curl "http://localhost:8000/api/scan?threshold=40&market=ASX"

# Find NSE stocks down 50%+
curl "http://localhost:8000/api/scan?threshold=50&market=NSE"

# Returns: List of beaten-down stocks sorted by quant composite score
```

### List Available Stocks
```bash
# Get all ASX stocks
curl "http://localhost:8000/api/stocks/list?market=ASX"

# Get all NSE stocks
curl "http://localhost:8000/api/stocks/list?market=NSE"

# Returns: Ticker + company name for all supported stocks
```

### Health Check
```bash
curl "http://localhost:8000/health"

# Returns: {"status": "ok", "service": "quantdesk-api"}
```

---

## 🐛 Troubleshooting

### "API unavailable"
- Ensure `api_server.py` is running in Terminal 1
- Check that you're accessing from the same machine (localhost:8000)
- Try visiting `http://localhost:8000/health` in your browser

### "No data for ticker"
- Ticker may be misspelled (try `CBA` not `CBA.AX`)
- Stock may not exist on Yahoo Finance
- Try a different stock first

### "CORS error" in browser console
- This is expected for cross-domain requests
- API server already includes CORS headers
- If still an issue, use Python to serve the HTML instead

### "Port 8000 is already in use"
```bash
# Use a different port:
python api_server.py --port 8001

# Update HTML: API_BASE = 'http://localhost:8001'
```

---

## 📊 Data Flow

```
┌─────────────────────┐
│  quantdesk.html     │ ← Browser (JavaScript)
└──────────┬──────────┘
           │ fetch() API calls
           ↓
┌─────────────────────┐
│  api_server.py      │ ← Python HTTP Server (port 8000)
└──────────┬──────────┘
           │ imports & calls
           ↓
┌─────────────────────┐
│  AXS.py + NSE.py    │ ← Real Analysis Logic
└──────────┬──────────┘
           │ calls
           ↓
┌─────────────────────┐
│  yfinance API       │ ← Yahoo Finance (Live Data)
└─────────────────────┘
```

---

## 🚀 Example: Live Analysis of CBA (Commonwealth Bank)

### Terminal 1 (API Server):
```bash
$ python api_server.py
============================================================
  QuantDesk API Server Running
  http://localhost:8000
============================================================
API request: GET /api/stock?ticker=CBA&period=1y&market=ASX
Analysed CBA.AX in 2.3s
```

### Terminal 2 (HTML/Browser):
```
Browser console:
Using real data from API: CBA
Rendered dashboard with:
- Current price: A$183.52
- Quant Composite: 58.3/100
- Game Theory Composite: 62.1/100
- Verdict: BUY
```

---

## 💡 Tips

1. **Keep the API server running** while using the dashboard
2. **Use the same machine** (localhost) for API + dashboard access
3. **Try multiple stocks** to see how scores vary
4. **Check the terminal** for API logs (helpful for debugging)
5. **Share the dashboard link** — only the HTML loads across networks, but the API must be local

---

**Happy analyzing! 📈**
