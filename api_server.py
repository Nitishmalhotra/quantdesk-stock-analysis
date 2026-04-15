"""
QuantDesk API Server
====================
Lightweight HTTP server bridge between Python analyzers and HTML dashboard.

Provides REST API endpoints for live stock data:
- GET /api/stock/CBA?period=1y&market=ASX
- GET /api/scan?threshold=40&market=ASX
- GET /health

Usage:
    python api_server.py --port 8000
    Then open dashboard and point to: http://localhost:8000/
"""

import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sys
import os
import traceback
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Lazy loading - only import when needed
_MODULES_LOADED = False
fetch_stock = None
quant_score = None
game_theory_score = None
ASX_STOCKS = {}
fetch_stock_nse = None
quant_score_nse = None
game_theory_score_nse = None
NIFTY50_STOCKS = {}

def load_modules():
    global _MODULES_LOADED, fetch_stock, quant_score, game_theory_score, ASX_STOCKS
    global fetch_stock_nse, quant_score_nse, game_theory_score_nse, NIFTY50_STOCKS
    
    if _MODULES_LOADED:
        return True
    
    try:
        from AXS import fetch_stock as fs, quant_score as qs, game_theory_score as gs, ASX_STOCKS as astocks
        from NSE_NIFTY50 import fetch_stock as fsn, quant_score as qsn, game_theory_score as gsn, NIFTY50_STOCKS as nstocks
        
        fetch_stock = fs
        quant_score = qs
        game_theory_score = gs
        ASX_STOCKS = astocks
        fetch_stock_nse = fsn
        quant_score_nse = qsn
        game_theory_score_nse = gsn
        NIFTY50_STOCKS = nstocks
        _MODULES_LOADED = True
        return True
    except Exception as e:
        print(f"Error loading modules: {e}")
        traceback.print_exc()
        return False


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for stock analysis API."""

    def do_GET(self):
        """Handle GET requests."""
        # Ensure modules are loaded
        if not load_modules():
            self.send_error(500, "Failed to load analyzer modules")
            return
            
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        try:
            # Route requests
            response = None
            
            if path == "/health":
                response = {"status": "ok", "service": "quantdesk-api"}

            elif path == "/api/stock":
                ticker = query.get("ticker", [""])[0].upper()
                period = query.get("period", ["1y"])[0]
                market = query.get("market", ["ASX"])[0].upper()

                if not ticker:
                    response = {"error": "Missing 'ticker' parameter", "status": "error"}
                else:
                    response = self._analyze_stock(ticker, period, market)

            elif path == "/api/scan":
                threshold = float(query.get("threshold", ["40"])[0])
                period = query.get("period", ["1y"])[0]
                market = query.get("market", ["ASX"])[0].upper()
                response = self._scan_stocks(threshold, period, market)

            elif path == "/api/stocks/list":
                market = query.get("market", ["ASX"])[0].upper()
                response = self._list_stocks(market)

            else:
                self.send_error(404, "Endpoint not found")
                return

            # Send successful response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            print(f"API Error: {e}")
            traceback.print_exc()
            self.send_error(500, f"Internal server error: {str(e)}")

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def _transform_quant_for_dashboard(self, q, hist):
        """Transform Python quant dict to JavaScript-friendly format."""
        if not q:
            return None
        
        # Extract raw values
        closes = hist["Close"].values.tolist()
        w52h = float(hist["Close"].max())
        current = float(hist["Close"].iloc[-1])
        dd = (current - w52h) / w52h * 100
        
        # Get EMA-10 trend
        ema10 = hist["Close"].ewm(span=10, adjust=False).mean().values
        uptrend = bool(ema10[-1] > ema10[-6] if len(ema10) >= 6 else False)
        
        return {
            "momScore": float(q.get("Momentum Score", 0)),
            "mrScore": float(q.get("Mean-Reversion Score", 0)),
            "volScore": float(q.get("Volatility Score", 0)),
            "vsScore": float(q.get("Volume Surge Score", 0)),
            "rsiScore": float(q.get("RSI Score", 0)),
            "trendScore": float(q.get("Trend Strength Score", 0)),
            "composite": float(q.get("── Quant Composite ──", 0)),
            "rsi": float(q.get("_rsi", 50)),
            "rv30": float(q.get("_rv30", 30)),
            "vr": float(q.get("_vol_ratio", 1.0)),
            "ret21": float(q.get("_ret_21", 0)),
            "z": float(q.get("_z", 0)),
            "ema20": float(q.get("EMA 20", current)),
            "ema50": float(q.get("EMA 50", current)),
            "ema100": float(q.get("EMA 100", current)),
            "ema200": float(q.get("EMA 200", current)),
            "vwap": float(q.get("VWAP", current)),
            "w52h": float(w52h),
            "dd": float(dd),
            "uptrend": 1 if uptrend else 0  # Convert bool to int for JSON
        }

    def _transform_gt_for_dashboard(self, g):
        """Transform Python game theory dict to JavaScript-friendly format."""
        if not g:
            return None
        
        # Extract strength percentage
        strength_str = str(g.get("NE — Equilibrium Strength", "0%")).replace("%", "").strip()
        strength = float(strength_str) if strength_str else 0
        
        # Extract divergence points
        div_str = str(g.get("Inst vs Retail Divergence", "0 pts")).split()[0]
        div = float(div_str) if div_str.isdigit() else 0
        
        return {
            "instSig": str(g.get("NE — Inst. Signal", "HOLD")),
            "retailSig": str(g.get("NE — Retail Signal", "HOLD")),
            "mmSig": str(g.get("NE — Market Maker", "HOLD")),
            "dominant": str(g.get("NE — Dominant Strategy", "HOLD")),
            "strength": float(strength),
            "neScore": float(g.get("NE Score", 50)),
            "R": float(g.get("PD — Payoff R (Reward)", 1.0)),
            "T": float(g.get("PD — Payoff T (Tempt.)", 1.0)),
            "S": float(g.get("PD — Payoff S (Sucker)", 1.0)),
            "P": float(g.get("PD — Payoff P (Punish)", 1.0)),
            "pdLabel": str(g.get("PD — Rational Choice", "HOLD")),
            "pdScore": float(g.get("PD Score", 50)),
            "div": float(div),
            "divScore": float(g.get("Divergence Score", 50)),
            "gtComp": float(g.get("── GT Composite ──", 50))
        }

    def _analyze_stock(self, ticker, period, market):
        """Fetch and analyze a single stock."""
        try:
            if market == "ASX":
                stock, asx_ticker = fetch_stock(ticker)
                hist = stock.history(period=period)
                if hist is None or hist.empty:
                    return {"error": f"No data for {asx_ticker}", "status": "error"}
                
                q = quant_score(hist)
                g = game_theory_score(hist, q, {})
                
                # Extract key metrics
                current_price = float(hist["Close"].iloc[-1])
                day_change = float(hist["Close"].pct_change().iloc[-1] * 100)
                
                # Transform to dashboard format
                q_transformed = self._transform_quant_for_dashboard(q, hist)
                g_transformed = self._transform_gt_for_dashboard(g)
                
                return {
                    "status": "success",
                    "ticker": asx_ticker,
                    "market": "ASX",
                    "price": round(current_price, 2),
                    "day_change": round(day_change, 2),
                    "period": period,
                    "quant_score": q_transformed,
                    "game_theory": g_transformed,
                    "history": [float(x) for x in hist["Close"].tail(30).tolist()]
                }
            
            elif market == "NSE":
                stock, nse_ticker = fetch_stock_nse(ticker)
                hist = stock.history(period=period)
                if hist is None or hist.empty:
                    return {"error": f"No data for {nse_ticker}", "status": "error"}
                
                q = quant_score_nse(hist)
                g = game_theory_score_nse(hist, q, {})
                
                current_price = float(hist["Close"].iloc[-1])
                day_change = float(hist["Close"].pct_change().iloc[-1] * 100)
                
                # Transform to dashboard format
                q_transformed = self._transform_quant_for_dashboard(q, hist)
                g_transformed = self._transform_gt_for_dashboard(g)
                
                return {
                    "status": "success",
                    "ticker": nse_ticker,
                    "market": "NSE",
                    "price": round(current_price, 2),
                    "day_change": round(day_change, 2),
                    "period": period,
                    "quant_score": q_transformed,
                    "game_theory": g_transformed,
                    "history": [float(x) for x in hist["Close"].tail(30).tolist()]
                }
        
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            traceback.print_exc()
            return {"error": str(e), "status": "error"}

    def _scan_stocks(self, threshold, period, market):
        """Scan for beaten-down stocks with uptrend signals."""
        try:
            if market == "ASX":
                stocks_dict = ASX_STOCKS
                fetch_fn = fetch_stock
                quant_fn = quant_score
            else:
                stocks_dict = NIFTY50_STOCKS
                fetch_fn = fetch_stock_nse
                quant_fn = quant_score_nse
            
            results = []
            for ticker in list(stocks_dict.keys())[:10]:  # Limit to first 10 for speed
                try:
                    stock, full_ticker = fetch_fn(ticker)
                    hist = stock.history(period=period)
                    if hist is None or hist.empty:
                        continue
                    
                    current = float(hist["Close"].iloc[-1])
                    high_52w = float(hist["Close"].max())
                    drawdown = (current - high_52w) / high_52w * 100
                    
                    if drawdown <= -threshold:
                        q = quant_fn(hist)
                        if q:
                            results.append({
                                "ticker": full_ticker,
                                "company": stocks_dict[ticker],
                                "price": round(current, 2),
                                "drawdown": round(drawdown, 1),
                                "quant_composite": q.get("── Quant Composite ──", 0)
                            })
                except Exception:
                    continue
            
            return {
                "status": "success",
                "market": market,
                "threshold": threshold,
                "results": sorted(results, key=lambda x: x["quant_composite"], reverse=True),
                "count": len(results)
            }
        
        except Exception as e:
            print(f"Scan error: {e}")
            traceback.print_exc()
            return {"error": str(e), "status": "error"}

    def _list_stocks(self, market):
        """List all available stocks for a market."""
        if market == "ASX":
            stocks = ASX_STOCKS
        else:
            stocks = NIFTY50_STOCKS
        
        return {
            "status": "success",
            "market": market,
            "stocks": [{"ticker": k, "company": v} for k, v in stocks.items()],
            "count": len(stocks)
        }


def run_server(port=8000):
    """Start the API server."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, APIHandler)
    
    print(f"\n{'='*60}")
    print(f"  QuantDesk API Server Running")
    print(f"  http://localhost:{port}")
    print(f"{'='*60}")
    print(f"\n  Endpoints:")
    print(f"  • GET /health")
    print(f"  • GET /api/stock?ticker=CBA&period=1y&market=ASX")
    print(f"  • GET /api/scan?threshold=40&market=ASX")
    print(f"  • GET /api/stocks/list?market=ASX")
    print(f"\n  Press Ctrl+C to stop.\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantDesk API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on (default: 8000)")
    args = parser.parse_args()
    
    run_server(args.port)
