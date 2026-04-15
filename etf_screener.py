"""
Enhanced iShares ETF Swing Trading Screener
Analyzes major ETFs with comprehensive technical & fundamental indicators
Version 2.0 - Enhanced Edition
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Try importing technical analysis library
try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False
    print("Warning: ta library not installed. Install with: pip install ta")


class EnhancedETFScreener:
    """Enhanced ETF Swing Trading Screener with Advanced Analytics"""
    
    def __init__(self, etf_list, period='6mo', interval='1d', benchmark='SPY'):
        self.etf_list = etf_list
        self.period = period
        self.interval = interval
        self.benchmark = benchmark
        self.results = []
        self.benchmark_data = None
        
    # ==================== BASIC INDICATORS ====================
    
    def calculate_sma(self, data, window):
        """Calculate Simple Moving Average"""
        try:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            
            if TA_LIB_AVAILABLE:
                return SMAIndicator(data, window=window).sma_indicator()
            else:
                return data.rolling(window=window).mean()
        except:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            return data.rolling(window=window).mean()
    
    def calculate_ema(self, data, window):
        """Calculate Exponential Moving Average"""
        try:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            
            if TA_LIB_AVAILABLE:
                return EMAIndicator(data, window=window).ema_indicator()
            else:
                return data.ewm(span=window, adjust=False).mean()
        except:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            return data.ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, data, window=14):
        """Calculate Relative Strength Index"""
        try:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            
            if TA_LIB_AVAILABLE:
                return RSIIndicator(data, window=window).rsi()
            else:
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
        except:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        try:
            if TA_LIB_AVAILABLE:
                close_data = pd.Series(data) if not isinstance(data, pd.Series) else data
                macd_obj = MACD(close_data, window_fast=fast, window_slow=slow, window_sign=signal)
                return macd_obj.macd(), macd_obj.macd_signal(), macd_obj.macd_diff()
            else:
                ema_fast = data.ewm(span=fast, adjust=False).mean()
                ema_slow = data.ewm(span=slow, adjust=False).mean()
                macd = ema_fast - ema_slow
                macd_signal = macd.ewm(span=signal, adjust=False).mean()
                macd_hist = macd - macd_signal
                return macd, macd_signal, macd_hist
        except:
            ema_fast = data.ewm(span=fast, adjust=False).mean()
            ema_slow = data.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist
    
    def calculate_adx(self, high, low, close, window=14):
        """Calculate Average Directional Index"""
        try:
            if TA_LIB_AVAILABLE:
                high_data = pd.Series(high) if not isinstance(high, pd.Series) else high
                low_data = pd.Series(low) if not isinstance(low, pd.Series) else low
                close_data = pd.Series(close) if not isinstance(close, pd.Series) else close
                adx_obj = ADXIndicator(high_data, low_data, close_data, window=window)
                return adx_obj.adx()
            else:
                tr = np.maximum(
                    np.maximum(high - low, np.abs(high - close.shift(1))),
                    np.abs(low - close.shift(1))
                )
                atr = tr.rolling(window=window).mean()
                
                plus_dm = np.where(high - high.shift(1) > low.shift(1) - low, high - high.shift(1), 0)
                minus_dm = np.where(low.shift(1) - low > high - high.shift(1), low.shift(1) - low, 0)
                
                plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
                minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)
                
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = dx.rolling(window=window).mean()
                
                return adx
        except:
            tr = np.maximum(
                np.maximum(high - low, np.abs(high - close.shift(1))),
                np.abs(low - close.shift(1))
            )
            atr = tr.rolling(window=window).mean()
            
            plus_dm = np.where(high - high.shift(1) > low.shift(1) - low, high - high.shift(1), 0)
            minus_dm = np.where(low.shift(1) - low > high - high.shift(1), low.shift(1) - low, 0)
            
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=window).mean()
            
            return adx
    
    # ==================== ENHANCED INDICATORS ====================
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        try:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            
            if TA_LIB_AVAILABLE:
                bb = BollingerBands(data, window=window, window_dev=num_std)
                return bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
            else:
                sma = data.rolling(window=window).mean()
                std = data.rolling(window=window).std()
                upper = sma + (std * num_std)
                lower = sma - (std * num_std)
                return sma, upper, lower
        except:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            sma = data.rolling(window=window).mean()
            std = data.rolling(window=window).std()
            upper = sma + (std * num_std)
            lower = sma - (std * num_std)
            return sma, upper, lower
    
    def calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range"""
        try:
            if TA_LIB_AVAILABLE:
                high_data = pd.Series(high) if not isinstance(high, pd.Series) else high
                low_data = pd.Series(low) if not isinstance(low, pd.Series) else low
                close_data = pd.Series(close) if not isinstance(close, pd.Series) else close
                return AverageTrueRange(high_data, low_data, close_data, window=window).average_true_range()
            else:
                tr1 = high - low
                tr2 = np.abs(high - close.shift(1))
                tr3 = np.abs(low - close.shift(1))
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                return tr.rolling(window=window).mean()
        except:
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            return tr.rolling(window=window).mean()
    
    def calculate_stochastic(self, high, low, close, window=14):
        """Calculate Stochastic Oscillator"""
        try:
            if TA_LIB_AVAILABLE:
                high_data = pd.Series(high) if not isinstance(high, pd.Series) else high
                low_data = pd.Series(low) if not isinstance(low, pd.Series) else low
                close_data = pd.Series(close) if not isinstance(close, pd.Series) else close
                stoch = StochasticOscillator(high_data, low_data, close_data, window=window)
                return stoch.stoch(), stoch.stoch_signal()
            else:
                lowest_low = low.rolling(window=window).min()
                highest_high = high.rolling(window=window).max()
                k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
                d = k.rolling(window=3).mean()
                return k, d
        except:
            lowest_low = low.rolling(window=window).min()
            highest_high = high.rolling(window=window).max()
            k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d = k.rolling(window=3).mean()
            return k, d
    
    def calculate_obv(self, close, volume):
        """Calculate On-Balance Volume"""
        try:
            if not isinstance(close, pd.Series):
                close = pd.Series(close)
            if not isinstance(volume, pd.Series):
                volume = pd.Series(volume)
            
            if TA_LIB_AVAILABLE:
                return OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            else:
                obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
                return obv
        except:
            if not isinstance(close, pd.Series):
                close = pd.Series(close)
            if not isinstance(volume, pd.Series):
                volume = pd.Series(volume)
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            return obv
    
    # ==================== PATTERN DETECTION ====================
    
    def detect_price_patterns(self, data):
        """Detect price patterns"""
        patterns = {
            'higher_highs': False,
            'higher_lows': False,
            'lower_highs': False,
            'lower_lows': False,
            'support_level': None,
            'resistance_level': None
        }
        
        try:
            recent = data.tail(20)
            highs = recent['high'].tail(10)
            lows = recent['low'].tail(10)
            
            if len(highs) >= 3 and len(lows) >= 3:
                patterns['higher_highs'] = highs.iloc[-1] > highs.iloc[-3] > highs.iloc[-5]
                patterns['higher_lows'] = lows.iloc[-1] > lows.iloc[-3] > lows.iloc[-5]
                patterns['lower_highs'] = highs.iloc[-1] < highs.iloc[-3] < highs.iloc[-5]
                patterns['lower_lows'] = lows.iloc[-1] < lows.iloc[-3] < lows.iloc[-5]
            
            if len(data) >= 50:
                last_50 = data.tail(50)
                patterns['support_level'] = last_50['low'].min()
                patterns['resistance_level'] = last_50['high'].max()
        except:
            pass
        
        return patterns
    
    # ==================== VOLUME ANALYSIS ====================
    
    def analyze_volume(self, data):
        """Analyze volume patterns"""
        volume_metrics = {
            'avg_volume_20': 0,
            'volume_ratio': 1.0,
            'volume_trend': 'neutral',
            'volume_breakout': False
        }
        
        try:
            volume_metrics['avg_volume_20'] = data['volume'].tail(20).mean()
            current_vol = data['volume'].iloc[-1]
            avg_vol = volume_metrics['avg_volume_20']
            volume_metrics['volume_ratio'] = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            recent_vol = data['volume'].tail(10).mean()
            prev_vol = data['volume'].tail(20).head(10).mean()
            
            if recent_vol > prev_vol * 1.2:
                volume_metrics['volume_trend'] = 'increasing'
            elif recent_vol < prev_vol * 0.8:
                volume_metrics['volume_trend'] = 'decreasing'
            else:
                volume_metrics['volume_trend'] = 'stable'
            
            volume_metrics['volume_breakout'] = volume_metrics['volume_ratio'] > 2.0
        except:
            pass
        
        return volume_metrics
    
    # ==================== FUNDAMENTAL ANALYSIS ====================
    
    def get_etf_fundamentals(self, ticker):
        """Get ETF fundamental data"""
        fundamentals = {
            'name': 'N/A',
            'category': 'N/A',
            'aum': 'N/A',
            'expense_ratio': 'N/A',
            'yield': 'N/A',
            'pe_ratio': 'N/A',
            'beta': 'N/A',
            'ytd_return': 'N/A',
            '52w_high': 'N/A',
            '52w_low': 'N/A'
        }
        
        try:
            etf = yf.Ticker(ticker)
            info = etf.info
            
            fundamentals['name'] = info.get('longName', 'N/A')
            fundamentals['category'] = info.get('category', 'N/A')
            
            if 'totalAssets' in info and info['totalAssets']:
                aum = info['totalAssets'] / 1e9
                fundamentals['aum'] = f"${aum:.2f}B"
            
            if 'annualReportExpenseRatio' in info and info['annualReportExpenseRatio']:
                fundamentals['expense_ratio'] = f"{info['annualReportExpenseRatio']*100:.2f}%"
            
            if 'yield' in info and info['yield']:
                fundamentals['yield'] = f"{info['yield']*100:.2f}%"
            elif 'trailingAnnualDividendYield' in info and info['trailingAnnualDividendYield']:
                fundamentals['yield'] = f"{info['trailingAnnualDividendYield']*100:.2f}%"
            
            if 'trailingPE' in info and info['trailingPE']:
                fundamentals['pe_ratio'] = f"{info['trailingPE']:.2f}"
            
            if 'beta' in info and info['beta']:
                fundamentals['beta'] = f"{info['beta']:.2f}"
            
            if 'ytdReturn' in info and info['ytdReturn']:
                fundamentals['ytd_return'] = f"{info['ytdReturn']*100:.2f}%"
            
            if 'fiftyTwoWeekHigh' in info and info['fiftyTwoWeekHigh']:
                fundamentals['52w_high'] = f"${info['fiftyTwoWeekHigh']:.2f}"
            
            if 'fiftyTwoWeekLow' in info and info['fiftyTwoWeekLow']:
                fundamentals['52w_low'] = f"${info['fiftyTwoWeekLow']:.2f}"
        except:
            pass
        
        return fundamentals
    
    # ==================== RISK METRICS ====================
    
    def calculate_risk_metrics(self, data):
        """Calculate risk and performance metrics"""
        metrics = {
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'return_1m': 0,
            'return_3m': 0,
            'return_6m': 0
        }
        
        try:
            returns = data['close'].pct_change().dropna()
            metrics['volatility'] = returns.std() * np.sqrt(252) * 100
            
            risk_free_rate = 0.02 / 252
            excess_returns = returns - risk_free_rate
            if returns.std() != 0:
                metrics['sharpe_ratio'] = (excess_returns.mean() / returns.std()) * np.sqrt(252)
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min() * 100
            
            if len(data) >= 21:
                metrics['return_1m'] = ((data['close'].iloc[-1] / data['close'].iloc[-21]) - 1) * 100
            if len(data) >= 63:
                metrics['return_3m'] = ((data['close'].iloc[-1] / data['close'].iloc[-63]) - 1) * 100
            if len(data) >= 126:
                metrics['return_6m'] = ((data['close'].iloc[-1] / data['close'].iloc[-126]) - 1) * 100
        except:
            pass
        
        return metrics
    
    # ==================== RELATIVE STRENGTH ====================
    
    def calculate_relative_strength(self, data, benchmark_data):
        """Calculate relative strength vs benchmark"""
        try:
            if benchmark_data is None or len(benchmark_data) == 0:
                return 0
            
            if len(data) > len(benchmark_data):
                data = data.tail(len(benchmark_data))
            elif len(benchmark_data) > len(data):
                benchmark_data = benchmark_data.tail(len(data))
            
            etf_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
            benchmark_return = (benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[0] - 1) * 100
            
            return etf_return - benchmark_return
        except:
            return 0
    
    # ==================== SCORING SYSTEM ====================
    
    def calculate_composite_score(self, signals):
        """Calculate composite score based on multiple factors"""
        score = 0
        
        if signals.get('trend') == 'bullish':
            score += 10
        elif signals.get('trend') == 'bearish':
            score -= 10
        
        if signals.get('macd_signal') == 'bullish':
            score += 10
        elif signals.get('macd_signal') == 'bearish':
            score -= 10
        
        if signals.get('rsi_zone') == 'oversold':
            score += 5
        elif signals.get('rsi_zone') == 'overbought':
            score -= 5
        elif signals.get('rsi_zone') == 'bullish':
            score += 5
        
        if signals.get('stoch_signal') == 'oversold':
            score += 5
        elif signals.get('stoch_signal') == 'overbought':
            score -= 5
        
        if signals.get('adx_strength') == 'strong':
            score += 10
        elif signals.get('adx_strength') == 'weak':
            score -= 5
        
        if signals.get('price_pattern') in ['higher_highs_lows', 'breakout']:
            score += 10
        elif signals.get('price_pattern') == 'lower_highs_lows':
            score -= 10
        
        if signals.get('volume_confirmation'):
            score += 10
        
        if signals.get('volume_breakout'):
            score += 10
        
        rel_strength = signals.get('relative_strength', 0)
        if rel_strength > 5:
            score += 10
        elif rel_strength > 0:
            score += 5
        elif rel_strength < -5:
            score -= 10
        elif rel_strength < 0:
            score -= 5
        
        if signals.get('bollinger_position') == 'oversold':
            score += 5
        elif signals.get('bollinger_position') == 'overbought':
            score -= 5
        
        score = max(0, min(100, 50 + score))
        return score
    
    # ==================== MAIN SCAN FUNCTION ====================
    
    def scan(self):
        """Scan all ETFs for trading signals"""
        
        try:
            print(f"Downloading benchmark ({self.benchmark}) data...")
            self.benchmark_data = yf.download(
                self.benchmark,
                period=self.period,
                interval=self.interval,
                progress=False
            )
            if isinstance(self.benchmark_data.columns, pd.MultiIndex):
                self.benchmark_data.columns = self.benchmark_data.columns.get_level_values(0)
            self.benchmark_data.columns = [str(col).lower() for col in self.benchmark_data.columns]
        except:
            print("Warning: Could not download benchmark data")
            self.benchmark_data = None
        
        for ticker in self.etf_list:
            try:
                print(f"Scanning {ticker}...", end=' ')
                
                data = yf.download(
                    ticker,
                    period=self.period,
                    interval=self.interval,
                    progress=False
                )
                
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                data.columns = [str(col).lower() for col in data.columns]
                
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    print(f"❌ (Missing: {missing_cols})")
                    continue
                
                if len(data) < 50:
                    print(f"❌ (Insufficient data)")
                    continue
                
                data['SMA_20'] = self.calculate_sma(data['close'], 20)
                data['SMA_50'] = self.calculate_sma(data['close'], 50)
                data['SMA_200'] = self.calculate_sma(data['close'], 200)
                data['EMA_12'] = self.calculate_ema(data['close'], 12)
                data['EMA_26'] = self.calculate_ema(data['close'], 26)
                data['RSI_14'] = self.calculate_rsi(data['close'], 14)
                
                macd, macd_signal, macd_hist = self.calculate_macd(data['close'])
                data['MACD'] = macd
                data['MACD_Signal'] = macd_signal
                data['MACD_Hist'] = macd_hist
                
                data['ADX_14'] = self.calculate_adx(data['high'], data['low'], data['close'], 14)
                
                bb_mid, bb_upper, bb_lower = self.calculate_bollinger_bands(data['close'], 20, 2)
                data['BB_Mid'] = bb_mid
                data['BB_Upper'] = bb_upper
                data['BB_Lower'] = bb_lower
                
                data['ATR_14'] = self.calculate_atr(data['high'], data['low'], data['close'], 14)
                
                stoch_k, stoch_d = self.calculate_stochastic(data['high'], data['low'], data['close'], 14)
                data['Stoch_K'] = stoch_k
                data['Stoch_D'] = stoch_d
                
                data['OBV'] = self.calculate_obv(data['close'], data['volume'])
                
                latest = data.iloc[-1]
                prev = data.iloc[-2]
                
                if pd.isna(latest['SMA_20']) or pd.isna(latest['RSI_14']) or pd.isna(latest['MACD']):
                    print(f"❌ (Incomplete indicators)")
                    continue
                
                signals = {}
                
                signals['trend'] = 'bullish' if latest['SMA_20'] > latest['SMA_50'] else 'bearish'
                
                macd_cross = (latest['MACD'] > latest['MACD_Signal']) and (prev['MACD'] <= prev['MACD_Signal'])
                signals['macd_signal'] = 'bullish' if macd_cross else 'bearish' if latest['MACD'] < latest['MACD_Signal'] else 'neutral'
                
                if latest['RSI_14'] < 30:
                    signals['rsi_zone'] = 'oversold'
                elif latest['RSI_14'] > 70:
                    signals['rsi_zone'] = 'overbought'
                elif 50 < latest['RSI_14'] < 70:
                    signals['rsi_zone'] = 'bullish'
                elif 30 < latest['RSI_14'] < 50:
                    signals['rsi_zone'] = 'bearish'
                else:
                    signals['rsi_zone'] = 'neutral'
                
                if not pd.isna(latest['Stoch_K']):
                    if latest['Stoch_K'] < 20:
                        signals['stoch_signal'] = 'oversold'
                    elif latest['Stoch_K'] > 80:
                        signals['stoch_signal'] = 'overbought'
                    else:
                        signals['stoch_signal'] = 'neutral'
                else:
                    signals['stoch_signal'] = 'neutral'
                
                signals['adx_strength'] = 'strong' if latest['ADX_14'] > 25 else 'weak' if latest['ADX_14'] < 20 else 'moderate'
                
                bb_position = (latest['close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) * 100
                if bb_position < 20:
                    signals['bollinger_position'] = 'oversold'
                elif bb_position > 80:
                    signals['bollinger_position'] = 'overbought'
                else:
                    signals['bollinger_position'] = 'neutral'
                
                volume_metrics = self.analyze_volume(data)
                signals['volume_confirmation'] = volume_metrics['volume_ratio'] > 1.2
                signals['volume_breakout'] = volume_metrics['volume_breakout']
                
                patterns = self.detect_price_patterns(data)
                if patterns['higher_highs'] and patterns['higher_lows']:
                    signals['price_pattern'] = 'higher_highs_lows'
                elif patterns['lower_highs'] and patterns['lower_lows']:
                    signals['price_pattern'] = 'lower_highs_lows'
                else:
                    signals['price_pattern'] = 'consolidation'
                
                signals['relative_strength'] = self.calculate_relative_strength(data, self.benchmark_data)
                risk_metrics = self.calculate_risk_metrics(data)
                fundamentals = self.get_etf_fundamentals(ticker)
                composite_score = self.calculate_composite_score(signals)
                
                if composite_score >= 70:
                    final_signal = "STRONG BUY"
                    signal_strength = 5
                elif composite_score >= 60:
                    final_signal = "BUY"
                    signal_strength = 4
                elif composite_score >= 55:
                    final_signal = "WEAK BUY"
                    signal_strength = 3
                elif composite_score <= 30:
                    final_signal = "STRONG SELL"
                    signal_strength = -5
                elif composite_score <= 40:
                    final_signal = "SELL"
                    signal_strength = -4
                elif composite_score <= 45:
                    final_signal = "WEAK SELL"
                    signal_strength = -3
                else:
                    final_signal = "HOLD"
                    signal_strength = 0
                
                self.results.append({
                    'Ticker': ticker,
                    'Name': fundamentals['name'][:30] if len(fundamentals['name']) > 30 else fundamentals['name'],
                    'Price': f"${latest['close']:.2f}",
                    'Score': composite_score,
                    'Signal': final_signal,
                    'Strength': signal_strength,
                    'RSI': f"{latest['RSI_14']:.1f}",
                    'MACD': 'Bull' if signals['macd_signal'] == 'bullish' else 'Bear',
                    'ADX': f"{latest['ADX_14']:.1f}",
                    'Stoch': f"{latest['Stoch_K']:.1f}" if not pd.isna(latest['Stoch_K']) else 'N/A',
                    'Vol_Ratio': f"{volume_metrics['volume_ratio']:.2f}x",
                    'Rel_Str': f"{signals['relative_strength']:+.1f}%",
                    'Volatility': f"{risk_metrics['volatility']:.1f}%",
                    'Sharpe': f"{risk_metrics['sharpe_ratio']:.2f}",
                    'MaxDD': f"{risk_metrics['max_drawdown']:.1f}%",
                    '1M': f"{risk_metrics['return_1m']:+.1f}%",
                    '3M': f"{risk_metrics['return_3m']:+.1f}%",
                    '6M': f"{risk_metrics['return_6m']:+.1f}%",
                    'AUM': fundamentals['aum'],
                    'Expense': fundamentals['expense_ratio'],
                    'Yield': fundamentals['yield'],
                    'Beta': fundamentals['beta']
                })
                
                print(f"✓ Score: {composite_score}/100 | {final_signal}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                continue
    
    def display_results(self):
        """Display scanning results"""
        if not self.results:
            print("\n❌ No results found")
            return
        
        df = pd.DataFrame(self.results)
        df = df.sort_values(by='Score', ascending=False)
        
        print("\n" + "="*180)
        print("ENHANCED ETF SWING TRADING SCREENER - COMPREHENSIVE ANALYSIS")
        print(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Benchmark: {self.benchmark}")
        print("="*180)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        print("\n" + "-"*180)
        print("TOP RANKED ETFs:")
        print("-"*180)
        top_10 = df.head(10)[['Ticker', 'Name', 'Price', 'Score', 'Signal', 'RSI', 'MACD', 'Vol_Ratio', 
                               'Rel_Str', '1M', '3M', 'AUM', 'Yield']]
        print(top_10.to_string(index=False))
        
        print("\n" + "-"*180)
        print("FULL ANALYSIS:")
        print("-"*180)
        print(df.to_string(index=False))
        
        print("\n" + "="*180)
        print("SUMMARY:")
        print("="*180)
        strong_buy = len(df[df['Signal'] == 'STRONG BUY'])
        buy = len(df[df['Signal'] == 'BUY'])
        weak_buy = len(df[df['Signal'] == 'WEAK BUY'])
        hold = len(df[df['Signal'] == 'HOLD'])
        weak_sell = len(df[df['Signal'] == 'WEAK SELL'])
        sell = len(df[df['Signal'] == 'SELL'])
        strong_sell = len(df[df['Signal'] == 'STRONG SELL'])
        
        print(f"  🟢 STRONG BUY: {strong_buy}")
        print(f"  🟢 BUY: {buy}")
        print(f"  🟡 WEAK BUY: {weak_buy}")
        print(f"  ⚪ HOLD: {hold}")
        print(f"  🟡 WEAK SELL: {weak_sell}")
        print(f"  🔴 SELL: {sell}")
        print(f"  🔴 STRONG SELL: {strong_sell}")
        print(f"\n  Average Score: {df['Score'].mean():.1f}/100")
        print(f"  Best Performer: {df.iloc[0]['Ticker']} ({df.iloc[0]['Score']}/100)")
        print("="*180 + "\n")
    
    def save_results(self, filename=None):
        """Save results to CSV"""
        if not self.results:
            print("No results to save")
            return
        
        if filename is None:
            filename = f"enhanced_etf_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(self.results)
        df = df.sort_values(by='Score', ascending=False)
        df.to_csv(filename, index=False)
        print(f"✓ Results saved to: {filename}")


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # ETF List - iShares Country/Region ETFs (29 ETFs)
    etf_list = [
        "EWS",    # iShares MSCI Singapore ETF
        "EWZ",    # iShares MSCI Brazil ETF
        "EWL",    # iShares MSCI Switzerland ETF
        "EWZS",   # iShares MSCI Brazil Small-Cap ETF
        "EWT",    # iShares MSCI Taiwan ETF
        "EEM",    # iShares MSCI Emerging Markets ETF
        "EWW",    # iShares MSCI Mexico ETF
        "EWQ",    # iShares MSCI France ETF
        "EWH",    # iShares MSCI Hong Kong ETF
        "EWN",    # iShares MSCI Netherlands ETF
        "EWK",    # iShares MSCI Belgium ETF
        "EWG",    # iShares MSCI Germany ETF
        "ECH",    # iShares MSCI Chile ETF
        "EWC",    # iShares MSCI Canada ETF
        "TUR",    # iShares MSCI Turkey ETF
        "EWA",    # iShares MSCI Australia ETF
        "GREK",   # Global X MSCI Greece ETF
        "EWD",    # iShares MSCI Sweden ETF
        "EWI",    # iShares MSCI Italy ETF
        "COLO",   # Global X MSCI Colombia ETF
        "ARGT",   # Global X MSCI Argentina ETF
        "EWJ",    # iShares MSCI Japan ETF
        "EWY",    # iShares MSCI South Korea ETF
        "EWP",    # iShares MSCI Spain ETF
        "EPHE",   # iShares MSCI Philippines ETF
        "EIUV",   # iShares MSCI UAE ETF
        "ENZL",   # iShares MSCI New Zealand ETF
        "ERUS",   # iShares MSCI Russia ETF
        "EZA",    # iShares MSCI South Africa ETF
        "COPX",   # Global X Copper Miners ETF
        "URA",    # Global X Uranium ETF
        "PICK",   # iShares MSCI Global Metals & Mining Producers ETF
        "EZU",    # iShares MSCI Eurozone ETF
        "IAU",    # iShares Gold Trust
        "TQQQ",   # ProShares UltraPro QQQ
        "VNM",    # VanEck Vectors Vietnam ETF
        
    ]
    
    # Settings
    period = "6mo"       # 6 months of data
    interval = "1d"      # Daily candles
    benchmark = "SPY"    # S&P 500 as benchmark
    
    # Create enhanced screener and run scan
    print("=" * 180)
    print("INITIALIZING ENHANCED ETF SCREENER...")
    print("=" * 180)
    
    screener = EnhancedETFScreener(etf_list, period=period, interval=interval, benchmark=benchmark)
    screener.scan()
    screener.display_results()
    screener.save_results()
