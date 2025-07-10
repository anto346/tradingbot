# Windows Unicode Fix - Added automatically
import sys
import os

# Fix Windows console encoding
if sys.platform.startswith('win'):
    try:
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass

#!/usr/bin/env python3
"""
Enhanced Zerodha Trading Bot v2.0
Complete algorithmic trading system with advanced features
"""

import pandas as pd
import numpy as np
import time
import logging
import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pickle
from dataclasses import dataclass
from enum import Enum

# Third-party imports with fallbacks
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    print("⚠️ KiteConnect not installed. Running in simulation mode only.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Scikit-learn not available. ML features disabled.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configuration
@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    # Risk Management
    risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.10
    max_positions: int = 5
    profit_target_multiplier: float = 2.0
    
    # Technical Analysis
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    bb_period: int = 20
    ma_short: int = 9
    ma_long: int = 21
    
    # Trading
    demo_mode: bool = True
    paper_trading: bool = True
    min_signal_strength: float = 3.0
    
    # Timing
    scan_interval_minutes: int = 5
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 30

class MarketRegime(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"

class EnhancedZerodhaBot:
    """Enhanced Trading Bot with advanced features"""
    
    def __init__(self, api_key: str = None, access_token: str = None, config: TradingConfig = None):
        """Initialize the enhanced trading bot"""
        
        self.config = config or TradingConfig()
        self.api_key = api_key
        self.access_token = access_token
        
        # Initialize Kite connection
        if KITE_AVAILABLE and api_key and not self.config.demo_mode:
            try:
                self.kite = KiteConnect(api_key=api_key)
                self.kite.set_access_token(access_token)
                self.api_connected = True
            except Exception as e:
                print(f"⚠️ Failed to connect to Kite API: {e}")
                self.api_connected = False
        else:
            self.kite = None
            self.api_connected = False
        
        # Initialize components
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {}
        self.current_regime = MarketRegime.SIDEWAYS_MARKET
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info("🚀 Enhanced Trading Bot Initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging with Windows compatibility"""
        import logging
        from datetime import datetime
        
        # Create logs directory
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Configure logging with safe encoding
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_filename = f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
        
        # Setup basic logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8', errors='ignore'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Override logging to handle Unicode errors
        original_info = self.logger.info
        original_error = self.logger.error
        
        def safe_log(level_func, message):
            try:
                level_func(message)
            except UnicodeEncodeError:
                # Remove emojis for Windows compatibility
                safe_message = str(message)
                emoji_map = {
                    '🚀': '[START]', '🔍': '[SCAN]', '📊': '[ANALYSIS]',
                    '💰': '[MONEY]', '📈': '[UP]', '✅': '[OK]', '❌': '[ERROR]'
                }
                for emoji, replacement in emoji_map.items():
                    safe_message = safe_message.replace(emoji, replacement)
                level_func(safe_message.encode('ascii', 'ignore').decode('ascii'))
        
        self.logger.info = lambda msg: safe_log(original_info, msg)
        self.logger.error = lambda msg: safe_log(original_error, msg)

    # Technical Analysis Methods
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd, signal)
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = self.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            # Moving Averages
            df['SMA_9'] = self.calculate_sma(df['close'], self.config.ma_short)
            df['SMA_21'] = self.calculate_sma(df['close'], self.config.ma_long)
            df['EMA_9'] = self.calculate_ema(df['close'], self.config.ma_short)
            df['EMA_21'] = self.calculate_ema(df['close'], self.config.ma_long)
            
            # Momentum Indicators
            df['RSI'] = self.calculate_rsi(df['close'])
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = self.calculate_macd(df['close'])
            
            # Volatility Indicators
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(df['close'])
            df['ATR'] = self.calculate_atr(df['high'], df['low'], df['close'])
            
            # Volume indicators
            df['volume_sma'] = self.calculate_sma(df['volume'], 20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return df

    def get_historical_data(self, symbol: str, interval: str = "5minute", days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data with fallback to simulation"""
        try:
            if self.api_connected:
                return self._get_real_historical_data(symbol, interval, days)
            else:
                return self._generate_realistic_data(symbol, days)
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return self._generate_realistic_data(symbol, days)
    
    def _generate_realistic_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate realistic market data for testing"""
        np.random.seed(hash(symbol) % 2**32)
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             periods=days, freq='D')
        
        # Different characteristics per symbol
        base_price = 100 + (hash(symbol) % 500)
        volatility = 0.015 + 0.01 * (hash(symbol) % 4)
        trend = 0.0005 * ((hash(symbol) % 5) - 2)
        
        prices = []
        volumes = []
        price = base_price
        
        for i in range(days):
            # Add market microstructure
            daily_change = np.random.normal(trend, volatility)
            price = max(10, price * (1 + daily_change))
            prices.append(price)
            
            # Volume with some correlation to price movement
            base_volume = 5000 + (hash(symbol) % 10000)
            volume = int(base_volume * np.random.uniform(0.8, 1.2))
            volumes.append(volume)
        
        # Create OHLC data
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = [p * np.random.uniform(0.995, 1.005) for p in prices]
        df['high'] = [max(o, c) * np.random.uniform(1.000, 1.015) for o, c in zip(df['open'], df['close'])]
        df['low'] = [min(o, c) * np.random.uniform(0.985, 1.000) for o, c in zip(df['open'], df['close'])]
        df['volume'] = volumes
        
        return df

    def generate_buy_signal(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Generate enhanced buy signals"""
        try:
            if len(df) < 50:
                return {'action': 'HOLD', 'score': 0, 'signals': [], 'confidence': 0}
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            signals = []
            score = 0
            
            # RSI Oversold
            if latest['RSI'] < self.config.rsi_oversold:
                signals.append(f"RSI Oversold ({latest['RSI']:.1f})")
                score += 2.0
            
            # MACD Bullish crossover
            if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
                signals.append("MACD Bullish Crossover")
                score += 2.5
            
            # Moving Average Analysis
            if latest['close'] > latest['SMA_9'] > latest['SMA_21']:
                signals.append("Price Above MAs")
                score += 1.5
            
            # Bollinger Bands
            bb_position = (latest['close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
            if bb_position <= 0.2:
                signals.append("Near Lower BB")
                score += 1.5
            
            # Volume confirmation
            if latest['volume_ratio'] > 1.5:
                signals.append("High Volume")
                score += 1.0
            
            confidence = min(100, (score / 8) * 100)
            
            # Risk management
            stop_loss = latest['close'] - (2 * latest['ATR'])
            target = latest['close'] + (4 * latest['ATR'])
            
            return {
                'action': 'BUY' if score >= self.config.min_signal_strength else 'HOLD',
                'score': round(score, 2),
                'confidence': round(confidence, 1),
                'signals': signals,
                'entry_price': latest['close'],
                'stop_loss': stop_loss,
                'target': target,
                'atr': latest['ATR']
            }
            
        except Exception as e:
            self.logger.error(f"Error generating buy signal for {symbol}: {e}")
            return {'action': 'HOLD', 'score': 0, 'signals': [], 'confidence': 0}

    def generate_sell_signal(self, df: pd.DataFrame, symbol: str, entry_price: float = None) -> Dict:
        """Generate enhanced sell signals"""
        try:
            if len(df) < 2:
                return {'action': 'HOLD', 'score': 0, 'signals': [], 'confidence': 0}
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            signals = []
            score = 0
            
            # RSI Overbought
            if latest['RSI'] > self.config.rsi_overbought:
                signals.append(f"RSI Overbought ({latest['RSI']:.1f})")
                score += 2.0
            
            # MACD Bearish crossover
            if latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
                signals.append("MACD Bearish Crossover")
                score += 2.5
            
            # Profit/Loss Management
            if entry_price:
                profit_pct = (latest['close'] - entry_price) / entry_price * 100
                
                if profit_pct >= 5:
                    signals.append(f"Big Profit ({profit_pct:.1f}%)")
                    score += 3.0
                elif profit_pct >= 3:
                    signals.append(f"Good Profit ({profit_pct:.1f}%)")
                    score += 2.0
                elif profit_pct <= -2:
                    signals.append(f"Stop Loss ({profit_pct:.1f}%)")
                    score += 4.0
            
            confidence = min(100, (score / 8) * 100)
            
            return {
                'action': 'SELL' if score >= self.config.min_signal_strength else 'HOLD',
                'score': round(score, 2),
                'confidence': round(confidence, 1),
                'signals': signals,
                'exit_price': latest['close']
            }
            
        except Exception as e:
            self.logger.error(f"Error generating sell signal for {symbol}: {e}")
            return {'action': 'HOLD', 'score': 0, 'signals': [], 'confidence': 0}

    def scan_market(self, watchlist: List[str]) -> List[Dict]:
        """Scan market for opportunities"""
        opportunities = []
        
        for symbol in watchlist:
            try:
                self.logger.info(f"🔍 Scanning {symbol}...")
                
                # Get data and calculate indicators
                df = self.get_historical_data(symbol, "5minute", 30)
                if df is None or len(df) < 50:
                    continue
                
                df = self.calculate_technical_indicators(df)
                
                # Generate signals
                if symbol in self.positions:
                    # Check for exit signals
                    entry_price = self.positions[symbol]['entry_price']
                    sell_signal = self.generate_sell_signal(df, symbol, entry_price)
                    
                    if sell_signal['action'] == 'SELL':
                        opportunities.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'type': 'EXIT',
                            'current_price': df.iloc[-1]['close'],
                            'signal_data': sell_signal
                        })
                else:
                    # Check for entry signals
                    if len(self.positions) < self.config.max_positions:
                        buy_signal = self.generate_buy_signal(df, symbol)
                        
                        if buy_signal['action'] == 'BUY':
                            opportunities.append({
                                'symbol': symbol,
                                'action': 'BUY',
                                'type': 'ENTRY',
                                'current_price': df.iloc[-1]['close'],
                                'signal_data': buy_signal
                            })
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        return opportunities

    def place_order(self, symbol: str, transaction_type: str, quantity: int, price: float = None) -> Optional[str]:
        """Place order with demo mode support"""
        try:
            order_params = {
                'tradingsymbol': symbol,
                'exchange': 'NSE',
                'transaction_type': transaction_type,
                'quantity': quantity,
                'product': 'MIS',
                'order_type': 'MARKET' if price is None else 'LIMIT',
                'validity': 'DAY'
            }
            
            if price:
                order_params['price'] = price
            
            if self.config.demo_mode or not self.api_connected:
                # Demo mode
                order_id = f"DEMO_{symbol}_{transaction_type}_{int(time.time())}"
                self.logger.info(f"[DEMO] Order placed: {order_params}")
                print(f"🔥 [DEMO] {transaction_type} {quantity} shares of {symbol}")
                return order_id
            else:
                # Real trading
                order_id = self.kite.place_order(**order_params)
                self.logger.info(f"Real order placed: {order_id}")
                return order_id
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk"""
        try:
            account_balance = self.get_account_balance()
            risk_amount = account_balance * self.config.risk_per_trade
            price_diff = abs(entry_price - stop_loss)
            
            if price_diff <= 0:
                return 1
            
            position_size = int(risk_amount / price_diff)
            return max(1, min(position_size, int(account_balance * 0.1 / entry_price)))
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1

    def get_account_balance(self) -> float:
        """Get account balance with fallback"""
        try:
            if self.api_connected and not self.config.demo_mode:
                margins = self.kite.margins()
                return margins['equity']['available']['live_balance']
            else:
                return 100000.0  # Demo balance
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return 100000.0

    def execute_opportunities(self, opportunities: List[Dict]):
        """Execute trading opportunities"""
        for opportunity in opportunities:
            try:
                symbol = opportunity['symbol']
                action = opportunity['action']
                signal_data = opportunity['signal_data']
                
                if action == 'BUY' and len(self.positions) < self.config.max_positions:
                    entry_price = signal_data['entry_price']
                    stop_loss = signal_data['stop_loss']
                    target = signal_data['target']
                    
                    quantity = self.calculate_position_size(entry_price, stop_loss)
                    order_id = self.place_order(symbol, 'BUY', quantity)
                    
                    if order_id:
                        self.positions[symbol] = {
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'stop_loss': stop_loss,
                            'target': target,
                            'order_id': order_id,
                            'entry_time': datetime.now(),
                            'confidence': signal_data['confidence']
                        }
                        
                        self.logger.info(f"✅ Bought {quantity} shares of {symbol}")
                
                elif action == 'SELL' and symbol in self.positions:
                    quantity = self.positions[symbol]['quantity']
                    order_id = self.place_order(symbol, 'SELL', quantity)
                    
                    if order_id:
                        entry_price = self.positions[symbol]['entry_price']
                        exit_price = signal_data['exit_price']
                        pnl = (exit_price - entry_price) * quantity
                        
                        self.trade_history.append({
                            'symbol': symbol,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'quantity': quantity,
                            'pnl': pnl,
                            'entry_time': self.positions[symbol]['entry_time'],
                            'exit_time': datetime.now()
                        })
                        
                        self.logger.info(f"✅ Sold {quantity} shares of {symbol}. P&L: ₹{pnl:.2f}")
                        del self.positions[symbol]
                
            except Exception as e:
                self.logger.error(f"Error executing opportunity: {e}")

    def generate_report(self, opportunities: List[Dict]):
        """Generate comprehensive trading report"""
        print("\n" + "="*80)
        print("📊 ENHANCED TRADING BOT - MARKET ANALYSIS REPORT")
        print("="*80)
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        account_balance = self.get_account_balance()
        
        print(f"📅 Date & Time: {current_time}")
        print(f"💰 Account Balance: ₹{account_balance:,.2f}")
        print(f"📈 Active Positions: {len(self.positions)}")
        print(f"🔍 Opportunities Found: {len(opportunities)}")
        
        # Current Positions
        if self.positions:
            print(f"\n📋 ACTIVE POSITIONS ({len(self.positions)}):")
            print("-" * 80)
            
            total_invested = 0
            total_current_value = 0
            
            for symbol, pos in self.positions.items():
                try:
                    current_df = self.get_historical_data(symbol, "minute", 1)
                    if current_df is not None and len(current_df) > 0:
                        current_price = current_df.iloc[-1]['close']
                        invested = pos['entry_price'] * pos['quantity']
                        current_value = current_price * pos['quantity']
                        pnl = current_value - invested
                        pnl_pct = (pnl / invested) * 100
                        
                        total_invested += invested
                        total_current_value += current_value
                        
                        status_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                        
                        print(f"\n{status_emoji} {symbol}")
                        print(f"   📊 Qty: {pos['quantity']} | Entry: ₹{pos['entry_price']:.2f} | Current: ₹{current_price:.2f}")
                        print(f"   💰 P&L: ₹{pnl:,.2f} ({pnl_pct:+.2f}%) | Confidence: {pos.get('confidence', 0):.1f}%")
                        print(f"   🎯 Target: ₹{pos['target']:.2f} | 🛡️ Stop: ₹{pos['stop_loss']:.2f}")
                        
                except Exception as e:
                    print(f"   ❌ Error getting price for {symbol}")
        
        # Trading Opportunities
        if opportunities:
            print(f"\n🎯 TRADING OPPORTUNITIES ({len(opportunities)}):")
            print("-" * 80)
            
            buy_opportunities = [opp for opp in opportunities if opp['action'] == 'BUY']
            sell_opportunities = [opp for opp in opportunities if opp['action'] == 'SELL']
            
            if buy_opportunities:
                print(f"\n🟢 BUY SIGNALS ({len(buy_opportunities)}):")
                for i, opp in enumerate(buy_opportunities, 1):
                    signal_data = opp['signal_data']
                    print(f"\n{i}. {opp['symbol']} - BUY Signal")
                    print(f"   💹 Price: ₹{opp['current_price']:.2f}")
                    print(f"   📊 Strength: {signal_data['score']:.1f}/10 | Confidence: {signal_data['confidence']:.1f}%")
                    print(f"   🎯 Target: ₹{signal_data['target']:.2f} | 🛡️ Stop: ₹{signal_data['stop_loss']:.2f}")
                    print(f"   📋 Signals: {', '.join(signal_data['signals'][:3])}")
            
            if sell_opportunities:
                print(f"\n🔴 SELL SIGNALS ({len(sell_opportunities)}):")
                for i, opp in enumerate(sell_opportunities, 1):
                    signal_data = opp['signal_data']
                    print(f"\n{i}. {opp['symbol']} - SELL Signal")
                    print(f"   💹 Price: ₹{opp['current_price']:.2f}")
                    print(f"   📊 Strength: {signal_data['score']:.1f}/10 | Confidence: {signal_data['confidence']:.1f}%")
                    print(f"   📋 Signals: {', '.join(signal_data['signals'][:3])}")
        
        # Performance Summary
        if self.trade_history:
            completed_trades = len(self.trade_history)
            total_pnl = sum(t['pnl'] for t in self.trade_history)
            winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
            win_rate = (winning_trades / completed_trades) * 100 if completed_trades > 0 else 0
            
            print(f"\n📈 PERFORMANCE SUMMARY:")
            print(f"   📊 Total Trades: {completed_trades}")
            print(f"   ✅ Win Rate: {win_rate:.1f}%")
            print(f"   💰 Total P&L: ₹{total_pnl:,.2f}")
        
        print("\n" + "="*80)

    def _is_market_open(self, current_time: datetime) -> bool:
        """Check if market is open"""
        try:
            if current_time.weekday() > 4:  # Weekend
                return False
            
            market_open = current_time.replace(
                hour=self.config.market_open_hour, 
                minute=self.config.market_open_minute, 
                second=0, microsecond=0
            )
            market_close = current_time.replace(
                hour=self.config.market_close_hour, 
                minute=self.config.market_close_minute, 
                second=0, microsecond=0
            )
            
            return market_open <= current_time <= market_close
        except Exception:
            return False

    def run_trading_session(self, watchlist: List[str]):
        """Main trading session loop"""
        print("🚀 Enhanced Trading Bot Starting...")
        self.logger.info("🚀 Enhanced Trading Bot Starting...")
        
        try:
            while True:
                current_time = datetime.now()
                
                if self._is_market_open(current_time):
                    print(f"\n⏰ Market OPEN - Scan at {current_time.strftime('%H:%M:%S')}")
                    
                    try:
                        # Scan for opportunities
                        opportunities = self.scan_market(watchlist)
                        
                        # Generate report
                        self.generate_report(opportunities)
                        
                        # Execute trades
                        if not self.config.demo_mode or self.config.paper_trading:
                            self.execute_opportunities(opportunities)
                        
                    except Exception as e:
                        self.logger.error(f"Error in trading loop: {e}")
                    
                    print(f"\n⏰ Next scan in {self.config.scan_interval_minutes} minutes...")
                    time.sleep(self.config.scan_interval_minutes * 60)
                    
                else:
                    print("🕒 Market CLOSED - Waiting...")
                    time.sleep(1800)  # Wait 30 minutes
                    
        except KeyboardInterrupt:
            print("\n🛑 Trading bot stopped by user")
            self.logger.info("Trading bot stopped by user")

# Test Functions
def run_quick_test():
    """Run quick test of the trading bot"""
    print("🧪 QUICK TRADING BOT TEST")
    print("=" * 50)
    
    # Create bot with demo config
    config = TradingConfig(demo_mode=True, paper_trading=True)
    bot = EnhancedZerodhaBot(config=config)
    
    # Test watchlist
    test_watchlist = ["RELIANCE", "TCS", "INFY"]
    
    print(f"📊 Testing {len(test_watchlist)} stocks...")
    
    # Quick scan
    opportunities = bot.scan_market(test_watchlist)
    
    # Generate report
    bot.generate_report(opportunities)
    
    print("\n✅ Quick test completed!")
    return bot, opportunities

def main():
    """Main function"""
    print("🚀 ENHANCED ZERODHA TRADING BOT v2.0")
    print("=" * 60)
    
    # Configuration
    config = TradingConfig(
        demo_mode=True,
        paper_trading=True,
        risk_per_trade=0.02,
        max_positions=5,
        min_signal_strength=3.0,
        scan_interval_minutes=5
    )
    
    # Watchlist
    watchlist = [
        "RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK",
        "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"
    ]
    
    print("🎯 Configuration:")
    print(f"   Demo Mode: {config.demo_mode}")
    print(f"   Risk per Trade: {config.risk_per_trade*100}%")
    print(f"   Max Positions: {config.max_positions}")
    
    # Run quick test by default
    run_quick_test()
    
    # Uncomment below for live trading session
    # bot = EnhancedZerodhaBot(config=config)
    # bot.run_trading_session(watchlist)

if __name__ == "__main__":
    main()
