#!/usr/bin/env python3
"""
Enhanced Demo Trading Bot - Shows Realistic Trading Signals
"""

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    risk_per_trade: float = 0.02
    max_positions: int = 5
    min_signal_strength: float = 2.5  # Lowered for demo
    rsi_oversold: int = 35  # Adjusted for more signals
    rsi_overbought: int = 65
    demo_mode: bool = True

class EnhancedDemoBot:
    """Enhanced demo bot with realistic trading signals"""
    
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.positions = {}
        self.trade_history = []
        print("[INIT] Enhanced Demo Trading Bot Initialized")
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2):
        sma = self.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def generate_realistic_market_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Generate realistic market data with trading patterns"""
        np.random.seed(hash(symbol) % 2**32)
        
        # Different market scenarios for different symbols
        market_scenarios = {
            'RELIANCE': {'trend': 0.0008, 'volatility': 0.025, 'base_price': 2800},
            'TCS': {'trend': 0.0005, 'volatility': 0.020, 'base_price': 3600},
            'INFY': {'trend': 0.0006, 'volatility': 0.022, 'base_price': 1650},
            'HDFC': {'trend': 0.0004, 'volatility': 0.018, 'base_price': 1580},
            'ICICIBANK': {'trend': 0.0007, 'volatility': 0.024, 'base_price': 950}
        }
        
        scenario = market_scenarios.get(symbol, {'trend': 0.0005, 'volatility': 0.02, 'base_price': 1000})
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
        
        # Generate price data with realistic patterns
        prices = []
        volumes = []
        price = scenario['base_price']
        
        for i in range(days):
            # Create realistic price movements with cycles
            cycle_effect = 0.0003 * np.sin(i * 0.1)  # Market cycles
            news_effect = 0.002 * np.random.choice([-1, 0, 0, 0, 1])  # Occasional news impact
            
            daily_change = scenario['trend'] + cycle_effect + news_effect + np.random.normal(0, scenario['volatility'])
            
            # Add some momentum and mean reversion
            if len(prices) > 5:
                momentum = np.mean([prices[-1]/prices[-2] - 1, prices[-2]/prices[-3] - 1]) * 0.3
                mean_reversion = -0.1 * (price - scenario['base_price']) / scenario['base_price']
                daily_change += momentum + mean_reversion
            
            price = max(scenario['base_price'] * 0.7, price * (1 + daily_change))
            prices.append(price)
            
            # Volume correlated with price movement and volatility
            base_volume = 100000 + (hash(symbol) % 50000)
            volume_multiplier = 1 + abs(daily_change) * 5 + np.random.uniform(-0.3, 0.3)
            volume = int(base_volume * volume_multiplier)
            volumes.append(volume)
        
        # Create OHLC data
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        
        # Generate realistic OHLC from close prices
        df['open'] = df['close'].shift(1) * np.random.uniform(0.998, 1.002, len(df))
        df['open'].iloc[0] = df['close'].iloc[0]
        
        # High and low based on intraday volatility
        intraday_range = scenario['volatility'] * 0.8
        df['high'] = df[['open', 'close']].max(axis=1) * np.random.uniform(1.0, 1 + intraday_range, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) * np.random.uniform(1 - intraday_range, 1.0, len(df))
        
        df['volume'] = volumes
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        # Moving Averages
        df['SMA_9'] = self.calculate_sma(df['close'], 9)
        df['SMA_21'] = self.calculate_sma(df['close'], 21)
        df['SMA_50'] = self.calculate_sma(df['close'], 50)
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['close'])
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = self.calculate_macd(df['close'])
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(df['close'])
        
        # Volume indicators
        df['volume_sma'] = self.calculate_sma(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['price_momentum'] = df['close'].rolling(5).mean() / df['close'].rolling(20).mean()
        
        return df
    
    def generate_buy_signal(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Generate buy signals with detailed analysis"""
        if len(df) < 50:
            return {'action': 'HOLD', 'score': 0, 'signals': [], 'confidence': 0}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        score = 0
        
        # 1. RSI Analysis
        if latest['RSI'] < self.config.rsi_oversold:
            signals.append(f"RSI Oversold ({latest['RSI']:.1f})")
            score += 2.5
        elif latest['RSI'] < 45:
            signals.append(f"RSI Favorable ({latest['RSI']:.1f})")
            score += 1.5
        
        # 2. MACD Analysis
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            signals.append("MACD Bullish Crossover")
            score += 3.0
        elif latest['MACD'] > latest['MACD_signal']:
            signals.append("MACD Above Signal")
            score += 1.0
        
        if latest['MACD_hist'] > prev['MACD_hist']:
            signals.append("MACD Histogram Rising")
            score += 1.0
        
        # 3. Moving Average Analysis
        if latest['close'] > latest['SMA_9'] > latest['SMA_21']:
            signals.append("Bullish MA Alignment")
            score += 2.0
        elif latest['close'] > latest['SMA_21']:
            signals.append("Price Above MA21")
            score += 1.0
        
        # 4. Bollinger Bands Analysis
        if not pd.isna(latest['BB_lower']):
            bb_position = (latest['close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
            if bb_position <= 0.25:
                signals.append("Near Lower Bollinger Band")
                score += 2.0
            elif bb_position <= 0.5:
                signals.append("Below BB Middle")
                score += 1.0
        
        # 5. Volume Analysis
        if latest['volume_ratio'] > 1.3:
            signals.append("High Volume Support")
            score += 1.5
        elif latest['volume_ratio'] > 1.1:
            signals.append("Above Average Volume")
            score += 0.5
        
        # 6. Price Momentum
        if latest['price_momentum'] > 1.02:
            signals.append("Strong Price Momentum")
            score += 1.5
        elif latest['price_momentum'] > 1.0:
            signals.append("Positive Momentum")
            score += 0.5
        
        # 7. Recent Price Action
        recent_low = df['close'].rolling(10).min().iloc[-1]
        if latest['close'] > recent_low * 1.02:
            signals.append("Breaking Recent Resistance")
            score += 1.0
        
        # Calculate confidence and risk parameters
        confidence = min(100, (score / 10) * 100)
        
        # Risk management
        atr = df['close'].rolling(14).std().iloc[-1] * 2
        stop_loss = latest['close'] - atr
        target = latest['close'] + (atr * 2)
        
        return {
            'action': 'BUY' if score >= self.config.min_signal_strength else 'HOLD',
            'score': round(score, 2),
            'confidence': round(confidence, 1),
            'signals': signals,
            'entry_price': latest['close'],
            'stop_loss': stop_loss,
            'target': target,
            'risk_reward': round((target - latest['close']) / (latest['close'] - stop_loss), 2)
        }
    
    def generate_sell_signal(self, df: pd.DataFrame, symbol: str, entry_price: float = None) -> Dict:
        """Generate sell signals"""
        if len(df) < 2:
            return {'action': 'HOLD', 'score': 0, 'signals': [], 'confidence': 0}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        score = 0
        
        # RSI Overbought
        if latest['RSI'] > self.config.rsi_overbought:
            signals.append(f"RSI Overbought ({latest['RSI']:.1f})")
            score += 2.5
        elif latest['RSI'] > 60:
            signals.append(f"RSI High ({latest['RSI']:.1f})")
            score += 1.0
        
        # MACD Bearish
        if latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            signals.append("MACD Bearish Crossover")
            score += 3.0
        
        # Profit Management
        if entry_price:
            profit_pct = (latest['close'] - entry_price) / entry_price * 100
            
            if profit_pct >= 8:
                signals.append(f"Excellent Profit ({profit_pct:.1f}%)")
                score += 4.0
            elif profit_pct >= 5:
                signals.append(f"Good Profit ({profit_pct:.1f}%)")
                score += 2.5
            elif profit_pct >= 3:
                signals.append(f"Decent Profit ({profit_pct:.1f}%)")
                score += 1.5
            elif profit_pct <= -3:
                signals.append(f"Stop Loss Trigger ({profit_pct:.1f}%)")
                score += 5.0
        
        confidence = min(100, (score / 8) * 100)
        
        return {
            'action': 'SELL' if score >= self.config.min_signal_strength else 'HOLD',
            'score': round(score, 2),
            'confidence': round(confidence, 1),
            'signals': signals,
            'exit_price': latest['close']
        }
    
    def scan_market_enhanced(self, watchlist: List[str]) -> List[Dict]:
        """Enhanced market scanning with detailed analysis"""
        opportunities = []
        
        print("\n[SCAN] Starting Enhanced Market Analysis...")
        print("=" * 60)
        
        for i, symbol in enumerate(watchlist, 1):
            print(f"\n[{i}/{len(watchlist)}] Analyzing {symbol}...")
            
            try:
                # Generate realistic data
                df = self.generate_realistic_market_data(symbol, 100)
                df = self.calculate_technical_indicators(df)
                
                current_price = df.iloc[-1]['close']
                
                # Show current market data
                print(f"   Current Price: Rs.{current_price:.2f}")
                print(f"   RSI: {df.iloc[-1]['RSI']:.1f}")
                print(f"   MACD: {df.iloc[-1]['MACD']:.2f}")
                
                # Generate signals
                if symbol in self.positions:
                    # Check exit signals for existing positions
                    entry_price = self.positions[symbol]['entry_price']
                    sell_signal = self.generate_sell_signal(df, symbol, entry_price)
                    
                    if sell_signal['action'] == 'SELL':
                        opportunities.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'type': 'EXIT',
                            'current_price': current_price,
                            'signal_data': sell_signal
                        })
                        print(f"   [SELL] Signal Strength: {sell_signal['score']:.1f}")
                else:
                    # Check entry signals
                    if len(self.positions) < self.config.max_positions:
                        buy_signal = self.generate_buy_signal(df, symbol)
                        
                        if buy_signal['action'] == 'BUY':
                            opportunities.append({
                                'symbol': symbol,
                                'action': 'BUY',
                                'type': 'ENTRY',
                                'current_price': current_price,
                                'signal_data': buy_signal
                            })
                            print(f"   [BUY] Signal Strength: {buy_signal['score']:.1f}")
                        else:
                            print(f"   [HOLD] Signal Strength: {buy_signal['score']:.1f} (Below threshold)")
                
            except Exception as e:
                print(f"   [ERROR] Failed to analyze {symbol}: {e}")
        
        print(f"\n[SCAN] Analysis complete. Found {len(opportunities)} opportunities.")
        return opportunities
    
    def generate_enhanced_report(self, opportunities: List[Dict]):
        """Generate detailed trading report"""
        print("\n" + "="*90)
        print("              ENHANCED DEMO TRADING BOT - DETAILED MARKET REPORT")
        print("="*90)
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        account_balance = 100000.0
        
        print(f"Date & Time: {current_time}")
        print(f"Account Balance: Rs.{account_balance:,.2f}")
        print(f"Active Positions: {len(self.positions)}")
        print(f"Signal Threshold: {self.config.min_signal_strength}")
        print(f"Opportunities Found: {len(opportunities)}")
        
        if opportunities:
            buy_opps = [o for o in opportunities if o['action'] == 'BUY']
            sell_opps = [o for o in opportunities if o['action'] == 'SELL']
            
            if buy_opps:
                print(f"\n[BUY OPPORTUNITIES] ({len(buy_opps)} found):")
                print("-" * 90)
                
                for i, opp in enumerate(buy_opps, 1):
                    signal = opp['signal_data']
                    print(f"\n{i}. {opp['symbol']} - BUY SIGNAL")
                    print(f"   Price: Rs.{opp['current_price']:.2f}")
                    print(f"   Signal Strength: {signal['score']:.1f}/10")
                    print(f"   Confidence: {signal['confidence']:.1f}%")
                    print(f"   Target: Rs.{signal['target']:.2f}")
                    print(f"   Stop Loss: Rs.{signal['stop_loss']:.2f}")
                    print(f"   Risk:Reward = 1:{signal.get('risk_reward', 0):.1f}")
                    
                    print(f"   Key Signals:")
                    for sig in signal['signals']:
                        print(f"     + {sig}")
                    
                    # Calculate position size
                    risk_amount = account_balance * self.config.risk_per_trade
                    price_diff = opp['current_price'] - signal['stop_loss']
                    if price_diff > 0:
                        quantity = int(risk_amount / price_diff)
                        investment = quantity * opp['current_price']
                        print(f"   Recommended Position:")
                        print(f"     Quantity: {quantity} shares")
                        print(f"     Investment: Rs.{investment:,.2f}")
                        print(f"     Risk: Rs.{quantity * price_diff:,.2f}")
            
            if sell_opps:
                print(f"\n[SELL OPPORTUNITIES] ({len(sell_opps)} found):")
                print("-" * 90)
                
                for i, opp in enumerate(sell_opps, 1):
                    signal = opp['signal_data']
                    print(f"\n{i}. {opp['symbol']} - SELL SIGNAL")
                    print(f"   Price: Rs.{opp['current_price']:.2f}")
                    print(f"   Signal Strength: {signal['score']:.1f}/10")
                    print(f"   Confidence: {signal['confidence']:.1f}%")
                    
                    print(f"   Key Signals:")
                    for sig in signal['signals']:
                        print(f"     + {sig}")
        
        else:
            print(f"\n[NO OPPORTUNITIES] No signals above threshold of {self.config.min_signal_strength}")
            print("Consider:")
            print("- Lowering signal threshold")
            print("- Checking different stocks")
            print("- Adjusting RSI parameters")
        
        # Performance metrics if we have trade history
        if self.trade_history:
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
            total_pnl = sum(t['pnl'] for t in self.trade_history)
            win_rate = (winning_trades / total_trades) * 100
            
            print(f"\n[PERFORMANCE SUMMARY]:")
            print(f"   Total Trades: {total_trades}")
            print(f"   Winning Trades: {winning_trades}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Total P&L: Rs.{total_pnl:,.2f}")
        
        print("\n" + "="*90)

def run_enhanced_demo():
    """Run enhanced demo with realistic signals"""
    print("[ROCKET] ENHANCED DEMO TRADING BOT")
    print("=" * 50)
    
    # Create bot with adjusted parameters for more signals
    config = TradingConfig(
        min_signal_strength=2.5,  # Lower threshold for demo
        rsi_oversold=35,         # More sensitive
        rsi_overbought=65
    )
    
    bot = EnhancedDemoBot(config)
    
    # Enhanced watchlist with different market cap stocks
    watchlist = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"]
    
    print(f"[CHART] Analyzing {len(watchlist)} stocks with enhanced signals...")
    print(f"[TARGET] Signal Threshold: {config.min_signal_strength}")
    
    # Run enhanced market scan
    opportunities = bot.scan_market_enhanced(watchlist)
    
    # Generate detailed report
    bot.generate_enhanced_report(opportunities)
    
    return bot, opportunities

def run_sensitivity_analysis():
    """Show how different parameters affect signal generation"""
    print("\n[TEST] SIGNAL SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    thresholds = [1.5, 2.0, 2.5, 3.0, 3.5]
    watchlist = ["RELIANCE", "TCS", "INFY"]
    
    for threshold in thresholds:
        config = TradingConfig(min_signal_strength=threshold)
        bot = EnhancedDemoBot(config)
        
        opportunities = []
        for symbol in watchlist:
            df = bot.generate_realistic_market_data(symbol, 100)
            df = bot.calculate_technical_indicators(df)
            signal = bot.generate_buy_signal(df, symbol)
            
            if signal['action'] == 'BUY':
                opportunities.append({'symbol': symbol, 'signal': signal})
        
        print(f"Threshold {threshold}: {len(opportunities)} opportunities")
        for opp in opportunities:
            print(f"  {opp['symbol']}: {opp['signal']['score']:.1f} ({opp['signal']['confidence']:.0f}%)")

def main():
    """Main demo function"""
    print("=" * 60)
    print("    ENHANCED DEMO TRADING BOT WITH REALISTIC SIGNALS")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Run Enhanced Demo (Recommended)")
        print("2. Signal Sensitivity Analysis") 
        print("3. Quick Test")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-3): ").strip()
        
        if choice == "1":
            run_enhanced_demo()
        elif choice == "2":
            run_sensitivity_analysis()
        elif choice == "3":
            config = TradingConfig(min_signal_strength=2.0)
            bot = EnhancedDemoBot(config)
            opportunities = bot.scan_market_enhanced(["RELIANCE", "TCS"])
            bot.generate_enhanced_report(opportunities)
        elif choice == "0":
            print("\n[STOP] Demo completed. Happy trading!")
            break
        else:
            print("[ERROR] Invalid choice")
        
        if choice != "0":
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()