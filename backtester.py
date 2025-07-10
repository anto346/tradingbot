"""
Simple backtesting module for the Enhanced Trading Bot
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

class SimpleBacktester:
    """Simple backtesting system"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.positions = {}
    
    def run_backtest(self, bot, symbol: str, days: int = 365) -> Dict:
        """Run simple backtest"""
        print(f"🧪 Running backtest for {symbol}...")
        
        # Get test data
        df = bot.get_historical_data(symbol, "day", days)
        if df is None or len(df) < 100:
            return {"error": "Insufficient data"}
        
        # Calculate indicators
        df = bot.calculate_technical_indicators(df)
        
        # Simulate trading
        for i in range(50, len(df) - 5):
            current_data = df.iloc[:i+1]
            current_price = current_data.iloc[-1]['close']
            
            # Check for buy signals
            if symbol not in self.positions:
                buy_signal = bot.generate_buy_signal(current_data, symbol)
                
                if buy_signal['action'] == 'BUY':
                    # Calculate position size (10% of capital)
                    position_value = self.current_capital * 0.1
                    quantity = int(position_value / current_price)
                    
                    if quantity > 0:
                        cost = quantity * current_price
                        self.current_capital -= cost
                        
                        self.positions[symbol] = {
                            'quantity': quantity,
                            'entry_price': current_price,
                            'entry_date': current_data.index[-1]
                        }
            else:
                # Check for sell signals
                position = self.positions[symbol]
                sell_signal = bot.generate_sell_signal(current_data, symbol, position['entry_price'])
                
                if sell_signal['action'] == 'SELL':
                    # Sell position
                    proceeds = position['quantity'] * current_price
                    self.current_capital += proceeds
                    
                    profit = proceeds - (position['quantity'] * position['entry_price'])
                    
                    self.trades.append({
                        'symbol': symbol,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'profit': profit,
                        'entry_date': position['entry_date'],
                        'exit_date': current_data.index[-1]
                    })
                    
                    del self.positions[symbol]
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict:
        """Analyze backtest results"""
        if not self.trades:
            return {"error": "No trades executed"}
        
        profits = [t['profit'] for t in self.trades]
        
        total_profit = sum(profits)
        winning_trades = [p for p in profits if p > 0]
        
        win_rate = len(winning_trades) / len(profits) * 100
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return_pct': total_return,
            'final_capital': self.current_capital,
            'avg_profit_per_trade': total_profit / len(self.trades)
        }

def run_simple_backtest():
    """Run a simple backtest"""
    print("🔬 Simple Backtesting Demo")
    print("=" * 40)
    
    from enhanced_trading_bot import EnhancedZerodhaBot, TradingConfig
    
    # Create bot
    config = TradingConfig(demo_mode=True)
    bot = EnhancedZerodhaBot(config=config)
    
    # Run backtest
    backtester = SimpleBacktester()
    results = backtester.run_backtest(bot, "RELIANCE", 365)
    
    print("📊 Backtest Results:")
    print("-" * 30)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    run_simple_backtest()
