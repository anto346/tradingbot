"""
Utility functions for the Enhanced Trading Bot
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List

def format_currency(amount: float) -> str:
    """Format currency"""
    return f"₹{amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format percentage"""
    return f"{value:+.2f}%"

def save_data(data: Dict, filename: str):
    """Save data to JSON file"""
    try:
        with open(f"data/{filename}", 'w') as f:
            json.dump(data, f, default=str, indent=2)
    except Exception as e:
        print(f"Error saving data: {e}")

def load_data(filename: str) -> Dict:
    """Load data from JSON file"""
    try:
        with open(f"data/{filename}", 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def calculate_performance_metrics(trades: List[Dict]) -> Dict:
    """Calculate trading performance metrics"""
    if not trades:
        return {}
    
    profits = [t.get('pnl', 0) for t in trades]
    
    total_trades = len(trades)
    winning_trades = len([p for p in profits if p > 0])
    total_profit = sum(profits)
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_profit': avg_profit
    }

class TradingLogger:
    """Simple trading logger"""
    
    def __init__(self):
        self.log_file = f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"
    
    def log(self, message: str):
        """Log message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"{timestamp} - {message}\n"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception:
            pass  # Fail silently
        
        print(f"[{timestamp}] {message}")
