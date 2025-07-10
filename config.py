"""
Configuration management for Enhanced Trading Bot
"""

import os
from dataclasses import dataclass, field
from typing import List
from enum import Enum

class TradingMode(Enum):
    DEMO = "demo"
    PAPER = "paper" 
    LIVE = "live"

@dataclass
class EnhancedTradingConfig:
    """Enhanced trading configuration"""
    
    # Core Settings
    trading_mode: TradingMode = TradingMode.DEMO
    risk_per_trade: float = 0.02
    max_positions: int = 5
    min_signal_strength: float = 3.0
    
    # Watchlist
    default_watchlist: List[str] = field(default_factory=lambda: [
        "RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK",
        "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"
    ])
    
    # Alerts
    enable_telegram_alerts: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

def load_config_from_env():
    """Load configuration from environment variables"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️ python-dotenv not installed. Using default config.")
    
    mode_str = os.getenv('TRADING_MODE', 'demo').lower()
    try:
        trading_mode = TradingMode(mode_str)
    except ValueError:
        trading_mode = TradingMode.DEMO
    
    return EnhancedTradingConfig(
        trading_mode=trading_mode,
        risk_per_trade=float(os.getenv('RISK_PER_TRADE', '0.02')),
        max_positions=int(os.getenv('MAX_POSITIONS', '5')),
        min_signal_strength=float(os.getenv('MIN_SIGNAL_STRENGTH', '3.0')),
        enable_telegram_alerts=os.getenv('ENABLE_TELEGRAM', 'false').lower() == 'true',
        telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN', ''),
        telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID', '')
    )
