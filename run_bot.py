#!/usr/bin/env python3
"""
Simple script to run the Enhanced Trading Bot
"""

import sys
import os

def main():
    """Main runner function"""
    print("🚀 Enhanced Trading Bot Launcher")
    print("=" * 50)
    
    # Check for mode argument
    mode = "demo"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ['demo', 'paper', 'live']:
            print("❌ Invalid mode. Use: demo, paper, or live")
            return
    
    print(f"🎯 Starting in {mode.upper()} mode...")
    
    # Set environment variable
    os.environ['TRADING_MODE'] = mode
    
    # Safety check for live mode
    if mode == 'live':
        print("\n⚠️  LIVE TRADING MODE!")
        print("This will use real money. Are you sure?")
        confirm = input("Type 'YES' to continue: ")
        if confirm != 'YES':
            print("❌ Cancelled. Use 'paper' mode for testing.")
            return
    
    # Import and run
    try:
        from enhanced_trading_bot import main as bot_main
        bot_main()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Make sure enhanced_trading_bot.py is in the same directory")
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")

if __name__ == "__main__":
    main()
