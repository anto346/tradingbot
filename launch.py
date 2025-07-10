#!/usr/bin/env python3
"""
Complete launch script for Enhanced Trading Bot
"""

import os
import sys

def print_banner():
    """Print welcome banner"""
    print("🚀" + "="*58 + "🚀")
    print("   ENHANCED ZERODHA TRADING BOT v2.0")
    print("   Advanced Algorithmic Trading System")
    print("🚀" + "="*58 + "🚀")

def check_setup():
    """Check if setup is complete"""
    required_files = [
        "enhanced_trading_bot.py",
        "config.py", 
        "requirements.txt",
        ".env"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nRun: python setup.py")
        return False
    
    return True

def show_menu():
    """Show main menu"""
    print("\n📋 SELECT OPTION:")
    print("1. 🧪 Quick Test (Demo)")
    print("2. 📊 Run Backtest")
    print("3. 🎮 Demo Trading")
    print("4. 📈 Paper Trading")
    print("5. 💰 Live Trading")
    print("6. ⚙️ Setup/Install")
    print("0. ❌ Exit")
    
    choice = input("\nEnter choice (0-6): ").strip()
    return choice

def run_option(choice):
    """Run selected option"""
    if choice == "1":
        print("\n🧪 Running Quick Test...")
        from enhanced_trading_bot import run_quick_test
        run_quick_test()
    
    elif choice == "2":
        print("\n📊 Running Backtest...")
        from backtester import run_simple_backtest
        run_simple_backtest()
    
    elif choice == "3":
        print("\n🎮 Starting Demo Trading...")
        os.environ['TRADING_MODE'] = 'demo'
        from enhanced_trading_bot import main
        main()
    
    elif choice == "4":
        print("\n📈 Starting Paper Trading...")
        os.environ['TRADING_MODE'] = 'paper'
        from enhanced_trading_bot import main
        main()
    
    elif choice == "5":
        print("\n💰 LIVE TRADING MODE!")
        print("⚠️  This uses real money!")
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm == 'CONFIRM':
            os.environ['TRADING_MODE'] = 'live'
            from enhanced_trading_bot import main
            main()
        else:
            print("❌ Live trading cancelled")
    
    elif choice == "6":
        print("\n⚙️ Running Setup...")
        os.system("python setup.py")
    
    elif choice == "0":
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice")

def main():
    """Main launcher function"""
    print_banner()
    
    if not check_setup():
        print("\n⚙️ Run setup first!")
        return
    
    try:
        while True:
            choice = show_menu()
            run_option(choice)
            
            if choice != "0":
                input("\n⏸️  Press Enter to continue...")
                
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
