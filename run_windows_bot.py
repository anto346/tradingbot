#!/usr/bin/env python3
"""
Windows-Compatible Trading Bot Launcher
"""

import sys
import os

# Fix Windows encoding issues
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul 2>&1')

# Import the fixed bot
try:
    from enhanced_trading_bot_fixed import run_quick_test, main
    
    print("Windows-Compatible Trading Bot")
    print("=" * 40)
    
    choice = input("1. Quick Test\n2. Full Demo\nChoice (1-2): ")
    
    if choice == "1":
        print("\nRunning quick test...")
        run_quick_test()
    elif choice == "2":
        print("\nStarting demo mode...")
        main()
    else:
        print("Invalid choice")
        
except Exception as e:
    print(f"Error: {e}")
    input("Press Enter to exit...")
