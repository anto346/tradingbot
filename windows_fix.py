#!/usr/bin/env python3
"""
Windows Unicode Fix for Enhanced Trading Bot
This fixes the emoji encoding issues on Windows systems
"""

import logging
import sys
import os

def fix_windows_console():
    """Fix Windows console encoding for Unicode support"""
    if sys.platform.startswith('win'):
        try:
            # Try to set UTF-8 encoding for Windows console
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
            
            # Set console code page to UTF-8
            os.system('chcp 65001 >nul 2>&1')
            
        except Exception:
            # If UTF-8 fails, we'll fall back to safe logging
            pass

class SafeFormatter(logging.Formatter):
    """Safe formatter that handles Unicode issues on Windows"""
    
    def format(self, record):
        # Get the original formatted message
        formatted = super().format(record)
        
        # If we're on Windows and have encoding issues, remove emojis
        if sys.platform.startswith('win'):
            try:
                # Test if the message can be encoded
                formatted.encode('cp1252')
                return formatted
            except UnicodeEncodeError:
                # Remove common emojis and Unicode characters
                emoji_map = {
                    '🚀': '[START]',
                    '🔍': '[SCAN]',
                    '📊': '[ANALYSIS]',
                    '💰': '[MONEY]',
                    '📈': '[UP]',
                    '📉': '[DOWN]',
                    '✅': '[OK]',
                    '❌': '[ERROR]',
                    '⚠️': '[WARNING]',
                    '🟢': '[GREEN]',
                    '🔴': '[RED]',
                    '🎯': '[TARGET]',
                    '🛡️': '[STOP]',
                    '⏰': '[TIME]',
                    '🕒': '[CLOCK]',
                    '🎮': '[DEMO]',
                    '💹': '[PRICE]',
                    '📋': '[LIST]',
                    '🧪': '[TEST]',
                    '⚖️': '[BALANCE]',
                    '📅': '[DATE]',
                    '🔧': '[CONFIG]',
                    '🌊': '[MARKET]',
                    '⚡': '[VOLATILITY]',
                    '🎯': '[OPPORTUNITY]',
                    '🛑': '[STOP]',
                    '📦': '[PACKAGE]',
                    '🔥': '[TRADE]'
                }
                
                # Replace emojis with safe text
                safe_formatted = formatted
                for emoji, replacement in emoji_map.items():
                    safe_formatted = safe_formatted.replace(emoji, replacement)
                
                # Remove any remaining Unicode characters that might cause issues
                safe_formatted = safe_formatted.encode('ascii', 'ignore').decode('ascii')
                
                return safe_formatted
        
        return formatted

def setup_safe_logging():
    """Setup logging that works on Windows with Unicode issues"""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create safe formatter
    formatter = SafeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler (always safe)
    from datetime import datetime
    log_filename = f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
    
    try:
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # Fallback to basic file handler
        try:
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass  # Skip file logging if it fails
    
    # Console handler with safe encoding
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    except Exception:
        pass  # Skip console logging if it fails
    
    return logger

# Enhanced Trading Bot with Windows fixes
class WindowsCompatibleBot:
    """Windows-compatible version of the trading bot"""
    
    def __init__(self, api_key=None, access_token=None, config=None):
        """Initialize with Windows compatibility fixes"""
        
        # Fix Windows console first
        fix_windows_console()
        
        # Setup safe logging
        self.logger = setup_safe_logging()
        
        # Import the original bot components
        try:
            from enhanced_trading_bot import TradingConfig, EnhancedZerodhaBot
            
            self.config = config or TradingConfig()
            
            # Initialize the original bot but override logging
            self.original_bot = EnhancedZerodhaBot(api_key, access_token, config)
            self.original_bot.logger = self.logger
            
            # Copy all methods from original bot
            for attr_name in dir(self.original_bot):
                if not attr_name.startswith('_') and not hasattr(self, attr_name):
                    setattr(self, attr_name, getattr(self.original_bot, attr_name))
            
            # Use safe console output
            self.safe_print("BOT INITIALIZED - Windows compatibility mode enabled")
            
        except Exception as e:
            print(f"Error initializing bot: {e}")
    
    def safe_print(self, message):
        """Safe print function that handles Windows encoding"""
        try:
            print(message)
        except UnicodeEncodeError:
            # Remove emojis and Unicode characters for Windows
            safe_message = message
            emoji_replacements = {
                '🚀': '[START]', '🔍': '[SCAN]', '📊': '[ANALYSIS]',
                '💰': '[MONEY]', '📈': '[UP]', '📉': '[DOWN]',
                '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARNING]',
                '🟢': '[GREEN]', '🔴': '[RED]', '🎯': '[TARGET]',
                '🛡️': '[STOP]', '⏰': '[TIME]', '🕒': '[CLOCK]'
            }
            
            for emoji, replacement in emoji_replacements.items():
                safe_message = safe_message.replace(emoji, replacement)
            
            # Encode to ASCII, ignoring problematic characters
            safe_message = safe_message.encode('ascii', 'ignore').decode('ascii')
            print(safe_message)
    
    def generate_report(self, opportunities):
        """Windows-safe report generation"""
        try:
            self.safe_print("\n" + "="*80)
            self.safe_print("ENHANCED TRADING BOT - MARKET ANALYSIS REPORT")
            self.safe_print("="*80)
            
            from datetime import datetime
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            account_balance = self.get_account_balance()
            
            self.safe_print(f"Date & Time: {current_time}")
            self.safe_print(f"Account Balance: Rs.{account_balance:,.2f}")
            self.safe_print(f"Active Positions: {len(self.positions)}")
            self.safe_print(f"Opportunities Found: {len(opportunities)}")
            
            # Show opportunities with safe printing
            if opportunities:
                self.safe_print(f"\nTRADING OPPORTUNITIES ({len(opportunities)}):")
                self.safe_print("-" * 80)
                
                for i, opp in enumerate(opportunities, 1):
                    signal_data = opp['signal_data']
                    self.safe_print(f"\n{i}. {opp['symbol']} - {opp['action']} Signal")
                    self.safe_print(f"   Price: Rs.{opp['current_price']:.2f}")
                    self.safe_print(f"   Strength: {signal_data['score']:.1f}/10")
                    self.safe_print(f"   Confidence: {signal_data['confidence']:.1f}%")
                    
                    if 'target' in signal_data:
                        self.safe_print(f"   Target: Rs.{signal_data['target']:.2f}")
                    if 'stop_loss' in signal_data:
                        self.safe_print(f"   Stop Loss: Rs.{signal_data['stop_loss']:.2f}")
                    
                    self.safe_print(f"   Signals: {', '.join(signal_data['signals'][:3])}")
            else:
                self.safe_print("\nNo trading opportunities found at this time.")
            
            self.safe_print("\n" + "="*80)
            
        except Exception as e:
            self.safe_print(f"Error generating report: {e}")

def run_windows_compatible_test():
    """Run test with Windows compatibility"""
    print("WINDOWS COMPATIBLE TRADING BOT TEST")
    print("=" * 50)
    
    try:
        # Import config
        from enhanced_trading_bot import TradingConfig
        
        # Create Windows-compatible bot
        config = TradingConfig(demo_mode=True, paper_trading=True)
        bot = WindowsCompatibleBot(config=config)
        
        # Test watchlist
        test_watchlist = ["RELIANCE", "TCS", "INFY"]
        
        print(f"Testing {len(test_watchlist)} stocks...")
        
        # Quick scan with safe output
        opportunities = bot.scan_market(test_watchlist)
        
        # Generate Windows-safe report
        bot.generate_report(opportunities)
        
        print("\nTest completed successfully!")
        return bot, opportunities
        
    except Exception as e:
        print(f"Error in test: {e}")
        return None, []

# Quick fix script
def apply_quick_fix():
    """Apply quick fix to existing bot files"""
    
    print("APPLYING WINDOWS UNICODE FIX...")
    print("=" * 40)
    
    # Create a patched version of the enhanced_trading_bot.py
    try:
        # Read the original file
        with open('enhanced_trading_bot.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add Windows compatibility import at the top
        fixed_content = '''# Windows Unicode Fix - Added automatically
import sys
import os

# Fix Windows console encoding
if sys.platform.startswith('win'):
    try:
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass

''' + content
        
        # Replace the setup_logging method with safe version
        setup_logging_replacement = '''    def setup_logging(self):
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
        self.logger.error = lambda msg: safe_log(original_error, msg)'''
        
        # Replace the original setup_logging method
        import re
        pattern = r'    def setup_logging\(self\):.*?self\.logger = logging\.getLogger\(self\.__class__\.__name__\)'
        
        fixed_content = re.sub(pattern, setup_logging_replacement, fixed_content, flags=re.DOTALL)
        
        # Save the fixed version
        with open('enhanced_trading_bot_fixed.py', 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print("✅ Created enhanced_trading_bot_fixed.py with Windows compatibility")
        
        # Also create a simple launcher
        launcher_content = '''#!/usr/bin/env python3
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
    
    choice = input("1. Quick Test\\n2. Full Demo\\nChoice (1-2): ")
    
    if choice == "1":
        print("\\nRunning quick test...")
        run_quick_test()
    elif choice == "2":
        print("\\nStarting demo mode...")
        main()
    else:
        print("Invalid choice")
        
except Exception as e:
    print(f"Error: {e}")
    input("Press Enter to exit...")
'''
        
        with open('run_windows_bot.py', 'w', encoding='utf-8') as f:
            f.write(launcher_content)
        
        print("✅ Created run_windows_bot.py launcher")
        print("\n🚀 QUICK START:")
        print("python run_windows_bot.py")
        
    except Exception as e:
        print(f"❌ Error applying fix: {e}")

if __name__ == "__main__":
    choice = input("1. Run Windows Test\n2. Apply Quick Fix\nChoice (1-2): ")
    
    if choice == "1":
        run_windows_compatible_test()
    elif choice == "2":
        apply_quick_fix()
    else:
        print("Invalid choice")