#!/usr/bin/env python3
"""
Setup script for Enhanced Trading Bot
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages"""
    print("📦 Installing packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed!")
        return True
    except Exception as e:
        print(f"❌ Error installing packages: {e}")
        return False

def setup_env():
    """Setup environment file"""
    if not os.path.exists('.env'):
        try:
            with open('.env.example', 'r') as f:
                content = f.read()
            with open('.env', 'w') as f:
                f.write(content)
            print("✅ Created .env file")
        except Exception as e:
            print(f"❌ Error creating .env: {e}")
    else:
        print("✅ .env file exists")

def main():
    """Setup function"""
    print("🚀 Enhanced Trading Bot Setup")
    print("=" * 40)
    
    # Install packages
    if not install_packages():
        return
    
    # Setup environment
    setup_env()
    
    print("\n" + "=" * 40)
    print("✅ Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit .env with your API credentials")
    print("2. Run: python run_bot.py demo")
    print("3. Test thoroughly before live trading")

if __name__ == "__main__":
    main()
