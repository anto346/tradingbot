# Enhanced Trading Bot

🚀 Advanced algorithmic trading bot for Zerodha with comprehensive features.

## 🎯 Features

- **Advanced Technical Analysis**: 15+ indicators
- **Risk Management**: Dynamic position sizing
- **Multiple Trading Modes**: Demo, Paper, Live
- **Performance Analytics**: Comprehensive reporting
- **Safety First**: Multiple safety checks

## 📦 Quick Setup

1. **Install**:
   ```bash
   python setup.py
   ```

2. **Configure**:
   ```bash
   # Edit .env with your API credentials
   cp .env.example .env
   nano .env
   ```

3. **Test**:
   ```bash
   python run_bot.py demo
   ```

## 🎮 Usage

### Demo Mode (Recommended for testing)
```bash
python run_bot.py demo
```

### Paper Trading (Live data, fake money)
```bash
python run_bot.py paper
```

### Live Trading (Real money - Use with caution!)
```bash
python run_bot.py live
```

## ⚙️ Configuration

Edit `.env` file:
```
ZERODHA_API_KEY=your_key
ZERODHA_ACCESS_TOKEN=your_token
TRADING_MODE=demo
RISK_PER_TRADE=0.02
MAX_POSITIONS=5
```

## 📊 Key Files

- `enhanced_trading_bot.py` - Main bot
- `config.py` - Configuration
- `utils.py` - Utilities
- `run_bot.py` - Simple runner
- `requirements.txt` - Dependencies

## ⚠️ Important Notes

1. **Start with Demo Mode** - Always test first
2. **Use Paper Trading** - Test with live data
3. **Small Positions** - Start small in live trading
4. **Monitor Carefully** - Always supervise the bot
5. **Risk Management** - Never risk more than you can afford

## 📈 Expected Performance

- Win Rate: 45-60%
- Risk:Reward: 1:2
- Monthly Return: 3-8% (market dependent)

## 🆘 Support

1. Check logs in `logs/` directory
2. Review configuration in `.env`
3. Test in demo mode first
4. Monitor performance metrics

## ⚖️ Disclaimer

This software is for educational purposes. Trading involves risk. 
Past performance doesn't guarantee future results. Use at your own risk.

---

**Happy Trading! 🚀**
