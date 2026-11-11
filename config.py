import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("BYBIT_API_KEY", "")
    API_SECRET = os.getenv("BYBIT_API_SECRET", "")
    BYBIT_DOMAIN = os.getenv("BYBIT_DOMAIN", "bybit")
    BYBIT_TLD = os.getenv("BYBIT_TLD", "eu")
    DEMO_MODE = os.getenv("BYBIT_DEMO_MODE", "true").lower() in {"1", "true", "yes"}
    BASE_ASSET = os.getenv("BASE_ASSET", "BTC").upper()
    QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDC").upper()
    SYMBOL = os.getenv("BYBIT_SYMBOL", f"{BASE_ASSET}{QUOTE_ASSET}")
    EURC_USDC_SYMBOL = os.getenv("EURC_USDC_SYMBOL", "EURCUSDC")
    CATEGORY = "spot"
    TIMEFRAME = "60"
    BACKTEST_CAPITAL_EUR = float(os.getenv("BACKTEST_CAPITAL_EUR", "1000"))
    BACKTEST_CAPITAL_USDC = float(os.getenv("BACKTEST_CAPITAL_USDC", "1000"))
    MIN_ORDER_NOTIONAL_EUR = float(os.getenv("MIN_ORDER_NOTIONAL_EUR", "50"))
    MAX_ORDER_NOTIONAL_EUR = float(os.getenv("MAX_ORDER_NOTIONAL_EUR", "1000"))
    MIN_ORDER_NOTIONAL_USDC = float(os.getenv("MIN_ORDER_NOTIONAL_USDC", "50"))
    MAX_ORDER_NOTIONAL_USDC = float(os.getenv("MAX_ORDER_NOTIONAL_USDC", "1000"))
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
    ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "1.5"))
    MIN_STOP_PCT = float(os.getenv("MIN_STOP_PCT", "0.015"))
    MAX_STOP_PCT = float(os.getenv("MAX_STOP_PCT", "0.035"))
    RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "2.0"))
    VOLATILITY_TARGET_PCT = float(os.getenv("VOLATILITY_TARGET_PCT", "1.0"))
    EMA200_TREND_FILTER = os.getenv("EMA200_TREND_FILTER", "true").lower() in {"1", "true", "yes"}
    FEE_RATE = float(os.getenv("FEE_RATE", "0.001"))
    SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.0005"))
    MAX_HOLDING_HOURS = float(os.getenv("MAX_HOLDING_HOURS", "32"))
    DEFAULT_SL_PCT = float(os.getenv("DEFAULT_SL_PCT", "0.018"))
    DEFAULT_TP_PCT = float(os.getenv("DEFAULT_TP_PCT", "0.038"))
    TRAIL_TIER1_TRIGGER = float(os.getenv("TRAIL_TIER1_TRIGGER", "0.014"))
    TRAIL_TIER1_LOCK = float(os.getenv("TRAIL_TIER1_LOCK", "0.0080"))
    TRAIL_TIER2_TRIGGER = float(os.getenv("TRAIL_TIER2_TRIGGER", "0.020"))
    TRAIL_TIER2_LOCK = float(os.getenv("TRAIL_TIER2_LOCK", "0.0150"))
    MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "15.0"))
    RL_ACTIVE = os.getenv("RL_ACTIVE", "true").lower() in {"1", "true", "yes"}
    RL_MIN_STATE_WR = float(os.getenv("RL_MIN_STATE_WR", "0.55"))
    RL_MAX_CONSECUTIVE_ERRORS = int(os.getenv("RL_MAX_CONSECUTIVE_ERRORS", "2"))
    RL_LEARNING_RATE = float(os.getenv("RL_LEARNING_RATE", "0.25"))
    RL_DYNAMIC_SIZING = os.getenv("RL_DYNAMIC_SIZING", "true").lower() in {"1", "true", "yes"}
    RL_USE_R_MULTIPLES = os.getenv("RL_USE_R_MULTIPLES", "true").lower() in {"1", "true", "yes"}
    FUNDING_RATE_FILTER = os.getenv("FUNDING_RATE_FILTER", "true").lower() in {"1", "true", "yes"}
    FUNDING_RATE_MAX_THRESHOLD = float(os.getenv("FUNDING_RATE_MAX_THRESHOLD", "0.00035"))
    PULLBACK_MIN_VOL_RATIO = float(os.getenv("PULLBACK_MIN_VOL_RATIO", "0.90"))

    # Tri-Engine Multi-Regime Strategy Upgrade Configuration
    TRI_ENGINE_MODE = os.getenv("TRI_ENGINE_MODE", "true").lower() in {"1", "true", "yes"}
    PULLBACK_RSI_MAX = float(os.getenv("PULLBACK_RSI_MAX", "48.0"))
    PULLBACK_RSI7_MAX = float(os.getenv("PULLBACK_RSI7_MAX", "38.0"))
    BREAKOUT_MIN_SCORE = int(os.getenv("BREAKOUT_MIN_SCORE", "2"))
    BREAKOUT_MIN_VOL_RATIO = float(os.getenv("BREAKOUT_MIN_VOL_RATIO", "1.15"))
    BREAKOUT_RSI_MIN = float(os.getenv("BREAKOUT_RSI_MIN", "52.0"))
    BREAKOUT_RSI_MAX = float(os.getenv("BREAKOUT_RSI_MAX", "76.0"))

    # Setup 3: Mean-Reversion & Bear/Range Bounce (Multi-bar rolling window & Range Market ADX < 24)
    MEAN_REV_SL_PCT = float(os.getenv("MEAN_REV_SL_PCT", "0.018"))
    MEAN_REV_TP_PCT = float(os.getenv("MEAN_REV_TP_PCT", "0.024"))
    MEAN_REV_MAX_HOLDING_HOURS = float(os.getenv("MEAN_REV_MAX_HOLDING_HOURS", "12.0"))
    MEAN_REV_RSI_MAX = float(os.getenv("MEAN_REV_RSI_MAX", "40.0"))
    MEAN_REV_RSI7_MAX = float(os.getenv("MEAN_REV_RSI7_MAX", "28.0"))
    MEAN_REV_DIST_EMA20_MIN = float(os.getenv("MEAN_REV_DIST_EMA20_MIN", "-1.25"))

    # RL Mistake Memory Half-Life Decay & Compounding
    RL_DECAY_HALF_LIFE_BARS = int(os.getenv("RL_DECAY_HALF_LIFE_BARS", "72"))
    RL_PROBED_SIZING_MULTIPLIER = float(os.getenv("RL_PROBED_SIZING_MULTIPLIER", "0.45"))
    RL_COMPOUNDING = os.getenv("RL_COMPOUNDING", "true").lower() in {"1", "true", "yes"}
    BASE_POSITION_PCT = float(os.getenv("BASE_POSITION_PCT", "0.75"))

    # Market Regime Detection & Choppiness Filter (Optional, zero collision by default)
    REGIME_FILTER_ACTIVE = os.getenv("REGIME_FILTER_ACTIVE", "false").lower() in {"1", "true", "yes"}
    ADX_MIN_TREND = float(os.getenv("ADX_MIN_TREND", "22.0"))
    CHOP_MAX_TREND = float(os.getenv("CHOP_MAX_TREND", "61.8"))

    # Backtest Parity with Live Circuit Breakers & Anti-Whipsaw (Optional)
    COOLDOWN_BARS_AFTER_LOSS = int(os.getenv("COOLDOWN_BARS_AFTER_LOSS", "0"))
    CIRCUIT_BREAKER_BACKTEST = os.getenv("CIRCUIT_BREAKER_BACKTEST", "false").lower() in {"1", "true", "yes"}
    DYNAMIC_SLIPPAGE_ENABLED = os.getenv("DYNAMIC_SLIPPAGE_ENABLED", "false").lower() in {"1", "true", "yes"}

    PAPER_INITIAL_CAPITAL_USDC = float(os.getenv("PAPER_INITIAL_CAPITAL_USDC", "1000"))
    PAPER_DB_PATH = os.getenv("PAPER_DB_PATH", "data/paper_trading_demo.sqlite3")
    PAPER_POLL_SECONDS = float(os.getenv("PAPER_POLL_SECONDS", "15"))
    PAPER_NEWS_POLL_SECONDS = float(os.getenv("PAPER_NEWS_POLL_SECONDS", "300"))
    PAPER_DAILY_LOSS_LIMIT_PCT = float(os.getenv("PAPER_DAILY_LOSS_LIMIT_PCT", "0.035"))
    PAPER_DAILY_LOSS_LIMIT_MIN_USDC = float(os.getenv("PAPER_DAILY_LOSS_LIMIT_MIN_USDC", "20.0"))
    PAPER_DAILY_LOSS_LIMIT_USDC = float(os.getenv("PAPER_DAILY_LOSS_LIMIT_USDC", "35.0"))
    PAPER_MAX_CONSECUTIVE_LOSSES = int(os.getenv("PAPER_MAX_CONSECUTIVE_LOSSES", "2"))
    PAPER_REVIEW_MIN_TRADES = int(os.getenv("PAPER_REVIEW_MIN_TRADES", "30"))
    LOCAL_TIMEZONE = os.getenv("LOCAL_TIMEZONE", "Europe/Rome")
    DATA_DIR = "data"
