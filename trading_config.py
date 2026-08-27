"""
Centralized quantitative trading configuration, fee structure, slippage models,
and data partitioning constants for the Bitcoin & Multi-Asset AI Analyst platform.
"""

from typing import Dict, List

# Transaction Costs (basis points: 1 bps = 0.01% = 0.0001)
FEE_BPS: float = 10.0          # Standard exchange taker fee (0.10%)
SLIPPAGE_BPS: float = 5.0      # Average market slippage for major pairs (0.05%)
TOTAL_COST_PER_SIDE_BPS: float = FEE_BPS + SLIPPAGE_BPS  # 15.0 bps (0.15%)
TOTAL_ROUNDTRIP_COST_PCT: float = (TOTAL_COST_PER_SIDE_BPS * 2.0) / 10_000.0  # 0.0030 (0.30%)

# Default Market Symbols & Timeframes
DEFAULT_SYMBOL: str = "BTCUSDT"
ACTIVE_SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SUPPORTED_TIMEFRAMES: List[str] = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
SUPPORTED_HORIZONS: List[str] = ["1h", "4h", "1d", "3d", "7d"]

# Dataset Partitioning & ML Validation
TRAIN_TEST_SPLIT_RATIO: float = 0.80
SPLIT_TIMESTAMP_MS: int = 1735689600000  # 2025-01-01T00:00:00Z

# Default Direction Probability Thresholds (Empirically Calibrated)
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.58
TIMEFRAME_THRESHOLDS: Dict[str, float] = {
    "4h": 0.61,
    "1h": 0.58,
    "30m": 0.56,
    "15m": 0.55,
}

# Per-Asset Calibrated Thresholds for 4h (Sniper Calibration & Noise Filtering)
ASSET_4H_THRESHOLDS: Dict[str, float] = {
    "BTCUSDT": 0.61,   # Baseline Champion
    "ETHUSDT": 0.58,   # Sniper Mode (Filters ~90% noise, pushes Win Rate to ~66.7%)
    "SOLUSDT": 0.55,   # High-Conviction Mode
}

# Asset-Specific Dynamic ATR Multipliers (Wick Buffers)
ASSET_ATR_SETTINGS: Dict[str, Dict[str, float]] = {
    "BTCUSDT": {"tp_mult": 1.5, "sl_mult": 1.0},
    "ETHUSDT": {"tp_mult": 1.8, "sl_mult": 1.2},  # Increased buffer against wicks
    "SOLUSDT": {"tp_mult": 2.2, "sl_mult": 1.5},  # High volatility expansion buffer
}

# Kelly Criterion Constraints (Quarter-Kelly Portfolio Sizing)
MAX_KELLY_FRACTION: float = 0.25   # Max 25% NAV allocation per trade
MIN_KELLY_FRACTION: float = 0.05   # Min 5% NAV allocation per trade
KELLY_SAFETY_FACTOR: float = 0.25  # Quarter-Kelly safety multiplier (1/4 f*)

# Dynamic ATR Trailing Stop Settings
TRAILING_STOP_ATR_TRIGGER: float = 1.0  # Profit threshold (+1.0x ATR) to activate trailing stop
TRAILING_STOP_ATR_DIST: float = 1.0     # Trailing distance (1.0x ATR) behind peak/trough

# --- Live Paper Trading Worker Configuration ---
INITIAL_BALANCE_USDT: float = 10_000.0
POSITION_SIZE_PCT: float = 0.20        # Default fallback position size
MAX_CONCURRENT_PER_SYMBOL: int = 1     # 1 open trade per symbol at any time
ATR_TP_MULTIPLIER: float = 1.5         # Take Profit = entry ± 1.5 × ATR(14)
ATR_SL_MULTIPLIER: float = 1.0         # Stop Loss   = entry ∓ 1.0 × ATR(14)
MAX_HOLD_BARS: int = 6                 # 6 × 4h = 24 hours max holding period
BACKEND_BASE_URL: str = "http://localhost:5197"

# --- Regime-Adaptive Multi-Strategy Configuration ---
REGIME_ADX_THRESHOLD: float = 25.0
MEAN_REVERSION_POSITION_PCT: float = 0.08  # 8% NAV fixed allocation
MEAN_REVERSION_RSI_OVERSOLD: float = 30.0
MEAN_REVERSION_RSI_OVERBOUGHT: float = 70.0
MEAN_REVERSION_TP_ATR_MULT: float = 1.0
MEAN_REVERSION_SL_ATR_MULT: float = 1.0
TREND_MOMENTUM_TP_ATR_MULT: float = 2.0
TREND_MOMENTUM_SL_ATR_MULT: float = 1.5

# --- Portfolio Circuit Breaker & Risk Limits ---
MAX_DAILY_DRAWDOWN: float = 0.04              # 4.0% max loss in rolling 24h window
VOLATILITY_HALT_THRESHOLD: float = 0.08       # 8.0% candle amplitude / body on 4h BTC bar
MAX_CONSECUTIVE_LOSSES: int = 4               # Max consecutive losing trades
CIRCUIT_BREAKER_COOLDOWN_HOURS: int = 24      # Cooldown duration after breaker trigger
ALTCOIN_VOLATILITY_HALT_BARS: int = 2         # Altcoin entry halt (2 bars = 8h) on BTC flash crash


