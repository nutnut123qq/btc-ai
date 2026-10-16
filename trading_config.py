"""Centralized quantitative trading configuration for the production market scope."""

from typing import Any, Dict, List

# Transaction Costs (basis points: 1 bps = 0.01% = 0.0001)
FEE_BPS: float = 10.0          # Standard exchange taker fee (0.10%)
SLIPPAGE_BPS: float = 5.0      # Average market slippage for major pairs (0.05%)
TOTAL_COST_PER_SIDE_BPS: float = FEE_BPS + SLIPPAGE_BPS  # 15.0 bps (0.15%)
TOTAL_ROUNDTRIP_COST_FRACTION: float = (TOTAL_COST_PER_SIDE_BPS * 2.0) / 10_000.0  # 0.0030 = 0.30%

# Supported market scope. Keep this policy centralized so scheduled scripts cannot
# silently recreate data that a database cleanup has removed.
DEFAULT_SYMBOL: str = "BTCUSDT"
ACTIVE_SYMBOLS: List[str] = [DEFAULT_SYMBOL]
ACTIVE_PRODUCTION_TIMEFRAMES: tuple[str, ...] = ("1h", "4h", "1d")
DEFAULT_TIMEFRAME: str = "4h"
# Backwards-compatible name for callers that expose production choices.
SUPPORTED_TIMEFRAMES: List[str] = list(ACTIVE_PRODUCTION_TIMEFRAMES)
SUPPORTED_HORIZONS: List[str] = ["1h", "4h", "1d", "3d", "7d"]

# Instrument registry — mirrors backend InstrumentCatalog (contract instrument-catalog-v1).
# Membership means the instrument has a declared contract; ACTIVE_SYMBOLS above remains
# the enabled production data scope, so crypto-only scripts (futures/liquidation/
# sentiment collectors, paper trader) stay BTCUSDT-only until the MBB lanes land.
# MBB: HOSE equity, VND, Asia/Ho_Chi_Minh, VCI primary + KBS cross-check, adjusted
# price basis, production timeframes exactly 1h and 1d (no synthetic 4h).
INSTRUMENTS: Dict[str, Dict[str, Any]] = {
    "BTCUSDT": {
        "symbol": "BTCUSDT",
        "label": "BTC/USDT",
        "venue": "Binance Spot",
        "asset_class": "crypto",
        "currency": "USDT",
        "timezone": "UTC",
        "data_provider": "Binance",
        "cross_check_provider": None,
        "price_basis": "raw",
        "aliases": ("BTC",),
        "timeframes": ACTIVE_PRODUCTION_TIMEFRAMES,
        "default_timeframe": DEFAULT_TIMEFRAME,
        "enabled": True,
        "capabilities": {
            "technical_analysis": True,
            "news": True,
            "sentiment": True,
            "futures": True,
            "liquidation": True,
            "order_book": True,
            "execution": True,
            "paper_trading": True,
        },
    },
    "MBB": {
        "symbol": "MBB",
        "label": "MBB (MB Bank)",
        "venue": "HOSE",
        "asset_class": "equity",
        "currency": "VND",
        "timezone": "Asia/Ho_Chi_Minh",
        "data_provider": "VCI",
        "cross_check_provider": "KBS",
        "price_basis": "adjusted",
        "aliases": (),
        "timeframes": ("1h", "1d"),
        "default_timeframe": "1d",
        "enabled": False,
        "capabilities": {
            "technical_analysis": True,
            "news": False,
            "sentiment": False,
            "futures": False,
            "liquidation": False,
            "order_book": False,
            "execution": False,
            "paper_trading": False,
        },
    },
}
KNOWN_SYMBOLS: tuple[str, ...] = tuple(INSTRUMENTS)

# Dataset Partitioning & ML Validation
TRAIN_TEST_SPLIT_RATIO: float = 0.80
SPLIT_TIMESTAMP_MS: int = 1735689600000  # 2025-01-01T00:00:00Z

# Default Direction Probability Thresholds (Empirically Calibrated)
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.58
TIMEFRAME_THRESHOLDS: Dict[str, float] = {
    "4h": 0.61,
    "1h": 0.58,
}

# Per-Asset Calibrated Thresholds for 4h (Sniper Calibration & Noise Filtering)
ASSET_4H_THRESHOLDS: Dict[str, float] = {
    "BTCUSDT": 0.61,
}

# Asset-Specific Dynamic ATR Multipliers (Wick Buffers)
ASSET_ATR_SETTINGS: Dict[str, Dict[str, float]] = {
    "BTCUSDT": {"tp_mult": 1.5, "sl_mult": 1.0},
}


def canonicalize_symbol(symbol: str) -> str:
    """Uppercase/trim and resolve declared aliases (BTC -> BTCUSDT)."""
    normalized = (symbol or "").strip().upper()
    for descriptor in INSTRUMENTS.values():
        if normalized == descriptor["symbol"] or normalized in descriptor["aliases"]:
            return descriptor["symbol"]
    return normalized


def require_known_symbol(symbol: str) -> str:
    """Reject symbols outside the shared instrument registry (fail closed)."""
    canonical = canonicalize_symbol(symbol)
    if canonical not in INSTRUMENTS:
        raise ValueError(
            f"Unknown symbol '{canonical or symbol}'. Known instruments: {', '.join(INSTRUMENTS)}."
        )
    return canonical


def require_active_symbol(symbol: str) -> str:
    """Normalize and reject symbols outside the enabled production data scope."""
    normalized = canonicalize_symbol(symbol)
    if normalized not in ACTIVE_SYMBOLS:
        raise ValueError(
            f"Unsupported symbol '{normalized or symbol}'. Allowed: {', '.join(ACTIVE_SYMBOLS)}."
        )
    return normalized


def require_active_symbols(symbols: List[str]) -> List[str]:
    """Validate a CLI symbol list and reject empty lists."""
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return [require_active_symbol(symbol) for symbol in symbols]


def require_active_timeframe(timeframe: str, symbol: str = DEFAULT_SYMBOL) -> str:
    """Reject candle timeframes outside the instrument's declared contract.

    Existing callers omit ``symbol`` and keep the BTCUSDT contract (1h/4h/1d).
    MBB declares exactly 1h and 1d; its 4h requests must fail closed.

    This is a catalog check only: it does not mean a known-but-disabled symbol is
    enabled for production traffic. Callers that gate live pipelines must also use
    ``require_active_symbol``.
    """
    canonical = require_known_symbol(symbol)
    normalized = timeframe.strip().lower()
    allowed = INSTRUMENTS[canonical]["timeframes"]
    if normalized not in allowed:
        raise ValueError(
            f"Inactive production timeframe '{normalized or timeframe}' for {canonical}. "
            f"Allowed: {', '.join(allowed)}."
        )
    return normalized


def active_timeframes_for(symbol: str) -> tuple[str, ...]:
    """Declared production timeframes for a known instrument."""
    return INSTRUMENTS[require_known_symbol(symbol)]["timeframes"]


def default_timeframe_for(symbol: str) -> str:
    """Declared default timeframe for a known instrument (MBB: 1d, BTCUSDT: 4h)."""
    return INSTRUMENTS[require_known_symbol(symbol)]["default_timeframe"]

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
