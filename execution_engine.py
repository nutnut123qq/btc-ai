"""Authoritative, unit-safe execution semantics for BTC research and paper replay.

The engine deliberately separates a decision from its fill.  A signal computed
from a finalized bar may fill no earlier than a later bar open.  It also makes
the price source and execution instrument explicit: shorting a spot reference
series is permitted only as a clearly labelled derivative *simulation*.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ExecutionMode(str, Enum):
    SPOT = "spot"
    SPOT_REFERENCE_DERIVATIVE_SIMULATION = "spot-reference-derivative-simulation"
    PERPETUAL = "perpetual"


class FillPolicy(str, Enum):
    NEXT_BAR_OPEN = "next-bar-open"


@dataclass(frozen=True)
class ExecutionCostSpec:
    fee_per_side_bps: float
    slippage_per_side_bps: float

    def __post_init__(self) -> None:
        if self.fee_per_side_bps < 0 or self.slippage_per_side_bps < 0:
            raise ValueError("Execution costs cannot be negative.")

    @property
    def fee_fraction(self) -> float:
        return self.fee_per_side_bps / 10_000.0

    @property
    def slippage_fraction(self) -> float:
        return self.slippage_per_side_bps / 10_000.0

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * (self.fee_per_side_bps + self.slippage_per_side_bps)

    def scaled(self, multiplier: float) -> "ExecutionCostSpec":
        if multiplier <= 0:
            raise ValueError("Cost multiplier must be positive.")
        return ExecutionCostSpec(
            fee_per_side_bps=self.fee_per_side_bps * multiplier,
            slippage_per_side_bps=self.slippage_per_side_bps * multiplier,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "feePerSideBps": self.fee_per_side_bps,
            "slippagePerSideBps": self.slippage_per_side_bps,
            "roundTripBps": self.round_trip_bps,
        }


@dataclass(frozen=True)
class ExecutionContract:
    mode: ExecutionMode = ExecutionMode.SPOT_REFERENCE_DERIVATIVE_SIMULATION
    price_source_market: str = "binance-spot-btcusdt"
    fill_policy: FillPolicy = FillPolicy.NEXT_BAR_OPEN
    max_positions: int = 1
    capital_fraction_per_trade: float = 1.0

    def __post_init__(self) -> None:
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least one.")
        if not 0 < self.capital_fraction_per_trade <= 1:
            raise ValueError("capital_fraction_per_trade must be in (0, 1].")

    def validate_side(self, side: str) -> None:
        if side not in {"long", "short"}:
            raise ValueError(f"Unsupported side: {side}")
        if side == "short" and self.mode == ExecutionMode.SPOT:
            raise ValueError("A spot execution contract cannot open a short position.")

    def to_dict(self) -> dict[str, object]:
        funding_policy = {
            ExecutionMode.SPOT: "not-applicable",
            ExecutionMode.SPOT_REFERENCE_DERIVATIVE_SIMULATION: "excluded-synthetic-simulation",
            ExecutionMode.PERPETUAL: "explicit-holding-period-pnl-required",
        }[self.mode]
        return {
            "mode": self.mode.value,
            "priceSourceMarket": self.price_source_market,
            "fillPolicy": self.fill_policy.value,
            "maxPositions": self.max_positions,
            "capitalFractionPerTrade": self.capital_fraction_per_trade,
            "fundingPolicy": funding_policy,
        }


@dataclass(frozen=True)
class DecisionGateResult:
    allowed: bool
    reason: str | None = None


def evaluate_decision_gate(
    *,
    model_available: bool,
    signal_bar_closed: bool,
    signal_available_ms: int,
    decision_ms: int,
    latest_observation_ms: int,
    max_staleness_ms: int | None,
) -> DecisionGateResult:
    """Fail closed when a decision is unavailable, premature, or stale."""
    if not model_available:
        return DecisionGateResult(False, "model-unavailable")
    if not signal_bar_closed:
        return DecisionGateResult(False, "signal-bar-not-finalized")
    if decision_ms < signal_available_ms:
        return DecisionGateResult(False, "signal-not-yet-available")
    if latest_observation_ms > decision_ms:
        return DecisionGateResult(False, "future-observation")
    if max_staleness_ms is not None:
        if max_staleness_ms < 0:
            raise ValueError("max_staleness_ms cannot be negative.")
        if decision_ms - latest_observation_ms > max_staleness_ms:
            return DecisionGateResult(False, "market-data-stale")
    return DecisionGateResult(True)


@dataclass(frozen=True)
class FillResult:
    entry_fill_price: float
    exit_fill_price: float
    gross_return: float
    net_return: float
    entry_fee_fraction: float
    exit_fee_fraction: float
    funding_pnl_fraction: float


def calculate_round_trip(
    *,
    side: Literal["long", "short"],
    entry_reference_price: float,
    exit_reference_price: float,
    costs: ExecutionCostSpec,
    contract: ExecutionContract,
    funding_pnl_fraction: float | None = None,
) -> FillResult:
    """Return PnL per unit of initial position notional.

    Fees are charged on actual entry and exit notionals rather than approximated
    by subtracting ``2 * fee`` from a price return.
    """
    contract.validate_side(side)
    if contract.mode == ExecutionMode.PERPETUAL and funding_pnl_fraction is None:
        raise ValueError("Perpetual execution requires explicit holding-period funding PnL.")
    if entry_reference_price <= 0 or exit_reference_price <= 0:
        raise ValueError("Execution prices must be positive.")

    slip = costs.slippage_fraction
    if side == "long":
        entry_fill = entry_reference_price * (1.0 + slip)
        exit_fill = exit_reference_price * (1.0 - slip)
        gross = exit_fill / entry_fill - 1.0
    else:
        entry_fill = entry_reference_price * (1.0 - slip)
        exit_fill = exit_reference_price * (1.0 + slip)
        gross = (entry_fill - exit_fill) / entry_fill

    # Quantity is defined by initial notional / entry fill.  Entry fee is
    # therefore exactly fee_fraction; exit fee varies with exit notional.
    exit_notional_fraction = exit_fill / entry_fill
    entry_fee = costs.fee_fraction
    exit_fee = exit_notional_fraction * costs.fee_fraction
    funding_pnl = float(funding_pnl_fraction or 0.0)
    return FillResult(
        entry_fill_price=entry_fill,
        exit_fill_price=exit_fill,
        gross_return=gross,
        net_return=gross - entry_fee - exit_fee + funding_pnl,
        entry_fee_fraction=entry_fee,
        exit_fee_fraction=exit_fee,
        funding_pnl_fraction=funding_pnl,
    )


@dataclass(frozen=True)
class BarrierResolution:
    triggered: bool
    exit_price: float | None
    reason: str | None
    ambiguous: bool
    lower_bound_exit_price: float | None = None
    upper_bound_exit_price: float | None = None


def resolve_tp_sl_bar(
    *,
    side: Literal["long", "short"],
    bar_high: float,
    bar_low: float,
    take_profit: float | None,
    stop_loss: float | None,
    scenario: Literal["conservative", "optimistic"] = "conservative",
) -> BarrierResolution:
    """Resolve OHLC barrier touches without fabricating intrabar ordering.

    When both TP and SL are touched the bar is marked ambiguous.  The selected
    exit is a declared bound: conservative chooses the stop, optimistic the TP.
    """
    if side not in {"long", "short"}:
        raise ValueError(f"Unsupported side: {side}")
    if bar_high < bar_low:
        raise ValueError("bar_high cannot be below bar_low.")

    if side == "long":
        tp_hit = take_profit is not None and bar_high >= take_profit
        sl_hit = stop_loss is not None and bar_low <= stop_loss
    else:
        tp_hit = take_profit is not None and bar_low <= take_profit
        sl_hit = stop_loss is not None and bar_high >= stop_loss

    if tp_hit and sl_hit:
        chosen_price = stop_loss if scenario == "conservative" else take_profit
        chosen_reason = "SL" if scenario == "conservative" else "TP"
        prices = sorted((float(stop_loss), float(take_profit)))
        return BarrierResolution(
            True,
            float(chosen_price),
            chosen_reason,
            True,
            lower_bound_exit_price=prices[0],
            upper_bound_exit_price=prices[1],
        )
    if sl_hit:
        return BarrierResolution(True, float(stop_loss), "SL", False)
    if tp_hit:
        return BarrierResolution(True, float(take_profit), "TP", False)
    return BarrierResolution(False, None, None, False)
