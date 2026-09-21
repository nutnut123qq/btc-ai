"""Shared, serializable contract for reproducible BTC technical research.

Unit-bearing names are intentional: 30 bps equals 0.30 percentage points and
0.003 as a decimal return.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from trading_config import DEFAULT_SYMBOL, FEE_BPS, SLIPPAGE_BPS, TOTAL_COST_PER_SIDE_BPS


RESEARCH_CONTRACT_VERSION = "btc-technical-research-v1"
HISTORICAL_ANALOG_TRIAL_FAMILY = (
    "returns_shape_v1_legacy",
    "returns_shape_v2_signed",
)


@dataclass(frozen=True)
class ExecutionCostSpec:
    """Screening cost assumptions; not a claim about realized fills."""

    fee_per_side_bps: float = FEE_BPS
    slippage_per_side_bps: float = SLIPPAGE_BPS

    @property
    def total_per_side_bps(self) -> float:
        return self.fee_per_side_bps + self.slippage_per_side_bps

    @property
    def round_trip_bps(self) -> float:
        return self.total_per_side_bps * 2.0

    @property
    def round_trip_return_fraction(self) -> float:
        return self.round_trip_bps / 10_000.0

    @property
    def round_trip_pct_points(self) -> float:
        return self.round_trip_bps / 100.0

    def to_dict(self) -> dict[str, float | str]:
        return {
            "feePerSideBps": self.fee_per_side_bps,
            "slippagePerSideBps": self.slippage_per_side_bps,
            "totalPerSideBps": self.total_per_side_bps,
            "roundTripBps": self.round_trip_bps,
            "roundTripReturnFraction": self.round_trip_return_fraction,
            "roundTripPctPoints": self.round_trip_pct_points,
            "role": "classification-dead-zone-floor; no fill or net-PnL claim",
        }


DEFAULT_EXECUTION_COSTS = ExecutionCostSpec()
assert DEFAULT_EXECUTION_COSTS.total_per_side_bps == TOTAL_COST_PER_SIDE_BPS


@dataclass(frozen=True)
class ResearchManifest:
    experiment: str
    symbol: str = DEFAULT_SYMBOL
    timeframe: str = "4h"
    signal_bar_state: str = "closed"
    decision_time: str = "after-signal-bar-close"
    outcome_price_basis: str = "signal-close-to-future-close"
    contract_version: str = RESEARCH_CONTRACT_VERSION

    def to_dict(
        self,
        *,
        parameters: dict[str, Any],
        data_provenance: dict[str, Any],
        costs: ExecutionCostSpec = DEFAULT_EXECUTION_COSTS,
        code_provenance: dict[str, Any] | None = None,
        runtime_dependencies: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contractVersion": self.contract_version,
            "experiment": self.experiment,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signalBarState": self.signal_bar_state,
            "decisionTime": self.decision_time,
            "outcomePriceBasis": self.outcome_price_basis,
            "costs": costs.to_dict(),
            "parameters": parameters,
            "dataProvenance": data_provenance,
        }
        if code_provenance is not None:
            payload["codeProvenance"] = code_provenance
        if runtime_dependencies is not None:
            payload["runtimeDependencies"] = runtime_dependencies
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload["manifestSha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload
