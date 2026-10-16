"""HOSE equity execution model for the MBB replay lane (MBB-04C).

Pure, DB-free, deterministic counterpart of ``execution_engine.py`` — that
engine is BTC/Binance semantics (bps fee+slippage, perpetual, next-bar-open)
and is intentionally left untouched.  Everything here models HOSE cash
equity rules for MBB only.

Declared conventions (contract, not tunables discovered on data):

- ``PRICE_BASIS = "adjusted"``: the InstrumentCandles series is VCI
  adjusted-price history.  Fills, returns and PnL are all computed on the
  adjusted basis; every emitted record carries ``priceBasis`` so consumers
  cannot mistake these for real matched prices.  Corporate-action cash
  flows are NOT modelled — adjusted returns already embed them.
- Lot size 100 shares.  Buys round down to whole lots; a sell liquidates
  the whole (lot-aligned) position, so qty is always an even lot.
- HOSE price band ±7% around a reference price.  The reference is the
  close of the previous bar of the SAME timeframe that precedes the fill
  bar (declared choice over prior-session-close; for 1d bars it coincides
  with prior session close, for 1h it is the previous closed hourly bar).
- Fill policy ``next-session-open``: a signal computed on a closed bar may
  fill no earlier than the next expected session open per
  ``vietnam_market_calendar`` (never ``open + interval_ms``).  Lunch,
  overnight, weekends and configured non-trading dates are handled by the
  calendar, not by arithmetic on milliseconds.
- Band-breach policy ``reject``: if the next open prints outside the
  ceiling/floor band the order is marked non-fill with an explicit reason.
  We do not clamp to the band edge — a fill at ceiling cannot be proven
  from OHLCV data.
- Costs: broker fee configurable (default 0.15% per side, on notional);
  sell-side tax 0.1% of gross proceeds; no buy-side tax.
- T+2 settlement counted in TRADING DAYS via the calendar: shares bought
  on day D are sellable on the 2nd following trading day; sell proceeds
  are usable cash only after the same T+2 lag.
- No shorting, no leverage: a sell requires settled shares equal to the
  whole position; a buy requires settled cash for at least one lot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

import vietnam_market_calendar as vn_calendar

PRICE_BASIS = "adjusted"
HOSE_LOT_SIZE = 100
DEFAULT_PRICE_BAND_FRACTION = 0.07
DEFAULT_BROKER_FEE_FRACTION = 0.0015  # 0.15% per side, configurable
DEFAULT_SELL_TAX_FRACTION = 0.001  # 0.1% of gross proceeds, sell side only
DEFAULT_SETTLEMENT_TRADING_DAYS = 2
FILL_POLICY = "next-session-open"
BAND_BREACH_POLICY = "reject"

# Bounded trading-day walk for settlement, mirroring the calendar's
# _LOOKAHEAD_DAYS so a pathological non-trading list cannot spin forever.
_SETTLE_LOOKAHEAD_DAYS = 45

BandState = Literal["within-band", "open-above-ceiling", "open-below-floor"]
OrderSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class VnEquityCostSpec:
    """HOSE cash-equity cost/constraint bundle (declared, configurable)."""

    broker_fee_fraction: float = DEFAULT_BROKER_FEE_FRACTION
    sell_tax_fraction: float = DEFAULT_SELL_TAX_FRACTION
    price_band_fraction: float = DEFAULT_PRICE_BAND_FRACTION
    lot_size: int = HOSE_LOT_SIZE
    settlement_trading_days: int = DEFAULT_SETTLEMENT_TRADING_DAYS
    price_basis: str = PRICE_BASIS

    def __post_init__(self) -> None:
        if self.broker_fee_fraction < 0 or self.sell_tax_fraction < 0:
            raise ValueError("Fee/tax fractions cannot be negative.")
        if not 0 < self.price_band_fraction < 1:
            raise ValueError("price_band_fraction must be in (0, 1).")
        if self.lot_size < 1:
            raise ValueError("lot_size must be at least one share.")
        if self.settlement_trading_days < 0:
            raise ValueError("settlement_trading_days cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "brokerFeeFraction": self.broker_fee_fraction,
            "sellTaxFraction": self.sell_tax_fraction,
            "priceBandFraction": self.price_band_fraction,
            "lotSize": self.lot_size,
            "settlementTradingDays": self.settlement_trading_days,
            "priceBasis": self.price_basis,
            "bandBreachPolicy": BAND_BREACH_POLICY,
            "fillPolicy": FILL_POLICY,
            "bandReference": "previous-closed-bar-same-timeframe",
        }


@dataclass(frozen=True)
class VnFill:
    side: OrderSide
    fill_open_ms: int
    expected_open_ms: int
    price: float  # adjusted basis
    shares: int
    gross_value_vnd: float
    fee_vnd: float
    tax_vnd: float
    settle_local_date: str  # ISO date, trading-day settlement
    price_basis: str = PRICE_BASIS

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "fillOpenMs": self.fill_open_ms,
            "expectedOpenMs": self.expected_open_ms,
            "price": self.price,
            "shares": self.shares,
            "grossValueVnd": self.gross_value_vnd,
            "feeVnd": self.fee_vnd,
            "taxVnd": self.tax_vnd,
            "settleLocalDate": self.settle_local_date,
            "priceBasis": self.price_basis,
        }


@dataclass(frozen=True)
class VnRejection:
    side: OrderSide
    fill_open_ms: int | None
    reason: str
    price_basis: str = PRICE_BASIS

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "fillOpenMs": self.fill_open_ms,
            "reason": self.reason,
            "priceBasis": self.price_basis,
        }


@dataclass(frozen=True)
class VnRoundTrip:
    """Closed long round-trip on the adjusted basis."""

    entry_open_ms: int
    exit_open_ms: int
    shares: int
    entry_price: float
    exit_price: float
    gross_pnl_vnd: float
    fee_vnd: float
    tax_vnd: float
    net_pnl_vnd: float
    gross_return_fraction: float
    net_return_fraction: float
    price_basis: str = PRICE_BASIS

    def to_dict(self) -> dict[str, Any]:
        return {
            "entryOpenMs": self.entry_open_ms,
            "exitOpenMs": self.exit_open_ms,
            "shares": self.shares,
            "entryPrice": self.entry_price,
            "exitPrice": self.exit_price,
            "grossPnlVnd": self.gross_pnl_vnd,
            "feeVnd": self.fee_vnd,
            "taxVnd": self.tax_vnd,
            "netPnlVnd": self.net_pnl_vnd,
            "grossReturnPct": self.gross_return_fraction * 100.0,
            "netReturnPct": self.net_return_fraction * 100.0,
            "priceBasis": self.price_basis,
        }


def settlement_date(
    trade_local_date: date,
    non_trading_dates: frozenset[date] = frozenset(),
    trading_days: int = DEFAULT_SETTLEMENT_TRADING_DAYS,
) -> date:
    """T+N settlement date counted in trading days after the trade date."""
    if trading_days < 0:
        raise ValueError("trading_days cannot be negative.")
    probe = trade_local_date
    counted = 0
    for _ in range(_SETTLE_LOOKAHEAD_DAYS):
        if counted == trading_days:
            return probe
        probe += timedelta(days=1)
        if vn_calendar.is_trading_day(probe, non_trading_dates):
            counted += 1
    raise ValueError(
        f"No settlement date within {_SETTLE_LOOKAHEAD_DAYS} days of {trade_local_date}."
    )


def band_bounds(
    reference_close: float, band_fraction: float = DEFAULT_PRICE_BAND_FRACTION
) -> tuple[float, float]:
    """(floor, ceiling) of the HOSE price band around the reference close."""
    if reference_close <= 0:
        raise ValueError("reference_close must be positive.")
    return (
        reference_close * (1.0 - band_fraction),
        reference_close * (1.0 + band_fraction),
    )


def classify_band(
    open_price: float,
    reference_close: float,
    band_fraction: float = DEFAULT_PRICE_BAND_FRACTION,
) -> BandState:
    floor, ceiling = band_bounds(reference_close, band_fraction)
    if open_price > ceiling:
        return "open-above-ceiling"
    if open_price < floor:
        return "open-below-floor"
    return "within-band"


def next_session_fill_open_ms(
    timeframe: str,
    signal_bar_open_ms: int,
    non_trading_dates: frozenset[date] = frozenset(),
) -> int | None:
    """Earliest modeled fill time = next expected session open (calendar)."""
    return vn_calendar.next_expected_open_ms(
        timeframe, signal_bar_open_ms, non_trading_dates
    )


def buy_shares_for_budget(
    settled_cash_vnd: float,
    price: float,
    costs: VnEquityCostSpec,
) -> int:
    """Max whole-lot share count affordable from settled cash incl. buy fee."""
    if price <= 0:
        raise ValueError("price must be positive.")
    per_share_cost = price * (1.0 + costs.broker_fee_fraction)
    shares = int(settled_cash_vnd // per_share_cost)
    return shares - (shares % costs.lot_size)


def close_round_trip(entry: VnFill, exit_fill: VnFill) -> VnRoundTrip:
    if entry.side != "buy" or exit_fill.side != "sell":
        raise ValueError("A round trip is buy entry + sell exit.")
    if exit_fill.shares != entry.shares:
        raise ValueError("Replay exits the whole position; share counts must match.")
    gross = (exit_fill.price - entry.price) * entry.shares
    fees = entry.fee_vnd + exit_fill.fee_vnd
    invested = entry.gross_value_vnd + entry.fee_vnd
    net = gross - fees - exit_fill.tax_vnd
    return VnRoundTrip(
        entry_open_ms=entry.fill_open_ms,
        exit_open_ms=exit_fill.fill_open_ms,
        shares=entry.shares,
        entry_price=entry.price,
        exit_price=exit_fill.price,
        gross_pnl_vnd=gross,
        fee_vnd=fees,
        tax_vnd=exit_fill.tax_vnd,
        net_pnl_vnd=net,
        gross_return_fraction=exit_fill.price / entry.price - 1.0,
        net_return_fraction=net / invested,
    )


@dataclass
class VnEquityAccount:
    """Single-position HOSE cash account with T+2 settlement tracking.

    ``cash_settled`` is spendable; sell proceeds sit in ``_cash_pending``
    until their trading-day settlement date.  ``shares_settled`` is
    sellable; freshly bought shares sit in ``_shares_pending`` the same
    way.  Sell executes only when the ENTIRE position is settled
    (whole-position exit keeps round-trip accounting exact and mirrors the
    declared no-partial-exit rule of this replay).
    """

    initial_cash_vnd: float
    costs: VnEquityCostSpec = field(default_factory=VnEquityCostSpec)
    non_trading_dates: frozenset[date] = frozenset()

    def __post_init__(self) -> None:
        if self.initial_cash_vnd < 0:
            raise ValueError("initial_cash_vnd cannot be negative.")
        self.non_trading_dates = frozenset(self.non_trading_dates)
        self._cash_settled: float = float(self.initial_cash_vnd)
        self._cash_pending: list[list[Any]] = []  # [settle_date, amount]
        self._shares_settled: int = 0
        self._shares_pending: list[list[Any]] = []  # [settle_date, shares]

    # -- settlement ----------------------------------------------------

    def settle_as_of(self, local_date: date) -> None:
        """Move pending cash/shares whose settle date arrived into settled."""
        ready_cash = [e for e in self._cash_pending if e[0] <= local_date]
        self._cash_pending = [e for e in self._cash_pending if e[0] > local_date]
        self._cash_settled += sum(e[1] for e in ready_cash)
        ready_shares = [e for e in self._shares_pending if e[0] <= local_date]
        self._shares_pending = [e for e in self._shares_pending if e[0] > local_date]
        self._shares_settled += sum(int(e[1]) for e in ready_shares)

    def _settle_for_bar(self, open_ms: int) -> date:
        local_date = vn_calendar.to_local(open_ms).date()
        self.settle_as_of(local_date)
        return local_date

    # -- views ----------------------------------------------------------

    @property
    def settled_cash_vnd(self) -> float:
        return self._cash_settled

    @property
    def settled_shares(self) -> int:
        return self._shares_settled

    @property
    def pending_shares(self) -> int:
        return sum(int(e[1]) for e in self._shares_pending)

    @property
    def total_shares(self) -> int:
        return self._shares_settled + self.pending_shares

    @property
    def total_cash_vnd(self) -> float:
        return self._cash_settled + sum(e[1] for e in self._cash_pending)

    def equity_vnd(self, mark_price: float) -> float:
        return self.total_cash_vnd + self.total_shares * mark_price

    # -- orders ---------------------------------------------------------

    def execute_buy(
        self,
        *,
        fill_open_ms: int,
        open_price: float,
        reference_close: float,
        expected_open_ms: int | None = None,
    ) -> VnFill | VnRejection:
        """Buy max affordable whole lots at the bar open, if inside band."""
        if open_price <= 0 or reference_close <= 0:
            raise ValueError("Prices must be positive.")
        band = classify_band(open_price, reference_close, self.costs.price_band_fraction)
        if band != "within-band":
            return VnRejection("buy", fill_open_ms, band)
        trade_date = self._settle_for_bar(fill_open_ms)
        shares = buy_shares_for_budget(self._cash_settled, open_price, self.costs)
        if shares < self.costs.lot_size:
            return VnRejection("buy", fill_open_ms, "insufficient-cash-for-lot")
        notional = shares * open_price
        fee = notional * self.costs.broker_fee_fraction
        self._cash_settled -= notional + fee
        settle = settlement_date(
            trade_date, self.non_trading_dates, self.costs.settlement_trading_days
        )
        self._shares_pending.append([settle, shares])
        return VnFill(
            side="buy",
            fill_open_ms=fill_open_ms,
            expected_open_ms=expected_open_ms
            if expected_open_ms is not None
            else fill_open_ms,
            price=open_price,
            shares=shares,
            gross_value_vnd=notional,
            fee_vnd=fee,
            tax_vnd=0.0,
            settle_local_date=settle.isoformat(),
        )

    def execute_sell(
        self,
        *,
        fill_open_ms: int,
        open_price: float,
        reference_close: float,
        expected_open_ms: int | None = None,
    ) -> VnFill | VnRejection:
        """Sell the whole position at the bar open if fully settled + in band."""
        if open_price <= 0 or reference_close <= 0:
            raise ValueError("Prices must be positive.")
        band = classify_band(open_price, reference_close, self.costs.price_band_fraction)
        if band != "within-band":
            return VnRejection("sell", fill_open_ms, band)
        trade_date = self._settle_for_bar(fill_open_ms)
        if self.total_shares <= 0:
            return VnRejection("sell", fill_open_ms, "no-position")
        if self._shares_settled != self.total_shares:
            return VnRejection("sell", fill_open_ms, "shares-not-settled")
        shares = self._shares_settled
        gross = shares * open_price
        fee = gross * self.costs.broker_fee_fraction
        tax = gross * self.costs.sell_tax_fraction
        self._shares_settled = 0
        settle = settlement_date(
            trade_date, self.non_trading_dates, self.costs.settlement_trading_days
        )
        self._cash_pending.append([settle, gross - fee - tax])
        return VnFill(
            side="sell",
            fill_open_ms=fill_open_ms,
            expected_open_ms=expected_open_ms
            if expected_open_ms is not None
            else fill_open_ms,
            price=open_price,
            shares=shares,
            gross_value_vnd=gross,
            fee_vnd=fee,
            tax_vnd=tax,
            settle_local_date=settle.isoformat(),
        )


__all__ = [
    "BAND_BREACH_POLICY",
    "DEFAULT_BROKER_FEE_FRACTION",
    "DEFAULT_PRICE_BAND_FRACTION",
    "DEFAULT_SELL_TAX_FRACTION",
    "DEFAULT_SETTLEMENT_TRADING_DAYS",
    "FILL_POLICY",
    "HOSE_LOT_SIZE",
    "PRICE_BASIS",
    "VnEquityAccount",
    "VnEquityCostSpec",
    "VnFill",
    "VnRejection",
    "VnRoundTrip",
    "band_bounds",
    "buy_shares_for_budget",
    "classify_band",
    "close_round_trip",
    "next_session_fill_open_ms",
    "settlement_date",
]
