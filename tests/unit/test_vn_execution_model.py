"""Unit tests for the HOSE equity execution model (MBB-04C).

All fixtures are deterministic and DB-free: session opens come from
``vietnam_market_calendar`` so lunch/weekend/holiday semantics match the
backend calendar instead of millisecond arithmetic.
"""

import sys
from datetime import date, time, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import vietnam_market_calendar as vn
from vn_execution_model import (
    PRICE_BASIS,
    VnEquityAccount,
    VnEquityCostSpec,
    VnFill,
    VnRejection,
    band_bounds,
    buy_shares_for_budget,
    classify_band,
    close_round_trip,
    next_session_fill_open_ms,
    settlement_date,
)


def _trading_days(start: date, count: int, non_trading=frozenset()) -> list[date]:
    days = []
    cursor = start
    while len(days) < count:
        if vn.is_trading_day(cursor, non_trading):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


DAYS = _trading_days(date(2026, 8, 3), 12)  # Mon 2026-08-03 .. two weeks
MON, TUE, WED, THU, FRI = DAYS[0], DAYS[1], DAYS[2], DAYS[3], DAYS[4]
assert MON.weekday() == 0 and FRI.weekday() == 4
MON2 = DAYS[5]


def _open_ms(day: date, local_time: time = time(9, 0)) -> int:
    return vn.to_utc_ms(day, local_time)


def test_lot_rounding_and_buy_fee():
    # 5,000,000 VND at 20,000/share + 0.15% fee -> 249 shares affordable,
    # rounded down to 2 lots = 200 shares.
    account = VnEquityAccount(5_000_000)
    fill = account.execute_buy(
        fill_open_ms=_open_ms(MON), open_price=20_000.0, reference_close=20_000.0
    )
    assert isinstance(fill, VnFill)
    assert fill.shares == 200 and fill.shares % 100 == 0
    assert fill.gross_value_vnd == 4_000_000.0
    assert fill.fee_vnd == pytest.approx(6_000.0)
    assert fill.tax_vnd == 0.0  # no buy-side tax
    assert account.settled_cash_vnd == pytest.approx(994_000.0)
    assert account.total_shares == 200
    assert account.settled_shares == 0  # pending T+2


def test_buy_shares_for_budget_helper():
    costs = VnEquityCostSpec()
    assert buy_shares_for_budget(5_000_000, 20_000, costs) == 200
    assert buy_shares_for_budget(1_999_999, 20_000, costs) == 0


def test_t2_settlement_rejects_sell_before_shares_settle():
    account = VnEquityAccount(5_000_000)
    buy = account.execute_buy(
        fill_open_ms=_open_ms(MON), open_price=20_000, reference_close=20_000
    )
    assert buy.settle_local_date == WED.isoformat()  # Mon + 2 trading days

    early = account.execute_sell(
        fill_open_ms=_open_ms(TUE), open_price=21_000, reference_close=20_000
    )
    assert isinstance(early, VnRejection) and early.reason == "shares-not-settled"

    fill = account.execute_sell(
        fill_open_ms=_open_ms(WED), open_price=21_000, reference_close=20_000
    )
    assert isinstance(fill, VnFill)
    assert fill.shares == 200
    assert fill.gross_value_vnd == 4_200_000.0
    assert fill.fee_vnd == pytest.approx(6_300.0)  # 0.15% of gross
    assert fill.tax_vnd == pytest.approx(4_200.0)  # 0.1% sell tax
    # proceeds = 4,200,000 - 6,300 - 4,200 = 4,189,500, pending until FRI
    assert fill.settle_local_date == FRI.isoformat()
    assert account.total_shares == 0
    assert account.settled_cash_vnd == pytest.approx(994_000.0)
    assert account.total_cash_vnd == pytest.approx(994_000 + 4_189_500.0)


def test_unsettled_proceeds_cannot_fund_a_new_buy():
    account = VnEquityAccount(5_000_000)
    account.execute_buy(
        fill_open_ms=_open_ms(MON), open_price=20_000, reference_close=20_000
    )
    account.execute_sell(
        fill_open_ms=_open_ms(WED), open_price=21_000, reference_close=20_000
    )
    # Thursday: proceeds still pending -> only the 994k leftover is spendable.
    blocked = account.execute_buy(
        fill_open_ms=_open_ms(THU), open_price=20_000, reference_close=20_000
    )
    assert isinstance(blocked, VnRejection)
    assert blocked.reason == "insufficient-cash-for-lot"
    # Friday: proceeds settled -> full budget is spendable again.
    fill = account.execute_buy(
        fill_open_ms=_open_ms(FRI), open_price=20_000, reference_close=20_000
    )
    assert isinstance(fill, VnFill) and fill.shares == 200


def test_price_band_rejects_opens_outside_ceiling_floor():
    floor, ceiling = band_bounds(20_000, 0.07)
    assert floor == pytest.approx(18_600.0)
    assert ceiling == pytest.approx(21_400.0)
    assert classify_band(21_400.0, 20_000) == "within-band"  # boundary ok
    assert classify_band(21_400.01, 20_000) == "open-above-ceiling"
    assert classify_band(18_599.99, 20_000) == "open-below-floor"

    account = VnEquityAccount(50_000_000)
    gap_up = account.execute_buy(
        fill_open_ms=_open_ms(MON), open_price=21_500.0, reference_close=20_000
    )
    assert isinstance(gap_up, VnRejection) and gap_up.reason == "open-above-ceiling"

    account.execute_buy(
        fill_open_ms=_open_ms(MON), open_price=20_000, reference_close=20_000
    )
    account.settle_as_of(WED)
    gap_down = account.execute_sell(
        fill_open_ms=_open_ms(WED), open_price=18_500.0, reference_close=20_000
    )
    assert isinstance(gap_down, VnRejection)
    assert gap_down.reason == "open-below-floor"


def test_next_session_open_spans_lunch_weekend_and_holiday():
    # 1h: the 11:00 bar's next open is 13:00 the same day (lunch is not a gap).
    eleven = vn.to_utc_ms(MON, time(11, 0))
    assert next_session_fill_open_ms("1h", eleven) == vn.to_utc_ms(MON, time(13, 0))
    # 1h: Friday 14:00 -> Monday 09:00 (weekend is not a gap).
    fri_last = vn.to_utc_ms(FRI, time(14, 0))
    assert next_session_fill_open_ms("1h", fri_last) == vn.to_utc_ms(
        MON2, time(9, 0)
    )
    # 1d: Friday -> next Monday's daily open.
    fri_daily = vn.daily_open_ms(FRI)
    assert next_session_fill_open_ms("1d", fri_daily) == vn.daily_open_ms(MON2)
    # Configured non-trading Monday pushes the fill to Tuesday.
    ntd = frozenset({MON2})
    assert next_session_fill_open_ms("1d", fri_daily, ntd) == vn.daily_open_ms(
        DAYS[6]
    )
    assert next_session_fill_open_ms("1h", fri_last, ntd) == vn.to_utc_ms(
        DAYS[6], time(9, 0)
    )


def test_settlement_date_skips_weekend_and_configured_holiday():
    # Friday trade -> settles Tuesday (Sat/Sun don't count).
    assert settlement_date(FRI) == DAYS[6]  # next Tuesday? DAYS[6]=Tue week2
    ntd = frozenset({DAYS[5], DAYS[6]})  # Mon+Tue of week 2 non-trading
    assert settlement_date(FRI, ntd) == DAYS[8]  # Wed(1) + Thu(2) trading days


def test_no_short_selling_and_no_oversell():
    account = VnEquityAccount(5_000_000)
    rejection = account.execute_sell(
        fill_open_ms=_open_ms(MON), open_price=20_000, reference_close=20_000
    )
    assert isinstance(rejection, VnRejection) and rejection.reason == "no-position"


def test_adjusted_price_basis_stamped_everywhere():
    account = VnEquityAccount(5_000_000)
    buy = account.execute_buy(
        fill_open_ms=_open_ms(MON), open_price=20_000, reference_close=20_000
    )
    sell = account.execute_sell(
        fill_open_ms=_open_ms(WED), open_price=21_000, reference_close=20_000
    )
    trip = close_round_trip(buy, sell)
    assert buy.price_basis == sell.price_basis == trip.price_basis == PRICE_BASIS
    for payload in (buy.to_dict(), sell.to_dict(), trip.to_dict()):
        assert payload["priceBasis"] == "adjusted"
    rejected = account.execute_sell(
        fill_open_ms=_open_ms(THU), open_price=21_000, reference_close=20_000
    )
    assert rejected.to_dict()["priceBasis"] == "adjusted"


def test_deterministic_round_trip_pnl_hand_computed():
    account = VnEquityAccount(5_000_000)
    buy = account.execute_buy(
        fill_open_ms=_open_ms(MON), open_price=20_000, reference_close=20_000
    )
    sell = account.execute_sell(
        fill_open_ms=_open_ms(WED), open_price=21_000, reference_close=20_000
    )
    trip = close_round_trip(buy, sell)
    # entry: 200 * 20,000 = 4,000,000 + 6,000 fee = invested 4,006,000
    # exit : 200 * 21,000 = 4,200,000 - 6,300 fee - 4,200 tax
    assert trip.gross_pnl_vnd == pytest.approx(200_000.0)
    assert trip.fee_vnd == pytest.approx(12_300.0)
    assert trip.tax_vnd == pytest.approx(4_200.0)
    assert trip.net_pnl_vnd == pytest.approx(183_500.0)
    assert trip.gross_return_fraction == pytest.approx(0.05)
    assert trip.net_return_fraction == pytest.approx(183_500.0 / 4_006_000.0)
    assert trip.entry_open_ms == _open_ms(MON)
    assert trip.exit_open_ms == _open_ms(WED)


def test_cost_spec_validation_and_dict():
    with pytest.raises(ValueError):
        VnEquityCostSpec(broker_fee_fraction=-0.1)
    with pytest.raises(ValueError):
        VnEquityCostSpec(price_band_fraction=0.0)
    spec = VnEquityCostSpec(broker_fee_fraction=0.002)
    payload = spec.to_dict()
    assert payload["brokerFeeFraction"] == 0.002
    assert payload["lotSize"] == 100
    assert payload["bandBreachPolicy"] == "reject"
    assert payload["fillPolicy"] == "next-session-open"
    assert payload["priceBasis"] == "adjusted"
