"""HOSE (Asia/Ho_Chi_Minh) session calendar for equity bars — mirrors
backend/Services/VietnamMarketCalendar.cs for the MBB lane.

Facts locked by the MBB-00 probe evidence and the backend calendar:

- The VN timezone has no DST; a fixed UTC+7 offset is exact for all of MBB's
  history, so no tzdata dependency is needed.
- HOSE sessions: morning continuous 09:00-11:30, afternoon 13:00-14:45 (incl.
  ATC). Hourly bars open at 09/10/11/13/14 local; the 11:00 bar closes 11:30
  and the 14:00 bar closes 14:45 — durations are session-aware, never open+60m.
- Daily bars open at 07:00 local (= 00:00 UTC of the trading date).
- Lunch, overnight, weekends and configured non-trading dates are NOT data
  gaps: contiguity is defined on the expected-open sequence, never on a fixed
  millisecond interval. A missing expected open is a real gap.
- Non-trading dates (holidays/suspensions) come from the MBB_NON_TRADING_DATES
  environment variable as comma-separated YYYY-MM-DD, matching the backend's
  MbbIngestion:NonTradingDates configuration surface.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone

VN_OFFSET = timezone(timedelta(hours=7), name="Asia/Ho_Chi_Minh")

HOURLY_OPENS_LOCAL = (time(9, 0), time(10, 0), time(11, 0), time(13, 0), time(14, 0))
DAILY_OPEN_LOCAL = time(7, 0)
MORNING_CLOSE_LOCAL = time(11, 30)
AFTERNOON_CLOSE_LOCAL = time(14, 45)

EQUITY_TIMEFRAMES = ("1h", "1d")

NON_TRADING_DATES_ENV = "MBB_NON_TRADING_DATES"

# Bound the next-trading-day walk so a pathological non-trading list cannot
# spin forever; 45 days covers the longest holiday chains with headroom.
_LOOKAHEAD_DAYS = 45


def to_utc_ms(local_date: date, local_time: time) -> int:
    return int(datetime.combine(local_date, local_time, tzinfo=VN_OFFSET).timestamp() * 1000)


def to_local(utc_ms: int) -> datetime:
    return datetime.fromtimestamp(utc_ms / 1000, tz=VN_OFFSET)


def is_trading_day(local_date: date, non_trading_dates: frozenset[date] = frozenset()) -> bool:
    """Trading day = weekday that is not a configured non-trading date."""
    return local_date.weekday() < 5 and local_date not in non_trading_dates


def hourly_opens_ms(local_date: date) -> tuple[int, ...]:
    return tuple(to_utc_ms(local_date, t) for t in HOURLY_OPENS_LOCAL)


def daily_open_ms(local_date: date) -> int:
    return to_utc_ms(local_date, DAILY_OPEN_LOCAL)


def session_close_ms(timeframe: str, open_ms: int) -> int:
    """Session-aware modeled bar close (UTC ms), same rules as the backend.

    1h closes: 09->10:00, 10->11:00, 11->11:30, 13->14:00, 14->14:45 local.
    1d closes at the same-day session end 14:45 local. An off-model open
    (e.g. the 2024-06-20 12:00 anomaly) echoes the open back: zero-duration.
    """
    local = to_local(open_ms)
    local_date = local.date()
    local_time = local.time().replace(tzinfo=None)
    if timeframe == "1d":
        return to_utc_ms(local_date, AFTERNOON_CLOSE_LOCAL)
    if timeframe == "1h":
        if local_time == time(9, 0):
            return to_utc_ms(local_date, time(10, 0))
        if local_time == time(10, 0):
            return to_utc_ms(local_date, time(11, 0))
        if local_time == time(11, 0):
            return to_utc_ms(local_date, MORNING_CLOSE_LOCAL)
        if local_time == time(13, 0):
            return to_utc_ms(local_date, time(14, 0))
        if local_time == time(14, 0):
            return to_utc_ms(local_date, AFTERNOON_CLOSE_LOCAL)
        return open_ms
    raise ValueError(f"Unsupported equity timeframe '{timeframe}'.")


def is_expected_open(
    timeframe: str, open_ms: int, non_trading_dates: frozenset[date] = frozenset()
) -> bool:
    """True when open_ms is a declared session open on a trading day."""
    local = to_local(open_ms)
    if not is_trading_day(local.date(), non_trading_dates):
        return False
    local_time = local.time().replace(tzinfo=None)
    if timeframe == "1d":
        return local_time == DAILY_OPEN_LOCAL
    if timeframe == "1h":
        return local_time in HOURLY_OPENS_LOCAL
    raise ValueError(f"Unsupported equity timeframe '{timeframe}'.")


def next_expected_open_ms(
    timeframe: str, open_ms: int, non_trading_dates: frozenset[date] = frozenset()
) -> int | None:
    """The expected open that must follow ``open_ms``.

    Returns ``None`` when ``open_ms`` is itself off-session (anomalous bars
    have no expected successor, so they can never be part of a contiguous run).
    """
    local = to_local(open_ms)
    local_date = local.date()
    if not is_trading_day(local_date, non_trading_dates):
        return None
    if timeframe == "1h":
        local_time = local.time().replace(tzinfo=None)
        if local_time not in HOURLY_OPENS_LOCAL:
            return None
        index = HOURLY_OPENS_LOCAL.index(local_time)
        if index + 1 < len(HOURLY_OPENS_LOCAL):
            return to_utc_ms(local_date, HOURLY_OPENS_LOCAL[index + 1])
        probe = local_date
    elif timeframe == "1d":
        if local.time().replace(tzinfo=None) != DAILY_OPEN_LOCAL:
            return None
        probe = local_date
    else:
        raise ValueError(f"Unsupported equity timeframe '{timeframe}'.")
    for _ in range(_LOOKAHEAD_DAYS):
        probe += timedelta(days=1)
        if is_trading_day(probe, non_trading_dates):
            return daily_open_ms(probe) if timeframe == "1d" else hourly_opens_ms(probe)[0]
    return None


def opens_are_adjacent(
    timeframe: str,
    prev_open_ms: int,
    next_open_ms: int,
    non_trading_dates: frozenset[date] = frozenset(),
) -> bool:
    """Session-aware contiguity: next_open must be prev's expected next open.

    11:00 -> 13:00 and Friday 14:00 -> Monday 09:00 are adjacent; a missing
    expected open makes the pair non-adjacent (a real data gap).
    """
    expected = next_expected_open_ms(timeframe, prev_open_ms, non_trading_dates)
    return expected is not None and expected == next_open_ms


def non_trading_dates_from_env() -> frozenset[date]:
    """Parse MBB_NON_TRADING_DATES (comma-separated YYYY-MM-DD); strict."""
    raw = os.getenv(NON_TRADING_DATES_ENV, "")
    parsed: set[date] = set()
    for token in raw.split(","):
        token = token.strip()
        if token:
            parsed.add(date.fromisoformat(token))
    return frozenset(parsed)


__all__ = [
    "AFTERNOON_CLOSE_LOCAL",
    "DAILY_OPEN_LOCAL",
    "EQUITY_TIMEFRAMES",
    "HOURLY_OPENS_LOCAL",
    "MORNING_CLOSE_LOCAL",
    "NON_TRADING_DATES_ENV",
    "VN_OFFSET",
    "daily_open_ms",
    "hourly_opens_ms",
    "is_expected_open",
    "is_trading_day",
    "next_expected_open_ms",
    "non_trading_dates_from_env",
    "opens_are_adjacent",
    "session_close_ms",
    "to_local",
    "to_utc_ms",
]
