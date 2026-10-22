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
- Non-trading dates (EQX v2 A1): the single source of truth is the backend's
  checked-in ``backend/Data/Calendars/vn-hose-non-trading-days.json``, synced
  by the backend into the ``MarketCalendarDays`` table. DB-backed Python jobs
  read that table (``load_calendar_from_cursor`` / ``load_calendar_db``). The
  MBB_NON_TRADING_DATES environment variable (comma-separated YYYY-MM-DD) is
  only an explicit override for fixtures/tests. An empty or unreadable table
  fails closed with ``calendar-unavailable`` — never "every weekday trades".
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable

VN_OFFSET = timezone(timedelta(hours=7), name="Asia/Ho_Chi_Minh")

HOURLY_OPENS_LOCAL = (time(9, 0), time(10, 0), time(11, 0), time(13, 0), time(14, 0))
DAILY_OPEN_LOCAL = time(7, 0)
MORNING_CLOSE_LOCAL = time(11, 30)
AFTERNOON_CLOSE_LOCAL = time(14, 45)

EQUITY_TIMEFRAMES = ("1h", "1d")

NON_TRADING_DATES_ENV = "MBB_NON_TRADING_DATES"

CALENDAR_VENUE = "HOSE"
CALENDAR_TABLE = "MarketCalendarDays"
CALENDAR_UNAVAILABLE = "calendar-unavailable"
CALENDAR_COVERAGE_EXPIRED = "calendar-coverage-expired"
BELOW_CALENDAR_COVERAGE = "below-calendar-coverage"
SOURCE_DB = f"db:{CALENDAR_TABLE}"
SOURCE_ENV = f"env:{NON_TRADING_DATES_ENV}"
SOURCE_EXPLICIT = "explicit"

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


class CalendarUnavailableError(RuntimeError):
    """The HOSE non-trading calendar could not be established (fail closed).

    ``reason`` is always ``calendar-unavailable``; the message says why (table
    missing/empty/unreadable, no override for a fixture run, ...).
    """

    reason = CALENDAR_UNAVAILABLE

    def __init__(self, message: str) -> None:
        super().__init__(f"{self.reason}: {message}")


class CalendarCoverageError(CalendarUnavailableError):
    """A date the computation needs lies after the calendar's coverage end (fail closed).

    After ``coverageTo`` the non-trading set is unknown, so treating weekdays as
    sessions would silently re-introduce the holiday-as-gap bug.
    """

    reason = CALENDAR_COVERAGE_EXPIRED


@dataclass(frozen=True)
class MarketCalendar:
    """Non-trading dates plus where they came from (recorded in every artifact)."""

    non_trading_dates: frozenset[date]
    source: str
    venue: str = CALENDAR_VENUE
    versions: tuple[str, ...] = ()
    unverified_count: int = 0
    # Coverage window of the calendar (MarketCalendarDays.CoverageFromDate/ToDate). None =
    # unbounded, which only explicit fixture/env calendars may use.
    coverage_from: date | None = None
    coverage_to: date | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def count(self) -> int:
        return len(self.non_trading_dates)

    @property
    def sha256(self) -> str:
        body = "\n".join(d.isoformat() for d in sorted(self.non_trading_dates))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    @property
    def version(self) -> str | None:
        return "+".join(self.versions) if self.versions else None

    def provenance(self) -> dict[str, Any]:
        dates = sorted(self.non_trading_dates)
        return {
            "source": self.source,
            "venue": self.venue,
            "version": self.version,
            "count": self.count,
            "unverifiedCount": self.unverified_count,
            "firstDate": dates[0].isoformat() if dates else None,
            "lastDate": dates[-1].isoformat() if dates else None,
            "sha256": self.sha256,
            "coverageFrom": self.coverage_from.isoformat() if self.coverage_from else None,
            "coverageTo": self.coverage_to.isoformat() if self.coverage_to else None,
        }

    def is_before_coverage(self, day: date) -> bool:
        return self.coverage_from is not None and day < self.coverage_from

    def require_covered(self, day: date, what: str = "date") -> None:
        """Fail closed when ``day`` is after the coverage end (session status unknown)."""
        if self.coverage_to is not None and day > self.coverage_to:
            raise CalendarCoverageError(
                f"{what} {day.isoformat()} is after the {self.venue} calendar coverage end "
                f"{self.coverage_to.isoformat()} (version {self.version}); add the next year's official notice"
            )


def calendar_from_dates(dates: Iterable[date], source: str = SOURCE_EXPLICIT) -> MarketCalendar:
    return MarketCalendar(frozenset(dates), source)


def calendar_from_env() -> MarketCalendar | None:
    """Explicit fixture/test override, or None when MBB_NON_TRADING_DATES is unset/blank.

    A blank value is NOT an override: it would silently mean "every weekday
    trades", which is exactly what the fail-closed rule forbids.
    """
    if not os.getenv(NON_TRADING_DATES_ENV, "").strip():
        return None
    return MarketCalendar(non_trading_dates_from_env(), SOURCE_ENV)


def load_calendar_from_cursor(cursor: Any, venue: str = CALENDAR_VENUE) -> MarketCalendar:
    """Read ``MarketCalendarDays`` for ``venue`` with an open DB-API cursor.

    Runs inside the caller's transaction (so a job's candle read and calendar
    read share one snapshot). Missing table / query error / zero rows raise
    :class:`CalendarUnavailableError`.
    """
    try:
        cursor.execute(
            f'SELECT "Date","Verification","CalendarVersion","CoverageFromDate","CoverageToDate" FROM "{CALENDAR_TABLE}" '
            'WHERE "Venue"=%s ORDER BY "Date"',
            (venue,),
        )
        rows = cursor.fetchall()
    except Exception as exc:  # noqa: BLE001 - any driver error => fail closed
        raise CalendarUnavailableError(
            f'"{CALENDAR_TABLE}" could not be read ({type(exc).__name__}); '
            "run the backend calendar sync (MarketCalendar:SyncOnStartup) first"
        ) from exc
    if not rows:
        raise CalendarUnavailableError(
            f'"{CALENDAR_TABLE}" has no {venue} rows; refusing to assume every weekday trades'
        )
    def as_date(value: Any) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))

    dates: set[date] = set()
    versions: set[str] = set()
    froms: set[date] = set()
    tos: set[date] = set()
    unverified = 0
    for raw_date, verification, version, cov_from, cov_to in rows:
        dates.add(as_date(raw_date))
        if cov_from is None or cov_to is None:
            raise CalendarUnavailableError(f'"{CALENDAR_TABLE}" row without coverage window')
        froms.add(as_date(cov_from))
        tos.add(as_date(cov_to))
        if version:
            versions.add(str(version))
        if verification != "verified":
            unverified += 1
    # Mixed versions: only the intersection of their coverage windows is known.
    return MarketCalendar(
        frozenset(dates), SOURCE_DB, venue, tuple(sorted(versions)), unverified, max(froms), min(tos)
    )


def load_calendar_db(venue: str = CALENDAR_VENUE, connection: Any = None) -> MarketCalendar:
    """Env override if set, else ``MarketCalendarDays`` via ``db_config`` (fail closed)."""
    override = calendar_from_env()
    if override is not None:
        return override
    owned = connection is None
    if owned:
        try:
            from db_config import get_db_connection

            connection = get_db_connection()
        except Exception as exc:  # noqa: BLE001
            raise CalendarUnavailableError(f"database unreachable ({type(exc).__name__})") from exc
    try:
        with connection.cursor() as cursor:
            calendar = load_calendar_from_cursor(cursor, venue)
        if owned:
            connection.rollback()
        return calendar
    finally:
        if owned:
            connection.close()


def require_calendar(
    non_trading_dates: Iterable[date] | None = None,
    calendar: MarketCalendar | None = None,
) -> MarketCalendar:
    """Calendar for a pure (fixture-capable) computation.

    Precedence: explicit ``calendar`` > explicit ``non_trading_dates`` > env
    override. Nothing given ⇒ fail closed; DB-backed callers load the table and
    pass it in.
    """
    if calendar is not None:
        return calendar
    if non_trading_dates is not None:
        return calendar_from_dates(non_trading_dates)
    override = calendar_from_env()
    if override is not None:
        return override
    raise CalendarUnavailableError(
        "no calendar supplied: DB-backed jobs read MarketCalendarDays, fixtures pass "
        f"non_trading_dates or set {NON_TRADING_DATES_ENV}"
    )


__all__ = [
    "AFTERNOON_CLOSE_LOCAL",
    "BELOW_CALENDAR_COVERAGE",
    "CALENDAR_COVERAGE_EXPIRED",
    "CALENDAR_UNAVAILABLE",
    "CalendarCoverageError",
    "CalendarUnavailableError",
    "MarketCalendar",
    "calendar_from_dates",
    "calendar_from_env",
    "load_calendar_db",
    "load_calendar_from_cursor",
    "require_calendar",
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
