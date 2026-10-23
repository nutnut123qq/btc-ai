"""Forward-paper fill/outcome observer for ``PaperObservations``.

Scheduled via ``backend/ops/run-ai-job.ps1 -Job PaperObserver`` (task ``BTCPaperObserver``).

Fills the nullable fill/outcome columns of directional forward-paper decisions
from a live quote observed inside a bounded window, and never otherwise.  This
is a simulated paper fill priced from a public quote, not an exchange
execution: no orders, no keys, no stored Klines.  Semantics are documented in
``docs/research/paper-observer.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Optional

from forward_paper_recorder import LiveQuote, fetch_binance_spot_quote

OBSERVER_VERSION = "forward-paper-observer-v1"
SYMBOL = "BTCUSDT"
FILL_WINDOW_MS = 15 * 60 * 1000  # aligned with the 15-minute BTCPaperTrader cadence
OUTCOME_HORIZON = "4h-gross"  # stored label: 4h after fill, costs NOT deducted
OUTCOME_HORIZON_MS = 4 * 60 * 60 * 1000
QUOTE_MAX_AGE_MS = 5_000
QUOTE_MAX_FUTURE_MS = 1_000
SELECT_SLACK_MS = 60_000  # pre-fetch selection is widened; exact checks use the quote time
# The database clock bounds every write, so a caller-supplied clock or a quote
# rebuilt from stored data cannot backdate a fill/outcome.
SERVER_PAST_TOLERANCE_MS = 10_000
SERVER_FUTURE_TOLERANCE_MS = 1_000

SIDES = {"long": 1, "short": -1}

SELECT_CANDIDATES_SQL = r'''
SELECT "Id", "Decision", "ObservedAtUtc", "AvailableTimeMs", "FillPrice", "FillObservedAtUtc"
FROM "PaperObservations"
WHERE "Symbol" = %s AND "Decision" IN ('long', 'short') AND (
    ("FillPrice" IS NULL AND "AvailableTimeMs" BETWEEN %s AND %s)
    OR ("FillPrice" IS NOT NULL AND "OutcomeReturn" IS NULL AND "FillObservedAtUtc" BETWEEN %s AND %s)
)
ORDER BY "AvailableTimeMs"
'''

_SERVER_CLOCK_GUARD = f'''
  AND %(quote_at)s::timestamptz BETWEEN NOW() - INTERVAL '{SERVER_PAST_TOLERANCE_MS} milliseconds'
                                    AND NOW() + INTERVAL '{SERVER_FUTURE_TOLERANCE_MS} milliseconds'
'''

FILL_UPDATE_SQL = r'''
UPDATE "PaperObservations"
SET "FillPrice" = %(price)s, "FillObservedAtUtc" = %(quote_at)s, "FillSource" = %(source)s
WHERE "Id" = %(id)s AND "Decision" IN ('long', 'short') AND "FillPrice" IS NULL
  AND "AvailableTimeMs" <= %(quote_ms)s AND "AvailableTimeMs" + %(window_ms)s >= %(quote_ms)s
  AND "ObservedAtUtc" <= %(quote_at)s''' + _SERVER_CLOCK_GUARD + '''
RETURNING "Id"
'''

OUTCOME_UPDATE_SQL = r'''
UPDATE "PaperObservations"
SET "OutcomeReturn" = %(gross)s, "OutcomeObservedAtUtc" = %(quote_at)s, "OutcomeHorizon" = %(horizon)s
WHERE "Id" = %(id)s AND "Decision" IN ('long', 'short') AND "FillPrice" IS NOT NULL
  AND "OutcomeReturn" IS NULL
  AND "FillObservedAtUtc" + %(horizon_ms)s * INTERVAL '1 millisecond' <= %(quote_at)s
  AND "FillObservedAtUtc" + (%(horizon_ms)s + %(window_ms)s) * INTERVAL '1 millisecond' >= %(quote_at)s''' + _SERVER_CLOCK_GUARD + '''
RETURNING "Id"
'''


def fill_params(row_id, price, quote_at: datetime, source: str) -> dict:
    return {"id": row_id, "price": price, "quote_at": quote_at, "source": source,
            "quote_ms": _ms(quote_at), "window_ms": FILL_WINDOW_MS}


def outcome_params(row_id, gross: float, quote_at: datetime) -> dict:
    return {"id": row_id, "gross": gross, "quote_at": quote_at, "horizon": OUTCOME_HORIZON,
            "horizon_ms": OUTCOME_HORIZON_MS, "window_ms": FILL_WINDOW_MS}


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def quote_rejection(quote: Any, now: datetime) -> Optional[str]:
    """Return why a quote cannot price a paper fill, or None if it can."""
    if not isinstance(quote, LiveQuote):
        return "quote-invalid-type"
    if not quote.price > 0:
        return "quote-nonpositive"
    if quote.received_at_utc.tzinfo is None:
        return "quote-naive-timestamp"
    if abs(_ms(quote.received_at_utc) - quote.received_time_ms) > 1:
        return "quote-timestamp-mismatch"
    now_ms = _ms(now)
    if now_ms - quote.received_time_ms > QUOTE_MAX_AGE_MS:
        return "quote-stale"
    if quote.received_time_ms - now_ms > QUOTE_MAX_FUTURE_MS:
        return "quote-from-future"
    return None


def plan_writes(rows, quote: LiveQuote) -> list[tuple]:
    """Pure decision: which fills/outcomes the observed quote may write.

    rows: (id, decision, observed_at_utc, available_time_ms, fill_price, fill_observed_at_utc)
    """
    q_ms, q_at = quote.received_time_ms, quote.received_at_utc
    writes: list[tuple] = []
    for row_id, decision, observed_at, available_ms, fill_price, fill_at in rows:
        side = SIDES.get(decision)
        if side is None:
            continue  # abstain never gets a fill or PnL
        if fill_price is None:
            if available_ms <= q_ms <= available_ms + FILL_WINDOW_MS and q_at >= observed_at:
                source = f"{quote.source}|receivedTimeMs={q_ms}|observer={OBSERVER_VERSION}"
                writes.append(("fill", row_id, quote.price, q_at, source))
        else:
            due = fill_at + timedelta(milliseconds=OUTCOME_HORIZON_MS)
            if due <= q_at <= due + timedelta(milliseconds=FILL_WINDOW_MS):
                gross = float(side * (Decimal(quote.price) / Decimal(fill_price) - 1))
                writes.append(("outcome", row_id, gross, q_at, OUTCOME_HORIZON))
    return writes


def observe_once(conn: Any) -> dict:
    """Public boundary: real clock and real live quote only."""
    return _observe_once(conn, quote_fetcher=fetch_binance_spot_quote,
                         clock=lambda: datetime.now(timezone.utc))


def _observe_once(
    conn: Any,
    *,
    quote_fetcher: Callable[[str], LiveQuote],
    clock: Callable[[], datetime],
) -> dict:
    """One pass: read candidates, fetch one live quote, conditionally write.

    Injected clock/fetcher exist for tests; the SQL server-clock guard still
    rejects any quote time that is not current per the database.
    """
    cur = conn.cursor()
    try:
        now_ms = _ms(clock())
        cur.execute(
            SELECT_CANDIDATES_SQL,
            (
                SYMBOL,
                now_ms - FILL_WINDOW_MS - SELECT_SLACK_MS,
                now_ms + SELECT_SLACK_MS,
                datetime.fromtimestamp((now_ms - OUTCOME_HORIZON_MS - FILL_WINDOW_MS - SELECT_SLACK_MS) / 1000, timezone.utc),
                datetime.fromtimestamp((now_ms - OUTCOME_HORIZON_MS + SELECT_SLACK_MS) / 1000, timezone.utc),
            ),
        )
        rows = cur.fetchall()
        conn.commit()
        summary = {"candidates": len(rows), "fills": 0, "outcomes": 0, "quote": "not-needed"}
        if not rows:
            return summary

        try:
            quote = quote_fetcher(SYMBOL)
        except Exception as exc:  # any fetch failure means no observation, so no write
            summary["quote"] = f"unavailable:{type(exc).__name__}"
            return summary
        rejection = quote_rejection(quote, clock())
        if rejection:
            summary["quote"] = rejection
            return summary
        summary["quote"] = "ok"

        for kind, row_id, value, quote_at, label in plan_writes(rows, quote):
            if kind == "fill":
                cur.execute(FILL_UPDATE_SQL, fill_params(row_id, value, quote_at, label))
            else:
                cur.execute(OUTCOME_UPDATE_SQL, outcome_params(row_id, value, quote_at))
            written = cur.fetchone() is not None
            conn.commit()
            if written:
                summary["fills" if kind == "fill" else "outcomes"] += 1
        return summary
    finally:
        cur.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Forward-paper fill/outcome observer (one pass).")
    parser.add_argument("--once", action="store_true", required=True, help="Run exactly one observation pass")
    parser.parse_args(argv)

    head = {"observer": OBSERVER_VERSION, "ranAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        from db_config import get_db_connection  # lazy: loads ai/.env only for the CLI

        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT to_regclass('public.\"PaperObservations\"') IS NOT NULL")
            exists = bool(cur.fetchone()[0])
            cur.close()
            summary = observe_once(conn) if exists else {"skipped": "PaperObservations table missing"}
        finally:
            conn.close()
    except Exception as exc:
        # stdout, not a traceback on stderr: the ops wrapper logs stdout, and Task Scheduler gets the exit code.
        print(json.dumps({**head, "error": type(exc).__name__, "message": str(exc)[:500]}))
        return 1
    print(json.dumps({**head, **summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
