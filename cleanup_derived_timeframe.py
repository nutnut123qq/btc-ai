#!/usr/bin/env python3
"""
Delete derived ML/indexing data for a specific timeframe while keeping raw Klines.

Usage:
    python cleanup_derived_timeframe.py 1m

This removes rows from:
    WindowClassificationDatasets
    MlFeatureStores
    PriceTargets
    PatternSequences
    WindowVectors
    CandlePatterns
    CandleVolumeStats
    TechnicalIndicators
    CandleSequenceSignals   (all, because it has no timeframe column and is tiny)

It intentionally does NOT touch Klines, News*, PriceAlertSettings, AppAlerts,
or CandleSequenceRules.
"""

import os
import sys
import time
import argparse

from db_config import get_db_connection

# Order: downstream-derived first, upstream last.
DELETE_QUERIES = [
    ('"WindowClassificationDatasets"', '"Symbol" = %s AND "Timeframe" = %s'),
    ('"MlFeatureStores"', '"Symbol" = %s AND "Timeframe" = %s'),
    ('"PriceTargets"', '"Symbol" = %s AND "Timeframe" = %s'),
    ('"PatternSequences"', '"Symbol" = %s AND "Timeframe" = %s'),
    ('"WindowVectors"', '"Symbol" = %s AND "Timeframe" = %s'),
    ('"CandlePatterns"', '"Symbol" = %s AND "Timeframe" = %s'),
    ('"CandleVolumeStats"', '"Symbol" = %s AND "Timeframe" = %s'),
    ('"TechnicalIndicators"', '"Symbol" = %s AND "Timeframe" = %s'),
    # CandleSequenceSignals has no timeframe column; it is tiny, delete all.
    ('"CandleSequenceSignals"', "TRUE"),
]


def get_counts(cur, symbol, timeframe):
    counts = {}
    for table, where in DELETE_QUERIES:
        if where == "TRUE":
            cur.execute(f"SELECT COUNT(*) FROM {table}")
        else:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", (symbol, timeframe))
        counts[table] = cur.fetchone()[0]
    return counts


def main():
    parser = argparse.ArgumentParser(description="Delete derived data for a timeframe")
    parser.add_argument("timeframe", help="e.g. 1m, 5m, 1h")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.timeframe

    conn = get_db_connection()
    conn.autocommit = False
    cur = conn.cursor()

    # Long timeout for huge DELETEs.
    cur.execute("SET statement_timeout = '30min'")

    print(f"Checking derived data for {symbol} {timeframe}...")
    before = get_counts(cur, symbol, timeframe)
    total_before = sum(before.values())
    print(f"Total rows to delete: {total_before:,}")
    for table, count in before.items():
        print(f"  {table}: {count:,}")

    if total_before == 0:
        print("Nothing to delete.")
        cur.close()
        conn.close()
        return

    if not args.yes:
        resp = input(f"\nDelete {total_before:,} rows for {symbol} {timeframe}? Klines will be kept. [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            cur.close()
            conn.close()
            return

    print("\nDeleting...")
    start = time.time()
    deleted = {}
    for table, where in DELETE_QUERIES:
        if where == "TRUE":
            cur.execute(f"DELETE FROM {table}")
        else:
            cur.execute(f"DELETE FROM {table} WHERE {where}", (symbol, timeframe))
        deleted[table] = cur.rowcount
        print(f"  {table}: {cur.rowcount:,} rows deleted")

    conn.commit()
    elapsed = time.time() - start
    print(f"\nDeleted {sum(deleted.values()):,} rows in {elapsed:.1f}s")

    print("\nVerifying...")
    after = get_counts(cur, symbol, timeframe)
    for table, count in after.items():
        if count != 0:
            print(f"  WARNING: {table} still has {count:,} rows")

    # Check Klines still present
    cur.execute('SELECT COUNT(*) FROM "Klines" WHERE "Symbol" = %s AND "Timeframe" = %s', (symbol, timeframe))
    klines_count = cur.fetchone()[0]
    print(f"\nKlines {timeframe} still present: {klines_count:,} rows")

    cur.close()
    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
