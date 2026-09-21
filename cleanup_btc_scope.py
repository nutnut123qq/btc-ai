#!/usr/bin/env python3
"""Preview or remove database rows outside the BTC production scope.

The production scope is BTCUSDT candles at 1h, 4h and 1d. BTC futures rows
in FuturesMetrics and MarketMetrics are retained at their native cadence.
NewsArticles and NewsChunks are never touched.

Dry-run is the default::

    python cleanup_btc_scope.py

Execution requires both an explicit flag and a fixed confirmation token::

    python cleanup_btc_scope.py --execute \
        --confirmation DELETE-NON-BTC-AND-SUBHOURLY
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Sequence

from db_config import get_db_connection
from trading_config import ACTIVE_PRODUCTION_TIMEFRAMES, DEFAULT_SYMBOL

CONFIRMATION_TOKEN = "DELETE-NON-BTC-AND-SUBHOURLY"


@dataclass(frozen=True)
class DeleteSpec:
    table: str
    predicate: str
    params: tuple[Any, ...]


def build_delete_specs(
    symbol: str = DEFAULT_SYMBOL,
    timeframes: Sequence[str] = ACTIVE_PRODUCTION_TIMEFRAMES,
) -> list[DeleteSpec]:
    """Return dependency-safe, exactly filtered cleanup operations."""
    scope_params = (symbol, list(timeframes))
    outside_scope = (
        '"Symbol" IS DISTINCT FROM %s OR "Timeframe" IS NULL '
        'OR "Timeframe" <> ALL(%s)'
    )
    non_btc = '"Symbol" IS DISTINCT FROM %s'

    specs = [
        # Children whose parents carry the authoritative scope identity.
        DeleteSpec(
            "BacktestTrades",
            'EXISTS (SELECT 1 FROM "BacktestRuns" r WHERE r."Id" = "BacktestTrades"."BacktestRunId" '
            f'AND ({outside_scope}))',
            scope_params,
        ),
        DeleteSpec(
            "ArchetypeOutcomes",
            'EXISTS (SELECT 1 FROM "CandleArchetypes" a WHERE a."Id" = "ArchetypeOutcomes"."ArchetypeId" '
            f'AND ({outside_scope}))',
            scope_params,
        ),
    ]

    # Tables with their own Symbol + Timeframe identity. Order keeps FK children
    # ahead of CandleArchetypes and raw Klines last.
    scoped_tables = (
        "ArchetypeTransitions",
        "ArchetypeSequences",
        "ArchetypeOccurrences",
        "WindowClassificationDatasets",
        "MlFeatureStores",
        "PriceTargets",
        "PatternSequences",
        "WindowVectors",
        "CandlePatterns",
        "CandleVolumeStats",
        "TechnicalIndicators",
        "ModelPredictions",
        "PaperTrades",
        "MarketRegimes",
        "RegimeTransitions",
        "VolumeProfileSnapshots",
        "SmartMoneyStructures",
        "LiquidationSnapshots",
        "EnsemblePredictionRecords",
        "CandleSequenceSignals",
        "CandleSequenceRules",
        "KlineGapStates",
        "CandleArchetypes",
        "BacktestRuns",
        "Klines",
    )
    specs.extend(DeleteSpec(table, outside_scope, scope_params) for table in scoped_tables)

    # Symbol-only tables. FuturesMetrics deliberately keeps every BTC row,
    # including the native 5m metrics requested for technical research.
    symbol_only_tables = (
        "ConfluenceSnapshots",
        "SentimentSnapshots",
        "FuturesMetrics",
        "MarketMetrics",
    )
    specs.extend(DeleteSpec(table, non_btc, (symbol,)) for table in symbol_only_tables)
    specs.append(
        DeleteSpec("WalletBalanceSnapshots", '"Symbol" IS NOT NULL AND "Symbol" <> %s', (symbol,))
    )
    return specs


def _existing_tables(cur) -> set[str]:
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = current_schema()"
    )
    return {row[0] for row in cur.fetchall()}


def _quoted_table(table: str) -> str:
    # All identifiers originate in the static list above. This assertion makes
    # accidental interpolation of CLI/user input impossible.
    if not table.replace("_", "").isalnum():
        raise ValueError(f"Unsafe table identifier: {table}")
    return f'"{table}"'


def collect_counts(cur, specs: Sequence[DeleteSpec]) -> list[tuple[DeleteSpec, int]]:
    existing = _existing_tables(cur)
    counts: list[tuple[DeleteSpec, int]] = []
    for spec in specs:
        if spec.table not in existing:
            continue
        cur.execute(
            f"SELECT COUNT(*) FROM {_quoted_table(spec.table)} WHERE {spec.predicate}",
            spec.params,
        )
        counts.append((spec, int(cur.fetchone()[0])))
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or delete non-BTC and non-1h/4h/1d project data."
    )
    parser.add_argument("--execute", action="store_true", help="Perform the reviewed deletes")
    parser.add_argument("--confirmation", help=f"Required with --execute: {CONFIRMATION_TOKEN}")
    args = parser.parse_args(argv)
    if args.execute and args.confirmation != CONFIRMATION_TOKEN:
        parser.error(f"--execute requires --confirmation {CONFIRMATION_TOKEN}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = build_delete_specs()
    conn = get_db_connection()
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '30min'")
            counts = collect_counts(cur, specs)
            total = sum(count for _, count in counts)
            mode = "EXECUTE" if args.execute else "DRY-RUN"
            print(f"[{mode}] Keep {DEFAULT_SYMBOL} candles: {', '.join(ACTIVE_PRODUCTION_TIMEFRAMES)}")
            print("[KEEP] FuturesMetrics BTC rows at native cadence")
            print("[KEEP] MarketMetrics BTC rows at native cadence")
            print("[KEEP] NewsArticles and NewsChunks")
            for spec, count in counts:
                print(f"  {spec.table}: {count:,}")
            print(f"Total rows outside scope: {total:,}")

            if not args.execute:
                conn.rollback()
                print("Dry-run complete; no rows were changed.")
                return 0

            for spec, expected in counts:
                if expected == 0:
                    continue
                cur.execute(
                    f"DELETE FROM {_quoted_table(spec.table)} WHERE {spec.predicate}",
                    spec.params,
                )
                print(f"  deleted {spec.table}: {cur.rowcount:,} (previewed {expected:,})")
            conn.commit()
            print("Cleanup committed.")
            return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
