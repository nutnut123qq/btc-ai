"""compute_metrics must emit only finite numbers (BTINF-1).

A run without a losing trade used to report profit_factor=float("inf"), which
json.dumps writes as non-standard `Infinity` into BacktestRuns.MetricsJson and
psycopg2 inserts into the non-null double ProfitFactor column. The convention
now mirrors backend EnsembleBacktestService: no loss -> 99.9 if anything was
won, else 0.
"""

import json
import math
import numbers

import numpy as np
import pytest

from backtest_strategy import compute_metrics, simulate_trades

HOUR_MS = 3_600_000


class LongModel:
    def predict_proba(self, values):
        result = np.full((len(values), 3), 0.05, dtype=float)
        result[:, 2] = 0.90  # XGB index 2 -> label 1 -> long
        return result


def _bars(opens):
    base = 1_700_000_000_000
    return [(base + i * HOUR_MS, p, p, p, p, 1_000.0) for i, p in enumerate(opens)]


def _run(opens):
    bars = _bars(opens)
    # Signals on bars 0 and 3; each fills at the next open and exits 2 bars later.
    rows = [(bars[0][0], [1.0], 1, 0.0), (bars[3][0], [1.0], 1, 0.0)]
    trades = simulate_trades(
        rows,
        bars,
        2 * HOUR_MS,
        model=LongModel(),
        meta={"model_name": "XGB_test"},
        fee_bps=0,
        slippage_bps=0,
    )
    assert len(trades) == 2
    return trades, compute_metrics(trades, bars)


def _numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        yield value


def _assert_strict_finite(metrics):
    values = list(_numbers(metrics))
    assert values
    assert all(math.isfinite(float(v)) for v in values)
    json.dumps(metrics, allow_nan=False)  # stricter than save_backtest_to_db, which keeps json.dumps' default allow_nan=True


def test_all_winning_trades_cap_profit_factor():
    trades, metrics = _run([100, 101, 102, 103, 104, 105, 106, 107])
    assert all(t["net_return"] > 0 for t in trades)
    assert metrics["losses"] == 0
    assert metrics["profit_factor"] == 99.9
    _assert_strict_finite(metrics)


def test_all_flat_trades_report_zero_profit_factor():
    trades, metrics = _run([100] * 8)
    assert all(t["net_return"] == 0 for t in trades)
    assert metrics["wins"] == 0 and metrics["losses"] == 2
    assert metrics["profit_factor"] == 0.0
    _assert_strict_finite(metrics)


def test_mixed_trades_keep_gross_profit_over_gross_loss():
    # Trade 1: 101 -> 110 (win); trade 2: 110 -> 99 (loss).
    trades, metrics = _run([100, 101, 105, 110, 110, 105, 99, 99])
    returns = [t["net_return"] for t in trades]
    assert returns[0] > 0 > returns[1]
    expected = returns[0] / abs(returns[1])
    assert metrics["profit_factor"] == pytest.approx(expected)
    _assert_strict_finite(metrics)
