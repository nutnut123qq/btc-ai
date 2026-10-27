"""Version-stamp regression tests for backtest_strategy.save_backtest_to_db.

Backtest runs written by this script used to rely on the DB column defaults
("legacy-unversioned" / "Legacy" from migration
AddResearchValidityAndAlertDeduplication), so GET /api/backtest/runs with
includeLegacy=false hid every Python-produced run. The INSERT now carries
explicit PipelineVersion / EvaluationVersion / ValidityStatus provenance.

FINITE-1: the writer stamps "Valid" itself, so it must also refuse NaN/+-Inf
anywhere it would insert (columns, trades, MetricsJson, EquityCurveJson).
"""

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import backtest_strategy


def _run_info():
    return {
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "window_size": 5,
        "horizon": "4h",
        "model_name": "fixture-model",
        "start_ms": 1_700_000_000_000,
        "end_ms": 1_700_100_000_000,
        "fee_bps": 10.0,
        "slippage_bps": 5.0,
        "execution_mode": "spot_reference_derivative_simulation",
        "metrics": {
            "total_trades": 1,
            "win_rate": 1.0,
            "total_return_pct": 1.5,
            "buy_hold_return_pct": 1.0,
            "max_drawdown_pct": 0.5,
            "sharpe_ratio": 2.0,
            "sortino_ratio": 2.5,
            "profit_factor": 3.0,
            "final_equity": 10_150.0,
            "equity_curve": [{"time": 1_700_000_000_000, "equity": 10_000.0}],
        },
    }


def _trade():
    return {
        "signal_time": 1_700_000_000_000,
        "decision_time": 1_700_014_400_000,
        "entry_time": 1_700_014_400_000,
        "exit_time": 1_700_028_800_000,
        "side": "long",
        "entry_price": 50_000.0,
        "exit_price": 50_750.0,
        "gross_return": 0.015,
        "net_return": 0.014,
        "pnl_pct": 1.4,
        "confidence": 0.7,
        "true_label": 1,
        "target_return": 0.01,
    }


def _mock_db(monkeypatch):
    conn = Mock()
    cursor = Mock()
    conn.cursor.return_value = cursor
    cursor.fetchone.return_value = (42,)
    monkeypatch.setattr(backtest_strategy, "get_connection", lambda: conn)
    return conn, cursor


def _save_with_mocked_db(monkeypatch, trades):
    conn, cursor = _mock_db(monkeypatch)
    run_id = backtest_strategy.save_backtest_to_db(_run_info(), trades)
    return run_id, conn, cursor


def _insert_columns(sql):
    match = re.search(r"INSERT INTO \"BacktestRuns\" \(([^)]*)\)", sql, re.S)
    assert match, "expected an INSERT INTO BacktestRuns statement"
    return [name.strip().strip('"') for name in match.group(1).split(",")]


def test_run_insert_stamps_pipeline_evaluation_and_validity_columns(monkeypatch):
    run_id, conn, cursor = _save_with_mocked_db(monkeypatch, [])

    assert run_id == 42
    run_sql, run_params = cursor.execute.call_args_list[0].args
    columns = _insert_columns(run_sql)
    values_by_column = dict(zip(columns, run_params))

    assert values_by_column["PipelineVersion"] == backtest_strategy.BACKTEST_PIPELINE_VERSION
    assert values_by_column["EvaluationVersion"] == backtest_strategy.BACKTEST_EVALUATION_VERSION
    assert values_by_column["ValidityStatus"] == backtest_strategy.BACKTEST_VALIDITY_STATUS
    conn.commit.assert_called_once()
    cursor.close.assert_called_once()
    conn.close.assert_called_once()


def test_stamp_values_are_versioned_and_valid_for_classifier():
    # Mirrors ResearchRecordClassifier.IsUnversioned: a record counts as
    # versioned when both stamps are non-empty and not "legacy-unversioned".
    stamps = (
        backtest_strategy.BACKTEST_PIPELINE_VERSION,
        backtest_strategy.BACKTEST_EVALUATION_VERSION,
    )
    for stamp in stamps:
        assert stamp.strip()
        assert stamp.lower() != "legacy-unversioned"
    # Data-pipeline pin mirrors ResearchVersions.DataPipeline; if the C# data
    # pipeline version ever bumps, this pin forces a fresh semantics review.
    assert backtest_strategy.BACKTEST_PIPELINE_VERSION == "quant-pipeline-v3"
    # Distinct engine id: NOT the C# ensemble "evaluation-v2" lineage.
    assert backtest_strategy.BACKTEST_EVALUATION_VERSION != "evaluation-v2"
    # ValidityStatuses.Valid so includeLegacy=false listings can show the run.
    assert backtest_strategy.BACKTEST_VALIDITY_STATUS == "Valid"


def test_trade_inserts_keep_ledger_columns_and_share_run_id(monkeypatch):
    run_id, _conn, cursor = _save_with_mocked_db(monkeypatch, [_trade(), _trade()])

    trade_calls = [
        call for call in cursor.execute.call_args_list
        if 'INSERT INTO "BacktestTrades"' in call.args[0]
    ]
    assert len(trade_calls) == 2
    for call in trade_calls:
        assert call.args[1][0] == run_id


def test_finite_run_inserts_same_parameters_in_same_order(monkeypatch):
    _run_id, conn, cursor = _save_with_mocked_db(monkeypatch, [_trade()])

    metrics = _run_info()["metrics"]
    run_params = cursor.execute.call_args_list[0].args[1]
    assert run_params[:-1] == (
        "BTCUSDT", "4h", 5, "4h", "fixture-model",
        1_700_000_000_000, 1_700_100_000_000, 10.0, 5.0,
        1, 1.0, 1.5, 1.0, 0.5, 2.0, 2.5, 3.0, 10_150.0,
        json.dumps(metrics), json.dumps(metrics["equity_curve"]),
        backtest_strategy.BACKTEST_PIPELINE_VERSION,
        backtest_strategy.BACKTEST_EVALUATION_VERSION,
        backtest_strategy.BACKTEST_VALIDITY_STATUS,
    )
    assert isinstance(run_params[-1], datetime)
    assert cursor.execute.call_args_list[1].args[1] == (
        42, 1_700_014_400_000, 1_700_028_800_000, "long",
        50_000.0, 50_750.0, 0.015, 0.014, 1.4, 0.7, 1, 0.01,
    )
    assert cursor.execute.call_count == 2
    conn.commit.assert_called_once()


def _set_metric(name, value):
    return lambda run_info, trades: run_info["metrics"].__setitem__(name, value)


@pytest.mark.parametrize(
    "poison",
    [
        pytest.param(_set_metric("sharpe_ratio", math.inf), id="run-metric-inf"),
        pytest.param(_set_metric("profit_factor", math.nan), id="run-metric-nan"),
        pytest.param(_set_metric("win_rate", np.float32("nan")), id="run-metric-numpy-nan"),
        pytest.param(lambda run_info, trades: run_info.__setitem__("fee_bps", -math.inf), id="run-column-outside-metrics"),
        pytest.param(lambda run_info, trades: trades[1].__setitem__("net_return", math.nan), id="trade-net-return-nan"),
        pytest.param(lambda run_info, trades: trades[0].__setitem__("target_return", math.inf), id="trade-target-return-inf"),
        pytest.param(
            lambda run_info, trades: run_info["metrics"]["equity_curve"][0].__setitem__("equity", math.inf),
            id="equity-curve-inf",
        ),
        pytest.param(
            _set_metric("mark_to_market_equity_curve", [{"time": 1, "equity": math.nan}]),
            id="metrics-json-only-nan",
        ),
    ],
)
def test_non_finite_value_raises_before_any_insert(monkeypatch, poison):
    conn, cursor = _mock_db(monkeypatch)
    connect = Mock(return_value=conn)
    monkeypatch.setattr(backtest_strategy, "get_connection", connect)
    run_info, trades = _run_info(), [_trade(), _trade()]
    poison(run_info, trades)

    with pytest.raises(ValueError, match="non-finite"):
        backtest_strategy.save_backtest_to_db(run_info, trades)

    connect.assert_not_called()
    conn.cursor.assert_not_called()
    cursor.execute.assert_not_called()
    conn.commit.assert_not_called()
