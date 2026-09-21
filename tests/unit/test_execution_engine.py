import numpy as np
import pytest

from backtest_strategy import compute_metrics, run_execution_stress_scenarios, simulate_trades
from execution_engine import (
    ExecutionContract,
    ExecutionCostSpec,
    ExecutionMode,
    calculate_round_trip,
    evaluate_decision_gate,
    resolve_tp_sl_bar,
)


class FixedModel:
    def __init__(self, label_index: int):
        self.label_index = label_index

    def predict_proba(self, values):
        result = np.full((len(values), 3), 0.05, dtype=float)
        result[:, self.label_index] = 0.90
        return result


def _bars(count=8, step_ms=3_600_000):
    base = 1_700_000_000_000
    rows = []
    for index in range(count):
        timestamp = base + index * step_ms
        open_price = 100.0 + index
        rows.append((timestamp, open_price, open_price + 2, open_price - 2, 150.0 + index, 1_000.0))
    return rows


def _signal(bar, value=1.0):
    return (bar[0], [value], 1, 0.01)


def test_signal_cannot_fill_on_same_bar_and_uses_next_open():
    bars = _bars()
    trades = simulate_trades(
        [_signal(bars[0])],
        bars,
        2 * 3_600_000,
        model=FixedModel(2),
        meta={"model_name": "XGB_test"},
        fee_bps=10,
        slippage_bps=0,
    )

    assert len(trades) == 1
    assert trades[0]["signal_time"] == bars[0][0]
    assert trades[0]["decision_time"] == bars[1][0]
    assert trades[0]["entry_time"] == bars[1][0]
    assert trades[0]["entry_price"] == bars[1][1]
    assert trades[0]["entry_price"] != bars[0][4]


def test_future_prices_cannot_change_the_prior_decision_or_entry():
    bars = _bars()
    mutated = list(bars)
    future = list(mutated[4])
    future[1:5] = [9_000.0, 9_100.0, 8_900.0, 9_050.0]
    mutated[4] = tuple(future)
    kwargs = {
        "model": FixedModel(2),
        "meta": {"model_name": "XGB_test"},
        "fee_bps": 0,
        "slippage_bps": 0,
    }
    original_trade = simulate_trades([_signal(bars[0])], bars, 3 * 3_600_000, **kwargs)[0]
    mutated_trade = simulate_trades([_signal(bars[0])], mutated, 3 * 3_600_000, **kwargs)[0]

    for field in ("signal_time", "decision_time", "entry_time", "side", "entry_price", "confidence"):
        assert mutated_trade[field] == original_trade[field]
    assert mutated_trade["exit_price"] != original_trade["exit_price"]


def test_sequential_capital_reconciles_and_overlapping_signal_is_skipped():
    bars = _bars()
    rows = [_signal(bars[0]), _signal(bars[1]), _signal(bars[3])]
    trades = simulate_trades(
        rows,
        bars,
        2 * 3_600_000,
        model=FixedModel(2),
        meta={"model_name": "XGB_test"},
        fee_bps=10,
        slippage_bps=0,
        initial_capital=10_000,
        capital_fraction_per_trade=0.5,
    )

    assert [trade["signal_time"] for trade in trades] == [bars[0][0], bars[3][0]]
    assert trades[1]["capital_before"] == pytest.approx(trades[0]["capital_after"])
    for trade in trades:
        assert trade["capital_after"] == pytest.approx(
            trade["capital_before"] + trade["position_notional"] * trade["net_return"]
        )
        entry_fee = trade["costs"]["feePerSideBps"] / 10_000
        assert trade["position_notional"] * (1 + entry_fee) <= trade["capital_before"] * 0.5 + 1e-9
    metrics = compute_metrics(trades, bars, initial_capital=10_000)
    assert metrics["final_equity"] == pytest.approx(trades[-1]["capital_after"])
    assert 0 < metrics["exposure_fraction"] <= 1
    assert metrics["turnover_initial_capital_multiple"] > 0


def test_long_and_short_cost_math_and_spot_short_rejection():
    costs = ExecutionCostSpec(fee_per_side_bps=10, slippage_per_side_bps=5)
    simulated = ExecutionContract(mode=ExecutionMode.SPOT_REFERENCE_DERIVATIVE_SIMULATION)
    long_fill = calculate_round_trip(
        side="long",
        entry_reference_price=100,
        exit_reference_price=110,
        costs=costs,
        contract=simulated,
    )
    short_fill = calculate_round_trip(
        side="short",
        entry_reference_price=100,
        exit_reference_price=90,
        costs=costs,
        contract=simulated,
    )

    assert long_fill.entry_fill_price == pytest.approx(100.05)
    assert long_fill.exit_fill_price == pytest.approx(109.945)
    assert long_fill.net_return < long_fill.gross_return
    assert short_fill.entry_fill_price == pytest.approx(99.95)
    assert short_fill.exit_fill_price == pytest.approx(90.045)
    assert short_fill.net_return < short_fill.gross_return

    with pytest.raises(ValueError, match="spot execution contract cannot open a short"):
        calculate_round_trip(
            side="short",
            entry_reference_price=100,
            exit_reference_price=90,
            costs=costs,
            contract=ExecutionContract(mode=ExecutionMode.SPOT),
        )
    with pytest.raises(ValueError, match="funding PnL"):
        calculate_round_trip(
            side="long",
            entry_reference_price=100,
            exit_reference_price=101,
            costs=costs,
            contract=ExecutionContract(mode=ExecutionMode.PERPETUAL),
        )


def test_same_bar_tp_sl_is_flagged_and_reported_as_bounds():
    conservative = resolve_tp_sl_bar(
        side="long",
        bar_high=112,
        bar_low=88,
        take_profit=110,
        stop_loss=90,
        scenario="conservative",
    )
    optimistic = resolve_tp_sl_bar(
        side="long",
        bar_high=112,
        bar_low=88,
        take_profit=110,
        stop_loss=90,
        scenario="optimistic",
    )

    assert conservative.ambiguous and optimistic.ambiguous
    assert conservative.exit_price == 90
    assert optimistic.exit_price == 110
    assert conservative.lower_bound_exit_price == 90
    assert conservative.upper_bound_exit_price == 110


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"model_available": False}, "model-unavailable"),
        ({"signal_bar_closed": False}, "signal-bar-not-finalized"),
        ({"decision_ms": 999}, "signal-not-yet-available"),
        ({"latest_observation_ms": 1_001}, "future-observation"),
        ({"decision_ms": 2_001, "max_staleness_ms": 1_000}, "market-data-stale"),
    ],
)
def test_decision_gate_abstains_on_untruthful_inputs(kwargs, reason):
    defaults = {
        "model_available": True,
        "signal_bar_closed": True,
        "signal_available_ms": 1_000,
        "decision_ms": 1_000,
        "latest_observation_ms": 1_000,
        "max_staleness_ms": 1_000,
    }
    defaults.update(kwargs)
    result = evaluate_decision_gate(**defaults)
    assert not result.allowed
    assert result.reason == reason


def test_cost_and_delay_stress_scenarios_are_declared():
    bars = _bars(10)
    scenarios = run_execution_stress_scenarios(
        [_signal(bars[0]), _signal(bars[4])],
        bars,
        2 * 3_600_000,
        model=FixedModel(2),
        meta={"model_name": "XGB_test"},
        fee_bps=10,
        slippage_bps=5,
    )
    assert set(scenarios) == {
        "cost_1x_delay_1_bar",
        "cost_1x_delay_2_bar",
        "cost_1.5x_delay_1_bar",
        "cost_1.5x_delay_2_bar",
        "cost_2x_delay_1_bar",
        "cost_2x_delay_2_bar",
    }
    assert scenarios["cost_2x_delay_1_bar"]["final_equity"] < scenarios["cost_1x_delay_1_bar"]["final_equity"]
