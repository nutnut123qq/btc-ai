import json

import numpy as np
import pytest

from technical_event_evidence import (
    INTERVAL_MS,
    BarData,
    EvaluatorConfig,
    EventOccurrence,
    ModuleEvents,
    build_bar_data,
    evaluate_all,
    evaluate_event_type,
    moving_block_bootstrap_mean_ci,
    pattern_decision_offset,
    wilson_interval,
    write_evidence,
)


def _bars(count: int = 500) -> BarData:
    rows = []
    close = 100.0
    for i in range(count):
        # Alternating short regimes avoid a permanently one-sided baseline.
        close *= 1.002 if (i // 5) % 2 == 0 else 0.998
        open_ms = i * INTERVAL_MS
        rows.append((open_ms, open_ms + INTERVAL_MS - 1, close))
    return build_bar_data(rows)


def test_build_bar_data_rejects_gap_outcomes_and_uses_closed_next_bar():
    rows = [
        (0, INTERVAL_MS - 1, 100.0),
        (INTERVAL_MS, 2 * INTERVAL_MS - 1, 101.0),
        (3 * INTERVAL_MS, 4 * INTERVAL_MS - 1, 102.0),
    ]
    bars = build_bar_data(rows)
    assert bars.next_returns[0] == pytest.approx(0.01)
    assert bars.label_available_ms[0] == 2 * INTERVAL_MS - 1
    assert np.isnan(bars.next_returns[1])
    assert bars.label_available_ms[1] == -1


def test_pattern_availability_uses_frozen_length_map_over_bad_stored_category():
    assert pattern_decision_offset("MorningStar", "Double") == 2
    assert pattern_decision_offset("EveningStar", "Double") == 2
    assert pattern_decision_offset("BullishEngulfing", "Double") == 1
    assert pattern_decision_offset("Doji", "Single") == 0
    assert pattern_decision_offset("UnknownFuturePattern", "Double") is None


def test_expanding_event_prior_is_strictly_available_before_each_decision():
    bars = _bars(450)
    occurrences = [EventOccurrence("fixture", i, bars.contexts[i]) for i in range(20, 430)]
    config = EvaluatorConfig(
        min_event_history=10,
        min_context_history=10,
        bootstrap_samples=40,
        block_size_events=4,
    )
    original = evaluate_event_type(bars, occurrences[:300], config)

    changed_returns = bars.next_returns.copy()
    changed_labels = bars.next_positive.copy()
    changed_returns[350:] = 0.9
    changed_labels[350:] = 1
    mutated = BarData(
        bars.open_ms,
        bars.close_ms,
        bars.close,
        bars.contexts,
        changed_returns,
        changed_labels,
        bars.label_available_ms,
    )
    repeated = evaluate_event_type(mutated, occurrences[:300], config)
    assert original == repeated


def test_all_modules_report_unavailable_explicitly_without_fabricating_trials():
    bars = _bars(120)
    unavailable = ModuleEvents("unavailable", "missing availability", "explicit timestamp required", (), {})
    modules = {name: unavailable for name in ("candle_patterns", "volume_anomaly", "market_regime", "causal_smc")}
    report = evaluate_all(
        bars,
        modules,
        EvaluatorConfig(min_event_history=5, min_context_history=5, bootstrap_samples=20),
    )
    assert all(module["status"] == "unavailable" for module in report["modules"].values())
    assert len(report["trials"]) == 4
    assert report["promotionAllowed"] is False
    assert report["pnlClaim"] is False


def test_familywise_bootstrap_refuses_unresolved_adjusted_tail():
    bars = _bars(120)
    event = EventOccurrence("fixture", 20, bars.contexts[20])
    evaluable = ModuleEvents("evaluable", None, "fixture", (event,), {})
    unavailable = ModuleEvents("unavailable", "fixture", "fixture", (), {})
    modules = {
        "candle_patterns": evaluable,
        "volume_anomaly": unavailable,
        "market_regime": unavailable,
        "causal_smc": unavailable,
    }
    with pytest.raises(ValueError, match="bootstrap_samples is too small"):
        evaluate_all(
            bars,
            modules,
            EvaluatorConfig(min_event_history=5, min_context_history=5, bootstrap_samples=20),
        )


def test_intervals_are_finite_and_adverse_signal_is_preserved():
    interval = wilson_interval(60, 100)
    assert interval is not None and 0 < interval["lower"] < interval["upper"] < 1
    ci = moving_block_bootstrap_mean_ci(
        np.full(80, -0.1), samples=100, block_size=8, alpha=0.05, seed=7
    )
    assert ci is not None and ci["upper"] < 0


def test_immutable_artifacts_allow_identical_replay_and_reject_collision(tmp_path):
    manifest = {"manifestSha256": "fixture"}
    evaluation = {"trials": [{"module": "fixture", "status": "inconclusive"}]}
    paths = write_evidence(tmp_path, manifest, evaluation)
    repeated = write_evidence(tmp_path, manifest, evaluation)
    assert paths == repeated
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["manifest"] == manifest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_evidence(tmp_path, manifest, {"trials": []})
