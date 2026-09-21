import json

import numpy as np
import pytest

import ml_evidence_walkforward as evaluator

from ml_evidence_walkforward import (
    CAUSAL_FEATURE_NAMES,
    DECLARED_BASELINES,
    DECLARED_MODEL_FAMILY,
    STORED_FEATURE_NAMES,
    BenchmarkData,
    EvaluatorConfig,
    build_manifest,
    drop_noncausal_stored_features,
    evaluate_walk_forward,
    extract_causal_feature,
    write_evidence,
)


def _dataset(rows=180, *, seed=17, predictive=False):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 8))
    if predictive:
        labels = np.where(features[:, 0] > 0.4, 1, np.where(features[:, 0] < -0.4, -1, 0))
    else:
        # Balanced but deliberately unrelated to the feature matrix.
        labels = np.resize(np.array([-1, 0, 1], dtype=np.int8), rows)
    decision = np.arange(1, rows + 1, dtype=np.int64) * 1_000
    label_available = decision + 1_000
    last_return = features[:, -1]
    return BenchmarkData(features, labels.astype(np.int8), decision, label_available, last_return)


def _config():
    return EvaluatorConfig(
        min_fit_rows=48,
        calibration_rows=24,
        test_rows=24,
        step_rows=24,
        block_size_rows=4,
        bootstrap_samples=100,
        minimum_gate_samples=48,
        minimum_class_samples=5,
        random_state=9,
    )


def test_stored_active_rule_count_is_removed_from_every_bar():
    stored_stride = len(STORED_FEATURE_NAMES)
    causal_stride = len(CAUSAL_FEATURE_NAMES)
    matrix = np.arange(2 * 3 * stored_stride, dtype=np.float64).reshape(2, 3 * stored_stride)
    filtered = drop_noncausal_stored_features(matrix, window_size=3)

    assert filtered.shape == (2, 3 * causal_stride)
    active_rule_offset = STORED_FEATURE_NAMES.index("ActiveRuleCount")
    removed = {bar * stored_stride + active_rule_offset for bar in range(3)}
    assert set(matrix[0]) - set(filtered[0]) == {matrix[0, index] for index in removed}


def test_last_bar_return_is_extracted_by_name_after_noncausal_column_is_removed():
    window_size = 3
    stored_stride = len(STORED_FEATURE_NAMES)
    matrix = np.zeros((2, window_size, stored_stride), dtype=np.float64)
    close_return_offset = STORED_FEATURE_NAMES.index("ClosePctChange1")
    body_offset = STORED_FEATURE_NAMES.index("BodyPct")
    active_rule_offset = STORED_FEATURE_NAMES.index("ActiveRuleCount")
    matrix[:, -1, close_return_offset] = [0.125, -0.25]
    matrix[:, -1, body_offset] = [91.0, 92.0]
    matrix[:, :, active_rule_offset] = 999.0

    causal = drop_noncausal_stored_features(matrix.reshape(2, -1), window_size)
    extracted = extract_causal_feature(
        causal,
        window_size=window_size,
        feature_name="ClosePctChange1",
    )

    np.testing.assert_array_equal(extracted, np.array([0.125, -0.25]))
    assert not np.isin(extracted, [91.0, 92.0]).any()


def test_outer_folds_purge_labels_and_keep_fit_calibration_before_test():
    report = evaluate_walk_forward(_dataset(), _config())

    assert len(report["folds"]) >= 2
    for fold in report["folds"]:
        assert fold["maxFitLabelAvailableMs"] <= fold["calibrationStartMs"]
        assert fold["maxCalibrationLabelAvailableMs"] <= fold["testStartMs"]
        assert fold["calibrationThroughMs"] < fold["testStartMs"]


def test_appending_future_rows_cannot_change_completed_predictions_or_metrics():
    original = _dataset(rows=144)
    extended = _dataset(rows=192)
    cutoff = int(original.decision_times_ms[-1])

    first = evaluate_walk_forward(original, _config(), evaluation_end_ms=cutoff)
    second = evaluate_walk_forward(extended, _config(), evaluation_end_ms=cutoff)

    assert first == second


def test_every_declared_trial_uses_identical_timestamp_coverage():
    report = evaluate_walk_forward(_dataset(), _config())
    trials = report["trials"]

    assert {trial["trial"] for trial in trials} == set(DECLARED_MODEL_FAMILY + DECLARED_BASELINES)
    assert len({trial["evaluationTimestampSha256"] for trial in trials}) == 1
    assert all(trial["coverage"] == 1.0 for trial in trials)
    assert report["promotionGate"]["checks"]["identicalTimestampCoverage"] is True


def test_evaluator_fails_closed_when_one_baseline_has_different_row_coverage(monkeypatch):
    original = evaluator._baseline_probabilities

    def mismatched_baselines(data, test_indices, config):
        outputs = original(data, test_indices, config)
        outputs["momentum"] = outputs["momentum"][:-1]
        return outputs

    monkeypatch.setattr(evaluator, "_baseline_probabilities", mismatched_baselines)
    with pytest.raises(ValueError, match="probability coverage mismatch for momentum"):
        evaluate_walk_forward(_dataset(), _config())


def test_coverage_validator_rejects_same_rows_in_different_timestamp_order():
    expected = np.array([1_000, 2_000, 3_000], dtype=np.int64)
    probabilities = {
        name: np.full((3, 3), 1.0 / 3.0)
        for name in DECLARED_MODEL_FAMILY + DECLARED_BASELINES
    }
    timestamps = {
        name: expected.copy() for name in DECLARED_MODEL_FAMILY + DECLARED_BASELINES
    }
    timestamps["momentum"] = expected[::-1]

    with pytest.raises(ValueError, match="timestamp coverage mismatch for momentum"):
        evaluator._validate_probability_coverage(probabilities, timestamps, expected)


def test_baseline_history_excludes_labels_unavailable_at_each_decision():
    data = _dataset(rows=12)
    delayed_availability = data.label_available_times_ms.copy()
    delayed_availability[:9] = data.decision_times_ms[10] + 1
    delayed = BenchmarkData(
        data.features,
        data.labels,
        data.decision_times_ms,
        delayed_availability,
        data.last_returns,
    )
    config = _config()
    test_indices = np.array([10], dtype=np.int64)

    original = evaluator._baseline_probabilities(delayed, test_indices, config)
    changed_labels = delayed.labels.copy()
    changed_labels[:9] = 1
    changed = BenchmarkData(
        delayed.features,
        changed_labels,
        delayed.decision_times_ms,
        delayed.label_available_times_ms,
        delayed.last_returns,
    )
    replay = evaluator._baseline_probabilities(changed, test_indices, config)

    np.testing.assert_array_equal(
        original["rolling_class_probs"], replay["rolling_class_probs"]
    )


def test_negative_or_inconclusive_candidate_is_recorded_and_not_promoted():
    report = evaluate_walk_forward(_dataset(predictive=False), _config())
    candidates = [trial for trial in report["trials"] if trial["kind"] == "candidate"]

    assert all(trial["status"] in {"negative", "inconclusive", "positive"} for trial in candidates)
    assert any(trial["status"] != "positive" for trial in candidates)
    assert report["negativeResultIsValid"] is True
    assert report["promotionGate"]["passed"] is False
    assert report["promotionGate"]["promotionAllowed"] is False


def test_manifest_has_replayable_code_data_runtime_provenance_and_is_data_bound():
    data = _dataset(rows=144)
    cutoff = int(data.label_available_times_ms[-1])
    first = build_manifest(data, _config(), decision_cutoff_ms=cutoff, source="npz:test-fixture")
    changed_data = BenchmarkData(
        data.features.copy(),
        data.labels.copy(),
        data.decision_times_ms.copy(),
        data.label_available_times_ms.copy(),
        data.last_returns.copy(),
    )
    changed_data.features[0, 0] += 0.01
    second = build_manifest(changed_data, _config(), decision_cutoff_ms=cutoff, source="npz:test-fixture")

    assert first["dataProvenance"]["datasetSha256"]
    assert first["codeProvenance"]["evaluatorSha256"]
    assert first["codeProvenance"]["researchContractSha256"]
    assert first["runtimeDependencies"]["scikitLearn"]
    assert first["manifestSha256"] != second["manifestSha256"]


def test_evidence_files_are_immutable_and_ledger_contains_all_trials(tmp_path):
    data = _dataset(rows=144)
    config = _config()
    cutoff = int(data.label_available_times_ms[-1])
    manifest = build_manifest(data, config, decision_cutoff_ms=cutoff, source="npz:test-fixture")
    evaluation = evaluate_walk_forward(data, config)

    paths = write_evidence(tmp_path, manifest, evaluation)
    write_evidence(tmp_path, manifest, evaluation)
    ledger = [json.loads(line) for line in paths["trialLedger"].read_text().splitlines()]

    assert len(ledger) == len(DECLARED_MODEL_FAMILY + DECLARED_BASELINES)
    assert {row["trial"] for row in ledger} == set(DECLARED_MODEL_FAMILY + DECLARED_BASELINES)
    assert paths["report"].exists()
