import copy
import hashlib
import json

import numpy as np
import pytest

import ml_evidence_v3 as v3
from ml_evidence_walkforward import BenchmarkData, CAUSAL_FEATURE_NAMES


def _dataset(rows=144, *, seed=71):
    rng = np.random.default_rng(seed)
    feature_dim = 5 * len(CAUSAL_FEATURE_NAMES)
    features = rng.normal(size=(rows, feature_dim))
    signal = features[:, 0] + 0.4 * features[:, -1]
    labels = np.where(signal > 0.45, 1, np.where(signal < -0.45, -1, 0)).astype(np.int8)
    decision = np.arange(1, rows + 1, dtype=np.int64) * 1_000
    available = decision + 1_000
    return BenchmarkData(features, labels, decision, available, features[:, -1])


def _config():
    return v3.V3Config(
        min_fit_rows=48,
        calibration_rows=24,
        test_rows=24,
        step_rows=24,
        bootstrap_samples=40,
        bootstrap_block_sizes_rows=(2, 4),
        adaptive_prior_windows_rows=(12, 24),
        minimum_gate_samples=48,
        minimum_class_samples=2,
        random_state=5,
    )


def test_feature_groups_partition_every_causal_feature_once():
    flattened = [feature for group in v3.FEATURE_GROUPS.values() for feature in group]

    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(CAUSAL_FEATURE_NAMES)


def test_adaptive_prior_only_uses_labels_available_at_decision_time():
    data = _dataset(rows=40)
    available = data.label_available_times_ms.copy()
    available[:30] = data.decision_times_ms[35] + 1
    delayed = BenchmarkData(
        data.features,
        data.labels,
        data.decision_times_ms,
        available,
        data.last_returns,
    )
    indices = np.array([35], dtype=np.int64)
    first = v3._baseline_probabilities(delayed, indices, _config())
    changed_labels = delayed.labels.copy()
    changed_labels[:30] = 1
    changed = BenchmarkData(
        delayed.features,
        changed_labels,
        delayed.decision_times_ms,
        delayed.label_available_times_ms,
        delayed.last_returns,
    )
    second = v3._baseline_probabilities(changed, indices, _config())

    for name in (v3.EXPANDING_PRIOR, "adaptive_class_prior_12_rows", "adaptive_class_prior_24_rows"):
        np.testing.assert_array_equal(first[name], second[name])


def test_v3_has_row_predictions_same_protocol_hgb_ablations_and_sensitivity():
    result = v3.evaluate_walk_forward_v3(_dataset(), _config())
    report = result.evaluation

    assert len(result.prediction_rows) == report["evaluationRows"]
    assert len(report["hgbFeatureGroupAblation"]) == len(v3.FEATURE_GROUPS)
    assert {row["blockSizeRows"] for row in report["pairedBrierSensitivity"]} == {2, 4}
    assert report["promotionGate"]["promotionAllowed"] is False
    assert report["negativeResultPolicy"]
    expected_trials = {
        v3.FULL_MODEL,
        *(f"hgb_without_{group}" for group in v3.FEATURE_GROUPS),
        *v3.declared_baselines(_config()),
    }
    for row in result.prediction_rows:
        assert set(row["probabilities"]) == expected_trials
        assert all(abs(sum(probability) - 1.0) < 1e-9 for probability in row["probabilities"].values())
    for fold in report["folds"]:
        assert fold["maxFitLabelAvailableMs"] <= fold["calibrationStartMs"]
        assert fold["maxCalibrationLabelAvailableMs"] <= fold["testStartMs"]


def test_future_rows_do_not_change_completed_v3_predictions():
    original = _dataset(rows=120)
    extended = _dataset(rows=168)
    cutoff = int(original.label_available_times_ms[-1])

    first = v3.evaluate_walk_forward_v3(original, _config(), evaluation_end_ms=cutoff)
    second = v3.evaluate_walk_forward_v3(extended, _config(), evaluation_end_ms=cutoff)

    assert first == second


def test_bundle_is_content_addressed_replayable_and_fails_closed_on_tamper(tmp_path):
    data = _dataset(rows=120)
    config = _config()
    cutoff = int(data.label_available_times_ms[-1])
    result = v3.evaluate_walk_forward_v3(data, config)
    manifest = v3.build_v3_manifest(
        data,
        config,
        decision_cutoff_ms=cutoff,
        source="immutable-npz-replay",
    )

    paths = v3.write_evidence_bundle(tmp_path, data, manifest, result)
    verified = v3.verify_evidence_bundle(paths["bundleManifest"])
    v3.write_evidence_bundle(tmp_path, data, manifest, result)

    assert verified["schemaVersion"] == v3.SCHEMA_VERSION
    assert set(verified["artifacts"]) == {"datasetSnapshot", "rowPredictions", "report"}
    with np.load(paths["datasetSnapshot"], allow_pickle=False) as snapshot:
        np.testing.assert_array_equal(snapshot["X"], data.features)
        np.testing.assert_array_equal(snapshot["y"], data.labels)
    first_prediction = json.loads(paths["rowPredictions"].read_text().splitlines()[0])
    assert {"foldId", "decisionTimeMs", "label", "probabilities"} <= first_prediction.keys()

    paths["rowPredictions"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        v3.verify_evidence_bundle(paths["bundleManifest"])


def test_npz_snapshot_bytes_and_bundle_paths_are_deterministic(tmp_path):
    data = _dataset(rows=120)
    config = _config()
    cutoff = int(data.label_available_times_ms[-1])
    result = v3.evaluate_walk_forward_v3(data, config)
    manifest = v3.build_v3_manifest(
        data,
        config,
        decision_cutoff_ms=cutoff,
        source="immutable-npz-replay",
    )

    assert v3._snapshot_bytes(data) == v3._snapshot_bytes(data)
    first = v3.write_evidence_bundle(tmp_path, data, manifest, result)
    second = v3.write_evidence_bundle(tmp_path, data, manifest, result)

    assert first == second
    assert first["datasetSnapshot"].read_bytes() == v3._snapshot_bytes(data)


def _snapshot_arrays(data):
    from io import BytesIO

    with np.load(BytesIO(v3._snapshot_bytes(data)), allow_pickle=False) as snapshot:
        return {name: snapshot[name].copy() for name in snapshot.files}


def _write_snapshot(path, arrays):
    np.savez_compressed(path, **arrays)


def test_v3_snapshot_loader_rejects_non_exact_keys_dtypes_width_names_and_schema(tmp_path):
    data = _dataset(rows=40)
    mutations = []

    extra_key = _snapshot_arrays(data)
    extra_key["unexpected"] = np.asarray(1, dtype=np.int64)
    mutations.append(extra_key)

    wrong_dtype = _snapshot_arrays(data)
    wrong_dtype["X"] = wrong_dtype["X"].astype(np.float32)
    mutations.append(wrong_dtype)

    wrong_width = _snapshot_arrays(data)
    wrong_width["X"] = wrong_width["X"][:, :-1]
    mutations.append(wrong_width)

    wrong_names = _snapshot_arrays(data)
    wrong_names["causal_feature_names"] = wrong_names["causal_feature_names"][::-1]
    mutations.append(wrong_names)

    wrong_schema = _snapshot_arrays(data)
    wrong_schema["schema_version"] = np.asarray("btc-ml-evidence-snapshot/v999")
    mutations.append(wrong_schema)

    for index, arrays in enumerate(mutations):
        path = tmp_path / f"invalid-{index}.npz"
        _write_snapshot(path, arrays)
        with pytest.raises(ValueError):
            v3._load_snapshot(path, expected_window_size=5)


def test_replay_loader_and_manifest_fail_closed_when_any_row_exceeds_cutoff(tmp_path):
    data = _dataset(rows=120)
    snapshot = tmp_path / "future-row.dataset.npz"
    snapshot.write_bytes(v3._snapshot_bytes(data))
    cutoff = int(data.label_available_times_ms[-2])

    with pytest.raises(ValueError, match="beyond decision cutoff"):
        v3.load_v3_npz(snapshot, decision_cutoff_ms=cutoff, window_size=5)
    with pytest.raises(ValueError, match="beyond decision cutoff"):
        v3.build_v3_manifest(
            data,
            _config(),
            decision_cutoff_ms=cutoff,
            source="immutable-npz-replay",
        )


def test_database_manifest_allows_only_redacted_exact_database_identity(monkeypatch):
    data = _dataset(rows=120)
    cutoff = int(data.label_available_times_ms[-1])
    monkeypatch.setattr(
        v3.v2,
        "get_db_params",
        lambda: {
            "host": "localhost",
            "port": 5432,
            "database": "bitcoin_analyst",
            "user": "not-persisted",
            "password": "not-persisted",
        },
    )
    manifest = v3.build_v3_manifest(
        data,
        _config(),
        decision_cutoff_ms=cutoff,
        source="postgresql-readonly",
    )

    assert manifest["dataProvenance"]["databaseIdentity"] == {
        "host": "localhost",
        "port": 5432,
        "database": "bitcoin_analyst",
        "passwordExcluded": True,
    }
    malformed = copy.deepcopy(manifest)
    malformed["dataProvenance"]["databaseIdentity"]["password"] = "must-not-appear"
    with pytest.raises(ValueError, match="data provenance schema mismatch|database identity"):
        v3._validate_research_manifest(malformed)


def test_writer_rejects_data_that_no_longer_matches_cutoff_bound_manifest(tmp_path):
    data = _dataset(rows=120)
    config = _config()
    cutoff = int(data.label_available_times_ms[-1])
    result = v3.evaluate_walk_forward_v3(data, config)
    manifest = v3.build_v3_manifest(
        data, config, decision_cutoff_ms=cutoff, source="immutable-npz-replay"
    )
    changed_availability = data.label_available_times_ms.copy()
    changed_availability[-1] = cutoff + 1
    changed = BenchmarkData(
        data.features,
        data.labels,
        data.decision_times_ms,
        changed_availability,
        data.last_returns,
    )

    with pytest.raises(ValueError, match="beyond decision cutoff"):
        v3.write_evidence_bundle(tmp_path, changed, manifest, result)


def test_gate_requires_every_baseline_metric_interval_at_every_block_size(monkeypatch):
    config = _config()
    labels = np.tile(np.asarray([-1, 0, 1], dtype=np.int8), 24)
    candidate = np.full((len(labels), 3), 0.1, dtype=np.float64)
    candidate[np.arange(len(labels)), np.searchsorted(v3.v2.CLASSES, labels)] = 0.8
    uniform = np.full((len(labels), 3), 1 / 3, dtype=np.float64)
    trials = {
        v3.FULL_MODEL: candidate,
        **{f"hgb_without_{group}": candidate for group in v3.FEATURE_GROUPS},
        **{name: uniform for name in v3.declared_baselines(config)},
    }

    def interval(_values, *, block_size, **_kwargs):
        return (0.01, 0.02) if block_size == 2 else (-0.01, 0.02)

    monkeypatch.setattr(v3.v2, "_paired_block_interval", interval)
    evidence = v3._statistical_evidence(labels, trials, config)

    assert len(evidence["baselineComparisons"]) == len(v3.declared_baselines(config))
    assert all(set(item["metrics"]) == {"brier", "logLoss"} for item in evidence["baselineComparisons"])
    assert all(
        len(metric["intervals"]) == len(config.bootstrap_block_sizes_rows)
        for comparison in evidence["baselineComparisons"]
        for metric in comparison["metrics"].values()
    )
    assert evidence["strongestBaselineByMetric"].keys() == {"brier", "logLoss"}
    assert not evidence["gateChecks"]["allBaselineMetricIntervalsPositiveAcrossBlockSizes"]


def test_semantic_verifier_rejects_report_claims_metrics_rows_and_manifest_schema():
    data = _dataset(rows=120)
    config = _config()
    cutoff = int(data.label_available_times_ms[-1])
    result = v3.evaluate_walk_forward_v3(data, config)
    manifest = v3.build_v3_manifest(
        data, config, decision_cutoff_ms=cutoff, source="immutable-npz-replay"
    )

    altered = copy.deepcopy(result.evaluation)
    altered["evidenceTier"] = "confirmatory"
    with pytest.raises(ValueError, match="semantic report mismatch"):
        v3._validate_prediction_semantics(
            data, manifest, altered, result.prediction_rows, recompute_statistics=True
        )

    altered = copy.deepcopy(result.evaluation)
    altered["metrics"][v3.FULL_MODEL]["brier"] += 0.01
    with pytest.raises(ValueError, match="semantic report mismatch"):
        v3._validate_prediction_semantics(
            data, manifest, altered, result.prediction_rows, recompute_statistics=True
        )

    rows = [copy.deepcopy(row) for row in result.prediction_rows]
    rows[0]["unexpected"] = True
    with pytest.raises(ValueError, match="prediction row schema mismatch"):
        v3._validate_prediction_semantics(
            data, manifest, result.evaluation, rows, recompute_statistics=True
        )

    rows = [copy.deepcopy(row) for row in result.prediction_rows]
    rows[0]["label"] = -rows[0]["label"] if rows[0]["label"] else 1
    with pytest.raises(ValueError, match="labels do not match"):
        v3._validate_prediction_semantics(
            data, manifest, result.evaluation, rows, recompute_statistics=True
        )

    malformed_manifest = copy.deepcopy(manifest)
    malformed_manifest["unexpected"] = True
    malformed_manifest.pop("manifestSha256")
    malformed_manifest["manifestSha256"] = hashlib.sha256(
        v3._canonical_json(malformed_manifest).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="manifest schema mismatch"):
        v3._validate_research_manifest(malformed_manifest)


def _write_rehashed_bundle(path, bundle):
    core = dict(bundle)
    core.pop("bundleManifestSha256", None)
    digest = hashlib.sha256(v3._canonical_json(core).encode("utf-8")).hexdigest()
    payload = core | {"bundleManifestSha256": digest}
    output = path / f"{digest}.manifest.json"
    output.write_text(json.dumps(payload), encoding="utf-8")
    return output


def test_bundle_verifier_rejects_wrong_artifact_keys_suffix_and_declared_size(tmp_path):
    data = _dataset(rows=120)
    config = _config()
    cutoff = int(data.label_available_times_ms[-1])
    result = v3.evaluate_walk_forward_v3(data, config)
    manifest = v3.build_v3_manifest(
        data, config, decision_cutoff_ms=cutoff, source="immutable-npz-replay"
    )
    paths = v3.write_evidence_bundle(tmp_path, data, manifest, result)
    original = json.loads(paths["bundleManifest"].read_text(encoding="utf-8"))

    wrong_keys = copy.deepcopy(original)
    wrong_keys["artifacts"]["unexpected"] = wrong_keys["artifacts"]["report"]
    with pytest.raises(ValueError, match="artifact keys"):
        v3.verify_evidence_bundle(_write_rehashed_bundle(tmp_path, wrong_keys))

    wrong_suffix = copy.deepcopy(original)
    wrong_suffix["artifacts"]["report"]["file"] = (
        wrong_suffix["artifacts"]["report"]["sha256"] + ".wrong.json"
    )
    with pytest.raises(ValueError, match="filename is not content-addressed"):
        v3.verify_evidence_bundle(_write_rehashed_bundle(tmp_path, wrong_suffix))

    wrong_size = copy.deepcopy(original)
    wrong_size["artifacts"]["report"]["bytes"] += 1
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        v3.verify_evidence_bundle(_write_rehashed_bundle(tmp_path, wrong_size))


def test_report_discloses_logistic_calibration_retrospective_tier_and_restart_risk():
    report = v3.evaluate_walk_forward_v3(_dataset(), _config()).evaluation

    assert "multinomial logistic recalibration" in report["protocol"]["calibration"]
    assert report["evidenceTier"] == "retrospective_selection_aware"
    assert report["promotionGate"]["confirmatory"] is False
    assert report["promotionGate"]["promotionAllowed"] is False
    assert report["executionPlan"]["checkpointResume"] is False
    assert any("must restart" in limitation for limitation in report["limitations"])
