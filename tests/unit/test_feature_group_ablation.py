import json

import numpy as np

from feature_group_ablation import (
    CAUSAL_FEATURE_NAMES,
    FEATURE_GROUPS,
    AblationConfig,
    AblationData,
    build_manifest,
    chronological_folds,
    evaluate_feature_groups,
    group_columns,
    validate_feature_groups,
    write_evidence,
)


def _data(rows=180, *, seed=11, predictive=False):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 5 * len(CAUSAL_FEATURE_NAMES)))
    if predictive:
        signal = features[:, 0]
        labels = np.where(signal > 0.35, 1, np.where(signal < -0.35, -1, 0))
    else:
        labels = np.resize(np.array([-1, 0, 1], dtype=np.int8), rows)
    decisions = np.arange(1, rows + 1, dtype=np.int64) * 1_000
    availability = decisions + 1_000
    return AblationData(features, labels.astype(np.int8), decisions, availability)


def _config():
    return AblationConfig(
        min_fit_rows=48,
        test_rows=24,
        step_rows=24,
        minimum_test_rows=12,
        max_iter=300,
        block_size_rows=4,
        bootstrap_samples=40,
        random_state=7,
    )


def test_groups_are_an_exact_nonoverlapping_partition_of_causal_schema():
    validate_feature_groups()
    flattened = [name for names in FEATURE_GROUPS.values() for name in names]
    columns = group_columns(5)

    assert len(flattened) == len(set(flattened)) == len(CAUSAL_FEATURE_NAMES)
    assert set(flattened) == set(CAUSAL_FEATURE_NAMES)
    assert sorted(np.concatenate(list(columns.values())).tolist()) == list(range(5 * len(CAUSAL_FEATURE_NAMES)))


def test_chronological_folds_only_fit_labels_available_by_test_start():
    data = _data()
    folds = list(chronological_folds(data, _config()))

    assert len(folds) >= 2
    for fold in folds:
        test_start = data.decision_times_ms[fold.test_indices].min()
        assert data.decision_times_ms[fold.train_indices].max() < test_start
        assert data.label_available_times_ms[fold.train_indices].max() <= test_start
        assert fold.train_indices.max() < fold.test_indices.min()


def test_unrelated_features_record_valid_negative_or_inconclusive_groups():
    report = evaluate_feature_groups(_data(predictive=False), _config())
    statuses = {group["incrementalStatus"] for group in report["groups"].values()}

    assert report["negativeResultIsValid"] is True
    assert statuses <= {"positive", "harmful", "inconclusive"}
    assert statuses & {"harmful", "inconclusive"}
    assert all(trial["coverage"] == 1.0 for trial in report["trials"].values())
    assert len({trial["evaluationTimestampSha256"] for trial in report["trials"].values()}) == 1


def test_predictive_price_group_is_detected_by_group_only_trial():
    report = evaluate_feature_groups(_data(rows=240, predictive=True), _config())
    price = report["groups"]["price_returns"]["groupOnly"]

    assert price["brier"] < report["baseline"]["brier"]
    assert price["brierLiftVsRollingPrior"] > 0


def test_manifest_is_data_bound_and_evidence_is_immutable(tmp_path):
    data = _data(rows=120)
    config = _config()
    cutoff = int(data.label_available_times_ms[-1])
    manifest = build_manifest(
        data, config, decision_cutoff_ms=cutoff, source="unit-test:synthetic"
    )
    report = evaluate_feature_groups(data, config)
    paths = write_evidence(tmp_path, manifest, report)
    repeated = write_evidence(tmp_path, manifest, report)

    assert paths == repeated
    assert json.loads(paths["manifest"].read_text())["dataProvenance"]["datasetSha256"]
    assert json.loads(paths["report"].read_text())["manifestSha256"] == manifest["manifestSha256"]

    changed = AblationData(
        data.features.copy(), data.labels.copy(), data.decision_times_ms, data.label_available_times_ms
    )
    changed.features[0, 0] += 0.01
    changed_manifest = build_manifest(
        changed, config, decision_cutoff_ms=cutoff, source="unit-test:synthetic"
    )
    assert changed_manifest["manifestSha256"] != manifest["manifestSha256"]
