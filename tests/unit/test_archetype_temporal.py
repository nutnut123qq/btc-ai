import json

import numpy as np
import pytest

from archetype_temporal import (
    assign_frozen,
    evaluate_temporal_archetypes,
    fit_temporal_archetypes,
    load_model,
    save_model_immutable,
)


def _separated_history(train_rows=80, future_rows=40):
    rng = np.random.default_rng(7)
    train_left = rng.normal(loc=-3.0, scale=0.15, size=(train_rows // 2, 3))
    train_right = rng.normal(loc=3.0, scale=0.15, size=(train_rows // 2, 3))
    train_x = np.vstack([train_left, train_right])
    train_y = np.array([-1] * len(train_left) + [1] * len(train_right), dtype=np.int8)
    future_left = rng.normal(loc=-3.0, scale=0.10, size=(future_rows // 2, 3))
    future_right = rng.normal(loc=3.0, scale=0.10, size=(future_rows // 2, 3))
    future_x = np.vstack([future_left, future_right])
    future_y = np.array([-1] * len(future_left) + [1] * len(future_right), dtype=np.int8)
    x = np.vstack([train_x, future_x])
    y = np.concatenate([train_y, future_y])
    times = np.arange(1, len(x) + 1, dtype=np.int64) * 1_000
    return x, y, times, train_rows * 1_000


def _fit(x, y, times, cutoff):
    return fit_temporal_archetypes(
        x,
        y,
        times,
        train_through_ms=cutoff,
        label_available_times_ms=times,
        n_clusters=2,
        min_cluster_members=2,
        distance_quantile=1.0,
    )


def test_appending_future_rows_cannot_change_version_or_historical_assignments():
    x, y, times, cutoff = _separated_history()
    model_before = _fit(x[:80], y[:80], times[:80], cutoff)
    model_after = _fit(x, y, times, cutoff)

    assert model_before.to_dict() == model_after.to_dict()
    before = assign_frozen(model_before, x[:80])
    after = assign_frozen(model_after, x[:80])
    np.testing.assert_array_equal(before.cluster_indices, after.cluster_indices)
    assert before.archetype_ids == after.archetype_ids


def test_future_labels_never_affect_scaler_centroids_or_predictions():
    x, y, times, cutoff = _separated_history()
    inverted_future = y.copy()
    inverted_future[80:] *= -1

    original = _fit(x, y, times, cutoff)
    inverted = _fit(x, inverted_future, times, cutoff)

    assert original.training_data_hash == inverted.training_data_hash
    assert original.version_id == inverted.version_id
    np.testing.assert_allclose(original.scaler_mean, inverted.scaler_mean)
    np.testing.assert_allclose(original.centroids_scaled, inverted.centroids_scaled)
    np.testing.assert_allclose(
        assign_frozen(original, x[80:]).class_probabilities,
        assign_frozen(inverted, x[80:]).class_probabilities,
    )


def test_version_file_is_immutable_and_round_trips(tmp_path):
    x, y, times, cutoff = _separated_history()
    model = _fit(x, y, times, cutoff)
    path = tmp_path / "frozen.json"

    save_model_immutable(model, path)
    save_model_immutable(model, path)
    assert load_model(path) == model

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["train_through_ms"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_model_immutable(model, path)


def test_oos_evaluator_records_a_valid_negative_result_against_baseline():
    x, y, times, cutoff = _separated_history(train_rows=80, future_rows=80)
    # Reverse the train relationship in frozen future data. The cluster-conditioned
    # forecast must lose to the balanced unconditional baseline, not be hidden.
    y[80:] *= -1

    model, report = evaluate_temporal_archetypes(
        x,
        y,
        times,
        train_through_ms=cutoff,
        label_available_times_ms=times,
        n_clusters=2,
        min_cluster_members=2,
        distance_quantile=1.0,
        min_evaluation_samples=30,
        block_size=4,
        bootstrap_samples=300,
    )

    assert report["versionId"] == model.version_id
    assert report["acceptedRows"] == 80
    assert report["coverage"] == 1.0
    assert report["brierLift"] < 0
    assert report["brierLift95Ci"][1] < 0
    assert report["conclusion"] == "negative"
    assert report["negativeResultIsValid"] is True


def test_distant_future_windows_abstain_instead_of_forcing_an_archetype():
    x, y, times, cutoff = _separated_history()
    model = fit_temporal_archetypes(
        x,
        y,
        times,
        train_through_ms=cutoff,
        label_available_times_ms=times,
        n_clusters=2,
        min_cluster_members=2,
        distance_quantile=0.95,
    )
    assignment = assign_frozen(model, np.array([[100.0, 100.0, 100.0]]))

    assert assignment.accepted.tolist() == [False]
    assert assignment.archetype_ids == (None,)


def test_label_realized_after_cutoff_is_purged_from_training():
    x, y, times, cutoff = _separated_history(train_rows=80, future_rows=40)
    label_available = times.copy()
    # Last five feature windows are visible at the cutoff, but their outcomes are not.
    label_available[75:80] = cutoff + 1_000

    model = fit_temporal_archetypes(
        x,
        y,
        times,
        train_through_ms=cutoff,
        label_available_times_ms=label_available,
        n_clusters=2,
        min_cluster_members=2,
        distance_quantile=1.0,
    )

    assert model.training_rows == 75
    assert model.training_label_policy == "label_available_time_ms <= train_through_ms"


def test_baseline_also_excludes_labels_that_mature_after_cutoff():
    x, y, times, cutoff = _separated_history(train_rows=80, future_rows=40)
    label_available = times.copy()
    # These labels are visible in the input array but were not knowable at the
    # decision cutoff. Make them a third class so leakage changes the baseline.
    y[70:80] = 0
    label_available[70:80] = cutoff + 1_000

    _, report = evaluate_temporal_archetypes(
        x,
        y,
        times,
        train_through_ms=cutoff,
        label_available_times_ms=label_available,
        n_clusters=2,
        min_cluster_members=2,
        distance_quantile=1.0,
        min_evaluation_samples=1,
        bootstrap_samples=50,
    )

    # Eligible train labels contain 40 down and 30 up samples. With Laplace
    # alpha=1, the probabilities are [41, 1, 31] / 73.
    expected = np.array([41 / 73, 1 / 73, 31 / 73])
    future_labels = y[80:]
    target = np.zeros((len(future_labels), 3))
    for class_index, class_value in enumerate((-1, 0, 1)):
        target[:, class_index] = future_labels == class_value
    expected_brier = float(np.mean(np.sum((expected - target) ** 2, axis=1)))
    assert report["baselineBrier"] == pytest.approx(expected_brier)
