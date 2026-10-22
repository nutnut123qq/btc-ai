"""Stage B protocol guards — synthetic bundle + mocked DB, never production."""
import dataclasses
import json
import shutil

import numpy as np
import pytest

import ml_evidence_v3 as v3
import ml_evidence_walkforward as v2
import stage_b_reserve_eval as sb
from test_ml_evidence_v3 import _config

I = v2.INTERVAL_MS
N_FROZEN, N_LATE, CAL_ROWS = 144, 5, 24
EXCLUDED_IDX = (20, N_FROZEN - 10)  # one fit row, one calibration row whose stored label breaks the contract
N_USED = N_FROZEN - len(EXCLUDED_IDX)


def _timeline(rows, seed=11):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 5 * len(v2.CAUSAL_FEATURE_NAMES)))
    signal = features[:, 0] + 0.4 * features[:, -1]
    labels = np.where(signal > 0.45, 1, np.where(signal < -0.45, -1, 0)).astype(np.int8)
    decision = 1_600_000_000_000 - (1_600_000_000_000 % I) + np.arange(1, rows + 1, dtype=np.int64) * I
    return v2.BenchmarkData(features, labels, decision, decision + I, features[:, -1])


def _closes(data):
    """Price path whose close-to-close moves reproduce every label (+-1% vs the 0.6% band)."""
    signal_open = data.decision_times_ms - I
    prices = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + data.labels.astype(float) / 100.0]))
    return {int(t): float(p) for t, p in zip(np.append(signal_open, signal_open[-1] + I), prices)}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    clean = _timeline(N_FROZEN + N_LATE + 1 + 400)
    closes = _closes(clean)
    labels = clean.labels.copy()
    labels[list(EXCLUDED_IDX)] = np.where(labels[list(EXCLUDED_IDX)] == 1, -1, 1)  # stored != Klines-derived
    full = v2.BenchmarkData(clean.features, labels, clean.decision_times_ms,
                            clean.label_available_times_ms, clean.last_returns)
    frozen = sb._subset(full, np.arange(len(full.labels)) < N_FROZEN)
    excluded_ms = tuple(int(full.decision_times_ms[i]) for i in EXCLUDED_IDX)
    used = sb._subset(frozen, ~np.isin(frozen.decision_times_ms, excluded_ms))
    # cutoff admits the 5 late rows' labels; the next row is the decision<=cutoff<label boundary row
    cutoff = int(full.label_available_times_ms[N_FROZEN + N_LATE - 1])
    config = dataclasses.replace(_config(), random_state=sb.SEED)
    manifest = v3.build_v3_manifest(frozen, config, decision_cutoff_ms=cutoff, source="test-fixture")
    result = v3.evaluate_walk_forward_v3(frozen, config)
    paths = v3.write_evidence_bundle(tmp_path_factory.mktemp("bundle"), frozen, manifest, result)
    bundle = json.loads(paths["bundleManifest"].read_text(encoding="utf-8"))
    return {"full": full, "frozen": frozen, "used": used, "excluded": excluded_ms, "cutoff": cutoff,
            "bundle": paths["bundleManifest"], "bundleSha": bundle["bundleManifestSha256"], "closes": closes}


@pytest.fixture
def env(world, monkeypatch, tmp_path):
    """Pin the protocol to the fixture bundle and route every DB read to synthetic data."""
    monkeypatch.setattr(sb, "FROZEN_CUTOFF_MS", world["cutoff"])
    monkeypatch.setattr(sb, "FROZEN_BUNDLE_MANIFEST_SHA256", world["bundleSha"])
    monkeypatch.setattr(sb, "FROZEN_DATASET_SHA256", v2.dataset_sha256(world["frozen"]))
    monkeypatch.setattr(sb, "FROZEN_ROWS", N_FROZEN)
    monkeypatch.setattr(sb, "CALIBRATION_ROWS", CAL_ROWS)
    monkeypatch.setattr(sb, "LABEL_EXCLUDED_DECISION_MS", world["excluded"])
    monkeypatch.setattr(sb, "STAGE_B_INPUT_ROWS", N_USED)
    monkeypatch.setattr(sb, "STAGE_B_INPUT_SHA256", v2.dataset_sha256(world["used"]))
    state = {"live": world["full"], "reserve": 150, "closes": dict(world["closes"]),
             "loads": [], "fits": [], "estimators": []}

    def load(*, decision_cutoff_ms, window_size):
        state["loads"].append(decision_cutoff_ms)
        live = state["live"]
        rows = sb._subset(live, live.label_available_times_ms <= decision_cutoff_ms)
        keep = (rows.decision_times_ms <= sb.FROZEN_CUTOFF_MS) | (
            np.cumsum(rows.decision_times_ms > sb.FROZEN_CUTOFF_MS) <= state["reserve"])
        return sb._subset(rows, keep)

    real_fit, real_build = v3._fit_calibrated_hgb, v2._build_estimator

    def fit(data, fit_idx, cal_idx, test_idx, columns, seed):
        state["fits"].append({"data": data, "fit": fit_idx, "cal": cal_idx, "test": test_idx,
                              "columns": columns, "seed": seed})
        return real_fit(data, fit_idx, cal_idx, test_idx, columns, seed)

    def build(name, seed):
        state["estimators"].append(name)
        return real_build(name, seed)

    monkeypatch.setattr(v2, "load_btc_4h_benchmark", load)
    monkeypatch.setattr(v3, "_fit_calibrated_hgb", fit)
    monkeypatch.setattr(v2, "_build_estimator", build)
    monkeypatch.setattr(sb, "fetch_closes", lambda _opens: state["closes"])
    monkeypatch.setattr(sb, "count_reserve_rows", lambda _now: {"rows": state["reserve"]})
    state["out"] = tmp_path / "reports" / "stage_b_report_test.json"
    state["score"] = ["--score", "--bundle", str(world["bundle"]), "--out", str(state["out"])]
    state["count"] = ["--count-only", "--bundle", str(world["bundle"])]
    return state


def test_149_reserve_rows_stop_before_any_fit(env):
    env["reserve"] = 149
    assert sb.main(env["score"]) == sb.EXIT_NOT_READY
    assert env["fits"] == [] and not env["out"].exists()


@pytest.mark.parametrize("rows", [149, 150, 400])
def test_count_only_never_fits_scores_or_loads_reserve(env, monkeypatch, rows):
    monkeypatch.setattr(sb, "score", lambda *_: pytest.fail("count-only reached scoring"))
    env["reserve"] = rows
    expected = sb.EXIT_READY_BY_COUNT if rows >= sb.MIN_RESERVE_ROWS else sb.EXIT_NOT_READY
    assert sb.main(env["count"]) == expected
    assert env["fits"] == [] and env["loads"] == [sb.FROZEN_CUTOFF_MS]
    assert not env["out"].parent.exists()


def test_explicit_score_uses_frozen_split_hgb_and_fixed_threshold(env, world):
    code = sb.main(env["score"])
    report = json.loads(env["out"].read_text(encoding="utf-8"))
    assert code == (sb.EXIT_PASSED if report["status"] == "STAGE_B_PASSED" else sb.EXIT_FAILED)
    assert report["status"] in {"STAGE_B_PASSED", "STAGE_B_FAILED"} and report["scoredOnce"]
    (call,) = env["fits"]
    np.testing.assert_array_equal(call["fit"], np.arange(N_USED - CAL_ROWS))
    np.testing.assert_array_equal(call["cal"], np.arange(N_USED - CAL_ROWS, N_USED))
    np.testing.assert_array_equal(call["test"], np.arange(N_USED, N_USED + 150))
    assert len(call["columns"]) == 170 and call["seed"] == sb.SEED
    assert set(env["estimators"]) == {"hist_gradient_boosting"}
    assert report["minReserveRows"] == 150 and report["reserve"]["rows"] == 150
    assert report["labelLineage"]["frozen"]["complete"] and report["labelLineage"]["reserve"]["complete"]
    assert report["frozenInput"]["labelExcludedDecisionTimesMs"] == list(world["excluded"])


def test_excluded_rows_leave_fit_calibration_and_baseline_history(env, world):
    sb.main(env["score"])
    (call,) = env["fits"]
    data = call["data"]  # the same matrix feeds the HGB fit, the calibrator and the rolling baselines
    assert not np.isin(world["excluded"], data.decision_times_ms).any()
    np.testing.assert_array_equal(data.features[:N_USED], world["used"].features)
    np.testing.assert_array_equal(data.labels[:N_USED], world["used"].labels)


def test_without_the_exclusions_the_frozen_labels_fail_lineage(env, world, monkeypatch):
    monkeypatch.setattr(sb, "LABEL_EXCLUDED_DECISION_MS", ())
    monkeypatch.setattr(sb, "STAGE_B_INPUT_ROWS", N_FROZEN)
    monkeypatch.setattr(sb, "STAGE_B_INPUT_SHA256", v2.dataset_sha256(world["frozen"]))
    assert sb.main(env["score"]) == sb.EXIT_EVIDENCE
    assert env["fits"] == []


def test_late_rows_change_neither_fit_calibration_nor_reserve(env, world, tmp_path):
    full = world["full"]
    without_late = sb._subset(full, ~np.isin(np.arange(len(full.labels)), np.arange(N_FROZEN, N_FROZEN + N_LATE)))
    runs = []
    for i, live in enumerate((full, without_late)):
        env["live"], env["fits"] = live, []
        out = tmp_path / f"run{i}" / "stage_b_report.json"
        sb.main(["--score", "--bundle", str(world["bundle"]), "--out", str(out)])
        runs.append((env["fits"][0], json.loads(out.read_text(encoding="utf-8"))))
    (a, ra), (b, rb) = runs
    frozen_part = slice(0, N_USED)
    np.testing.assert_array_equal(a["data"].features[frozen_part], world["used"].features)
    np.testing.assert_array_equal(a["data"].features[frozen_part], b["data"].features[frozen_part])
    np.testing.assert_array_equal(a["data"].decision_times_ms, b["data"].decision_times_ms)
    late = full.decision_times_ms[N_FROZEN:N_FROZEN + N_LATE]
    assert not np.isin(late, a["data"].decision_times_ms).any()
    assert ra["reserve"] == rb["reserve"]
    assert ra["drift"]["lateRowsExcludedFromFitCalibration"] == N_LATE
    assert rb["drift"]["lateRowsExcludedFromFitCalibration"] == 0


def _tamper_npz(env, world, tmp_path):
    copy = tmp_path / "bundle"
    shutil.copytree(world["bundle"].parent, copy)
    npz = next(copy.glob("*.dataset.npz"))
    content = bytearray(npz.read_bytes())
    content[len(content) // 2] ^= 0xFF
    npz.write_bytes(bytes(content))
    env["score"][2] = str(copy / world["bundle"].name)


def _drop_bar(env, world, tmp_path):
    env["closes"].pop(min(env["closes"]))


def _flip_label_source(env, world, tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "LABEL_SOURCE", "PriceTargets.TargetDirectionTb4h")


def _move_reserve_close(env, world, tmp_path):
    row = N_FROZEN + N_LATE + 150  # last reserve row (after the decision<=cutoff<label boundary row)
    decision, label = int(world["full"].decision_times_ms[row]), int(world["full"].labels[row])
    c0 = env["closes"][decision - I]
    env["closes"][decision] = c0 * (1.02 if label == 0 else 1.0)  # realized move no longer matches label


def _drift_live(env, world, tmp_path):
    live = env["live"]
    features = live.features.copy()
    features[3, 0] += 1.0
    env["live"] = v2.BenchmarkData(features, live.labels, live.decision_times_ms,
                                   live.label_available_times_ms, live.last_returns)


def _wrong_schema(env, world, tmp_path):
    live = env["live"]
    env["live"] = v2.BenchmarkData(live.features[:, :-1], live.labels, live.decision_times_ms,
                                   live.label_available_times_ms, live.last_returns)


@pytest.mark.parametrize("breakage", [_tamper_npz, _drop_bar, _flip_label_source, _move_reserve_close,
                                      _drift_live, _wrong_schema, "pin", "runtime", "input-pin",
                                      "unknown-exclusion"])
def test_broken_evidence_fails_closed_before_scoring(env, world, tmp_path, monkeypatch, breakage):
    if breakage == "pin":
        monkeypatch.setattr(sb, "FROZEN_BUNDLE_MANIFEST_SHA256", "0" * 64)
    elif breakage == "input-pin":
        monkeypatch.setattr(sb, "STAGE_B_INPUT_SHA256", "0" * 64)
    elif breakage == "unknown-exclusion":  # a timestamp outside the snapshot must not silently shrink the list
        monkeypatch.setattr(sb, "LABEL_EXCLUDED_DECISION_MS", (world["excluded"][0], 1))
    elif breakage == "runtime":
        monkeypatch.setattr(sb, "_runtime", lambda: {"python": "0", "numpy": "0", "scikitLearn": "0", "psycopg2": "0"})
    elif breakage is _flip_label_source:
        breakage(env, world, tmp_path, monkeypatch)
    else:
        breakage(env, world, tmp_path)
    assert sb.main(env["score"]) == sb.EXIT_EVIDENCE
    assert env["fits"] == [] and not env["out"].exists()


def test_scoring_is_one_shot(env):
    first = sb.main(env["score"])
    assert first in (sb.EXIT_PASSED, sb.EXIT_FAILED)
    env["fits"] = []
    again = env["out"].with_name("stage_b_report_retry.json")
    assert sb.main(["--score", "--bundle", env["score"][2], "--out", str(again)]) == sb.EXIT_EVIDENCE
    assert env["fits"] == [] and not again.exists()


@pytest.mark.parametrize("argv", [
    [], ["--bundle", "x"], ["--score", "--bundle", "x"],
    ["--count-only", "--bundle", "x", "--min-reserve", "10"], ["--count-only", "--bundle", "x", "--strict"],
    ["--score", "--count-only", "--bundle", "x"], ["--count-only", "--bundle", "x", "--candidate", "xgb"],
])
def test_cli_offers_no_threshold_candidate_or_implicit_scoring(argv):
    with pytest.raises(SystemExit) as exc:
        sb.main(argv)
    assert exc.value.code == 2


def test_label_contract_band_edges_and_class_mapping():
    got = sb.ctc_direction(np.array([0.6, -0.6, 0.6000001, -0.6000001, 0.0]))
    np.testing.assert_array_equal(got, [0, 0, 1, -1, 0])
    assert list(v2.CLASSES) == [-1, 0, 1] and sb.LABEL_REMAP == {-1: 0, 0: 1, 1: 2}


def test_lineage_uses_signal_close_and_next_bar_not_a_later_bar(world):
    data, closes = world["used"], world["closes"]
    assert sb.label_lineage(data.decision_times_ms, data.labels, closes)["complete"]
    shifted = {t - I: c for t, c in closes.items()}  # pairs close[t+4h] with close[t+8h]
    assert not sb.label_lineage(data.decision_times_ms, data.labels, shifted)["complete"]
