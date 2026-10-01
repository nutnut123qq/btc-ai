import json
import stat
from datetime import datetime, timezone

import pytest

import run_technical_evidence_pipeline as pipeline
from tests.unit.test_technical_event_descriptive_evidence import _causal_snapshot


FIXED = datetime(2026, 9, 22, 1, 2, 3, tzinfo=timezone.utc)


def test_file_lock_rejects_overlap_and_only_owner_removes(tmp_path):
    path = tmp_path / pipeline.LOCK_FILE
    with pipeline.FileLock(path, lambda: FIXED):
        assert path.exists()
        with pytest.raises(pipeline.PipelineLockedError):
            with pipeline.FileLock(path, lambda: FIXED):
                pass
        assert path.exists()
    assert not path.exists()


def test_explicit_stale_lock_recovery_requires_dead_same_host_owner(tmp_path):
    path = tmp_path / pipeline.LOCK_FILE
    path.write_text(json.dumps({
        "schema": pipeline.LOCK_SCHEMA,
        "token": "owner-token",
        "pid": 1234,
        "host": "test-host",
        "startedAtUtc": "2026-09-21T01:02:03Z",
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="still alive"):
        pipeline.recover_stale_lock(
            path, max_age=pipeline.timedelta(hours=12), now=lambda: FIXED,
            host="test-host", pid_alive=lambda pid: True,
        )
    recovered = pipeline.recover_stale_lock(
        path, max_age=pipeline.timedelta(hours=12), now=lambda: FIXED,
        host="test-host", pid_alive=lambda pid: False,
    )
    assert not path.exists()
    assert recovered.exists()
    with pipeline.FileLock(path, lambda: FIXED):
        assert path.exists()


def test_atomic_status_hash_rejects_tamper(tmp_path):
    path = tmp_path / pipeline.STATUS_FILE
    written = pipeline._write_status_atomic(path, {"schema": pipeline.STATUS_SCHEMA, "running": False})
    assert pipeline.read_verified_status(path) == written
    changed = json.loads(path.read_text())
    changed["running"] = True
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        pipeline.read_verified_status(path)


def test_failed_batch_publishes_no_manifest_and_preserves_failure_status(tmp_path):
    def load(timeframe, cutoff):
        if timeframe == "4h":
            raise RuntimeError("secret=do-not-publish")
        return _causal_snapshot(timeframe=timeframe)

    with pytest.raises(RuntimeError):
        pipeline.run_pipeline(
            tmp_path,
            forced_cutoff_ms=10**15,
            bootstrap_samples=10,
            now=lambda: FIXED,
            snapshot_loader=load,
        )
    assert not list(tmp_path.glob("*.manifest.json"))
    status = pipeline.read_verified_status(tmp_path / pipeline.STATUS_FILE)
    assert status["running"] is False
    assert status["lastFailedAtUtc"] == "2026-09-22T01:02:03Z"
    assert "secret" not in status["lastError"]
    assert not (tmp_path / pipeline.LOCK_FILE).exists()


def test_golden_semantic_self_check_runs_before_any_snapshot_or_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "verify_golden_fixture",
        lambda: (_ for _ in ()).throw(ValueError("golden mismatch")),
    )
    called = False

    def load(timeframe, cutoff):
        nonlocal called
        called = True
        return _causal_snapshot(timeframe)

    with pytest.raises(ValueError, match="golden mismatch"):
        pipeline.run_pipeline(
            tmp_path,
            forced_cutoff_ms=10**15,
            bootstrap_samples=10,
            now=lambda: FIXED,
            snapshot_loader=load,
        )
    assert called is False
    assert not list(tmp_path.glob("*.manifest.json"))
    assert not (tmp_path / pipeline.RUN_POINTER_FILE).exists()


def test_success_publishes_three_verified_manifests_last_and_retains_prior(tmp_path):
    old = tmp_path / ("0" * 64 + ".manifest.json")
    old.write_text("old", encoding="utf-8")

    status = pipeline.run_pipeline(
        tmp_path,
        forced_cutoff_ms=10**15,
        bootstrap_samples=10,
        now=lambda: FIXED,
        snapshot_loader=lambda timeframe, cutoff: _causal_snapshot(timeframe=timeframe),
    )

    assert old.read_text(encoding="utf-8") == "old"
    manifests = [path for path in tmp_path.glob("*.manifest.json") if path != old]
    assert len(manifests) == 3
    assert all(pipeline.verify_bundle(path)["valid"] for path in manifests)
    assert {item["timeframe"] for item in status["timeframes"]} == {"1h", "4h", "1d"}
    assert all(item["semanticVerification"] for item in status["timeframes"])
    pointer = json.loads((tmp_path / pipeline.RUN_POINTER_FILE).read_text(encoding="utf-8"))
    pointer_core = {key: value for key, value in pointer.items() if key != "pointerSha256"}
    assert pointer["pointerSha256"] == pipeline.sha256_bytes(pipeline.canonical_json(pointer_core).encode())
    index_path = tmp_path / pointer["runIndexFileName"]
    assert pointer["runIndexSha256"] == pipeline.sha256_bytes(index_path.read_bytes())
    run_index = json.loads(index_path.read_text(encoding="utf-8"))
    assert [item["timeframe"] for item in run_index["bundles"]] == ["1h", "4h", "1d"]
    assert {item["manifestSha256"] for item in run_index["bundles"]} == {
        path.name.split(".", 1)[0] for path in manifests
    }
    assert all(item["runtimeFileName"] == f"{item['runtimeSha256']}.runtime.json" for item in run_index["bundles"])
    assert all((tmp_path / item["runtimeFileName"]).is_file() for item in run_index["bundles"])
    assert status["runIndexSha256"] == pointer["runIndexSha256"]
    assert pipeline.read_verified_status(tmp_path / pipeline.STATUS_FILE) == status
    verified = pipeline.read_verified_success(tmp_path)
    assert verified["valid"] is True
    assert [item["timeframe"] for item in verified["bundles"]] == ["1h", "4h", "1d"]


def test_run_reader_rejects_missing_or_tampered_referenced_runtime_sidecar(tmp_path):
    pipeline.run_pipeline(
        tmp_path,
        forced_cutoff_ms=10**15,
        bootstrap_samples=5,
        now=lambda: FIXED,
        snapshot_loader=lambda timeframe, cutoff: _causal_snapshot(timeframe=timeframe),
    )
    pointer = json.loads((tmp_path / pipeline.RUN_POINTER_FILE).read_text(encoding="utf-8"))
    index = json.loads((tmp_path / pointer["runIndexFileName"]).read_text(encoding="utf-8"))
    runtime_path = tmp_path / index["bundles"][0]["runtimeFileName"]
    original = runtime_path.read_bytes()
    runtime_path.chmod(stat.S_IWRITE | stat.S_IREAD)
    runtime_path.unlink()
    with pytest.raises(FileNotFoundError):
        pipeline.read_verified_success(tmp_path, verify_semantics=False)
    runtime_path.write_bytes(original)
    runtime_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime sidecar content hash mismatch"):
        pipeline.read_verified_success(tmp_path, verify_semantics=False)


def test_partial_publish_is_orphaned_pointer_stays_atomic_and_retry_recovers(tmp_path):
    pipeline.run_pipeline(
        tmp_path,
        forced_cutoff_ms=10**15,
        bootstrap_samples=5,
        now=lambda: FIXED,
        snapshot_loader=lambda timeframe, cutoff: _causal_snapshot(timeframe=timeframe),
    )
    pointer_before = (tmp_path / pipeline.RUN_POINTER_FILE).read_bytes()
    calls = 0

    def fail_second(paths, output_dir):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        return pipeline._publish_staged_bundle(paths, output_dir)

    with pytest.raises(OSError, match="injected publish failure"):
        pipeline.run_pipeline(
            tmp_path,
            forced_cutoff_ms=10**15,
            bootstrap_samples=5,
            dedup_bars=5,
            now=lambda: FIXED + pipeline.timedelta(minutes=1),
            snapshot_loader=lambda timeframe, cutoff: _causal_snapshot(timeframe=timeframe),
            bundle_publisher=fail_second,
        )
    assert (tmp_path / pipeline.RUN_POINTER_FILE).read_bytes() == pointer_before
    assert pipeline.read_verified_success(tmp_path)["valid"] is True
    failed_status = pipeline.read_verified_status(tmp_path / pipeline.STATUS_FILE)
    assert failed_status["lastFailedAtUtc"] is not None

    recovered = pipeline.run_pipeline(
        tmp_path,
        forced_cutoff_ms=10**15,
        bootstrap_samples=5,
        dedup_bars=5,
        now=lambda: FIXED + pipeline.timedelta(minutes=2),
        snapshot_loader=lambda timeframe, cutoff: _causal_snapshot(timeframe=timeframe),
    )
    assert recovered["lastSucceededAtUtc"] == "2026-09-22T01:04:03Z"
    assert pipeline.read_verified_success(tmp_path)["valid"] is True


def test_pointer_reader_rejects_tamper_and_never_discovers_orphan_manifest(tmp_path):
    pipeline.run_pipeline(
        tmp_path,
        forced_cutoff_ms=10**15,
        bootstrap_samples=5,
        now=lambda: FIXED,
        snapshot_loader=lambda timeframe, cutoff: _causal_snapshot(timeframe=timeframe),
    )
    orphan = tmp_path / ("f" * 64 + ".manifest.json")
    orphan.write_text("orphan", encoding="utf-8")
    assert len(pipeline.read_verified_success(tmp_path)["bundles"]) == 3

    pointer_path = tmp_path / pipeline.RUN_POINTER_FILE
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["runIndexSha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="pointer hash mismatch"):
        pipeline.read_verified_success(tmp_path)
