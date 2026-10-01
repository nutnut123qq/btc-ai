import hashlib

import pytest

import run_technical_evidence_pipeline as pipeline
from technical_evidence_retention import apply_retention_plan, build_retention_plan
from tests.unit.test_technical_event_descriptive_evidence import _causal_snapshot


def _run(directory, dedup_bars, minute):
    return pipeline.run_pipeline(
        directory,
        forced_cutoff_ms=10**15,
        bootstrap_samples=5,
        dedup_bars=dedup_bars,
        now=lambda: pipeline.datetime(2026, 9, 22, 2, minute, tzinfo=pipeline.timezone.utc),
        snapshot_loader=lambda timeframe, cutoff: _causal_snapshot(timeframe=timeframe),
    )


def _orphan(directory, content=b"orphan\n"):
    digest = hashlib.sha256(content).hexdigest()
    path = directory / f"{digest}.report.json"
    path.write_bytes(content)
    return path


def test_retention_dry_run_blocks_until_two_successful_runs_and_deletes_nothing(tmp_path):
    _run(tmp_path, 6, 0)
    orphan = _orphan(tmp_path)
    plan = build_retention_plan(tmp_path, keep_successful_runs=2)
    assert plan["blockedReason"].startswith("only 1 valid successful run")
    assert plan["candidateCount"] == 0
    assert orphan.exists()
    assert pipeline.read_verified_success(tmp_path)["valid"] is True


def test_retention_keeps_latest_two_and_shared_files_requires_exact_plan_hash(tmp_path):
    _run(tmp_path, 6, 0)
    first_pointer = pipeline.read_verified_success(tmp_path)
    _run(tmp_path, 5, 1)
    _run(tmp_path, 4, 2)
    orphan = _orphan(tmp_path)

    plan = build_retention_plan(tmp_path, keep_successful_runs=2)
    assert plan["blockedReason"] is None
    assert plan["validSuccessfulRunCount"] == 3
    assert plan["candidateCount"] > 0
    assert orphan.name in {item["fileName"] for item in plan["candidates"]}
    assert f"{first_pointer['runIndexSha256']}.run-index.json" not in plan["keptRunIndexFileNames"]
    with pytest.raises(ValueError, match="plan hash changed"):
        apply_retention_plan(
            tmp_path,
            keep_successful_runs=2,
            expected_plan_sha256="0" * 64,
        )
    assert orphan.exists()

    result = apply_retention_plan(
        tmp_path,
        keep_successful_runs=2,
        expected_plan_sha256=plan["planSha256"],
    )
    assert result["removedCount"] == plan["candidateCount"]
    assert not orphan.exists()
    assert pipeline.read_verified_success(tmp_path)["valid"] is True
    after = build_retention_plan(tmp_path, keep_successful_runs=2)
    assert after["candidateCount"] == 0


def test_retention_fails_closed_when_candidate_changes_after_review(tmp_path):
    _run(tmp_path, 6, 0)
    _run(tmp_path, 5, 1)
    orphan = _orphan(tmp_path)
    plan = build_retention_plan(tmp_path, keep_successful_runs=2)
    orphan.write_bytes(b"changed after review\n")
    with pytest.raises(ValueError, match="plan hash changed"):
        apply_retention_plan(
            tmp_path,
            keep_successful_runs=2,
            expected_plan_sha256=plan["planSha256"],
        )
    assert pipeline.read_verified_success(tmp_path)["valid"] is True
