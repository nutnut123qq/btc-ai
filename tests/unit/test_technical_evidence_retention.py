import hashlib
from datetime import date

import pytest

import run_technical_evidence_pipeline as pipeline
import vietnam_market_calendar as vn
from technical_event_descriptive_evidence import build_postgresql_snapshot_from_rows
from technical_evidence_retention import apply_retention_plan, build_retention_plan
from tests.unit.test_technical_event_descriptive_evidence import _causal_snapshot
from tests.unit.test_technical_evidence_pipeline_mbb import (
    _mbb_daily_rows,
    _mbb_hourly_rows,
    _trading_days,
)


def _run(directory, dedup_bars, minute):
    return pipeline.run_pipeline(
        directory,
        forced_cutoff_ms=10**15,
        bootstrap_samples=5,
        dedup_bars=dedup_bars,
        now=lambda: pipeline.datetime(2026, 9, 22, 2, minute, tzinfo=pipeline.timezone.utc),
        snapshot_loader=lambda timeframe, cutoff: _causal_snapshot(timeframe=timeframe),
    )


def _run_equity(directory, symbol, dedup_bars, minute):
    rows = {
        "1h": _mbb_hourly_rows(_trading_days(date(2026, 8, 3), 20)),
        "1d": _mbb_daily_rows(_trading_days(date(2026, 6, 1), 60)),
    }
    return pipeline.run_pipeline(
        directory,
        symbol=symbol,
        forced_cutoff_ms=10**15,
        bootstrap_samples=5,
        dedup_bars=dedup_bars,
        now=lambda: pipeline.datetime(2026, 9, 22, 2, minute, tzinfo=pipeline.timezone.utc),
        snapshot_loader=lambda timeframe, cutoff: build_postgresql_snapshot_from_rows(
            timeframe, 10**15, rows[timeframe], [], smc_columns=[], symbol=symbol
        ),
    )


def _top_level_files(directory):
    return sorted(
        (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in directory.iterdir()
        if path.is_file()
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


def test_retention_is_scoped_to_one_instrument_directory(tmp_path, monkeypatch):
    # The pipeline isolates equities under <dir>/<SYMBOL>/; retention must plan
    # and delete inside exactly that directory, with that instrument's declared
    # timeframes, and never touch the BTCUSDT top-level run (or vice versa).
    monkeypatch.setenv(vn.NON_TRADING_DATES_ENV, "2000-01-03")
    _run(tmp_path, 6, 0)
    btc_files = _top_level_files(tmp_path)
    for dedup_bars, minute in ((6, 1), (5, 2), (4, 3)):
        _run_equity(tmp_path, "MBB", dedup_bars, minute)
    _run_equity(tmp_path, "BID", 6, 4)
    mbb_dir = tmp_path / "MBB"
    orphan = _orphan(mbb_dir)
    bid_files = _top_level_files(tmp_path / "BID")

    plan = build_retention_plan(tmp_path, keep_successful_runs=2, symbol="MBB")
    assert plan["symbol"] == "MBB"
    assert plan["directory"] == str(mbb_dir.resolve())
    assert plan["blockedReason"] is None
    assert plan["validSuccessfulRunCount"] == 3
    assert orphan.name in {item["fileName"] for item in plan["candidates"]}
    result = apply_retention_plan(
        tmp_path,
        keep_successful_runs=2,
        expected_plan_sha256=plan["planSha256"],
        symbol="MBB",
    )
    assert result["removedCount"] == plan["candidateCount"] > 1
    assert not orphan.exists()
    assert pipeline.read_verified_success(mbb_dir, symbol="MBB", timeframes=("1h", "1d"))["valid"] is True
    assert build_retention_plan(tmp_path, keep_successful_runs=2, symbol="MBB")["candidateCount"] == 0

    # Neighbours are untouched: the BTC top level and the BID directory.
    assert _top_level_files(tmp_path) == btc_files
    assert _top_level_files(tmp_path / "BID") == bid_files
    btc_plan = build_retention_plan(tmp_path, keep_successful_runs=2)
    assert btc_plan["symbol"] == "BTCUSDT"
    assert btc_plan["validSuccessfulRunCount"] == 1
    assert pipeline.read_verified_success(tmp_path)["valid"] is True


def test_retention_rejects_unknown_symbol_and_a_foreign_run_index(tmp_path, monkeypatch):
    monkeypatch.setenv(vn.NON_TRADING_DATES_ENV, "2000-01-03")
    with pytest.raises(ValueError, match="Unknown symbol"):
        build_retention_plan(tmp_path, keep_successful_runs=2, symbol="VNM")
    _run_equity(tmp_path, "MBB", 6, 0)
    # MBB's run copied under BID/ must not be planned as BID's.
    (tmp_path / "BID").mkdir()
    for path in (tmp_path / "MBB").iterdir():
        if path.is_file():
            (tmp_path / "BID" / path.name).write_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        build_retention_plan(tmp_path, keep_successful_runs=2, symbol="BID")
