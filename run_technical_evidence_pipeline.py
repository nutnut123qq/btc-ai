#!/usr/bin/env python3
"""Finite, lock-protected publisher for BTC technical descriptive evidence.

This is an operational wrapper, not a daemon.  A scheduler may invoke it; one
invocation attempts exactly one 1h/4h/1d batch and exits.  Bundle manifests are
published only after the staged bundle passes semantic verification.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from technical_event_descriptive_evidence import (
    EvaluationConfig,
    TIMEFRAME_MS,
    canonical_json,
    evaluate,
    load_postgresql_snapshot,
    sha256_bytes,
    sha256_file,
    verify_bundle,
    write_bundle,
)
from technical_event_modules import verify_golden_fixture


STATUS_SCHEMA = "btc-technical-evidence-pipeline-status/v1"
TIMEFRAMES = ("1h", "4h", "1d")
STATUS_FILE = "pipeline-status.json"
LOCK_FILE = ".pipeline.lock"
LOCK_SCHEMA = "btc-technical-evidence-pipeline-lock/v1"
RUN_INDEX_SCHEMA = "btc-technical-evidence-run-index/v1"
RUN_POINTER_SCHEMA = "btc-technical-evidence-run-pointer/v1"
RUN_POINTER_FILE = "latest-success.json"


class PipelineLockedError(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _status_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("statusSha256", None)
    result["statusSha256"] = sha256_bytes(canonical_json(result).encode())
    return result


def _write_status_atomic(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _status_with_hash(payload)
    content = (canonical_json(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    content = (canonical_json(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_run_index(output_dir: Path, completed: datetime, published: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    ordered = sorted(published, key=lambda item: TIMEFRAMES.index(str(item["timeframe"])))
    if [item["timeframe"] for item in ordered] != list(TIMEFRAMES):
        raise ValueError("run index requires exact 1h, 4h and 1d bundle coverage")
    definitions = {item.get("definitionsSha256") for item in ordered}
    if len(definitions) != 1 or None in definitions:
        raise ValueError("run index contract definitions hash is inconsistent")
    bundles = [
        {
            "timeframe": item["timeframe"],
            "cutoffMs": item["cutoffMs"],
            "manifestFileName": item["manifestFileName"],
            "manifestSha256": item["manifestSha256"],
            "stored": item["stored"],
            "eligible": item["eligible"],
            "excluded": item["excluded"],
            "realizedAtMaxHorizon": item["realizedAtMaxHorizon"],
            "semanticVerification": item["semanticVerification"],
            "runtimeFileName": item.get("runtimeFileName"),
            "runtimeSha256": item.get("runtimeSha256"),
        }
        for item in ordered
    ]
    index = {
        "schema": RUN_INDEX_SCHEMA,
        "claimType": "descriptive_technical_event_history",
        "symbol": "BTCUSDT",
        "status": "succeeded",
        "completedAtUtc": _iso(completed),
        "contractDefinitionsSha256": next(iter(definitions)),
        "bundles": bundles,
    }
    index_bytes = (canonical_json(index) + "\n").encode()
    index_sha = sha256_bytes(index_bytes)
    index_path = output_dir / f"{index_sha}.run-index.json"
    if index_path.exists():
        if index_path.read_bytes() != index_bytes:
            raise FileExistsError("conflicting content-addressed run index")
    else:
        descriptor = os.open(index_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(index_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    pointer_core = {
        "schema": RUN_POINTER_SCHEMA,
        "runIndexFileName": index_path.name,
        "runIndexSha256": index_sha,
        "updatedAtUtc": _iso(completed),
    }
    pointer = {**pointer_core, "pointerSha256": sha256_bytes(canonical_json(pointer_core).encode())}
    _write_json_atomic(output_dir / RUN_POINTER_FILE, pointer)
    return index_path, pointer


def _safe_child(directory: Path, file_name: Any) -> Path:
    if not isinstance(file_name, str) or not file_name or Path(file_name).name != file_name:
        raise ValueError("evidence pointer contains an unsafe file name")
    path = directory / file_name
    if path.parent.resolve() != directory.resolve():
        raise ValueError("evidence pointer path traversal rejected")
    return path


def read_verified_success(output_dir: Path, *, verify_semantics: bool = True) -> dict[str, Any]:
    """Fail-closed reader for the exact atomic trio referenced by latest-success."""
    directory = output_dir.resolve()
    pointer_path = directory / RUN_POINTER_FILE
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema") != RUN_POINTER_SCHEMA:
        raise ValueError("unsupported evidence run pointer schema")
    pointer_sha = pointer.get("pointerSha256")
    pointer_core = {key: value for key, value in pointer.items() if key != "pointerSha256"}
    if pointer_sha != sha256_bytes(canonical_json(pointer_core).encode()):
        raise ValueError("evidence run pointer hash mismatch")

    index_path = _safe_child(directory, pointer.get("runIndexFileName"))
    index_bytes = index_path.read_bytes()
    index_sha = sha256_bytes(index_bytes)
    if pointer.get("runIndexSha256") != index_sha or index_path.name != f"{index_sha}.run-index.json":
        raise ValueError("evidence run index hash mismatch")
    index = json.loads(index_bytes)
    if (
        index.get("schema") != RUN_INDEX_SCHEMA
        or index.get("claimType") != "descriptive_technical_event_history"
        or index.get("symbol") != "BTCUSDT"
        or index.get("status") != "succeeded"
    ):
        raise ValueError("unsupported or unsuccessful evidence run index")
    bundles = index.get("bundles")
    if not isinstance(bundles, list) or [item.get("timeframe") for item in bundles] != list(TIMEFRAMES):
        raise ValueError("evidence run index does not contain the exact 1h/4h/1d trio")

    verified: list[dict[str, Any]] = []
    for item in bundles:
        manifest_sha = item.get("manifestSha256")
        if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
            raise ValueError("evidence run manifest hash is invalid")
        manifest_path = _safe_child(directory, item.get("manifestFileName"))
        if manifest_path.name != f"{manifest_sha}.manifest.json":
            raise ValueError("evidence run manifest name/hash mismatch")
        if sha256_bytes(manifest_path.read_bytes()) != manifest_sha:
            raise ValueError("evidence run manifest content hash mismatch")
        if item.get("semanticVerification") is not True:
            raise ValueError("evidence run contains a bundle not marked semantically verified")
        runtime_name = item.get("runtimeFileName")
        runtime_sha = item.get("runtimeSha256")
        if (runtime_name is None) != (runtime_sha is None):
            raise ValueError("evidence runtime sidecar declaration is incomplete")
        if runtime_name is not None:
            if not isinstance(runtime_sha, str) or runtime_name != f"{runtime_sha}.runtime.json":
                raise ValueError("evidence runtime sidecar name/hash mismatch")
            runtime_path = _safe_child(directory, runtime_name)
            runtime_bytes = runtime_path.read_bytes()
            if sha256_bytes(runtime_bytes) != runtime_sha:
                raise ValueError("evidence runtime sidecar content hash mismatch")
            runtime = json.loads(runtime_bytes)
            if (
                runtime.get("schema") != "btc-technical-evidence-runtime/v1"
                or runtime.get("semanticManifestFileName") != manifest_path.name
                or runtime.get("semanticManifestSha256") != manifest_sha
                or runtime.get("semanticHashIncludesRuntimeMetadata") is not False
            ):
                raise ValueError("evidence runtime sidecar semantic binding mismatch")
        result = verify_bundle(manifest_path) if verify_semantics else {"valid": True}
        if result.get("valid") is not True:
            raise ValueError("evidence run bundle semantic verification failed")
        verified.append({"timeframe": item["timeframe"], **result})
    return {
        "valid": True,
        "pointerSha256": pointer_sha,
        "runIndexSha256": index_sha,
        "contractDefinitionsSha256": index.get("contractDefinitionsSha256"),
        "bundles": verified,
    }


def read_verified_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != STATUS_SCHEMA:
        raise ValueError("unsupported pipeline status schema")
    declared = value.get("statusSha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError("pipeline status hash missing")
    expected = _status_with_hash(value)["statusSha256"]
    if declared != expected:
        raise ValueError("pipeline status hash mismatch")
    return value


class FileLock:
    def __init__(self, path: Path, now: Callable[[], datetime] = _utc_now):
        self.path = path
        self.now = now
        self.token = uuid.uuid4().hex
        self._owned = False

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise PipelineLockedError(f"pipeline lock already exists: {self.path.name}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json({
                "schema": LOCK_SCHEMA,
                "token": self.token,
                "pid": os.getpid(),
                "host": platform.node(),
                "startedAtUtc": _iso(self.now()),
            }) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._owned = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._owned:
            return
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if current.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        finally:
            self._owned = False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def recover_stale_lock(
    lock_path: Path,
    *,
    max_age: timedelta,
    now: Callable[[], datetime] = _utc_now,
    host: str | None = None,
    pid_alive: Callable[[int], bool] = _pid_alive,
) -> Path:
    """Explicit conservative recovery; never called by the scheduled run."""
    if not lock_path.exists():
        raise FileNotFoundError("pipeline lock does not exist")
    value = json.loads(lock_path.read_text(encoding="utf-8"))
    if value.get("schema") != LOCK_SCHEMA:
        raise ValueError("pipeline lock schema is invalid; manual inspection required")
    token = value.get("token")
    pid = value.get("pid")
    lock_host = value.get("host")
    started_raw = value.get("startedAtUtc")
    if not isinstance(token, str) or not token or not isinstance(pid, int) or not isinstance(lock_host, str) or not isinstance(started_raw, str):
        raise ValueError("pipeline lock fields are invalid; manual inspection required")
    current_host = host if host is not None else platform.node()
    if lock_host != current_host:
        raise RuntimeError("lock belongs to a different host; manual inspection required")
    started = datetime.fromisoformat(started_raw.replace("Z", "+00:00"))
    if now() - started <= max_age:
        raise RuntimeError("lock is not old enough for recovery")
    if pid_alive(pid):
        raise RuntimeError("lock owner PID is still alive; recovery refused")
    recovered = lock_path.with_name(f"{lock_path.name}.recovered-{token}.json")
    os.replace(lock_path, recovered)
    return recovered


def latest_cutoffs() -> dict[str, int]:
    """Return the latest finalized persisted candle close for each supported timeframe."""
    from db_config import get_db_connection

    connection = get_db_connection()
    try:
        # Keep all three MAX queries on one database snapshot.  The timeframes
        # may legitimately have different latest closes, but the chosen trio
        # must describe one coherent persisted state.
        connection.set_session(
            readonly=True,
            autocommit=False,
            isolation_level="REPEATABLE READ",
        )
        with connection.cursor() as cursor:
            result: dict[str, int] = {}
            for timeframe in TIMEFRAMES:
                cursor.execute(
                    'SELECT MAX("CloseTimeMs") FROM "Klines" WHERE "Symbol"=%s AND "Timeframe"=%s',
                    ("BTCUSDT", timeframe),
                )
                value = cursor.fetchone()[0]
                if not isinstance(value, int) or value <= 0:
                    raise RuntimeError(f"no finalized BTCUSDT {timeframe} candle available")
                result[timeframe] = value
        connection.rollback()
        return result
    finally:
        connection.close()


def _publish_staged_bundle(staged: Mapping[str, Path], output_dir: Path) -> Path:
    """Publish immutable artifacts first and the verified manifest last."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for role in ("snapshot", "ledger", "report", "runtime"):
        source = staged[role]
        destination = output_dir / source.name
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size or sha256_file(destination) != sha256_file(source):
                raise FileExistsError(f"conflicting content-addressed artifact: {destination.name}")
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            descriptor = os.open(destination, flags, 0o444)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    with source.open("rb") as source_stream:
                        shutil.copyfileobj(source_stream, stream, length=1024 * 1024)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                destination.unlink(missing_ok=True)
                raise
    manifest_source = staged["manifest"]
    manifest_destination = output_dir / manifest_source.name
    if manifest_destination.exists():
        if manifest_destination.stat().st_size != manifest_source.stat().st_size or sha256_file(manifest_destination) != sha256_file(manifest_source):
            raise FileExistsError(f"conflicting content-addressed manifest: {manifest_destination.name}")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(manifest_destination, flags, 0o444)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                with manifest_source.open("rb") as source_stream:
                    shutil.copyfileobj(source_stream, stream, length=1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            manifest_destination.unlink(missing_ok=True)
            raise
    return manifest_destination


def run_pipeline(
    output_dir: Path,
    *,
    forced_cutoff_ms: int | None = None,
    bootstrap_samples: int = 2_000,
    block_size_events: int = 8,
    dedup_bars: int = 6,
    now: Callable[[], datetime] = _utc_now,
    cutoff_loader: Callable[[], Mapping[str, int]] = latest_cutoffs,
    snapshot_loader: Callable[[str, int], Mapping[str, Any]] = load_postgresql_snapshot,
    bundle_publisher: Callable[[Mapping[str, Path], Path], Path] = _publish_staged_bundle,
    run_index_publisher: Callable[[Path, datetime, list[dict[str, Any]]], tuple[Path, dict[str, Any]]] = _publish_run_index,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / STATUS_FILE
    lock_path = output_dir / LOCK_FILE
    with FileLock(lock_path, now):
        started = now()
        previous = read_verified_status(status_path) or {}
        base = {
            "schema": STATUS_SCHEMA,
            "running": True,
            "locked": True,
            "lastStartedAtUtc": _iso(started),
            "lastSucceededAtUtc": previous.get("lastSucceededAtUtc"),
            "lastFailedAtUtc": previous.get("lastFailedAtUtc"),
            "lastError": None,
            "updatedAtUtc": _iso(started),
            "staleAfterUtc": _iso(started + timedelta(hours=48)),
            "timeframes": previous.get("timeframes", []),
            "runIndexFileName": previous.get("runIndexFileName"),
            "runIndexSha256": previous.get("runIndexSha256"),
            "goldenSemanticLedgerSha256": previous.get("goldenSemanticLedgerSha256"),
        }
        _write_status_atomic(status_path, base)
        staging = output_dir / f".staging-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            golden = verify_golden_fixture()
            cutoffs = (
                {timeframe: forced_cutoff_ms for timeframe in TIMEFRAMES}
                if forced_cutoff_ms is not None
                else dict(cutoff_loader())
            )
            if set(cutoffs) != set(TIMEFRAMES) or any(not isinstance(value, int) or value <= 0 for value in cutoffs.values()):
                raise ValueError("cutoffs must contain positive 1h, 4h and 1d timestamps")
            staged_runs: list[tuple[str, Mapping[str, Path], Mapping[str, Any], Mapping[str, Any]]] = []
            for timeframe in TIMEFRAMES:
                cutoff = int(cutoffs[timeframe])
                snapshot = snapshot_loader(timeframe, cutoff)
                config = EvaluationConfig(
                    cutoff_ms=cutoff,
                    bootstrap_samples=bootstrap_samples,
                    block_size_events=block_size_events,
                    dedup_bars=dedup_bars,
                )
                frozen, rows, report = evaluate(snapshot, config)
                timeframe_stage = staging / timeframe
                paths = write_bundle(timeframe_stage, frozen, rows, report, config)
                verification = verify_bundle(paths["manifest"])
                staged_runs.append((timeframe, paths, report, verification))

            published: list[dict[str, Any]] = []
            for timeframe, paths, report, verification in staged_runs:
                manifest = bundle_publisher(paths, output_dir)
                contract = report.get("modules", {}).get("contract", {})
                published.append(
                    {
                        "timeframe": timeframe,
                        "cutoffMs": report["scope"]["cutoffMs"],
                        "manifestSha256": manifest.name.split(".", 1)[0],
                        "manifestFileName": manifest.name,
                        "stored": report["counts"]["stored"],
                        "eligible": report["counts"]["eligible"],
                        "excluded": report["counts"]["excluded"],
                        "realizedAtMaxHorizon": report["counts"]["realizedAtMaxHorizon"],
                        "definitionsSha256": contract.get("definitionsSha256"),
                        "semanticVerification": verification.get("valid") is True,
                        "runtimeFileName": paths["runtime"].name,
                        "runtimeSha256": paths["runtime"].name.split(".", 1)[0],
                    }
                )
            completed = now()
            run_index_path, run_pointer = run_index_publisher(output_dir, completed, published)
            success = {
                **base,
                "running": False,
                "locked": False,
                "lastSucceededAtUtc": _iso(completed),
                "lastError": None,
                "updatedAtUtc": _iso(completed),
                "staleAfterUtc": _iso(completed + timedelta(hours=48)),
                "timeframes": published,
                "runIndexFileName": run_index_path.name,
                "runIndexSha256": run_pointer["runIndexSha256"],
                "goldenSemanticLedgerSha256": golden["crossLanguageSemanticLedgerSha256"],
            }
            return _write_status_atomic(status_path, success)
        except Exception as exc:
            failed = now()
            # Do not serialize connection strings, paths, SQL or exception repr.
            safe_error = f"{type(exc).__name__}: pipeline run failed; inspect local service logs"
            failure = {
                **base,
                "running": False,
                "locked": False,
                "lastFailedAtUtc": _iso(failed),
                "lastError": safe_error,
                "updatedAtUtc": _iso(failed),
                "staleAfterUtc": _iso(failed + timedelta(hours=48)),
            }
            _write_status_atomic(status_path, failure)
            raise
        finally:
            if staging.parent == output_dir and staging.name.startswith(".staging-"):
                shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite BTC technical descriptive evidence pipeline")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/research/evidence/technical-descriptive"))
    parser.add_argument("--cutoff-ms", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--block-size-events", type=int, default=8)
    parser.add_argument("--dedup-bars", type=int, default=6)
    parser.add_argument("--recover-stale-lock", action="store_true")
    parser.add_argument("--max-lock-age-hours", type=float, default=12.0)
    args = parser.parse_args()
    if args.recover_stale_lock:
        recovered = recover_stale_lock(
            args.output_dir.resolve() / LOCK_FILE,
            max_age=timedelta(hours=args.max_lock_age_hours),
        )
        print(json.dumps({"recoveredLock": str(recovered)}, indent=2))
        return
    try:
        status = run_pipeline(
            args.output_dir,
            forced_cutoff_ms=args.cutoff_ms,
            bootstrap_samples=args.bootstrap_samples,
            block_size_events=args.block_size_events,
            dedup_bars=args.dedup_bars,
        )
    except PipelineLockedError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
