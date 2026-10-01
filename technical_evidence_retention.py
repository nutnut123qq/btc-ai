#!/usr/bin/env python3
"""Reference-aware retention for immutable technical-evidence bundles.

Planning is always read-only.  Applying requires the exact hash of a fresh plan;
every candidate is re-hashed immediately before deletion.  Only known
content-addressed evidence file names directly inside the selected directory are
eligible.  The active latest-success graph is never deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from run_technical_evidence_pipeline import (
    RUN_INDEX_SCHEMA,
    RUN_POINTER_FILE,
    TIMEFRAMES,
    canonical_json,
    read_verified_success,
    sha256_bytes,
)


PLAN_SCHEMA = "btc-technical-evidence-retention-plan/v1"
CONTENT_FILE = re.compile(
    r"^[0-9a-f]{64}\.(?:run-index\.json|manifest\.json|snapshot\.json|ledger\.jsonl|report\.json|runtime\.json)$"
)


def _safe_child(directory: Path, name: str) -> Path:
    if Path(name).name != name or not CONTENT_FILE.fullmatch(name):
        raise ValueError(f"unsafe or unsupported evidence file name: {name}")
    path = directory / name
    if path.parent.resolve() != directory.resolve():
        raise ValueError("retention path traversal rejected")
    return path


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest_graph(directory: Path) -> tuple[dict[str, set[str]], set[str]]:
    graph: dict[str, set[str]] = {}
    all_known: set[str] = set()
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not CONTENT_FILE.fullmatch(path.name):
            continue
        all_known.add(path.name)
        if not path.name.endswith(".manifest.json"):
            continue
        expected = path.name.split(".", 1)[0]
        if _content_hash(path) != expected:
            raise ValueError(f"manifest content hash mismatch: {path.name}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {"snapshot", "ledger", "report"}:
            raise ValueError(f"manifest artifact graph is invalid: {path.name}")
        references = {path.name}
        for role in ("snapshot", "ledger", "report"):
            item = artifacts[role]
            name = item.get("fileName") if isinstance(item, Mapping) else None
            if not isinstance(name, str):
                raise ValueError(f"manifest artifact name missing: {path.name}")
            artifact = _safe_child(directory, name)
            if not artifact.is_file() or _content_hash(artifact) != item.get("sha256"):
                raise ValueError(f"manifest artifact integrity mismatch: {path.name}:{role}")
            references.add(name)
        graph[path.name] = references
    return graph, all_known


def _load_run_indexes(directory: Path, manifest_graph: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.run-index.json")):
        if not CONTENT_FILE.fullmatch(path.name):
            raise ValueError(f"unsupported run-index name: {path.name}")
        digest = _content_hash(path)
        if path.name != f"{digest}.run-index.json":
            raise ValueError(f"run-index content hash mismatch: {path.name}")
        value = json.loads(path.read_text(encoding="utf-8"))
        bundles = value.get("bundles")
        if (
            value.get("schema") != RUN_INDEX_SCHEMA
            or value.get("status") != "succeeded"
            or not isinstance(bundles, list)
            or [bundle.get("timeframe") for bundle in bundles] != list(TIMEFRAMES)
        ):
            raise ValueError(f"run-index schema/trio mismatch: {path.name}")
        references = {path.name}
        for bundle in bundles:
            manifest_name = bundle.get("manifestFileName")
            manifest_sha = bundle.get("manifestSha256")
            if not isinstance(manifest_name, str) or manifest_name != f"{manifest_sha}.manifest.json":
                raise ValueError(f"run-index manifest declaration mismatch: {path.name}")
            if manifest_name not in manifest_graph:
                raise ValueError(f"run-index manifest graph missing: {path.name}:{manifest_name}")
            references.update(manifest_graph[manifest_name])
            runtime_name = bundle.get("runtimeFileName")
            runtime_sha = bundle.get("runtimeSha256")
            if (runtime_name is None) != (runtime_sha is None):
                raise ValueError(f"run-index runtime declaration mismatch: {path.name}")
            if runtime_name is not None:
                if runtime_name != f"{runtime_sha}.runtime.json":
                    raise ValueError(f"run-index runtime name/hash mismatch: {path.name}")
                runtime_path = _safe_child(directory, runtime_name)
                if not runtime_path.is_file() or _content_hash(runtime_path) != runtime_sha:
                    raise ValueError(f"run-index runtime integrity mismatch: {path.name}:{runtime_name}")
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                if (
                    runtime.get("schema") != "btc-technical-evidence-runtime/v1"
                    or runtime.get("semanticManifestFileName") != manifest_name
                    or runtime.get("semanticManifestSha256") != manifest_sha
                    or runtime.get("semanticHashIncludesRuntimeMetadata") is not False
                ):
                    raise ValueError(f"run-index runtime semantic binding mismatch: {path.name}:{runtime_name}")
                references.add(runtime_name)
        try:
            completed = datetime.fromisoformat(str(value["completedAtUtc"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"run-index completion timestamp invalid: {path.name}") from exc
        runs.append({"name": path.name, "sha256": digest, "completed": completed, "references": references})
    return sorted(runs, key=lambda run: (run["completed"], run["name"]), reverse=True)


def build_retention_plan(output_dir: Path, *, keep_successful_runs: int = 2) -> dict[str, Any]:
    if keep_successful_runs < 2:
        raise ValueError("at least two successful runs must be retained")
    directory = output_dir.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"evidence directory does not exist: {directory}")
    active = read_verified_success(directory, verify_semantics=False)
    pointer = json.loads((directory / RUN_POINTER_FILE).read_text(encoding="utf-8"))
    active_index = str(pointer["runIndexFileName"])
    manifest_graph, all_known = _load_manifest_graph(directory)
    runs = _load_run_indexes(directory, manifest_graph)

    blocked_reason = None
    if len(runs) < keep_successful_runs:
        blocked_reason = (
            f"only {len(runs)} valid successful run(s) exist; retention requires "
            f"at least {keep_successful_runs}"
        )
    kept_run_names = {run["name"] for run in runs[:keep_successful_runs]}
    kept_run_names.add(active_index)
    protected: set[str] = set()
    for run in runs:
        if run["name"] in kept_run_names:
            protected.update(run["references"])
    if active_index not in protected:
        raise ValueError("active run index is not present in the validated retention graph")

    candidates: list[dict[str, Any]] = []
    if blocked_reason is None:
        for name in sorted(all_known - protected):
            path = _safe_child(directory, name)
            candidates.append({
                "fileName": name,
                "sha256": _content_hash(path),
                "sizeBytes": path.stat().st_size,
            })
    core = {
        "schema": PLAN_SCHEMA,
        "directory": str(directory),
        "keepSuccessfulRuns": keep_successful_runs,
        "activeRunIndexFileName": active_index,
        "activeRunIndexSha256": active["runIndexSha256"],
        "validSuccessfulRunCount": len(runs),
        "keptRunIndexFileNames": sorted(kept_run_names),
        "protectedFileCount": len(protected),
        "blockedReason": blocked_reason,
        "candidateCount": len(candidates),
        "reclaimableBytes": sum(item["sizeBytes"] for item in candidates),
        "candidates": candidates,
    }
    return {**core, "planSha256": sha256_bytes(canonical_json(core).encode())}


def apply_retention_plan(
    output_dir: Path,
    *,
    keep_successful_runs: int,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    plan = build_retention_plan(output_dir, keep_successful_runs=keep_successful_runs)
    if plan["blockedReason"] is not None:
        raise RuntimeError(plan["blockedReason"])
    if expected_plan_sha256 != plan["planSha256"]:
        raise ValueError("retention plan hash changed; rerun dry-run and review")
    directory = output_dir.resolve()
    removed: list[dict[str, Any]] = []
    for item in plan["candidates"]:
        path = _safe_child(directory, item["fileName"])
        if not path.is_file() or _content_hash(path) != item["sha256"] or path.stat().st_size != item["sizeBytes"]:
            raise RuntimeError(f"retention candidate changed before apply: {item['fileName']}")
        original_mode = path.stat().st_mode
        try:
            path.chmod(original_mode | stat.S_IWRITE)
            path.unlink()
        except Exception:
            if path.exists():
                path.chmod(original_mode)
            raise
        removed.append(dict(item))
    return {
        "schema": PLAN_SCHEMA,
        "appliedPlanSha256": plan["planSha256"],
        "removedCount": len(removed),
        "reclaimedBytes": sum(item["sizeBytes"] for item in removed),
        "removed": removed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-aware technical evidence retention")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/research/evidence/technical-descriptive"))
    parser.add_argument("--keep-successful-runs", type=int, default=2)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-plan-sha256")
    args = parser.parse_args()
    if args.apply:
        if not args.confirm_plan_sha256:
            parser.error("--apply requires --confirm-plan-sha256 from a reviewed dry-run")
        result = apply_retention_plan(
            args.output_dir,
            keep_successful_runs=args.keep_successful_runs,
            expected_plan_sha256=args.confirm_plan_sha256,
        )
    else:
        result = build_retention_plan(args.output_dir, keep_successful_runs=args.keep_successful_runs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
