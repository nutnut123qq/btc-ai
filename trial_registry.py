"""Append-only trial registry (roadmap 0.5): every strategy/model/config ever evaluated, so that the
multiple-testing count used by the deflated Sharpe ratio (DSR) or PBO is the real one, not "the 6 configs of
this run".

Storage: ``docs/research/trial-registry/registry.jsonl`` (committed, one JSON object per line). Each entry carries
``prevSha256`` (the previous line's ``sha256``) and its own ``sha256`` over the canonical JSON of every other field,
so an edit or deletion of an earlier line breaks verification. Entries are never edited: a correction is a new entry
with ``supersedes`` naming the old id.

    python trial_registry.py verify
    python trial_registry.py count [--market HOSE] [--status run]

Counting rule: ``trials`` is the number of distinct configurations, rules, prompts/skill versions or hypotheses
actually evaluated against outcomes (not folds, not bootstrap resamples). ``trialsExact = false`` marks a documented
lower bound. Planned entries (``status = planned``) are registered before any result is seen and counted once run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

REGISTRY_SCHEMA = "trial-registry/v1"
REGISTRY_PATH = Path(__file__).resolve().parent / "docs" / "research" / "trial-registry" / "registry.jsonl"
GENESIS = "0" * 64
STATUSES = ("planned", "run", "abandoned")
MARKETS = ("HOSE", "BTC")
REQUIRED = ("schema", "id", "registeredAt", "market", "family", "status", "trials", "trialsExact",
            "description", "source", "prevSha256", "sha256")


class RegistryError(ValueError):
    """The registry file is malformed or was modified after the fact."""


def _canonical(entry: Mapping[str, Any]) -> bytes:
    body = {k: v for k, v in entry.items() if k != "sha256"}
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def entry_hash(entry: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(entry)).hexdigest()


def _validate(entry: Mapping[str, Any], line_no: int) -> None:
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise RegistryError(f"line {line_no}: missing {', '.join(missing)}")
    if entry["schema"] != REGISTRY_SCHEMA:
        raise RegistryError(f"line {line_no}: schema {entry['schema']!r} != {REGISTRY_SCHEMA!r}")
    if entry["status"] not in STATUSES:
        raise RegistryError(f"line {line_no}: status {entry['status']!r} not in {STATUSES}")
    if entry["market"] not in MARKETS:
        raise RegistryError(f"line {line_no}: market {entry['market']!r} not in {MARKETS}")
    if not isinstance(entry["trials"], int) or isinstance(entry["trials"], bool) or entry["trials"] < 1:
        raise RegistryError(f"line {line_no}: trials must be a positive integer")
    if not isinstance(entry["trialsExact"], bool):
        raise RegistryError(f"line {line_no}: trialsExact must be a boolean")


@dataclass(frozen=True)
class Registry:
    entries: tuple[dict[str, Any], ...]

    @property
    def head(self) -> str:
        return self.entries[-1]["sha256"] if self.entries else GENESIS

    def active(self) -> list[dict[str, Any]]:
        """Entries not superseded by a later correction."""
        superseded = {e["supersedes"] for e in self.entries if e.get("supersedes")}
        return [e for e in self.entries if e["id"] not in superseded]

    def count(self, market: str | None = None, statuses: Iterable[str] = ("run",)) -> dict[str, Any]:
        wanted = set(statuses)
        rows = [e for e in self.active() if e["status"] in wanted and (market is None or e["market"] == market)]
        return {
            "market": market or "all",
            "statuses": sorted(wanted),
            "entries": len(rows),
            "trials": sum(e["trials"] for e in rows),
            "lowerBound": any(not e["trialsExact"] for e in rows),
            "byFamily": {f: sum(e["trials"] for e in rows if e["family"] == f) for f in sorted({e["family"] for e in rows})},
        }


def load(path: Path | str = REGISTRY_PATH) -> Registry:
    path = Path(path)
    if not path.exists():
        return Registry(())
    entries: list[dict[str, Any]] = []
    prev, ids = GENESIS, set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise RegistryError(f"line {line_no}: blank line")
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"line {line_no}: not JSON ({exc.msg})") from exc
        _validate(entry, line_no)
        if entry["prevSha256"] != prev:
            raise RegistryError(f"line {line_no}: prevSha256 does not chain to the previous entry")
        if entry_hash(entry) != entry["sha256"]:
            raise RegistryError(f"line {line_no}: sha256 mismatch (entry {entry['id']!r} was modified)")
        if entry["id"] in ids:
            raise RegistryError(f"line {line_no}: duplicate id {entry['id']!r}")
        if entry.get("supersedes") and entry["supersedes"] not in ids:
            raise RegistryError(f"line {line_no}: supersedes unknown id {entry['supersedes']!r}")
        ids.add(entry["id"])
        prev = entry["sha256"]
        entries.append(entry)
    return Registry(tuple(entries))


def append(fields: Mapping[str, Any], path: Path | str = REGISTRY_PATH) -> dict[str, Any]:
    """Append one entry (fields without schema/prevSha256/sha256), chained to the verified current head."""
    path = Path(path)
    registry = load(path)
    entry = {"schema": REGISTRY_SCHEMA, **dict(fields), "prevSha256": registry.head}
    entry["sha256"] = entry_hash(entry)
    _validate(entry, len(registry.entries) + 1)
    if any(e["id"] == entry["id"] for e in registry.entries):
        raise RegistryError(f"duplicate id {entry['id']!r}")
    if entry.get("supersedes") and not any(e["id"] == entry["supersedes"] for e in registry.entries):
        raise RegistryError(f"supersedes unknown id {entry['supersedes']!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return entry


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Append-only trial registry")
    parser.add_argument("command", choices=("verify", "count"))
    parser.add_argument("--market", choices=MARKETS)
    parser.add_argument("--status", action="append", choices=STATUSES)
    parser.add_argument("--path", default=str(REGISTRY_PATH))
    args = parser.parse_args(argv)
    try:
        registry = load(args.path)
    except RegistryError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    if args.command == "verify":
        print(json.dumps({"ok": True, "entries": len(registry.entries), "head": registry.head}))
    else:
        print(json.dumps(registry.count(args.market, args.status or ("run",)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
