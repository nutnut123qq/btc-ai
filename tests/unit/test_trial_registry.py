"""trial_registry: append-only, hash-chained, counts trials for multiple-testing corrections."""

import json

import pytest

import trial_registry as tr


def _fields(i: str, market: str = "HOSE", status: str = "run", trials: int = 3, exact: bool = True, **extra):
    return {"id": i, "registeredAt": "2026-11-06", "market": market, "family": "f", "status": status,
            "trials": trials, "trialsExact": exact, "description": "d", "source": "s", **extra}


def test_append_chains_and_counts(tmp_path):
    path = tmp_path / "r.jsonl"
    a = tr.append(_fields("a"), path)
    b = tr.append(_fields("b", trials=5, exact=False), path)
    tr.append(_fields("c", market="BTC", trials=100), path)
    tr.append(_fields("d", status="planned", trials=7), path)
    assert a["prevSha256"] == tr.GENESIS and b["prevSha256"] == a["sha256"]
    reg = tr.load(path)
    hose = reg.count("HOSE")
    assert (hose["entries"], hose["trials"], hose["lowerBound"]) == (2, 8, True)
    assert reg.count("HOSE", ("run", "planned"))["trials"] == 15
    assert reg.count()["trials"] == 108


def test_editing_an_earlier_line_is_detected(tmp_path):
    path = tmp_path / "r.jsonl"
    tr.append(_fields("a"), path)
    tr.append(_fields("b"), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["trials"] = 1  # quietly lowering a trial count
    lines[0] = json.dumps(first, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(tr.RegistryError, match="sha256 mismatch"):
        tr.load(path)


def test_deleting_a_line_breaks_the_chain(tmp_path):
    path = tmp_path / "r.jsonl"
    for i in "abc":
        tr.append(_fields(i), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    with pytest.raises(tr.RegistryError, match="chain"):
        tr.load(path)


def test_supersedes_replaces_in_counts_and_must_exist(tmp_path):
    path = tmp_path / "r.jsonl"
    tr.append(_fields("a", trials=2), path)
    tr.append(_fields("a2", trials=9, supersedes="a"), path)
    assert tr.load(path).count("HOSE")["trials"] == 9
    with pytest.raises(tr.RegistryError, match="supersedes unknown"):
        tr.append(_fields("x", supersedes="nope"), path)


@pytest.mark.parametrize("bad", [{"trials": 0}, {"trials": True}, {"status": "done"}, {"market": "NYSE"}, {"trialsExact": "yes"}])
def test_invalid_entries_are_refused(tmp_path, bad):
    with pytest.raises(tr.RegistryError):
        tr.append({**_fields("a"), **bad}, tmp_path / "r.jsonl")


def test_duplicate_id_is_refused(tmp_path):
    path = tmp_path / "r.jsonl"
    tr.append(_fields("a"), path)
    with pytest.raises(tr.RegistryError, match="duplicate"):
        tr.append(_fields("a"), path)


def test_committed_registry_verifies():
    reg = tr.load()
    assert reg.entries, "the committed registry must not be empty"
    assert any(e["id"] == "hose-vn30-1-rotation" for e in reg.entries)


def test_frozen_protocol_files_match_their_registered_hash():
    import hashlib
    from pathlib import Path

    root = Path(tr.__file__).resolve().parent
    frozen = [e for e in tr.load().active() if e.get("specSha256")]
    assert any(e["id"] == "hose-vn30-2" or e.get("supersedes") == "hose-vn30-2" for e in frozen)
    for entry in frozen:
        raw = (root / entry["source"]).read_bytes()
        assert b"\r" not in raw, f"{entry['source']} must stay LF (its hash is frozen)"
        assert hashlib.sha256(raw).hexdigest() == entry["specSha256"], f"{entry['source']} changed after it was frozen"
