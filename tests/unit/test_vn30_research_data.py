"""VN30-2 research snapshot: written once, never overwritten, every hash verified before data is returned
(fail closed), zero-volume bars kept but not sessions, membership point in time."""
import json
import os
import stat
import sys
from datetime import datetime
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AI_ROOT))
sys.path.insert(0, str(AI_ROOT / "scripts"))

import vn30_research_data as vrd  # noqa: E402

FIRST = [f"T{i:02d}" for i in range(30)]
SECOND = FIRST[:-1] + ["T30"]


def _membership(tmp_path: Path) -> Path:
    path = tmp_path / "membership.json"
    path.write_text(json.dumps({"periods": [
        {"effectiveDate": "2018-01-22", "constituents": FIRST, "removed": ["OLD"]},
        {"effectiveDate": "2018-07-23", "constituents": SECOND}]}), encoding="utf-8")
    return path


def _rows(dates, start=10.0, zero=()):
    return [{"date": d, "open": start + i, "high": start + i + 0.5, "low": start + i - 0.5, "close": start + i + 0.25,
             "volume": 0 if d in zero else 1000 + i} for i, d in enumerate(dates)]


DATES = ["2018-01-19", "2018-01-22", "2018-01-23", "2018-01-24", "2018-07-23", "2018-07-24"]


def _series():
    return {"VNINDEX": _rows(DATES, 900.0, zero={"2018-01-23"}),
            "T00": _rows(DATES, 10.1, zero={"2018-01-23"}),
            "T29": _rows(DATES[:5], 20.0),
            "T30": _rows(DATES[3:], 30.0)}


ROLES = {"VNINDEX": "index", "T00": "member", "T29": "member", "T30": "member"}


def _write(tmp_path, **kw):
    return vrd.write_snapshot(tmp_path / "snaps", "20261106", _series(), ROLES, _membership(tmp_path),
                              meta={"source": {"name": "test"}}, missing={"ZZZ": "HTTPError 404"}, **kw)


def _writable(path: Path) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def test_round_trip_verifies_and_exposes_calendar_bars_and_members(tmp_path):
    snap_dir = _write(tmp_path)
    snap = vrd.load_snapshot(snap_dir)
    assert snap_dir.name == f"20261106-{snap.sha256[:12]}"
    assert snap.sha256 == vrd.sha256_bytes((snap_dir / "manifest.json").read_bytes())
    assert snap.symbols == ("T00", "T29", "T30", "VNINDEX")
    # the zero-volume VNINDEX bar is kept in the file but is not a session
    assert "2018-01-23" not in snap.sessions and len(snap.sessions) == len(DATES) - 1
    assert [b.date for b in snap.bars("T00", include_zero_volume=True)] == DATES
    assert "2018-01-23" not in [b.date for b in snap.bars("T00")]
    assert snap.bars("T00", include_zero_volume=True)[2].zero_volume
    assert snap.bars("T00")[0].close == 10.35  # float written and read back exactly
    assert snap.members_on("2018-01-19") is None
    assert snap.members_on("2018-07-20") == frozenset(FIRST)
    assert snap.members_on("2018-07-23") == frozenset(SECOND)
    m = snap.manifest
    assert m["missing"] == {"ZZZ": "HTTPError 404"}
    assert m["symbols"]["T00"]["zeroVolumeDates"] == ["2018-01-23"]
    assert m["symbols"]["T29"]["inForce"] == [{"from": "2018-01-22", "toExclusive": "2018-07-23"}]
    assert m["symbols"]["T30"]["inForce"] == [{"from": "2018-07-23", "toExclusive": None}]
    assert "inForce" not in m["symbols"]["VNINDEX"]
    assert set(m["files"]) == {"membership.json", "T00.csv", "T29.csv", "T30.csv", "VNINDEX.csv"}
    with pytest.raises(KeyError, match="missing: HTTPError 404"):
        snap.bars("ZZZ")


def test_writer_refuses_to_overwrite_an_existing_snapshot(tmp_path):
    first = _write(tmp_path)
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    with pytest.raises(vrd.SnapshotExistsError):
        _write(tmp_path)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before
    assert [p.name for p in (tmp_path / "snaps").iterdir()] == [first.name]  # staging cleaned up


def test_files_are_read_only_after_writing(tmp_path):
    snap_dir = _write(tmp_path)
    for p in snap_dir.iterdir():
        assert not os.access(p, os.W_OK)


def test_tampered_csv_fails_closed(tmp_path):
    snap_dir = _write(tmp_path)
    p = snap_dir / "T00.csv"
    _writable(p)
    p.write_bytes(p.read_bytes().replace(b"10.35", b"10.36"))
    with pytest.raises(vrd.SnapshotError, match="sha256 mismatch for T00.csv"):
        vrd.load_snapshot(snap_dir)


def test_tampered_manifest_fails_closed(tmp_path):
    snap_dir = _write(tmp_path)
    p = snap_dir / "manifest.json"
    _writable(p)
    p.write_bytes(p.read_bytes().replace(b'"test"', b'"other"'))
    with pytest.raises(vrd.SnapshotError, match="manifest hash"):
        vrd.load_snapshot(snap_dir)


def test_extra_or_missing_files_fail_closed(tmp_path):
    snap_dir = _write(tmp_path)
    (snap_dir / "EXTRA.csv").write_text("date\n", encoding="utf-8")
    with pytest.raises(vrd.SnapshotError, match="unlisted"):
        vrd.load_snapshot(snap_dir)
    (snap_dir / "EXTRA.csv").unlink()
    _writable(snap_dir / "T30.csv")
    (snap_dir / "T30.csv").unlink()
    with pytest.raises(vrd.SnapshotError, match="absent"):
        vrd.load_snapshot(snap_dir)


def test_renamed_folder_and_wrong_expected_hash_fail_closed(tmp_path):
    snap_dir = _write(tmp_path)
    sha = vrd.load_snapshot(snap_dir).sha256
    assert vrd.load_snapshot(snap_dir, expected_sha256=sha[:12]).sha256 == sha
    with pytest.raises(vrd.SnapshotError, match="not the expected"):
        vrd.load_snapshot(snap_dir, expected_sha256="0" * 64)
    with pytest.raises(vrd.SnapshotError, match="not the expected"):
        vrd.load_snapshot(snap_dir, expected_sha256=sha[:6])  # too short to pin a snapshot
    moved = snap_dir.with_name("20261106-000000000000")
    snap_dir.rename(moved)
    with pytest.raises(vrd.SnapshotError, match="folder name"):
        vrd.load_snapshot(moved)


def test_writer_rejects_unusable_input(tmp_path):
    membership = _membership(tmp_path)
    series = _series()
    with pytest.raises(ValueError, match="VNINDEX is required"):
        vrd.write_snapshot(tmp_path, "20261106", {"T00": series["T00"]}, ROLES, membership)
    dup = dict(series, T00=series["T00"] + [series["T00"][-1]])
    with pytest.raises(ValueError, match="strictly increasing"):
        vrd.write_snapshot(tmp_path, "20261106", dup, ROLES, membership)
    with pytest.raises(ValueError, match="unsafe symbol"):
        vrd.write_snapshot(tmp_path, "20261106", dict(series, **{"../X": series["T00"]}), ROLES, membership)
    with pytest.raises(ValueError, match="stamp"):
        vrd.write_snapshot(tmp_path, "2026-11-06", series, ROLES, membership)
    assert not any(tmp_path.glob("2026*"))


def test_summary_flags_ohlc_anomalies_without_dropping_bars():
    rows = _rows(DATES[:3])
    rows[1]["high"] = rows[1]["low"] - 1
    s = vrd.summarize("X", rows)
    assert s["bars"] == 3 and s["ohlcAnomalyDates"] == [DATES[1]]


def test_third_thursday():
    assert vrd.third_thursday(2026, 10) == "2026-10-15" and vrd.third_thursday(2026, 1) == "2026-01-15"
    assert vrd.third_thursday(2025, 5) == "2025-05-15"


def test_front_month_roll_check_follows_expiry_and_skips_unserved_months():
    def r(d, c):
        return {"date": d, "open": c, "close": c}
    contracts = {(2026, 10): [r("2026-10-14", 1.0), r("2026-10-15", 2.0)],
                 (2026, 11): [r("2026-10-14", 9.0), r("2026-10-15", 9.0), r("2026-10-16", 3.0), r("2026-11-19", 4.0)]}
    good = [r("2026-10-13", 7.0), r("2026-10-14", 1.0), r("2026-10-15", 2.0), r("2026-10-16", 3.0),
            r("2026-11-19", 4.0), r("2026-11-20", 5.0)]
    res = vrd.verify_front_month_roll(good, contracts)
    # 10-13 precedes the first served bar; 11-20 rolls to December, which is not served
    assert (res["daysChecked"], res["daysMatchingFront"], res["daysFrontNotServed"]) == (4, 4, 1)
    assert res["contracts"]["2026-10"]["expiry"] == "2026-10-15"
    assert res["contracts"]["2026-11"]["expiry"] == "2026-11-19"  # still live: third Thursday
    bad = good[:2] + [r("2026-10-15", 9.0)] + good[3:]
    res = vrd.verify_front_month_roll(bad, contracts)
    assert res["mismatches"] == [{"date": "2026-10-15", "front": "2026-10"}]
    assert vrd.verify_front_month_roll(good, {})["daysChecked"] == 0


def test_exclusion_keeps_bars_hashed_but_out_of_bars_by_default(tmp_path):
    snap_dir = vrd.write_snapshot(tmp_path / "snaps", "20261106", _series(), ROLES, _membership(tmp_path),
                                  exclusions={"T00": {"before": "2018-07-23", "reason": "ticker reuse"}})
    snap = vrd.load_snapshot(snap_dir)
    e = snap.manifest["symbols"]["T00"]
    assert (e["excludeBefore"], e["barsExcluded"], e["usableFirst"]) == ("2018-07-23", 4, "2018-07-23")
    assert [b.date for b in snap.bars("T00")] == ["2018-07-23", "2018-07-24"]
    assert len(snap.bars("T00", include_zero_volume=True, include_excluded=True)) == len(DATES)
    assert [c for c in snap.coverage() if c["symbol"] == "T00"][0]["usableFirst"] == "2018-07-23"
    with pytest.raises(ValueError, match="exclusion"):
        vrd.write_snapshot(tmp_path / "x", "20261106", _series(), ROLES, _membership(tmp_path),
                           exclusions={"T00": {"before": "2018-07-23"}})


def test_builder_roles_and_final_bar_cutoff():
    import build_vn30_research_snapshot as b
    from probe_mbb_market_data import VN_TZ
    roles = b.symbol_roles(AI_ROOT / "contracts" / "vn30-membership.json")
    assert roles["VNINDEX"] == roles["VN30"] == "index" and roles["VN30F1M"] == "future-continuous"
    assert roles["E1VFVN30"] == roles["FUEVFVND"] == "etf"
    assert roles["ROS"] == "member" and roles["KBC"] == "leaver-at-first-period"
    assert sum(v == "member" for v in roles.values()) == 54
    assert b.EXCLUSIONS["VHM"]["before"] == "2018-05-17"
    assert b.cutoff_exclusive(datetime(2026, 11, 6, 15, 59, tzinfo=VN_TZ), "16:00") == "2026-11-06"
    assert b.cutoff_exclusive(datetime(2026, 11, 6, 16, 0, tzinfo=VN_TZ), "16:00") == "2026-11-07"


def test_verify_cli(tmp_path, capsys):
    snap_dir = _write(tmp_path)
    assert vrd.main(["verify", str(snap_dir)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["lastSession"] == "2018-07-24" and out["missing"] == {"ZZZ": "HTTPError 404"}
    _writable(snap_dir / "T29.csv")
    (snap_dir / "T29.csv").write_text("x", encoding="utf-8")
    assert vrd.main(["verify", str(snap_dir)]) == 1
