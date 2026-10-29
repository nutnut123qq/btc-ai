"""Skill ``equity-technical`` (EQTECH-1 lane K).

Pins the chat skill against the code it describes: the direction-score-v1
thresholds and horizons must equal the barrier configs of ``equity_base_rates``
(imported, never re-typed), the note it records must carry no probability, and
it may only point the session at the technical/candles/research-notes endpoints.
Reads files only — no DB, no network.
"""

import json
import re
import sys
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AI_ROOT))

import equity_base_rates as ebr

SKILL_DIR = AI_ROOT / "skills" / "equity-technical"
SKILL_PATH = SKILL_DIR / "SKILL.md"

ALLOWED_API_PATHS = {
    "/api/equities/{symbol}/technical",
    "/api/equities/{symbol}/candles",
    "/api/equities/{symbol}/research-notes",
    "/api/equities/{symbol}/research-notes/direction-scores",
}
# Backend route segments that serve fundamentals, events, documents, news or the
# mixed dossier/snapshot payloads; the technical-only skill must not name them.
FORBIDDEN_ROUTE_SEGMENTS = (
    "dossier", "context", "fundamentals", "financial-facts", "source-documents", "documents",
    "news", "events", "llm-runs", "research-snapshot", "research-readiness", "base-rates", "rag",
)
NOTE_BODY_KEYS = {
    "schemaVersion", "thesis", "evidence", "counterEvidence", "scenarios", "limitations",
    "missingData", "outcomeCheck", "probability",
}


def _pct(fraction: float) -> float:
    return round(fraction * 100, 6)


@pytest.fixture(scope="module")
def skill() -> str:
    assert SKILL_PATH.is_file(), f"missing skill file {SKILL_PATH}"
    # read_text uses universal newlines, so a CRLF checkout (core.autocrlf) reads the same.
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def note_example(skill: str) -> dict:
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", skill, flags=re.S)
    assert len(blocks) == 1, "expected exactly one ```json note example"
    return json.loads(blocks[0])


def test_front_matter_names_the_skill(skill):
    m = re.match(r"---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", skill, flags=re.S)
    assert m, "SKILL.md must start with a --- front matter block"
    fields = dict(re.findall(r"^(name|description):[ \t]*([^\r\n]+)", m.group(1), flags=re.M))
    assert fields.get("name", "").strip() == "equity-technical"
    description = fields.get("description", "")
    assert "MBB" in description and "BID" in description
    assert "/api/equities/{symbol}/technical" in description


def test_threshold_table_equals_barrier_configs(skill):
    rows = re.findall(
        r"^[ \t]*\|\s*(\d+)\s*\|\s*\+(\d+(?:\.\d+)?)%\s*\|\s*[-−](\d+(?:\.\d+)?)%\s*\|\s*`(tb-[a-z0-9-]+)`\s*\|",
        skill, flags=re.M)
    stated = {(int(h), float(up), float(down), cid) for h, up, down, cid in rows}
    expected = {(c.horizon_sessions, _pct(c.target_pct), _pct(c.stop_pct), c.config_id) for c in ebr.CONFIGS}
    assert len(rows) == len(expected), rows
    assert stated == expected
    # The scorer's bounds are inclusive and the down threshold is negative.
    assert set(re.findall(r"\bR\s*([<>]=?)", skill)) == {">=", "<="}
    assert not re.search(r"\bR\s*<=\s*\+", skill)
    assert "`downThresholdPct` -0.04" in skill


def test_only_the_config_horizons_are_allowed(skill, note_example):
    horizons = sorted(c.horizon_sessions for c in ebr.CONFIGS)
    assert f"**{' or '.join(str(h) for h in horizons)} only**" in skill
    assert note_example["horizonSessions"] in horizons
    h = note_example["horizonSessions"]
    assert all(s["horizonSessions"] == h for s in note_example["body"]["scenarios"])
    assert note_example["body"]["outcomeCheck"]["dueAfterSessions"] == h
    cfg = next(c for c in ebr.CONFIGS if c.horizon_sessions == h)
    how = note_example["body"]["outcomeCheck"]["how"]
    assert "direction-score-v1" in how
    assert f"R >= +{_pct(cfg.target_pct):g}%" in how
    assert f"R <= -{_pct(cfg.stop_pct):g}%" in how


def test_names_rule_versions_and_endpoints(skill):
    assert "direction-score-v1" in skill
    assert "scoring.ruleVersion" in skill
    assert 'skillVersion: "equity-technical-v1"' in skill
    assert 'methodVersion: "equity-technical-method-v1"' in skill
    assert "GET /api/equities/{symbol}/technical" in skill
    assert "GET /api/equities/{symbol}/research-notes/direction-scores" in skill
    assert "POST /api/equities/{symbol}/research-notes" in skill
    assert "X-Admin-Key" in skill
    assert "nEffective >= 30" in skill and "insufficient-sample" in skill
    assert "superseded-not-scored" in skill


def test_note_example_has_null_probability_and_versions(skill, note_example):
    assert "`body.probability` is always `null`" in skill
    assert "probability" in note_example["body"] and note_example["body"]["probability"] is None
    assert note_example["skillVersion"] == "equity-technical-v1"
    assert note_example["methodVersion"] == "equity-technical-method-v1"
    assert note_example["mode"] == "forward"
    assert note_example["actionBasis"] == "market-only"
    assert note_example["direction"] in {"up", "down", "sideways", "no-view"}
    assert set(note_example["body"]) == NOTE_BODY_KEYS
    assert note_example["body"]["schemaVersion"] == "equity-research-note-v1"
    assert isinstance(note_example["requestedAsOf"], str)
    assert re.fullmatch(r"[0-9a-f]{64}", note_example["snapshotSha256"])


def test_uses_the_technical_payload_field_names(skill):
    for field in (
        "timeframes[]", "candles", "indicators.methods", "candlePatterns", "conditions", "dataQuality",
        "freshness", "baseRates", "relativeStrength", "calendar.forwardNoteOpen",
        "calendar.nextSessionOpenUtcMs", "noteBinding.snapshotId", "noteBinding.snapshotSha256",
        "noteBinding.requestedAsOf", "noteBinding.signalBarOpenTimeMs", "scoring.ruleVersion", "limitations",
    ):
        assert f"`{field}`" in skill, field


def test_base_rates_are_labelled_unvalidated(skill):
    assert "`baseRates.status`" in skill
    assert "`abstained`" in skill and "`gate-not-passed`" in skill
    assert "**unvalidated**" in skill


def test_only_technical_endpoints_are_named(skill):
    paths = {p.rstrip(".,);:`") for p in re.findall(r"/api/[A-Za-z0-9_{}./-]+", skill)}
    assert paths, "the skill must name its endpoints"
    assert paths <= ALLOWED_API_PATHS, paths - ALLOWED_API_PATHS
    for segment in FORBIDDEN_ROUTE_SEGMENTS:
        assert not re.search(rf"/{re.escape(segment)}\b", skill), segment
    assert "research_snapshot.py export" not in skill
    assert "--include-portfolio" not in skill


def test_reuses_investment_research_references(skill):
    refs = set(re.findall(r"`(\.\./investment-research/[^`]+\.md)`", skill))
    assert "../investment-research/references/journal-note.md" in refs
    for ref in refs:
        assert (SKILL_DIR / ref).resolve().is_file(), ref
