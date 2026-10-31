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
    assert 'skillVersion: "equity-technical-v4"' in skill
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
    assert note_example["skillVersion"] == "equity-technical-v4"
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
    # Runnable examples name a real symbol (/api/equities/MBB/...); compare them as {symbol}.
    paths = {re.sub(r"^/api/equities/(MBB|BID)/", "/api/equities/{symbol}/", p.rstrip(".,);:`\""))
             for p in re.findall(r"/api/[A-Za-z0-9_{}./-]+", skill)}
    assert paths, "the skill must name its endpoints"
    assert paths <= ALLOWED_API_PATHS, paths - ALLOWED_API_PATHS
    for segment in FORBIDDEN_ROUTE_SEGMENTS:
        assert not re.search(rf"/{re.escape(segment)}\b", skill), segment
    assert "research_snapshot.py export" not in skill
    assert "--include-portfolio" not in skill


def test_states_no_edge_from_the_blind_test(skill, note_example):
    # SKILLS-1 (#12): the 2026-10-30 blind test showed no edge, so every view and note says so.
    flat = " ".join(skill.split())
    assert "**No edge yet: the view is for reference only.**" in flat
    assert "**5/14** right, against **7/14** for always answering `sideways`, 7/14 for the prior and 4/14 for momentum" in flat
    sentence = "no edge shown: blind test 2026-10-30 scored 5/14 vs 7/14 for always sideways (H20); view is for reference only"
    assert f"`\"{sentence}\"`" in flat
    assert sentence in note_example["body"]["limitations"]
    assert "`nEffective >= 30`" in flat and "`baselinePrior`" in flat


def test_blind_test_numbers_match_the_scored_cases(skill):
    score = AI_ROOT.parent / "evidence" / "blind-test-20261030" / "score.json"
    if not score.is_file():
        pytest.skip("blind-test evidence lives outside the ai repo (btc workspace only)")
    rows = json.loads(score.read_text(encoding="utf-8"))
    n = len(rows)
    hits = {key: sum(r[key] == r["realized"] for r in rows) for key in ("prediction", "prior", "momentum")}
    sideways = sum(r["realized"] == "sideways" for r in rows)
    flat = " ".join(skill.split())
    assert (f"**{hits['prediction']}/{n}** right, against **{sideways}/{n}** for always answering `sideways`, "
            f"{hits['prior']}/{n} for the prior and {hits['momentum']}/{n} for momentum") in flat
    assert f"scored {hits['prediction']}/{n} vs {sideways}/{n} for always sideways" in flat


CONDITIONS_SERVICE = AI_ROOT.parent / "backend" / "Services" / "CurrentConditionsService.cs"


def test_condition_statistics_are_read_as_descriptive_history(skill, note_example):
    # EQEVID-1: per-symbol bundles now attach statistics to MBB/BID conditions; the skill
    # must keep them descriptive (no probability, no expected return) and gated on sample size.
    flat = " ".join(skill.split())
    assert "**Condition statistics are descriptive history, not a forecast.**" in flat
    assert "Never turn a cell into a probability or an expected return." in flat
    assert "They are not the 20/60 sessions of the call" in flat
    sentence = ("condition statistics are descriptive history (spec declared on BTCUSDT, "
                "run unchanged on HOSE bars), not a forecast")
    assert f"`\"{sentence}\"`" in flat
    assert "Only a cell with `passesDeclaredFdr: true` may be called a difference" in flat
    # A state condition (e.g. "close below SMA50") is joined to the crossing event's history.
    assert "The cells belong to the **event type**, matched by `module` + `eventType`." in flat
    assert "not every bar spent in the state" in flat


def test_sample_gate_and_fdr_level_match_the_statistical_spec(skill):
    spec = json.loads((AI_ROOT / "contracts" / "technical-evidence-statistical-spec.json").read_text(encoding="utf-8"))
    flat = " ".join(skill.split())
    gate = spec["multipleTesting"]
    assert gate["method"] == "Benjamini-Yekutieli" and "Benjamini-Yekutieli" in flat
    assert f"at least {gate['minimumNonOverlappingPairs']} non-overlapping pairs" in flat
    assert f"q at most {gate['declaredQAlpha']:g}" in flat
    assert spec["scope"]["horizonsBars"] == [1, 3, 6]
    assert spec["scope"]["metrics"] == ["forwardReturn", "mfe", "mae"]
    assert '`evidence["1"|"3"|"6"]["forwardReturn"|"mfe"|"mae"]`' in flat
    assert spec["scope"]["symbol"] == "BTCUSDT" and "declared once on BTCUSDT" in flat


def test_conditions_candle_rule_set_is_quoted_from_the_module_contract(skill):
    contract = json.loads((AI_ROOT / "contracts" / "technical-module-contract.json").read_text(encoding="utf-8"))
    event_types = contract["modules"]["candlePatterns"]["parameters"]["eventTypes"]
    flat = " ".join(skill.split())
    listed = ", ".join(f"`{name}`" for name in event_types[:-1]) + f", `{event_types[-1]}`"
    assert f"is a different, smaller rule set: only {listed} (no Marubozu" in flat
    assert not any("MARUBOZU" in name.upper() for name in event_types)
    assert "rule-set difference, not two sources disagreeing" in flat


@pytest.mark.skipif(not CONDITIONS_SERVICE.is_file(), reason="backend repo not next to ai/ (separate-repo CI)")
def test_evidence_fields_and_reasons_exist_in_the_backend_cells(skill):
    source = CONDITIONS_SERVICE.read_text(encoding="utf-8")
    cell_keys = set(re.findall(r'\["(\w+)"\] = Scalar\(hypothesis', source)) | {"tested"}
    named = {"tested", "rawP", "adjustedQValue", "passesDeclaredFdr", "sufficientSample",
             "nonOverlappingPairs", "effect", "ciLower", "ciUpper", "meanPairedDifference"}
    assert named == cell_keys, cell_keys ^ named
    untested = set(re.findall(r'UntestedCell\(\s*"([a-z-]+)"', source))
    untested |= set(re.findall(r'\? "([a-z-]+)"\s*:\s*"([a-z-]+)"', source)[0])
    unavailable = set(re.findall(r'EvidenceContext\.Unavailable\(\s*"([a-z-]+)"', source))
    assert untested == {"evidence-unavailable", "untested", "insufficient-sample"}, untested
    assert len(unavailable) == 6, unavailable
    assert "`tested: true`" in skill and "`tested: false`" in skill
    for name in sorted((named - {"tested"}) | untested | unavailable):
        assert f"`{name}`" in skill, name
    kinds = re.search(r"ValidKinds = new\(StringComparer\.Ordinal\)\s*\{([^}]*)\}", source).group(1)
    assert set(re.findall(r'"(\w+)"', kinds)) == {"triggeredOnBar", "state", "activeZone", "operativeLeg"}
    contract = json.loads((AI_ROOT / "contracts" / "technical-module-contract.json").read_text(encoding="utf-8"))
    declared = {name for module in contract["modules"].values() for name in module["parameters"]["eventTypes"]}
    assert {"RSI_ZONE_NEUTRAL", "EMA_FAST_BELOW_SLOW"}.isdisjoint(declared)
    assert "CLOSE_BELOW_SMA50" in declared
    assert "StatisticalHorizons = [1, 3, 6]" in source
    assert 'StatisticalMetrics = ["forwardReturn", "mfe", "mae"]' in source
    for envelope in ("available", "reason", "cutoffMs", "evidenceAgeBars"):
        assert f'["{envelope}"]' in source and f"`{envelope}`" in skill
