"""Skill ``equity-technical`` (EQTECH-1 lane K).

Pins the chat skill against the code it describes: the direction-score-v1
thresholds and horizons must equal the barrier configs of ``equity_base_rates``
(imported, never re-typed), the note it records must carry no probability, and
it may only point the session at the technical/candles/research-notes endpoints.
Since v7 (VN30-WEB) the forward thesis is published verbatim on the public site
behind the keyword filter of ``export_vn30_web``: the Step 4 writing rule is
pinned, and its own examples are parsed back out of it and run through that
filter in every natural position of a sentence.
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
import trading_config

SKILL_DIR = AI_ROOT / "skills" / "equity-technical"
SKILL_PATH = SKILL_DIR / "SKILL.md"
SKILL_VERSION = "equity-technical-v7"

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
    assert "MBB" in description and "BID" in description and "30 VN30 constituents" in description
    assert "/api/equities/{symbol}/technical" in description


def _symbols(line: str) -> set[str]:
    return set(re.findall(r"`([A-Z]{3,})`", line))


def test_symbol_scope_equals_the_technical_analysis_registry(skill):
    # VN30-TA: the skill's symbol list is the registry's TA equities (imported, never re-typed),
    # split by the decision lane that only MBB/BID carry.
    scope = re.search(r"- \*\*Symbols:\*\*(.*?)\r?\n- \*\*Endpoints", skill, flags=re.S)
    assert scope, "Scope must list the symbols before the endpoints"
    decision_line, rest = scope.group(1).split("plus the decision lane", 1)
    others = rest.split("1d only, no decision lane", 1)[0]
    equities = {s: d for s, d in trading_config.INSTRUMENTS.items() if d["asset_class"] == "equity"}
    assert all(d["capabilities"]["technical_analysis"] for d in equities.values())
    decision = {s for s, d in equities.items() if d["capabilities"]["decision"]}
    assert _symbols(decision_line) == decision == {"MBB", "BID"}
    assert _symbols(others) == set(equities) - decision
    assert len(equities) == 30 and "(the other 28)" in others
    assert all(equities[s]["timeframes"] == ("1d",) for s in set(equities) - decision)
    banks = re.search(r"The banks among the 30 are (.*?)\.", " ".join(skill.split()))
    # the relative-strength advice hinges on this list, so pin it exactly (TPB left VN30 on 2026-08-03)
    assert banks and _symbols(banks.group(1)) == {"MBB", "BID", "VCB", "CTG", "ACB", "HDB", "LPB", "SHB", "SSB",
                                                  "STB", "TCB", "VIB", "VPB"}


def test_non_decision_symbols_work_without_base_rates_or_a_bundle(skill):
    flat = " ".join(skill.split())
    assert "`baseRatesUnavailableReason` `no-decision-capability`" in flat
    assert '`"no base rates: no decision lane for this symbol (no-decision-capability)"`' in flat
    assert "Only MBB and BID have verified per-symbol bundles." in flat
    assert "the symbol has no historical statistics" in flat and "never borrow numbers from MBB, BID or BTCUSDT" in flat
    assert "For a non-bank name, weigh VNINDEX" in flat
    assert "Both tests used MBB and BID only. For the other 28 symbols no blind test exists" in flat
    assert "`equity-position` skill (MBB and BID only;" in flat


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
    assert f'skillVersion: "{SKILL_VERSION}"' in skill
    # no stale version left anywhere in the file (the Versions history names old ones as "v6 (", never quoted)
    assert set(re.findall(r'"equity-technical-v(\d+)"', skill)) == {"7"}
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
    assert note_example["skillVersion"] == SKILL_VERSION
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


NO_EDGE_SENTENCE = ("no edge shown: blind tests 2026-10-30/31 scored 28/74 vs 34/74 for always sideways (H20), "
                    "on MBB/BID only (none for other symbols); view is for reference only")


def test_states_no_edge_from_the_blind_tests(skill, note_example):
    # SKILLS-1 (#12) and the 2026-10-31 anonymised history test: no edge shown, so every view and note says so.
    flat = " ".join(skill.split())
    assert "**No edge yet: the view is for reference only.**" in flat
    assert "**5/14** right, against **7/14** for always answering `sideways`, 7/14 for the prior and 4/14 for momentum" in flat
    assert "**23/60** right, against **27/60** for always answering `sideways`, 24/60 for the prior and 21/60 for momentum" in flat
    assert "Together: **28/74** against **34/74** for always `sideways`, about 66-68 independent cases" in flat
    assert "their 20-session windows overlap (about 3-4 independent observations per symbol" in flat
    assert "forward-dated" not in flat  # the 2026-10-30 test was historical cases after the cutoff, not a forward journal
    assert f"`\"{NO_EDGE_SENTENCE}\"`" in flat
    assert NO_EDGE_SENTENCE in note_example["body"]["limitations"]
    assert "blind test 2026-10-30 scored 5/14" not in flat  # the old single-test sentence is gone
    assert "`nEffective >= 30`" in flat and "`baselinePrior`" in flat


@pytest.mark.parametrize("folder", ["blind-test-20261030", "blind-test-hist-20261031"])
def test_blind_test_numbers_match_the_scored_cases(skill, folder):
    score = AI_ROOT.parent / "evidence" / folder / "score.json"
    if not score.is_file():
        pytest.skip("blind-test evidence lives outside the ai repo (btc workspace only)")
    rows = json.loads(score.read_text(encoding="utf-8"))
    n = len(rows)
    hits = {key: sum(r[key] == r["realized"] for r in rows) for key in ("prediction", "prior", "momentum")}
    sideways = sum(r["realized"] == "sideways" for r in rows)
    flat = " ".join(skill.split())
    assert (f"**{hits['prediction']}/{n}** right, against **{sideways}/{n}** for always answering `sideways`, "
            f"{hits['prior']}/{n} for the prior and {hits['momentum']}/{n} for momentum") in flat


def test_combined_blind_test_total_matches_both_score_files(skill):
    rows = []
    for folder in ("blind-test-20261030", "blind-test-hist-20261031"):
        score = AI_ROOT.parent / "evidence" / folder / "score.json"
        if not score.is_file():
            pytest.skip("blind-test evidence lives outside the ai repo (btc workspace only)")
        rows += json.loads(score.read_text(encoding="utf-8"))
    hits = sum(r["prediction"] == r["realized"] for r in rows)
    sideways = sum(r["realized"] == "sideways" for r in rows)
    assert f"scored {hits}/{len(rows)} vs {sideways}/{len(rows)} for always sideways" in NO_EDGE_SENTENCE
    assert f"Together: **{hits}/{len(rows)}** against **{sideways}/{len(rows)}**" in " ".join(skill.split())


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


# --- v7 (VN30-WEB): the forward thesis is published verbatim, behind a fail-closed keyword filter ---

# Contract VN30-WEB r3 E1: the canonical action-word list of the thesis filter (export_vn30_web.py), as the skill
# names it; "in any form" stands for the inflections the filter withholds too. "held", a bare "stopped" and
# "accumulation" are not on the list (the filter lets them through), so the skill may allow them.
THESIS_ACTION_WORDS = (
    "buy", "sell", "hold", "holder", "reduce", "add", "accumulate", "trim", "enter", "entry", "exit", "wait",
    "stop", "stop-loss", "stopped out", "take profit", "profit-taking", "cut loss", "stand aside", "sideline",
    "new money", "limit or market order", "scale in or out", "average down", "go long or short", "shorting",
    "short-sell", "recommend",
)
# Every "quoted" or `code` item of Step 4's `thesis` bullet is either one of THESIS_RULE_NAMES or one of the rule's
# examples, sorted into the three tuples below. The test parses them back out of the rule both ways, so an example
# added later fails until it is sorted here, and then meets the filter.
# - THESIS_PUBLISHABLE: what the rule calls fine or tells the writer to write. The filter must publish each one in
#   every position of THESIS_POSITIONS.
THESIS_PUBLISHABLE = (
    "rose 1.2%", "moved into oversold", "the decline", "fell 4%", "for reference only", "selling pressure",
    "buying pressure", "accumulation/distribution", "support held", "price stopped at the low", "20 sessions later",
    "range position 0.45", "volume 2.1x its 20-session average", "20- and 60-session", "ex-date adjustment",
    "SMA50 19877", "19600. The close", "SMA200", "closed 18750. The stock", "2026-11-02", "not yet confirmed",
    "is flat", "measured-move level",
)
# - THESIS_WITHHELD: what the rule tells the writer not to write because the filter withholds it; the filter must
#   withhold each one in every position. Since r3 this includes phrasing the filter withholds beyond the word lists
#   ("<number> <ticker>" also across a full stop, a 13-digit epoch, awaiting, stays flat, price target), so a filter
#   change that publishes one of them turns this test red and the rule is brought back in line.
THESIS_WITHHELD = (
    "not a recommendation", "close(T+20)", "20-session/60-session", "cash dividend", "conditions.payload",
    "contract-definitions-sha-mismatch", "snapshot", "1792281600000", "sold off", "SMA50:19877", "19600.The close",
    "200 DMA", "closed 18750. MBB", "awaiting confirmation", "stays flat", "price target", "you",
)
# - THESIS_AVOIDED: what the rule tells the writer not to write for a reason other than the filter withholding it.
#   The skill may be stricter than the filter, never looser, so these meet no filter check: a word glued to "_" or a
#   digit and an event name (the filter is not built to read them, contract r3 E4), the placeholder "T+n", and
#   "sell-off" (a form of "sell"; the r3 filter lets it through as a noun after a word, "the sell-off", and withholds
#   it after ". ", ", ", "; " or ": ").
THESIS_AVOIDED = ("RSI_ENTER_OVERSOLD", "buy18600", "T+n", "sell-off")
# Quoted or `code` items of the rule that are names, not phrases: note fields and the character "_".
THESIS_RULE_NAMES = ("body", "actionView", "evidence", "limitations", "_")
THESIS_CARRIER = "MBB 1d closed near SMA50 ({}). View: sideways over 20 sessions, for reference only."
# Where a phrase can stand in a thesis, as (template, capitalised): inside "( ... )" (THESIS_CARRIER), first word of
# the thesis, after ". ", ", " and "; ", after a word, and after ": " at the very end. The filter reads what stands
# around a word (review r3: the r3 filter publishes "(sell-off" and withholds "Sell-off volume faded"), so a phrase
# is fine only when it is fine in all of them.
THESIS_POSITIONS = (
    ("MBB 1d closed near SMA50 ({}). View: sideways over 20 sessions, for reference only.", False),
    ("{} on the last bar. View: sideways over 20 sessions, for reference only.", True),
    ("MBB 1d closed near SMA50. {} on the last bar.", True),
    ("MBB 1d closed near SMA50, {} on the last bar.", False),
    ("MBB 1d closed near SMA50; {} on the last bar.", False),
    ("MBB 1d closed near SMA50 with {} on the last bar.", False),
    ("MBB 1d, last bar: {}", False),
)
_QUOTED = r'(?:"[^"]+"|`[^`]+`)'


def _section(skill: str, heading: str) -> str:
    m = re.search(rf"^## {re.escape(heading)}[ \t]*\n(.*?)(?=^## |\Z)", skill, flags=re.S | re.M)
    assert m, f"missing section ## {heading}"
    return m.group(1)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _thesis_rule(skill: str) -> tuple[str, list[str]]:
    """Step 4's `thesis` bullet: its lead text and its sub-bullets, each flattened."""
    step4 = _section(skill, "Step 4: Record the note (once)")
    m = re.search(r"^  - `thesis`:(.*?)(?=^  - `evidence\[\]`)", step4, flags=re.S | re.M)
    assert m, "Step 4 must keep the `thesis` bullet right before `evidence[]`"
    lead, *items = re.split(r"^    - ", m.group(1), flags=re.M)
    return _flat(lead), [_flat(item) for item in items]


def _rule_item(items: list[str], start: str) -> str:
    found = [item for item in items if item.startswith(start)]
    assert len(found) == 1, f"the thesis rule needs one sub-bullet starting {start!r}"
    return found[0]


def _names(text: str, word: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text) is not None


def _items(text: str) -> list[str]:
    """The "quoted" and `code` items of a text, in order."""
    return [quoted or code for quoted, code in re.findall(r'"([^"]+)"|`([^`]+)`', text)]


def _thesis_examples(rule: str) -> tuple[set[str], set[str]]:
    """The rule's examples by what the rule says of them: (fine, not). Fine: the "These are fine:" list, the
    "Describe the move instead (...)" list and every X of 'X, not Y'; not: every Y. X is one or more items joined by
    ", " or " or ", Y one or more joined by " or "."""
    fine, avoid = [], []
    for x, y in re.findall(rf"({_QUOTED}(?:(?:, | or ){_QUOTED})*), not ({_QUOTED}(?: or {_QUOTED})*)", rule):
        fine += _items(x)
        avoid += _items(y)
    listed = re.search(rf"These are fine: ({_QUOTED}(?:(?:, | or | and ){_QUOTED})*)\.", rule)
    moves = re.search(rf"Describe the move instead \(({_QUOTED}(?:, {_QUOTED})*)\)", rule)
    assert listed and moves, "the rule must keep its fine list and its describe-the-move examples"
    return set(fine + _items(listed.group(1)) + _items(moves.group(1))), set(avoid)


def _in_position(position: tuple[str, bool], phrase: str) -> str:
    template, capitalised = position
    return template.format(phrase[:1].upper() + phrase[1:] if capitalised else phrase)


def test_forward_thesis_is_written_for_the_public_site(skill):
    # The exporter publishes the forward thesis verbatim (ai/export_vn30_web.py); the writer is the cheap
    # lever against both a leak and a thesis withheld for a technical sentence.
    lead, items = _thesis_rule(skill)
    assert lead.startswith("non-empty string, at most 4000 characters; the whole `body` at most 64 KiB.")
    assert ("The thesis of a **forward** note is published verbatim on the public VN30 website after a "
            "fail-closed keyword filter, and a thesis the filter withholds is not shown at all (one flagged word "
            "hides the whole text);") in lead
    # the filter also withholds a thesis with Greek, Cyrillic or modifier letters (contract VN30-WEB r3 E4)
    assert "So write it in English (no Greek letters such as σ)" in lead
    assert "describe only the candles and indicators:" in lead
    assert len(items) == 6


def test_thesis_length_rule_matches_the_export_cut(skill):
    # Review r3: "published verbatim" holds only up to export_vn30_web.THESIS_MAX characters; a longer thesis is cut
    # on the site (its closing view and "for reference only" can go), so the rule names the same number.
    import export_vn30_web as web

    lead, _ = _thesis_rule(skill)
    assert f"the site cuts a thesis longer than {web.THESIS_MAX} characters" in lead
    assert f"keep it within {web.THESIS_MAX} characters" in lead
    text = (THESIS_CARRIER.format("RSI14 45") + " ") * (2 * web.THESIS_MAX // len(THESIS_CARRIER))
    whole, longer = text[:web.THESIS_MAX], text[:web.THESIS_MAX + 1]
    assert web.public_thesis(whole) == (whole, False)
    shown, withheld = web.public_thesis(longer)
    assert not withheld and len(shown) == web.THESIS_MAX and shown != longer


def test_thesis_rule_bans_every_filtered_action_word_and_keeps_the_action_in_action_view(skill):
    _, items = _thesis_rule(skill)
    actions = _rule_item(items, "no action words")
    assert actions.startswith("no action words, even in their market sense and in any form (buys, buying, bought): ")
    named = [word.strip() for word in actions.split(": ", 1)[1].split(".", 1)[0].split(",")]
    assert len(named) == len(set(named)), named
    assert set(named) == set(THESIS_ACTION_WORDS), set(named) ^ set(THESIS_ACTION_WORDS)
    assert '"for reference only", not "not a recommendation"' in actions
    # Review r3: the r3 filter withholds "sell-off" at the start of a sentence or after ", ", "; " or ": ", so the
    # rule offers "the decline" instead and never calls it fine.
    assert 'write "the decline" or "fell 4%", not "sell-off" or "sold off"' in actions
    allowed = re.search(r"These are fine: (.*?)\. Gluing", actions)
    assert allowed and _items(allowed.group(1)) == [
        "selling pressure", "buying pressure", "accumulation/distribution", "support held", "price stopped at the low"]
    # The filter does not read a word glued to "_" or a digit (contract r3 E4, export_vn30_web docstring: "the writer
    # is told not to write them"); the conditions' event names glue listed words that way (RSI_ENTER_OVERSOLD).
    assert 'Gluing a word to "_" or a digit does not make it fine ("buy18600").' in actions
    assert "not by an event name (`RSI_ENTER_OVERSOLD` belongs in `evidence`)" in actions
    contract = json.loads((AI_ROOT / "contracts" / "technical-module-contract.json").read_text(encoding="utf-8"))
    declared = {name for module in contract["modules"].values() for name in module["parameters"]["eventTypes"]}
    assert "RSI_ENTER_OVERSOLD" in declared
    assert "The action goes only in `actionView` (Step 3), never in the thesis" in actions
    # actionView keeps its own default: the thesis rule does not change Step 3.
    assert ("Use `market-only` with `wait` by default, or `hold` only when the user says they already hold the "
            "stock") in _flat(skill)


def test_thesis_rule_bans_holdings_money_and_internal_identifiers(skill):
    _, items = _thesis_rule(skill)
    holdings = _rule_item(items, "no position")
    for word in ("position", "number of shares", "price paid", "cost", "money amount", "budget", "NAV", '"T+n"'):
        assert _names(holdings, word), word
    identifiers = _rule_item(items, "no URL")
    for word in ("URL", "host name", "file path", "dotted field path", "id", "epoch timestamp",
                 "code of 32 or more characters", 'the word "snapshot"'):
        assert _names(identifiers, word), word
    # review r3: the binding's signalBarOpenTimeMs is the epoch a writer could copy; the filter reads it as a digit run
    assert 'epoch timestamp ("2026-11-02", not "1792281600000")' in identifiers
    assert "cite field paths and codes in `evidence` and `limitations` instead" in identifiers
    phrasing = _rule_item(items, "volume as a multiple")
    assert '"volume 2.1x its 20-session average"' in phrasing and "never a share count" in phrasing
    assert '"20- and 60-session", not "20-session/60-session"' in phrasing
    assert '"ex-date adjustment", not "cash dividend"' in phrasing


def test_thesis_rule_warns_about_numbers_and_words_beyond_its_lists(skill):
    # Review r3: the filter also withholds plain chart phrasing the lists never named: "SMA50:19877" reads as
    # host:port, "19600.The" as a host name, a number right before three capitals as "<number> <ticker>" (r3, even
    # across a full stop), and words such as awaiting, stays flat, price target or you (whether the filter really
    # withholds each of them is checked in test_thesis_rule_examples_agree_with_the_export_filter).
    _, items = _thesis_rule(skill)
    numbers = _rule_item(items, "numbers:")
    for text in ('"SMA50 19877", not "SMA50:19877"', '("19600. The close", not "19600.The close")',
                 '("SMA200", not "200 DMA"; "closed 18750. The stock", not "closed 18750. MBB")',
                 "even across a full stop", "the filter can read it as a share count"):
        assert text in numbers, text
    beyond = _rule_item(items, "the filter withholds more words")
    assert beyond.startswith("the filter withholds more words than these lists, so a purely technical thesis can be "
                             "hidden too: ")
    for text in ('"not yet confirmed", not "awaiting confirmation"', '"is flat", not "stays flat"',
                 '"measured-move level", not "price target"', 'never address the reader ("you")'):
        assert text in beyond, text


def _shown(web, text: str) -> bool:
    return web.public_thesis(text) == (text, False)


def test_thesis_rule_examples_agree_with_the_export_filter(skill, note_example):
    # Producer -> consumer (AGENTS.md #10): every example of the rule, parsed back out of it, goes through the real
    # filter in every position of THESIS_POSITIONS. The skill never calls fine what the site withholds wherever it
    # stands in a sentence, and what it says the site withholds is withheld wherever it stands.
    import export_vn30_web as web

    lead, items = _thesis_rule(skill)
    rule = " ".join([lead, *items])
    examples = THESIS_PUBLISHABLE + THESIS_WITHHELD + THESIS_AVOIDED
    assert len(examples) == len(set(examples)), "an example is sorted twice"
    assert set(_items(rule)) - set(THESIS_RULE_NAMES) == set(examples), set(_items(rule)) ^ set(examples)
    fine, avoid = _thesis_examples(rule)
    assert fine <= set(THESIS_PUBLISHABLE), fine - set(THESIS_PUBLISHABLE)
    assert avoid <= set(THESIS_WITHHELD + THESIS_AVOIDED), avoid - set(THESIS_WITHHELD + THESIS_AVOIDED)
    assert THESIS_POSITIONS[0] == (THESIS_CARRIER, False)
    # no position withholds a neutral phrase by itself (otherwise the withheld check below would pass for nothing)
    assert [p for p in THESIS_POSITIONS if not _shown(web, _in_position(p, "RSI14 45"))] == []
    lost = [_in_position(p, phrase) for phrase in THESIS_PUBLISHABLE for p in THESIS_POSITIONS
            if not _shown(web, _in_position(p, phrase))]
    assert lost == [], lost
    published = [_in_position(p, phrase) for phrase in THESIS_WITHHELD for p in THESIS_POSITIONS
                 if web.public_thesis(_in_position(p, phrase)) != (None, True)]
    assert published == [], published
    example = note_example["body"]["thesis"]  # a forward note: this is what the site would show
    assert note_example["mode"] == "forward" and _shown(web, example)


def test_skill_version_is_v7_in_all_three_places(skill, note_example):
    step4 = _flat(_section(skill, "Step 4: Record the note (once)"))
    assert f'plus `skillVersion: "{SKILL_VERSION}"` and `methodVersion: "equity-technical-method-v1"`' in step4
    assert note_example["skillVersion"] == SKILL_VERSION
    versions = _flat(_section(skill, "Versions"))
    assert versions.startswith(f'`skillVersion: "{SKILL_VERSION}"`, `methodVersion: "equity-technical-method-v1"`.')
    v7 = re.search(r"\bv7 \(2026-11-03, VN30-WEB\): (.*)$", versions)
    assert v7, "Versions must log v7 (2026-11-03, VN30-WEB)"
    assert "the thesis of a forward note is published verbatim on the public VN30 website" in v7.group(1)
    assert set(re.findall(r'"equity-technical-v(\d+)"', skill)) == {"7"}
