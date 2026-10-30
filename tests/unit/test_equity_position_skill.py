"""Skill ``equity-position``.

Pins the chat skill against the code it describes: it may only name the
position/cash/decision endpoints, the costs it quotes must equal
``vn_execution_model`` (imported, never re-typed), and - when the backend repo
sits next to ``ai/`` as in the btc workspace - every reason code and field name
it quotes must exist in the backend source. Reads files only: no DB, no network.
"""

import re
import sys
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AI_ROOT))

import vn_execution_model as vem

SKILL_DIR = AI_ROOT / "skills" / "equity-position"
SKILL_PATH = SKILL_DIR / "SKILL.md"
BACKEND = AI_ROOT.parent / "backend"
DECISION_SERVICE = BACKEND / "Services" / "MbbDecisionService.cs"
POSITION_MODELS = BACKEND / "Services" / "Models" / "MbbPositionModels.cs"

ALLOWED_API_PATHS = {
    "/api/equities/{symbol}/position",
    "/api/equities/cash",
    "/api/equities/{symbol}/decision",
}
FORBIDDEN_ROUTE_SEGMENTS = (
    "dossier", "context", "fundamentals", "financial-facts", "source-documents", "documents",
    "news", "events", "llm-runs", "research-snapshot", "research-notes", "technical", "candles",
)

needs_backend = pytest.mark.skipif(
    not (DECISION_SERVICE.is_file() and POSITION_MODELS.is_file()),
    reason="backend repo not next to ai/ (separate-repo CI)")


def _pct(fraction: float) -> str:
    return f"{fraction * 100:g}%"


@pytest.fixture(scope="module")
def skill() -> str:
    assert SKILL_PATH.is_file(), f"missing skill file {SKILL_PATH}"
    return SKILL_PATH.read_text(encoding="utf-8")


def test_front_matter_names_the_skill(skill):
    m = re.match(r"---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", skill, flags=re.S)
    assert m, "SKILL.md must start with a --- front matter block"
    fields = dict(re.findall(r"^(name|description):[ \t]*([^\r\n]+)", m.group(1), flags=re.M))
    assert fields.get("name", "").strip() == "equity-position"
    description = fields.get("description", "")
    assert "MBB" in description and "BID" in description
    assert "/api/equities/{symbol}/decision" in description
    assert len(description) <= 1024


def test_only_position_endpoints_are_named(skill):
    paths = {p.rstrip(".,);:`") for p in re.findall(r"/api/[A-Za-z0-9_{}./-]+", skill)}
    assert paths == ALLOWED_API_PATHS, paths ^ ALLOWED_API_PATHS
    for segment in FORBIDDEN_ROUTE_SEGMENTS:
        assert not re.search(rf"/api/[^\s`]*/{re.escape(segment)}\b", skill), segment


def test_states_the_write_and_safety_rules(skill):
    # The position PUT deletes every lot and overwrites the budget: the skill must say so.
    assert "**The position PUT replaces everything.**" in skill
    assert "**clears**" in skill
    assert "show the user the exact lot list" in skill
    assert "X-Admin-Key" in skill and "never print" in skill
    assert "never places an order" in skill.lower()
    assert "**No probabilities, no forecasts.**" in skill
    assert "never pre-adjust a lot" in skill
    assert "never copy `availableCashVnd` into a" in skill
    assert 'skillVersion: "equity-position-v3"' in skill


def test_quotes_the_backend_band_and_levels_instead_of_recomputing(skill):
    # Since backend rules declared-rules-v4 (EQDEC-1) the decision returns the next-session band
    # (around the signal close) and both cost triggers on every branch: the skill must not redo them.
    flat = " ".join(skill.split())  # phrases may wrap across lines
    assert "**The band in the decision is the next session's band.**" in flat
    assert "not the next session's band" not in flat
    assert "`market.previousDailyCloseVnd`" in flat and "`market.lastDailyCloseVnd`" in flat
    assert "`declared-rules-v4`" in flat and "data-limitations.md` item 12" in flat
    # The band is only the next session's while the 1d bar is fresh (cold-check EQDEC-1 MINOR).
    assert "while the 1d entry of `dataFreshness[]` has `isStale: false`" in flat
    assert "do not recompute" in flat and "do not compute levels yourself" in flat
    for recompute in ("x 1.07", "x 0.93", "x 0.92", "x 0.96"):
        assert recompute not in flat, recompute
    limitations = (SKILL_DIR / "../investment-research/references/data-limitations.md").resolve()
    item12 = re.search(r"^12\. (.*?)(?=^\d+\. )", limitations.read_text(encoding="utf-8"), flags=re.M | re.S)
    assert item12 and "fixed in backend rules `declared-rules-v4`" in item12.group(1)
    assert "known backend issue" not in item12.group(1)


def test_quoted_costs_equal_the_execution_model(skill):
    spec = vem.VnEquityCostSpec()
    assert f"Broker fee {_pct(spec.broker_fee_fraction)} per side" in skill
    assert f"sell tax {_pct(spec.sell_tax_fraction)}" in skill
    assert f"lot {spec.lot_size}" in skill
    assert f"T+{spec.settlement_trading_days}" in skill
    assert f"band ±{_pct(spec.price_band_fraction)}" in skill


def test_reuses_investment_research_references(skill):
    refs = set(re.findall(r"`(\.\./investment-research/[^`]+\.md)`", skill))
    assert "../investment-research/references/cli.md" in refs
    for ref in refs:
        assert (SKILL_DIR / ref).resolve().is_file(), ref


@needs_backend
def test_reason_codes_exist_in_the_decision_service(skill):
    source = DECISION_SERVICE.read_text(encoding="utf-8")
    table = re.findall(r"^[ \t]*\|([^|\n]+)\|[^|\n]+\|[ \t]*$", skill, flags=re.M)
    # Prefixed codes are written `code:<placeholder>` and appear in the source as "code:...".
    codes = {c for cell in table for c in re.findall(r"`([a-z][a-z-]+)(?::<[a-z]+>)?`", cell)}
    codes -= {"reason", "down", "up"}
    assert len(codes) >= 20, codes
    missing = {c for c in codes if f'"{c}"' not in source and f'"{c}:' not in source}
    assert not missing, missing


@needs_backend
def test_stop_and_reduce_levels_equal_the_declared_rules(skill):
    source = DECISION_SERVICE.read_text(encoding="utf-8")
    stop = re.search(r"StopLossFraction = (0\.\d+)m", source).group(1)
    reduce_ = re.search(r"ReduceLossFraction = (0\.\d+)m", source).group(1)
    assert f"-{_pct(float(stop))} stop" in skill and f"-{_pct(float(reduce_))} reduce" in skill
    assert f"avg cost -{_pct(float(stop))}" in skill and f"avg cost -{_pct(float(reduce_))}" in skill


@needs_backend
def test_field_names_exist_in_the_dtos(skill):
    models = POSITION_MODELS.read_text(encoding="utf-8")
    # Record parameters (optionally with "= default") and init properties.
    declared = {m[0].lower() + m[1:] for m in re.findall(
        r"\b([A-Z][A-Za-z0-9]+)(?=\s*(?:=\s*[^,)\n]+)?[,)]|\s*\{\s*get;)", models)}
    for field in (
        "lots", "quantity", "costPerShareVnd", "acquireDate", "settleDate", "settled", "totalShares",
        "settledShares", "pendingShares", "avgCostPerShareVnd", "sharedCashVnd", "buyBudgetVnd",
        "cashSource", "cashNotes", "action", "reasons", "triggers", "invalidation", "buyMore", "market",
        "cash", "dataFreshness", "evidenceStatus", "priceBasis", "generatedAtUtc", "sellableSharesNow",
        "lastPriceVnd", "unrealizedPnlVnd", "unrealizedPnlPct", "netProceedsIfSellableNowVnd",
        "costBasisComparable", "costBasisNote", "corporateActions", "spendableVnd", "maxBuyableShares",
        "maxBuyableLots", "estimatedCostVnd", "ceilingPriceVnd", "floorPriceVnd", "priceLevelVnd",
        "withinPriceBand", "nonFillRisk", "isStale", "availableCashVnd", "previousDailyCloseVnd",
        "lastDailyCloseVnd", "totalDeclaredBudgetsVnd", "updatedAtUtc",
    ):
        assert f"`{field}`" in skill or f".{field}`" in skill or f"`{field}[]`" in skill, f"skill no longer names {field}"
        assert field in declared, f"{field} is not a DTO property"
