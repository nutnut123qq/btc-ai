from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

from forward_paper_recorder import (
    INSERT_SQL,
    RECORDER_VERSION,
    SCHEMA_SQL,
    PaperObservation,
    deterministic_decision_id,
    insert_observation,
)


def _observation() -> PaperObservation:
    close_ms = 1_799_999_999_999
    observed_ms = close_ms + 1
    return PaperObservation(
        decision_id=deterministic_decision_id("BTCUSDT", "4h", 1_799_985_600_000, close_ms),
        symbol="BTCUSDT",
        timeframe="4h",
        signal_bar_open_ms=1_799_985_600_000,
        signal_bar_close_ms=close_ms,
        observed_at_utc=datetime.fromtimestamp(observed_ms / 1000, timezone.utc),
        available_time_ms=observed_ms,
        model_version=None,
        decision="abstain",
        confidence=None,
        abstention_reason="model-unavailable",
        quote_source="binance-spot-bookTicker-bid-ask-mid",
        quote_price=Decimal("50000.125"),
        quote_received_at_utc=datetime.fromtimestamp(observed_ms / 1000, timezone.utc),
        quote_received_time_ms=observed_ms,
        config_provenance={"threshold": 0.61},
        evidence_provenance={"nextBarOpenRead": False},
    )


def test_decision_id_is_stable_per_signal_bar_and_changes_with_bar():
    first = deterministic_decision_id("BTCUSDT", "4h", 100, 200)
    retry = deterministic_decision_id("btcusdt", "4h", 100, 200)
    next_bar = deterministic_decision_id("BTCUSDT", "4h", 201, 300)

    assert first == retry
    assert first != next_bar
    assert len(first) == 64


def test_schema_is_separate_unique_and_never_inserts_synthetic_results():
    assert 'CREATE TABLE IF NOT EXISTS "PaperObservations"' in SCHEMA_SQL
    assert '"DecisionId" character varying(64) NOT NULL UNIQUE' in SCHEMA_SQL
    assert '"AvailableTimeMs" >= "SignalBarCloseTimeMs"' in SCHEMA_SQL
    assert '"CK_PaperObservations_Scope"' in SCHEMA_SQL
    assert '"CK_PaperObservations_DecisionEvidence"' in SCHEMA_SQL
    assert '"CK_PaperObservations_QuoteLineage"' in SCHEMA_SQL
    assert '"CK_PaperObservations_ProspectiveClock"' in SCHEMA_SQL
    assert '"CK_PaperObservations_FillCoherence"' in SCHEMA_SQL
    assert '"CK_PaperObservations_OutcomeCoherence"' in SCHEMA_SQL
    assert '"TR_PaperObservations_ImmutableEvidence"' in SCHEMA_SQL
    assert "PaperObservations cannot be deleted" in SCHEMA_SQL
    assert "decision evidence is immutable" in SCHEMA_SQL
    assert 'ON CONFLICT ("DecisionId") DO NOTHING' in INSERT_SQL
    assert 'NULL, NULL, NULL' in INSERT_SQL
    assert '"FillPrice"' in INSERT_SQL
    assert '"OutcomeReturn"' in INSERT_SQL


def test_insert_is_idempotent_when_database_reports_conflict():
    conn = Mock()
    cursor = Mock()
    cursor.fetchone.return_value = None

    inserted = insert_observation(conn, cursor, _observation())

    assert inserted is False
    sql, params = cursor.execute.call_args.args
    assert sql == INSERT_SQL
    assert params[2] == RECORDER_VERSION
    assert params[9] is None  # no fabricated model version
    assert params[10] == "abstain"
    conn.commit.assert_called_once()


def test_observation_rejects_lookahead_timestamp():
    observation = _observation()
    with pytest.raises(ValueError, match="cannot predate"):
        PaperObservation(
            **{
                **observation.__dict__,
                "available_time_ms": observation.signal_bar_close_ms - 1,
            }
        )


def test_observation_rejects_non_btc_scope_and_incomplete_quote_lineage():
    observation = _observation()
    with pytest.raises(ValueError, match="BTCUSDT 4h"):
        PaperObservation(**{**observation.__dict__, "symbol": "ETHUSDT"})
    with pytest.raises(ValueError, match="present together"):
        PaperObservation(**{**observation.__dict__, "quote_received_time_ms": None})


def test_directional_observation_requires_a_model_and_bounded_confidence():
    observation = _observation()
    with pytest.raises(ValueError, match="requires model"):
        PaperObservation(
            **{
                **observation.__dict__,
                "decision": "long",
                "abstention_reason": None,
                "confidence": 0.7,
            }
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PaperObservation(**{**observation.__dict__, "confidence": 1.1})
