"""Immutable-core prospective decision observations for forward paper research.

This module deliberately records observations rather than simulated trades.  A
fill, position, or outcome may only be populated by a separate timestamped
observer after that event actually happens.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, Optional


RECORDER_VERSION = "forward-paper-observation-v1"
BINANCE_SPOT_BOOK_TICKER = "https://api.binance.com/api/v3/ticker/bookTicker"

SCHEMA_SQL = r'''
CREATE TABLE IF NOT EXISTS "PaperObservations" (
    "Id" uuid PRIMARY KEY,
    "DecisionId" character varying(64) NOT NULL UNIQUE,
    "RecorderVersion" character varying(80) NOT NULL,
    "Symbol" character varying(20) NOT NULL,
    "Timeframe" character varying(10) NOT NULL,
    "SignalBarOpenTimeMs" bigint NOT NULL,
    "SignalBarCloseTimeMs" bigint NOT NULL,
    "ObservedAtUtc" timestamp with time zone NOT NULL,
    "AvailableTimeMs" bigint NOT NULL,
    "ModelVersion" text,
    "Decision" character varying(16) NOT NULL,
    "Confidence" double precision,
    "AbstentionReason" text,
    "QuoteSource" text NOT NULL,
    "QuotePrice" numeric(28, 12),
    "QuoteReceivedAtUtc" timestamp with time zone,
    "QuoteReceivedTimeMs" bigint,
    "ConfigProvenanceJson" jsonb NOT NULL,
    "EvidenceProvenanceJson" jsonb NOT NULL,
    "FillPrice" numeric(28, 12),
    "FillObservedAtUtc" timestamp with time zone,
    "FillSource" text,
    "OutcomeReturn" double precision,
    "OutcomeObservedAtUtc" timestamp with time zone,
    "OutcomeHorizon" character varying(20),
    "CreatedAtUtc" timestamp with time zone NOT NULL DEFAULT NOW(),
    CONSTRAINT "CK_PaperObservations_Decision" CHECK ("Decision" IN ('abstain', 'long', 'short')),
    CONSTRAINT "CK_PaperObservations_Chronology" CHECK ("AvailableTimeMs" >= "SignalBarCloseTimeMs"),
    CONSTRAINT "CK_PaperObservations_SignalBar" CHECK ("SignalBarOpenTimeMs" < "SignalBarCloseTimeMs"),
    CONSTRAINT "CK_PaperObservations_Scope" CHECK ("Symbol" = 'BTCUSDT' AND "Timeframe" = '4h'),
    CONSTRAINT "CK_PaperObservations_Confidence" CHECK ("Confidence" IS NULL OR ("Confidence" >= 0 AND "Confidence" <= 1)),
    CONSTRAINT "CK_PaperObservations_DecisionEvidence" CHECK (
        ("Decision" = 'abstain' AND "AbstentionReason" IS NOT NULL)
        OR ("Decision" IN ('long', 'short') AND "AbstentionReason" IS NULL AND "ModelVersion" IS NOT NULL AND "Confidence" IS NOT NULL)
    ),
    CONSTRAINT "CK_PaperObservations_QuoteLineage" CHECK (
        ("QuotePrice" IS NULL AND "QuoteReceivedAtUtc" IS NULL AND "QuoteReceivedTimeMs" IS NULL)
        OR ("QuotePrice" > 0 AND "QuoteReceivedAtUtc" IS NOT NULL AND "QuoteReceivedTimeMs" IS NOT NULL AND "AvailableTimeMs" >= "QuoteReceivedTimeMs")
    ),
    CONSTRAINT "CK_PaperObservations_ProspectiveClock" CHECK (
        "EvidenceProvenanceJson" @> '{"isProspectivePaperEvidence": true}'::jsonb
        AND "ObservedAtUtc" >= "CreatedAtUtc" - INTERVAL '10 minutes'
        AND "ObservedAtUtc" <= "CreatedAtUtc" + INTERVAL '1 minute'
        AND ABS("AvailableTimeMs" - (EXTRACT(EPOCH FROM "ObservedAtUtc") * 1000)::bigint) <= 1000
    ),
    CONSTRAINT "CK_PaperObservations_FillCoherence" CHECK (
        ("FillPrice" IS NULL AND "FillObservedAtUtc" IS NULL AND "FillSource" IS NULL)
        OR ("Decision" IN ('long', 'short') AND "FillPrice" > 0
            AND "FillObservedAtUtc" >= "ObservedAtUtc" AND NULLIF(BTRIM("FillSource"), '') IS NOT NULL)
    ),
    CONSTRAINT "CK_PaperObservations_OutcomeCoherence" CHECK (
        ("OutcomeReturn" IS NULL AND "OutcomeObservedAtUtc" IS NULL AND "OutcomeHorizon" IS NULL)
        OR ("Decision" IN ('long', 'short') AND "FillPrice" IS NOT NULL
            AND "OutcomeReturn" IS NOT NULL AND "OutcomeObservedAtUtc" >= "FillObservedAtUtc"
            AND NULLIF(BTRIM("OutcomeHorizon"), '') IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS "IX_PaperObservations_Symbol_Timeframe_SignalClose"
    ON "PaperObservations" ("Symbol", "Timeframe", "SignalBarCloseTimeMs");
DO $constraints$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'CK_PaperObservations_Chronology') THEN
        ALTER TABLE "PaperObservations" ADD CONSTRAINT "CK_PaperObservations_Chronology"
            CHECK ("AvailableTimeMs" >= "SignalBarCloseTimeMs");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'CK_PaperObservations_SignalBar') THEN
        ALTER TABLE "PaperObservations" ADD CONSTRAINT "CK_PaperObservations_SignalBar"
            CHECK ("SignalBarOpenTimeMs" < "SignalBarCloseTimeMs");
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'CK_PaperObservations_Scope') THEN
        ALTER TABLE "PaperObservations" ADD CONSTRAINT "CK_PaperObservations_Scope"
            CHECK ("Symbol" = 'BTCUSDT' AND "Timeframe" = '4h');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'CK_PaperObservations_Confidence') THEN
        ALTER TABLE "PaperObservations" ADD CONSTRAINT "CK_PaperObservations_Confidence"
            CHECK ("Confidence" IS NULL OR ("Confidence" >= 0 AND "Confidence" <= 1));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'CK_PaperObservations_DecisionEvidence') THEN
        ALTER TABLE "PaperObservations" ADD CONSTRAINT "CK_PaperObservations_DecisionEvidence"
            CHECK (("Decision" = 'abstain' AND "AbstentionReason" IS NOT NULL)
                OR ("Decision" IN ('long', 'short') AND "AbstentionReason" IS NULL AND "ModelVersion" IS NOT NULL AND "Confidence" IS NOT NULL));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'CK_PaperObservations_QuoteLineage') THEN
        ALTER TABLE "PaperObservations" ADD CONSTRAINT "CK_PaperObservations_QuoteLineage"
            CHECK (("QuotePrice" IS NULL AND "QuoteReceivedAtUtc" IS NULL AND "QuoteReceivedTimeMs" IS NULL)
                OR ("QuotePrice" > 0 AND "QuoteReceivedAtUtc" IS NOT NULL AND "QuoteReceivedTimeMs" IS NOT NULL AND "AvailableTimeMs" >= "QuoteReceivedTimeMs"));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'CK_PaperObservations_ProspectiveClock') THEN
        ALTER TABLE "PaperObservations" ADD CONSTRAINT "CK_PaperObservations_ProspectiveClock"
            CHECK ("EvidenceProvenanceJson" @> '{"isProspectivePaperEvidence": true}'::jsonb
                AND "ObservedAtUtc" >= "CreatedAtUtc" - INTERVAL '10 minutes'
                AND "ObservedAtUtc" <= "CreatedAtUtc" + INTERVAL '1 minute'
                AND ABS("AvailableTimeMs" - (EXTRACT(EPOCH FROM "ObservedAtUtc") * 1000)::bigint) <= 1000);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'CK_PaperObservations_FillCoherence') THEN
        ALTER TABLE "PaperObservations" ADD CONSTRAINT "CK_PaperObservations_FillCoherence"
            CHECK (("FillPrice" IS NULL AND "FillObservedAtUtc" IS NULL AND "FillSource" IS NULL)
                OR ("Decision" IN ('long', 'short') AND "FillPrice" > 0
                    AND "FillObservedAtUtc" >= "ObservedAtUtc" AND NULLIF(BTRIM("FillSource"), '') IS NOT NULL));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'CK_PaperObservations_OutcomeCoherence') THEN
        ALTER TABLE "PaperObservations" ADD CONSTRAINT "CK_PaperObservations_OutcomeCoherence"
            CHECK (("OutcomeReturn" IS NULL AND "OutcomeObservedAtUtc" IS NULL AND "OutcomeHorizon" IS NULL)
                OR ("Decision" IN ('long', 'short') AND "FillPrice" IS NOT NULL
                    AND "OutcomeReturn" IS NOT NULL AND "OutcomeObservedAtUtc" >= "FillObservedAtUtc"
                    AND NULLIF(BTRIM("OutcomeHorizon"), '') IS NOT NULL));
    END IF;
END
$constraints$;

CREATE OR REPLACE FUNCTION prevent_paper_observation_rewrite()
RETURNS trigger LANGUAGE plpgsql AS $guard$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'PaperObservations cannot be deleted';
    END IF;
    IF (OLD."Id", OLD."DecisionId", OLD."RecorderVersion", OLD."Symbol", OLD."Timeframe",
        OLD."SignalBarOpenTimeMs", OLD."SignalBarCloseTimeMs", OLD."ObservedAtUtc", OLD."AvailableTimeMs",
        OLD."ModelVersion", OLD."Decision", OLD."Confidence", OLD."AbstentionReason", OLD."QuoteSource",
        OLD."QuotePrice", OLD."QuoteReceivedAtUtc", OLD."QuoteReceivedTimeMs", OLD."ConfigProvenanceJson",
        OLD."EvidenceProvenanceJson", OLD."CreatedAtUtc") IS DISTINCT FROM
       (NEW."Id", NEW."DecisionId", NEW."RecorderVersion", NEW."Symbol", NEW."Timeframe",
        NEW."SignalBarOpenTimeMs", NEW."SignalBarCloseTimeMs", NEW."ObservedAtUtc", NEW."AvailableTimeMs",
        NEW."ModelVersion", NEW."Decision", NEW."Confidence", NEW."AbstentionReason", NEW."QuoteSource",
        NEW."QuotePrice", NEW."QuoteReceivedAtUtc", NEW."QuoteReceivedTimeMs", NEW."ConfigProvenanceJson",
        NEW."EvidenceProvenanceJson", NEW."CreatedAtUtc") THEN
        RAISE EXCEPTION 'PaperObservations decision evidence is immutable';
    END IF;
    IF OLD."FillPrice" IS NOT NULL AND
       (OLD."FillPrice", OLD."FillObservedAtUtc", OLD."FillSource") IS DISTINCT FROM
       (NEW."FillPrice", NEW."FillObservedAtUtc", NEW."FillSource") THEN
        RAISE EXCEPTION 'PaperObservations fill evidence is write-once';
    END IF;
    IF OLD."OutcomeReturn" IS NOT NULL AND
       (OLD."OutcomeReturn", OLD."OutcomeObservedAtUtc", OLD."OutcomeHorizon") IS DISTINCT FROM
       (NEW."OutcomeReturn", NEW."OutcomeObservedAtUtc", NEW."OutcomeHorizon") THEN
        RAISE EXCEPTION 'PaperObservations outcome evidence is write-once';
    END IF;
    RETURN NEW;
END
$guard$;
DO $trigger$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'TR_PaperObservations_ImmutableEvidence') THEN
        CREATE TRIGGER "TR_PaperObservations_ImmutableEvidence"
        BEFORE UPDATE OR DELETE ON "PaperObservations"
        FOR EACH ROW EXECUTE FUNCTION prevent_paper_observation_rewrite();
    END IF;
END
$trigger$;
'''

INSERT_SQL = r'''
INSERT INTO "PaperObservations" (
    "Id", "DecisionId", "RecorderVersion", "Symbol", "Timeframe",
    "SignalBarOpenTimeMs", "SignalBarCloseTimeMs", "ObservedAtUtc", "AvailableTimeMs",
    "ModelVersion", "Decision", "Confidence", "AbstentionReason",
    "QuoteSource", "QuotePrice", "QuoteReceivedAtUtc", "QuoteReceivedTimeMs",
    "ConfigProvenanceJson", "EvidenceProvenanceJson",
    "FillPrice", "FillObservedAtUtc", "FillSource",
    "OutcomeReturn", "OutcomeObservedAtUtc", "OutcomeHorizon"
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s::jsonb, %s::jsonb,
    NULL, NULL, NULL,
    NULL, NULL, NULL
)
ON CONFLICT ("DecisionId") DO NOTHING
RETURNING "Id";
'''


@dataclass(frozen=True)
class LiveQuote:
    source: str
    price: Decimal
    received_at_utc: datetime
    received_time_ms: int


@dataclass(frozen=True)
class PaperObservation:
    decision_id: str
    symbol: str
    timeframe: str
    signal_bar_open_ms: int
    signal_bar_close_ms: int
    observed_at_utc: datetime
    available_time_ms: int
    model_version: Optional[str]
    decision: str
    confidence: Optional[float]
    abstention_reason: Optional[str]
    quote_source: str
    quote_price: Optional[Decimal]
    quote_received_at_utc: Optional[datetime]
    quote_received_time_ms: Optional[int]
    config_provenance: Mapping[str, Any]
    evidence_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.symbol != "BTCUSDT" or self.timeframe != "4h":
            raise ValueError("Forward paper observations are restricted to BTCUSDT 4h")
        if self.signal_bar_open_ms >= self.signal_bar_close_ms:
            raise ValueError("Signal bar open must predate its close")
        if self.decision not in {"abstain", "long", "short"}:
            raise ValueError(f"Unsupported decision: {self.decision}")
        if self.available_time_ms < self.signal_bar_close_ms:
            raise ValueError("A prospective observation cannot predate its signal bar close")
        if self.observed_at_utc.tzinfo is None:
            raise ValueError("observed_at_utc must be timezone-aware")
        if self.decision == "abstain" and not self.abstention_reason:
            raise ValueError("An abstention must disclose its reason")
        if self.decision != "abstain":
            if (
                self.abstention_reason is not None
                or not self.model_version
                or self.confidence is None
                or self.quote_price is None
            ):
                raise ValueError("A directional decision requires model, confidence, live quote, and no abstention reason")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        quote_parts = (
            self.quote_price,
            self.quote_received_at_utc,
            self.quote_received_time_ms,
        )
        if any(value is None for value in quote_parts) and any(value is not None for value in quote_parts):
            raise ValueError("Quote price and receipt timestamps must be present together")
        if self.quote_price is not None and self.quote_price <= 0:
            raise ValueError("quote_price must be positive")
        if self.quote_received_at_utc is not None and self.quote_received_at_utc.tzinfo is None:
            raise ValueError("quote_received_at_utc must be timezone-aware")
        if self.quote_received_time_ms is not None and self.available_time_ms < self.quote_received_time_ms:
            raise ValueError("Availability cannot predate quote receipt")


def deterministic_decision_id(
    symbol: str,
    timeframe: str,
    signal_bar_open_ms: int,
    signal_bar_close_ms: int,
) -> str:
    """One logical prospective decision per finalized signal bar."""
    payload = (
        f"{RECORDER_VERSION}|{symbol.upper()}|{timeframe}|"
        f"{signal_bar_open_ms}|{signal_bar_close_ms}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fetch_binance_spot_quote(symbol: str, timeout_seconds: float = 8.0) -> LiveQuote:
    """Fetch a current observable quote; this is never a historical fill."""
    url = BINANCE_SPOT_BOOK_TICKER + "?" + urllib.parse.urlencode({"symbol": symbol})
    request = urllib.request.Request(url, headers={"User-Agent": "btc-research-forward-paper/1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    bid = Decimal(str(payload["bidPrice"]))
    ask = Decimal(str(payload["askPrice"]))
    if bid <= 0 or ask <= 0 or ask < bid:
        raise ValueError("Binance returned an invalid top-of-book quote")
    received = datetime.now(timezone.utc)
    return LiveQuote(
        source="binance-spot-bookTicker-bid-ask-mid",
        price=(bid + ask) / Decimal("2"),
        received_at_utc=received,
        received_time_ms=int(received.timestamp() * 1000),
    )


def ensure_observation_schema(conn: Any, cursor: Any) -> None:
    cursor.execute(SCHEMA_SQL)
    conn.commit()


def insert_observation(conn: Any, cursor: Any, observation: PaperObservation) -> bool:
    """Insert once. A retry of the same signal bar is an idempotent no-op."""
    cursor.execute(
        INSERT_SQL,
        (
            str(uuid.uuid4()),
            observation.decision_id,
            RECORDER_VERSION,
            observation.symbol,
            observation.timeframe,
            observation.signal_bar_open_ms,
            observation.signal_bar_close_ms,
            observation.observed_at_utc,
            observation.available_time_ms,
            observation.model_version,
            observation.decision,
            observation.confidence,
            observation.abstention_reason,
            observation.quote_source,
            observation.quote_price,
            observation.quote_received_at_utc,
            observation.quote_received_time_ms,
            json.dumps(observation.config_provenance, sort_keys=True, separators=(",", ":")),
            json.dumps(observation.evidence_provenance, sort_keys=True, separators=(",", ":")),
        ),
    )
    inserted = cursor.fetchone() is not None
    conn.commit()
    return inserted
