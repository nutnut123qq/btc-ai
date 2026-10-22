"""SQL for extraction rows and financial facts (psycopg2 connection owned by the orchestrator; never commits)."""
from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

FACT_COLUMNS = ["Id", "FactKey", "Symbol", "MetricCode", "MetricDefinitionVersion", "MetricGroup", "Statement", "PeriodLabel",
                "PeriodType", "PeriodStart", "PeriodEnd", "FiscalYear", "Consolidation", "Value", "Unit", "Scale", "RawLabel",
                "RawValueText", "RawUnitText", "ColumnRole", "PageNumber", "Location", "DocumentId", "ExtractionId", "Status",
                "ValidationJson", "IsDerived", "DerivationJson", "AvailableAtUtc", "CreatedAtUtc"]
JSON_COLS = {"ValidationJson", "DerivationJson"}
UUID_COLS = {"Id", "DocumentId", "ExtractionId"}


def _json(v: Any) -> str | None:
    return None if v is None else json.dumps(v, ensure_ascii=False, default=_default)


def _default(o: Any):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, uuid.UUID):
        return str(o)
    raise TypeError(type(o).__name__)


class PgRepo:
    def __init__(self, conn):
        self.conn = conn

    def _q(self, sql: str, args: Iterable[Any] = (), fetch: str | None = None):
        with self.conn.cursor() as cur:
            cur.execute(sql, tuple(args))
            if fetch is None:
                return None
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall() if fetch == "all" else ([cur.fetchone()] if fetch == "one" else [])
            out = [dict(zip(cols, r)) for r in rows if r is not None]
            return out if fetch == "all" else (out[0] if out else None)

    # -- extraction --------------------------------------------------------------------------
    def get_extraction(self, doc_id: str, version: str) -> dict | None:
        return self._q('SELECT "Id"::text AS "Id", "Status" FROM "EquityDocumentExtractions" '
                       'WHERE "DocumentId"=%s::uuid AND "ExtractorVersion"=%s FOR UPDATE', (str(doc_id), version), "one")

    def delete_failed_extraction(self, extraction_id: str) -> None:
        self._q('DELETE FROM "EquityDocumentExtractions" WHERE "Id"=%s::uuid AND "Status"=\'failed\'', (extraction_id,))

    def insert_extraction(self, row: dict) -> None:
        self._q('INSERT INTO "EquityDocumentExtractions" ("Id","DocumentId","ExtractorVersion","Method","Status","PageCount",'
                '"PagesOcr","ArtifactPath","QualityJson","Error","StartedAtUtc","FinishedAtUtc") VALUES '
                '(%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s)',
                (row["Id"], row["DocumentId"], row["ExtractorVersion"], row["Method"], row["Status"], row["PageCount"],
                 row["PagesOcr"], row.get("ArtifactPath"), _json(row.get("QualityJson")), (row.get("Error") or None) and row["Error"][:2000],
                 row["StartedAtUtc"], row["FinishedAtUtc"]))

    # -- facts ------------------------------------------------------------------------------------
    def insert_facts(self, facts: list[dict], on_conflict_skip: bool = False) -> int:
        if not facts:
            return 0
        placeholders = []
        for c in FACT_COLUMNS:
            placeholders.append("%s::uuid" if c in UUID_COLS else ("%s::jsonb" if c in JSON_COLS else "%s"))
        sql = (f'INSERT INTO "EquityFinancialFacts" ({",".join(chr(34) + c + chr(34) for c in FACT_COLUMNS)}) '
               f'VALUES ({",".join(placeholders)})')
        if on_conflict_skip:
            sql += ' ON CONFLICT ("FactKey") DO NOTHING'
        n = 0
        with self.conn.cursor() as cur:
            for f in facts:
                args = []
                for c in FACT_COLUMNS:
                    v = f.get(c)
                    if c in JSON_COLS:
                        v = _json(v)
                    elif c in UUID_COLS and v is not None:
                        v = str(v)
                    args.append(v)
                cur.execute(sql, args)
                n += cur.rowcount
        return n

    def facts_for_symbol(self, symbol: str) -> list[dict]:
        return self._q('SELECT f."Id"::text AS "Id", f."FactKey", f."Symbol", f."MetricCode", f."MetricDefinitionVersion", f."MetricGroup", '
                       'f."Statement", f."PeriodLabel", f."PeriodType", f."PeriodStart", f."PeriodEnd", f."FiscalYear", '
                       'f."Consolidation", f."Value", f."Unit", f."Scale", f."ColumnRole", f."DocumentId"::text AS "DocumentId", '
                       'f."Status", f."ValidationJson", f."IsDerived", f."DerivationJson", f."AvailableAtUtc", f."PageNumber", '
                       'd."ReportingPeriod" AS "DocPeriod", d."Revision" AS "DocRevision", d."DocType" AS "DocType", '
                       'd."AuditStatus" AS "DocAuditStatus", '
                       'd."SupersedesDocumentId"::text AS "DocSupersedes", d."ContentSha256" AS "DocSha", '
                       'f."ExtractionId"::text AS "ExtractionId", e."ExtractorVersion" AS "ExtractorVersion", '
                       'e."FinishedAtUtc" AS "ExtractionFinishedAtUtc" '
                       'FROM "EquityFinancialFacts" f JOIN "EquitySourceDocuments" d ON d."Id"=f."DocumentId" '
                       'LEFT JOIN "EquityDocumentExtractions" e ON e."Id"=f."ExtractionId" '
                       'WHERE f."Symbol"=%s ORDER BY f."FactKey"', (symbol,), "all") or []

    def update_fact_status(self, fact_id: str, status: str, validation: dict) -> None:
        self._q('UPDATE "EquityFinancialFacts" SET "Status"=%s, "ValidationJson"=%s::jsonb WHERE "Id"=%s::uuid',
                (status, _json(validation), fact_id))
