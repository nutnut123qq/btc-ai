"""EQX v3 issuer-document pipeline orchestrator (spec-v3-document-pipeline.md §1, §6, §7).

    python ai/equity_pipeline.py run --symbol {BID|MBB} --mode {backfill|incremental} [--run-id <guid>]
        [--budget N] [--max-docs N] [--dry-run] [--no-extract] [--head-check] [--notice-budget N]

Stages: discover (issuer IR listings) -> download/version (EquitySourceDocuments + Locations + Attempts) ->
[MB] notice scan for issuer-stated publication dates -> availability fill -> extract (ai/equity_facts, worker B,
behind an import seam) -> derive_and_cross_validate. The database is the checkpoint: every document is written in
its own transaction, so a killed run resumes where it stopped and never leaves a partial document row.

Exit codes: 0 succeeded, 2 partial (some step failed, budget/max-docs reached, listing incomplete, extractor
unavailable), 75 busy (advisory lock held by another run), 1 failed (unexpected error / bad arguments).
DB connection comes from DB_HOST/DB_PORT/DB_USER/DB_PASS/DB_NAME (no default database; .env is not read).
Document root: EQUITY_DOCUMENTS_ROOT (or EQX_DOCUMENTS_ROOT / --root). The backend launches
`python -I -X utf8 ai/equity_pipeline.py run --symbol X --mode M --run-id G` and owns the run row (also
EQUITY_PIPELINE_RUN_ID); this script updates its counts/checkpoint/heartbeat and final status.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))  # trusted ai/ dir, also under `python -I`
import equity_documents as ed  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PIPELINE_VERSION = "equity-pipeline-v1"
EXIT_OK, EXIT_PARTIAL, EXIT_FAILED, EXIT_BUSY = 0, 2, 1, 75
SCOPE_FROM_YEAR = 2016
HEARTBEAT_S = 30
STALE_RUN_MINUTES = 15
# A CLI run that holds the advisory lock knows no other pipeline process is alive; a "running" row whose heartbeat
# is older than this is a killed run. Fresher rows may belong to a backend run that has not started yet -> busy.
CLI_STALE_MINUTES = 2
DEFAULT_BUDGET = {"backfill": 450, "incremental": 60}
DEFAULT_NOTICE_BUDGET = 200
LATEST_CONSOLIDATED_QUARTERS = 8
# Statutory-ish deadlines after period end used only to say "not yet due" instead of "not listed".
DUE_DAYS = {"Q": 45, "H1": 60, "FY": 100}
# MB server Last-Modified is usable only for uploads after the 2019-07-03 server migration (A0 §4), and only when it
# falls in a plausible filing window after period end.
MB_LASTMOD_MIN = datetime(2019, 7, 4, tzinfo=UTC)
AVAILABILITY_WINDOW_DAYS = 150
# MB in-place overwrite detection: incremental runs HEAD-check MB files of the last 2 fiscal years when the last
# head-check (latest attempt with Source=HEAD_CHECK_SOURCE) is older than this; --head-check forces it.
HEAD_CHECK_SOURCE = "mb-ir-head-check"
HEAD_CHECK_INTERVAL_DAYS = 7

ISSUER = {
    "BID": {"entity": ed.ISSUERS["BID"].entity, "provider": "issuer:BIDV", "hosts": ["bidv.com.vn", "www.bidv.com.vn"]},
    "MBB": {"entity": ed.ISSUERS["MBB"].entity, "provider": "issuer:MB", "hosts": ["mbbank.com.vn", "www.mbbank.com.vn"]},
}

# Gaps verified by the A0 inventory (evidence/equity-expansion/v3/a0-source-inventory.md §3.1). Attached to the
# not-listed attempt so coverage can explain the gap; dynamic evidence (MB notices) is added at run time.
KNOWN_GAP_EVIDENCE = {
    ("BID", "2017Q4", "separate"): "A0 2026-10-19: absent from VN/EN library (2018 group lists only consolidated Q4/2017) "
                                   "and from the disclosure feed; SSC IDS history starts 2025-01; HOSE archive unreachable. "
                                   "Publication unverified.",
    ("BID", "2018Q4", "separate"): "A0 2026-10-19: absent from VN/EN library (2019 group lists only consolidated Q4/2018) "
                                   "and from the disclosure feed; SSC IDS from 2025-01; HOSE unreachable. Unverified.",
    ("MBB", "2016", "separate"): "A0 2026-10-19: published per MB notice 589 (2017-03-29, audited 2016 separate+consolidated) "
                                 "which links the retired SharePoint page; file no longer hosted by MB.",
    ("MBB", "2017", "consolidated"): "A0 2026-10-19: published per MB notice 825 (2018-03-05, audited 2017 separate+consolidated, "
                                     "no link); file not listed by MB.",
    ("MBB", "2017", "separate"): "A0 2026-10-19: published per MB notice 825 (2018-03-05); file not listed by MB.",
    ("MBB", "2017H1", "consolidated"): "A0 2026-10-19: not listed and no notice found in 2017 notices (which are incomplete). Unverified.",
    ("MBB", "2017H1", "separate"): "A0 2026-10-19: not listed and no notice found in 2017 notices (incomplete). Unverified.",
}

log = logging.getLogger("equity_pipeline")


class RunBusy(Exception):
    """Another run row for the symbol is still running (unique index UX_EquityPipelineRuns_OneRunningPerSymbol)."""


def utcnow() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z") if dt else None


def default_root() -> Path:
    return Path(os.environ.get("EQUITY_DOCUMENTS_ROOT") or os.environ.get("EQX_DOCUMENTS_ROOT")
                or r"D:\code\btc\data\equity-documents")


def period_kind(reporting_period: str) -> str:
    return "Q" if "Q" in reporting_period else ("H1" if "H" in reporting_period else "FY")


def expected_periods(today: date, from_year: int = SCOPE_FROM_YEAR) -> list[str]:
    """All Q/H1/FY periods from from_year whose period has ended by `today`."""
    out = []
    for y in range(from_year, today.year + 1):
        for p in (f"{y}Q1", f"{y}Q2", f"{y}H1", f"{y}Q3", f"{y}Q4", f"{y}"):
            if ed.period_end(p).date() < today:
                out.append(p)
    return out


def due_date(reporting_period: str) -> date:
    return (ed.period_end(reporting_period) + timedelta(days=DUE_DAYS[period_kind(reporting_period)])).date()


def published_from_listing_date(day: str | None) -> datetime | None:
    """Issuer-stated date only (ICT) -> available from 00:00 ICT of the next day."""
    if not day:
        return None
    d = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
    return d.replace(hour=24 - ed.ICT_OFFSET_H)


def ict_local_to_utc(local: str) -> datetime:
    dt = datetime.fromisoformat(local.split(".")[0])
    return (dt - timedelta(hours=ed.ICT_OFFSET_H)).replace(tzinfo=UTC)


# ============================================================================================
# Repository: PostgreSQL implementation (psycopg2) + the interface the pipeline uses.
# ============================================================================================

DOC_COLUMNS = ["Id", "Symbol", "Entity", "DocType", "Title", "SourceUrl", "Provider", "ContentSha256", "StoragePath",
               "ByteSize", "MimeType", "ReportingPeriod", "Consolidation", "AuditStatus", "PublishedAtUtc",
               "FirstSeenAtUtc", "FetchedAtUtc", "Revision", "SupersedesDocumentId", "ExtractionJson", "CreatedAtUtc"]
LOC_COLUMNS = ["Id", "DocumentId", "SourceUrl", "Provider", "ListingTitle", "ListingDate", "FirstSeenAtUtc",
               "LastSeenAtUtc", "LastFetchedAtUtc", "HttpEtag", "HttpLastModified"]


class PgRepo:
    """All SQL of the pipeline. `tx()` = one transaction; attempts outside a tx are committed immediately."""

    def __init__(self, dsn_params: dict[str, Any]):
        import psycopg2
        import psycopg2.extras

        psycopg2.extras.register_uuid()
        self._pg, self._extras = psycopg2, psycopg2.extras
        self.conn = psycopg2.connect(**dsn_params)
        self.conn.autocommit = False
        self.lock_conn = psycopg2.connect(**dsn_params)
        self.lock_conn.autocommit = True
        self.hb_conn = psycopg2.connect(**dsn_params)
        self.hb_conn.autocommit = True
        self._in_tx = False
        self._hb_lock = threading.Lock()

    # -- connection / tx ----------------------------------------------------------------------
    @staticmethod
    def params_from_env() -> dict[str, Any]:
        name = os.environ.get("DB_NAME") or os.environ.get("PGDATABASE")
        pw = os.environ.get("DB_PASS") or os.environ.get("DB_PASSWORD") or os.environ.get("PGPASSWORD")
        if not name or not pw:
            raise SystemExit("DB_NAME and DB_PASS must be set explicitly (no default database)")
        return {"host": os.environ.get("DB_HOST", os.environ.get("PGHOST", "localhost")),
                "port": int(os.environ.get("DB_PORT", os.environ.get("PGPORT", "5432"))),
                "user": os.environ.get("DB_USER", os.environ.get("PGUSER", "postgres")), "password": pw, "dbname": name}

    @contextlib.contextmanager
    def tx(self):
        assert not self._in_tx, "nested tx"
        self._in_tx = True
        try:
            yield
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
        finally:
            self._in_tx = False

    def _exec(self, sql: str, args: Iterable[Any] = (), fetch: str | None = None) -> Any:
        with self.conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
            cur.execute(sql, tuple(args))
            res = cur.fetchall() if fetch == "all" else (cur.fetchone() if fetch == "one" else None)
        if not self._in_tx:
            self.conn.commit()
        return res

    def close(self) -> None:
        for c in (self.conn, self.hb_conn, self.lock_conn):
            with contextlib.suppress(Exception):
                c.close()

    # -- lock -----------------------------------------------------------------------------------
    def try_lock(self, symbol: str) -> bool:
        with self.lock_conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (f"equity-docs:{symbol}",))
            return bool(cur.fetchone()[0])

    def unlock(self, symbol: str) -> None:
        with self.lock_conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (f"equity-docs:{symbol}",))

    # -- runs ----------------------------------------------------------------------------------
    def run_get(self, run_id: uuid.UUID) -> dict | None:
        return self._exec('SELECT * FROM "EquityPipelineRuns" WHERE "Id"=%s', (run_id,), "one")

    def run_fail_stale(self, symbol: str, now: datetime) -> int:
        rows = self._exec('UPDATE "EquityPipelineRuns" SET "Status"=\'failed\', "FinishedAtUtc"=%s, "Error"=%s '
                          'WHERE "Symbol"=%s AND "Status"=\'running\' AND "HeartbeatAtUtc" < %s RETURNING "Id"',
                          (now, "stale-heartbeat (marked by equity_pipeline holding the advisory lock)", symbol,
                           now - timedelta(minutes=CLI_STALE_MINUTES)), "all")
        return len(rows or [])

    def run_create(self, run_id: uuid.UUID, symbol: str, mode: str, trigger: str, now: datetime) -> None:
        try:
            self._run_create(run_id, symbol, mode, trigger, now)
        except self._pg.IntegrityError as exc:
            self.conn.rollback()
            raise RunBusy(str(exc).splitlines()[0]) from exc

    def _run_create(self, run_id: uuid.UUID, symbol: str, mode: str, trigger: str, now: datetime) -> None:
        self._exec('INSERT INTO "EquityPipelineRuns" ("Id","Symbol","Mode","Trigger","Status","StartedAtUtc",'
                   '"HeartbeatAtUtc","CountsJson","CheckpointJson") VALUES (%s,%s,%s,%s,\'running\',%s,%s,%s,%s)',
                   (run_id, symbol, mode, trigger, now, now, self._extras.Json({}), self._extras.Json({})))

    def run_heartbeat(self, run_id: uuid.UUID, counts: dict, checkpoint: dict, log_tail: str) -> None:
        with self._hb_lock, self.hb_conn.cursor() as cur:
            cur.execute('UPDATE "EquityPipelineRuns" SET "HeartbeatAtUtc"=GREATEST(now(), "StartedAtUtc"), "CountsJson"=%s, '
                        '"CheckpointJson"=%s, "LogTail"=%s WHERE "Id"=%s AND "Status"=\'running\'',
                        (self._extras.Json(counts), self._extras.Json(checkpoint), log_tail[-8000:], run_id))

    def run_finish(self, run_id: uuid.UUID, status: str, counts: dict, checkpoint: dict, error: str | None,
                   log_tail: str) -> None:
        with self._hb_lock, self.hb_conn.cursor() as cur:
            cur.execute('UPDATE "EquityPipelineRuns" SET "Status"=%s, "FinishedAtUtc"=GREATEST(now(), "StartedAtUtc"), '
                        '"HeartbeatAtUtc"=GREATEST(now(), "StartedAtUtc"), "CountsJson"=%s, "CheckpointJson"=%s, '
                        '"Error"=%s, "LogTail"=%s WHERE "Id"=%s AND "Status"=\'running\'',
                        (status, self._extras.Json(counts), self._extras.Json(checkpoint),
                         (error or None) and error[:2000], log_tail[-8000:], run_id))

    # -- attempts -------------------------------------------------------------------------------
    def attempt(self, a: dict[str, Any]) -> int:
        row = self._exec('INSERT INTO "EquitySourceAttempts" ("RunId","Symbol","Source","TargetPeriod","DocType",'
                         '"Consolidation","Url","AttemptedAtUtc","Outcome","HttpStatus","Detail") VALUES '
                         '(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING "Id"',
                         (a.get("RunId"), a["Symbol"], a["Source"], a.get("TargetPeriod"), a.get("DocType"),
                          a.get("Consolidation"), (a.get("Url") or None) and a["Url"][:1000], a["AttemptedAtUtc"],
                          a["Outcome"], a.get("HttpStatus"), (a.get("Detail") or None) and a["Detail"][:2000]), "one")
        return row["Id"]

    def last_attempt_at(self, symbol: str, source: str) -> datetime | None:
        row = self._exec('SELECT max("AttemptedAtUtc") AS t FROM "EquitySourceAttempts" WHERE "Symbol"=%s AND "Source"=%s',
                         (symbol, source), "one")
        return row["t"] if row else None

    def attempts_for_source(self, symbol: str, source: str, outcome: str) -> list[dict]:
        return self._exec('SELECT "Url","Detail","AttemptedAtUtc" FROM "EquitySourceAttempts" WHERE "Symbol"=%s '
                          'AND "Source"=%s AND "Outcome"=%s ORDER BY "AttemptedAtUtc"', (symbol, source, outcome), "all")

    # -- documents / locations -----------------------------------------------------------------
    def location_by_url(self, url: str) -> dict | None:
        return self._exec('SELECT l.*, d."ByteSize" AS "DocByteSize", d."Revision" AS "DocRevision", '
                          'd."ContentSha256" AS "DocSha" FROM "EquitySourceDocumentLocations" l JOIN '
                          '"EquitySourceDocuments" d ON d."Id"=l."DocumentId" WHERE l."SourceUrl"=%s '
                          'ORDER BY d."Revision" DESC, l."LastFetchedAtUtc" DESC NULLS LAST LIMIT 1', (url,), "one")

    def latest_doc_for_identity(self, symbol: str, identity: str) -> dict | None:
        if identity.startswith("bidv-wcm:"):
            pat = f"%/connect/{identity.split(':', 1)[1]}/%"
            return self._exec('SELECT d.* FROM "EquitySourceDocuments" d JOIN "EquitySourceDocumentLocations" l ON '
                              'l."DocumentId"=d."Id" WHERE d."Symbol"=%s AND lower(l."SourceUrl") LIKE %s '
                              'ORDER BY d."Revision" DESC, d."FetchedAtUtc" DESC LIMIT 1', (symbol, pat), "one")
        if identity.startswith("url:"):
            return self._exec('SELECT d.* FROM "EquitySourceDocuments" d JOIN "EquitySourceDocumentLocations" l ON '
                              'l."DocumentId"=d."Id" WHERE d."Symbol"=%s AND l."SourceUrl"=%s '
                              'ORDER BY d."Revision" DESC LIMIT 1', (symbol, identity[4:]), "one")
        return None

    def doc_by_sha(self, symbol: str, sha: str) -> dict | None:
        return self._exec('SELECT * FROM "EquitySourceDocuments" WHERE "Symbol"=%s AND "ContentSha256"=%s',
                          (symbol, sha), "one")

    def docs_without_locations(self, symbol: str) -> list[dict]:
        return self._exec('SELECT d.* FROM "EquitySourceDocuments" d WHERE d."Symbol"=%s AND NOT EXISTS (SELECT 1 '
                          'FROM "EquitySourceDocumentLocations" l WHERE l."DocumentId"=d."Id")', (symbol,), "all")

    def docs(self, symbol: str) -> list[dict]:
        return self._exec('SELECT * FROM "EquitySourceDocuments" WHERE "Symbol"=%s', (symbol,), "all")

    def insert_document(self, d: dict[str, Any]) -> None:
        vals = [self._extras.Json(d[c]) if c == "ExtractionJson" and d[c] is not None else d[c] for c in DOC_COLUMNS]
        self._exec('INSERT INTO "EquitySourceDocuments" (' + ",".join(f'"{c}"' for c in DOC_COLUMNS) + ") VALUES ("
                   + ",".join(["%s"] * len(DOC_COLUMNS)) + ")", vals)

    def upsert_location(self, loc: dict[str, Any]) -> None:
        cols = [c for c in LOC_COLUMNS if c != "Id"]
        self._exec('INSERT INTO "EquitySourceDocumentLocations" (' + ",".join(f'"{c}"' for c in cols) + ") VALUES ("
                   + ",".join(["%s"] * len(cols)) + ') ON CONFLICT ("DocumentId","SourceUrl") DO UPDATE SET '
                   '"LastSeenAtUtc"=GREATEST(EXCLUDED."LastSeenAtUtc","EquitySourceDocumentLocations"."LastSeenAtUtc"), '
                   '"LastFetchedAtUtc"=COALESCE(EXCLUDED."LastFetchedAtUtc","EquitySourceDocumentLocations"."LastFetchedAtUtc"), '
                   '"HttpEtag"=COALESCE(EXCLUDED."HttpEtag","EquitySourceDocumentLocations"."HttpEtag"), '
                   '"HttpLastModified"=COALESCE(EXCLUDED."HttpLastModified","EquitySourceDocumentLocations"."HttpLastModified"), '
                   '"ListingTitle"=COALESCE("EquitySourceDocumentLocations"."ListingTitle",EXCLUDED."ListingTitle"), '
                   '"ListingDate"=COALESCE("EquitySourceDocumentLocations"."ListingDate",EXCLUDED."ListingDate")',
                   [loc.get(c) for c in cols])

    def fill_published(self, doc_id: uuid.UUID, published: datetime, provenance: dict) -> bool:
        row = self._exec('UPDATE "EquitySourceDocuments" SET "PublishedAtUtc"=%s, "ExtractionJson"=COALESCE('
                         '"ExtractionJson", \'{}\'::jsonb) || jsonb_build_object(\'availability\', %s::jsonb) '
                         'WHERE "Id"=%s AND "PublishedAtUtc" IS NULL RETURNING "Id"',
                         (published, json.dumps(provenance, ensure_ascii=False), doc_id), "one")
        return row is not None

    # -- extraction ----------------------------------------------------------------------------
    def docs_needing_extraction(self, symbol: str, version: str, consolidations: list[str]) -> list[dict]:
        return self._exec('SELECT d.* FROM "EquitySourceDocuments" d WHERE d."Symbol"=%s AND d."DocType"=\'financial-statement\' '
                          'AND d."Consolidation" = ANY(%s) AND NOT EXISTS (SELECT 1 FROM "EquityDocumentExtractions" e '
                          'WHERE e."DocumentId"=d."Id" AND e."ExtractorVersion"=%s AND e."Status" IN (\'succeeded\',\'partial\')) '
                          'ORDER BY d."ReportingPeriod" DESC NULLS LAST, d."Revision" DESC', (symbol, consolidations, version), "all")

    def clear_failed_extraction(self, doc_id: uuid.UUID, version: str) -> None:
        self._exec('DELETE FROM "EquityDocumentExtractions" WHERE "DocumentId"=%s AND "ExtractorVersion"=%s '
                   "AND \"Status\"='failed'", (doc_id, version))

    def record_failed_extraction(self, doc_id: uuid.UUID, version: str, error: str, started: datetime, now: datetime) -> None:
        self._exec('INSERT INTO "EquityDocumentExtractions" ("Id","DocumentId","ExtractorVersion","Method","Status",'
                   '"PageCount","PagesOcr","QualityJson","Error","StartedAtUtc","FinishedAtUtc") VALUES '
                   '(%s,%s,%s,\'mixed\',\'failed\',0,0,%s,%s,%s,%s) ON CONFLICT ("DocumentId","ExtractorVersion") DO UPDATE '
                   'SET "Error"=EXCLUDED."Error", "FinishedAtUtc"=EXCLUDED."FinishedAtUtc" '
                   'WHERE "EquityDocumentExtractions"."Status"=\'failed\'',
                   (uuid.uuid4(), doc_id, version, self._extras.Json({"recordedBy": PIPELINE_VERSION,
                                                                       "note": "extractor raised; transaction rolled back"}),
                    error[:2000], started, max(now, started)))


# ============================================================================================
# Run context: counts, log tail, heartbeat.
# ============================================================================================

class _TailHandler(logging.Handler):
    def __init__(self, buf: deque):
        super().__init__()
        self.buf = buf

    def emit(self, record: logging.LogRecord) -> None:
        self.buf.append(self.format(record))


class RunContext:
    def __init__(self, repo: Any, run_id: uuid.UUID | None, symbol: str, mode: str, dry_run: bool,
                 clock: Callable[[], datetime] = utcnow):
        self.repo, self.run_id, self.symbol, self.mode, self.dry_run, self.clock = repo, run_id, symbol, mode, dry_run, clock
        self.counts: dict[str, Any] = {"pipelineVersion": PIPELINE_VERSION, "discover": {}, "notices": {}, "download": {},
                                       "availability": {}, "extract": {}, "derive": {}, "requests": {}}
        self.checkpoint: dict[str, Any] = {"stage": "start"}
        self.problems: list[str] = []
        self.tail: deque[str] = deque(maxlen=120)
        self._handler = _TailHandler(self.tail)
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        log.addHandler(self._handler)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_beat = 0.0
        self.pending_links: dict[str, dict] = {}  # dry-run: legacy documents the link step would attach

    def inc(self, stage: str, key: str, n: int = 1) -> None:
        self.counts[stage][key] = self.counts[stage].get(key, 0) + n

    def problem(self, msg: str) -> None:
        self.problems.append(msg)
        log.warning(msg)

    def stage(self, name: str, **extra: Any) -> None:
        self.checkpoint = {"stage": name, **extra}
        log.info("stage %s %s", name, json.dumps(extra, ensure_ascii=False) if extra else "")
        self.beat(force=True)

    def log_tail(self) -> str:
        return "\n".join(self.tail)[-8000:]

    def beat(self, force: bool = False) -> None:
        """Heartbeat (also the callback handed to the extractor): writes at most every 20 s unless forced."""
        if self.dry_run or self.run_id is None:
            return
        now = time.monotonic()
        if not force and now - self._last_beat < 20:
            return
        self._last_beat = now
        try:
            self.repo.run_heartbeat(self.run_id, copy.deepcopy(self.counts), dict(self.checkpoint), self.log_tail())
        except Exception as exc:  # noqa: BLE001 - a lost heartbeat must not kill the run
            log.warning("heartbeat failed: %s", exc)

    def start_heartbeat(self) -> None:
        if self.dry_run or self.run_id is None:
            return

        def loop() -> None:
            while not self._stop.wait(HEARTBEAT_S):
                self.beat(force=True)

        self._thread = threading.Thread(target=loop, name="equity-pipeline-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.removeHandler(self._handler)


# ============================================================================================
# Discovery
# ============================================================================================

class ListingResult:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []      # library items (statements + skipped)
        self.complete = True
        self.errors: list[str] = []
        self.unstorable: list[dict[str, Any]] = []  # in-scope statements listed in a form we cannot store
        self.covered_from: datetime | None = None  # incremental newest-first listing: oldest period end read


def _listing_attempt(ctx: RunContext, source: str, url: str, outcome: str, status: int | None, detail: str) -> None:
    if ctx.dry_run:
        return
    ctx.repo.attempt({"RunId": ctx.run_id, "Symbol": ctx.symbol, "Source": source, "Url": url,
                      "AttemptedAtUtc": ctx.clock(), "Outcome": outcome, "HttpStatus": status, "Detail": detail})


def discover_bid(ctx: RunContext, http: ed.PoliteHttp) -> ListingResult:
    res = ListingResult()
    for source, url, kind in (("bidv-ir-library", ed.BIDV_LIBRARY_VI, "library"),
                              ("bidv-ir-disclosure-feed", ed.BIDV_DISCLOSURE_FEED, "feed")):
        r = http.request(url)
        ctx.inc("discover", "listingRequests")
        if r["outcome"] != "ok":
            res.complete = False
            res.errors.append(f"{source}: {r['outcome']} {r.get('status')}")
            _listing_attempt(ctx, source, url, r["outcome"], r.get("status"), f"listing fetch failed: {r.get('detail')}")
            continue
        items = ed.parse_bidv_items(r["resp"].text, url)
        if kind == "library":
            fs = [i for i in items if "/baocaotaichinh/" in (i.get("contentPath") or "")]
            if not fs:
                res.complete = False
                res.errors.append(f"{source}: 0 library items parsed (layout change?)")
                _listing_attempt(ctx, source, url, "parse-error", r["status"], "no `var item` in the financial-statement library")
                continue
            for i in fs:
                i["source"] = source
                i["classification"] = ed.classify_v3(i["title"], i["fileName"])
            res.items.extend(fs)
            _listing_attempt(ctx, source, url, "listed", r["status"],
                             f"{len(fs)} library items ({len(items)} items on page); single page, no pagination")
        else:
            picked = 0
            for i in items:
                c = ed.classify_v3(i["title"], i["fileName"])
                if c["restatement"] or (c["skipReason"] or "").startswith("profit-variance"):
                    i["source"], i["classification"] = source, c
                    res.items.append(i)
                    picked += 1
            _listing_attempt(ctx, source, url, "listed", r["status"],
                             f"{len(items)} disclosure items; {picked} restatement/explanation items kept")
    return res


class MbApi:
    """MB IR XHR API client. The page embeds a per-session anti-forgery token (public, not a credential) that the
    API requires together with the session cookies. MB's WAF answers some sessions with a sticky 403 (observed
    2026-10-19: 4 retries on the same session all 403, a fresh session succeeds at once), so a 403 that survives
    one backoff retry triggers a session refresh (new cookies + token), at most `max_refresh` times per call."""

    def __init__(self, http: ed.PoliteHttp, ctx: RunContext, page_url: str, stage: str, max_refresh: int = 3):
        self.http, self.ctx, self.page_url, self.stage, self.max_refresh = http, ctx, page_url, stage, max_refresh
        self.headers: dict[str, str] | None = None

    def _session(self, fresh: bool) -> ed.FetchOutcome | None:
        if fresh and hasattr(self.http.session, "cookies"):
            self.http.session.cookies.clear()
        r = self.http.request(self.page_url)
        self.ctx.inc(self.stage, "sessionPageRequests")
        if r["outcome"] != "ok":
            return r
        m = re.search(r'name="__RequestVerificationToken" type="hidden" value="([^"]+)"', r["resp"].text)
        self.headers = {"Referer": self.page_url, "X-Requested-With": "XMLHttpRequest"}
        if m:
            self.headers["MB-XSRF-Token-FormOnline"] = m.group(1)
        return None

    def get(self, url: str) -> ed.FetchOutcome:
        if self.headers is None:
            err = self._session(False)
            if err is not None:
                err["detail"] = f"IR page {self.page_url} failed: {err.get('detail')}"
                return err
        r: ed.FetchOutcome | None = None
        for i in range(self.max_refresh + 1):
            r = self.http.request(url, headers=self.headers, max_retries=1)
            if r["outcome"] == "ok" or r.get("status") != 403:
                return r
            if i < self.max_refresh:
                self.ctx.inc(self.stage, "sessionRefreshes")
                err = self._session(True)
                if err is not None:
                    return err
        assert r is not None
        r["detail"] = f"{r.get('detail')}; still 403 after {self.max_refresh} session refreshes"
        return r


def discover_mb(ctx: RunContext, http: ed.PoliteHttp) -> ListingResult:
    res = ListingResult()
    source = "mb-ir-finance-api"
    api = MbApi(http, ctx, ed.MB_FS_PAGE, "discover")
    for cat, cat_cons in ed.MB_FS_CATEGORIES.items():
        page, total = 1, 1
        while page <= total:
            url = f"{ed.MB_ORIGIN}/api/GetListFinance/{cat}/{page}/0"
            r = api.get(url)
            ctx.inc("discover", "listingRequests")
            if r["outcome"] != "ok":
                res.complete = False
                res.errors.append(f"cat {cat} page {page}: {r['outcome']} {r.get('status')}")
                _listing_attempt(ctx, source, url, r["outcome"], r.get("status"), f"listing page failed: {r.get('detail')}")
                break
            try:
                items, total = ed.parse_mb_finance_page(r["resp"].json(), cat)
            except Exception as exc:  # noqa: BLE001
                res.complete = False
                res.errors.append(f"cat {cat} page {page}: parse {exc}")
                _listing_attempt(ctx, source, url, "parse-error", r["status"], f"JSON parse failed: {exc}")
                break
            for i in items:
                i["source"] = source
                i["classification"] = ed.classify_v3(i["title"], i["fileName"], cat_cons)
            res.items.extend(items)
            _listing_attempt(ctx, source, url, "listed", r["status"], f"category {cat} ({cat_cons}) page {page}/{total}: {len(items)} items")
            page += 1
            if ctx.mode == "incremental" and page <= total:
                # Newest first: stop once a page overlaps what is already stored. Coverage cells are then only
                # claimed for periods at/after the oldest item read (ListingResult.covered_from).
                known = [i for i in items if i.get("url") and ctx.repo.location_by_url(i["url"]) is not None]
                if known:
                    ends = [ed.period_end(i["classification"]["reportingPeriod"]) for i in items
                            if i["classification"].get("reportingPeriod")]
                    if ends:
                        cut = min(ends)
                        res.covered_from = cut if res.covered_from is None else max(res.covered_from, cut)
                    ctx.inc("discover", "pagesSkippedIncremental", total - page + 1)
                    break
    return res


# ============================================================================================
# Planning
# ============================================================================================

def in_scope(item: dict[str, Any]) -> bool:
    c = item["classification"]
    return bool(c["isFinancialStatement"] and (c["fiscalYear"] or 0) >= SCOPE_FROM_YEAR and item.get("url")
                and (item.get("fileType") or "") == "pdf")


def skip_reason(item: dict[str, Any]) -> str | None:
    c = item["classification"]
    if c["skipReason"]:
        return c["skipReason"]
    if not item.get("url"):
        return "listing item has no file attached"
    if (item.get("fileType") or "") != "pdf":
        return f"not a PDF ({item.get('fileType')})"
    if (c["fiscalYear"] or 0) < SCOPE_FROM_YEAR:
        return f"before scope ({c['fiscalYear']} < {SCOPE_FROM_YEAR})"
    return None


def item_year(item: dict[str, Any]) -> int:
    c = item.get("classification") or {}
    if c.get("fiscalYear"):
        return int(c["fiscalYear"])
    years = re.findall(r"(?<!\d)(20\d\d)(?!\d)", item.get("title") or "")
    if years:
        return int(years[-1])
    if str(item.get("group") or "").isdigit():
        return int(item["group"])
    return int((item.get("listedDate") or "0")[:4] or 0)


def doc_consolidation(item: dict[str, Any]) -> str:
    c = item["classification"]
    return c["consolidation"] if c["consolidation"] in ("consolidated", "separate") else "unknown"


def priority_order(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """consolidated latest 8 quarters -> consolidated history -> restatements -> separate (newest first)."""
    def end(i: dict) -> datetime:
        return ed.period_end(i["classification"]["reportingPeriod"])

    cons_q = sorted({i["classification"]["reportingPeriod"] for i in targets
                     if doc_consolidation(i) == "consolidated" and period_kind(i["classification"]["reportingPeriod"]) == "Q"},
                    key=lambda p: ed.period_end(p), reverse=True)[:LATEST_CONSOLIDATED_QUARTERS]

    def tier(i: dict) -> int:
        cons, rp = doc_consolidation(i), i["classification"]["reportingPeriod"]
        if cons == "consolidated" and rp in cons_q:
            return 0
        if cons == "consolidated":
            return 1
        if i["classification"]["restatement"]:
            return 2
        return 3

    return sorted(targets, key=lambda i: (tier(i), -end(i).timestamp(), i.get("url") or ""))


# ============================================================================================
# Download / version
# ============================================================================================

def _attempt(ctx: RunContext, item: dict[str, Any] | None, outcome: str, *, status: int | None = None,
             detail: str = "", url: str | None = None, source: str | None = None, period: str | None = None,
             cons: str | None = None, doc_type: str | None = "financial-statement") -> None:
    if ctx.dry_run:
        return
    c = (item or {}).get("classification") or {}
    ctx.repo.attempt({"RunId": ctx.run_id, "Symbol": ctx.symbol, "Source": source or (item or {}).get("source") or "pipeline",
                      "TargetPeriod": period or c.get("reportingPeriod"), "DocType": doc_type,
                      "Consolidation": cons or (doc_consolidation(item) if item else None),
                      "Url": url or (item or {}).get("url"), "AttemptedAtUtc": ctx.clock(), "Outcome": outcome,
                      "HttpStatus": status, "Detail": detail})


def _listing_provenance(item: dict[str, Any]) -> dict[str, Any]:
    keys = ("title", "listingUrl", "listedDate", "listedSize", "group", "category", "mbId", "cmsLastSaveLocal",
            "filenameDatePrefix", "contentPath", "identity", "versionToken", "source")
    return {k: item.get(k) for k in keys if item.get(k) is not None}


def _availability_from_listing(ctx: RunContext, item: dict[str, Any]) -> tuple[datetime | None, dict | None]:
    if ctx.symbol == "BID" and item.get("listedDate"):
        pub = published_from_listing_date(item["listedDate"])
        return pub, {"basis": "issuer IR listing publishdate (date only, ICT) -> 00:00 ICT next day",
                     "statedDate": item["listedDate"], "source": item.get("source")}
    return None, None


def process_target(ctx: RunContext, http: ed.PoliteHttp, root: Path, item: dict[str, Any], *, head_check: bool,
                   max_bytes: int) -> str:
    """Returns 'skipped-unchanged' | 'downloaded' | 'changed' | 'unchanged' | 'duplicate-content' | outcome-of-failure."""
    repo, sym, url = ctx.repo, ctx.symbol, item["url"]
    c = item["classification"]
    loc = repo.location_by_url(url)
    if loc is None and ctx.dry_run and url in ctx.pending_links:
        d = ctx.pending_links[url]
        loc = {"DocumentId": d["Id"], "DocByteSize": d["ByteSize"], "DocRevision": d["Revision"],
               "FirstSeenAtUtc": d["FirstSeenAtUtc"], "HttpEtag": None, "HttpLastModified": None}
        ctx.inc("download", "wouldLinkExisting")
    now = ctx.clock()
    listing_date = date.fromisoformat(item["listedDate"]) if item.get("listedDate") else None
    if loc is not None:
        size_ok = ed.listed_size_matches(item.get("listedSize"), int(loc["DocByteSize"]))
        # HEAD re-validation is MB-only. MB serves fixed static paths that can be overwritten in place, possibly with
        # the same size, so only ETag/Last-Modified reveal it. BIDV URLs carry the WCM version token
        # (CACHEID=ROOTWORKSPACE-{uuid}-{token}); WCM issues a new token when the item's file is replaced, so new
        # content arrives as a new URL, and a same-URL change still shows through the listed size compared on every
        # run (see the fixture drill). BIDV also sends no Last-Modified, so HEAD would add requests without a better
        # signal than the token + size.
        need_head = head_check and sym == "MBB" and (c["fiscalYear"] or 0) >= now.year - 2
        if size_ok is not False and not need_head:
            if not ctx.dry_run:
                pub, prov = _availability_from_listing(ctx, item)
                if pub is not None and repo.fill_published(loc["DocumentId"], pub, prov):
                    ctx.inc("availability", "filledFromListingOnKnownDoc")
                with repo.tx():
                    repo.upsert_location({"DocumentId": loc["DocumentId"], "SourceUrl": url, "Provider": ISSUER[sym]["provider"],
                                          "ListingTitle": item["title"][:500], "ListingDate": listing_date,
                                          "FirstSeenAtUtc": loc["FirstSeenAtUtc"], "LastSeenAtUtc": now,
                                          "LastFetchedAtUtc": None, "HttpEtag": None, "HttpLastModified": None})
                    if ctx.mode == "backfill":
                        _attempt(ctx, item, "unchanged", detail=(
                            f"stored as document {loc['DocumentId']} rev {loc['DocRevision']}; listed size "
                            f"{item.get('listedSize')!r} {'matches' if size_ok else 'not stated'} stored {loc['DocByteSize']} bytes; not re-fetched"))
            ctx.inc("download", "unchangedNoFetch")
            return "skipped-unchanged"
        if need_head and size_ok is not False:
            h = http.request(url, method="HEAD")
            ctx.inc("download", "headChecks")
            if h["outcome"] != "ok":
                _attempt(ctx, item, h["outcome"], source=HEAD_CHECK_SOURCE, status=h.get("status"),
                         detail=f"HEAD check failed ({h.get('detail')}); falling back to GET")
            else:
                et, lm = h["resp"].headers.get("ETag"), h["resp"].headers.get("Last-Modified")
                old_et, old_lm = loc.get("HttpEtag"), loc.get("HttpLastModified")
                differs = bool((old_et and et and et != old_et) or (old_lm and lm and lm != old_lm))
                baseline = not old_et and not old_lm
                if not differs and (not baseline or size_ok):
                    if not ctx.dry_run:
                        with repo.tx():
                            repo.upsert_location({"DocumentId": loc["DocumentId"], "SourceUrl": url, "Provider": ISSUER[sym]["provider"],
                                                  "ListingTitle": item["title"][:500], "ListingDate": listing_date,
                                                  "FirstSeenAtUtc": loc["FirstSeenAtUtc"], "LastSeenAtUtc": now,
                                                  "LastFetchedAtUtc": None, "HttpEtag": et, "HttpLastModified": lm})
                            _attempt(ctx, item, "unchanged", source=HEAD_CHECK_SOURCE, status=h.get("status"),
                                     detail=(f"HEAD check: ETag {et!r} / Last-Modified {lm!r} "
                                             + ("recorded as baseline" if baseline else "unchanged") + f"; document {loc['DocumentId']}"))
                    ctx.inc("download", "unchangedNoFetch")
                    return "skipped-unchanged"
                ctx.inc("download", "headChanged")
                if not ctx.dry_run:
                    _attempt(ctx, item, "changed", source=HEAD_CHECK_SOURCE, status=h.get("status"),
                             detail=f"HEAD check: validators changed (ETag {old_et!r}->{et!r}, Last-Modified {old_lm!r}->{lm!r}); re-fetching")
    if ctx.dry_run:
        ctx.inc("download", "wouldFetch")
        return "would-fetch"
    r = ed.download_pdf(http, url, root / sym, max_bytes)
    ctx.inc("download", "fetchAttempts")
    if r["outcome"] != "ok":
        key = {"http-error": "httpError", "rate-limited": "rateLimited", "timeout": "timeout", "too-large": "tooLarge",
               "parse-error": "notPdf"}[r["outcome"]]
        ctx.inc("download", key)
        _attempt(ctx, item, r["outcome"], status=r.get("status"), detail=r.get("detail") or "")
        ctx.problem(f"download {c['reportingPeriod']} {doc_consolidation(item)}: {r['outcome']} {r.get('detail')}")
        return r["outcome"]
    sha, size, tmp = r["sha256"], r["bytes"], Path(r["tmpPath"])
    rel = f"{sym}/{sha}.pdf"
    dest = root / rel
    if dest.exists():
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(dest)  # content-addressed: an orphan after a crash is harmless and reused next run
    ctx.inc("download", "bytes", size)
    existing = repo.doc_by_sha(sym, sha)
    prior = None
    if existing is None:
        prior = repo.latest_doc_for_identity(sym, item.get("identity") or f"url:{url}")
    loc_row = {"SourceUrl": url, "Provider": ISSUER[sym]["provider"], "ListingTitle": item["title"][:500],
               "ListingDate": listing_date, "FirstSeenAtUtc": now, "LastSeenAtUtc": now, "LastFetchedAtUtc": now,
               "HttpEtag": r.get("etag"), "HttpLastModified": r.get("lastModified")}
    with repo.tx():
        if existing is not None:
            repo.upsert_location({**loc_row, "DocumentId": existing["Id"]})
            if loc is not None and loc["DocumentId"] == existing["Id"]:
                outcome, detail, key = "unchanged", f"re-fetched; same sha256 as document {existing['Id']}", "unchanged"
            else:
                outcome, key = "downloaded", "duplicateContent"
                detail = f"bytes identical to stored document {existing['Id']} ({existing['ContentSha256'][:12]}); location added, no new document"
            _attempt(ctx, item, outcome, status=r["status"], detail=detail)
        else:
            pub, prov = _availability_from_listing(ctx, item)
            doc_id = uuid.uuid4()
            extraction_json = {"collector": ed.V3_COLLECTOR_VERSION, "pipeline": PIPELINE_VERSION,
                               "listing": _listing_provenance(item), "classification": c,
                               "http": {"etag": r.get("etag"), "lastModified": r.get("lastModified"),
                                        "contentType": r.get("contentType"), "finalUrl": r.get("finalUrl")}}
            if prov:
                extraction_json["availability"] = prov
            if c.get("restatement"):
                extraction_json["restatement"] = {"isRestatement": True, "covers": c.get("covers"),
                                                  "note": "issuer adjustment/supplement document for an audited FY statement; "
                                                          "original FS document kept unchanged"}
            doc = {"Id": doc_id, "Symbol": sym, "Entity": ISSUER[sym]["entity"], "DocType": "financial-statement",
                   "Title": item["title"][:500], "SourceUrl": url[:1000], "Provider": ISSUER[sym]["provider"],
                   "ContentSha256": sha, "StoragePath": rel, "ByteSize": size, "MimeType": "application/pdf",
                   "ReportingPeriod": c["reportingPeriod"], "Consolidation": doc_consolidation(item),
                   "AuditStatus": c["auditStatus"] if c["auditStatus"] in ("audited", "reviewed") else "unknown",
                   "PublishedAtUtc": pub, "FirstSeenAtUtc": now, "FetchedAtUtc": now,
                   "Revision": (prior["Revision"] + 1) if prior else 1, "SupersedesDocumentId": prior["Id"] if prior else None,
                   "ExtractionJson": extraction_json, "CreatedAtUtc": now}
            repo.insert_document(doc)
            repo.upsert_location({**loc_row, "DocumentId": doc_id})
            if prior:
                outcome, key = "changed", "changed"
                detail = (f"new bytes for listing identity {item.get('identity')}: document {doc_id} rev {doc['Revision']} "
                          f"supersedes {prior['Id']} rev {prior['Revision']} ({prior['ContentSha256'][:12]} -> {sha[:12]})")
            else:
                outcome, key, detail = "downloaded", "downloaded", f"document {doc_id} ({size} bytes, sha256 {sha[:12]})"
            _attempt(ctx, item, outcome, status=r["status"], detail=detail)
    ctx.inc("download", key)
    log.info("%s %s %s %s", key, c["reportingPeriod"], doc_consolidation(item), sha[:12])
    return key


def head_check_due(ctx: RunContext, forced: bool) -> bool:
    """MB only: forced by --head-check, or automatic in incremental mode when the last head-check (derived from
    EquitySourceAttempts, no schema change) is missing or older than HEAD_CHECK_INTERVAL_DAYS."""
    if ctx.symbol != "MBB":
        return False
    if forced:
        ctx.counts["download"]["headCheckReason"] = "forced (--head-check)"
        return True
    if ctx.mode != "incremental":
        return False
    last = ctx.repo.last_attempt_at(ctx.symbol, HEAD_CHECK_SOURCE)
    due = last is None or ctx.clock() - last > timedelta(days=HEAD_CHECK_INTERVAL_DAYS)
    ctx.counts["download"]["headCheckReason"] = (
        f"auto: last head-check {iso(last) if last else 'never'}" + (" (due)" if due else f" (< {HEAD_CHECK_INTERVAL_DAYS} days, skipped)"))
    return due


def reconcile_existing(ctx: RunContext) -> int:
    """Documents registered before v3 (no Location rows) get their Location from SourceUrl; no re-download.
    Dry-run only records what would be linked so the plan does not report those documents as "would fetch"."""
    if ctx.dry_run:
        ctx.pending_links = {d["SourceUrl"]: d for d in ctx.repo.docs_without_locations(ctx.symbol)}
        ctx.counts["download"]["wouldReconcileLegacyDocuments"] = len(ctx.pending_links)
        return 0
    n = 0
    for d in ctx.repo.docs_without_locations(ctx.symbol):
        with ctx.repo.tx():
            ctx.repo.upsert_location({"DocumentId": d["Id"], "SourceUrl": d["SourceUrl"], "Provider": d["Provider"],
                                      "ListingTitle": (d.get("Title") or "")[:500] or None, "ListingDate": None,
                                      "FirstSeenAtUtc": d["FirstSeenAtUtc"], "LastSeenAtUtc": d["FetchedAtUtc"],
                                      "LastFetchedAtUtc": d["FetchedAtUtc"], "HttpEtag": None, "HttpLastModified": None})
        n += 1
    ctx.counts["download"]["reconciledLegacyDocuments"] = n
    return n


def write_period_attempts(ctx: RunContext, listing: ListingResult, notice_index: dict | None, today: date) -> None:
    """One listed/not-listed attempt per (period, consolidation) cell; backfill writes all cells, incremental only
    gaps and the last 8 periods. Never claims 'not-listed' when the listing was incomplete."""
    listed: dict[tuple[str, str], list[dict]] = {}
    for it in listing.items:
        if in_scope(it) and not it["classification"]["restatement"]:
            listed.setdefault((it["classification"]["reportingPeriod"], doc_consolidation(it)), []).append(it)
    periods = expected_periods(today)
    if listing.covered_from is not None:
        periods = [p for p in periods if ed.period_end(p) > listing.covered_from]  # same end date (Q4 vs FY, Q2 vs H1) may sit on the next page
    recent = set(periods[-8:])
    checked = {"BID": "BIDV VN library bao-cao-va-tai-lieu (all items, single page) + disclosure feed thong-tin-co-dong",
               "MBB": "MB GetListFinance categories 7 (consolidated) and 13 (separate), all pages, year=0"}[ctx.symbol]
    stats = {"listed": 0, "notListed": 0, "notYetDue": 0}
    for p in periods:
        for cons in ("consolidated", "separate"):
            hits = listed.get((p, cons), [])
            if hits:
                stats["listed"] += 1
                if ctx.mode == "backfill" or p in recent:
                    _attempt(ctx, hits[0], "listed", source="pipeline-coverage", period=p, cons=cons,
                             detail=f"{len(hits)} listing item(s): " + "; ".join(h["title"][:120] for h in hits))
                continue
            if not listing.complete:
                continue
            if today <= due_date(p):
                stats["notYetDue"] += 1
                _attempt(ctx, None, "not-listed", source="pipeline-coverage", period=p, cons=cons, url=None,
                         detail=f"not yet due (statutory window ends ~{due_date(p)}); checked {checked}")
                continue
            stats["notListed"] += 1
            parts = [f"checked {checked}"]
            known = KNOWN_GAP_EVIDENCE.get((ctx.symbol, p, cons))
            if known:
                parts.append(known)
            for u in listing.unstorable:
                uc = u["classification"]
                if uc["reportingPeriod"] == p and cons in (uc["covers"] or [doc_consolidation(u)]):
                    parts.append(f"listed but not storable: {skip_reason(u)} ({(u.get('url') or 'no file')[:200]})")
            for n in (notice_index or {}).get((p, cons), [])[:3]:
                parts.append(f"MB notice {n['id']} ({n['createdLocal']}): '{n['title'][:100]}' announces this statement")
            _attempt(ctx, None, "not-listed", source="pipeline-coverage", period=p, cons=cons, url=None,
                     detail=" | ".join(parts))
    ctx.counts["discover"]["periodCells"] = stats


# ============================================================================================
# MB notices (issuer-stated publication dates) + availability
# ============================================================================================

NOTICE_SOURCE = "mb-ir-notices"


def scan_mb_notices(ctx: RunContext, http: ed.PoliteHttp, cap: int, today: date) -> dict[tuple[str, str], list[dict]]:
    """Bounded, resumable scan of GetListMessage/{page}/{year}. Each fetched page is an attempt whose Detail holds
    the FS notices it contained (the checkpoint). Past years are immutable: pages already recorded are not
    re-fetched; the current year's page 1 is always re-read (new notices land there)."""
    index: dict[tuple[str, str], list[dict]] = {}
    done: dict[str, dict] = {}
    for a in ctx.repo.attempts_for_source(ctx.symbol, NOTICE_SOURCE, "listed"):
        with contextlib.suppress(Exception):
            done[a["Url"]] = json.loads(a["Detail"])

    def add(notices: list[dict]) -> None:
        for n in notices:
            for p, cons in ed.classify_notice(n["title"]):
                if n.get("createdLocal"):
                    index.setdefault((p, cons), []).append(n)

    for d in done.values():
        add(d.get("fs", []))
    years = [today.year] if ctx.mode == "incremental" else list(range(today.year, SCOPE_FROM_YEAR - 1, -1))
    if ctx.mode == "incremental" and today.month <= 4:
        years.append(today.year - 1)
    start_used = http.used
    api = MbApi(http, ctx, ed.MB_NOTICE_PAGE.format(year=today.year), "notices")
    stats = {"pagesFetched": 0, "pagesFromCheckpoint": 0, "errors": 0, "fsNotices": 0}
    for y in years:
        page, total = 1, None
        while total is None or page <= total:
            url = f"{ed.MB_ORIGIN}/api/GetListMessage/{page}/{y}"
            fresh_needed = (y == today.year and page == 1) or url not in done
            if not fresh_needed:
                stats["pagesFromCheckpoint"] += 1
                total = done[url].get("totalPage", total or 1)
                page += 1
                continue
            if http.used - start_used >= cap:
                ctx.counts["notices"] = {**ctx.counts["notices"], **stats, "capReached": True}
                _attempt(ctx, None, "budget-exhausted", source=NOTICE_SOURCE, url=url, doc_type=None,
                         detail=f"notice scan cap {cap} reached; resumes from this page next run")
                ctx.problem(f"MB notice scan cap {cap} reached")
                return index
            r = api.get(url)
            if r["outcome"] != "ok":
                stats["errors"] += 1
                _attempt(ctx, None, r["outcome"], source=NOTICE_SOURCE, url=url, status=r.get("status"), doc_type=None,
                         detail=f"notice page failed: {r.get('detail')}")
                ctx.problem(f"MB notices {y} p{page}: {r['outcome']}")
                break
            try:
                notices, total = ed.parse_mb_notices(r["resp"].json())
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                _attempt(ctx, None, "parse-error", source=NOTICE_SOURCE, url=url, status=r["status"], doc_type=None,
                         detail=f"JSON parse failed: {exc}")
                break
            fs = [n for n in notices if ed.classify_notice(n["title"])]
            stats["pagesFetched"] += 1
            stats["fsNotices"] += len(fs)
            add(fs)
            detail = json.dumps({"totalPage": total, "items": len(notices),
                                 "fs": [{"id": n["id"], "createdLocal": n["createdLocal"], "title": n["title"][:110]} for n in fs]},
                                ensure_ascii=False)
            if len(detail) > 2000:
                detail = json.dumps({"totalPage": total, "items": len(notices),
                                     "fs": [{"id": n["id"], "createdLocal": n["createdLocal"], "title": n["title"][:50]} for n in fs]},
                                    ensure_ascii=False)[:2000]
            _attempt(ctx, None, "listed", source=NOTICE_SOURCE, url=url, status=r["status"], doc_type=None, detail=detail)
            page += 1
            if ctx.mode == "incremental":
                break  # page 1 only
    ctx.counts["notices"] = {**ctx.counts["notices"], **stats}
    return index


def fill_availability(ctx: RunContext, notice_index: dict | None) -> None:
    """Fill PublishedAtUtc once (only when NULL) from an issuer-stated date: BID listing date; MB notice
    `created` (ICT) or server Last-Modified (post-2019-07 uploads, plausible window). Never invents."""
    stats: dict[str, int] = {}
    for d in ctx.repo.docs(ctx.symbol):
        if d["PublishedAtUtc"] is not None or not d["ReportingPeriod"]:
            continue
        end = ed.period_end(d["ReportingPeriod"])
        window = (end, end + timedelta(days=AVAILABILITY_WINDOW_DAYS))
        prov, pub = None, None
        ej = d.get("ExtractionJson") or {}
        if isinstance(ej, str):
            ej = json.loads(ej)
        listing = ej.get("listing") or {}
        if ctx.symbol == "BID" and listing.get("listedDate"):
            pub = published_from_listing_date(listing["listedDate"])
            prov = {"basis": "issuer IR listing publishdate (date only, ICT) -> 00:00 ICT next day", "statedDate": listing["listedDate"]}
        if pub is None and ctx.symbol == "MBB":
            cands = []
            covers = [d["Consolidation"]] if d["Consolidation"] in ("consolidated", "separate") else ["consolidated", "separate"]
            for cons in covers:
                for n in (notice_index or {}).get((d["ReportingPeriod"], cons), []):
                    t = ict_local_to_utc(n["createdLocal"])
                    if window[0] <= t <= window[1]:
                        cands.append((t, n))
            if cands:
                t, n = min(cands, key=lambda x: x[0])
                pub, prov = t, {"basis": "MB investor notice created time (ICT)", "noticeId": n["id"],
                                "noticeCreatedLocal": n["createdLocal"], "noticeTitle": n["title"][:200]}
            else:
                lm = ((ej.get("http") or {}).get("lastModified")) or _location_lastmod(ctx, d)
                if lm:
                    with contextlib.suppress(Exception):
                        t = parsedate_to_datetime(lm).astimezone(UTC)
                        if t >= MB_LASTMOD_MIN and window[0] <= t <= window[1]:
                            pub, prov = t, {"basis": "MB server Last-Modified (upload time; accepted only after the "
                                                     "2019-07-03 server migration and within 150 days of period end)",
                                            "httpLastModified": lm}
                        else:
                            stats["lastModifiedRejected"] = stats.get("lastModifiedRejected", 0) + 1
        if pub is None:
            stats["noStatedDate"] = stats.get("noStatedDate", 0) + 1
            continue
        if not ctx.dry_run and ctx.repo.fill_published(d["Id"], pub, prov):
            stats[prov["basis"].split(" (")[0]] = stats.get(prov["basis"].split(" (")[0], 0) + 1
    ctx.counts["availability"] = stats


def _location_lastmod(ctx: RunContext, d: dict) -> str | None:
    loc = ctx.repo.location_by_url(d["SourceUrl"])
    return (loc or {}).get("HttpLastModified")


# ============================================================================================
# Extraction seam (worker B: ai/equity_facts)
# ============================================================================================

def load_extractor() -> tuple[Any | None, str | None]:
    try:
        import equity_facts  # type: ignore
    except ImportError as exc:
        return None, f"ai/equity_facts not importable ({exc})"
    for name in ("EXTRACTOR_VERSION", "extract_document", "derive_and_cross_validate"):
        if not hasattr(equity_facts, name):
            return None, f"ai/equity_facts lacks {name}"
    return equity_facts, None


def run_extraction(ctx: RunContext, extractor: Any, root: Path, max_docs: int | None, scope: list[str]) -> None:
    repo, version = ctx.repo, extractor.EXTRACTOR_VERSION
    todo = repo.docs_needing_extraction(ctx.symbol, version, scope)
    ctx.counts["extract"].update({"extractorVersion": version, "pending": len(todo)})
    for i, doc in enumerate(todo):
        if max_docs is not None and i >= max_docs:
            ctx.counts["extract"]["deferred"] = len(todo) - i
            ctx.problem(f"extraction: max-docs {max_docs} reached, {len(todo) - i} deferred")
            break
        ctx.stage("extract", documentId=str(doc["Id"]), period=doc["ReportingPeriod"], index=i + 1, of=len(todo))
        started = ctx.clock()
        try:
            with repo.tx():
                # a failed row of this version is retried; it comes back via rollback + re-record if it fails again
                repo.clear_failed_extraction(doc["Id"], version)
                res = extractor.extract_document(repo.conn, dict(doc), root, run_id=str(ctx.run_id) if ctx.run_id else None,
                                                 heartbeat=ctx.beat)
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop the run
            err = f"{type(exc).__name__}: {exc}"
            ctx.inc("extract", "failed")
            ctx.problem(f"extract {doc['ReportingPeriod']} {doc['Consolidation']} failed: {err}")
            log.debug(traceback.format_exc())
            with contextlib.suppress(Exception), repo.tx():
                repo.record_failed_extraction(doc["Id"], version, err, started, ctx.clock())
                _attempt(ctx, None, "parse-error", source="pipeline-extract", period=doc["ReportingPeriod"],
                         cons=doc["Consolidation"], url=doc["SourceUrl"], detail=f"extraction {version} failed: {err}"[:2000])
            continue
        res = res or {}
        if res.get("skipped"):
            ctx.inc("extract", "skipped")
            continue
        ctx.inc("extract", f"status:{res.get('status', 'unknown')}")
        for k in ("facts", "validated", "reviewNeeded", "quarantined", "pagesOcr"):
            if isinstance(res.get(k), int):
                ctx.inc("extract", k, res[k])
        if res.get("status") == "failed":
            ctx.problem(f"extract {doc['ReportingPeriod']} {doc['Consolidation']}: status failed")


# ============================================================================================
# Run
# ============================================================================================

def run(args: argparse.Namespace, *, repo: Any, http: ed.PoliteHttp, root: Path, extractor_loader=load_extractor,
        clock: Callable[[], datetime] = utcnow) -> int:
    symbol, mode = args.symbol, args.mode
    if not repo.try_lock(symbol):
        log.error("busy: advisory lock equity-docs:%s held by another run", symbol)
        return EXIT_BUSY
    run_id: uuid.UUID | None = None
    ctx: RunContext | None = None
    try:
        if not args.dry_run:
            if args.run_id:
                run_id = uuid.UUID(args.run_id)
                row = repo.run_get(run_id)
                if row is None or row["Status"] != "running" or row["Symbol"] != symbol:
                    log.error("run %s not found / not running / wrong symbol", run_id)
                    return EXIT_FAILED
            else:
                stale = repo.run_fail_stale(symbol, clock())
                if stale:
                    log.warning("marked %d stale running run(s) failed", stale)
                run_id = uuid.uuid4()
                try:
                    repo.run_create(run_id, symbol, mode, "cli", clock())
                except RunBusy as exc:
                    log.error("busy: %s", exc)
                    return EXIT_BUSY
        ctx = RunContext(repo, run_id, symbol, mode, args.dry_run, clock)
        ctx.start_heartbeat()
        status, error = _run_stages(ctx, args, http, root, extractor_loader)
    except Exception as exc:  # noqa: BLE001
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        log.error("run failed: %s\n%s", error, traceback.format_exc())
    finally:
        if ctx is not None:
            ctx.counts["requests"] = {"total": http.used, "budget": http.budget, "retries": http.retries,
                                      "byOutcome": dict(http.by_outcome)}
            ctx.stop()
        try:
            repo.unlock(symbol)
        except Exception:  # noqa: BLE001
            pass
    assert ctx is not None
    if ctx.problems and status == "succeeded":
        status = "partial"
        error = "; ".join(ctx.problems)[:2000]
    if not args.dry_run and run_id is not None:
        repo.run_finish(run_id, status, ctx.counts, ctx.checkpoint, error, ctx.log_tail())
    summary = {"runId": str(run_id) if run_id else None, "status": status, "counts": ctx.counts, "problems": ctx.problems[:20]}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return {"succeeded": EXIT_OK, "partial": EXIT_PARTIAL}.get(status, EXIT_FAILED)


def _run_stages(ctx: RunContext, args: argparse.Namespace, http: ed.PoliteHttp, root: Path, extractor_loader) -> tuple[str, str | None]:
    today = ctx.clock().date()
    for stale in (root / ctx.symbol).glob(".partial-*"):
        stale.unlink(missing_ok=True)
    ctx.stage("reconcile")
    reconcile_existing(ctx)

    ctx.stage("discover")
    listing = None
    try:
        listing = discover_bid(ctx, http) if ctx.symbol == "BID" else discover_mb(ctx, http)
    except ed.BudgetExceeded as exc:
        _attempt(ctx, None, "budget-exhausted", source="pipeline-discover", doc_type=None, detail=str(exc))
        ctx.problem(f"discover: {exc}")
        return "succeeded", None
    if not listing.complete:
        ctx.problem("listing incomplete: " + "; ".join(listing.errors))
    targets, skipped = [], {}
    for it in listing.items:
        why = skip_reason(it)
        if why is None:
            targets.append(it)
            continue
        skipped[why] = skipped.get(why, 0) + 1
        if item_year(it) < SCOPE_FROM_YEAR:
            continue  # pre-2016 archive items (incl. BIDV 2008-2012 .zip, MB 2013 .rar) are only counted
        if it["classification"]["isFinancialStatement"]:
            # An in-scope statement we cannot store (non-PDF file, no file attached): a real coverage loss. It gets a
            # failure-type attempt carrying period/consolidation (every run) and makes the run partial.
            outcome = "not-listed" if why.startswith("listing item has no file") else "parse-error"
            _attempt(ctx, it, outcome, detail=f"in-scope statement listed but not stored: {why}; title '{it['title'][:150]}'")
            listing.unstorable.append(it)
            if not it["classification"]["restatement"] or it.get("url"):
                ctx.problem(f"in-scope statement not stored ({why}): {it['classification']['reportingPeriod']} {doc_consolidation(it)}")
        else:
            # not a statement (e.g. BIDV profit-variance letters): recorded so coverage can show why it was skipped
            _attempt(ctx, it, "listed", doc_type="other", detail=f"skipped: {why}; title '{it['title'][:150]}'")
    ctx.counts["discover"].update({"items": len(listing.items), "targets": len(targets), "skipped": skipped,
                                   "complete": listing.complete})
    order = priority_order(targets)

    head_check = head_check_due(ctx, args.head_check)
    ctx.stage("download", targets=len(order), headCheck=head_check)
    fetched = 0
    stop_reason = None
    for idx, it in enumerate(order):
        if stop_reason:
            ctx.inc("download", "deferred")
            continue
        loc = ctx.repo.location_by_url(it["url"]) if args.max_docs is not None else None
        if args.max_docs is not None and fetched >= args.max_docs and loc is None:
            stop_reason = f"max-docs {args.max_docs} reached"
            ctx.inc("download", "deferred")
            continue
        try:
            out = process_target(ctx, http, root, it, head_check=head_check, max_bytes=args.max_bytes)
        except ed.BudgetExceeded as exc:
            stop_reason = f"request budget exhausted ({exc})"
            ctx.inc("download", "deferred")
            _attempt(ctx, it, "budget-exhausted", detail=f"{exc}; this and later targets deferred to the next run")
            continue
        if out not in ("skipped-unchanged",):
            fetched += 1
        if idx % 10 == 0:
            ctx.beat()
    if stop_reason:
        if "max-docs" in stop_reason and not ctx.dry_run:
            _attempt(ctx, None, "budget-exhausted", source="pipeline-download", doc_type=None,
                     detail=f"{stop_reason}; {ctx.counts['download'].get('deferred', 0)} targets deferred")
        ctx.problem(f"download stopped: {stop_reason}; deferred {ctx.counts['download'].get('deferred', 0)}")

    notice_index = None
    if ctx.symbol == "MBB" and not args.skip_notices and not ctx.dry_run:
        ctx.stage("notices")
        cap = min(args.notice_budget, max(0, http.budget - http.used))
        try:
            notice_index = scan_mb_notices(ctx, http, cap, today)
        except ed.BudgetExceeded as exc:
            ctx.problem(f"notices: {exc}")

    if not ctx.dry_run:
        ctx.stage("coverage-attempts")
        write_period_attempts(ctx, listing, notice_index, today)
        ctx.stage("availability")
        fill_availability(ctx, notice_index)

    if args.no_extract or ctx.dry_run:
        ctx.counts["extract"]["disabled"] = True
    else:
        extractor, why = extractor_loader()
        if extractor is None:
            ctx.counts["extract"]["unavailable"] = why
            ctx.problem(f"extraction unavailable: {why}")
        else:
            ctx.stage("extract")
            run_extraction(ctx, extractor, root, args.max_docs, ["consolidated", "unknown"] if args.extract_scope == "consolidated"
                           else ["consolidated", "separate", "unknown"])
            ctx.stage("derive")
            try:
                with ctx.repo.tx():
                    res = extractor.derive_and_cross_validate(ctx.repo.conn, ctx.symbol,
                                                              run_id=str(ctx.run_id) if ctx.run_id else None)
                ctx.counts["derive"] = res or {}
            except Exception as exc:  # noqa: BLE001
                ctx.problem(f"derive failed: {type(exc).__name__}: {exc}")
    ctx.stage("done")
    return "succeeded", None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--symbol", required=True, choices=["BID", "MBB"])
    r.add_argument("--mode", required=True, choices=["backfill", "incremental"])
    r.add_argument("--run-id")
    r.add_argument("--budget", type=int, help="max HTTP requests this run (default backfill 450, incremental 60)")
    r.add_argument("--max-docs", type=int, help="max documents fetched (and extracted) this run")
    r.add_argument("--dry-run", action="store_true", help="fetch listings and print the plan; no DB writes, no downloads")
    r.add_argument("--no-extract", action="store_true", help="skip extraction/derive explicitly (not counted as partial)")
    r.add_argument("--extract-scope", choices=["consolidated", "all"], default="consolidated")
    r.add_argument("--head-check", action="store_true", help="MB: HEAD stored files of the last 2 fiscal years for in-place overwrites")
    r.add_argument("--notice-budget", type=int, default=DEFAULT_NOTICE_BUDGET)
    r.add_argument("--skip-notices", action="store_true")
    r.add_argument("--max-bytes", type=int, default=ed.MAX_DOCUMENT_BYTES)
    r.add_argument("--delay", type=float, default=ed.DEFAULT_DELAY_S)
    r.add_argument("--root", type=Path, default=None)
    return ap


FIXTURE_ENV = "EQUITY_PIPELINE_FIXTURE_ORIGIN"
FIXTURE_FORBIDDEN_DBS = {"bitcoin_analyst"}


def apply_fixture_origin(origin: str, db_name: str) -> list[str]:
    """TEST-ONLY seam (drills against a local fixture server). Unset in production, where nothing changes.
    Redirects the issuer listing roots to http://127.0.0.1:<port> (loopback only) and refuses production DBs."""
    p = urlparse(origin)
    if p.scheme != "http" or p.hostname not in ("127.0.0.1", "localhost") or not p.port or p.path not in ("", "/"):
        raise ValueError(f"{FIXTURE_ENV} must be http://127.0.0.1:<port>")
    if db_name in FIXTURE_FORBIDDEN_DBS:
        raise ValueError(f"{FIXTURE_ENV} refused for database {db_name}")
    base = f"http://{p.hostname}:{p.port}"
    ed.BIDV_LIBRARY_VI = f"{base}/vn/quan-he-nha-dau-tu/bao-cao-va-tai-lieu/"
    ed.BIDV_DISCLOSURE_FEED = f"{base}/vn/quan-he-nha-dau-tu/thong-tin-co-dong/"
    ed.MB_ORIGIN = base
    ed.MB_FS_PAGE = f"{base}/Investor/bao-cao-tai-chinh"
    ed.MB_NOTICE_PAGE = f"{base}/Investor/thong-bao-nha-dau-tu/{{year}}/0//0"
    return [p.hostname]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_FAILED if exc.code else EXIT_OK
    if not args.run_id and os.environ.get("EQUITY_PIPELINE_RUN_ID"):
        args.run_id = os.environ["EQUITY_PIPELINE_RUN_ID"]  # the backend runner also passes --run-id
    if args.run_id:
        try:
            uuid.UUID(args.run_id)
        except ValueError:
            log.error("--run-id is not a GUID")
            return EXIT_FAILED
    budget = args.budget if args.budget is not None else DEFAULT_BUDGET[args.mode]
    root = args.root or default_root()
    hosts, loopback = ISSUER[args.symbol]["hosts"], False
    fixture = os.environ.get(FIXTURE_ENV)
    if fixture:
        try:
            hosts, loopback = apply_fixture_origin(fixture, os.environ.get("DB_NAME") or ""), True
        except ValueError as exc:
            log.error("%s", exc)
            return EXIT_FAILED
        log.warning("TEST FIXTURE ORIGIN %s: issuer listing/document URLs are redirected", fixture)
    try:
        repo = PgRepo(PgRepo.params_from_env())
    except SystemExit as exc:
        log.error("%s", exc)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001
        log.error("database connection failed: %s", type(exc).__name__)
        return EXIT_FAILED
    http = ed.PoliteHttp(hosts, budget, args.delay, loopback_http=loopback)
    try:
        return run(args, repo=repo, http=http, root=root)
    finally:
        repo.close()


if __name__ == "__main__":
    raise SystemExit(main())
