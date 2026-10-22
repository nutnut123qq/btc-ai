"""Offline tests for ai/equity_pipeline.py (v3 orchestrator): in-memory repository, fake HTTP, synthetic fixtures."""

from __future__ import annotations

import contextlib
import copy
import json
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import equity_documents as ed  # noqa: E402
import equity_pipeline as ep  # noqa: E402

NOW = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------------------------
# fakes


class Repo:
    """In-memory twin of PgRepo (same method surface). tx() snapshots state and restores it on any exception,
    including BaseException (a simulated kill), like a PostgreSQL rollback."""

    def __init__(self):
        self.store: dict = {}
        self.locs: list[dict] = []
        self.attempts: list[dict] = []
        self.runs: dict = {}
        self.extractions: list[dict] = []
        self.held_elsewhere: set[str] = set()
        self.locked: set[str] = set()
        self.conn = object()
        self._in_tx = False
        self.fail_hook = None  # callable(op, payload): raise to simulate a crash

    @contextlib.contextmanager
    def tx(self):
        assert not self._in_tx
        snap = copy.deepcopy((self.store, self.locs, self.attempts, self.extractions))
        self._in_tx = True
        try:
            yield
        except BaseException:
            self.store, self.locs, self.attempts, self.extractions = snap
            raise
        finally:
            self._in_tx = False

    # lock / runs
    def try_lock(self, symbol):
        if symbol in self.held_elsewhere or symbol in self.locked:
            return False
        self.locked.add(symbol)
        return True

    def unlock(self, symbol):
        self.locked.discard(symbol)

    def run_get(self, run_id):
        return self.runs.get(run_id)

    def run_fail_stale(self, symbol, now):
        n = 0
        for r in self.runs.values():
            if (r["Symbol"] == symbol and r["Status"] == "running"
                    and r["HeartbeatAtUtc"] < now - timedelta(minutes=ep.CLI_STALE_MINUTES)):
                r.update(Status="failed", FinishedAtUtc=now, Error="stale-heartbeat")
                n += 1
        return n

    def run_create(self, run_id, symbol, mode, trigger, now):
        if any(r["Symbol"] == symbol and r["Status"] == "running" for r in self.runs.values()):
            raise ep.RunBusy("running row exists")
        self.runs[run_id] = {"Id": run_id, "Symbol": symbol, "Mode": mode, "Trigger": trigger, "Status": "running",
                             "StartedAtUtc": now, "HeartbeatAtUtc": now, "FinishedAtUtc": None, "CountsJson": {}}

    def run_heartbeat(self, run_id, counts, checkpoint, tail):
        r = self.runs[run_id]
        if r["Status"] == "running":
            r.update(CountsJson=counts, CheckpointJson=checkpoint, LogTail=tail)

    def run_finish(self, run_id, status, counts, checkpoint, error, tail):
        self.runs[run_id].update(Status=status, CountsJson=counts, CheckpointJson=checkpoint, Error=error,
                                 FinishedAtUtc=NOW, LogTail=tail)

    # attempts
    def attempt(self, a):
        if self.fail_hook:
            self.fail_hook("attempt", a)
        a = dict(a, Id=len(self.attempts) + 1)
        self.attempts.append(a)
        return a["Id"]

    def last_attempt_at(self, symbol, source):
        ts = [a["AttemptedAtUtc"] for a in self.attempts if a["Symbol"] == symbol and a["Source"] == source]
        return max(ts) if ts else None

    def attempts_for_source(self, symbol, source, outcome):
        return [a for a in self.attempts if a["Symbol"] == symbol and a["Source"] == source and a["Outcome"] == outcome]

    # documents / locations
    def location_by_url(self, url):
        hits = [{**l, "DocByteSize": self.store[l["DocumentId"]]["ByteSize"],
                 "DocRevision": self.store[l["DocumentId"]]["Revision"],
                 "DocSha": self.store[l["DocumentId"]]["ContentSha256"]} for l in self.locs if l["SourceUrl"] == url]
        return max(hits, key=lambda l: l["DocRevision"]) if hits else None

    def latest_doc_for_identity(self, symbol, identity):
        if identity.startswith("bidv-wcm:"):
            frag = f"/connect/{identity.split(':', 1)[1]}/"
            ids = {l["DocumentId"] for l in self.locs if frag in l["SourceUrl"].lower()}
        else:
            ids = {l["DocumentId"] for l in self.locs if l["SourceUrl"] == identity[4:]}
        docs = [self.store[i] for i in ids if self.store[i]["Symbol"] == symbol]
        return max(docs, key=lambda d: d["Revision"]) if docs else None

    def doc_by_sha(self, symbol, sha):
        return next((d for d in self.store.values() if d["Symbol"] == symbol and d["ContentSha256"] == sha), None)

    def docs_without_locations(self, symbol):
        with_loc = {l["DocumentId"] for l in self.locs}
        return [d for d in self.store.values() if d["Symbol"] == symbol and d["Id"] not in with_loc]

    def docs(self, symbol):
        return [d for d in self.store.values() if d["Symbol"] == symbol]

    def insert_document(self, d):
        if self.fail_hook:
            self.fail_hook("insert_document", d)
        assert self.doc_by_sha(d["Symbol"], d["ContentSha256"]) is None, "unique (Symbol, ContentSha256)"
        self.store[d["Id"]] = copy.deepcopy(d)

    def upsert_location(self, loc):
        if self.fail_hook:
            self.fail_hook("upsert_location", loc)
        assert loc["DocumentId"] in self.store, "FK"
        for l in self.locs:
            if l["DocumentId"] == loc["DocumentId"] and l["SourceUrl"] == loc["SourceUrl"]:
                l["LastSeenAtUtc"] = max(l["LastSeenAtUtc"], loc["LastSeenAtUtc"])
                for k in ("LastFetchedAtUtc", "HttpEtag", "HttpLastModified"):
                    if loc.get(k) is not None:
                        l[k] = loc[k]
                for k in ("ListingTitle", "ListingDate"):
                    l[k] = l.get(k) or loc.get(k)
                return
        self.locs.append(dict(loc, Id=len(self.locs) + 1))

    def fill_published(self, doc_id, published, provenance):
        d = self.store[doc_id]
        if d["PublishedAtUtc"] is not None:
            return False
        d["PublishedAtUtc"] = published
        d["ExtractionJson"] = {**(d["ExtractionJson"] or {}), "availability": provenance}
        return True

    # extraction
    def docs_needing_extraction(self, symbol, version, consolidations):
        done = {e["DocumentId"] for e in self.extractions
                if e["ExtractorVersion"] == version and e["Status"] in ("succeeded", "partial")}
        return [d for d in self.docs(symbol) if d["DocType"] == "financial-statement"
                and d["Consolidation"] in consolidations and d["Id"] not in done]

    def clear_failed_extraction(self, doc_id, version):
        self.extractions = [e for e in self.extractions
                            if not (e["DocumentId"] == doc_id and e["ExtractorVersion"] == version and e["Status"] == "failed")]

    def record_failed_extraction(self, doc_id, version, error, started, now):
        self.extractions.append({"DocumentId": doc_id, "ExtractorVersion": version, "Status": "failed", "Error": error})


class FakeResp:
    def __init__(self, url, body=b"", status=200, headers=None):
        self.url, self.status_code = url, status
        self._body = body if isinstance(body, bytes) else body.encode("utf-8")
        self.headers = {"Content-Type": "application/pdf" if self._body.startswith(b"%PDF-") else "text/html",
                        **(headers or {})}

    @property
    def text(self):
        return self._body.decode("utf-8")

    def json(self):
        return json.loads(self._body)

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]


class FakeCookies(dict):
    cleared = 0

    def clear(self):
        FakeCookies.cleared += 1
        super().clear()


class FakeSession:
    """routes: url -> list of FakeResp (consumed in order, last one repeats)."""

    def __init__(self, routes):
        self.routes = {k: list(v) if isinstance(v, list) else [v] for k, v in routes.items()}
        self.headers = {}
        self.cookies = FakeCookies()
        self.calls: list[tuple[str, str]] = []

    def request(self, method, url, **kw):
        self.calls.append((method, url))
        if url not in self.routes:
            return FakeResp(url, b"not found", 404)
        q = self.routes[url]
        r = q.pop(0) if len(q) > 1 else q[0]
        if method == "HEAD":
            return FakeResp(r.url, b"", r.status_code, {k: v for k, v in r.headers.items()})
        return r


def make_http(routes, symbol="BID", budget=500):
    sleeps = []
    http = ed.PoliteHttp(ep.ISSUER[symbol]["hosts"], budget, delay_s=0, session=FakeSession(routes),
                         sleep=sleeps.append, backoff=(3, 6, 12))
    return http, sleeps


def args(*extra, symbol="BID", mode="backfill"):
    return ep.build_parser().parse_args(["run", "--symbol", symbol, "--mode", mode, "--no-extract", *extra])


def pdf(tag: str, size: int = 2048) -> bytes:
    body = f"%PDF-1.4 {tag} ".encode()
    return body + b"x" * (size - len(body))


def mb_size(b: bytes) -> str:
    return f"{len(b) / 2**20:.2f} MB"


# ---------------------------------------------------------------------------------------------
# fixtures


def bidv_item(title, date_, href, size, group="2026", path="/bidv/quan-he-nha-dau-tu/bao-cao-va-tai-lieu/baocaotaichinh/2026/x"):
    return ("var item = {\n\ttype_pdf: 'File PDF' === 'File PDF',\n\tparenttitle: formatTitle('%s'),\n"
            "parentname: formatTitle('%s'),\n\ttitle: formatTitle('%s'),\n\tpublishdate: '%s',\n"
            "\tfile_title: formatTitle('%s'),\n\tfile_size: '%s',\n\tpath: '%s'\n};\n") % (group, group, title, date_, href, size, path)


def wcm(uuid_: str, name: str, token: str) -> str:
    return f"/wps/wcm/connect/{uuid_}/{name}?MOD=AJPERES&amp;CACHEID=ROOTWORKSPACE-{uuid_}-{token}"


U1, U2, U3, U4 = ("11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222",
                  "33333333-3333-3333-3333-333333333333", "44444444-4444-4444-4444-444444444444")


def bid_world(token1="tokA", body1=None, body2=None, extra_items="", sizes=None):
    body1 = body1 or pdf("bid-2026q2-cons")
    body2 = body2 or pdf("bid-2026q2-sep", 3072)
    lib = (bidv_item("Báo cáo tài chính hợp nhất giữa niên độ Quý II năm 2026", "30/07/2026", wcm(U1, "q2hn.pdf", token1),
                     (sizes or {}).get(1, "0.00 MB") if sizes else f"{len(body1) / 1024:.0f} KB")
           + bidv_item("Báo cáo tài chính riêng giữa niên độ Quý II năm 2026", "30/07/2026", wcm(U2, "q2rl.pdf", "tokB"),
                       f"{len(body2) / 1024:.0f} KB")
           + bidv_item("Giải trình biến động lợi nhuận sau thuế tại Báo cáo tài chính hợp nhất Quý II/2016", "26/08/2016",
                       wcm(U3, "gt.pdf", "tokC"), "278 KB", group="2016")
           + bidv_item("Báo cáo tài chính hợp nhất 2011", "31/12/2011", "/wps/wcm/connect/x/bctc2011.zip?MOD=A", "6 MB", group="Khac")
           + extra_items)
    feed = bidv_item("Điều chỉnh Báo cáo tài chính riêng lẻ và hợp nhất năm 2024 đã được kiểm toán", "23/10/2025",
                     wcm(U4, "dieuchinh.pdf", "tokD"), "2 KB", group="2025",
                     path="/bidv/quan-he-nha-dau-tu/thong-tin-co-dong/congbothongtin/2025/x")
    base = "https://bidv.com.vn"
    routes = {
        ed.BIDV_LIBRARY_VI: FakeResp(ed.BIDV_LIBRARY_VI, "<html>" + lib + "</html>"),
        ed.BIDV_DISCLOSURE_FEED: FakeResp(ed.BIDV_DISCLOSURE_FEED, "<html>" + feed + "</html>"),
        f"{base}{wcm(U1, 'q2hn.pdf', token1).replace('&amp;', '&')}": FakeResp("u1", body1),
        f"{base}{wcm(U2, 'q2rl.pdf', 'tokB').replace('&amp;', '&')}": FakeResp("u2", body2),
        f"{base}{wcm(U4, 'dieuchinh.pdf', 'tokD').replace('&amp;', '&')}": FakeResp("u4", pdf("restatement")),
    }
    for k, v in routes.items():
        v.url = k
    return routes


def run_once(repo, routes, tmp_path, *extra, symbol="BID", mode="backfill", budget=500, extractor_loader=None, clock=None):
    http, sleeps = make_http(routes, symbol, budget)
    code = ep.run(args(*extra, symbol=symbol, mode=mode), repo=repo, http=http, root=tmp_path,
                  extractor_loader=extractor_loader or (lambda: (None, "not installed")), clock=clock or (lambda: NOW))
    return code, http, sleeps


def last_run(repo):
    return max(repo.runs.values(), key=lambda r: r["StartedAtUtc"])


# ---------------------------------------------------------------------------------------------
# classification


@pytest.mark.parametrize("title, fname, cat, period, cons, audit, fs, restated", [
    ("Báo cáo tài chính hợp nhất giữa niên độ Quý II năm 2026", "", None, "2026Q2", "consolidated", "unknown", True, False),
    ("Báo cáo tài chính hợp nhất giữa niên độ đã được soát xét cho kỳ hoạt động 6 tháng kết thúc ngày 30/06/2026", "", None,
     "2026H1", "consolidated", "reviewed", True, False),
    ("Báo cáo tài chính riêng giữa niên độ đã được soát xét 2017", "", None, "2017H1", "separate", "reviewed", True, False),
    ("Báo cáo tài chính kiểm toán hợp nhất 2018", "", None, "2018", "consolidated", "audited", True, False),
    ("Báo cáo tài chính hợp nhất đã kiểm toán 2016", "", None, "2016", "consolidated", "audited", True, False),
    ("CBTT Báo cáo tài chính kiểm toán năm 2020", "BCTC+HN+kiem+toan+2020.pdf", None, "2020", "consolidated", "audited", True, False),
    ("Báo cáo tài chính hợp nhất Quý I.2016", "", None, "2016Q1", "consolidated", "unknown", True, False),
    ("Báo cáo tài chính riêng lẻ Quý IV/2021", "", None, "2021Q4", "separate", "unknown", True, False),
    ("Báo cáo tài chính riêng lẻ bán niên soát xét năm 2018", "", "consolidated", "2018H1", "separate", "reviewed", True, False),
    ("Báo cáo tài chính hợp nhất Quý 4.2025", "", None, "2025Q4", "consolidated", "unknown", True, False),
    ("Consolidated Financial Statements for 3rd Quarter 2018", "", None, "2018Q3", "consolidated", "unknown", True, False),
    ("Separate financial statements of the first half of 2022", "", None, "2022H1", "separate", "unknown", True, False),
    ("Giải trình biến động lợi nhuận sau thuế tại Báo cáo tài chính hợp nhất Quý II/2016", "", None, None, "unknown", "unknown", False, False),
    ("Điều chỉnh Báo cáo tài chính riêng lẻ và hợp nhất năm 2024 đã được kiểm toán", "", None, "2024", "unknown", "audited", True, True),
    ("CBTT bổ sung Báo cáo tài chính riêng lẻ và hợp nhất năm 2021 đã kiểm toán", "", None, "2021", "unknown", "audited", True, True),
    ("Bản tin nhà đầu tư", "", None, None, "unknown", "unknown", False, False),
])
def test_classify_v3(title, fname, cat, period, cons, audit, fs, restated):
    c = ed.classify_v3(title, fname, cat)
    assert (c["reportingPeriod"], c["consolidation"], c["auditStatus"], c["isFinancialStatement"], c["restatement"]) == \
        (period, cons, audit, fs, restated)
    if not fs:
        assert c["skipReason"]


def test_classify_basis_records_filename_and_category_fallbacks():
    assert ed.classify_v3("CBTT Báo cáo tài chính kiểm toán năm 2020", "BCTC+HN+kiem+toan+2020.pdf")["basis"] == {"consolidation": "filename"}
    assert ed.classify_v3("Báo cáo tài chính Quý 1 năm 2019", "x.pdf", "separate")["basis"] == {"consolidation": "listing category"}


def test_classify_notice_covers_both_when_title_names_both_or_neither():
    assert ed.classify_notice("MBB.CBTT Báo cáo tài chính hợp nhất & riêng lẻ Quý III năm 2018") == [
        ("2018Q3", "consolidated"), ("2018Q3", "separate")]
    assert ed.classify_notice("Công bố thông tin báo cáo tài chính MB Quý 4 năm 2016") == [("2016Q4", "consolidated"), ("2016Q4", "separate")]
    assert ed.classify_notice("UBCKNN chấp thuận gia hạn thời gian công bố Báo cáo tài chính") == []


@pytest.mark.parametrize("listed, size, ok", [("12.84 MB", int(12.84 * 2**20), True), ("12.84 MB", int(12.80 * 2**20), False),
                                              ("487 KB", 487 * 1024 + 100, True), ("0 MB", 5, None), (None, 5, None)])
def test_listed_size_matches(listed, size, ok):
    assert ed.listed_size_matches(listed, size) is ok


def test_parse_bidv_items_title_not_confused_with_parenttitle():
    items = ed.parse_bidv_items(bidv_item("Báo cáo tài chính hợp nhất Quý 1/2018", "29/04/2018", wcm(U1, "a.pdf", "t1"), "6.23 MB"),
                                ed.BIDV_LIBRARY_VI)
    assert items[0]["title"] == "Báo cáo tài chính hợp nhất Quý 1/2018"
    assert items[0]["identity"] == f"bidv-wcm:{U1}" and items[0]["versionToken"] == "t1"
    assert "&amp;" not in items[0]["url"] and items[0]["listedDate"] == "2018-04-29"


# ---------------------------------------------------------------------------------------------
# HTTP


def test_rate_limit_backoff_honours_retry_after_then_succeeds():
    url = "https://bidv.com.vn/x"
    http, sleeps = make_http({url: [FakeResp(url, "busy", 429, {"Retry-After": "20"}), FakeResp(url, "ok", 200)]})
    r = http.request(url)
    assert r["outcome"] == "ok" and r["attempts"] == 2 and http.used == 2 and http.retries == 1
    assert sleeps and max(sleeps) >= 20


def test_persistent_403_is_rate_limited_not_silent_success():
    url = "https://www.mbbank.com.vn/api/x"
    http, sleeps = make_http({url: FakeResp(url, "", 403)}, "MBB")
    r = http.request(url)
    assert r["outcome"] == "rate-limited" and r["status"] == 403 and r["attempts"] == 4
    assert sleeps == [3, 6, 12] and http.by_outcome == {"rate-limited": 1}


def test_budget_exceeded_raises():
    url = "https://bidv.com.vn/x"
    http, _ = make_http({url: FakeResp(url, "ok")}, budget=1)
    http.request(url)
    with pytest.raises(ed.BudgetExceeded):
        http.request(url)


def test_non_issuer_url_refused():
    http, _ = make_http({})
    with pytest.raises(ValueError):
        http.request("https://evil.example/x.pdf")


# ---------------------------------------------------------------------------------------------
# pipeline: discovery, download, versioning, idempotency


def test_backfill_downloads_in_priority_order_and_records_skips_and_coverage(tmp_path):
    repo = Repo()
    code, http, _ = run_once(repo, bid_world(), tmp_path)
    assert code == ep.EXIT_OK  # --no-extract, complete listing, every download ok
    docs = repo.docs("BID")
    assert {(d["ReportingPeriod"], d["Consolidation"]) for d in docs} == {("2026Q2", "consolidated"), ("2026Q2", "separate"),
                                                                          ("2024", "unknown")}
    restatement = next(d for d in docs if d["ReportingPeriod"] == "2024")
    assert restatement["ExtractionJson"]["restatement"]["isRestatement"] is True
    q2 = next(d for d in docs if d["Consolidation"] == "consolidated")
    assert q2["PublishedAtUtc"] == datetime(2026, 7, 30, 17, 0, tzinfo=UTC)  # 00:00 ICT next day
    assert q2["StoragePath"] == f"BID/{q2['ContentSha256']}.pdf" and (tmp_path / q2["StoragePath"]).read_bytes().startswith(b"%PDF-")
    pdf_calls = [u for m, u in http.session.calls if ".pdf" in u]
    assert "q2hn.pdf" in pdf_calls[0] and "q2rl.pdf" in pdf_calls[-1]  # consolidated first, separate last
    skipped = [a for a in repo.attempts if a.get("DocType") == "other"]
    assert len(skipped) == 1 and "explanation" in skipped[0]["Detail"]
    cells = [a for a in repo.attempts if a["Source"] == "pipeline-coverage"]
    assert any(a["TargetPeriod"] == "2026Q2" and a["Outcome"] == "listed" for a in cells)
    assert any(a["TargetPeriod"] == "2017Q4" and a["Consolidation"] == "separate" and a["Outcome"] == "not-listed"
               and "A0 2026-10-19" in a["Detail"] for a in cells)
    assert any(a["TargetPeriod"] == "2026Q3" and "not yet due" in a["Detail"] for a in cells)
    run = last_run(repo)
    assert run["Status"] == "succeeded" and run["Error"] is None  # --no-extract is not a problem
    assert run["CountsJson"]["download"]["downloaded"] == 3


def test_second_run_is_idempotent_zero_downloads_zero_duplicates(tmp_path):
    repo = Repo()
    run_once(repo, bid_world(), tmp_path)
    n_docs, n_locs = len(repo.docs("BID")), len(repo.locs)
    code, http, _ = run_once(repo, bid_world(), tmp_path, mode="incremental", clock=lambda: NOW + timedelta(hours=1))
    assert code == ep.EXIT_OK
    assert len(repo.docs("BID")) == n_docs and len(repo.locs) == n_locs
    assert not [u for m, u in http.session.calls if ".pdf" in u]
    assert last_run(repo)["CountsJson"]["download"].get("fetchAttempts", 0) == 0


def test_same_url_new_bytes_creates_new_revision_superseding_old(tmp_path):
    repo = Repo()
    run_once(repo, bid_world(), tmp_path)
    v1 = next(d for d in repo.docs("BID") if d["ReportingPeriod"] == "2026Q2" and d["Consolidation"] == "consolidated")
    new_body = pdf("bid-2026q2-cons-REISSUED", 5000)
    # same WCM item, new version token -> different URL, same identity
    code, _, _ = run_once(repo, bid_world(token1="tokNEW", body1=new_body), tmp_path, mode="incremental",
                          clock=lambda: NOW + timedelta(days=1))
    assert code == ep.EXIT_OK
    revs = sorted((d for d in repo.docs("BID") if d["ReportingPeriod"] == "2026Q2" and d["Consolidation"] == "consolidated"),
                  key=lambda d: d["Revision"])
    assert [d["Revision"] for d in revs] == [1, 2] and revs[1]["SupersedesDocumentId"] == v1["Id"]
    assert any(a["Outcome"] == "changed" for a in repo.attempts)


def test_mb_static_url_overwritten_in_place_detected_by_listed_size(tmp_path):
    repo = Repo()
    a, b = pdf("mb-q2-a", 100_000), pdf("mb-q2-b", 200_000)
    code, _, _ = run_once(repo, mb_world(a), tmp_path, "--skip-notices", symbol="MBB")
    assert code == ep.EXIT_OK
    code, _, _ = run_once(repo, mb_world(b), tmp_path, "--skip-notices", symbol="MBB", mode="incremental",
                          clock=lambda: NOW + timedelta(days=1))
    assert code == ep.EXIT_OK
    revs = sorted((d for d in repo.docs("MBB") if d["ReportingPeriod"] == "2026Q2"), key=lambda d: d["Revision"])
    assert [d["Revision"] for d in revs] == [1, 2] and revs[1]["SupersedesDocumentId"] == revs[0]["Id"]
    assert revs[0]["SourceUrl"] == revs[1]["SourceUrl"]


def test_same_hash_two_urls_one_document_two_locations(tmp_path):
    repo = Repo()
    same = pdf("identical")
    routes = bid_world(body1=same, body2=same)
    code, _, _ = run_once(repo, routes, tmp_path)
    q2 = [d for d in repo.docs("BID") if d["ReportingPeriod"] == "2026Q2"]
    assert len(q2) == 1
    assert len([l for l in repo.locs if l["DocumentId"] == q2[0]["Id"]]) == 2
    assert any("bytes identical" in (a["Detail"] or "") for a in repo.attempts)
    assert code == ep.EXIT_OK


def test_html_error_page_is_rejected_not_stored(tmp_path):
    repo = Repo()
    routes = bid_world()
    sep_url = next(k for k in routes if "q2rl.pdf" in k)
    routes[sep_url] = [FakeResp(sep_url, "<html>error404</html>", 200)]
    code, _, _ = run_once(repo, routes, tmp_path)
    assert code == ep.EXIT_PARTIAL
    assert not [d for d in repo.docs("BID") if d["Consolidation"] == "separate"]
    bad = [a for a in repo.attempts if a["Outcome"] == "parse-error"]
    assert bad and "not a PDF" in bad[0]["Detail"]
    assert last_run(repo)["CountsJson"]["download"]["notPdf"] == 1
    assert not list((tmp_path / "BID").glob(".partial-*"))


def test_too_large_rejected(tmp_path):
    repo = Repo()
    code, _, _ = run_once(repo, bid_world(), tmp_path, "--max-bytes", "1000")
    assert code == ep.EXIT_PARTIAL and not repo.docs("BID")
    assert {a["Outcome"] for a in repo.attempts if a["Source"] in ("bidv-ir-library", "bidv-ir-disclosure-feed") and a.get("TargetPeriod")} == {"too-large"}


def test_budget_exhaustion_is_recorded_and_partial(tmp_path):
    repo = Repo()
    code, http, _ = run_once(repo, bid_world(), tmp_path, budget=3)  # 2 listing pages + 1 document
    assert code == ep.EXIT_PARTIAL
    assert len(repo.docs("BID")) == 1
    ex = [a for a in repo.attempts if a["Outcome"] == "budget-exhausted"]
    assert ex and "deferred" in ex[0]["Detail"]
    run = last_run(repo)
    assert run["Status"] == "partial" and run["CountsJson"]["download"]["deferred"] == 2
    # next run resumes: only the deferred documents are fetched
    code, http, _ = run_once(repo, bid_world(), tmp_path, clock=lambda: NOW + timedelta(minutes=5))
    assert code == ep.EXIT_OK and len(repo.docs("BID")) == 3
    assert len([u for m, u in http.session.calls if ".pdf" in u]) == 2


def test_max_docs_defers_rest(tmp_path):
    repo = Repo()
    code, _, _ = run_once(repo, bid_world(), tmp_path, "--max-docs", "1")
    assert code == ep.EXIT_PARTIAL and len(repo.docs("BID")) == 1
    assert last_run(repo)["CountsJson"]["download"]["deferred"] == 2


def test_listing_failure_never_claims_not_listed(tmp_path):
    repo = Repo()
    routes = bid_world()
    routes[ed.BIDV_LIBRARY_VI] = [FakeResp(ed.BIDV_LIBRARY_VI, "", 503)]
    code, _, _ = run_once(repo, routes, tmp_path)
    assert code == ep.EXIT_PARTIAL
    assert not [a for a in repo.attempts if a["Outcome"] == "not-listed"]
    assert [a for a in repo.attempts if a["Source"] == "bidv-ir-library" and a["Outcome"] == "http-error"]


class SimulatedKill(BaseException):
    pass


def test_resume_after_crash_mid_document_leaves_no_partial_rows(tmp_path):
    repo = Repo()
    state = {"armed": True}

    def crash(op, payload):
        if op == "upsert_location" and "q2rl.pdf" in payload["SourceUrl"] and state["armed"]:
            state["armed"] = False
            raise SimulatedKill()

    repo.fail_hook = crash
    with pytest.raises(SimulatedKill):
        run_once(repo, bid_world(), tmp_path)
    # the consolidated doc (committed earlier) is kept; the separate doc's insert was rolled back with its location
    assert {d["Consolidation"] for d in repo.docs("BID")} == {"consolidated", "unknown"}
    assert all(l["DocumentId"] in {d["Id"] for d in repo.docs("BID")} for l in repo.locs)
    assert "BID" not in repo.locked  # lock released by finally
    killed = last_run(repo)
    assert killed["Status"] == "running"  # a real kill leaves the row; the next run marks it stale
    code, http, _ = run_once(repo, bid_world(), tmp_path, clock=lambda: NOW + timedelta(minutes=5))
    assert code == ep.EXIT_OK
    assert killed["Status"] == "failed" and killed["Error"] == "stale-heartbeat"
    assert sorted((d["ReportingPeriod"], d["Consolidation"]) for d in repo.docs("BID")) == [
        ("2024", "unknown"), ("2026Q2", "consolidated"), ("2026Q2", "separate")]
    assert len({d["ContentSha256"] for d in repo.docs("BID")}) == 3
    assert "q2hn.pdf" not in " ".join(u for m, u in http.session.calls)  # committed doc not re-fetched


def test_lock_busy_exits_75_without_touching_state(tmp_path):
    repo = Repo()
    repo.held_elsewhere.add("BID")
    code, http, _ = run_once(repo, bid_world(), tmp_path)
    assert code == ep.EXIT_BUSY and http.used == 0 and not repo.runs and not repo.attempts


def test_fresh_running_row_without_lock_holder_is_busy(tmp_path):
    repo = Repo()
    repo.runs[uuid.uuid4()] = {"Id": "x", "Symbol": "BID", "Status": "running", "StartedAtUtc": NOW, "HeartbeatAtUtc": NOW}
    code, http, _ = run_once(repo, bid_world(), tmp_path)
    assert code == ep.EXIT_BUSY and http.used == 0


def test_run_id_from_backend_is_used(tmp_path):
    repo = Repo()
    rid = uuid.uuid4()
    repo.runs[rid] = {"Id": rid, "Symbol": "BID", "Mode": "backfill", "Trigger": "admin", "Status": "running",
                      "StartedAtUtc": NOW, "HeartbeatAtUtc": NOW}
    code, _, _ = run_once(repo, bid_world(), tmp_path, "--run-id", str(rid))
    assert code == ep.EXIT_OK and repo.runs[rid]["Status"] == "succeeded" and len(repo.runs) == 1
    assert all(a["RunId"] == rid for a in repo.attempts)


def test_legacy_documents_are_reconciled_not_duplicated(tmp_path):
    repo = Repo()
    routes = bid_world()
    body = pdf("bid-2026q2-cons")
    url = next(k for k in routes if "q2hn.pdf" in k)
    import hashlib
    sha = hashlib.sha256(body).hexdigest()
    legacy_id = uuid.uuid4()
    repo.store[legacy_id] = {"Id": legacy_id, "Symbol": "BID", "DocType": "financial-statement", "ReportingPeriod": "2026Q2",
                             "Consolidation": "consolidated", "ContentSha256": sha, "ByteSize": len(body), "Revision": 1,
                             "SourceUrl": url, "Provider": "issuer:BIDV", "Title": "legacy", "PublishedAtUtc": None,
                             "FirstSeenAtUtc": NOW - timedelta(days=1), "FetchedAtUtc": NOW - timedelta(days=1),
                             "ExtractionJson": {"listing": {}}}
    code, http, _ = run_once(repo, routes, tmp_path)
    assert code == ep.EXIT_OK
    assert "q2hn.pdf" not in " ".join(u for m, u in http.session.calls)
    assert len([d for d in repo.docs("BID") if d["ReportingPeriod"] == "2026Q2" and d["Consolidation"] == "consolidated"]) == 1
    assert [l for l in repo.locs if l["DocumentId"] == legacy_id]


# ---------------------------------------------------------------------------------------------
# MB: pagination, notices, availability


def mb_world(q2_body, *, cat7_pages=2, notices=True):
    origin = ed.MB_ORIGIN
    page = FakeResp(ed.MB_FS_PAGE, '<input name="__RequestVerificationToken" type="hidden" value="tok123" />')
    routes = {ed.MB_FS_PAGE: page}

    def item(i, title, path, size):
        return {"id": i, "title": title, "file_path": path, "file_zise": size, "file_type": "pdf", "year": 2026,
                "last_Save_Date": "2018-05-23T00:00:00"}

    q2 = item(1, "Báo cáo tài chính hợp nhất Quý 2 năm 2026", "/resources/files/NhaDauTu/2026/BCTC/20260730---q2-hn.pdf", mb_size(q2_body))
    p1 = {"lst": [q2], "currentPageinfo": [{"totalPage": cat7_pages}]}
    old = pdf("mb-2018q3")
    p2 = {"lst": [item(2, "Báo cáo tài chính hợp nhất quý 3 năm 2018", "/resources/files/q3-2018-hn.pdf", mb_size(old))],
          "currentPageinfo": [{"totalPage": cat7_pages}]}
    sep = {"lst": [], "currentPageinfo": [{"totalPage": 1}]}
    routes[f"{origin}/api/GetListFinance/7/1/0"] = [FakeResp("", "", 403), FakeResp("", json.dumps(p1))]  # WAF hiccup
    routes[f"{origin}/api/GetListFinance/7/2/0"] = FakeResp("", json.dumps(p2))
    routes[f"{origin}/api/GetListFinance/13/1/0"] = FakeResp("", json.dumps(sep))
    routes[f"{origin}/resources/files/NhaDauTu/2026/BCTC/20260730---q2-hn.pdf"] = FakeResp(
        "", q2_body, headers={"Last-Modified": "Thu, 30 Jul 2026 12:42:59 GMT", "ETag": '"e1"'})
    routes[f"{origin}/resources/files/q3-2018-hn.pdf"] = FakeResp("", old, headers={"Last-Modified": "Wed, 03 Jul 2019 07:22:17 GMT"})
    if notices:
        for y in range(2016, 2027):
            routes[ed.MB_NOTICE_PAGE.format(year=y)] = FakeResp("", '<input name="__RequestVerificationToken" type="hidden" value="t" />')
            news = []
            if y == 2018:
                news = [{"id": "1254", "title": "MBB.CBTT Báo cáo tài chính hợp nhất & riêng lẻ Quý III năm 2018",
                         "created": "2018-10-19T17:00:10"},
                        {"id": "880", "title": "Mb Ký Hợp Đồng Cung Cấp Dịch Vụ Kiểm Toán Năm 2018,2019", "created": "2018-04-16T15:45:00"}]
            routes[f"{origin}/api/GetListMessage/1/{y}"] = FakeResp("", json.dumps({"otherNews": news, "currentPageinfo": [{"totalPage": 1}]}))
    for k, v in routes.items():
        for r in (v if isinstance(v, list) else [v]):
            r.url = k
    return routes


def test_mb_pagination_retry_and_availability_from_notice_and_last_modified(tmp_path):
    repo = Repo()
    q2 = pdf("mb-q2", 4000)
    code, http, sleeps = run_once(repo, mb_world(q2), tmp_path, symbol="MBB")
    assert code == ep.EXIT_OK
    calls = [u for m, u in http.session.calls]
    assert calls.count(f"{ed.MB_ORIGIN}/api/GetListFinance/7/1/0") == 2 and f"{ed.MB_ORIGIN}/api/GetListFinance/7/2/0" in calls
    assert http.retries >= 1 and 3 in sleeps
    docs = {d["ReportingPeriod"]: d for d in repo.docs("MBB")}
    assert set(docs) == {"2026Q2", "2018Q3"}
    # 2018Q3: notice created 2018-10-19 17:00:10 ICT -> 10:00:10 UTC; server Last-Modified 2019-07-03 (migration) ignored
    assert docs["2018Q3"]["PublishedAtUtc"] == datetime(2018, 10, 19, 10, 0, 10, tzinfo=UTC)
    assert docs["2018Q3"]["ExtractionJson"]["availability"]["noticeId"] == "1254"
    # 2026Q2: no notice -> Last-Modified (post-2019, within window)
    assert docs["2026Q2"]["PublishedAtUtc"] == datetime(2026, 7, 30, 12, 42, 59, tzinfo=UTC)
    assert "Last-Modified" in docs["2026Q2"]["ExtractionJson"]["availability"]["basis"]
    # the MB CMS save date is never used
    assert "2018-05-23" not in json.dumps(docs["2018Q3"]["ExtractionJson"]["availability"])
    notice_pages = [a for a in repo.attempts if a["Source"] == ep.NOTICE_SOURCE and a["Outcome"] == "listed"]
    assert len(notice_pages) == 11
    # resume: second backfill re-reads only the current year's page 1 (+ its token page)
    code, http, _ = run_once(repo, mb_world(q2), tmp_path, symbol="MBB", clock=lambda: NOW + timedelta(hours=1))
    notice_calls = [u for m, u in http.session.calls if "GetListMessage" in u]
    assert notice_calls == [f"{ed.MB_ORIGIN}/api/GetListMessage/1/2026"]


def test_mb_listing_403_exhausted_is_partial_and_no_gap_claims(tmp_path):
    repo = Repo()
    routes = mb_world(pdf("q2"), notices=False)
    routes[f"{ed.MB_ORIGIN}/api/GetListFinance/13/1/0"] = [FakeResp(f"{ed.MB_ORIGIN}/api/GetListFinance/13/1/0", "", 403)]
    code, http, _ = run_once(repo, routes, tmp_path, "--skip-notices", symbol="MBB")
    assert code == ep.EXIT_PARTIAL
    assert [a for a in repo.attempts if a["Outcome"] == "rate-limited"]
    assert not [a for a in repo.attempts if a["Outcome"] == "not-listed"]
    counts = last_run(repo)["CountsJson"]
    assert counts["requests"]["byOutcome"]["rate-limited"] == 4  # 1 + 3 session refreshes, each with 1 retry
    assert counts["discover"]["sessionRefreshes"] == 3
    bad = next(a for a in repo.attempts if a["Outcome"] == "rate-limited")
    assert "session refreshes" in bad["Detail"]


def test_mb_sticky_403_recovers_with_fresh_session(tmp_path):
    repo = Repo()
    routes = mb_world(pdf("q2", 100_000), notices=False)
    url = f"{ed.MB_ORIGIN}/api/GetListFinance/13/1/0"
    ok = routes[url]
    routes[url] = [FakeResp(url, "", 403), FakeResp(url, "", 403), ok]
    before = FakeCookies.cleared
    code, http, _ = run_once(repo, routes, tmp_path, "--skip-notices", symbol="MBB")
    assert code == ep.EXIT_OK and FakeCookies.cleared == before + 1
    assert last_run(repo)["CountsJson"]["discover"]["sessionRefreshes"] == 1


# ---------------------------------------------------------------------------------------------
# extraction seam


class FakeExtractor:
    EXTRACTOR_VERSION = "fake-v1"

    def __init__(self, repo, fail_on=None):
        self.repo, self.calls, self.fail_on, self.derived = repo, [], fail_on, 0

    def extract_document(self, conn, document, root, *, run_id, heartbeat):
        heartbeat()
        self.calls.append(document["ReportingPeriod"])
        if self.fail_on and document["ReportingPeriod"] == self.fail_on:
            raise RuntimeError("ocr exploded")
        self.repo.extractions.append({"DocumentId": document["Id"], "ExtractorVersion": self.EXTRACTOR_VERSION,
                                     "Status": "succeeded"})
        return {"status": "succeeded", "facts": 3, "validated": 2, "reviewNeeded": 1, "quarantined": 0}

    def derive_and_cross_validate(self, conn, symbol, *, run_id):
        self.derived += 1
        return {"derived": 0}


def test_extraction_seam_per_document_and_failure_isolated(tmp_path):
    repo = Repo()
    fx = FakeExtractor(repo, fail_on="2024")
    http, _ = make_http(bid_world())
    a = ep.build_parser().parse_args(["run", "--symbol", "BID", "--mode", "backfill"])
    code = ep.run(a, repo=repo, http=http, root=tmp_path, extractor_loader=lambda: (fx, None), clock=lambda: NOW)
    assert code == ep.EXIT_PARTIAL
    assert sorted(fx.calls) == ["2024", "2026Q2"]  # consolidated + restatement (unknown); separate out of default scope
    failed = [e for e in repo.extractions if e["Status"] == "failed"]
    assert len(failed) == 1 and "ocr exploded" in failed[0]["Error"]
    assert fx.derived == 1
    counts = last_run(repo)["CountsJson"]["extract"]
    assert counts["failed"] == 1 and counts["facts"] == 3 and counts["status:succeeded"] == 1
    # rerun: the succeeded one is not re-extracted, the failed one is retried
    fx2 = FakeExtractor(repo)
    http, _ = make_http(bid_world())
    ep.run(a, repo=repo, http=http, root=tmp_path, extractor_loader=lambda: (fx2, None), clock=lambda: NOW + timedelta(hours=1))
    assert fx2.calls == ["2024"]
    assert not [e for e in repo.extractions if e["Status"] == "failed"]  # failed row replaced by the successful retry


def test_missing_extractor_is_partial_not_silent(tmp_path):
    repo = Repo()
    http, _ = make_http(bid_world())
    a = ep.build_parser().parse_args(["run", "--symbol", "BID", "--mode", "backfill"])
    code = ep.run(a, repo=repo, http=http, root=tmp_path, extractor_loader=lambda: (None, "ai/equity_facts not importable"),
                  clock=lambda: NOW)
    assert code == ep.EXIT_PARTIAL
    assert "extraction unavailable" in last_run(repo)["Error"]


def test_dry_run_writes_nothing(tmp_path):
    repo = Repo()
    http, _ = make_http(bid_world())
    a = ep.build_parser().parse_args(["run", "--symbol", "BID", "--mode", "backfill", "--dry-run"])
    code = ep.run(a, repo=repo, http=http, root=tmp_path, clock=lambda: NOW)
    assert code == ep.EXIT_OK and not repo.runs and not repo.attempts and not repo.docs("BID")
    assert not [u for m, u in http.session.calls if ".pdf" in u]


def test_cli_rejects_bad_run_id():
    assert ep.main(["run", "--symbol", "BID", "--mode", "backfill", "--run-id", "nope"]) == ep.EXIT_FAILED


def test_mb_incremental_reads_newest_pages_until_overlap(tmp_path):
    repo = Repo()
    q2 = pdf("mb-q2", 100_000)
    assert run_once(repo, mb_world(q2, notices=False), tmp_path, "--skip-notices", symbol="MBB")[0] == ep.EXIT_OK
    code, http, _ = run_once(repo, mb_world(q2, notices=False), tmp_path, "--skip-notices", symbol="MBB", mode="incremental",
                             clock=lambda: NOW + timedelta(hours=1))
    assert code == ep.EXIT_OK
    calls = [u for m, u in http.session.calls]
    assert f"{ed.MB_ORIGIN}/api/GetListFinance/7/2/0" not in calls  # page 1 overlapped stored documents
    cells = [a for a in repo.attempts if a["Source"] == "pipeline-coverage" and a["RunId"] == last_run(repo)["Id"]]
    assert cells and min(a["TargetPeriod"] for a in cells) >= "2026"  # no gap claims outside the window read


def test_mb_incremental_window_excludes_shared_period_end_on_unread_page(tmp_path):
    """Page 1 ends with FY2025 (31/12) while Q4/2025 (also 31/12) sits on page 2: no gap may be claimed for 2025Q4."""
    origin = ed.MB_ORIGIN
    fy, q4 = pdf("fy2025", 100_000), pdf("q4-2025", 120_000)

    def it(i, title, path, body):
        return {"id": i, "title": title, "file_path": path, "file_zise": mb_size(body), "file_type": "pdf", "year": 2025}

    routes = {ed.MB_FS_PAGE: FakeResp(ed.MB_FS_PAGE, '<input name="__RequestVerificationToken" type="hidden" value="t" />'),
              f"{origin}/api/GetListFinance/7/1/0": FakeResp("", json.dumps({"lst": [it(1, "Báo cáo tài chính kiểm toán hợp nhất năm 2025", "/f/fy.pdf", fy)],
                                                                             "currentPageinfo": [{"totalPage": 2}]})),
              f"{origin}/api/GetListFinance/7/2/0": FakeResp("", json.dumps({"lst": [it(2, "Báo cáo tài chính hợp nhất Quý 4.2025", "/f/q4.pdf", q4)],
                                                                             "currentPageinfo": [{"totalPage": 2}]})),
              f"{origin}/api/GetListFinance/13/1/0": FakeResp("", json.dumps({"lst": [], "currentPageinfo": [{"totalPage": 1}]})),
              f"{origin}/f/fy.pdf": FakeResp("", fy), f"{origin}/f/q4.pdf": FakeResp("", q4)}
    for k, v in routes.items():
        v.url = k
    repo = Repo()
    assert run_once(repo, routes, tmp_path, "--skip-notices", symbol="MBB")[0] == ep.EXIT_OK
    run_once(repo, routes, tmp_path, "--skip-notices", symbol="MBB", mode="incremental", clock=lambda: NOW + timedelta(hours=1))
    rid = last_run(repo)["Id"]
    claims = [a for a in repo.attempts if a["RunId"] == rid and a["Source"] == "pipeline-coverage" and a["Outcome"] == "not-listed"
              and not a["Detail"].startswith("not yet due")]
    assert not [a for a in claims if a["TargetPeriod"] in ("2025Q4", "2025")]


def test_dry_run_reports_legacy_documents_as_link_not_fetch(tmp_path):
    import hashlib
    repo = Repo()
    routes = bid_world()
    body = pdf("bid-2026q2-cons")
    url = next(k for k in routes if "q2hn.pdf" in k)
    lid = uuid.uuid4()
    repo.store[lid] = {"Id": lid, "Symbol": "BID", "DocType": "financial-statement", "ReportingPeriod": "2026Q2",
                       "Consolidation": "consolidated", "ContentSha256": hashlib.sha256(body).hexdigest(), "ByteSize": len(body),
                       "Revision": 1, "SourceUrl": url, "Provider": "issuer:BIDV", "Title": "legacy", "PublishedAtUtc": NOW,
                       "FirstSeenAtUtc": NOW, "FetchedAtUtc": NOW, "ExtractionJson": {}}
    http, _ = make_http(routes)
    a = ep.build_parser().parse_args(["run", "--symbol", "BID", "--mode", "backfill", "--dry-run"])
    assert ep.run(a, repo=repo, http=http, root=tmp_path, clock=lambda: NOW) == ep.EXIT_OK
    assert not repo.locs and not repo.attempts  # still nothing written
    # summary printed by run() is not captured here; check the counters through a second dry run's context
    ctx = ep.RunContext(repo, None, "BID", "backfill", True, lambda: NOW)
    ep.reconcile_existing(ctx)
    item = {"url": url, "title": "x", "listedSize": f"{len(body) / 1024:.0f} KB", "source": "bidv-ir-library",
            "classification": ed.classify_v3("Báo cáo tài chính hợp nhất Quý II năm 2026")}
    assert ep.process_target(ctx, http, tmp_path, item, head_check=False, max_bytes=10**7) == "skipped-unchanged"
    assert ctx.counts["download"] == {"wouldReconcileLegacyDocuments": 1, "wouldLinkExisting": 1, "unchangedNoFetch": 1}
    ctx.stop()


def test_fixture_origin_seam_is_loopback_only_and_refuses_production_db(monkeypatch):
    for k in ("BIDV_LIBRARY_VI", "BIDV_DISCLOSURE_FEED", "MB_ORIGIN", "MB_FS_PAGE", "MB_NOTICE_PAGE"):
        monkeypatch.setattr(ed, k, getattr(ed, k))
    with pytest.raises(ValueError):
        ep.apply_fixture_origin("https://bidv.com.vn", "btc_eqx_pipetest")
    with pytest.raises(ValueError):
        ep.apply_fixture_origin("http://10.0.0.5:8080", "btc_eqx_pipetest")
    with pytest.raises(ValueError):
        ep.apply_fixture_origin("http://127.0.0.1:8765", "bitcoin_analyst")
    assert ep.apply_fixture_origin("http://127.0.0.1:8765", "btc_eqx_pipetest") == ["127.0.0.1"]
    assert ed.BIDV_LIBRARY_VI.startswith("http://127.0.0.1:8765/") and ed.MB_ORIGIN == "http://127.0.0.1:8765"
    http = ed.PoliteHttp(["127.0.0.1"], 5, delay_s=0, session=FakeSession({}), loopback_http=True)
    http._check("http://127.0.0.1:8765/x.pdf")
    prod = ed.PoliteHttp(["bidv.com.vn"], 5, delay_s=0, session=FakeSession({}), loopback_http=True)
    with pytest.raises(ValueError):  # loopback_http never relaxes https for real issuer hosts
        prod._check("http://bidv.com.vn/x.pdf")


def test_mb_incremental_auto_head_check_detects_same_size_overwrite_and_respects_interval(tmp_path):
    repo = Repo()
    a, b = pdf("mb-q2-a", 100_000), pdf("mb-q2-b", 100_000)  # same size: the listing cannot show the overwrite
    assert run_once(repo, mb_world(a, notices=False), tmp_path, "--skip-notices", symbol="MBB")[0] == ep.EXIT_OK
    q2url = f"{ed.MB_ORIGIN}/resources/files/NhaDauTu/2026/BCTC/20260730---q2-hn.pdf"

    def world(body, etag):
        r = mb_world(body, notices=False)
        r[q2url] = FakeResp(q2url, body, headers={"ETag": etag, "Last-Modified": "Mon, 19 Oct 2026 03:00:00 GMT"})
        return r

    # 1st incremental, no flag: never head-checked -> automatic HEAD -> ETag differs -> GET -> revision 2
    code, http, _ = run_once(repo, world(b, '"e2"'), tmp_path, "--skip-notices", symbol="MBB", mode="incremental",
                             clock=lambda: NOW + timedelta(days=1))
    assert code == ep.EXIT_OK
    revs = sorted((d for d in repo.docs("MBB") if d["ReportingPeriod"] == "2026Q2"), key=lambda d: d["Revision"])
    assert [d["Revision"] for d in revs] == [1, 2] and revs[1]["SupersedesDocumentId"] == revs[0]["Id"]
    assert ("HEAD", q2url) in http.session.calls
    heads = [x for x in repo.attempts if x["Source"] == ep.HEAD_CHECK_SOURCE]
    assert [x["Outcome"] for x in heads] == ["changed"]  # 2018Q3 is outside the 2-year window
    assert "never" in last_run(repo)["CountsJson"]["download"]["headCheckReason"]
    # 3 days later: last head-check is recent -> no HEAD at all
    code, http, _ = run_once(repo, world(b, '"e2"'), tmp_path, "--skip-notices", symbol="MBB", mode="incremental",
                             clock=lambda: NOW + timedelta(days=4))
    assert code == ep.EXIT_OK and not [c for c in http.session.calls if c[0] == "HEAD"]
    assert "skipped" in last_run(repo)["CountsJson"]["download"]["headCheckReason"]
    # --head-check forces it; unchanged validators -> nothing re-fetched
    code, http, _ = run_once(repo, world(b, '"e2"'), tmp_path, "--skip-notices", "--head-check", symbol="MBB",
                             mode="incremental", clock=lambda: NOW + timedelta(days=5))
    assert code == ep.EXIT_OK and ("HEAD", q2url) in http.session.calls
    assert not [c for c in http.session.calls if c[0] == "GET" and c[1] == q2url]
    assert len([d for d in repo.docs("MBB") if d["ReportingPeriod"] == "2026Q2"]) == 2
    assert [x["Outcome"] for x in repo.attempts if x["Source"] == ep.HEAD_CHECK_SOURCE][-1] == "unchanged"
    # more than 7 days after the last head-check: due again automatically
    code, http, _ = run_once(repo, world(b, '"e2"'), tmp_path, "--skip-notices", symbol="MBB", mode="incremental",
                             clock=lambda: NOW + timedelta(days=14))
    assert code == ep.EXIT_OK and ("HEAD", q2url) in http.session.calls


def test_bid_never_head_checks(tmp_path):
    repo = Repo()
    run_once(repo, bid_world(), tmp_path)
    code, http, _ = run_once(repo, bid_world(), tmp_path, "--head-check", mode="incremental",
                             clock=lambda: NOW + timedelta(days=30))
    assert code == ep.EXIT_OK and not [c for c in http.session.calls if c[0] == "HEAD"]



def test_in_scope_statement_in_unsupported_format_gets_attempt_and_coverage_reason(tmp_path):
    """A 2016+ statement listed only as an archive is a real coverage loss: attempt with period/consolidation,
    partial run, and the gap cell names the reason. Pre-2016 archives (BIDV 2008-2012 .zip) are only counted."""
    zip_item = bidv_item("Báo cáo tài chính riêng lẻ giữa niên độ Quý IV năm 2025", "30/01/2026",
                         "/wps/wcm/connect/55555555-5555-5555-5555-555555555555/q4rl.zip?MOD=AJPERES&amp;CACHEID=ROOTWORKSPACE-55555555-5555-5555-5555-555555555555-t",
                         "6 MB", group="2025")
    for mode in ("backfill", "incremental"):
        repo = Repo()
        code, http, _ = run_once(repo, bid_world(extra_items=zip_item), tmp_path, mode=mode)
        assert code == ep.EXIT_PARTIAL
        bad = [a for a in repo.attempts if a["Outcome"] == "parse-error"]
        assert len(bad) == 1 and bad[0]["TargetPeriod"] == "2025Q4" and bad[0]["Consolidation"] == "separate"
        assert "not a PDF (zip)" in bad[0]["Detail"] and bad[0]["DocType"] == "financial-statement"
        assert not [u for m, u in http.session.calls if u.endswith(".zip") or ".zip?" in u]  # never downloaded
        cell = [a for a in repo.attempts if a["Source"] == "pipeline-coverage" and a["TargetPeriod"] == "2025Q4"
                and a["Consolidation"] == "separate"]
        if mode == "backfill":
            assert cell and "listed but not storable: not a PDF (zip)" in cell[0]["Detail"]
        assert not [a for a in repo.attempts if "2011" in (a.get("Detail") or "") and a["Source"] != "pipeline-coverage"]
