"""Offline tests for ai/equity_documents.py (no network, synthetic fixtures only)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import equity_documents as ed  # noqa: E402

ENTITY_FIELDS = {
    "Symbol", "Entity", "DocType", "Title", "SourceUrl", "Provider", "ContentSha256", "StoragePath", "ByteSize",
    "MimeType", "ReportingPeriod", "Consolidation", "AuditStatus", "PublishedAtUtc", "FirstSeenAtUtc",
    "FetchedAtUtc", "Revision", "SupersedesDocumentId", "ExtractionJson",
}


# --- classification / selection ---------------------------------------------------------------

@pytest.mark.parametrize("title, period, kind, cons, audit", [
    ("Báo cáo tài chính hợp nhất giữa niên độ Quý II năm 2026", "2026Q2", "quarterly", "consolidated", "unknown"),
    ("Báo cáo tài chính hợp nhất Quý 4.2025", "2025Q4", "quarterly", "consolidated", "unknown"),
    ("BCTC hợp nhất Quý 1 năm 2026", "2026Q1", "quarterly", "consolidated", "unknown"),
    ("Báo cáo tài chính hợp nhất Quý IV/2021", "2021Q4", "quarterly", "consolidated", "unknown"),
    ("Báo cáo tài chính kiểm toán hợp nhất năm 2025", "2025", "annual", "consolidated", "audited"),
    ("Báo cáo tài chính hợp nhất kiểm toán năm 2024", "2024", "annual", "consolidated", "audited"),
    ("Báo cáo tài chính kiểm toán riêng Ngân hàng 2025", "2025", "annual", "separate", "audited"),
    ("Báo cáo tài chính riêng giữa niên độ Quý II năm 2026", "2026Q2", "quarterly", "separate", "unknown"),
    ("Báo cáo tài chính hợp nhất soát xét bán niên năm 2026", None, "semiannual", "consolidated", "reviewed"),
])
def test_classify_title(title, period, kind, cons, audit):
    c = ed.classify_title(title)
    assert (c["period"], c["kind"], c["consolidation"], c["auditStatus"]) == (period, kind, cons, audit)
    assert c["isFinancialStatement"]


def _cand(title, url):
    return {"title": title, "sourceUrl": url, **ed.classify_title(title)}


def test_select_targets_latest_four_quarters_plus_latest_audited_annual_consolidated_only():
    cands = [_cand(t, f"https://x/{i}") for i, t in enumerate([
        "Báo cáo tài chính hợp nhất Quý 2 năm 2026", "Báo cáo tài chính riêng Quý 2 năm 2026",
        "BCTC hợp nhất Quý 1 năm 2026", "Báo cáo tài chính kiểm toán hợp nhất năm 2025",
        "Báo cáo tài chính hợp nhất Quý 4.2025", "Báo cáo tài chính hợp nhất Quý 3.2025",
        "Báo cáo tài chính hợp nhất quý 2.2025", "Báo cáo tài chính hợp nhất kiểm toán năm 2024",
        "Báo cáo tài chính hợp nhất soát xét bán niên năm 2026",
    ])]
    picked = ed.select_targets(cands)
    assert [p["period"] for p in picked] == ["2026Q2", "2026Q1", "2025Q4", "2025Q3", "2025"]
    assert all(p["consolidation"] == "consolidated" for p in picked)


# --- listing parsers ---------------------------------------------------------------------------

BIDV_FIXTURE = """
var item = {
\ttype_pdf: 'File PDF' === 'File PDF',
\tparenttitle: formatTitle('2026'),
parentname: formatTitle('2026'),
\ttitle: formatTitle('Báo cáo tài chính hợp nhất giữa niên độ Quý II năm 2026'),
\tpublishdate: '30/07/2026',
\tfile_title: formatTitle('/wps/wcm/connect/abc/20260730+-+BID+-+CBTT+BCTC+HN+Q2.pdf?MOD=AJPERES&amp;CACHEID=ROOT-abc'),
\tfile_size: '12.84 MB',
\tpath: '/bidv/x'
};
var item = {
\ttype_pdf: 'File PDF' === 'File PDF',
\tparenttitle: formatTitle('2026'),
parentname: formatTitle('2026'),
\ttitle: formatTitle('Bản tin nhà đầu tư'),
\tpublishdate: '01/08/2026',
\tfile_title: formatTitle('/wps/wcm/connect/def/video.mp4'),
\tfile_size: '1 MB',
\tpath: '/bidv/y'
};
"""


def test_parse_bidv_listing_unescapes_url_and_keeps_stated_date_only():
    items = ed.parse_bidv_listing(BIDV_FIXTURE, "https://bidv.com.vn/vn/quan-he-nha-dau-tu/bao-cao-va-tai-lieu/")
    assert len(items) == 1  # non-PDF entry dropped
    it = items[0]
    assert it["sourceUrl"] == ("https://bidv.com.vn/wps/wcm/connect/abc/20260730+-+BID+-+CBTT+BCTC+HN+Q2.pdf"
                               "?MOD=AJPERES&CACHEID=ROOT-abc")
    assert it["publishedDate"] == "2026-07-30" and it["period"] == "2026Q2"


def test_parse_mb_listing_has_no_published_date():
    payload = {"lst": [{"title": "Báo cáo tài chính hợp nhất  Quý 2 năm 2026", "title_en": "Q2",
                        "file_path": "/resources/files/NhaDauTu/2026/BCTC/q2.pdf", "file_zise": "3.51 MB",
                        "last_Save_Date": "2026-07-30T19:42:59.283", "year": 2026}]}
    it = ed.parse_mb_listing(payload, "https://www.mbbank.com.vn/api/GetListFinance/7/1/2026")[0]
    assert it["sourceUrl"] == "https://www.mbbank.com.vn/resources/files/NhaDauTu/2026/BCTC/q2.pdf"
    assert it["publishedDate"] is None  # CMS save time is not a publication statement
    assert it["listingLastSaveLocal"] == "2026-07-30T19:42:59.283"


# --- numbers -----------------------------------------------------------------------------------

@pytest.mark.parametrize("tok, val", [
    ("1.234.567", 1234567), ("(17.113.184)", -17113184), ("(4,786.217)", -4786217), ("1,234", 1234),
    ("-3.454", -3454), ("26.340,00", None), ("12.5", None), ("30", 30),
])
def test_parse_number(tok, val):
    assert ed.parse_number(tok) == val


# --- OCR income statement ----------------------------------------------------------------------

LABELS = """Ngan hang ThLPO'ng m9i Co phan Quan dc}i
BAO cAo Kl:T QUA HO� T DQNG H9'P NHAT
Quy 11/2026
Thu nh$p lai va cac khoan thu nh$p tU'ang ti,!'
Chi phi lai va cac khoan chi phf tU'ang ti,!'
Thu nh�p lai thucln
Thu nh,;Jp tu ho?t dong d[ch vu
Chi phi ho?t d¢ng d[ch vu
Lai thu§n tiP ho�t d(ing djch v1,1
T6NG LQ'I NHU�N TRU'O'C THUi:
Chi phf thu� TNDN hi$n hanh
Chi phi thue TNDN hoan l?i
Chi phi thu� TNDN trong ky
LQ'l NHUAN SAU THUi:
LQ'i fch cua c6 dong khong kiem soat
LQ'I NHU,?.N SAU THUE CUA NGAN HANG
S6 phat sinh quy II
Nam nay Nam trU'O'C
trieu aong tri�u aong
"""
# rows: interest income, interest expense, NII, fee income, fee expense, net fee, PBT, cit cur, cit def, cit, PAT, NCI, PAT bank
COL1 = ["34.006.838", "(17.113.184)", "16.893.654", "5.249.714", "(3.444.039)", "1.805.675", "10.559.872",
        "(2.120.910)", "6.511", "(2.114.399)", "8.445.473", "216.411", "8.229.062"]
COL2 = ["21.099.126", "(8.727.009)", "12.372.117", "4.613.875", "(2.698.332)", "1.915.543", "7.502.991",
        "(1.496.624)", "(1.661)", "(1.498.285)", "6.004.706", "127.101", "5.877.605"]


def test_ocr_npat_paired_columns_takes_column_one():
    page = LABELS + "\n".join(f"{a} {b}" for a, b in zip(COL1, COL2))
    hit, why = ed.extract_ocr_quarter_net_profit(["cover", page], "2026Q2")
    assert hit and not why
    assert hit["valueOriginal"] == 8445473 and hit["value"] == 8_445_473_000_000 and hit["unit"] == "VND"
    assert hit["page"] == 2 and "8.445.473" in hit["sourceLocation"]
    assert hit["alignmentCheck"]["profitBeforeTax"] == 10559872


def test_ocr_npat_single_column_blocks():
    page = LABELS + "\n".join(COL1) + "\nBa Nguyen Thi Thanh Nga\n" + "\n".join(COL2)
    hit, _ = ed.extract_ocr_quarter_net_profit([page], "2026Q2")
    assert hit["valueOriginal"] == 8445473


def test_ocr_npat_does_not_drift_into_column_two_when_column_one_is_garbled():
    garbled = COL1[:10] + ["8.445.47 3x"] + COL1[11:]  # PAT unreadable in column 1
    page = LABELS + "\n".join(garbled) + "\n" + "\n".join(COL2)
    hit, why = ed.extract_ocr_quarter_net_profit([page], "2026Q2")
    assert hit is None and "column 1" in why


def test_ocr_npat_rejects_annual_period_and_missing_unit():
    page = LABELS + "\n".join(f"{a} {b}" for a, b in zip(COL1, COL2))
    assert ed.extract_ocr_quarter_net_profit([page], "2025")[0] is None
    hit, why = ed.extract_ocr_quarter_net_profit([page.replace("trieu aong", "x").replace("tri�u aong", "x")], "2026Q2")
    assert hit is None and "million-VND unit" in why


# --- generic same-line extraction --------------------------------------------------------------

def test_extract_metrics_same_line_with_unit_and_ratio_definitions():
    bs = ("BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT\nĐơn vị: triệu đồng\nTỔNG TÀI SẢN 1.234.567 1.100.000\n"
          "VIII Vốn chủ sở hữu 98.765 90.000\n")
    inc = "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT\nĐơn vị: triệu đồng\nXIII Lợi nhuận sau thuế 4.321 3.000\n"
    npl = "Tỷ lệ nợ xấu / tổng dư nợ cho vay khách hàng (nợ nhóm 3 đến nhóm 5)\nTỷ lệ nợ xấu 1,45%\n"
    out = ed.extract_metrics([bs, inc, npl], "2026Q2", "consolidated")
    found = {f["metric"]: f for f in out["found"]}
    assert found["TOTAL_ASSETS"]["value"] == 1_234_567_000_000 and found["TOTAL_ASSETS"]["page"] == 1
    assert found["EQUITY"]["valueOriginal"] == 98765
    assert found["NET_PROFIT"]["page"] == 2 and found["NET_PROFIT"]["unitOriginal"] == "triệu đồng"
    assert found["NPL"]["value"] == 1.45 and found["NPL"]["unit"] == "percent"
    assert {m["metric"] for m in out["missing"]} == {"CAR", "COVERAGE"}


def test_extract_metrics_requires_explicit_unit_and_ratio_definition():
    bs = "BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT\nTỔNG TÀI SẢN 1.234.567\n"
    npl = "Tỷ lệ nợ xấu 1,45%\n"  # no definition on the page
    out = ed.extract_metrics([bs, npl], "2026Q2", "consolidated")
    assert not out["found"]
    reasons = {m["metric"]: m["reason"] for m in out["missing"]}
    assert "unit" in reasons["TOTAL_ASSETS"] and "definition" in reasons["NPL"]


# --- fetch: dedup / resume / safety ------------------------------------------------------------

class FakeResp:
    def __init__(self, url, body: bytes, status=200, ctype="application/pdf"):
        self.url, self.content, self.status_code = url, body, status
        self.headers = {"Content-Type": ctype}
        self.text = body.decode("latin-1")

    def iter_content(self, n):
        for i in range(0, len(self.content), n):
            yield self.content[i:i + n]


class FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def get(self, url, **_):
        self.calls.append(url)
        return self.routes[url]


def _setup_catalog(root: Path, urls: list[str]):
    sdir = root / "MBB"
    sdir.mkdir(parents=True)
    cands = [{"title": f"Báo cáo tài chính hợp nhất Quý {i + 1} năm 2026", "sourceUrl": u, "listingUrl": "l",
              "publishedDate": None, **ed.classify_title(f"Báo cáo tài chính hợp nhất Quý {i + 1} năm 2026")}
             for i, u in enumerate(urls)]
    ed.write_json(sdir / "catalog.json", {"discoveredAtUtc": "2026-10-18T00:00:00Z", "candidates": cands,
                                          "targets": urls})


def test_fetch_dedups_by_sha_resumes_and_rejects_non_pdf(tmp_path):
    base = "https://www.mbbank.com.vn/resources/files/"
    a, b, c = base + "a.pdf", base + "b.pdf", base + "c.pdf"
    pdf = b"%PDF-1.7 fake body"
    _setup_catalog(tmp_path, [a, b, c])
    sess = FakeSession({a: FakeResp(a, pdf), b: FakeResp(b, pdf), c: FakeResp(c, b"<html>nope", ctype="text/html")})
    rep = ed.cmd_fetch(tmp_path, "MBB", budget=10, delay_s=0, session=sess)
    sha = hashlib.sha256(pdf).hexdigest()
    assert [f["sha256"] for f in rep["fetched"]] == [sha]
    assert rep["skipped"][0]["reason"] == "duplicate-content"
    assert "not a PDF" in rep["failed"][0]["reason"]
    assert (tmp_path / "MBB" / f"{sha}.pdf").read_bytes() == pdf
    assert not list((tmp_path / "MBB").glob(".partial-*"))
    log = [json.loads(x) for x in (tmp_path / "MBB" / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {e["url"] for e in log} == {a, b, c} and log[0]["sha256"] == sha
    # resume: the stored URL is not requested again
    sess2 = FakeSession({b: FakeResp(b, pdf), c: FakeResp(c, b"%PDF-other")})
    rep2 = ed.cmd_fetch(tmp_path, "MBB", budget=10, delay_s=0, session=sess2)
    assert a not in sess2.calls and rep2["skipped"][0]["reason"] == "already-stored"


def test_fetch_budget_and_host_allowlist(tmp_path):
    base = "https://www.mbbank.com.vn/resources/files/"
    _setup_catalog(tmp_path, [base + "a.pdf", base + "b.pdf"])
    sess = FakeSession({base + "a.pdf": FakeResp(base + "a.pdf", b"%PDF-a"), base + "b.pdf": FakeResp(base + "b.pdf", b"%PDF-b")})
    rep = ed.cmd_fetch(tmp_path, "MBB", budget=1, delay_s=0, session=sess)
    assert len(rep["fetched"]) == 1 and rep["failed"][0]["reason"] == "budget-exhausted"
    http = ed.Http(tmp_path / "MBB", ed.ISSUERS["MBB"], 5, 0, "t", FakeSession({}))
    with pytest.raises(ValueError):
        http.request("https://evil.example.com/x.pdf")
    with pytest.raises(ValueError):
        http.request("http://www.mbbank.com.vn/x.pdf")  # https only


# --- manifest / register -----------------------------------------------------------------------

def _state():
    listing = {"title": "Báo cáo tài chính hợp nhất Quý II năm 2026", "listingUrl": "l", "publishedDate": "2026-07-30",
               **ed.classify_title("Báo cáo tài chính hợp nhất Quý II năm 2026")}
    found = {"metric": "NET_PROFIT", "period": "2026Q2", "value": 8_445_473_000_000, "unit": "VND",
             "valueOriginal": 8445473, "unitOriginal": "triệu đồng", "consolidation": "consolidated",
             "sourceLocation": "page 6: row", "definition": "PAT incl. NCI"}
    annual = dict(found, period="2025")
    return {"documents": [{
        "documentId": "d1", "sourceUrl": "https://bidv.com.vn/a.pdf", "contentSha256": "ab" * 32,
        "storagePath": "BID/" + "ab" * 32 + ".pdf", "byteSize": 10, "mimeType": "application/pdf",
        "firstSeenAtUtc": "2026-10-18T00:00:00Z", "fetchedAtUtc": "2026-10-18T00:00:01Z", "revision": 1,
        "supersedesDocumentId": None, "listing": listing,
        "extraction": {"auditStatement": {"status": "unaudited"}, "found": [found, annual],
                       "missing": [{"metric": "TOTAL_ASSETS", "reason": "scanned"}]},
    }]}


def test_manifest_has_every_source_document_field_and_flags_annual_rows():
    m = ed.build_manifest("BID", _state())
    doc = m["documents"][0]
    assert ENTITY_FIELDS <= set(doc)
    assert doc["Provider"] == "issuer:BIDV" and doc["AuditStatus"] == "unaudited"  # stated in document
    # Date-only issuer statement -> available from 00:00 ICT of the next day (17:00Z same day).
    assert doc["PublishedDate"] == "2026-07-30" and doc["PublishedAtUtc"] == "2026-07-30T17:00:00Z"
    assert "00:00 ICT next day" in doc["PublishedAtBasis"]
    no_date = _state()
    no_date["documents"][0]["listing"]["publishedDate"] = None
    mb = ed.build_manifest("BID", no_date)["documents"][0]
    assert mb["PublishedAtUtc"] is None and mb["PublishedDate"] is None and mb["PublishedAtBasis"] is None
    assert "publishedDate" not in ed.register_payload(mb) and "publishedAtUtc" not in ed.register_payload(mb)
    assert doc["ExtractionJson"]["missing"][0]["metric"] == "TOTAL_ASSETS"
    rows = m["fundamentals"]
    assert [r["importable"] for r in rows] == [True, False]
    assert rows[0]["source"] == "issuer-report" and rows[0]["documentId"] == "d1"


def test_register_posts_documents_then_fundamentals(tmp_path, monkeypatch):
    sdir = tmp_path / "BID"
    sdir.mkdir()
    ed.write_json(sdir / "manifest.json", ed.build_manifest("BID", _state()))
    monkeypatch.setenv("EQX_ADMIN_KEY", "k")
    posts = []

    class R:
        status_code, text = 201, "{}"

        @staticmethod
        def json():
            return {"created": True, "item": {"documentId": "backend-guid-1"}}

    def poster(url, json, headers, timeout):
        posts.append((url, json, headers))
        return R()

    # Before any document is registered, no fundamentals row can be imported (needs a backend id).
    assert ed.cmd_register(tmp_path, "BID", "http://x", "fundamentals", poster)[0]["status"] is None
    ed.cmd_register(tmp_path, "BID", "http://localhost:5197/", "documents", poster)
    url, body, headers = posts[0]
    assert url == "http://localhost:5197/api/equities/BID/admin/source-documents"
    assert headers == {"X-Admin-Key": "k"} and body["contentSha256"] == "ab" * 32
    # EquitySourceDocumentRegistration shape: extraction is an object, backend assigns id/revision.
    assert isinstance(body["extraction"], dict) and body["extraction"]["missing"][0]["metric"] == "TOTAL_ASSETS"
    assert not {"documentId", "revision", "extractionJson", "byteSize"} & set(body)
    # The backend takes the date-only value and derives PublishedAtUtc itself (never both).
    assert body["publishedDate"] == "2026-07-30" and "publishedAtUtc" not in body
    assert body["storagePath"].startswith("BID/") and body["docType"] == "financial-statement"
    assert ed.read_json(tmp_path / "BID" / "registered.json", {})["ab" * 32]["documentId"] == "backend-guid-1"
    ed.cmd_register(tmp_path, "BID", "http://localhost:5197", "fundamentals", poster)
    url, body, _ = posts[1]
    assert url.endswith("/admin/fundamentals") and len(body) == 1 and body[0]["period"] == "2026Q2"
    assert body[0]["documentId"] == "backend-guid-1" and body[0]["sourceLocation"] == "page 6: row"


def test_register_requires_admin_key(tmp_path, monkeypatch):
    monkeypatch.delenv("EQX_ADMIN_KEY", raising=False)
    with pytest.raises(SystemExit):
        ed.cmd_register(tmp_path, "BID", "http://x")
