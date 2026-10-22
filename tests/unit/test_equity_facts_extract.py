"""equity_facts: extract_document orchestration (idempotency, fail-closed, DB-contract shape) and an integration
test on real cropped page OCR fixtures with hand-checked ground truth (evidence/equity-expansion/v3/b0-samples)."""
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from equity_facts import EXTRACTOR_VERSION
from equity_facts.extract import build_facts, extract_with
from equity_facts.periods import doc_period
from equity_facts.tables import build_table
from equity_facts.textnorm import detect_unit
from equity_facts.validate import evaluate

FIX = Path(__file__).parent / "fixtures" / "equity_facts"
GT_TO_METRIC = {
    "Tiền mặt, vàng bạc, đá quý": "CASH_AND_GOLD", "Tiền gửi tại Ngân hàng Trung ương": "DEPOSITS_AT_SBV",
    "Tiền, vàng gửi tại và cho vay TCTD khác": "INTERBANK_PLACEMENTS_NET", "Dự phòng rủi ro cho vay khách hàng": "LOAN_LOSS_PROVISION_BALANCE",
    "Chứng khoán đầu tư": "INVESTMENT_SECURITIES_NET", "Góp vốn, đầu tư dài hạn": "LONG_TERM_INVESTMENTS_NET", "TỔNG TÀI SẢN CÓ": "TOTAL_ASSETS",
    "Thu nhập lãi và các khoản thu nhập tương tự": "INTEREST_INCOME", "Chi phí lãi và các khoản chi phí tương tự": "INTEREST_EXPENSE",
    "Chi phí lãi và các chi phí tương tự": "INTEREST_EXPENSE", "Thu nhập lãi thuần": "NET_INTEREST_INCOME",
    "Thu nhập từ hoạt động dịch vụ": "FEE_INCOME_GROSS", "Chi phí hoạt động dịch vụ": "FEE_EXPENSE", "Lãi thuần từ hoạt động dịch vụ": "FEE_INCOME",
    "Lãi/ lỗ thuần từ hoạt động dịch vụ": "FEE_INCOME", "Lãi thuần từ hoạt động kinh doanh ngoại hối": "FX_INCOME_NET",
    "Lãi thuần từ hoạt động kinh doanh khác": "OTHER_INCOME_NET", "Lãi/ lỗ thuần từ hoạt động khác": "OTHER_INCOME_NET",
    "Thu nhập từ góp vốn mua cổ phần": "EQUITY_INVESTMENT_INCOME", "Thu nhập từ góp vốn, mua cổ phần": "EQUITY_INVESTMENT_INCOME",
    "TỔNG THU NHẬP HOẠT ĐỘNG": "TOTAL_OPERATING_INCOME", "TỔNG CHI PHÍ HOẠT ĐỘNG": "OPERATING_EXPENSES", "Chi phí hoạt động": "OPERATING_EXPENSES",
    "Chi phí dự phòng rủi ro": "CREDIT_PROVISION_EXPENSE", "Chi phí dự phòng rủi ro tín dụng": "CREDIT_PROVISION_EXPENSE",
    "TỔNG LỢI NHUẬN TRƯỚC THUẾ": "PROFIT_BEFORE_TAX", "Tổng lợi nhuận trước thuế": "PROFIT_BEFORE_TAX",
    "Chi phí thuế TNDN hiện hành": "CIT_CURRENT", "Chi phí thuế TNDN trong kỳ": "INCOME_TAX_EXPENSE", "Chi phí thuế TNDN": "INCOME_TAX_EXPENSE",
    "LỢI NHUẬN SAU THUẾ": "PROFIT_AFTER_TAX", "Lợi nhuận sau thuế": "PROFIT_AFTER_TAX",
    "Lợi ích của cổ đông không kiểm soát": "MINORITY_INTEREST_PROFIT", "Lợi ích của cổ đông thiểu số": "MINORITY_INTEREST_PROFIT",
    "LỢI NHUẬN SAU THUẾ CỦA NGÂN HÀNG": "PROFIT_AFTER_TAX_PARENT", "Lợi nhuận còn lại của Ngân hàng": "PROFIT_AFTER_TAX_PARENT",
    "Nợ đủ tiêu chuẩn": "LOAN_GROUP1", "Nợ cần chú ý": "NPL_GROUP2", "Nợ dưới tiêu chuẩn": "NPL_GROUP3", "Nợ nghi ngờ": "NPL_GROUP4",
    "Nợ có khả năng mất vốn": "NPL_GROUP5",
}


def load(name):
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------
# integration: real OCR boxes -> facts == hand-checked ground truth
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pages", [
    ["mbb_q2_2026_p6_is"],
    ["bid_q2_2026_p4_bs", "bid_q2_2026_p7_is", "bid_q2_2026_p22_note_loans"],
    ["bid_q1_2026_p7_is"],
])
def test_real_page_fixtures_match_ground_truth(pages):
    fx = [load(p) for p in pages]
    sym, rp = fx[0]["symbol"], fx[0]["reportingPeriod"]
    kinds, tables, units = {}, {}, {}
    for i, f in enumerate(fx, start=1):
        kinds[i] = f["pageKind"]
        tables[i] = build_table(f["words"], f["boxes"], f["width"], f["height"])
        units[i] = detect_unit("\n".join(w["text"] for w in f["words"]))
    facts, dashes, unreadable, _ = build_facts({"Symbol": sym, "Id": "doc"}, doc_period(rp), kinds, tables, units,
                                               {i: "" for i in kinds}, {})
    evaluate(facts, dashes, unreadable)
    by = {(f["metric"], f["page"], f["col"]): f for f in facts}
    checked = 0
    for i, f in enumerate(fx, start=1):
        for label, values in f["groundTruth"]["rows"]:
            metric = GT_TO_METRIC.get(label)
            if not metric:
                continue
            for col, want in enumerate(values):
                got = by.get((metric, i, col))
                assert got is not None, (f["groundTruth"]["doc"], label, col)
                assert got["value"] == want, (label, col, got["raw"])
                assert got["status"] == "validated", (label, col, got.get("checks"))
                checked += 1
    assert checked >= 30
    assert all(f["unit"][0] == 1_000_000 for f in facts)


def test_mbb_income_statement_column_roles_from_headers():
    f = load("mbb_q2_2026_p6_is")
    table = build_table(f["words"], f["boxes"], f["width"], f["height"])
    facts, *_ = build_facts({"Symbol": "MBB", "Id": "d"}, doc_period("2026Q2"), {1: "income_statement"}, {1: table},
                            {1: (1_000_000, "triệu đồng")}, {1: ""}, {})
    pbt = {(x["periodLabel"], x["periodType"], x["role"]): x["value"] for x in facts if x["metric"] == "PROFIT_BEFORE_TAX"}
    assert pbt == {("2026Q2", "quarter", "current"): 10_559_872, ("2025Q2", "quarter", "comparative"): 7_502_991,
                   ("2026-6M", "ytd", "cumulative-current"): 20_188_258, ("2025-6M", "ytd", "cumulative-comparative"): 15_889_316}


# ------------------------------------------------------------------------------------------------
# extract_with: idempotency, fail-closed, DB contract
# ------------------------------------------------------------------------------------------------

class FakeRepo:
    def __init__(self, existing=None):
        self.existing = existing
        self.extractions, self.facts, self.deleted = [], [], []

    def get_extraction(self, doc_id, version):
        return self.existing

    def delete_failed_extraction(self, extraction_id):
        self.deleted.append(extraction_id)

    def insert_extraction(self, row):
        self.extractions.append(row)

    def insert_facts(self, facts, on_conflict_skip=False):
        self.facts.extend(facts)
        return len(facts)


class FakePages:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def close(self):
        pass


class FakeEngine:
    """Page plan of a BIDV quarterly report; full-page OCR served from the real fixtures."""

    PAGES = {2: "bid_q2_2026_p4_bs", 3: "bid_q2_2026_p7_is", 5: "bid_q2_2026_p22_note_loans"}
    CHEAP = {
        1: "CÔNG BỐ THÔNG TIN ĐỊNH KỲ\nNgân hàng TMCP Đầu tư và Phát triển Việt Nam",
        2: "BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT GIỮA NIÊN ĐỘ\n" + "\n".join(f"Chỉ tiêu {i} 1.234.567 2.345.678" for i in range(12)),
        3: "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG HỢP NHẤT GIỮA NIÊN ĐỘ\n" + "\n".join(f"Chỉ tiêu {i} 1.234.567 (2.345.678)" for i in range(12)),
        4: "THUYẾT MINH BÁO CÁO TÀI CHÍNH HỢP NHẤT CHỌN LỌC\n1. Đặc điểm hoạt động",
        5: ("3. CHO VAY KHÁCH HÀNG\nNợ đủ tiêu chuẩn 2.425.506.836 2.306.515.484\nNợ cần chú ý 30.581.448 31.462.771\n"
            "Nợ dưới tiêu chuẩn 12.003.167 4.473.881\nNợ nghi ngờ 7.293.693 4.677.985\nNợ có khả năng mất vốn 26.421.899 25.824.953"),
        6: "Trang cuối",
    }

    def __init__(self):
        self.full_calls = []

    def open(self, path):
        return FakePages(6)

    def cheap_text(self, pages, page_no):
        return self.CHEAP[page_no]

    def full_page(self, pages, page_no):
        self.full_calls.append(page_no)
        f = load(self.PAGES[page_no])
        return {"words": f["words"], "boxes": f["boxes"], "width": f["width"], "height": f["height"], "skewDeg": 0.0,
                "pdfRotation": 0}

    def text_layer(self, pages, page_no):
        return ""

    def versions(self):
        return {"fake": "1"}


def document(tmp_path, **kw):
    pdf = tmp_path / "BID" / "abc.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4 fake")
    doc = {"Id": "b905cf9d-753b-429c-b7d8-0b7b81cb2725", "Symbol": "BID", "StoragePath": "BID/abc.pdf", "ReportingPeriod": "2026Q2",
           "DocType": "financial-statement", "Consolidation": "consolidated", "ContentSha256": "abc", "Revision": 1,
           "PublishedAtUtc": datetime(2026, 7, 30, 17, 0, tzinfo=UTC), "FirstSeenAtUtc": datetime(2026, 10, 18, 8, 0, tzinfo=UTC),
           "SupersedesDocumentId": None, "Title": "Báo cáo tài chính hợp nhất giữa niên độ Quý II năm 2026"}
    doc.update(kw)
    return doc


CLOCK = lambda: datetime(2026, 10, 19, 10, 0, tzinfo=UTC)


def test_extract_succeeds_and_rows_satisfy_db_checks(tmp_path):
    repo, eng, beats = FakeRepo(), FakeEngine(), []
    res = extract_with(repo, eng, document(tmp_path), tmp_path, run_id="r1", heartbeat=lambda: beats.append(1), clock=CLOCK)
    assert res["status"] == "succeeded" and res["facts"] == len(repo.facts) > 50
    assert res["validated"] == res["facts"] - res["extracted"] - res["reviewNeeded"] - res["quarantined"]
    assert eng.full_calls == [2, 3, 5]                 # only classified pages get the expensive OCR
    assert len(beats) >= 5                              # heartbeat per page
    ex = repo.extractions[0]
    assert ex["ExtractorVersion"] == EXTRACTOR_VERSION and ex["Method"] == "ocr" and ex["PagesOcr"] <= ex["PageCount"] == 6
    art = tmp_path / ex["ArtifactPath"]
    assert (art / "pages.json").is_file() and (art / "p003.table.json").is_file() and (art / "p005.words.json").is_file()
    keys = set()
    for f in repo.facts:
        assert re.fullmatch(r"\d{4}(Q[1-4]|H[12]|-9M|-6M|-3M)?", f["PeriodLabel"])
        assert f["PeriodType"] in {"quarter", "ytd", "half-year", "annual", "point-in-time"}
        assert f["ColumnRole"] in {"current", "comparative", "cumulative-current", "cumulative-comparative"}
        assert f["Status"] in {"extracted", "validated", "review-needed", "quarantined"}
        assert f["MetricGroup"] in {"balance-sheet", "income-statement", "cash-flow", "credit-quality", "deposits-funding", "capital", "shares"}
        assert f["Statement"] in {"balance-sheet", "income-statement", "cash-flow", "note"}
        assert not f["IsDerived"] and f["DerivationJson"] is None and f["ExtractionId"] == ex["Id"]
        assert f["PageNumber"] > 0 and f["RawValueText"].strip() and f["Scale"] == 1_000_000 and f["Unit"] == "VND"
        assert f["Consolidation"] == "consolidated" and len(f["FactKey"]) <= 300 and len(f["Location"]) <= 500
        assert f["AvailableAtUtc"] == datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
        keys.add(f["FactKey"])
    assert len(keys) == len(repo.facts)
    ta = {(f["PeriodLabel"], f["ColumnRole"]): (f["Value"], f["Status"]) for f in repo.facts if f["MetricCode"] == "TOTAL_ASSETS"}
    assert ta == {("2026Q2", "current"): (3_440_843_537, "validated"), ("2025Q4", "comparative"): (3_330_825_720, "validated")}


def test_existing_successful_extraction_is_skipped(tmp_path):
    repo, eng = FakeRepo(existing={"Id": "e1", "Status": "partial"}), FakeEngine()
    res = extract_with(repo, eng, document(tmp_path), tmp_path, run_id=None, heartbeat=lambda: None, clock=CLOCK)
    assert res["skipped"] is True and repo.extractions == [] and eng.full_calls == []


def test_failed_extraction_is_deleted_and_retried(tmp_path):
    repo = FakeRepo(existing={"Id": "old-failed", "Status": "failed"})
    res = extract_with(repo, FakeEngine(), document(tmp_path), tmp_path, run_id=None, heartbeat=lambda: None, clock=CLOCK)
    assert repo.deleted == ["old-failed"] and res["status"] == "succeeded"


@pytest.mark.parametrize("kw,needle", [
    ({"ReportingPeriod": None}, "ReportingPeriod"),
    ({"StoragePath": "BID/missing.pdf"}, "missing"),
    ({"Symbol": "VCB"}, "out of scope"),
])
def test_unusable_document_records_failed_extraction(tmp_path, kw, needle):
    repo = FakeRepo()
    res = extract_with(repo, FakeEngine(), document(tmp_path, **kw), tmp_path, run_id=None, heartbeat=lambda: None, clock=CLOCK)
    assert res["status"] == "failed" and needle in res["error"]
    assert repo.extractions[0]["Status"] == "failed" and repo.facts == []


def test_consolidation_mismatch_quarantines_everything(tmp_path):
    repo = FakeRepo()
    res = extract_with(repo, FakeEngine(), document(tmp_path, Consolidation="separate"), tmp_path, run_id=None,
                       heartbeat=lambda: None, clock=CLOCK)
    titled = [f for f in repo.facts if f["Statement"] in ("balance-sheet", "income-statement")]  # pages titled HỢP NHẤT
    assert titled and all(f["Status"] == "quarantined" for f in titled)
    assert all("consolidation-mismatch" in " ".join(f["ValidationJson"]["flags"]) for f in titled)
    assert res["quarantined"] >= len(titled)


def test_first_seen_availability_when_issuer_date_unknown(tmp_path):
    repo = FakeRepo()
    extract_with(repo, FakeEngine(), document(tmp_path, PublishedAtUtc=None), tmp_path, run_id=None, heartbeat=lambda: None,
                 clock=CLOCK)
    assert {f["AvailableAtUtc"] for f in repo.facts} == {datetime(2026, 10, 18, 8, 0, tzinfo=UTC)}


def test_real_ocr_smoke_on_cropped_png():
    """Runs the real engines on a small crop when the portable Tesseract + RapidOCR are installed (skipped in CI)."""
    from equity_facts import ocr

    ok, why = ocr.ocr_available()
    if not ok:
        pytest.skip(why)
    from PIL import Image

    img = Image.open(FIX / "mbb_q2_2026_p6_is_crop.png").convert("L")
    words = ocr.tesseract_words(img, 300)
    boxes = ocr.rapid_boxes(img)
    table = build_table(words, boxes, img.width, img.height)
    rows = {r["labelNorm"][0] if r["labelNorm"] else "": r for r in table["rows"] if r["cells"]}
    nii = next(r for k, r in rows.items() if "THU NHAP LAI THUAN" in k)
    assert [nii["cells"][str(k)]["value"] for k in range(4)] == [16_893_654, 12_372_117, 31_806_771, 24_064_301]
