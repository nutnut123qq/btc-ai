"""equity_facts: number / unit / label normalization (V3-B)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from equity_facts.textnorm import (
    detect_unit,
    find_dates,
    fold,
    label_score,
    norm_label,
    parse_amount,
)


@pytest.mark.parametrize("raw,value", [
    ("1.234.567", 1234567),
    ("12,065,997", 12065997),           # BIDV Q1 2026 prints comma thousands
    ("(17.113.184)", -17113184),        # parentheses = negative
    ("(62,209)", -62209),
    ("(1.452. 7 42)", -1452742),        # spaces inside a well-formed grouped number (old text layer)
    ("787", 787),
    ("-1.661", -1661),
    ("457,353,489.", 457353489),        # trailing scan speckle
    ("3.388.221.517", 3388221517),
])
def test_parse_amount_numbers(raw, value):
    p = parse_amount(raw)
    assert p["kind"] == "number" and p["value"] == value


@pytest.mark.parametrize("raw", ["-", "–", "—"])
def test_parse_amount_dash_is_nil_not_zero_fact(raw):
    p = parse_amount(raw)
    assert p["kind"] == "dash" and p["value"] is None


@pytest.mark.parametrize("raw", ["6.004,.706", "1.234,567", "125117", "4249)", "12.34", "2026", "1.2345.678", "abc", ""])
def test_parse_amount_never_guesses(raw):
    p = parse_amount(raw)
    assert p["kind"] == "malformed" and p["value"] is None


@pytest.mark.parametrize("text,scale", [
    ("Đơn vị: Triệu VND", 1_000_000),
    ("Đơn vị: Triệu ND", 1_000_000),     # Tesseract reading of the BIDV header
    ("30/06/2026 triệu đồng", 1_000_000),
    ("Đơn vị tính: tỷ đồng", 1_000_000_000),
    ("Đơn vị tính: đồng", 1),
])
def test_detect_unit(text, scale):
    u = detect_unit(text)
    assert u is not None and u[0] == scale and u[1]


def test_detect_unit_absent_or_conflicting_is_none():
    assert detect_unit("Tiền gửi của khách hàng 963.815.357") is None
    assert detect_unit("Đơn vị: Triệu VND ... Đơn vị tính: tỷ đồng") is None


def test_fold_and_norm_label():
    assert fold("Tiền gửi của khách hàng") == "TIEN GUI CUA KHACH HANG"
    assert fold("Đầu tư") == "DAU TU"
    assert norm_label("VII Chứng khoán đầu tư 5") == "CHUNG KHOAN DAU TU"
    assert norm_label("Thu nhập lãi thuần IV.1") == "THU NHAP LAI THUAN"
    assert norm_label("3 _ Các khoản phải trả và công nợ khác 11") == "CAC KHOAN PHAI TRA VA CONG NO KHAC"
    assert norm_label('Tiền gửi tại Ngân hàng Nhà nước ("NHNN")') == "TIEN GUI TAI NGAN HANG NHA NUOC NHNN"


def test_label_score_tolerates_ocr_noise_but_separates_neighbours():
    gross = norm_label("Cho vay khách hàng")
    assert label_score(gross, norm_label("CHO VAYKHACHHANG")) >= 0.95
    assert label_score(norm_label("Chứng khoán đầu tư"), norm_label("VH Chứng khoán đầu tư _5")) >= 0.9
    assert label_score(gross, norm_label("Dự phòng rủi ro cho vay khách hàng")) < 0.86
    assert label_score(norm_label("Nợ đủ tiêu chuẩn"), norm_label("Nợ đủ tiêu chuẩn")) == 1.0


def test_find_dates():
    assert find_dates("Từ 01/04/2026 đến 30/06/2026") == [(2026, 4, 1), (2026, 6, 30)]
    assert find_dates("31/13/2025 99/99/9999") == []
