"""equity_facts: cheap page classification (statement titles, continuation pages, loan-quality note by structure)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from equity_facts.classify import classify_text, plan_pages, scan_done

NUMS = "\n".join(f"Dòng {i} 1.234.567 (2.345.678)" for i in range(10))


def info(page, text):
    return {"page": page, **classify_text(text)}


def test_statement_titles_continuation_and_notes():
    pages = [
        info(1, "CÔNG BỐ THÔNG TIN ĐỊNH KỲ"),
        info(2, "MỤC LỤC\nBÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT 5-7\nBÁO CÁO KẾT QUẢ HOẠT ĐỘNG HỢP NHẤT 8"),  # TOC: no amounts
        info(3, "NGÂN HÀNG TMCP ĐẦU TƯ VÀ PHÁT TRIỂN VIỆT NAM\nBÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT GIỮA NIÊN ĐỘ\n" + NUMS),
        info(4, "NGÂN HÀNG TMCP ĐẦU TƯ VÀ PHÁT TRIỂN VIỆT NAM\nBáo cáo tài chính hợp nhất giữa niên độ\n" + NUMS),  # continuation
        info(5, "BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT (tiếp theo)\nCÁC CHỈ TIÊU NGOÀI BÁO CÁO TÌNH HÌNH TÀI CHÍNH\n" + NUMS),
        info(6, "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG HỢP NHẤT\n" + NUMS),
        info(7, "BÁO CÁO LƯU CHUYỂN TIỀN TỆ HỢP NHẤT\n" + NUMS),
        info(8, "THUYẾT MINH BÁO CÁO TÀI CHÍNH HỢP NHẤT\n1. Đặc điểm hoạt động"),
    ]
    plan = plan_pages(pages)
    assert plan["kinds"] == {3: "balance_sheet", 4: "balance_sheet", 5: "off_balance", 6: "income_statement", 7: "cash_flow"}
    assert plan["notesStart"] == 8 and set(plan["consolidationTitles"]) == {"consolidated"}
    assert not scan_done(plan["kinds"], plan["notesStart"])


def test_loan_quality_note_needs_the_customer_loan_group_table():
    interbank = ("Phân tích chất lượng dư nợ tiền gửi tại các TCTD khác và cho vay các TCTD khác:\n"
                 "Nợ đủ tiêu chuẩn 141.740.943 103.903.210\nNợ có khả năng mất vốn - 1.127")
    policy = "Nợ đủ tiêu chuẩn là các khoản nợ được đánh giá có khả năng thu hồi đầy đủ ... cho vay khách hàng"
    customer = ("3. CHO VAY KHÁCH HÀNG\nNợ đủ tiêu chuẩn 2.425.506.836 2.306.515.484\nNợ cần chú ý 30.581.448 31.462.771\n"
                "Nợ dưới tiêu chuẩn 12.003.167 4.473.881\nNợ nghi ngờ 7.293.693 4.677.985\nNợ có khả năng mất vốn 26.421.899 25.824.953")
    assert not classify_text(interbank)["loanQualityTable"]
    assert not classify_text(policy)["loanQualityTable"]
    assert classify_text(customer)["loanQualityTable"]
    pages = [info(1, "BÁO CÁO TÌNH HÌNH TÀI CHÍNH HỢP NHẤT\n" + NUMS), info(2, "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG HỢP NHẤT\n" + NUMS),
             info(3, "THUYẾT MINH BÁO CÁO TÀI CHÍNH HỢP NHẤT"), info(4, interbank), info(5, customer)]
    plan = plan_pages(pages)
    assert plan["kinds"][5] == "note_loan_quality" and 4 not in plan["kinds"]
    assert scan_done(plan["kinds"], plan["notesStart"])


def test_separate_statement_title_detected():
    assert classify_text("BÁO CÁO TÌNH HÌNH TÀI CHÍNH RIÊNG\n" + NUMS)["consolidation"] == "separate"
