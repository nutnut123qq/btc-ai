# STOPS-1 — kiểm định ngược luật cắt lỗ -8% / giảm tỷ trọng -4% (đóng băng 2026-11-06)

Bản chuẩn là [`stops-1.json`](stops-1.json). SHA-256 của file đó (`6ba6d2dd…`) được ghi vào sổ đăng ký thử nghiệm
(mục `hose-stops-1`, trạng thái planned, 4 lần thử) trước khi chạy. Code: `vn30_stops_eval.py`.

**Câu hỏi:** luật rủi ro của `MbbDecisionService` (declared-rules-v4) có cải thiện danh mục VN30 sau phí không?
- Cắt lỗ: giá đóng cửa 1d ≤ giá vốn × 0,92 (làm tròn lên bước giá) → bán hết ở giá mở cửa phiên sau.
- Giảm tỷ trọng: xu hướng giảm (đóng cửa < SMA50 và RSI14 < 50) và đóng cửa ≤ giá vốn × 0,96 → bán một nửa
  (backend không nêu khối lượng; một nửa là lựa chọn khai báo), tối đa một lần giữa hai kỳ tái cân bằng.

**Thiết kế:** 2 danh mục (B-ew trọng số đều; B-mom_6 top 10) × 3 biến thể (không luật; cắt lỗ; cắt lỗ + giảm),
chạy bằng `vn_portfolio_backtest` với luật VN30-2, snapshot `20261106-9afb8020ec16`, chỉ cửa sổ phát triển
2018-01-22 → 2023-12-31 (không đọc holdout). Luật được coi là "có bằng chứng" chỉ khi CAGR, Sharpe và MDD đều tốt hơn
bản không luật **và** DSR của chuỗi vượt trội hằng ngày ≥ 0,95 (N = số lần thử HOSE trong sổ đăng ký).

**Kết quả (2026-11-06, N = 34): không có bằng chứng (not supported).**

| Danh mục / biến thể | CAGR ròng | Sharpe | MDD | Vòng quay/năm | DSR vượt trội |
|---|---|---|---|---|---|
| B-ew / không luật | 1,23% | 0,165 | -48,7% | 1,16 | |
| B-ew / cắt lỗ -8% | 3,76% | 0,288 | -40,3% | 3,52 | 0,154 |
| B-ew / cắt lỗ + giảm -4% | 3,80% | 0,293 | -38,8% | 4,23 | 0,120 |
| B-mom_6 / không luật | 3,12% | 0,250 | -52,7% | 4,28 | |
| B-mom_6 / cắt lỗ -8% | 3,33% | 0,262 | -47,9% | 6,69 | 0,013 |
| B-mom_6 / cắt lỗ + giảm -4% | 3,97% | 0,294 | -45,8% | 7,56 | 0,019 |

Ước lượng điểm đều nghiêng về có luật (CAGR, Sharpe cao hơn, MDD nông hơn), nhưng DSR vượt trội rất xa ngưỡng 0,95:
khác biệt nằm trong nhiễu của 6 năm và 30 mã. Luật -8% / -4% vẫn là một lựa chọn khẩu vị rủi ro được khai báo, không
phải một cải thiện đã được chứng minh. Kết quả đầy đủ: `docs/research/evidence/vn30-2/stops-1-20261106T120635Z/result.json`
(thư mục bị gitignore, trong bản checkout chính).
