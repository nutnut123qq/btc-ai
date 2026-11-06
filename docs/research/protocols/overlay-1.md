# OVERLAY-1 — lớp phủ chế độ thị trường (MA200) và mục tiêu biến động (15%) (đóng băng 2026-11-06)

Bản chuẩn là [`overlay-1.json`](overlay-1.json). SHA-256 của file đó (`e0a10361…`) được ghi vào sổ đăng ký thử nghiệm
(mục `hose-overlay-1`, trạng thái planned, 4 lần thử) trước khi tính bất cứ kết quả nào. Code: `vn30_overlay_eval.py`
(và hook `exposure` của `vn_portfolio_backtest.run_backtest`).

**Câu hỏi (lộ trình 3.2):** giảm tỷ trọng cổ phiếu theo chế độ thị trường hoặc theo biến động có giảm sụt giảm
(drawdown) của danh mục VN30 sau phí không? Không nằm trong giao thức VN30-2 đóng băng, nên là một thử nghiệm riêng.

**Thiết kế:**
- Danh mục: B-ew (trọng số đều mọi mã VN30 đủ lịch sử) và B-composite top 10 (VN30-2 lần chạy 20261106T121249Z không có
  mô hình thắng, nên dùng B-composite như lộ trình quy định).
- Biến thể: không lớp phủ (đối chứng); `ma200-cash`: ở phiên quyết định, nếu giá đóng cửa chỉ số VN30 < trung bình 200
  phiên thì tỷ trọng 0 (bán hết, giữ tiền mặt), ngược lại 1; `vol-target-15`: tỷ trọng = min(1; 15% / biến động năm hoá
  60 phiên của rổ trọng số đều các mã mục tiêu).
- Tỷ trọng chỉ đổi ở phiên quyết định (mỗi 20 phiên), không đòn bẩy, tiền mặt lãi 0%. Cùng động cơ backtest, phí, thuế,
  T+2 và quy tắc lô như VN30-2; snapshot `20261106-9afb8020ec16`; chỉ cửa sổ phát triển 2018-01-22 → 2023-12-31 (không
  đọc holdout).
- Lớp phủ "có bằng chứng" chỉ khi so với đối chứng CAGR, Sharpe đều cao hơn, MDD nông hơn **và** DSR của chuỗi vượt
  trội hằng ngày ≥ 0,95 (N = số lần thử HOSE trong sổ đăng ký).

**Kết quả (2026-11-06, N = 46): không có bằng chứng (not supported).**

| Danh mục / biến thể | CAGR ròng | Sharpe | MDD | Tỷ trọng TB | Vòng quay/năm | DSR vượt trội |
|---|---|---|---|---|---|---|
| B-ew / không | 1,23% | 0,165 | -48,6% | 1,00 | 1,16 | |
| B-ew / MA200 | 2,39% | 0,245 | -34,9% | 0,43 | 2,37 | 0,011 |
| B-ew / vol 15% | -1,94% | -0,043 | -41,3% | 0,76 | 1,69 | 0,000 |
| B-composite / không | 4,40% | 0,306 | -46,6% | 1,00 | 3,10 | |
| B-composite / MA200 | 5,45% | 0,421 | -35,7% | 0,43 | 2,83 | 0,011 |
| B-composite / vol 15% | 1,09% | 0,149 | -34,6% | 0,75 | 3,05 | 0,000 |

- MA200 giảm MDD khoảng 11-14 điểm phần trăm và nâng CAGR khoảng 1 điểm/năm trên cả hai danh mục, nhưng DSR của chuỗi
  vượt trội chỉ 0,011, rất xa 0,95. Lớp phủ ở trạng thái tiền mặt tại 57% số phiên quyết định (2018-2023 có nhiều giai
  đoạn giảm), nên kết quả phụ thuộc vào vài pha giảm lớn (2018, 2022): 6 năm không đủ để phân biệt với may rủi.
- Mục tiêu biến động 15% giảm MDD nhưng làm giảm CAGR và Sharpe rõ rệt (tiền mặt lãi 0%, giảm tỷ trọng sau các đợt
  biến động mạnh rồi bỏ lỡ nhịp hồi).
- Kết luận: không lớp phủ nào là cải thiện đã được chứng minh. MA200 có thể giữ làm một lựa chọn khẩu vị rủi ro được khai
  báo (giảm sụt giảm), không phải nguồn lợi nhuận. Kết quả đầy đủ:
  `docs/research/evidence/vn30-2/overlay-1-20261106T122937Z/result.json` (thư mục bị gitignore, trong bản checkout chính,
  sha256 `d7521b72…`); mục sổ đăng ký `hose-overlay-1-run`.
