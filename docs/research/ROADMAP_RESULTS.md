# Kết quả triển khai lộ trình VN30 (đợt 2026-11-06)

Tài liệu này tổng hợp các hạng mục trong lộ trình của `DANH_GIA_DU_AN_VN30.md` đã được triển khai trong worktree `btc-ai-roadmap` (nhánh `roadmap/phase1-3`). Mọi thay đổi đều **chưa commit**. Số liệu chép đúng như các lần chạy đã tính, không làm tròn lại.

> Đây không phải tư vấn đầu tư. Các con số dưới đây đánh giá phương pháp nghiên cứu, không phải khuyến nghị mua hay bán. Quyết định phân bổ vốn là của người dùng.

## 1. Tóm tắt

- **Kết luận chính: không phương pháp nào được kiểm định cho thấy lợi thế có ý nghĩa thống kê so với danh mục đều trọng số VN30 (B-ew) sau chi phí.** VN30-2 cho kết quả no-go: cả hai mô hình ML (ridge, LightGBM) đều trượt cổng phát triển, và đều thua chính baseline composite.
- Quy tắc cắt lỗ -8% / giảm -4% (STOPS-1) và lớp phủ MA200 / vol-target (OVERLAY-1) làm drawdown nông hơn trên cửa sổ phát triển, nhưng không qua cổng DSR. Vì vậy chúng chỉ là **lựa chọn quản trị rủi ro đã khai báo**, chưa phải cải thiện đã được chứng minh.
- Holdout 2024-01-01..2026-09-18 của VN30-2 **đã bị đọc một lần, vĩnh viễn**. Từ nay mọi biến thể phải có giao thức mới và chỉ được đánh giá trên dữ liệu forward.
- Bộ ghi paper forward cho 8 mô hình VN30-2 đã sẵn sàng. Nó chỉ chạy được sau khi code được commit và merge vào checkout chính `btc-ai`. Ngày quyết định forward đầu tiên dự kiến là **khoảng 2026-11-19**; nếu chưa merge kịp thì ngày đó mất hẳn, vì không có backfill.

## 2. Trạng thái từng hạng mục

| Hạng mục lộ trình | Trạng thái | Reviewer |
|---|---|---|
| 1.2 + 1.4 Snapshot VN30 theo thời điểm, gồm mã đã rời rổ, 2011+, bất biến, sha256 | Xong (một phần so với lộ trình, xem 3.1) | Duyệt, 6 vấn đề nhỏ |
| 1.3 Chuỗi VN30, VN30F1M, E1VFVN30 | Có trong snapshot (một phần) | Duyệt cùng 1.2 + 1.4 |
| 1.1 Giá thô, giá tham chiếu, trần/sàn, giá trị giao dịch | Xong | Duyệt, 7 vấn đề nhỏ |
| 2.1 Backtester danh mục đa mã theo luật HOSE | Xong, đã sửa sau review | Duyệt, 8 vấn đề nhỏ |
| 2.3 Bộ đánh giá panel (purge/embargo, rank IC, DSR, PBO, khóa holdout) | Xong, đã sửa một vòng | **Không duyệt**: còn 1 vấn đề lớn và 4 vấn đề nhỏ |
| 3.1 Chạy giao thức đóng băng VN30-2 | Đã chạy một lần: **no-go** | Duyệt, 5 vấn đề nhỏ |
| 2.5 Backtest cắt lỗ -8% / giảm -4% (STOPS-1) | Đã chạy: **không được ủng hộ** | Duyệt, 6 vấn đề nhỏ |
| 3.2 Lớp phủ MA200 / vol-target (OVERLAY-1) | Đã chạy: **không được ủng hộ** | Duyệt, 6 vấn đề nhỏ |
| 2.6 + 3.3 Sổ paper forward cho các mô hình VN30-2 | Code xong, sổ còn trống | Duyệt, 5 vấn đề nhỏ |

Số lần thử HOSE trong trial registry (run + planned, dùng làm N cho DSR) tăng dần trong đợt này: 30 (trước đợt) → 34 (STOPS-1) → 42 (VN30-2 raw-limit) → 46 (OVERLAY-1). Chuỗi hash của registry kiểm tra đạt, 18 mục, head `fc3743d5`.

## 3. Chi tiết từng hạng mục

### 3.1 Hạng mục 1.2 + 1.4 (+ 1.3): snapshot nghiên cứu VN30

**Đã xây dựng**
- `vn30_research_data.py`: bộ ghi, bộ nạp và kiểm tra roll VN30F1M.
- `scripts/build_vn30_research_snapshot.py`: dùng lại `fetch_vn30_candles.fetch` và `probe_mbb_market_data._vci_rows`, không sửa hai file đó.
- `tests/unit/test_vn30_research_data.py` (14 test, đều đạt).
- `docs/research/vn30-research-snapshot.md`.

**Dữ liệu**
- Snapshot cần dùng: `D:\nutnutmeiggy\chungkhoan\data\vn30-research\snapshots\20261106-9afb8020ec16`, sha256 `9afb8020ec16a1c26669b046b5621453012e3cd09a5dc24b70da754648acc94e`, khoảng 9.7 MB.
- Gồm 61 mã, không thiếu mã nào, và 3950 phiên VNINDEX từ 2011-01-04 đến 2026-11-06. Lần tải bắt đầu lúc 17:01 ICT, sau mốc chốt 16:00, nên có nến 2026-11-06.
- Nguồn: OHLCV ngày đã điều chỉnh từ endpoint gap-chart công khai của VCI. Các request chạy tuần tự, cách nhau 1.2 s.
- Phạm vi:
  - 54 mã từng nằm trong một kỳ VN30 của `contracts/vn30-membership.json` (thành viên hiện tại và đã rời rổ);
  - KBC và PVD (rời rổ ngay ở kỳ đầu);
  - VNINDEX, VN30, E1VFVN30, FUEVFVND, VN30F1M.
- Độ phủ tiêu biểu:
  - 27 chuỗi đủ 3952 nến từ 2011-01-04;
  - VN30 index 3682 nến từ 2012-02-06;
  - E1VFVN30 2702 nến từ 2016-01-07;
  - VN30F1M 2294 nến từ 2017-08-10;
  - VHM có 2117 nến dùng được từ 2018-05-17;
  - ROS dừng ở 2022-08-11 (hủy niêm yết);
  - TCX có 261 nến, VPL có 374 nến.
- Quyết định đầu tiên khả thi là 2018-01-22: 28/30 thành viên khi đó có hơn 260 phiên lịch sử.
- Bộ nạp `load_snapshot` fail-closed: nó kiểm tra hash của manifest, hash của từng file, file thừa hoặc thiếu, và số nến cùng ngày đầu/cuối của từng mã.

**Lỗi dữ liệu đã phát hiện và cách xử lý**
1. VHM bị dùng lại mã: nến 2011-11-10..2018-05-16 thuộc một chứng khoán khác. Đã đánh dấu `excludeBefore 2018-05-17` (1412 nến), và mặc định bị loại.
2. 2018-01-23 và 2018-01-24: mọi mã HOSE có khối lượng 0 và giá đứng yên. Nhiều khả năng đây là lỗ dữ liệu của nhà cung cấp. Hai ngày này không được tính là phiên, nên lịch có thể thiếu 2 phiên thật.
3. Các mã chuyển sàn có cả lịch sử HNX/UPCoM trước khi lên HOSE. Chỉ ảnh hưởng đến giai đoạn khởi động đặc trưng.
4. VN30F1M là chuỗi hợp đồng gần liên tục, không điều chỉnh ngược. Từ 2025-05-16, 308/308 ngày so sánh được đều khớp hợp đồng gần nhất; 63 ngày khác không so được vì nguồn không còn phục vụ hợp đồng tương ứng. Lịch roll trước 2025-05-16 chỉ là giả định.

**Review**: duyệt. Các vấn đề nhỏ cần biết:
- Snapshot cũ `20261106-a9922c1255f9` vẫn còn trên đĩa và vẫn nạp được. Manifest của nó không có loại trừ VHM. **Không được dùng.** Xóa hay cách ly nó là quyết định của chủ dự án.
- `expected_sha256` trong `load_snapshot` là tùy chọn. Các entry point nghiên cứu nên luôn truyền hash đầy đủ.
- Chưa xác định VCI có điều chỉnh khối lượng hay không. Nếu khối lượng là số thô, đặc trưng thanh khoản `liq_20` có thiên lệch nhìn trước nhẹ.
- Snapshot chỉ đúng theo thời điểm từ 2018-01-22. Mọi quyết định trước 2018 sẽ bị thiên lệch sống sót.

**So với lộ trình**
- 1.2 chưa làm phần "trong backend" và phần mở rộng universe lên top 100-200 HOSE.
- 1.4 là snapshot CSV chụp một lần, không phải parquet hằng ngày.
- 1.3 có chuỗi giá E1VFVN30, nhưng chưa có cổ tức của ETF.

### 3.2 Hạng mục 1.1: giá thô, giá tham chiếu, trần/sàn

**Đã xây dựng**
- `vn_raw_prices.py`:
  - bộ tải tuần tự, tối thiểu 1.2 s giữa các request, có thể chạy tiếp khi bị ngắt, mỗi file chỉ ghi một lần, manifest có sha256;
  - các hàm bước giá HOSE, `price_limits`, `reference_price` và `band_fraction` (±7%; ±20% ở phiên đầu hoặc sau khi tạm dừng hơn 25 phiên);
  - `RawPrices.limit_locked_open`, `traded_value` và lệnh `crosscheck`.
- `tests/unit/test_vn_raw_prices.py` (46 test, đều đạt).

**Dữ liệu**
- Nguồn: endpoint JSON lịch sử giá công khai của CafeF (robots.txt cho phép). Mỗi bản ghi gồm OHLC thô, giá đóng cửa điều chỉnh, mức thay đổi (dùng để suy ra giá tham chiếu), khối lượng và giá trị khớp lệnh, khối lượng và giá trị thỏa thuận.
- Tải đủ 54/54 mã, 129,378 nến, từ 2016-01-04 đến 2026-11-06. Lưu tại `D:\nutnutmeiggy\chungkhoan\data\vn30-research\raw\` (54 file + `manifest.json`).

**Kết quả kiểm tra**
- Trên 65,197 phiên "sạch" khi mã là thành viên VN30, chỉ 1 phiên vượt trần/sàn tính từ giá đóng cửa thô phiên trước: MWG 2021-08-30, do CafeF bỏ sót điều chỉnh chia cổ tức bằng cổ phiếu.
- Dữ liệu ủng hộ cách làm tròn trần/sàn vào trong: 1,164 phiên giá cao chạm trần và 1,350 phiên giá thấp chạm sàn. Nếu làm tròn gần nhất thì chỉ có 557 và 702.
- Phiên mở cửa bị khóa trần/sàn trên dữ liệu thô: 120 phía mua, 226 phía bán.
- So với proxy của giao thức:
  - phía mua: cả hai cùng bắt 111, chỉ proxy bắt 24, chỉ dữ liệu thô bắt 9;
  - phía bán: cả hai 201, chỉ proxy 67, chỉ dữ liệu thô 25.
- Lỗi của CafeF được báo cáo, không sửa vào file:
  - 109 phiên thành viên có trong VCI nhưng thiếu ở CafeF;
  - một số cụm ngày giá đóng cửa lệch giá chính thức;
  - 165 bất thường trong bản ghi.

**Review**: duyệt. Các vấn đề nhỏ:
- Con số 165 gộp nhiều loại. Đếm lại có 139 bản ghi giá mở cửa ngoài khoảng low..high và 23 bản ghi giá đóng cửa ngoài khoảng; trong phiên thành viên có 4 giá đóng cửa sai, và chúng ảnh hưởng tới giá tham chiếu của phiên sau.
- Cơ sở `exchange` của `reference_price` dùng chính nến của phiên đó. Nó chỉ được dùng để mô phỏng khớp lệnh, không được dùng làm đặc trưng tại thời điểm quyết định.
- Manifest không ghi phiên bản code.
- Quy ước khóa trần/sàn khác `vn_execution_model.classify_band`.

**Ghi chú**: VN30-2 vẫn dùng proxy giá điều chỉnh theo giao thức. Dữ liệu thô chỉ được dùng cho một trial độ nhạy đã đăng ký riêng (mục 3.5).

### 3.3 Hạng mục 2.1: backtester danh mục

**Đã xây dựng**: `vn_portfolio_backtest.py` và `tests/unit/test_vn_portfolio_backtest.py` (27 test sau khi sửa, đều đạt).

**Luật mô phỏng**
- Khớp lệnh ở giá mở cửa phiên d+1. Chọn top 10, giữ mã đang có khi hạng còn ≤ 15, tái cân bằng mỗi 20 phiên.
- Lô 100 cổ phiếu. Phí 0.15% mỗi chiều, thuế bán 0.1%, trượt giá 1 bước giá.
- Cổ phiếu chỉ bán được từ phiên T+2.
- Không khớp khi phiên mở cửa bị khóa trần/sàn (proxy 1.065 / 0.935 hoặc dữ liệu thô). Lệnh trượt được thử lại ở phiên sau.
- Mỗi lần quyết định chỉ xét thành viên đang hiệu lực, có ít nhất 260 phiên lịch sử.

**Kiểm thử**
- Có test vàng tính tay. Ví dụ một vòng mua-bán khép kín: NAV cuối 108,985,487.5.
- Test chống nhìn trước: nhiễu toàn bộ nến sau mốc cắt không làm đổi gì trước mốc.
- Có các test bất biến trên thị trường ngẫu nhiên.

**Sửa sau review**
- Vấn đề lớn: lệnh mua thiếu tiền từng bị bỏ. Nay lệnh được giữ và thử lại khi còn lệnh bán mở hoặc còn tiền bán chưa về.
- Tách hai luật dùng dữ liệu thô: phát hiện khóa trần/sàn, và tính trượt giá theo bước giá thô.
- Thêm các tùy chọn T+2, mồi giá đóng cửa ban đầu, lọc 260 phiên, chi phí chỉnh tỷ trọng mã giữ lại, báo cáo vị thế "treo".

**Review**: duyệt. Các vấn đề nhỏ còn mở:
- Chưa kiểm tra nến NaN hoặc giá ≤ 0.
- Nến khối lượng 0 vẫn được coi là phiên giao dịch. Lớp dữ liệu đã loại chúng, nhưng backtester tự nó không chặn.
- `first_decision_index` dùng 260 cố định.
- Quy mô lệnh tính bằng giá mở cửa d+1, một lợi thế thông tin nhỏ đã được khai báo.
- Vị thế bị hủy niêm yết vẫn được định giá ở giá đóng cửa cuối cùng.
- **Lựa chọn quan trọng**: `cash_settlement_sessions = 0`, nghĩa là tiền bán dùng mua ngay trong phiên và không tính lãi ứng trước. Theo phép thử của reviewer, áp T+2 tiền chặt có thể giảm NAV cuối khoảng 4-8%. Chưa có trial độ nhạy cho lựa chọn này.

### 3.4 Hạng mục 2.3: bộ đánh giá panel

**Đã xây dựng**: `vn30_panel_eval.py` và `tests/unit/test_vn30_panel_eval.py` (29 test, đều đạt).

**Nội dung**
- Purge 21 và embargo 20. Embargo được hiểu là khoảng đệm thêm trước mỗi khối kiểm tra.
- Walk-forward mở rộng, fit lại theo năm.
- Rank IC Spearman theo ngày, chỉ giữ các ngày cách nhau ít nhất 20 phiên. Bootstrap khối theo ngày.
- DSR (Bailey & López de Prado) với N lấy từ registry.
- PBO bằng CSCV, 16 khối, 12,870 cách chia.
- `HoldoutLock` dùng file độc quyền: mỗi protocol chỉ mở holdout được một lần.

**Kiểm tra với đáp án đã biết**
- Ví dụ trong bài báo cho SR0 0.113172 và DSR 0.900397, khớp giá trị công bố 0.1132 và 0.9004.
- PBO trên nhiễu trung bình khoảng 0.53 qua 40 seed. Khi cài sẵn một lợi thế, PBO < 0.05.

**Sửa một vòng**
- Bắt buộc có `last_session_in_window`.
- `fit_predict` không còn thấy cột nhãn.
- Cố định cách ước lượng V[SR].
- Holdout chỉ mở được khi có khóa hợp lệ.
- Cố định thống kê cổng PBO (`gatePbo`, logit ≤ 0).

**Review lần hai: KHÔNG DUYỆT.** Vấn đề lớn còn mở:
- Chốt chặn holdout và chốt chặn tràn cửa sổ chỉ nằm trong `yearly_folds`. Đường tính rank IC cho các baseline (`rank_ic_by_date`, `ic_summary`, `paired_ic_difference`) không bị chặn.
- `walk_forward_scores` cũng nhận Fold tự dựng.

Các vấn đề nhỏ:
- `verify_holdout_lock` chấp nhận khóa lấy ở thư mục khác.
- Phép kiểm `last_session_in_window` có thể trễ tới 19 phiên.
- Cột nhãn viết sai tên không gây lỗi.
- Các lựa chọn ngoài giao thức (ước lượng V[SR], `gatePbo`, `min_names=10`, cách hiểu purge/embargo) chưa được ghi vào phụ lục hay registry.

**Ảnh hưởng tới lần chạy VN30-2**: theo báo cáo lần chạy, pha phát triển cắt thị trường ở 2023-12-29, ghi `dev-results.json` trước khi lấy khóa, rồi mới mở holdout dưới `HoldoutLock`. Như vậy khe hở trên không bị khai thác trong lần chạy này. Dù vậy, **phải sửa nó trước bất kỳ giao thức mới nào**.

### 3.5 Hạng mục 3.1: chạy VN30-2 (run `20261106T121249Z`)

**Đã xây dựng**
- `vn30_2_run.py` và `tests/unit/test_vn30_2_run.py` (16 test, đều đạt).
- Sửa một assertion trong mỗi file `test_vn30_panel_eval.py` và `test_trial_registry.py`, để chúng chấp nhận mục run thay thế `hose-vn30-2`.

**Bằng chứng**
- Thư mục `D:\nutnutmeiggy\chungkhoan\btc-ai\docs\research\evidence\vn30-2\20261106T121249Z\` chứa `results.json` (sha256 `cbdb626bd740c8effd4c8d3eb3abf6eca2a179671b67bc3b81da122619e185d9`), `report.md`, `dev-results.json`, `daily-nav.csv` và `holdout.lock.json`.
- Protocol sha256 là `f9cae5e9...`. Snapshot là `9afb8020ec16...`. Code ở git HEAD `d24cc07` cộng hash phần diff `aa06f7de...`.

**Cổng phát triển** (fold 2019-2023, 62 ngày quyết định không chồng lấn). Một mô hình chỉ qua khi IC > composite, IC > 0 và t ≥ 2.

| Mô hình | IC trung bình | t | Qua cổng |
|---|---|---|---|
| M-ridge | 0.0495 | 1.33 | Không (dưới composite, t < 2) |
| M-lgbm | 0.0462 | 1.53 | Không (dưới composite, t < 2) |
| B-composite (mốc so sánh) | 0.0684 | 1.72 | — |

Hiệu chênh IC từng cặp so với composite: ridge -0.0190 (t -0.68), lgbm -0.0223 (t -0.63). **Kết luận: no-go, không có mô hình thắng ở pha phát triển.**

**Phát triển, mọi mô hình** (danh mục 2019-01-10..2023-12-27, lợi nhuận ròng sau chi phí):

| Mô hình | IC (t) | Vượt B-ew /năm | Vượt VN30 /năm | Vượt E1VFVN30 /năm | CAGR | Sharpe | MDD | Vòng quay /năm |
|---|---|---|---|---|---|---|---|---|
| B-ew | — | — | 0.85% | -0.05% | 6.30% | 0.40 | -39.69% | 1.36 |
| B-mom_12_1 | 0.0535 (1.39) | 2.68% | 3.53% | 2.63% | 8.98% | 0.50 | -43.65% | 3.34 |
| B-mom_6 | 0.0753 (1.91) | -0.06% | 0.80% | -0.11% | 6.24% | 0.38 | -39.21% | 4.49 |
| B-rev_1 | -0.0702 (-2.01) | -12.36% | -11.51% | -12.41% | -6.07% | -0.15 | -51.67% | 12.24 |
| B-lowvol | 0.0222 (0.65) | -1.01% | -0.15% | -1.06% | 5.29% | 0.39 | -37.78% | 3.63 |
| B-composite | 0.0684 (1.72) | 1.42% | 2.28% | 1.37% | 7.72% | 0.45 | -41.33% | 3.34 |
| M-ridge | 0.0495 (1.33) | -1.27% | -0.41% | -1.32% | 5.03% | 0.34 | -38.79% | 5.50 |
| M-lgbm | 0.0462 (1.53) | 0.41% | 1.26% | 0.36% | 6.71% | 0.42 | -37.42% | 10.01 |

Kiểm định nhiều giả thuyết ở pha phát triển (chỉ để tham khảo; DSR tính cho B-mom_6, mô hình có IC phát triển cao nhất): DSR 0.0044, PBO 0.8464.

**Holdout, chỉ báo cáo** (2024-01-24..2026-08-25, 32 ngày quyết định). Cổng holdout không được áp dụng vì không mô hình nào qua cổng phát triển.

| Mô hình | IC (t) | Vượt B-ew /năm | Vượt VN30 /năm | Vượt E1VFVN30 /năm | CAGR | Sharpe | MDD |
|---|---|---|---|---|---|---|---|
| B-ew | — | — | -3.82% | -5.92% | 17.32% | 0.97 | -17.40% |
| B-mom_12_1 (bán niêm phong) | 0.0827 (1.56) | 12.22% | 8.39% | 6.29% | 29.53% | 1.30 | -21.06% |
| B-mom_6 (bán niêm phong) | 0.0507 (0.84) | 2.10% | -1.72% | -3.82% | 19.42% | 0.93 | -24.94% |
| B-rev_1 | -0.0092 (-0.18) | -6.96% | -10.79% | -12.89% | 10.35% | 0.60 | -21.07% |
| B-lowvol | 0.0080 (0.17) | -0.34% | -4.16% | -6.26% | 16.98% | 0.97 | -18.50% |
| B-composite | 0.0627 (1.04) | -2.20% | -6.03% | -8.12% | 15.11% | 0.76 | -24.11% |
| M-ridge | 0.0234 (0.44) | -2.22% | -6.05% | -8.15% | 15.09% | 0.79 | -22.97% |
| M-lgbm | -0.0255 (-0.68) | 4.65% | 0.83% | -1.27% | 21.97% | 1.13 | -17.33% |

- Kiểm định nhiều giả thuyết trên holdout (DSR tính cho M-ridge, mô hình ML có IC phát triển cao hơn): DSR 0.0067 (N = 42, T = 641 lợi nhuận ngày) và PBO 0.3313. Cả hai đều ở phía trượt ngưỡng (DSR ≥ 0.95, PBO < 0.3).
- B-mom_12_1 và B-mom_6 là "bán niêm phong": VN30-1 đã xem chúng trên cửa sổ này, nên các con số đó **không xác nhận được điều gì**.
- M-lgbm vượt B-ew 4.65%/năm nhưng có IC âm (-0.0255). Lợi nhuận và IC trái dấu, nên phần lợi nhuận này không đến từ khả năng xếp hạng.
- Trial độ nhạy khóa trần/sàn theo giá thô (`hose-vn30-2-rawlimits`, 8 trial, đăng ký trước khi tính) chỉ làm lợi nhuận vượt thay đổi tối đa khoảng 0.2 điểm phần trăm mỗi năm. Ví dụ ở pha phát triển, M-ridge là -1.15% thay vì -1.27%.

**Registry**: đã thêm `hose-vn30-2-rawlimits` (planned), `hose-vn30-2-run-20261106T121249Z` (run, thay thế `hose-vn30-2`) và `hose-vn30-2-rawlimits-run-20261106T121249Z`.

**Review**: duyệt. Các vấn đề nhỏ:
- Đã có một lần chạy thử trên dữ liệu thật trước lần chạy chính thức, và nó đã ghi IC cùng kết luận pha phát triển ra đĩa. Code được sửa 25 giây trước lần chạy chính thức; phần sửa chỉ đổi tập mã cho dữ liệu thô. IC và lợi nhuận vượt pha phát triển trùng khớp từng bit giữa hai lần chạy.
- `dry_run` không bắt buộc trial raw-limit phải được đăng ký trước.
- Lựa chọn `cash_settlement_sessions = 0` lạc quan, và không có trial độ nhạy.
- Mẫu phát triển thực tế là 2019-01-10..2023-11-28, vì năm 2018 không có dữ liệu huấn luyện.
- 30/2228 dòng bị loại vì thiếu nến ở d+1 hoặc d+21.
- So sánh với VN30 và E1VFVN30 dùng giá đóng cửa, **không gồm cổ tức**, tính từ đóng cửa ngày quyết định. Trong khi đó danh mục dùng giá điều chỉnh, tức đã gồm cổ tức. Vì vậy cột "vượt VN30" và "vượt E1VFVN30" nghiêng về phía có lợi cho danh mục.
- Holdout đã bị tiêu thụ dù cổng phát triển đã trượt. Giao thức cho phép báo cáo, nhưng VN30-2 không thể chạy lại.

### 3.6 Hạng mục 2.5: STOPS-1 (cắt lỗ -8% / giảm -4%)

**Đã xây dựng**
- Giao thức đóng băng `docs/research/protocols/stops-1.json` (sha256 `6ba6d2dd7f54f3bb9cbc1e6aee5fbe4ffe822c6277f45d4d0e26b5709c6d7fd1`) và `stops-1.md`.
- Mục registry `hose-stops-1` (4 trial) được đăng ký **trước** khi chạy, sau đó là `hose-stops-1-run`.
- `vn30_stops_eval.py`, `tests/unit/test_vn30_stops_eval.py`, và hook tùy chọn `risk_exit` trong backtester.

**Quy tắc** (lấy từ `MbbDecisionService declared-rules-v4` của btc-be, chỉ đọc)
- Cắt lỗ: giá đóng cửa ≤ giá vốn × 0.92 thì bán toàn bộ ở giá mở cửa phiên sau.
- Giảm: khi xu hướng giảm (giá đóng cửa < SMA50 và RSI14 < 50) và giá đóng cửa ≤ giá vốn × 0.96, bán một nửa. Tối đa một lần giữa hai kỳ tái cân bằng.

**Cửa sổ**: 2018-01-22..2023-12-29 (1484 phiên). Mọi nến sau 2023-12-31 bị loại trước khi mô phỏng, nên holdout không bị đọc. N = 34.

| Danh mục / biến thể | CAGR ròng | Sharpe | MDD | Vòng quay /năm | DSR phần vượt so với đối chứng |
|---|---|---|---|---|---|
| P-ew / không | 0.0123 | 0.165 | -0.4865 | 1.16 | (đối chứng) |
| P-ew / stop-8 | 0.0376 | 0.288 | -0.4033 | 3.52 | 0.154 |
| P-ew / stop-8+reduce-4 | 0.0380 | 0.293 | -0.3883 | 4.23 | 0.120 |
| P-mom6 / không | 0.0312 | 0.250 | -0.5271 | 4.28 | (đối chứng) |
| P-mom6 / stop-8 | 0.0333 | 0.262 | -0.4785 | 6.69 | 0.013 |
| P-mom6 / stop-8+reduce-4 | 0.0397 | 0.294 | -0.4579 | 7.56 | 0.019 |

- **Kết luận: cả 4 biến thể đều "không được ủng hộ".** Ước lượng điểm đều tốt hơn đối chứng ở CAGR, Sharpe và MDD, nhưng DSR phần vượt đều thấp xa ngưỡng 0.95. Vòng quay tăng mạnh.
- DSR của chính chuỗi lợi nhuận ngày ròng chỉ 0.04-0.08 cho cả 6 chuỗi.

**Review**: duyệt. Các vấn đề nhỏ:
- Đối chứng ở đây bắt đầu từ 2018-01-22, còn danh mục VN30-2 bắt đầu từ 2019-01-10. Vì vậy số liệu khác nhau: CAGR của B-ew là 0.0123 ở đây so với 0.063 ở VN30-2. Phần lợi của quy tắc nhiều khả năng dựa vào đợt sụt giảm năm 2018.
- Tiền bán từ lệnh cắt lỗ đôi khi tài trợ cho một lệnh mua đang chờ, trái điều khoản của spec. Ảnh hưởng không đáng kể.
- Số lệnh khớp chỉ được báo cáo dưới dạng tổng, không tách theo loại.
- Lệnh giảm cũng thay thế lệnh mua đang mở, điều spec không nói.

### 3.7 Hạng mục 3.2: OVERLAY-1 (MA200, vol-target 15%)

**Đã xây dựng**
- Giao thức đóng băng `docs/research/protocols/overlay-1.json` (sha256 `e0a10361...`) và `overlay-1.md`.
- Mục registry `hose-overlay-1` (4 trial) được đăng ký trước, sau đó là `hose-overlay-1-run`.
- `vn30_overlay_eval.py`, `tests/unit/test_vn30_overlay_eval.py` (11 test), và hook tùy chọn `exposure` trong backtester.

**Biến thể**
- ma200-cash: nếu VN30 đóng cửa dưới trung bình 200 phiên thì tỷ trọng 0, ngược lại là 1.
- vol-target-15: tỷ trọng = min(1, 0.15 / σ60), với σ60 là biến động năm hóa 60 phiên của rổ mục tiêu.
- Chỉ đặt tỷ trọng ở ngày quyết định. Không đòn bẩy. Tiền mặt lãi 0%.

**Cửa sổ**: 2018-01-22..2023-12-31, 75 quyết định, N = 46.

**Kết quả**: chạy một lần (`20261106T122937Z`, `result.json` sha256 `d7521b72d2d37e71143a81a8c6fe58974457cfc2548db82c95d1c3340be5eb00`).

| Chuỗi | CAGR ròng | Sharpe | MDD | Tỷ trọng TB | DSR phần vượt |
|---|---|---|---|---|---|
| P-ew / không | 1.23% | 0.165 | -48.65% | 1.00 | (đối chứng) |
| P-ew / ma200-cash | 2.39% | 0.245 | -34.87% | 0.427 | 0.011 |
| P-ew / vol-target-15 | -1.94% | -0.043 | -41.28% | 0.763 | 0.0001 |
| P-composite / không | 4.40% | 0.306 | -46.57% | 1.00 | (đối chứng) |
| P-composite / ma200-cash | 5.45% | 0.421 | -35.73% | 0.427 | 0.011 |
| P-composite / vol-target-15 | 1.09% | 0.149 | -34.64% | 0.752 | 0.0002 |

- **Kết luận: cả 4 lớp phủ đều "không được ủng hộ".**
- MA200 tốt hơn đối chứng ở cả CAGR (+1.16 / +1.05 điểm phần trăm), Sharpe (+0.080 / +0.115) và MDD (nông hơn 13.8 / 10.8 điểm phần trăm). Tuy nhiên DSR phần vượt chỉ 0.011. Phần vượt ngày có trung bình hơi âm; CAGR cao hơn là nhờ giảm hao hụt do biến động. MA200 ở tiền mặt tại 57% số ngày quyết định.
- Vol-target-15 có MDD nông hơn, nhưng CAGR và Sharpe đều kém đối chứng.
- P-ew / không tái lập đúng đối chứng của STOPS-1 (0.0123 / 0.165 / -0.4865).

**Review**: duyệt. Các vấn đề nhỏ:
- Như STOPS-1, đối chứng khác số liệu VN30-2 vì khác thời điểm bắt đầu.
- Cổng DSR đo chênh lệch lợi nhuận trung bình, không đo chênh lệch Sharpe. Khi tiền mặt lãi 0%, một lớp phủ giảm rủi ro hầu như không thể qua cổng này. Đây là giới hạn của thiết kế đã đóng băng, không phải sai lệch so với giao thức.
- Phép kiểm composite không kiểm tra tập mã đủ điều kiện.
- Thêm hook `exposure` làm đổi sha256 của `vn_portfolio_backtest.py` so với giá trị VN30-2 đã ghi. Bỏ riêng hook thì tái lập đúng sha `fb75cac8...`, nên hook là thay đổi duy nhất.

**So với lộ trình 3.2**
- Tiêu chí "MDD ngoài mẫu thấp hơn E1VFVN30 sau phí" **chưa đo được**, vì holdout đã tiêu thụ và OVERLAY-1 chỉ chạy trên cửa sổ phát triển.
- Tùy chọn phòng hộ bằng VN30F chưa được thử.

### 3.8 Hạng mục 2.6 + 3.3: sổ paper forward VN30-2

**Đã xây dựng**
- `vn30_2_forward.py`, với các lệnh `record`, `evaluate`, `verify`.
- `tests/unit/test_vn30_2_forward.py`: reviewer đếm lại được 27 test, không phải 29 như báo cáo. Bốn file test liên quan cộng lại là 82 test, đều đạt.
- Các script: `scripts/run_vn30_2_forward.ps1`, `scripts/install_vn30_2_forward_task.ps1`, và chỉnh `scripts/run_vn30_screen.ps1` để gọi `record` sau bước ghi forward.

**Cách hoạt động**
- Lưới quyết định: mỗi 20 phiên kể từ 2018-01-22, nối tiếp trên lịch thực. Lần ghi đầu tiên được phép từ 2026-11-09 trở đi.
- Sổ ghi tại `D:\nutnutmeiggy\chungkhoan\data\forward\vn30-2.jsonl`, nối chuỗi hash, không backfill. Mỗi bản ghi phải được ghi trước 09:00 ICT của phiên kế tiếp.
- Đặc trưng và mô hình dùng đúng code của VN30-2, fit lại theo năm như giao thức. Có test xác nhận điểm số khớp `walk_forward_scores` tới 1e-12.
- Dữ liệu lịch sử lấy từ snapshot (đã kiểm hash). Phiên mới lấy từ backend 127.0.0.1 qua GET, được kiểm với lịch HOSE và với 60 phiên chồng lấn.
- `evaluate` tính rank IC và lợi nhuận ròng bằng chính `run_backtest` theo luật giao thức.
- Nếu dữ liệu chưa đủ, chương trình fail-closed: trả `data-not-ready` (mã thoát 4) hoặc `missed-deadline` (mã thoát 3).

**Dữ liệu**
- Sổ chính thức chưa có dòng nào. Ngày 2026-11-06 chưa phải ngày quyết định: ngày quyết định gần nhất là 2026-10-22, ngày kế tiếp cách 9 phiên, khoảng 2026-11-19 (con số tính ra, chưa quan sát).
- Mô phỏng một quyết định ngày 2026-10-22 trên sổ nháp ghi được bản ghi hợp lệ: 29 mã đủ điều kiện, không có khoảng trống dữ liệu live, danh mục backend khớp 30 thành viên.
- Giá đóng cửa backend khớp snapshot, 0 sai lệch trên MBB, FPT, VIC, SSB và VNINDEX.

**Review**: duyệt. Các vấn đề nhỏ:
- **Một quyết định không ghi được vẫn có thể bị mất trong im lặng**: phiên sau đó trả `not-a-decision-date` với mã thoát 0. Điều này quan trọng vì hệ thống sẽ chạy không người trông.
- Cách tính kỳ cuối không đồng đều với các kỳ trước.
- Cách năm hóa (12.6 kỳ/năm) khác `vn30_2_run`.
- `splice()` chấp nhận mọi tỷ lệ không đổi, kể cả khi đổi đơn vị giá.
- Task thử lại `BTCVn30TwoForward` chưa được đăng ký.

## 4. Kết luận trung thực

**Có phương pháp nào thắng VN30 đều trọng số (B-ew) hoặc chỉ số sau chi phí không? Không có bằng chứng.**

1. **Mô hình ML (ridge, LightGBM)** thua chính baseline composite trên cửa sổ phát triển, t < 2, nên trượt cổng đã đóng băng.
   - Trên holdout, M-ridge thua B-ew 2.22%/năm.
   - M-lgbm vượt B-ew 4.65%/năm nhưng có IC âm.
   - DSR holdout là 0.0067 với N = 42.
   - Kết quả này đúng với điều lộ trình đã dự báo: chỉ với 30 mã và đặc trưng giá, cổng t > 3 gần như không thể đạt.
2. **Baseline đơn giản.**
   - Momentum 12-1 vượt B-ew 2.68%/năm ở pha phát triển và 12.22%/năm trên holdout.
   - Nhưng holdout đó đã bị VN30-1 xem trước (bán niêm phong). DSR phát triển của nhóm là 0.0044 và PBO là 0.8464.
   - Vì vậy không thể coi đây là lợi thế đã được xác nhận.
3. **So với chỉ số.**
   - Ở pha phát triển, B-ew gần như hòa E1VFVN30 (-0.05%/năm).
   - Trên holdout, B-ew thua VN30 3.82%/năm và thua E1VFVN30 5.92%/năm, dù phép so sánh đã nghiêng về phía danh mục (danh mục gồm cổ tức, chỉ số và giá ETF thì không).
   - Trong 2024-2026, chuẩn mua-và-giữ E1VFVN30 khó đánh bại hơn cả danh mục đều trọng số.
4. **Cắt lỗ -8% / giảm -4% và MA200** làm MDD nông hơn khoảng 5-15 điểm phần trăm trên cửa sổ phát triển. Tuy nhiên:
   - không biến thể nào qua cổng DSR;
   - chỉ có một cửa sổ 6 năm, phần lợi nghiêng về năm 2018;
   - vòng quay của các biến thể cắt lỗ tăng mạnh.

   Có thể giữ chúng như **lựa chọn quản trị rủi ro theo khẩu vị**, nhưng không được trình bày là "đã chứng minh tăng lợi nhuận".
5. Theo tiêu chí 3.4 của lộ trình, kết luận nghiêng rõ về **no-go cho alpha từ dữ liệu chỉ có nến**. Hệ thống nên chuyển sang vai trò quản trị rủi ro và đo so với chuẩn mua-và-giữ E1VFVN30. Quyết định cuối thuộc về chủ dự án, sau giai đoạn paper forward.

## 5. Việc còn lại

### 5.1 Việc chỉ thời gian mới làm được

- **Paper forward 6-12 tháng** (3.3) cho 8 mô hình VN30-2:
  - Ngày quyết định đầu tiên dự kiến khoảng 2026-11-19. Bản ghi đầu tiên đáo hạn (có giá mở cửa d+21) khoảng tháng 12/2026.
  - Mỗi năm có khoảng 12-13 kỳ quyết định 20 phiên. Sau 6-12 tháng mới có khoảng 6-12 điểm IC: đủ để kiểm tra độ tỉnh táo (IC live ≥ 1/2 IC backtest), **không đủ để xác nhận thống kê**.
  - Vì VN30-2 là no-go, cổng forward của giao thức không áp dụng cho mô hình nào; sổ forward chỉ để quan sát.
- Mọi ý tưởng mới (biến thể VN30-3, phòng hộ VN30F, tiền mặt có lãi) phải đóng băng giao thức mới, đăng ký trước, và chỉ đánh giá trên dữ liệu forward, vì holdout 2024-2026 đã dùng.

### 5.2 Việc chủ dự án hoặc coordinator cần làm (theo thứ tự ưu tiên)

1. **Commit và merge vào `btc-ai` trước khoảng 2026-11-19.** Task định kỳ đọc checkout chính; nếu chưa merge thì ngày quyết định đầu tiên mất hẳn.
2. Sau khi merge, chạy `scripts\install_vn30_2_forward_task.ps1` ở `btc-ai`: `-WhatIf` trước, rồi cài thật, rồi `-AuditOnly`. Đây là thay đổi cấu hình lâu dài nên cần người dùng quyết định.
3. Thứ tự commit `vn_portfolio_backtest.py`: commit bản trước hook `exposure` (sha `fb75cac8...`) cùng lần chạy VN30-2, sau đó commit hook (sha `48723baa...`, đã ghi trong kết quả OVERLAY-1).
4. Chuẩn hóa `registry.jsonl` về LF (thay đổi `.gitattributes` `*.jsonl text eol=lf`; hiện file trộn CRLF và LF), rồi kiểm lại chuỗi hash.
5. Cách ly hoặc xóa snapshot cũ `20261106-a9922c1255f9`.
6. Sửa vấn đề lớn của bộ đánh giá 2.3 (đường rank IC của baseline không bị chặn holdout) và ghi phụ lục các lựa chọn ngoài giao thức, trước bất kỳ giao thức mới nào.
7. Sửa lỗi forward ghi thiếu mà không báo (mã thoát 0) để chạy không người trông an toàn hơn.
8. Sau mỗi kỳ review VN30, cập nhật cả `contracts/vn30-membership.json` và danh mục mã của btc-be. File membership phải được cập nhật trước khoảng 2027-02-09, nếu không `record` sẽ từ chối chạy. Lịch HOSE 2027 hiện vẫn tạm thời và phải gia hạn trước 2028.
9. Chạy lại toàn bộ test khi máy rảnh bộ nhớ. Lần gần nhất: 2721 đạt, 5 bỏ qua, 1 lỗi môi trường (`test_technical_evidence_launcher` cần `btc-ai-roadmap\venv`, worktree không có). Một số lần trước có MemoryError hoặc timeout khi máy tải nặng; các test đó chạy riêng thì đạt.

### 5.3 Việc chưa làm và lý do

| Việc | Lý do |
|---|---|
| 1.6 Dữ liệu trực giao (khối ngoại, room, cơ bản theo thời điểm) từ SSI FastConnect / FiinGroup | Cần quyết định mua dữ liệu trả phí. Đợt này chỉ được dùng dữ liệu công khai miễn phí. Thiếu nó, cổng 3.1 trên thực tế khó đạt |
| 1.2 Mở rộng universe lên top 100-200 HOSE; universe theo thời điểm trong backend | Ngoài phạm vi đợt này (chỉ làm snapshot nghiên cứu ngoài repo). Đây là cách chính để tăng độ rộng mẫu |
| 1.3 Cổ tức của E1VFVN30 | Snapshot chỉ có chuỗi giá; so sánh với ETF đang dùng giá đóng cửa |
| 1.4 Snapshot parquet hằng ngày | Mới có một snapshot CSV bất biến; chưa tự động hóa |
| 2.4 Bậc thang baseline đầy đủ (bảng riêng cho E1VFVN30 mua-và-giữ có IR, ghi registry riêng) | Phần lớn đã nằm trong VN30-2 (B-ew, 12-1, 6M, đảo chiều 1M, biến động thấp, composite) và OVERLAY-1 (MA200); chưa có bảng riêng |
| 3.2 Phòng hộ VN30F; đo MDD ngoài mẫu so với E1VFVN30 | Chưa thiết kế giao thức; holdout đã tiêu thụ nên chỉ còn đo được bằng dữ liệu forward |
| Các trial độ nhạy: T+2 tiền chặt, tiền mặt có lãi tiền gửi, kiểm định chênh lệch Sharpe (HAC) | Mỗi cái là một trial mới phải đăng ký trước; chưa đăng ký nên chưa chạy |
| 1.5, 2.2 và các mục giai đoạn 0 | Không được xử lý trong đợt này; báo cáo này không đánh giá tình trạng của chúng |
| 3.4 Quyết định go/no-go chính thức; 3.5 cập nhật website | 3.4 phụ thuộc kết quả forward và là quyết định của chủ dự án; 3.5 nằm ở `btc-fe`, nơi đợt này không được phép sửa |
