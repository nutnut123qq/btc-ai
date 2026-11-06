# Snapshot nến cho VN30-2

Giao thức [`protocols/vn30-2.json`](protocols/vn30-2.json) yêu cầu mỗi lần chạy đọc đúng một snapshot nến đóng băng và
ghi lại sha256 của nó cùng kết quả. Snapshot nằm ngoài repo, ở `<ROOT>\data\vn30-research\snapshots\<YYYYMMDD>-<sha12>\`.

**Tạo snapshot** (từ thư mục repo, python của venv):

    python scripts/build_vn30_research_snapshot.py

- Nguồn: endpoint công khai VCI gap-chart, `ONE_DAY`, giá đã điều chỉnh. Gọi tuần tự, cách nhau ≥ 1 giây.
- Mã lấy về:
  - mọi mã từng có trong `contracts/vn30-membership.json`, gồm cả mã đã rời rổ;
  - KBC và PVD, hai mã bị loại ở kỳ đầu tiên;
  - chỉ số VNINDEX và VN30;
  - ETF E1VFVN30 và FUEVFVND;
  - hợp đồng tương lai liên tục VN30F1M.
- Khoảng thời gian: từ 2011-01-01 (hoặc ngày niêm yết) đến phiên đã chốt gần nhất. Nến của ngày chạy chỉ được giữ
  khi chạy sau 16:00 giờ Việt Nam.
- Script không bao giờ ghi đè snapshot đã có. Mã nào không lấy được thì được ghi vào mục `missing` của manifest.

**Đọc snapshot:** dùng `vn30_research_data.load_snapshot(path, expected_sha256)`. Hàm kiểm lại mọi hash và từ chối nếu:

- có file lạ hoặc thiếu file;
- tên thư mục không khớp hash;
- hash khác với hash mà người gọi đưa vào.

Hàm trả về:

- `sessions`: lịch phiên theo VNINDEX, bỏ các phiên khối lượng 0;
- `bars(symbol)`;
- `members_on(date)`: thành phần VN30 theo thời điểm, lấy từ bản sao `membership.json` nằm trong snapshot.

Muốn kiểm tra nhanh thì chạy `python vn30_research_data.py verify <thư mục> [--sha256 <hash>]`.

**Snapshot hiện tại:** `20261106-9afb8020ec16`, sha256 `9afb8020ec16a1c26669b046b5621453012e3cd09a5dc24b70da754648acc94e`.

- Gồm 61 mã, không thiếu mã nào.
- Có 3950 phiên, từ 2011-01-04 đến 2026-11-06.

Thư mục `20261106-a9922c1255f9` chứa cùng dữ liệu CSV, byte y hệt, nhưng manifest cũ hơn: chưa loại các nến VHM trước
ngày niêm yết, và phần kiểm tra roll tính sai. Không dùng thư mục này.

**Lưu ý dữ liệu** (chi tiết nằm trong `knownIssues` của manifest):

- **VHM:** mã này đã được dùng lại. Nến trước 2018-05-17 là của một chứng khoán khác. Các nến đó vẫn nằm trong file,
  nhưng `bars()` bỏ chúng đi (`excludeBefore`).
- **Ngày 2018-01-23 và 2018-01-24:** mọi mã HOSE đều có khối lượng 0, nhiều khả năng là dữ liệu nhà cung cấp bị
  thiếu. Hai ngày này không được tính là phiên.
- **Mã chuyển sàn sang HOSE** (ACB, SHB, VIB, LPB, ...): lịch sử gồm cả giai đoạn ở HNX/UPCoM.
- **ROS:** dữ liệu dừng ở 2022-08-11 vì mã bị huỷ niêm yết.
- **VN30F1M:** chuỗi do nhà cung cấp ghép, không điều chỉnh khi roll. Roll diễn ra tại ngày đáo hạn (thứ Năm thứ ba
  của tháng). Đã khớp 308/308 ngày với từng hợp đồng mà nguồn còn cung cấp (từ 2025-05-16). Trước mốc đó chưa kiểm được.
