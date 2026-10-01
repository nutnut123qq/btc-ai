# BTC Technical-Event Descriptive Evidence — Tóm tắt cho người đọc

> Mục đích: trả lời câu hỏi "bằng chứng đâu?" bằng số liệu có thể kiểm chứng.
> Đây là **descriptive evidence** — lịch sử quan sát có đối chứng, **không phải** dự báo,
> tín hiệu giao dịch, tỷ lệ thắng, hay bằng chứng nhân quả/pnl.

**Run mới nhất:** `a10d1d1ad169fa2abf77ba38907e593e021069638d8f68368887727194df3ac3`
(hoàn tất 2026-10-01T09:24:52Z, status `succeeded`, cả 3 khung `semanticVerification=true`).

## 1. Đo cái gì, trên dữ liệu nào

- Symbol duy nhất: **BTCUSDT**; khung: **1h, 4h, 1d**.
- Mỗi "sự kiện" là một điều kiện kỹ thuật được phát hiện trên nến **đã đóng và đã có sẵn**
  (availability time — điểm dữ liệu thực sự xuất hiện, không phải open-time danh nghĩa).
- Mỗi sự kiện được ghép với **matched controls**: các thời điểm cùng context nhưng không có
  sự kiện, được chọn **strictly trước** thời điểm quyết định — outcome của control phải kết
  thúc trước khi event được quyết định. Đây là ràng buộc chống look-ahead ở cấp dữ liệu.
- Outcome đo ở **horizon cố định** 1/3/6 nến, 3 metric: `forwardReturn` (close-to-close),
  `mfe` (maximum favorable excursion), `mae` (maximum adverse excursion).
- Family khai báo trước: **378 hypotheses** = 8 module × ≤42 event-type × 3 horizon × 3 metric.
  Mọi hypothesis trong family đều được giữ lại trong báo cáo — kể cả âm tính, không đủ mẫu,
  hay không kiểm định được (no result-based filtering).

| Khung | Nến lưu trữ | Sự kiện eligible | Bị loại | Lý do loại chính |
|-------|------------:|-----------------:|--------:|------------------|
| 1h | 162,956 | 127,531 | 35,425 | overlap_deduplicated 29,492; unknown_availability 5,741; non_contiguous 192 |
| 4h | — | 31,764 | 14,773 | (cùng cơ chế) |
| 1d | — | 5,154 | 1,141 | (cùng cơ chế) |

`realizedAtMaxHorizon` ≈ `eligible` → gần như toàn bộ sự kiện đã có outcome đầy đủ tới
horizon xa nhất tại thời điểm cutoff.

## 2. Thống kê như thế nào

- Ước lượng: **paired difference** event − matched controls (cùng context), báo cáo mean,
  median, standardized mean difference (d), phân số dương/âm.
- Khoảng tin cậy: **moving block bootstrap** trên chuỗi sự kiện theo thứ tự thời gian
  (2000 resample, block = 8 events) — vì outcome window chồng lấn, KHÔNG giả định độc lập.
- Đa kiểm định: **Benjamini–Yekutieli FDR** ở q = 0.05 trên toàn bộ family đã khai báo
  (BY valid under arbitrary dependence, conservative).
- Chẩn đoán mẫu (per hypothesis, đều được giữ trong report): nominal matched pairs,
  unique decision times, **maximum greedy non-overlapping outcome windows** (tập con
  không chồng lấn lớn nhất), **effective sample size** theo autocorrelation bậc ≤20.

## 3. Kết quả headline

| Khung | Hypotheses | Đủ mẫu để test | Đạt BY-FDR q<0.05 | …trong đó n≥50 cặp |
|-------|-----------:|---------------:|------------------:|-------------------:|
| 1h | 378 | 360 | 227 | 211 |
| 4h | 378 | 333 | 140 | 131 |
| 1d | 378 | 297 | 26 | 25 |

Top effect có **n ≥ 50 cặp** (ảnh chụp mạnh nhất có mẫu đủ lớn):

| Khung | Sự kiện | Horizon / Metric | d | Mean diff | Bootstrap CI | q | n |
|-------|---------|------------------|---:|----------:|--------------|---:|--:|
| 4h | volumeProfile / CLOSE_ABOVE_VAL | h6 fwdRet | +0.48 | +2.03% | [+1.46%, +2.50%] | 0.011 | 375 |
| 4h | volumeProfile / CLOSE_BELOW_VAH | h6 fwdRet | −0.49 | −1.90% | [−2.31%, −1.51%] | 0.011 | 468 |
| 4h | causalSmc / FVG_BULL | h6 mfe | +0.46 | +1.19% | [+0.99%, +1.41%] | 0.011 | 982 |
| 1h | causalSmc / FVG_BULL | h6 mfe | +0.37 | +0.48% | [+0.44%, +0.54%] | 0.006 | 4,196 |
| 1h | volumeProfile / CLOSE_ABOVE_VAL | h6 fwdRet | +0.37 | +0.89% | [+0.74%, +1.05%] | 0.006 | 1,605 |
| 1d | volumeProfile / CLOSE_ABOVE_VAL | h6 fwdRet | +0.62 | +6.42% | [+3.84%, +7.75%] | 0.036 | 65 |

Cách đọc: sau sự kiện "đóng trên Value Area High" khung 4h, forward return 6 nến trung bình
cao hơn matched control ~2.0% (CI bootstrap [1.46%, 2.50%], q-BY = 0.011). Đây là mô tả
lịch sử, không phải cam kết tương lai.

Lưu ý trung thực: report cũng chứa các hypothesis `marketRegime/*` có |d| lớn (ví dụ d≈12)
nhưng với **n = 2–8 cặp** — hiệu ứng đó thống kê yếu về mẫu, được giữ lại đúng chính sách
retention chứ không phải kết luận. Bảng trên cố tình chỉ lấy n≥50.

## 4. Giới hạn đã khai báo (trong report)

- `independenceClaimed = false` — outcome windows được phép chồng lấn; bootstrap block là
  cơ chế xử lý phụ thuộc, không phải giả định độc lập.
- `economicClaim = false`, `probabilityClaim = false`, `promotionAllowed = false`.
- Effective sample size (autocorrelation-adjusted) nhỏ hơn nhiều so với nominal pairs trên
  một số hypothesis — ESS được report per-hypothesis, đọc kèm trước khi kết luận.
- `sensitivityAudit` và `stability` (theo năm/regime) nằm trong report đầy đủ; một số hướng
  nâng cấp (block-size tuning, sensitivity mở rộng) đã được audit độc lập và defer có chủ đích.

## 5. Provenance — kiểm chứng được

- Run pointer: `latest-success.json` → `runIndexSha256 = a10d1d1a…df3ac3`
- Manifest per khung: `1h 9775fe0d854a1a75…`, `4h fea49eb2c78c530f…`, `1d cc34398eddc16bb9…`
- Contract definitions: `921f98700bc2afab…a46057`
- Mọi artifact là **content-addressed**: tên file = sha256 nội dung. Backend chỉ publish
  qua run pointer nguyên tử và re-verify hash trước khi phục vụ.
- Reproduce: `pwsh ai/scripts/run_technical_evidence_pipeline.ps1` (lock + atomic publish +
  semantic verification + retention). Snapshot/ledger/report hash phải khớp byte nếu dữ
  liệu đầu vào và code không đổi.

Artifacts nằm tại `ai/docs/research/evidence/technical-descriptive/` (không commit vào git
vì kích thước ~700MB; integrity vẫn được bảo đảm bởi hash + manifest + backup DB định kỳ).
