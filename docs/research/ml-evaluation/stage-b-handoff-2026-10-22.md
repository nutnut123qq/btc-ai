# Bàn giao nghiên cứu BTC 4h + chuẩn bị Stage B — 2026-10-22

Tài liệu bàn giao, chỉ đọc/đối chiếu. Lượt này **không** chấm reserve, không
train/calibrate, không đổi registry/model/DB/config, không deploy. Đi kèm
`stage-b-runbook.md` (thủ tục), `evaluation-protocol.md` (tiêu chí),
`holdout-assessment.md` (định nghĩa reserve), `quarantine-provenance.md`
(lý do quarantine).

Evidence local (ngoài 3 repo, không commit):
`D:\code\btc\evidence\btc-research-handoff-20261022\`
- `server-ai-dirty\` — bản sao các file dirty trên server AI chưa có trong git
  (giữ đường dẫn tương đối, sha256 khớp server, xem §2).
- `probes\` — script count-only/drift-check đã dùng + output JSON/text.

## 0. State summary (kiểm 2026-10-22 22:08–22:20 ICT = 15:08–15:20Z)

| Hạng mục | Trạng thái thật | Evidence |
|---|---|---|
| Git local | backend/ai/frontend sạch; HEAD `dd3c22e` / `4eaf335` / `211cced` | `git status` |
| Git server | BE `dd3c22e`, FE `211cced` sạch; AI `4eaf335` + **17 mục dirty** (27 file, untracked dir gộp) | `probes/server_ai_dirty_inventory_20261022.txt` |
| Reserve Stage B | **99/150 — NOT_READY**, thiếu 51 (strict 240: thiếu 141) | `probes/reserve_count_20261022T151417Z.json` |
| Model serving | `status: quarantined`; AI `/api/predict/models` → `{"models":[]}`; BE `/api/prediction/latest` → 503 `MODEL_ARTIFACT_INCOMPATIBLE` | `probes/server_ops_health_20261022.txt` |
| Paper journal | 123 bản ghi BTCUSDT 4h, 100% `abstain/model-unavailable`, ModelVersion null, **0 fill, 0 outcome**; 123 nến tín hiệu liên tục 2026-10-02 00:00Z → 10-22 08:00Z, 1 RecorderVersion | `probes/paper_observations_20261022.txt` |
| Dataset builders | 7 worker healthy (Embedding `partial` — timeout mạng, không liên quan Stage B); MlDatasetBuilder succeed 15:10Z; WindowDatasetBuilder succeed 10:15Z (chu kỳ 6h); 5 hàng reserve tạo trong 24h, 42 trong 7 ngày (≈6/ngày) → **đang chờ mẫu, không phải pipeline đứng** | cùng file |
| Technical/SMC replay | 1h/4h/1d `checkpointStatus=complete` nhưng `lastProcessedCloseTimeMs` = 10-11 14:59Z / 10-11 11:59Z / 10-10 23:59Z — đứng ở lần rebuild 11/10, **không** cập nhật tới hôm nay | `GET /api/smart-money/replay/coverage?timeframe=…` |
| Watchdog | Vòng restart mỗi giờ :12:26 kết thúc sau 10-14 22:12 ICT. Sau đó: 10-19 03:12 (backend-workers: KlinesIngestionWorker fail lúc 03:01) và 4 lần 10-22 (10:10, 10:25, 16:48, 17:04 — "Unable to connect", khớp cửa sổ deploy equity lane) | `watchdog.log`; `D:\code\btc\evidence\equity-expansion\deploy-20261022\README.md` |

## 1. Kết luận nghiên cứu còn hiệu lực

1. **Model cũ bị quarantine — giữ nguyên, không hồi sinh.**
   `BTCUSDT_4h_ws5_h4h_XGB_v20260901025507` (byte-identical với `v20260825`,
   sha256 `1f1542d7…`). Lý do (đã tái lập, `quarantine-provenance.md`): fail
   macro F1 / balanced accuracy / MCC / per-class F1 (không dự đoán lớp up);
   `val_acc 0.9378` bị nhiễm calibration (replay thật 0.521); label contract
   trước rebuild 21/09; test window đã tiêu. Protocol §4: artifact này tốt nhất
   chỉ là footnote.
   Trên server còn thêm lý do fail-closed thứ hai: registry trỏ
   `models/BTCUSDT_4h_ws5_h4h_XGB_v20260901025507.joblib` nhưng file đã bị
   `run-ml-pipeline.ps1` (05/10) move sang `models/_archive_ws5_h4h_20261005/`.
   Check đầu tiên (`status != active`) đã chặn trước, nên đây là thông tin cho
   Stage C, không phải lỗi đang chảy.
2. **Candidate/recipe freeze cho Stage B = HGB causal (v3), không phải XGB.**
   `stage_b_reserve_eval.py`: `v3._fit_calibrated_hgb` (HGB + PriorOnlyCalibrator),
   cột causal `_feature_columns(5)` = 170 (175 stored bỏ `ActiveRuleCount`),
   label `PriceTargets.TargetDirection4h` (close-to-close), fit = hàng có
   label_available ≤ cutoff trừ 120 hàng cuối làm calibration, seed 42.
   Gate = 9 check `PROMOTION_THRESHOLDS` + mọi khoảng paired block bootstrap
   (block 6/12/24, 1000 mẫu, Bonferroni) so với baseline v3 phải > 0.
3. **Stage A (retrospective) — bundle freeze 05/10** `evidence/v3_20261005_050530/`
   (chỉ có trên server, chưa vào git; đã copy local):
   `promotionGate.status = retrospective_screen_passed`, `confirmatory = false`,
   `promotionAllowed = false`; 117 fold, 14,040 hàng đánh giá; baseline mạnh nhất
   `adaptive_class_prior_180_rows`; dataset 14,674 hàng, `datasetSha256 e876003b…`,
   `decisionCutoffMs 1791151530514`; git `0addeb3` **dirty=true**; runtime
   python 3.12.8 / numpy 2.5.2 / sklearn 1.9.0. Đây là bằng chứng *sàng lọc*,
   không chứng minh hiệu quả đầu tư.
4. **Discovery 4h 128 trials → 0 rule = kết quả nghiên cứu, không phải bug**
   (TASKS.md, verdict 10/15): selection WR 0.390, net −0.31%; 2 trial qua
   selection đều chết held-out. Lever là candidate generation/cost model theo TF,
   không phải nới gate.
5. **Paper journal chứng minh được:** recorder chạy idempotent, ghi đủ mỗi nến
   4h đóng từ 02/10, fail-closed đúng (abstain khi không có model). **Chưa chứng
   minh:** bất kỳ quyết định giao dịch, fill, outcome hay hiệu quả nào — chưa có
   observer fill/outcome và chưa có model active.
6. **Snapshot của evidence:** Stage A = snapshot DB cutoff 2026-10-04T22:05:30Z.
   Replay SMC = checkpoint 10–11/10. Backtest run id=8 (10/11) = model
   quarantined trên cửa sổ 2026-04-25→08-24 (pre-cutoff, retrospective).
   Các report 10/05 khác (§2) là pre-cutoff, đã "spent".

## 2. Inventory 17 mục dirty trên `E:\BTC\btc-ai` (không mục nào bị dọn)

Owner chung: phiên freeze 05/10 ~04:50–05:42 ICT chạy `D:\code\btc\run-ml-pipeline.ps1`
+ `run-ml-validate.ps1` (script local, ngoài git) và phiên deploy 11/10
(TASKS.md dòng ~621-625). Session cụ thể: **unknown**.

| # | St | Path | Size / sha256 (đầu) | Nguồn gốc | Stage B? | Serving? |
|---|---|---|---|---|---|---|
| 1 | M | `baseline_advanced_report.md` | 1766 / `07551219` | `train_baseline_advanced.py` 10/05 04:55 (chỉ 4h ws5 h4h, 14,674 mẫu; best LGB acc 0.618 < majority 0.639) | không | không |
| 2 | M | `calibration_report_BTCUSDT_4h_ws5_h4h.json` | 1564 / `c22e9f2c` | `calibrate_threshold.py` 10/05 05:04 (XGB tự fit trong script, test từ 2025-07-08 — spent) | không | không |
| 3–10 | D | `models/BTCUSDT_4h_ws5_h4h_XGB_{balanced,calibrated,v20260825,v20260901025507}.{joblib,json}` | HEAD blob: `b1792d86`,`0e8b52ad`,`7163e3bb`,`0f8daa7e`,`a66fb1f7`,`4b725a2c`,`a66fb1f7`,`05320b1b` | move bởi `run-ml-pipeline.ps1` step 0 | không | registry vẫn trỏ #9 (xem §1.1) |
| 11 | ?? | `models/_archive_ws5_h4h_20261005/` (8 file) | git blob của **cả 8** = HEAD blob tương ứng (khác byte chỉ do CRLF ở .json) | đích move của #3–10 | không | không |
| 12 | ?? | `evidence/v3_20261005_050530/` (4 file) | report `68773ff7`, dataset.npz `efb0c8f4`, predictions `070b5f66`, manifest file `7d662f64…` (sha256 nội dung `00218223`) | `ml_evidence_v3.py` freeze 10/05 05:42 | **provenance của recipe + cutoff** | không |
| 13 | ?? | `models/BTCUSDT_4h_ws5_h4h_LGB_balanced.{joblib,json}` | `3d807fe7` / `3c76c81e` | trainer 10/05 04:55; không đăng ký, manifest thiếu trường gate → không servable | không | không |
| 14 | ?? | `backtest_report_BTCUSDT_4h_ws5_h4h_XGB_v20260901025507.json` | 635505 / `2fb20ee8` | tạo 10/05 04:50, **ghi đè** 10/11 22:39 bởi rerun backtest id=8; nội dung 10/05 không còn (UNVERIFIED) | không | không |
| 15 | ?? | `stage_b_reserve_eval.remote-copy` | 5811 / `be133f7b` | bản script cũ trước `d14e9c3` (không `--strict`, comment TZ sai), rename 10/11 trước pull; logic chấm giống bản hiện hành | không dùng | không |
| 16 | ?? | `wfv_ws5_h4h_20261005.json` | 14132 / `fdc62cac` | `walkforward_validate.py` 10/05 04:58 (9 fold, legacy tool) — retrospective | không | không |

Không mục nào là "đã mất": 8 file D có bản byte-khớp trong `_archive` **và** trong
git HEAD (repo local còn nguyên ở `ai/models/`). Bản sao local + hash đầy đủ:
`server-ai-dirty\` (#1, #2, #12–#16, `modified-files.diff` sha256 `dc863846…`,
wrapper `_tmp\run-stageb.ps1` `a1751aa1…`). Model #3–#11 không copy vì đã có trong git.

**Chưa bàn giao xong (cần user quyết, không làm trong lượt này):** bundle #12 là
evidence bắt buộc của Stage B nhưng chưa nằm trong repo nào. Đề xuất: đưa vào
`ai/docs/research/evidence/` theo convention content-addressed — cần duyệt riêng
(≈21 MB).

## 3. Reserve count-only (production DB, read-only)

Không chạy `stage_b_reserve_eval.py`. Probe `probes/stage_b_reserve_count_only.py`
(sha256 `1ecc1db8…`) lặp đúng predicate của `load_btc_4h_benchmark` +
`stage_b_reserve_eval.py`, chỉ SELECT timestamp/COUNT, session
`transaction_read_only=on`:

- Cohort: `BTCUSDT`/`4h`/WindowSize 5/Horizon `4h`/FeatureDim 175, JOIN
  `PriceTargets` (Symbol, Timeframe, `OpenTimeMs = WindowEndMs`),
  `TargetDirection4h IS NOT NULL` (không đọc giá trị).
- decision = WindowEndMs + 4h; label_available = decision + 4h ≤ thời điểm kiểm;
  reserve = decision > `FROZEN_CUTOFF_MS = 1791151530514` (2026-10-04T22:05:30.514Z).

Kết quả lúc **2026-10-22T15:14:17Z**:

| Trường | Giá trị |
|---|---|
| reserveRows | **99** (distinct 99) → `NOT_READY`, thiếu **51** tới 150, **141** tới 240 |
| decision đầu / cuối | 2026-10-05T00:00Z / 2026-10-21T08:00Z (WindowEnd cuối 10-21 04:00Z = 11:00 ICT) — 99 nến liên tục, không gap |
| hàng post-cutoff bất kỳ / lệch FeatureDim / chờ label | 99 / 0 / 0 |
| trùng WindowEndMs / vector lỗi (len≠175 hoặc NaN) toàn cohort | 0 / 0 (loader `validate()` sẽ không nổ) |
| fit pool (label_available ≤ cutoff) | 14,679; 1 hàng biên (decision 10-04 20:00Z) không thuộc bên nào — đúng thiết kế |

**ETA (ước lượng, giả định ingestion không gián đoạn, 6 nến/ngày, lag
WindowEnd→hàng reserve ≈ 28–36h như quan sát hôm nay):** hàng thứ 150 có
WindowEnd 2026-10-29 16:00Z → xuất hiện ~**31/10 sáng ICT**; hàng 240 → ~15/11.

**Drift pre-cutoff** (`probes/frozen_drift_check.py`, loader với cutoff = freeze
nên SQL không chọn hàng reserve; chỉ in hash/đếm lệch): 14,674 hàng DB khớp
snapshot **byte-for-byte** (`e876003b…`, 0 thiếu, 0 lệch feature, 0 lệch label).
DB có thêm **5 hàng** pre-cutoff materialize sau freeze (decision 10-04
00:00–16:00Z) → sẽ vào fit/calibration của Stage B dù không có trong snapshot.

## 4. Gói chạy Stage B — đã đối chiếu, CHƯA chạy

### 4.1 Đối chiếu

- Hash local = server = giá trị freeze ghi trong bundle #12:
  `ml_evidence_walkforward.py c977ebe6…`, `ml_evidence_v3.py ac244d29…`
  (= `evaluatorSha256` / `v3EvaluatorSha256`). `stage_b_reserve_eval.py`
  `757c46c5…` (blob `00ef4c29`, HEAD), `rolling_retrainer.py ab2677ce…`,
  `prediction_service.py 468e858e…`. `db_config.py` khác byte (CRLF) nhưng
  không dirty trong git. Từ 2026-10-04 tới nay chỉ có 2 commit chạm các file
  trên: `7965493` (10/06, thêm `stage_b_reserve_eval.py` — viết *sau* freeze;
  bản nháp đã có trên server từ 10/05 12:29 = #15) và `d14e9c3` (10/07, chỉ
  comment + `--strict`). Evaluator v2/v3 không đổi kể từ freeze.
- Constants khớp runbook: cutoff `1791151530514`, `MIN_RESERVE_ROWS 150`,
  `STRICT_RESERVE_ROWS 240`, `CALIBRATION_ROWS 120`, `SEED 42`.
- Server venv: python 3.12.8, numpy 2.5.2, scikit-learn 1.9.0, psycopg2-binary
  2.9.12, joblib 1.5.3 = runtime của freeze; import `ml_evidence_v3`,
  `stage_b_reserve_eval`, `rolling_retrainer` OK, `_feature_columns(5)` = 170.
- DB: script dùng `db_config.get_db_connection()` ← env `DB_PASS` do
  `Import-OpsSecrets` (DPAPI clixml của user server) nạp; chạy được qua SSH;
  không có credential trong lệnh/tài liệu.

### 4.2 ⚠️ Bẫy đã thấy

1. **Runbook §1 "pre-flight count" và `run-stage-b.ps1` đều gọi thẳng script
   chấm.** Khi reserve ≥150 (hoặc ai đó truyền `--min-reserve` nhỏ) lệnh "xem
   count" sẽ **chấm luôn** = tiêu thụ lượt one-shot. Wrapper server
   `E:\BTC\tmp\run-stageb.ps1` không có `--out` → report mặc định
   `E:\BTC\btc-ai\stage_b_report_<YYYYMMDD>.json`, output chỉ ra console SSH.
   **Đếm chỉ bằng probe count-only §3.**
2. Exit code 1 = `STAGE_B_FAILED` **hoặc** crash (traceback). Phân biệt bằng
   file report có tồn tại + trường `status`, không bằng exit code.
3. Report của script không tự ghi git commit, dataset hash, runtime hay hash
   evaluator → phải ghi ngoài (checklist §4.4).

### 4.3 Điểm phải quyết TRƯỚC khi chạy (người duyệt; không tự đổi)

> *Đã quyết 2026-10-23 — xem §8.* Danh sách dưới giữ nguyên làm lịch sử.

1. Ngưỡng: 150 (mặc định) hay `--strict` 240. Protocol gọi 240 là "worth waiting
   for". Phải ghi lựa chọn vào TASKS.md **trước** khi chạy; không đổi sau khi
   thấy kết quả.
2. Protocol Stage B đòi thêm `label_lineage.complete` trên snapshot dùng —
   **script không kiểm**.
3. Protocol khai họ 4 candidate (frozen XGB, retrained XGB, HGB, logistic);
   **script chỉ chấm HGB**. Theo code, Stage B = HGB-only.
4. 5 hàng materialize trễ (§3) vào fit/cal pool, không nằm trong snapshot freeze.
5. `holdout-assessment.md` mô tả reserve từ cutoff v3 cũ 2026-09-20T16:00Z; code
   và runbook dùng cutoff freeze 2026-10-04T22:05:30Z (runbook coi ms constant là
   authoritative). Khoảng 09-20→10-04 đã bị Stage A 05/10 đọc.

### 4.4 Checklist + lệnh (chạy đúng MỘT lần, khi đủ điều kiện)

> *Superseded 2026-10-23:* lệnh dưới dành cho script cũ (không `--score`,
> fit trên live pool). Lệnh hiện hành: `stage-b-runbook.md` §1–§2.

Trước khi chạy:
- [ ] Probe count-only (§3) cho `state = READY_BY_COUNT_ONLY` ở ngưỡng đã chọn; lưu JSON.
- [ ] `git -C E:\BTC\btc-ai rev-parse HEAD` + status; hash 5 file §4.1 vẫn khớp.
- [ ] `frozen_drift_check.py` → `restrictedEqualsFrozen: true`, 0 thiếu/lệch.
- [ ] Venv versions khớp §4.1.
- [ ] Các điểm §4.3 đã có quyết định bằng chữ trong TASKS.md PROMO-1 P4.
- [ ] File report đích **chưa tồn tại**.

Lệnh (PowerShell 5.1 trên server, foreground, < 1 phút; thêm `--strict` chỉ khi đã chọn 240):

```powershell
. E:\BTC\btc-be\ops\common.ps1; Import-OpsSecrets
if (-not $env:DB_PASS) { $env:DB_PASS = $env:PGPASSWORD }
$ts  = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$dir = 'E:\BTC\stage-b'; New-Item -ItemType Directory -Force $dir | Out-Null
$out = Join-Path $dir "stage_b_report_$ts.json"; $log = Join-Path $dir "stage_b_run_$ts.log"
if (Test-Path $out) { throw "report exists: $out" }
Set-Location E:\BTC\btc-ai
cmd /c "venv\Scripts\python.exe stage_b_reserve_eval.py --out `"$out`" > `"$log`" 2>&1"
"STAGE_B_EXIT=$LASTEXITCODE"; Get-FileHash $out, $log -Algorithm SHA256
```

Sau khi chạy:
- [ ] Copy report + log về local (`ai/docs/research/evidence/stage-b/`, giữ sha256);
      ghi verdict, đường dẫn, sha256, git HEAD, versions vào TASKS.md P4.
- [ ] **Không** chạy lại trên cùng reserve với bất kỳ tham số nào khác.

### 4.5 Ba nhánh

- **Chưa đủ mẫu** → chờ, chỉ dùng probe count-only; không chấm.
- **STAGE_B_FAILED / inconclusive** → giữ `quarantined`, giữ nguyên report
  (evidence âm tính là kết quả hợp lệ); không sửa tham số rồi chạy lại.
- **STAGE_B_PASSED** → chỉ *đủ điều kiện xem xét* Stage C. Stage C (train
  recipe lên toàn bộ pool pre-reserve + manifest đủ `_REQUIRED_MANIFEST_FIELDS`
  + registry) **chưa có công cụ** (`stage_c_assemble_*.py` không tồn tại) —
  bước riêng, cần review, chỉ làm nếu PASSED. Không tự `status: active`; flip
  active = bắt đầu forward-paper (runbook §3).

## 5. Thật sự dở / cố ý tắt / đang chờ

- **Đang chờ dữ liệu:** reserve 99/150 (P4).
- **Cố ý tắt:** model 4h (quarantined, fail-closed 503); paper trader abstain.
- **Chưa triển khai:** Stage C assembly; observer fill/outcome cho PaperObservations.
- **Dở thật:** bundle freeze #12 chỉ có trên server; registry server trỏ file đã
  move (chỉ quan trọng ở Stage C); replay SMC đứng 10–11/10.

## 6. Trình tự tiếp quản và điều cấm

1. Mỗi lần kiểm: chạy probe count-only (lệnh trong `probes/run_remote_py.sh`
   + `stage_b_reserve_count_only.py`), cập nhật P4 bằng số có timestamp.
2. Khi đủ ngưỡng đã chọn: làm checklist §4.4, chạy đúng một lần, áp nhánh §4.5.
3. Cấm: chạy `stage_b_reserve_eval.py`/`run-stage-b.ps1` để "xem count"; đổi
   cutoff/gate/min sample/schema/label/recipe; sửa registry khi chưa có report
   PASSED; backfill/rebuild DB; dọn/stash/checkout đè 17 mục dirty trên server.

## 7. Claim UNVERIFIED còn lại

- **Reserve chưa từng bị chấm: UNVERIFIED.** Không tìm thấy `stage_b_report*`
  ở bất kỳ đâu dưới `E:\BTC` hay workspace local, không log nào chứa
  `STAGE_B_PASSED/FAILED/scoredOnce`, các lần chạy có ghi nhận đều ở count < 150
  (9/15/17/53). Nhưng không loại trừ được lần chạy chỉ in console, chạy với
  `--min-reserve` nhỏ, hay file bị xoá.
- Session/người chạy pipeline 05/10 và nội dung bản 05/10 của #14: unknown.
- Quyết định §4.3: chưa có. *(Đính chính 2026-10-23: đã quyết, §8.)*
- `EmbeddingBackfillWorker partial` (timeout mạng): ngoài phạm vi, chưa điều tra.

## 8. Đính chính 2026-10-23 — protocol đã chốt + lineage BLOCKED

**Quyết định (user chốt trước khi chấm, không dựa trên kết quả reserve):**
150 mẫu; chỉ HGB causal v3; fit/cal chỉ từ snapshot freeze 14.674 hàng
(hash-pinned); 5 hàng trễ loại khỏi fit/cal/baseline history, không xóa, không
thành reserve; nhãn `PriceTargets.TargetDirection4h` phải có lineage, thiếu thì
fail-closed. Bản ghi chính thức: `evaluation-protocol.md` §5. Code:
`stage_b_reserve_eval.py` (`--count-only` / `--score`, exit 0/10/20/21/30).
Lệnh: `stage-b-runbook.md` §1–§2. Server **chưa** có script mới (deploy là bước
riêng cần duyệt).

**Lineage snapshot freeze — chứng minh được tới đâu** (count-only thật trên
server, code local chạy trong bộ nhớ, `probes/countonly_preflight_20261023.json`,
chi tiết `probes/lineage_findings_20261023.txt`):
- Bundle verify đầy đủ PASS (local, kể cả recompute thống kê Stage A, 8m44s);
  drift live pre-cutoff = 0 lệch.
- Hợp đồng nhãn khớp code backend: `MlDatasetService.ComputeTargets` →
  `FutureReturn(klines, idx, 1)` + `Direction(r, 0.6)`; hàng chỉ ghi khi có
  `TargetReturn1d`.
- **14.664/14.674 hàng khớp hợp đồng; 10 hàng không:**
  - 1 hàng (signal 2020-02-19 08:00Z): Klines thiếu bar 12:00Z; backend dùng
    bar *tồn tại kế tiếp* (list index +1) ⇒ nhãn thực là move 8h. Đây là lệch
    giữa code và hợp đồng "4h", không phải lỗi dữ liệu đơn lẻ.
  - 9 hàng (signal 2026-09-14 16:00Z → 09-16 00:00Z): `TargetReturn4h` lưu ≠
    return tính từ close Klines hiện tại ở mọi horizon 1–3 bar; 5 hàng đổi nhãn.
    Nguyên nhân **UNVERIFIED** (giả thuyết: Klines được sửa sau khi tính target;
    bảng Klines không có cột updated-at để chứng minh).
- Không có per-row stamp phiên bản code sinh nhãn → lineage chỉ chứng minh được
  bằng tái tính, không bằng version.
- Reserve: chỉ kiểm metadata — Klines 4h sau cutoff 109 bar, 0 gap. Lineage
  reserve chưa kiểm (chỉ chạy trong `--score`).

**Trạng thái Stage B: BLOCKED (evidence), không chỉ chờ mẫu.** Count-only trả
`EVIDENCE_ERROR` exit 30 với reserve **99/150**. Kể cả khi đủ 150, `--score`
sẽ dừng ở lineage frozen. Gỡ BLOCKED là quyết định riêng của user, ví dụ: chấp
nhận nhãn snapshot "as-recorded" cho 10 hàng đã biết (ghi danh sách cố định),
hay loại 10 hàng khỏi fit/cal, hay sửa nguồn và freeze lại — mỗi lựa chọn đổi
input đã chốt nên phải được ghi thành protocol version mới **trước** khi chấm.
Không tự chọn trong lượt này.

**No-peek:** vẫn UNVERIFIED (không đổi so với §7).

### 8.1 Đính chính v2 cùng ngày — đã chọn phương án B (loại 10 hàng)

User chọn B (trong 3 phương án: giữ nguyên nhãn như đã ghi / loại / sửa
nguồn rồi freeze lại), trước khi chấm. Protocol `…/2026-10-23-v2`
(`evaluation-protocol.md` §5.1): đúng 10 hàng ở trên bị loại khỏi
fit/cal/baseline history; input Stage B = 14.664 hàng, sha256 `3e2e73b4…`
(ghim trong code); snapshot, DB, reserve không đổi.

Count-only v2 trên server (code local chạy trong bộ nhớ, không ghi file;
`probes/countonly_preflight_v2_20261023.json`, 2026-10-22T16:22Z):
`NOT_READY`, exit 20, reserve **101/150** (thiếu 49), drift khớp 14.674/14.674,
lineage input Stage B **14.664/14.664 complete**. ⇒ Lineage phía frozen hết
BLOCKED; Stage B giờ chỉ còn **chờ mẫu**, cộng hai điều kiện trước khi chạy:
deploy script mới lên server (bước riêng cần duyệt) và lineage reserve phải
đạt trong `--score` (strict, chưa kiểm).
