# Stage B → Promotion Runbook

Operational runbook for executing the no-peek reserve gate and applying its
verdict. Companion to `evaluation-protocol.md` (criteria) and
`holdout-assessment.md` (reserve definition). Single-operator, single-machine.

## 0. Fixed constants (do not re-derive)

- Frozen cutoff: `FROZEN_CUTOFF_MS = 1791151530514` = **2026-10-04T22:05:30Z**
  (2026-10-05 05:05 +07) — captured wall-time of the eval-config freeze.
  (An older annotation read `2026-10-05T04:45:30Z`; that was a mislabelled
  TZ conversion, the ms constant is authoritative.)
- Reserve = rows with `decision_time > cutoff` (decision = WindowEndMs + 4h).
- Gate: `MIN_RESERVE_ROWS = 150`; stricter `STRICT_RESERVE_ROWS = 240`.
- Script: `ai/stage_b_reserve_eval.py`; wrapper: `D:\code\btc\run-stage-b.ps1`
  (runs on remote `E:\BTC\btc-ai\` via ssh — the remote DB is canonical).
- Structural lag: a reserve row materializes ~24–30 h after its decision
  (PriceTargets null-gates on `TargetReturn1d` = +6 bars for 4h, plus the
  6-hourly `WindowDatasetBuilder` cadence). Zero reserve rows on day 1 is
  expected, not a stall.

## 1. Pre-flight (any time; safe, no peeking)

```powershell
# Reserve count without scoring — exits RESERVE_NOT_READY, prints count only.
ssh "my pc"@100.83.92.101 "cd /d E:\BTC\btc-ai && venv\Scripts\python.exe stage_b_reserve_eval.py"
# or locally (local DB may lag/stale — remote count is authoritative):
cd ai; ./venv/Scripts/python.exe stage_b_reserve_eval.py
```

Before the real run: confirm remote `stage_b_reserve_eval.py` matches repo
HEAD (`fc` / hash-compare `E:\BTC\btc-ai\stage_b_reserve_eval.py` vs local).

## 2. Execute Stage B (once — the reserve is scored exactly once)

```powershell
D:\code\btc\run-stage-b.ps1        # ssh wrapper → remote run
```

Or directly on remote:
```powershell
ssh "my pc"@100.83.92.101 "cd /d E:\BTC\btc-ai && venv\Scripts\python.exe stage_b_reserve_eval.py --out stage_b_report_YYYYMMDD.json"
# stricter bound: add --strict (requires 240 rows)
```

Record the verdict + report path in `TASKS.md` under `PROMO-1` P4.

## 3. Verdict branches

### `RESERVE_NOT_READY` (exit 2)
Do nothing. Re-check next session. Do NOT re-run to "peek" — the gate is
one-shot by design; intermediate runs only ever print the count.

### `STAGE_B_FAILED` (exit 1)
- Keep `status: quarantined` in `models/model_registry.json`.
- Copy the report into evidence: `ai/docs/research/evidence/` (immutable
  bundle convention, sha-named) or record its path + sha256 in TASKS.md.
- "Inconclusive" is a valid terminal state (protocol §2) — do not retry with
  tweaked parameters; that would be a second peek at the same reserve.

### `STAGE_B_PASSED` (exit 0) → Stage C artifact assembly
Passing only makes the candidate *eligible* — nothing serves yet. To produce
a servable artifact:

1. **Train + persist**: fit the exact Stage-B recipe
   (`v2._build_estimator(FULL_MODEL)` + `PriorOnlyCalibrator`,
   `ml_evidence_v3._fit_calibrated_hgb` internals, causal columns
   `_feature_columns(5)`) on the full pre-reserve pool, then `joblib.dump`
   to `models/BTCUSDT_4h_ws5_h4h_<family>_v<date>.joblib`. NOTE: no existing
   tool writes a gate-compliant manifest — a small `stage_c_assemble_*.py`
   (or careful manual assembly) is required; treat it as its own reviewed
   step.
2. **Manifest** `<artifact>.json` must satisfy `prediction_service.py`
   `_REQUIRED_MANIFEST_FIELDS`:
   - `artifact_sha256` (sha256 of the .joblib), `feature_schema_version`,
     `feature_schema_hash` (sha256 of compact-JSON `feature_names`),
     `feature_names` (length == `feature_dim`), `feature_dim`,
     `class_mapping` = `{"0":-1,"1":0,"2":1}`,
     `library_versions` {joblib, scikit-learn, xgboost} = **serving env
     versions exactly** (loader compares with `importlib.metadata`),
     `data_provenance` {`identity` = `BTCUSDT_4h_ws5_h4h`, `row_count`,
     `dataset_sha256` (64-hex), `label_lineage.complete` = true +
     `source_column`}, `promotion_gate.passed` = true (+ gate detail),
     `version` matching the registry entry.
3. **Registry**: update `models/model_registry.json` — entry
   `BTCUSDT_4h_ws5_h4h`: `active_model_file`, `version`,
   `status: "active"`, keep prior entry in `history` (archived). There is no
   `probation` status in code — `paper_trader.py` loads via `load_model`,
   so the flip IS what starts forward-paper decisions.
4. **Verify serve path fail-safe first**: before flipping, confirm a bad
   manifest still 503s (`/api/prediction/latest` →
   `MODEL_ARTIFACT_INCOMPATIBLE`) on a copy/staging — fail-closed must not
   regress.
5. **Forward-paper probation**: after `status:active`, `BTCPaperTrader`
   polls every 15 min and starts writing real (non-abstain)
   `PaperObservations`. Probation = observe ≥ N forward bars (agree N with
   the operator before flipping; suggestion: 2 weeks ≈ 84 bars) — decisions
   must produce sane non-abstain rates and calibrated confidence. If
   decisions look degenerate → re-quarantine (set `status:"quarantined"`,
   keep artifact + note reason in `quarantine_reason`).
6. **Deploy**: ai service runs on remote (`E:\BTC\btc-ai` + venv). Artifact +
   manifest + registry must be present on the remote before flipping —
   verify sha256 of `.joblib` local-vs-remote identical.

## 4. Never-do list (protocol invariants)

- Never edit registry `status` without a Stage-B-passed report on file.
- Never re-run Stage B on the same reserve with adjusted parameters.
- Never backfill reserve rows' labels into selection/calibration.
- Never serve an artifact whose manifest fails any `_resolve_active_artifact`
  check — the 503 is the intended state when evidence is incomplete.
