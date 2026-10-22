# Stage B → Promotion Runbook

Operational runbook for executing the no-peek reserve gate and applying its
verdict. Companion to `evaluation-protocol.md` (criteria) and
`holdout-assessment.md` (reserve definition). Single-operator, single-machine.

> **Revision 2026-10-23 (protocol `btc-4h-stage-b-confirmatory/2026-10-23`).**
> §0–§3 rewritten: count-only is now a separate, non-scoring mode; fit /
> calibration come only from the hash-pinned 2026-10-05 freeze snapshot;
> label lineage is checked before any fit; `--min-reserve` / `--strict` were
> removed; exit codes changed. The pre-revision commands (bare script run,
> `run-stage-b.ps1`, `E:\BTC\tmp\run-stageb.ps1`, exit 2/1/0) are obsolete —
> on the old script they could score while "checking the count". Decision
> record: `evaluation-protocol.md` §5, `stage-b-handoff-2026-10-22.md` §8.
>
> **Revision v2 (same day, `…/2026-10-23-v2`):** fit/calibration input =
> snapshot minus 10 pinned label-contract rows (14,664 rows, sha256
> `3e2e73b4…`); `evaluation-protocol.md` §5.1.

## 0. Fixed constants (do not re-derive — pinned in `stage_b_reserve_eval.py`)

- Frozen cutoff: `FROZEN_CUTOFF_MS = 1791151530514` = **2026-10-04T22:05:30Z**
  (2026-10-05 05:05 +07) — captured wall-time of the eval-config freeze.
  (An older annotation read `2026-10-05T04:45:30Z`; that was a mislabelled
  TZ conversion, the ms constant is authoritative.)
- Frozen input: v3 bundle `7d662f64….manifest.json`, dataset sha256
  `e876003b…`, 14,674 rows (`evidence/v3_20261005_050530/` on the server).
  Stage B uses it minus 10 pinned label-contract rows: 14,664 rows, sha256
  `3e2e73b4…` (fit 14,544 + calibration 120).
- Reserve = live rows with `decision_time > cutoff` (decision = WindowEndMs + 4h)
  and label available (decision + 4h) by run time.
- Gate: `MIN_RESERVE_ROWS = 150` (fixed). The 240-row bound is a reference
  variant only, not this confirmatory run.
- Candidate: v3 causal HGB + PriorOnlyCalibrator, 170 columns, seed 42.
- Structural lag: a reserve row materializes ~24–36 h after its signal bar
  (PriceTargets null-gates on `TargetReturn1d` = +6 bars for 4h, plus the
  6-hourly `WindowDatasetBuilder` cadence). Zero reserve rows on day 1 is
  expected, not a stall.

## 1. Count-only pre-flight (any time; never fits or scores)

```powershell
# On the server, after the revised script is deployed (deploy = separately approved step):
. E:\BTC\btc-be\ops\common.ps1; Import-OpsSecrets
if (-not $env:DB_PASS) { $env:DB_PASS = $env:PGPASSWORD }
Set-Location E:\BTC\btc-ai
venv\Scripts\python.exe stage_b_reserve_eval.py --count-only `
  --bundle E:\BTC\btc-ai\evidence\v3_20261005_050530\7d662f64c72a45634dc8f3eaa3b5f8a9793b724ec1abf113d3b8cc726c99af19.manifest.json
```

Counts the reserve (metadata only), verifies the bundle hashes (without
recomputing Stage A statistics), checks live pre-cutoff rows against the
snapshot (drift) and re-derives the frozen labels from Klines (lineage). It
never loads a reserve label. Before deploy, the same code can run from local
sources in memory: `evidence/btc-research-handoff-20261022/probes/countonly_runner.py`
via `run_remote_py.sh` (writes nothing on the server).

## 2. Execute Stage B (once — explicit `--score`, never implied)

Preconditions: count-only returned `READY_BY_COUNT_ONLY` (exit 21) — **not**
`EVIDENCE_ERROR`; the deployed script hash equals repo HEAD; no
`stage_b_report*.json` exists in the output folder.

```powershell
. E:\BTC\btc-be\ops\common.ps1; Import-OpsSecrets
if (-not $env:DB_PASS) { $env:DB_PASS = $env:PGPASSWORD }
$ts  = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$dir = 'E:\BTC\stage-b'; $out = Join-Path $dir "stage_b_report_$ts.json"; $log = Join-Path $dir "stage_b_run_$ts.log"
New-Item -ItemType Directory -Force $dir | Out-Null
Set-Location E:\BTC\btc-ai
cmd /c "venv\Scripts\python.exe stage_b_reserve_eval.py --score --bundle E:\BTC\btc-ai\evidence\v3_20261005_050530\7d662f64c72a45634dc8f3eaa3b5f8a9793b724ec1abf113d3b8cc726c99af19.manifest.json --out `"$out`" > `"$log`" 2>&1"
"STAGE_B_EXIT=$LASTEXITCODE"; Get-FileHash $out, $log -Algorithm SHA256
```

`--score` re-verifies the bundle in full (Stage A statistics recomputed,
~9 min), re-checks drift + lineage for frozen **and** reserve rows, then fits
once. It refuses to run when any `stage_b_report*.json` already sits in the
output folder and opens `--out` with exclusive create. The report records
protocol version, frozen bundle/dataset hashes, late-row exclusion, reserve
identity (count, first/last decision, dataset sha), lineage, git/file hashes
and runtime.

Record the verdict + report path + sha256 in `TASKS.md` under `PROMO-1` P4.

## 3. Verdict branches

| Exit | Status | Meaning / action |
|---|---|---|
| 20 | `NOT_READY` / `RESERVE_NOT_READY` | < 150 rows. Wait; nothing fitted. |
| 21 | `READY_BY_COUNT_ONLY` | count-only only: eligible *by count*. Not a verdict. |
| 30 | `EVIDENCE_ERROR` | hash / drift / lineage / runtime / schema failure, or a prior report exists. No verdict, no report. Fix the evidence (separately approved), never the gate. |
| 1 / 2 | crash / CLI usage | No verdict. Investigate; a crash after fitting still counts as a consumed run if a report exists. |
| 10 | `STAGE_B_FAILED` | see below |
| 0 | `STAGE_B_PASSED` | see below |

### `STAGE_B_FAILED` (exit 10)
- Keep `status: quarantined` in `models/model_registry.json`.
- Copy the report into evidence: `ai/docs/research/evidence/` (immutable
  bundle convention, sha-named) or record its path + sha256 in TASKS.md.
- "Inconclusive" is a valid terminal state (protocol §2) — do not retry with
  tweaked parameters; that would be a second peek at the same reserve.

### `STAGE_B_PASSED` (exit 0) → Stage C artifact assembly (not implemented)
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
