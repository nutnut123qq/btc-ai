# Holdout assessment — what is left that is honest

Which slices of the BTCUSDT `4h ws5 h4h` window dataset have already been
consumed by fitting, calibration, selection, drift checks, or evaluation —
and what could still serve as a holdout. All timestamps verified against the
live DB (read-only, 2026-10-02) and the artifacts under `ai/models/` and
`ai/docs/research/evidence/`.

## 1. Dataset extent today

- `WindowClassificationDatasets` (`BTCUSDT`, `4h`, `ws=5`, `h=4h`,
  non-null vector+label): **14,380 rows**, `WindowEndMs`
  2020-02-03T20:00Z → 2026-09-30T08:00Z.
- `Klines` 4h: 14,793 bars, 2020-01-01T00:00Z → **2026-10-01T12:00Z**
  (the dataset lags because each label needs the next bar to close).
- Overall label distribution (TB semantics): down 4,120 / sideways 3,572 /
  up 6,688.
- **Every historical row was rewritten on 2026-09-21** — all 14,321
  pre-existing rows share `CreatedAtUtc` ≈ 2026-09-21 (full
  `MlDatasetRebuildService` rebuild, `backend/Services/MlDatasetRebuildService.cs:219-259`);
  ~59 rows have been appended incrementally since. There is no row-level
  provenance into the pre-rebuild table.

## 2. The artifact's declared windows (registry, exact)

| Split | Window (UTC) | Rows | Fate |
|---|---|---|---|
| train | 2024-10-23T22:14:24 → 2026-04-24T20:19:12 | 3,288 | XGB fit + (in cv=5) calibration |
| val | 2026-04-24T20:19:12 → 2026-07-25T04:00 | 547 | **calibration-contaminated** — `cv=5` was fit on Train+Val then scored on Val (`06f2df7^:rolling_retrainer.py:392-401`); threshold 0.62 also scanned here. Frozen-artifact honest replay on these rows: acc 0.521, not 0.9378 |
| test | 2026-07-25T04:00 → 2026-08-24T04:00 | 181 | drift-check input (`before_metrics`) on every retrainer run + candidate `oos_metrics` + the Sep-1 "audit" |

Notes:

- The old writer used **adjacent masks with zero purge**
  (`06f2df7^:rolling_retrainer.py:313-318`: train ends at `val_start`, val ends
  at `test_start`). With a 1-bar label horizon, the last train row's label
  bar falls inside the val window — small but real leakage the current
  `PURGE_BARS = 5` (`rolling_retrainer.py:69, 438-441`) exists to prevent.
- The test window was queried by drift detection at least twice (Aug-25 and
  Sep-1 runs) plus the unrecorded audit — it is a **selection input, not a
  holdout**. "Test" in the registry is a misnomer.
- The stored metrics were computed under a label semantics that no longer
  matches the table (see `feature-label-contract.md` §3) — the stored "test"
  metrics describe data that does not exist anymore.

## 3. Everything else that has touched the data

| Consumer | Window(s) consumed | When |
|---|---|---|
| `backtest_strategy.py` reports (`backtest_report_*_4h_ws5_h4h_*.json`) | post-2025-01-01 test windows | 2026-07-16 |
| `calibrate_threshold.py` (`calibration_report_…json`) | cal 2024-07-08→2025-07-01, test from 2025-07-08 | 2026-07-17 |
| `meta_labeling.py` (negative result, deferred) | test from 2025-07-08 | 2026-07-17 |
| `walkforward_validate.py` (`walkforward_report_…json`) | 8 folds, last test 2025-08-01→2026-02-01 | 2026-07-17 |
| `run_blind_oos_audit.py` (`oos_blind_performance_report.*`) | all rows ≥ 2025-01-01 | 2026-08-28 |
| `ml_evidence_walkforward.py` v2 (`docs/research/evidence/ml-v2/ab79a19c…`) | 13,680 OOS rows, test blocks 2020-06-20 → **2026-09-17T04:00Z** | 2026-09-21 |
| `ml_evidence_v3.py` (`docs/research/evidence/ml-v3/`, manifest `fb18d819…`) | same rows; decision cutoff **2026-09-20T16:00Z**, dataset sha `158797e6…`, 14,320 rows | 2026-09-21 |

**Every timestamp ≤ 2026-09-20T16:00Z has been read by at least one
evaluation, selection, or drift process.** In particular, v2's candidate
family was *selected* on overlapping OOS history — the v3 report itself
declares `evidenceTier = "retrospective_selection_aware"` and
`promotionAllowed = false` for that reason
(`docs/research/evidence/ml-v3/d4048c86…report.json`; policy text at
`ml_evidence_v3.py:576-583`).

## 4. What is actually left

Only one slice is untouched by any ML evaluator:

- **`WindowEndMs > 2026-09-20T16:00Z` (v3 cutoff) → 2026-09-30T08:00Z:
  exactly 58 rows**, label dist down 13 / sideways 24 / up 21, all created
  2026-09-22→10-01 (post-rebuild, consistent TB lineage).

That is it. 58 rows, growing ~6/day with ingestion.

## 5. Can any window serve as an honest holdout?

- **Pre-cutoff data: no.** Not because it is "dirty" — because it is *spent*.
  Any claim from it inherits the v2 selection bias; it can power a
  retrospective screen (v3 tier) but never a confirmatory one.
- **The 181-row stored test window: no.** Reused for drift + candidate
  scoring + the audit; additionally its stored metrics were computed under
  the pre-rebuild labels.
- **Post-cutoff data (2026-09-20T16:00Z onward): yes, but not yet large
  enough.** 58 rows vs `minimum_samples = 150`
  (`rolling_retrainer.py:71`) — reachable ~2026-11-05 at 6 bars/day; the
  stricter v3 `minimum_gate_samples = 240` (`ml_evidence_v3.py:70`) is
  reachable ~2026-11-29. Feasible only if ingestion keeps running and nobody
  evaluates the window early — a "no-peek" reserve.
- **A subtlety:** the reserve only stays honest if the *evaluation label*
  is declared now (TB vs CTC — see `feature-label-contract.md`). Rescoring it
  under both semantics later and picking the nicer one would repeat the
  original sin.

## 6. Recommendation

A clean confirmatory evaluation is **not feasible today**; it becomes
feasible ~Nov 2026 when the reserve reaches gate size. Until then the only
honest options are:

1. **Frozen-snapshot replication** — the v3 `*.dataset.npz`
   (`docs/research/evidence/ml-v3/d5dd2d18…npz`, 14,320 rows) is
   byte-stable and hash-bound; rerunning against it measures the *current*
   code deterministically, but inherits retrospective-selection status.
2. **Declared forward reserve** — freeze evaluation config now (label =
   `TargetDirection4h` recommended for comparability with the v2/v3
   evidence; `TargetDirectionTb4h` recommended only if the production
   contract is deliberately re-pinned), accumulate to ≥150 rows, evaluate
   once.
3. **Forward-paper observations** — worthless for the current model (every
   poll abstains `model-unavailable`, `paper_trader.py:975-977`); only useful
   once a promoted artifact exists.
