# Proposed evaluation protocol — BTCUSDT 4h direction model

> **Status (2026-10-23):** §0–§4 are the original 2026-10-02 *proposal* and
> are kept unchanged as history. The protocol that governs the current
> confirmatory Stage B run is **§5 as revised by §5.1** (`btc-4h-stage-b-confirmatory/2026-10-23-v2`):
> one pre-selected candidate (v3 causal HGB), 150 rows, frozen-snapshot
> fit/calibration, label lineage checked before fit. Where §1–§2 differ
> (four-candidate family, 240-row "worth waiting for", reserve from the
> 2026-09-20 cutoff), §5 wins for this run.

A review-first proposal for re-evaluating the quarantined artifact
(`BTCUSDT_4h_ws5_h4h_XGB_v20260901025507.joblib`) and any successor candidate.
No training or scoring run is implied by this document. It reuses the gate
vocabulary of `rolling_retrainer.py` and the evidence machinery of
`ml_evidence_walkforward.py` / `ml_evidence_v3.py` so results slot into
existing enforcement (`prediction_service._resolve_active_artifact`).

## 0. Decisions required BEFORE any run (pre-declaration)

1. **Label contract.** Declare one label source for the evaluation:
   `PriceTargets.TargetDirection4h` (close-to-close, ±0.6% dead zone,
   `MlDatasetService.cs:481-499`) or `TargetDirectionTb4h` (triple-barrier
   first-touch, same band, `:447-479`). The stored `Label` column is TB today;
   the stored OOS metrics were recorded under CTC semantics; the v2/v3
   evidence is all CTC. **Recommendation: CTC** — it matches the research
   corpus and is what `PredictionController`'s rescore approximates (its
   ±0.15% band is a separate bug to fix, `:186/:308/:326`). Whatever is chosen
   must be written into `data_provenance.label_lineage.source_column` and
   verified to cover 100% of rows (`rolling_retrainer.py:280-321`).
2. **Feature contract.** The served schema is 175 cols (35/bar, includes
   `ActiveRuleCount`); the causal research schema is 170 cols (excludes it,
   `ml_evidence_walkforward.py:76-79`). Declare which one the candidate uses —
   they cannot be compared as the same model. Evaluating the *frozen* artifact
   requires its native 175-col matrix (present in `FeatureVector`).
3. **Freeze.** Persist the evaluation input as a content-addressed snapshot
   (v3 pattern: `*.dataset.npz` + `dataset_sha256`, `evaluationDatasetSha256`),
   record git commit + dirty flag (`ml_evidence_walkforward.py:682-694`) and
   runtime versions. All metrics must be recomputable from the snapshot alone.

## 1. Design

### Windows — expanding walk-forward, purged by label availability

Reuse `v2._folds` semantics (`ml_evidence_walkforward.py:479-510`):

- **fit** — all rows whose `label_available_times_ms <= calibration_start`
  and `>= min_fit_rows` (500);
- **calibration** — next `calibration_rows` (120) rows whose labels were
  available before `test_start`;
- **test** — next `test_rows` (120) rows; step forward `step_rows` (120).

Purge is structural: a row enters fit/calibration only after its label is
realized (`label_available = decision + 4h`), so no boundary-crossing label
is ever fit — equivalent to `PURGE_BARS=5` in `rolling_retrainer.py:69`
which exceeds the 1-bar label horizon and the 5-bar feature window overlap.
On the frozen v3 snapshot this yields 114 folds / 13,680 OOS rows
(2020-06-20 → 2026-09-17); on the reserve extension it appends post-2026-09-20
blocks.

### Candidates (declared family — nothing added after seeing results)

> *Superseded for the Stage B confirmatory run (2026-10-23, §5):* this
> four-candidate family belonged to the Stage A screening proposal. Stage B
> scores only the HGB pre-selected by the v3 screen; it does not score four
> models and keep the best.

- `frozen_quarantined_xgb` — the artifact as-is; honest OOS only for rows
  after its fit data ends (~2026-07-25). Inside the frozen v3 snapshot that
  is the last ~2–3 fold test blocks (2026-07-25 → 2026-09-17, ~320 rows),
  plus the growing forward reserve (58 rows today).
- `retrained_xgb_current_recipe` — `rolling_retrainer.py:509-524` parameters
  (200/6/0.04, FrozenEstimator + isotonic on the calibration partition).
- `hist_gradient_boosting` — the v3 candidate, for continuity with the only
  passing (retrospective) evidence.
- `logistic_scaled` — the v2 control (`ml_evidence_walkforward.py:83`).

### Baselines (identical timestamps, fail-closed coverage)

- `majority_baseline` — constant train-window class prior
  (`rolling_retrainer.py:225-229`) — the gate's own baseline;
- `expanding_historical_class_prior` — Laplace-smoothed calibrated null
  (`ml_evidence_v3.py:30-31, 236-237`);
- `expanding_historical_majority` — soft majority at confidence 0.8;
- `adaptive_class_prior_{180,540}` — regime-adaptive priors (≈30/90d);
- `momentum` / `reversion` — sign of final-bar `ClosePctChange1`
  (persistence for a direction task);
- `persistence` — previous realized label replayed at prior confidence.

### Metrics (per fold + pooled)

- Proper scores: multi-class Brier, log-loss (clipped) —
  `evaluate_probabilities` (`rolling_retrainer.py:170-211`).
- Discrimination: accuracy, balanced accuracy, MCC, per-class F1, macro F1 —
  same function; plus confusion counts (`v2._metrics`).
- Calibration: ECE + per-class reliability bins; propose adding a calibration
  slope (multinomial logit of outcome on model logits) — currently absent
  from both evaluators.
- Per-regime stability: bucket test rows by the existing regime classifier
  (`run_blind_oos_audit.py:103-126`, SMA20/SMA50 + 20-bar return); report
  brier/balanced-acc per regime — a model that is only calibrated in chop is
  the failure mode that quarantined the artifact.
- Coverage: fraction of rows with max prob ≥ serving threshold (0.61,
  `trading_config.py:33-35`), and accuracy-on-covered-subset — separates
  "model is wrong" from "model abstains".

### Uncertainty

Paired circular block bootstrap on row-level Brier/log-loss lift vs each
baseline, block sizes {6,12,24} rows, 1,000 resamples, Bonferroni over
baselines × metrics (`ml_evidence_v3.py:326-359`). A comparison is
`positive` only if the lower bound clears 0 at **all** block sizes.

## 2. Promotion criteria — reuse of existing gate vocabulary

Stage A (retrospective screen, on frozen snapshot): apply
`assess_promotion_gate` (`rolling_retrainer.py:232-265`) with
`PROMOTION_THRESHOLDS` (`:70-79`) to the **two most recent non-overlapping
test windows** that are inside the snapshot but post-date the candidate's
fit, plus the v3 selection-aware checks (`ml_evidence_v3.py:404-414`):
minimum samples, min class support, beats strongest baseline on brier AND
log-loss, all paired intervals positive across block sizes, identical
timestamp coverage. Passing yields status `retrospective_screen_passed` —
same word as `ml_evidence_v3.py:578` — and `promotionAllowed = false`.

*(2026-10-23: the Stage B paragraph below is the original proposal; the
binding version is §5.)*
Stage B (confirmatory, on the no-peek reserve defined in
`holdout-assessment.md`): the same 9-check gate on the reserve window,
scored **once**, plus `label_lineage.complete` on the snapshot used.
Requires ≥150 rows (≈2026-11-05 earliest); the 240-row bound
(`ml_evidence_v3.py:70`) is the stricter variant worth waiting for.

Stage C (artifact assembly): only a Stage-B pass may produce a manifest with
`data_provenance` + `promotion_gate.passed = true` — the two fields
`prediction_service.py:137-152` enforces at serve time — followed by
forward-paper probation before any registry `status: active`.

### Pass/fail interpretation

| Outcome | Meaning | Action |
|---|---|---|
| Stage A pass + Stage B pass | candidate demonstrated on frozen + untouched data | eligible for artifact build & probation — **never** auto-serve |
| Stage A pass + Stage B fail/inconclusive | history looked good, new data doesn't confirm | **stay quarantined**; retain evidence row |
| Stage A fail | retrospective evidence absent | stay quarantined; do not proceed to B |
| Inconclusive anywhere (small n, CI straddles 0) | no signal either way | stay quarantined; "inconclusive" is a valid terminal state (`negativeResultIsValid`, `ml_evidence_v3.py:585`) |
| Any coverage/lineage/hash mismatch | evidence is broken, not negative | fail closed; fix tooling, not the verdict |

## 3. Resource budget (measured on this machine, 2026-10-02)

| Operation | Measured |
|---|---|
| Fetch full dataset (14,380 × 175 floats) | ~1.1 s |
| `joblib.load` quarantined artifact (9.8 MB) | ~4.4 s |
| HGB fit, 500×170 | ~4.4 s |
| HGB fit, 14,000×170 | ~3.1 s |
| XGB retrainer-recipe fit, 3,288×175 | ~38 s |

Derived estimates:

- **Stage A screen** (v3 protocol): 114 folds × 8 fits (1 full + 7 ablations)
  ≈ 912 HGB fits × ~3–4 s ≈ **55–70 min**, + baselines/bootstrap (minutes).
  The frozen artifact needs no refits — scoring it on the ~3 post-2026-07-25
  test blocks (~320 rows) is seconds.
- **Candidate retrainer-style eval** (one XGB fit + calibration + 2 window
  scores): ~1–2 min per fold if folded in — keep Stage A XGB to a few folds
  (e.g., 6 recent folds ≈ 6 × 40 s) rather than all 114.
- **Stage B reserve eval**: trivial compute (<1 min); the binding constraint
  is calendar time (150 rows ≈ Nov 5; 240 rows ≈ Nov 29).
- No per-fold checkpointing exists (`ml_evidence_v3.py:456-462`); an
  interrupted production-size run restarts — schedule Stage A accordingly.

## 4. Non-goals / honest limits

- This protocol evaluates probabilistic direction quality only. Fees,
  slippage, fills, capacity, and PnL remain separate stages (the
  `oos_blind` backtest's +274% claims are a different — and currently
  suspect — evidence class).
- Nothing here un-quarantines `v20260901025507`: its label contract predates
  the Sep-21 rebuild, its calibration metric is contaminated, and its test
  window is spent. Best case for that artifact is a historical footnote; the
  protocol is aimed at the *next* candidate — or at confirming quarantine
  permanently.

## 5. Confirmatory Stage B protocol — `btc-4h-stage-b-confirmatory/2026-10-23`

Declared 2026-10-23, before any reserve scoring (reserve 99/150 at the time;
no reserve label, prediction or metric had been read for this decision).
Enforced in code by `stage_b_reserve_eval.py`; tests in
`tests/unit/test_stage_b_reserve_eval.py`.

| Item | Binding value |
|---|---|
| Minimum reserve | **150** rows, fixed (no CLI override; 240 = reference variant only) |
| Candidate | **v3 causal HGB** (`hist_gradient_boosting`) + `PriorOnlyCalibrator`, `v3._fit_calibrated_hgb`, seed 42 — the only candidate |
| Features | 170 causal columns = 5 bars × 34 (stored 35/bar minus `ActiveRuleCount`) |
| Fit / calibration input | **only** the 2026-10-05 v3 freeze snapshot: bundle `7d662f64…`, dataset sha256 `e876003b…`, 14,674 rows; fit = first 14,554, calibration = last 120, time order kept |
| Late rows | the 5 pre-cutoff rows materialized after the freeze (decisions 2026-10-04 00:00–16:00Z) are excluded from fit, calibration **and** baseline history; not deleted; not reserve |
| Reserve | live rows with decision > 1791151530514 and label available by run time; not replaced by any other row set later |
| Label | `PriceTargets.TargetDirection4h`: `r = (close[t+4h] − close[t]) / close[t] × 100`, `r > 0.6 → 1`, `r < −0.6 → −1`, else 0 (edge ±0.6 → 0); classes −1/0/1 remapped 0/1/2; decision = signal open + 4h, label available = decision + 4h |
| Lineage | re-derived from `Klines` 4h closes (signal bar + the exact next 4h bar) for every frozen and reserve row before any fit; any missing bar or mismatch → `EVIDENCE_ERROR`, no verdict |
| Drift | live pre-cutoff rows must reproduce the snapshot exactly (features + labels) |
| Gate | unchanged: 9 `PROMOTION_THRESHOLDS` checks + paired block bootstrap (6/12/24, 1,000, Bonferroni over 6 baselines × 2 metrics) all positive |
| Modes | `--count-only` never fits/scores/loads reserve labels; `--score` is explicit, one-shot (refuses if a report exists), exclusive-create report |

Passing makes the candidate *eligible* for Stage C review only; nothing is
served and the registry is not touched by the script.

**State at declaration:** the frozen snapshot does **not** pass the lineage
check (1 row whose next 4h Kline is missing — stored label is an 8 h move;
5 rows whose stored label differs from the current Klines closes, cluster
2026-09-14/15). Stage B is therefore BLOCKED on evidence, independent of the
reserve count; resolving it is a separate decision (see
`stage-b-handoff-2026-10-22.md` §8). The rule is not relaxed to fit the data.

### 5.1 Revision v2 — `btc-4h-stage-b-confirmatory/2026-10-23-v2` (binding)

Decided 2026-10-23 by the user (option B of three: keep-as-recorded /
exclude / fix-and-refreeze), still before any reserve scoring (reserve
99/150; no reserve label, prediction or metric read). Everything in §5 holds
except the fit/calibration input:

- **Exactly 10 frozen rows are excluded** from fit, calibration and baseline
  history — the full list found by the lineage check, no more, no fewer
  (decision times, UTC): 2020-02-19 12:00 (next 4h Kline missing → stored label
  is an 8h move) and 2026-09-14 20:00 → 2026-09-16 04:00, nine consecutive bars
  (stored `TargetReturn4h` ≠ current Klines; 5 of them change the label). The
  list is pinned in code (`LABEL_EXCLUDED_DECISION_MS`); a timestamp missing
  from the snapshot is an evidence error, not a silent shrink.
- **Stage B input** = snapshot `e876003b…` minus those 10 = **14,664 rows,
  sha256 `3e2e73b4…`** (pinned). Fit = first 14,544, calibration = last 120
  (the window starts 9 rows earlier than in v1; one excluded row was in fit,
  nine in calibration).
- The snapshot file, the DB and the reserve are unchanged; drift is still
  checked against all 14,674 snapshot rows.
- Lineage on the Stage B input: 14,664/14,664 rows match the contract
  (server, read-only, 2026-10-22T16:22Z). Reserve lineage stays strict and
  runs only under `--score`; a reserve row with a missing next 4h Kline or a
  mismatch stops the run with `EVIDENCE_ERROR` — no exclusion rule exists
  for reserve rows.

Not addressed by v2 (separate work, does not change this run): the backend
labels a row against the next *existing* Kline instead of the exact next 4h
bar, and the root cause of the 2026-09-14/16 return mismatch is unverified.
