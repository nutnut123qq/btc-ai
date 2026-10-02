# Feature & label contract — BTCUSDT_4h_ws5_h4h

What the artifact expects, what the dataset actually stores, and where the
training/serving semantics diverge. Verified against `ai/` HEAD `25501cd`,
backend HEAD, and the live database (read-only queries, 2026-10-02).

## 1. Feature contract

### Order and width

`feature_dim = 175 = window_size 5 × 35 per-bar features`. Layout is bar-major:
`ws5_bar{i}_{name}` for `i = 0..4` (bar0 = oldest, bar4 = signal bar), with the
35 names in this exact order (artifact manifest `feature_names`,
`models/BTCUSDT_4h_ws5_h4h_XGB_v20260901025507.json:11-187`):

```
CloseZscore, ClosePctChange1, ClosePctChange4, ClosePctChange24,
HighLowRangePct, BodyPct, UpperWickPct, LowerWickPct,
Rsi14, Rsi14Slope, MacdNorm, MacdSignalNorm, MacdHistogramNorm,
Ema12Dist, Ema26Dist, Ema50Dist, Ema200Dist, Sma50Dist, Sma200Dist,
BollingerWidth, BollingerPosition, Atr14Pct, ObvEmaDist, VwapDist,
RollingVwapDist, VolumeZscore, VolumeSma20Ratio, TakerBuyRatio,
RecentPatternEncoded, ActiveRuleCount,
HourSin, HourCos, DayOfWeekSin, DayOfWeekCos, IsWeekend
```

The same list exists three times and must stay in lockstep:

- Backend builder: `WindowDatasetService.FeatureNames`
  (`backend/Services/WindowDatasetService.cs:20-57`), emitted by
  `BuildFeatureVector` (`:254-341`) — 28 stored `MlFeatureStores` indicator
  columns, then `RecentPatternEncoded` (offset 28) + `ActiveRuleCount`
  (offset 29), then 5 time features (offsets 30-34) computed from
  `OpenTimeMs` UTC (`:327-337`; `DayOfWeek` = `System.DayOfWeek`, Sunday=0).
- Trainer-side schema: `FEATURE_NAMES_35` (`ai/train_baseline_advanced.py:61-71`),
  expanded by `infer_feature_names()` (`:94-101`); the schema name is
  `FEATURE_SCHEMA_VERSION = "window-dataset-35-v1"` (`rolling_retrainer.py:68`).
- Serving feature builder: `paper_trader.py:86-95` (`FEATURE_COLS` +
  `TIME_FEATURE_COLS`); `build_vector_at` re-orders stored columns to match the
  manifest's `feature_names` (`paper_trader.py:224-230`), and `time_features`
  (`:172-182`) replicates .NET's Sunday=0 convention.

### Hash enforcement

`feature_schema_hash` = sha256 over `json.dumps(feature_names,
separators=(",",":"))` (`prediction_service.py:57-59`,
`rolling_retrainer.py:355-357`). Verified: manifest value
`781c4e2a…ad80f` recomputes exactly. Loader enforces it at
`prediction_service.py:116-124`, plus `len(feature_names) == feature_dim` and
`model.n_features_in_ == feature_dim` (`:172-176`).

**Note:** `ActiveRuleCount` (offset 29) is present in the serving contract but
flagged non-causal by the research evaluator — it "reflects rules enabled at
rebuild time and has no point-in-time rule-version lineage"
(`ml_evidence_walkforward.py:76-79`, `docs/research/ML_EVIDENCE_WALKFORWARD.md:17-19`).
The v2/v3 evaluators drop it (170 cols); the served artifact keeps it. Any
candidate artifact trained under the new protocol must decide this explicitly —
the two feature contracts are not interchangeable.

## 2. Label contract

Stored label column: `WindowClassificationDatasets.Label ∈ {-1, 0, 1}`
(-1 down, 0 sideways, 1 up), remapped to {0,1,2} for XGBoost by
`LABEL_REMAP` (`rolling_retrainer.py:65-66`); manifest `class_mapping`
`{"0": -1, "1": 0, "2": 1}` restores semantics at serving
(`prediction_service.py:112-114`, `:215-221`).

The label is **copied at dataset-build time** from `PriceTargets` by
`ExtractLabelAndReturn` (`WindowDatasetService.cs:384-406`), which selects on
`IndexingOptions.WindowDatasetLabelType` — default `"TripleBarrier"`
(`backend/Options/IndexingOptions.cs:73`; **not** overridden in
`appsettings.json`). For horizon `4h` on 4h bars (`BarsForHorizon` = 1,
`MlDatasetService.cs:400-426`):

- **Triple-barrier (current default):** `TargetDirectionTb4h`
  (`MlDatasetService.cs:447-479`). Next bar wins: `1` if its High ≥
  close×1.006, `-1` if its Low ≤ close×0.994 (high checked first — a bar that
  sweeps both barriers resolves to **up**), else `0` with the close-to-close
  return. Dead-zone **±0.6%** via `DirectionThreshold("4h")` (`:491-499`).
- **Close-to-close (alternative):** `TargetDirection4h` (`:481-487`):
  `1` if 1-bar close-to-close return > +0.6%, `-1` if < −0.6%, else `0`.

Today the stored `Label` column is **100% triple-barrier** for the whole
`4h ws5 h4h` dataset — verified globally: all 14,380 rows equal
`TargetDirectionTb4h`, while only 8,839 (61.5%) also equal
`TargetDirection4h`. `dataset_provenance()` (`rolling_retrainer.py:280-321`)
would now resolve `label_lineage.source_column =
PriceTargets.TargetDirectionTb4h` unambiguously.

## 3. The threshold/semantics mismatch

**Serving rescore uses a different label than training — twice over.**

`PredictionController` audit + accuracy paths compute the "actual" label as
close-to-close with a **±0.15%** dead zone:

- `Controllers/PredictionController.cs:186` — history `actualLabel`
- `Controllers/PredictionController.cs:308` — accuracy `actualLabel`
- `Controllers/PredictionController.cs:326` — canonical win-rate `actual`
- `:245-261` — `TargetReturn` itself = `(targetKline.Close − entryKline.Close)/entryKline.Close`, pure close-to-close.

vs training label: **triple-barrier first-touch at ±0.6%**
(`MlDatasetService.cs:452-478`, threshold `:494`; selected by
`IndexingOptions.cs:73`).

So serving-side "accuracy"/`winRatePct` scores the model on a *different task*:
a model can be "correct" under TB (price touched +0.6% intrabar) while the
rescore marks it wrong (closed +0.4%). Independent of the ±0.6% vs ±0.15%
band, the TB-vs-CTC semantics alone guarantee disagreement: on the stored test
window today only **78.5%** of rows have identical labels under the two
definitions.

### Does the mismatch bias the *stored* OOS metrics?

The registry `oos_*` numbers were produced by `evaluate_model_performance`
inside the retrainer — scored against the dataset `Label` column as it existed
on 2026-09-01. Two findings matter:

1. **The stored OOS metrics replay under close-to-close labels, not under
   today's stored TB labels.** Re-scoring the frozen artifact on the stored
   test window (2026-07-25T04:00 → 2026-08-24T04:00, 181 rows):
   - vs `TargetDirection4h` (CTC): acc **0.7403** (exact match to registry),
     f1_macro 0.2872 (stored 0.2863), brier 0.4076 (stored 0.3906),
     ll 0.7488 (stored 0.7063);
   - vs today's `Label` (TB): acc 0.558, f1_macro 0.2618, brier 0.5932, ll 1.017.
   The 4-decimal accuracy match means the Sep-1 evaluation labels were
   close-to-close for that window; the residual brier/ll gap is consistent
   with a few rows' labels/features having changed in the rebuild.
2. **The entire dataset was rebuilt 2026-09-21** (`CreatedAtUtc` for all
   14,321 historical rows = 2026-09-21; `MlDatasetRebuildService` deletes +
   reinserts, `Services/MlDatasetRebuildService.cs:219-259`), after commit
   `e6cefcd` "Build leakage-safe BTC research platform". So whatever label
   semantics the Sep-1 metrics describe, the current table carries different
   labels for ~21.5% of the test window.

Consequence: the registry metrics are **not reproducible** and **not scored
against the label contract the serving stack now stores**. They are best read
as "evaluated against `TargetDirection4h`-style CTC labels on a pre-rebuild
dataset". For any re-evaluation, the label source must be pinned first —
`data_provenance.label_lineage` (`prediction_service.py:137-149` enforces it
at serve time) exists precisely because this ambiguity was hit in production.

### Related thresholds (for completeness, not bugs)

- Serving decision threshold for the paper trader is
  `ASSET_4H_THRESHOLDS["BTCUSDT"] = 0.61` (`trading_config.py:33-35`), read at
  `paper_trader.py:992` — the manifest's `optimal_threshold: 0.62` (scanned on
  the calibration-contaminated val window, `06f2df7^:rolling_retrainer.py:404-417`)
  is stored but **not** consulted by the forward paper path.
- `TIMEFRAME_THRESHOLDS` (`trading_config.py:27-30`) holds the older 4h=0.61 /
  1h=0.58 pair used by `run_blind_oos_audit.py`.
