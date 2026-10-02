# Quarantine provenance — BTCUSDT_4h_ws5_h4h v20260901025507

Survey of what actually wrote the quarantine, what the gate is, and what a
reproducible audit artifact would have had to contain. All citations verified
against `ai/` HEAD `25501cd` and backend HEAD.

## 1. What the registry entry actually is

`models/model_registry.json` (current working tree, lines 12–13):

```json
"status": "quarantined",
"quarantine_reason": "Independent OOS audit failed majority-baseline and class-discrimination promotion gates.",
```

Attached metrics: `validation_accuracy 0.9378`, `oos_brier_score 0.3906`,
`oos_log_loss 0.7063`, `oos_accuracy 0.7403`, `optimal_threshold 0.62`,
windows train `2024-10-23T22:14:24Z → 2026-04-24T20:19:12Z` (3288 rows),
val `→ 2026-07-25T04:00Z` (547), test `→ 2026-08-24T04:00Z` (181).

### Timeline reconstructed from git + filesystem

| When (UTC) | Event | Evidence |
|---|---|---|
| 2026-08-25 04:48 | `v20260825` trained, registered active | history[] entry in registry; `models/BTCUSDT_4h_ws5_h4h_XGB_v20260825.json` |
| 2026-09-01 02:55 | `v20260901025507` written by the **pre-gate** `rolling_retrainer.py` writer; registry updated `status: active`, v20260825 archived | `created_at_utc` 02:55:08; registry `updated_at_utc` before the commit = `2026-09-01T02:55:08.115990` (visible at `06f2df7^:models/model_registry.json`); manifest field `"calibration": "isotonic (cv=5)"` — a string only the old writer emitted |
| ~2026-09-01 05:05 | registry entry flipped to `status: quarantined` + `quarantine_reason` added; `updated_at_utc = 05:05:07` | `git show 06f2df7 -- models/model_registry.json` |
| 2026-09-01 07:35 | commit `06f2df7` "fix: enforce independent model promotion gates" packages the quarantined registry **plus** the hardened code | commit stat |

### The writer did not write that reason string

The only committed code that ever writes `status: "quarantined"` /
`quarantine_reason` is `rolling_retrainer.py:563-566` (identical at `06f2df7`
and at HEAD):

```python
existing["status"] = "quarantined"
existing["quarantine_reason"] = "Artifact lacks a passing independent-window promotion gate."
```

`git log --all -S "Independent OOS audit failed"` returns **one** hit — the
registry JSON hunk inside `06f2df7`. The string exists in no `.py` at any
commit. Conclusion: the quarantine entry was a **data edit inside the commit**
(hand-edit or uncommitted one-off tooling), not the output of the committed
writer path. It *is* git-auditable (author nutnut123qq, 2026-09-01 14:35 +07,
commit `06f2df7`), but it is **not** a machine-generated record: no per-check
verdicts, no baseline metrics, no dataset hash were persisted with it.

The same commit's `docs/local-production.md` diff is the closest thing to a
written rationale: *"an independent temporal audit showed that it does not
beat the recent majority baseline and collapses on the directional classes."*

### Why the writer's path could not have produced this entry anyway

`retrain_symbol_rolling()` only quarantines (`rolling_retrainer.py:559-566`)
when a **freshly trained candidate** fails the gate — and on that path the
candidate artifact is never saved. `v20260901025507.joblib` exists on disk,
because it was trained by the **old** writer that had no gate at all
(`06f2df7^:rolling_retrainer.py:392-397` — `CalibratedClassifierCV(cv=5)` on
stacked Train+Val). The gate machinery and the quarantine it would have
triggered did not exist when the artifact was produced.

## 2. The gate it plausibly failed — `two-independent-windows-v1`

`assess_promotion_gate()` (`rolling_retrainer.py:232-265`) runs **9 checks
independently on each of two untouched windows** (calibration-era "validation"
window and last-30d "oos" window), with thresholds from
`PROMOTION_THRESHOLDS` (`:70-79`):

1. `sample count` ≥ 150
2. `class support` — min class count ≥ 15
3. `macro F1` ≥ 0.40
4. `balanced accuracy` ≥ 0.40
5. `MCC` ≥ 0.10
6. `per-class F1` — min over down/sideways/up ≥ 0.10
7. `ECE` ≤ 0.20
8. `Brier improvement` — model ≤ majority-baseline − 0.01
9. `log-loss improvement` — model ≤ majority-baseline − 0.01

plus a dataset-level check appended at `:533-535`: `label_lineage.complete`
must be true (exactly one `PriceTargets` label column reproduces every stored
label). Baselines are `majority_baseline_metrics()` (`:225-229`): the
constant train-prior probability vector, never fit on eval labels.

### Reproduced verdict on today's data (survey measurement, 2026-10)

I re-scored the frozen artifact (`joblib.load`, 4.4 s, sha256 verified equal
to `manifest.artifact_sha256 = 1f1542d7…`) on the exact registry windows
reconstructed from today's `WindowClassificationDatasets` and ran
`assess_promotion_gate` — the gate **still fails**, on:

- `oos: macro F1` (0.26 vs 0.40)
- `oos: balanced accuracy` (0.34 vs 0.40)
- `oos: MCC` (0.08 vs 0.10)
- `oos: per-class F1` (up = 0.00; model predicts only down/sideways — `prediction_counts [8, 173, 0]`)

The loss-improvement checks **pass today** (model brier 0.593 < baseline
0.676 − 0.01; log-loss 1.017 < 1.110 − 0.01) under the current triple-barrier
labels, and also pass under close-to-close labels (0.408 < 0.696;
0.749 < 1.145). So the registry's "majority-baseline" clause is **not
reproducible on current data** — either the September snapshot's baseline
differed, or the phrase loosely covered the loss-improvement/discrimination
family. The "class-discrimination" clause reproduces exactly.

Note also `val_acc 0.9378` is calibration-contaminated as suspected: the old
writer fit `cv=5` isotonic on Train+Val and scored on Val. Replaying the
frozen artifact on those same rows gives acc **0.521** — the stored number was
in-sample for 4/5 of the calibration ensemble.

## 3. Byte-identical predecessor

`v20260825.joblib` and `v20260901025507.joblib` are byte-identical
(sha256 `1f1542d7…` for both; manifest `artifact_sha256` matches). The Sep-1
"retrain" was a deterministic re-run on unchanged data (`random_state=42`,
same windows). `before_metrics` ≡ `oos_metrics` in the manifest for the same
reason — not a copy-paste bug.

## 4. What a reproducible quarantine artifact must contain

The gap in `06f2df7` is that the verdict exists but the evidence doesn't.
Proposal — a `promotion-audit` JSON stored next to the artifact and hashed
into `manifest.data_provenance` on future runs:

```json
{
  "schema": "btc-promotion-audit/v1",
  "policy": "two-independent-windows-v1",
  "created_at_utc": "…",
  "code": {"git_commit": "…", "git_dirty": false, "evaluator": "rolling_retrainer@<sha256 of file>"},
  "dataset": {
    "source_table": "WindowClassificationDatasets",
    "identity": "BTCUSDT_4h_ws5_h4h",
    "row_count": 14380,
    "first_window_end_ms": 1580760000000,
    "last_window_end_ms": 1790755200000,
    "dataset_sha256": "<sha256 over times+X+y, same digest as dataset_provenance()>",
    "label_lineage": {"complete": true, "source_column": "PriceTargets.TargetDirectionTb4h",
                      "close_to_close_matches": 0, "triple_barrier_matches": 14380}
  },
  "windows": {
    "train":      {"start_ms": …, "end_ms": …, "rows": 3288},
    "calibration": {"start_ms": …, "end_ms": …, "rows": …},
    "gate":       {"start_ms": …, "end_ms": …, "rows": …},
    "oos":        {"start_ms": …, "end_ms": …, "rows": 181},
    "purge_bars": 5
  },
  "artifact": {"file": "…joblib", "artifact_sha256": "…",
               "feature_schema_hash": "…", "library_versions": {"joblib": "…", "scikit-learn": "…", "xgboost": "…"}},
  "metrics": {
    "gate": {"samples": …, "class_counts": […], "brier_score": …, "log_loss": …,
             "accuracy": …, "balanced_accuracy": …, "mcc": …, "ece": …,
             "f1_per_class": {"down": …, "sideways": …, "up": …}, "f1_macro": …, "f1_weighted": …},
    "oos":  { "…": "same shape" },
    "gate_baseline": {"brier_score": …, "log_loss": …},
    "oos_baseline":  {"brier_score": …, "log_loss": …}
  },
  "gate_verdict": {"passed": false, "thresholds": {"…": "copy of PROMOTION_THRESHOLDS"},
                   "failures": ["oos: macro F1", "oos: balanced accuracy", "oos: MCC", "oos: per-class F1"]},
  "registry_mutation": {"key": "BTCUSDT_4h_ws5_h4h", "status_before": "active",
                        "status_after": "quarantined", "quarantine_reason": "…"},
  "audit_sha256": "<sha256 of canonical JSON of everything above>"
}
```

Requirements the schema enforces:

- **Dataset hash + label lineage** — without it, a post-hoc rebuild (like the
  2026-09-21 `MlDatasetRebuildService` run) silently changes what the metrics
  mean; this is exactly what makes the Sep-1 metrics non-reproducible today.
- **Row-level predictions or at least per-window probability dumps** — the
  v3 bundle (`docs/research/evidence/ml-v3/`, manifest
  `fb18d819…`) is the in-repo template: `*.dataset.npz` +
  `*.predictions.jsonl` + `*.report.json` + `*.manifest.json`, all
  content-addressed. Adopt it; don't invent a parallel format.
- **Failures enumerated, not summarized** — the registry reason string should
  be derivable from `gate_verdict.failures`, not authored prose.
