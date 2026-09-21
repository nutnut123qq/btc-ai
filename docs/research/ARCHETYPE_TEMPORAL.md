# BTC archetype temporal contract

`archetype_temporal.py` is the predictive research path for candle-window
archetypes. `cluster_archetypes.py` remains a descriptive gallery rebuild: it
refits all loaded history and its outcome summaries are in-sample.

## Frozen version

A version fits only rows satisfying both conditions:

- the feature window is available at or before `train_through_ms`;
- the label outcome is realized at or before `train_through_ms`.

The second condition purges boundary-crossing labels. The artifact stores the
training-only scaler, canonicalized KMeans centroids, per-cluster member counts,
distance gates and Laplace-smoothed class probabilities. Its `version_id` hashes
the scope, cutoff, training rows and fitted state. An archetype ID therefore has
the form `arch-<version hash>:A-0000` and only has meaning within that immutable
version. Reusing a path with different bytes is rejected.

Future windows use a manual transform plus nearest frozen centroid. No `fit`,
`fit_transform` or centroid update exists in the assignment path. Windows beyond
the training distance quantile, or assigned to an undersized cluster, abstain.

## Evaluation

The holdout begins strictly after the training cutoff and an optional row embargo.
The candidate and baseline are scored on the same accepted timestamps. The
baseline is the unconditional Laplace-smoothed class distribution learned from
training rows. Primary evidence is multiclass Brier-loss lift with a paired
time-block bootstrap interval; accuracy is diagnostic. Reports include sample
counts, coverage, abstention and the accepted timestamps.

`positive`, `negative`, `inconclusive` and `insufficient_evidence` are all valid
outcomes. The evaluator never promotes a candidate merely because the pipeline
ran.

## Reproducible command

By default the command reads the frozen BTCUSDT 4h benchmark from PostgreSQL in
a read-only transaction and removes the non-causal stored `ActiveRuleCount`
feature through the shared benchmark loader:

```powershell
.\venv\Scripts\python.exe archetype_temporal.py `
  --train-through-ms 1735689599000 `
  --output-dir .\artifacts\archetypes `
  --timeframe 4h --window-size 5 --horizon 4h --clusters 8
```

For an offline replay, prepare an `.npz` with aligned arrays:

- `X`: finite feature matrix;
- `y`: labels in `-1, 0, 1`;
- `end_ms`: feature availability timestamps in chronological order;
- `label_available_ms`: timestamps at which each future-dependent label is known.

Then run:

```powershell
.\venv\Scripts\python.exe archetype_temporal.py `
  --input-npz .\data\btc_4h_ws5.npz `
  --train-through-ms 1735689599000 `
  --output-dir .\artifacts\archetypes `
  --timeframe 4h --window-size 5 --horizon 4h --clusters 8
```

The command writes a frozen model and an evaluation report. It does not write
PostgreSQL. Production/database promotion is a separate decision and requires a
reviewed positive artifact; a negative artifact remains useful evidence.
