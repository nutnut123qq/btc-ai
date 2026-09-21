# BTC 4h ML evidence bundle v3

`ml_evidence_v3.py` creates audit evidence, not a deployable model. A passing
predictive gate never updates the model registry, paper trader, or live trader.

## Research question

On BTCUSDT 4h, does the fixed histogram gradient boosting classifier improve
proper probabilistic scores for next-bar close-to-close direction over a
predeclared set of baselines on future walk-forward blocks?

This is a **retrospective, selection-aware screen**, not an independent
confirmation: HGB was selected in the earlier v2 work using overlapping OOS
history. A confirmatory claim requires untouched post-selection holdout or
forward observations.

The baseline family is deliberately explicit:

- an **expanding historical class prior** (the old `rolling_class_probs` name was
  inaccurate);
- an expanding soft majority;
- finite adaptive priors over the last 180 and 540 labels available at decision
  time (about 30 and 90 days of 4h bars);
- fixed momentum and reversion rules.

## Leakage controls

- Test folds are non-overlapping, forward-only blocks.
- A row may enter fit or calibration only after its target label is available.
- The model is fit on the earlier fit partition. Penalized multinomial logistic
  recalibration over log base probabilities is fit on a separate, later
  calibration partition. If that partition contains only one class, the base
  probabilities are retained. The test partition is touched only for evaluation.
- `ActiveRuleCount` remains excluded because its historical values lack
  point-in-time rule-version lineage.
- Every model, ablation, and baseline must cover the exact same OOS timestamps.
  The evaluator fails closed if coverage differs.

## What is persisted

Each run writes four immutable, content-addressed files:

1. `*.dataset.npz`: exact feature matrix, labels, decision/availability times,
   final-bar return, and causal feature names.
2. `*.predictions.jsonl`: one row per OOS decision with fold ID, realized label,
   and class probabilities for every model, ablation, and baseline.
3. `*.report.json`: hypothesis, protocol, folds, proper scores, reliability bins,
   feature-group ablation, uncertainty sensitivity, limitations, and a
   non-mutating promotion gate.
4. `*.manifest.json`: byte hashes and sizes for the three artifacts plus research,
   code, data, and runtime provenance.

`verify_evidence_bundle()` checks the manifest filename/hash and every referenced
artifact hash and size. It also validates the frozen feature schema and cutoff,
joins every prediction timestamp and label to the snapshot, validates probability
families and sums, reconstructs deterministic fold coverage, and recomputes
metrics, uncertainty intervals, and the retrospective gate. Consumers fail
closed on any byte or semantic mismatch.

## Same-protocol feature evidence

The evaluator reruns HGB while omitting one complete feature group at a time.
All ablations use the full model's folds, calibration partitions, timestamps,
hyperparameters, and random seeds. `meanBrierDamageWhenOmitted > 0` means the
full model scored better, but the familywise block-bootstrap interval controls
the interpretation: `useful`, `harmful`, or `inconclusive`.

This is conditional predictive evidence for the fixed protocol. It is not proof
that a feature group causes market returns.

## Uncertainty sensitivity

The full HGB is compared separately with every baseline for both Brier and log
loss. Bonferroni correction covers all declared baseline × metric comparisons.
Every comparison is recomputed with predeclared circular block sizes of 6, 12,
and 24 rows; the retrospective gate requires positive lower bounds at every
block size. Brier and log-loss strongest baselines are selected independently.
Finite adaptive prior windows of 180 and 540 rows make regime adaptation an
explicit baseline rather than an after-the-fact explanation.

## Run

Production-size read-only PostgreSQL evaluation:

```powershell
python ml_evidence_v3.py `
  --decision-cutoff-ms 1788206399999 `
  --output-dir docs/research/evidence/ml-v3
```

Replay a previously frozen **v3** NPZ without touching PostgreSQL. The loader
requires the exact schema version, window size, feature width, names, and order;
generic or legacy NPZ files are rejected:

```powershell
python ml_evidence_v3.py `
  --decision-cutoff-ms 1788206399999 `
  --input-npz path/to/frozen.dataset.npz `
  --output-dir docs/research/evidence/ml-v3
```

For a fast protocol smoke test, reduce fold sizes and bootstrap samples on a
fixture. Do not present smoke-test metrics as project evidence.

## Honest limits

This protocol tests one symbol, timeframe, target, model family, and frozen
feature schema. It does not include fees, slippage, latency, fills, capacity,
paper outcomes, or live PnL. Brier score combines calibration and discrimination,
so the reliability tables and log loss must be inspected too. Economic,
forward-paper, and live evidence remain separate promotion stages.

A production-size default run performs one full plus seven leave-one-group-out
HGB fits per fold. The report records the exact fit count. Per-fold immutable
checkpoints are not implemented yet, so an interrupted run must restart; run a
resource preflight before starting and never present smoke-test output as project
evidence.
