# BTCUSDT 4h next-bar ML evidence contract

`ml_evidence_walkforward.py` is the common predictive evaluator for the first
bounded ML benchmark. It is an evidence generator, not a training or production
promotion script. It never writes PostgreSQL, the model registry, or paper/live
trading state.

## Frozen scope

- Symbol and venue key: `BTCUSDT`.
- Timeframe: closed 4-hour bars.
- Decision: after the signal bar closes.
- Outcome: close-to-close direction of the next 4-hour bar from
  `PriceTargets.TargetDirection4h`; the triple-barrier label stored in some window
  datasets is deliberately not used.
- Features: five bars by default, sourced from the 35 stored features per bar.
  `ActiveRuleCount` (offset 29) is removed from every bar because the historical
  value reflects rules enabled at rebuild time and has no point-in-time
  rule-version lineage.
- Declared model family: scaled logistic regression and histogram gradient
  boosting. Adding or tuning another candidate creates a new declared family and
  manifest; it cannot be silently inserted after inspecting results.

## Temporal protocol

Each expanding outer fold has three chronological partitions:

1. base fit;
2. later probability calibration;
3. untouched outer test.

Rows are admitted to fit/calibration only when their next-bar label is available
before the following boundary. This purges boundary-crossing labels. The scaler is
inside the logistic pipeline and is therefore fit on base-fit rows only. The tree
has no global preprocessing. A multinomial sigmoid calibrator sees calibration
probabilities and labels only; it never sees outer-test outcomes.

Four predeclared causal baselines run on every identical outer-test timestamp:
rolling class probabilities, rolling majority, last-bar momentum and last-bar
reversion. Rolling baselines can use only labels whose availability timestamp is
not later than the current decision.

Reports include multiclass Brier loss, log loss, class-wise reliability bins,
balanced accuracy and confusion matrices. Candidate Brier loss is paired against
the strongest baseline on identical timestamps with a time-block bootstrap.
Every declared candidate and baseline is written to the immutable JSONL trial
ledger, including negative and inconclusive results.
The two-candidate family uses a Bonferroni-adjusted bootstrap interval so choosing
the better-looking declared candidate does not silently retain an unadjusted 5%
threshold.

The promotion gate fails closed unless the selected declared model beats the
strongest baseline on both losses, its paired Brier interval excludes zero, all
timestamps have equal coverage, sample support is adequate, and the trial family
is complete. Passing the gate only sets evidence status; this command still does
not mutate any production registry.

## Reproducible command

The explicit cutoff is mandatory so a later rerun cannot silently include new
rows:

```powershell
.\venv\Scripts\python.exe ml_evidence_walkforward.py `
  --decision-cutoff-ms 1758326400000 `
  --output-dir .\artifacts\ml-evidence
```

This opens PostgreSQL in a read-only transaction. For an offline replay, pass
`--input-npz path`, where the archive contains finite aligned arrays `X`, `y`,
`decision_ms`, `label_available_ms`, and `last_return`.

The report manifest binds the exact array bytes, cutoff, feature/label semantics,
parameters, evaluator and research-contract hashes, Git state, and runtime library
versions. Artifacts are named by manifest hash and cannot be overwritten with
different content.
