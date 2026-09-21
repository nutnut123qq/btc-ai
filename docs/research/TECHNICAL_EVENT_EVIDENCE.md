# BTCUSDT 4h technical-event evidence

`technical_event_evidence.py` is the read-only evidence audit for stored candle
patterns, volume anomalies, market regimes, and causal Smart Money Concepts (SMC)
events. It measures conditional next-bar behavior. It does not simulate execution,
calculate PnL, select a production signal, or update application tables.

## Frozen scope and timing

- Symbol/timeframe: `BTCUSDT` closed `4h` bars.
- Decision: immediately after the event's defining bar has closed.
- Outcome: return from that close to the next contiguous 4h close; direction is
  positive only when that return is greater than zero.
- Baseline: all prior 4h bars in the same causal six-close trend context, using
  only next-bar outcomes already available at the event decision.
- Candidate estimate: the Laplace-smoothed positive rate from prior occurrences
  of the same event type whose outcomes were already available.
- No event type is selected after observing results. Every stored type is
  reported. Volume thresholds `1.5x` and `2.0x` are declared in advance.

Candle-pattern availability is reconstructed conservatively from a frozen map of
the recognizer's known one-, two-, and three-candle pattern names. This avoids a
known storage ambiguity where Morning/Evening Star can be categorized as Double
despite requiring three candles. Unknown multi-pattern names and rows that cannot
be joined contiguously are excluded and counted. Volume and regime rows are
available at the matching finalized candle close because their formulas use that
candle and prior bars.

SMC is evaluated only when the table exposes `OriginTimeMs`, `AvailableTimeMs`,
and `CalculationVersion`, and only for `smc-causal-v2`. The availability timestamp
must join exactly to a finalized 4h close. Later-updated mitigation fields are
never used. Missing lineage produces an explicit `unavailable` module instead of
an inferred signal.

## Reported evidence

For every event type the report includes stored/realized/OOS counts, coverage,
positive-rate Wilson interval, mean-return moving-block bootstrap interval, and a
chronological expanding-prior comparison with the context baseline. Paired Brier
lift and return-minus-prior-context expectation also receive block-bootstrap
intervals. Positive, adverse, inconclusive, insufficient, and unavailable results
are all retained in the immutable trial ledger.

All reported event types form one declared comparison family. Evidence status
uses a Bonferroni-adjusted paired-Brier bootstrap interval across that full family;
the report retains both pointwise and family-adjusted descriptive intervals.
The evaluator rejects a bootstrap count too small to resolve the adjusted
two-sided tail; the default is 2,000 resamples.

The status `positive_predictive_evidence` means only that the event's expanding
prior improved next-bar Brier loss in this bounded test. It is not a claim of
economic value, independent observations, fill quality, or tradable PnL.

## Reproducible command

The cutoff is mandatory, so later market data cannot silently enter a rerun:

```powershell
.\venv\Scripts\python.exe technical_event_evidence.py `
  --decision-cutoff-ms 1789948799999 `
  --output-dir .\docs\research\evidence\technical-events
```

PostgreSQL is opened in a read-only transaction. The manifest binds the exact
closed bars and event rows, cutoff, parameters, code hashes, Git state, database
identity without its password, and runtime versions. A report and JSONL trial
ledger use the manifest hash as their filename and refuse conflicting overwrite.
