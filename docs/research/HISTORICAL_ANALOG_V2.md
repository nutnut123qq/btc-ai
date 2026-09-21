# Historical Analog v2 evidence workflow

Run from `ai/`:

```powershell
.\venv\Scripts\python.exe historical_analog_walkforward.py `
  --timeframe 4h `
  --output docs/research/historical_analog_4h_v2.json
```

The command reads BTCUSDT closed candles, evaluates chronologically, writes a
content-addressed manifest into the report, and appends the outcome (including
failed/inconclusive outcomes) to `historical_analog_trials.jsonl`.
Corrections never silently erase an earlier trial; invalidated manifest hashes
and their replacements are recorded in `historical_analog_trial_corrections.jsonl`.

Defaults are intentionally aligned with the product search: 15-bar windows,
50 non-overlapping neighbours and a 20,000-bar candidate lookback. The declared
trial family is limited to the legacy unsigned representation and
`returns_shape_v2_signed`; adding another representation requires changing the
declared family before evaluating it.

## Interpretation

- `roundTripPctPoints = 0.30` means 30 bps round trip (10 bps fee plus 5 bps
  slippage per side). It is only the floor of the direction-label dead zone.
  This report does not calculate fills or net PnL.
- `acceptedQueryCoverage` is the fraction of otherwise eligible query windows
  whose selected neighbours clear the predeclared mean-similarity threshold.
- `rollingCandidateMajorityAccuracy` uses the same accepted query timestamps as
  the analog and only eligible candidates inside the configured rolling
  lookback. It is not an expanding-history baseline.
- `accuracyLift95PctInterval` is a paired circular moving-block bootstrap over
  query order. `inconclusive`, `adverse`, and `insufficient_evidence` are valid
  completed outcomes; none should be rewritten as a probability of winning.

To replay a past result, use the exact parameters and the Klines snapshot whose
row count, timestamp range, and SHA-256 fingerprint are embedded in its
manifest. Appending future candles cannot change a run with a fixed
`evaluation_end_index` in the Python API.
