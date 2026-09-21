# BTC 4h feature-group contribution protocol

`feature_group_ablation.py` measures the out-of-sample classification contribution
of eight declared technical feature groups: price/returns, candle geometry, trend,
momentum, volatility, volume, pattern, and UTC time.

The evaluator reads `WindowClassificationDatasets` and `PriceTargets` in a
read-only PostgreSQL transaction. It uses the complete five-bar closed 4h window
at each decision, removes `ActiveRuleCount` from every bar because that stored
field has no point-in-time rule-version lineage, and trains on labels whose
outcome was already available at the start of each chronological test fold.

The fixed trial family contains a rolling class-prior baseline, a full scaled
multinomial logistic model, each group by itself, and the full model with each
group removed. All trials use identical timestamps. Reports contain Brier score,
log loss, balanced accuracy, coverage, paired moving-block bootstrap intervals,
and a Bonferroni familywise correction. A group is called incrementally positive
only when removing it worsens Brier score and the adjusted confidence interval is
strictly above zero. Negative and inconclusive outcomes remain valid evidence.

The report is a feature-diagnostic artifact. It does not simulate fills, costs,
positions, returns, drawdown, or PnL, and therefore cannot promote a trading
strategy.

Example:

```powershell
$cutoff = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
.\venv\Scripts\python.exe feature_group_ablation.py `
  --decision-cutoff-ms $cutoff `
  --output-dir docs\research\evidence\feature-groups
```

The manifest and report filenames are derived from the manifest SHA-256. Reusing
the same manifest is idempotent; conflicting content is rejected.
