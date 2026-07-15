# Advanced Baseline Training Report

Generated: 2026-07-15T17:06:16.657180+00:00 UTC
Symbol: BTCUSDT, Timeframe: 1h
Split: train on WindowEndMs < 1735689600000 (2025-01-01 UTC), test >= split

## Summary by (WindowSize, Horizon)

| WS | Horizon | Samples | Best model | Best acc | Best F1 | Majority acc | Majority F1 |
|----|---------|---------|------------|----------|---------|--------------|-------------|
| 5 | 1h | 56885 | XGB_balanced | 0.6330 | 0.6280 | 0.3612 | 0.1917 |

## Detailed Results

### WindowSize=5, Horizon=1h

- Total: 56885, Train: 43551, Test: 13334
- Label distribution (train): {np.int8(1): 21010, np.int8(0): 10150, np.int8(-1): 12391}
- Label distribution (test):  {np.int8(-1): 3667, np.int8(1): 4816, np.int8(0): 4851}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3612 | 0.1917 | 0.00 |
| Random | 0.3343 | 0.3273 | 0.00 |
| LR_balanced | 0.5259 | 0.4983 | 14.88 |
| RF_balanced | 0.5390 | 0.5153 | 24.42 |
| HGB_balanced | 0.6214 | 0.6135 | 18.15 |
| LGB_balanced | 0.6222 | 0.6144 | 12.49 |
| XGB_balanced | 0.6330 | 0.6280 | 26.49 |
| LR_SMOTE | 0.5287 | 0.5052 | 22.94 |
| LR_Undersample | 0.5253 | 0.4980 | 11.09 |

Top 10 important features:
- ws5_bar4_Atr14Pct: 0.120542
- ws5_bar4_HighLowRangePct: 0.052431
- ws5_bar4_RecentPatternEncoded: 0.028573
- ws5_bar3_RecentPatternEncoded: 0.016627
- ws5_bar4_ClosePctChange1: 0.014137
- ws5_bar0_IsWeekend: 0.012481
- ws5_bar1_HourCos: 0.012311
- ws5_bar3_HourCos: 0.012229
- ws5_bar3_HighLowRangePct: 0.011549
- ws5_bar2_IsWeekend: 0.010073

## Notes
- 'balanced' = sklearn class_weight='balanced'.
- SMOTE/undersample results only shown if imbalanced-learn is installed.
- Time-based split prevents look-ahead leakage.
