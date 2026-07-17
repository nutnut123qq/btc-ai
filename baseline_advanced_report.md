# Advanced Baseline Training Report

Generated: 2026-07-16T02:06:02.303301+00:00 UTC
Symbol: BTCUSDT, Timeframe: 4h
Split: train on WindowEndMs < 1735689600000 (2025-01-01 UTC), test >= split

## Summary by (WindowSize, Horizon)

| WS | Horizon | Samples | Best model | Best acc | Best F1 | Majority acc | Majority F1 |
|----|---------|---------|------------|----------|---------|--------------|-------------|
| 20 | 1d | 14083 | XGB_balanced | 0.5588 | 0.5577 | 0.4224 | 0.2509 |
| 20 | 4h | 14083 | XGB_balanced | 0.5999 | 0.5915 | 0.3790 | 0.2083 |
| 25 | 1d | 14073 | XGB_balanced | 0.5543 | 0.5527 | 0.4224 | 0.2509 |
| 25 | 4h | 10102 | HGB_balanced | 0.5656 | 0.5668 | 0.4587 | 0.2885 |

## Detailed Results

### WindowSize=20, Horizon=1d

- Total: 14083, Train: 10724, Test: 3359
- Label distribution (train): {np.int8(1): 5149, np.int8(-1): 4537, np.int8(0): 1038}
- Label distribution (test):  {np.int8(1): 1419, np.int8(0): 548, np.int8(-1): 1392}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.4224 | 0.2509 | 0.00 |
| Random | 0.3921 | 0.3808 | 0.00 |
| LR_balanced | 0.4183 | 0.4135 | 27.32 |
| RF_balanced | 0.4183 | 0.3968 | 14.56 |
| HGB_balanced | 0.5109 | 0.5116 | 154.01 |
| LGB_balanced | 0.5293 | 0.5309 | 239.59 |
| XGB_balanced | 0.5588 | 0.5577 | 289.63 |
| LR_SMOTE | 0.4204 | 0.4139 | 65.87 |
| LR_Undersample | 0.3876 | 0.3807 | 29.42 |

Top 10 important features:
- ws20_bar19_Atr14Pct: 0.021864
- ws20_bar7_DayOfWeekCos: 0.021592
- ws20_bar5_DayOfWeekCos: 0.014853
- ws20_bar11_Atr14Pct: 0.012302
- ws20_bar8_DayOfWeekCos: 0.012042
- ws20_bar4_DayOfWeekCos: 0.007775
- ws20_bar6_DayOfWeekCos: 0.006593
- ws20_bar19_RecentPatternEncoded: 0.006578
- ws20_bar6_Atr14Pct: 0.006396
- ws20_bar15_DayOfWeekSin: 0.006348

### WindowSize=20, Horizon=4h

- Total: 14083, Train: 10724, Test: 3359
- Label distribution (train): {np.int8(1): 5386, np.int8(-1): 3088, np.int8(0): 2250}
- Label distribution (test):  {np.int8(0): 1103, np.int8(-1): 983, np.int8(1): 1273}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3790 | 0.2083 | 0.00 |
| Random | 0.3257 | 0.3144 | 0.00 |
| LR_balanced | 0.4990 | 0.4762 | 59.81 |
| RF_balanced | 0.4799 | 0.4417 | 13.77 |
| HGB_balanced | 0.5781 | 0.5687 | 32.97 |
| LGB_balanced | 0.5936 | 0.5860 | 39.41 |
| XGB_balanced | 0.5999 | 0.5915 | 150.45 |
| LR_SMOTE | 0.4966 | 0.4780 | 45.32 |
| LR_Undersample | 0.4885 | 0.4633 | 16.27 |

Top 10 important features:
- ws20_bar19_Atr14Pct: 0.032050
- ws20_bar3_DayOfWeekCos: 0.012457
- ws20_bar4_DayOfWeekCos: 0.012134
- ws20_bar19_HighLowRangePct: 0.006897
- ws20_bar19_RecentPatternEncoded: 0.006556
- ws20_bar14_DayOfWeekSin: 0.005803
- ws20_bar0_HourSin: 0.005428
- ws20_bar2_HourCos: 0.004416
- ws20_bar15_DayOfWeekSin: 0.004341
- ws20_bar19_ClosePctChange1: 0.004063

### WindowSize=25, Horizon=1d

- Total: 14073, Train: 10714, Test: 3359
- Label distribution (train): {np.int8(-1): 4535, np.int8(0): 1036, np.int8(1): 5143}
- Label distribution (test):  {np.int8(1): 1419, np.int8(0): 548, np.int8(-1): 1392}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.4224 | 0.2509 | 0.00 |
| Random | 0.3921 | 0.3808 | 0.00 |
| LR_balanced | 0.4221 | 0.4194 | 17.54 |
| RF_balanced | 0.4230 | 0.4075 | 12.30 |
| HGB_balanced | 0.5049 | 0.5071 | 19.32 |
| LGB_balanced | 0.5234 | 0.5243 | 24.86 |
| XGB_balanced | 0.5543 | 0.5527 | 73.54 |
| LR_SMOTE | 0.4207 | 0.4179 | 24.07 |
| LR_Undersample | 0.4019 | 0.4001 | 6.35 |

Top 10 important features:
- ws25_bar24_Atr14Pct: 0.015592
- ws25_bar10_DayOfWeekCos: 0.012709
- ws25_bar9_DayOfWeekCos: 0.009813
- ws25_bar0_DayOfWeekSin: 0.008977
- ws25_bar16_Atr14Pct: 0.007258
- ws25_bar0_Atr14Pct: 0.006100
- ws25_bar18_Atr14Pct: 0.006019
- ws25_bar24_RecentPatternEncoded: 0.005737
- ws25_bar16_RollingVwapDist: 0.004969
- ws25_bar24_HighLowRangePct: 0.004896

### WindowSize=25, Horizon=4h

- Total: 10102, Train: 8081, Test: 2021
- Label distribution (train): {np.int8(1): 4161, np.int8(-1): 2310, np.int8(0): 1610}
- Label distribution (test):  {np.int8(1): 927, np.int8(-1): 603, np.int8(0): 491}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.4587 | 0.2885 | 0.00 |
| Random | 0.3652 | 0.3552 | 0.00 |
| LR_balanced | 0.4592 | 0.4564 | 14.87 |
| RF_balanced | 0.4127 | 0.3939 | 6.92 |
| HGB_balanced | 0.5656 | 0.5668 | 42.78 |
| LGB_balanced | 0.5611 | 0.5642 | 25.38 |
| XGB_balanced | 0.5755 | 0.5667 | 99.18 |
| LR_SMOTE | 0.4537 | 0.4542 | 35.65 |
| LR_Undersample | 0.4488 | 0.4477 | 15.68 |

Top 10 important features:
- ws25_bar24_Atr14Pct: 0.030707
- ws25_bar9_DayOfWeekCos: 0.006809
- ws25_bar2_HourSin: 0.005438
- ws25_bar24_Sma50Dist: 0.005358
- ws25_bar22_Atr14Pct: 0.004819
- ws25_bar24_RecentPatternEncoded: 0.004806
- ws25_bar23_Ema200Dist: 0.004644
- ws25_bar3_DayOfWeekSin: 0.004449
- ws25_bar9_Atr14Pct: 0.003885
- ws25_bar24_HighLowRangePct: 0.003626

## Notes
- 'balanced' = sklearn class_weight='balanced'.
- SMOTE/undersample results only shown if imbalanced-learn is installed.
- Time-based split prevents look-ahead leakage.
