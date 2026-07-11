# Advanced Baseline Training Report

Generated: 2026-07-11T19:04:43.337577 UTC
Symbol: BTCUSDT, Timeframe: 1h
Split: train on WindowEndMs < 1735689600000 (2025-01-01 UTC), test >= split

## Summary by (WindowSize, Horizon)

| WS | Horizon | Samples | Best model | Best acc | Best F1 | Majority acc | Majority F1 |
|----|---------|---------|------------|----------|---------|--------------|-------------|
| 5 | 1d | 56885 | XGB_balanced | 0.5133 | 0.5126 | 0.4150 | 0.2435 |
| 5 | 1h | 56885 | XGB_balanced | 0.6330 | 0.6280 | 0.3612 | 0.1917 |
| 5 | 4h | 56885 | XGB_balanced | 0.5767 | 0.5745 | 0.3374 | 0.1702 |
| 10 | 1d | 56805 | XGB_balanced | 0.5170 | 0.5167 | 0.4150 | 0.2435 |
| 10 | 1h | 56805 | XGB_balanced | 0.6306 | 0.6251 | 0.3612 | 0.1917 |
| 10 | 4h | 56805 | XGB_balanced | 0.5765 | 0.5745 | 0.3374 | 0.1702 |
| 15 | 1d | 56725 | XGB_balanced | 0.5180 | 0.5177 | 0.4150 | 0.2435 |
| 15 | 1h | 56725 | XGB_balanced | 0.6330 | 0.6278 | 0.3612 | 0.1917 |
| 15 | 4h | 56725 | XGB_balanced | 0.5807 | 0.5784 | 0.3374 | 0.1702 |
| 20 | 1d | 56645 | XGB_balanced | 0.5188 | 0.5185 | 0.4150 | 0.2435 |
| 20 | 1h | 56645 | XGB_balanced | 0.6318 | 0.6261 | 0.3612 | 0.1917 |
| 20 | 4h | 56645 | XGB_balanced | 0.5805 | 0.5781 | 0.3374 | 0.1702 |
| 25 | 1d | 56565 | XGB_balanced | 0.5129 | 0.5125 | 0.4150 | 0.2435 |
| 25 | 1h | 56565 | XGB_balanced | 0.6309 | 0.6251 | 0.3612 | 0.1917 |
| 25 | 4h | 56565 | XGB_balanced | 0.5824 | 0.5798 | 0.3374 | 0.1702 |

## Detailed Results

### WindowSize=5, Horizon=1d

- Total: 56885, Train: 43551, Test: 13334
- Label distribution (train): {np.int8(-1): 19529, np.int8(1): 19855, np.int8(0): 4167}
- Label distribution (test):  {np.int8(-1): 5630, np.int8(1): 5534, np.int8(0): 2170}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.4150 | 0.2435 | 0.00 |
| Random | 0.3903 | 0.3794 | 0.00 |
| LR_balanced | 0.3795 | 0.3704 | 17.35 |
| RF_balanced | 0.4300 | 0.4201 | 29.09 |
| HGB_balanced | 0.4837 | 0.4833 | 21.42 |
| LGB_balanced | 0.4777 | 0.4778 | 19.62 |
| XGB_balanced | 0.5133 | 0.5126 | 33.44 |
| LR_SMOTE | 0.3864 | 0.3786 | 23.55 |
| LR_Undersample | 0.3786 | 0.3696 | 5.79 |

Top 10 important features:
- ws5_bar4_Atr14Pct: 0.076959
- ws5_bar4_DayOfWeekSin: 0.036005
- ws5_bar2_DayOfWeekSin: 0.023242
- ws5_bar4_RecentPatternEncoded: 0.022592
- ws5_bar1_DayOfWeekSin: 0.018522
- ws5_bar3_DayOfWeekSin: 0.016972
- ws5_bar0_DayOfWeekSin: 0.016798
- ws5_bar4_ClosePctChange1: 0.011367
- ws5_bar1_HourSin: 0.011358
- ws5_bar3_RecentPatternEncoded: 0.011296

### WindowSize=5, Horizon=1h

- Total: 56885, Train: 43551, Test: 13334
- Label distribution (train): {np.int8(1): 21010, np.int8(0): 10150, np.int8(-1): 12391}
- Label distribution (test):  {np.int8(-1): 3667, np.int8(1): 4816, np.int8(0): 4851}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3612 | 0.1917 | 0.00 |
| Random | 0.3343 | 0.3273 | 0.00 |
| LR_balanced | 0.5259 | 0.4983 | 15.27 |
| RF_balanced | 0.5390 | 0.5153 | 27.95 |
| HGB_balanced | 0.6214 | 0.6135 | 14.86 |
| LGB_balanced | 0.6222 | 0.6144 | 15.47 |
| XGB_balanced | 0.6330 | 0.6280 | 32.44 |
| LR_SMOTE | 0.5287 | 0.5052 | 25.62 |
| LR_Undersample | 0.5253 | 0.4980 | 12.30 |

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

### WindowSize=5, Horizon=4h

- Total: 56885, Train: 43551, Test: 13334
- Label distribution (train): {np.int8(1): 18357, np.int8(-1): 16032, np.int8(0): 9162}
- Label distribution (test):  {np.int8(-1): 4439, np.int8(0): 4396, np.int8(1): 4499}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3374 | 0.1702 | 0.00 |
| Random | 0.3317 | 0.3250 | 0.00 |
| LR_balanced | 0.4804 | 0.4466 | 17.56 |
| RF_balanced | 0.5052 | 0.4752 | 27.88 |
| HGB_balanced | 0.5631 | 0.5512 | 15.58 |
| LGB_balanced | 0.5624 | 0.5493 | 15.56 |
| XGB_balanced | 0.5767 | 0.5745 | 32.92 |
| LR_SMOTE | 0.4847 | 0.4541 | 21.87 |
| LR_Undersample | 0.4756 | 0.4408 | 10.31 |

Top 10 important features:
- ws5_bar4_Atr14Pct: 0.101592
- ws5_bar4_RecentPatternEncoded: 0.027615
- ws5_bar4_HighLowRangePct: 0.021973
- ws5_bar4_HourCos: 0.021741
- ws5_bar0_IsWeekend: 0.017385
- ws5_bar3_RecentPatternEncoded: 0.016197
- ws5_bar4_ClosePctChange1: 0.015950
- ws5_bar2_HourCos: 0.013630
- ws5_bar0_DayOfWeekSin: 0.012579
- ws5_bar1_HourCos: 0.012169

### WindowSize=10, Horizon=1d

- Total: 56805, Train: 43471, Test: 13334
- Label distribution (train): {np.int8(1): 19812, np.int8(-1): 19492, np.int8(0): 4167}
- Label distribution (test):  {np.int8(-1): 5630, np.int8(1): 5534, np.int8(0): 2170}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.4150 | 0.2435 | 0.00 |
| Random | 0.3901 | 0.3794 | 0.00 |
| LR_balanced | 0.3863 | 0.3773 | 36.15 |
| RF_balanced | 0.4282 | 0.4161 | 56.43 |
| HGB_balanced | 0.4894 | 0.4887 | 80.54 |
| LGB_balanced | 0.4868 | 0.4868 | 37.00 |
| XGB_balanced | 0.5170 | 0.5167 | 78.85 |
| LR_SMOTE | 0.3934 | 0.3862 | 46.89 |
| LR_Undersample | 0.3836 | 0.3744 | 18.84 |

Top 10 important features:
- ws10_bar9_Atr14Pct: 0.050034
- ws10_bar9_DayOfWeekSin: 0.026105
- ws10_bar8_DayOfWeekSin: 0.020723
- ws10_bar9_RecentPatternEncoded: 0.013672
- ws10_bar0_DayOfWeekSin: 0.012804
- ws10_bar7_DayOfWeekSin: 0.011343
- ws10_bar1_DayOfWeekSin: 0.008686
- ws10_bar9_HighLowRangePct: 0.007600
- ws10_bar5_DayOfWeekSin: 0.007145
- ws10_bar9_ClosePctChange1: 0.006883

### WindowSize=10, Horizon=1h

- Total: 56805, Train: 43471, Test: 13334
- Label distribution (train): {np.int8(1): 20962, np.int8(-1): 12365, np.int8(0): 10144}
- Label distribution (test):  {np.int8(-1): 3667, np.int8(1): 4816, np.int8(0): 4851}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3612 | 0.1917 | 0.00 |
| Random | 0.3343 | 0.3273 | 0.00 |
| LR_balanced | 0.5243 | 0.4975 | 34.15 |
| RF_balanced | 0.5302 | 0.5043 | 42.59 |
| HGB_balanced | 0.6228 | 0.6146 | 21.64 |
| LGB_balanced | 0.6209 | 0.6125 | 27.47 |
| XGB_balanced | 0.6306 | 0.6251 | 65.00 |
| LR_SMOTE | 0.5268 | 0.5041 | 45.90 |
| LR_Undersample | 0.5268 | 0.5005 | 22.59 |

Top 10 important features:
- ws10_bar9_Atr14Pct: 0.079040
- ws10_bar9_HighLowRangePct: 0.032334
- ws10_bar9_RecentPatternEncoded: 0.017390
- ws10_bar8_RecentPatternEncoded: 0.009988
- ws10_bar1_DayOfWeekSin: 0.009263
- ws10_bar9_ClosePctChange1: 0.008668
- ws10_bar8_HighLowRangePct: 0.007744
- ws10_bar2_DayOfWeekSin: 0.007352
- ws10_bar2_HourSin: 0.007324
- ws10_bar0_IsWeekend: 0.007042

### WindowSize=10, Horizon=4h

- Total: 56805, Train: 43471, Test: 13334
- Label distribution (train): {np.int8(1): 18320, np.int8(-1): 15994, np.int8(0): 9157}
- Label distribution (test):  {np.int8(-1): 4439, np.int8(0): 4396, np.int8(1): 4499}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3374 | 0.1702 | 0.00 |
| Random | 0.3315 | 0.3248 | 0.00 |
| LR_balanced | 0.4791 | 0.4466 | 33.71 |
| RF_balanced | 0.4948 | 0.4615 | 38.18 |
| HGB_balanced | 0.5631 | 0.5515 | 46.63 |
| LGB_balanced | 0.5649 | 0.5524 | 39.18 |
| XGB_balanced | 0.5765 | 0.5745 | 75.65 |
| LR_SMOTE | 0.4814 | 0.4519 | 38.58 |
| LR_Undersample | 0.4745 | 0.4408 | 19.82 |

Top 10 important features:
- ws10_bar9_Atr14Pct: 0.066162
- ws10_bar9_RecentPatternEncoded: 0.016335
- ws10_bar9_HighLowRangePct: 0.015794
- ws10_bar3_HourSin: 0.013778
- ws10_bar0_DayOfWeekSin: 0.011706
- ws10_bar0_IsWeekend: 0.009699
- ws10_bar8_RecentPatternEncoded: 0.009241
- ws10_bar9_ClosePctChange1: 0.008931
- ws10_bar9_DayOfWeekCos: 0.008173
- ws10_bar0_HourSin: 0.007585

### WindowSize=15, Horizon=1d

- Total: 56725, Train: 43391, Test: 13334
- Label distribution (train): {np.int8(-1): 19456, np.int8(1): 19768, np.int8(0): 4167}
- Label distribution (test):  {np.int8(-1): 5630, np.int8(1): 5534, np.int8(0): 2170}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.4150 | 0.2435 | 0.00 |
| Random | 0.3901 | 0.3794 | 0.00 |
| LR_balanced | 0.3906 | 0.3823 | 44.85 |
| RF_balanced | 0.4273 | 0.4155 | 59.74 |
| HGB_balanced | 0.4953 | 0.4954 | 68.45 |
| LGB_balanced | 0.4888 | 0.4886 | 47.97 |
| XGB_balanced | 0.5180 | 0.5177 | 128.26 |
| LR_SMOTE | 0.4010 | 0.3938 | 83.28 |
| LR_Undersample | 0.3849 | 0.3758 | 19.28 |

Top 10 important features:
- ws15_bar14_Atr14Pct: 0.034056
- ws15_bar14_DayOfWeekSin: 0.026839
- ws15_bar13_DayOfWeekSin: 0.017963
- ws15_bar2_DayOfWeekSin: 0.014984
- ws15_bar6_DayOfWeekSin: 0.009172
- ws15_bar14_RecentPatternEncoded: 0.009085
- ws15_bar2_IsWeekend: 0.007419
- ws15_bar12_DayOfWeekSin: 0.007199
- ws15_bar3_DayOfWeekSin: 0.007197
- ws15_bar4_DayOfWeekSin: 0.006290

### WindowSize=15, Horizon=1h

- Total: 56725, Train: 43391, Test: 13334
- Label distribution (train): {np.int8(1): 20913, np.int8(-1): 12343, np.int8(0): 10135}
- Label distribution (test):  {np.int8(-1): 3667, np.int8(1): 4816, np.int8(0): 4851}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3612 | 0.1917 | 0.00 |
| Random | 0.3342 | 0.3273 | 0.00 |
| LR_balanced | 0.5244 | 0.4976 | 61.19 |
| RF_balanced | 0.5247 | 0.4968 | 44.89 |
| HGB_balanced | 0.6225 | 0.6145 | 28.63 |
| LGB_balanced | 0.6224 | 0.6143 | 24.83 |
| XGB_balanced | 0.6330 | 0.6278 | 71.00 |
| LR_SMOTE | 0.5271 | 0.5043 | 50.02 |
| LR_Undersample | 0.5211 | 0.4944 | 22.64 |

Top 10 important features:
- ws15_bar14_Atr14Pct: 0.059014
- ws15_bar14_HighLowRangePct: 0.024239
- ws15_bar14_RecentPatternEncoded: 0.012046
- ws15_bar1_DayOfWeekSin: 0.007576
- ws15_bar13_RecentPatternEncoded: 0.006941
- ws15_bar2_DayOfWeekSin: 0.006448
- ws15_bar3_IsWeekend: 0.006270
- ws15_bar13_HighLowRangePct: 0.006177
- ws15_bar14_ClosePctChange1: 0.006091
- ws15_bar5_HourSin: 0.005145

### WindowSize=15, Horizon=4h

- Total: 56725, Train: 43391, Test: 13334
- Label distribution (train): {np.int8(1): 18279, np.int8(0): 9147, np.int8(-1): 15965}
- Label distribution (test):  {np.int8(-1): 4439, np.int8(0): 4396, np.int8(1): 4499}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3374 | 0.1702 | 0.00 |
| Random | 0.3315 | 0.3248 | 0.00 |
| LR_balanced | 0.4834 | 0.4517 | 36.18 |
| RF_balanced | 0.4882 | 0.4524 | 30.24 |
| HGB_balanced | 0.5658 | 0.5544 | 32.59 |
| LGB_balanced | 0.5670 | 0.5546 | 23.88 |
| XGB_balanced | 0.5807 | 0.5784 | 94.09 |
| LR_SMOTE | 0.4864 | 0.4571 | 51.63 |
| LR_Undersample | 0.4848 | 0.4532 | 27.23 |

Top 10 important features:
- ws15_bar14_Atr14Pct: 0.046741
- ws15_bar14_RecentPatternEncoded: 0.011477
- ws15_bar1_DayOfWeekSin: 0.010844
- ws15_bar14_HighLowRangePct: 0.010801
- ws15_bar2_HourCos: 0.010054
- ws15_bar0_DayOfWeekSin: 0.008381
- ws15_bar4_IsWeekend: 0.006903
- ws15_bar13_RecentPatternEncoded: 0.006833
- ws15_bar14_ClosePctChange1: 0.006630
- ws15_bar5_HourSin: 0.006235

### WindowSize=20, Horizon=1d

- Total: 56645, Train: 43311, Test: 13334
- Label distribution (train): {np.int8(-1): 19430, np.int8(1): 19719, np.int8(0): 4162}
- Label distribution (test):  {np.int8(-1): 5630, np.int8(1): 5534, np.int8(0): 2170}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.4150 | 0.2435 | 0.00 |
| Random | 0.3901 | 0.3794 | 0.00 |
| LR_balanced | 0.3951 | 0.3859 | 54.59 |
| RF_balanced | 0.4248 | 0.4126 | 59.57 |
| HGB_balanced | 0.4861 | 0.4862 | 67.41 |
| LGB_balanced | 0.4873 | 0.4873 | 50.81 |
| XGB_balanced | 0.5188 | 0.5185 | 122.85 |
| LR_SMOTE | 0.4051 | 0.3975 | 84.77 |
| LR_Undersample | 0.3928 | 0.3839 | 17.41 |

Top 10 important features:
- ws20_bar19_Atr14Pct: 0.026223
- ws20_bar19_DayOfWeekSin: 0.022318
- ws20_bar18_DayOfWeekSin: 0.017075
- ws20_bar7_DayOfWeekSin: 0.013210
- ws20_bar19_RecentPatternEncoded: 0.006782
- ws20_bar4_DayOfWeekSin: 0.006656
- ws20_bar9_DayOfWeekSin: 0.006374
- ws20_bar17_DayOfWeekSin: 0.006123
- ws20_bar11_DayOfWeekSin: 0.005707
- ws20_bar3_DayOfWeekSin: 0.005406

### WindowSize=20, Horizon=1h

- Total: 56645, Train: 43311, Test: 13334
- Label distribution (train): {np.int8(-1): 12327, np.int8(1): 20860, np.int8(0): 10124}
- Label distribution (test):  {np.int8(-1): 3667, np.int8(1): 4816, np.int8(0): 4851}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3612 | 0.1917 | 0.00 |
| Random | 0.3343 | 0.3274 | 0.00 |
| LR_balanced | 0.5228 | 0.4966 | 57.08 |
| RF_balanced | 0.5229 | 0.4940 | 56.25 |
| HGB_balanced | 0.6222 | 0.6144 | 43.71 |
| LGB_balanced | 0.6234 | 0.6151 | 52.90 |
| XGB_balanced | 0.6318 | 0.6261 | 124.65 |
| LR_SMOTE | 0.5261 | 0.5031 | 78.24 |
| LR_Undersample | 0.5241 | 0.4982 | 40.67 |

Top 10 important features:
- ws20_bar19_Atr14Pct: 0.046136
- ws20_bar19_HighLowRangePct: 0.019606
- ws20_bar19_RecentPatternEncoded: 0.009534
- ws20_bar4_DayOfWeekSin: 0.005331
- ws20_bar18_RecentPatternEncoded: 0.005256
- ws20_bar2_Atr14Pct: 0.004908
- ws20_bar19_ClosePctChange1: 0.004782
- ws20_bar0_DayOfWeekSin: 0.004511
- ws20_bar7_DayOfWeekSin: 0.004498
- ws20_bar18_HighLowRangePct: 0.004412

### WindowSize=20, Horizon=4h

- Total: 56645, Train: 43311, Test: 13334
- Label distribution (train): {np.int8(-1): 15945, np.int8(1): 18237, np.int8(0): 9129}
- Label distribution (test):  {np.int8(-1): 4439, np.int8(0): 4396, np.int8(1): 4499}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3374 | 0.1702 | 0.00 |
| Random | 0.3316 | 0.3249 | 0.00 |
| LR_balanced | 0.4838 | 0.4525 | 57.25 |
| RF_balanced | 0.4912 | 0.4563 | 64.84 |
| HGB_balanced | 0.5589 | 0.5459 | 56.62 |
| LGB_balanced | 0.5629 | 0.5501 | 56.24 |
| XGB_balanced | 0.5805 | 0.5781 | 131.98 |
| LR_SMOTE | 0.4836 | 0.4546 | 74.10 |
| LR_Undersample | 0.4810 | 0.4501 | 38.25 |

Top 10 important features:
- ws20_bar19_Atr14Pct: 0.035646
- ws20_bar4_DayOfWeekSin: 0.011658
- ws20_bar6_DayOfWeekSin: 0.008825
- ws20_bar19_HighLowRangePct: 0.008784
- ws20_bar19_RecentPatternEncoded: 0.008680
- ws20_bar3_Atr14Pct: 0.007959
- ws20_bar2_DayOfWeekCos: 0.007267
- ws20_bar1_HourSin: 0.007002
- ws20_bar2_DayOfWeekSin: 0.006202
- ws20_bar4_HourCos: 0.005902

### WindowSize=25, Horizon=1d

- Total: 56565, Train: 43231, Test: 13334
- Label distribution (train): {np.int8(1): 19672, np.int8(-1): 19402, np.int8(0): 4157}
- Label distribution (test):  {np.int8(-1): 5630, np.int8(1): 5534, np.int8(0): 2170}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.4150 | 0.2435 | 0.00 |
| Random | 0.3902 | 0.3796 | 0.00 |
| LR_balanced | 0.4012 | 0.3931 | 72.59 |
| RF_balanced | 0.4265 | 0.4156 | 66.12 |
| HGB_balanced | 0.4878 | 0.4877 | 86.16 |
| LGB_balanced | 0.4812 | 0.4813 | 66.67 |
| XGB_balanced | 0.5129 | 0.5125 | 156.38 |
| LR_SMOTE | 0.4105 | 0.4036 | 94.94 |
| LR_Undersample | 0.4008 | 0.3925 | 21.96 |

Top 10 important features:
- ws25_bar24_Atr14Pct: 0.022265
- ws25_bar24_DayOfWeekSin: 0.018606
- ws25_bar23_DayOfWeekSin: 0.014259
- ws25_bar12_DayOfWeekSin: 0.010471
- ws25_bar22_DayOfWeekSin: 0.008280
- ws25_bar24_RecentPatternEncoded: 0.005703
- ws25_bar8_DayOfWeekSin: 0.005254
- ws25_bar3_DayOfWeekSin: 0.005046
- ws25_bar13_DayOfWeekSin: 0.004945
- ws25_bar3_DayOfWeekCos: 0.004215

### WindowSize=25, Horizon=1h

- Total: 56565, Train: 43231, Test: 13334
- Label distribution (train): {np.int8(-1): 12302, np.int8(1): 20818, np.int8(0): 10111}
- Label distribution (test):  {np.int8(-1): 3667, np.int8(1): 4816, np.int8(0): 4851}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3612 | 0.1917 | 0.00 |
| Random | 0.3343 | 0.3275 | 0.00 |
| LR_balanced | 0.5257 | 0.5002 | 71.44 |
| RF_balanced | 0.5202 | 0.4910 | 62.31 |
| HGB_balanced | 0.6252 | 0.6174 | 56.42 |
| LGB_balanced | 0.6225 | 0.6144 | 73.14 |
| XGB_balanced | 0.6309 | 0.6251 | 173.75 |
| LR_SMOTE | 0.5274 | 0.5058 | 98.00 |
| LR_Undersample | 0.5208 | 0.4944 | 49.17 |

Top 10 important features:
- ws25_bar24_Atr14Pct: 0.038109
- ws25_bar24_HighLowRangePct: 0.016463
- ws25_bar24_RecentPatternEncoded: 0.007931
- ws25_bar5_DayOfWeekSin: 0.005252
- ws25_bar23_Atr14Pct: 0.004980
- ws25_bar12_DayOfWeekSin: 0.004767
- ws25_bar23_HighLowRangePct: 0.004631
- ws25_bar23_RecentPatternEncoded: 0.004290
- ws25_bar7_Atr14Pct: 0.004162
- ws25_bar3_DayOfWeekSin: 0.004154

### WindowSize=25, Horizon=4h

- Total: 56565, Train: 43231, Test: 13334
- Label distribution (train): {np.int8(-1): 15915, np.int8(1): 18201, np.int8(0): 9115}
- Label distribution (test):  {np.int8(-1): 4439, np.int8(0): 4396, np.int8(1): 4499}

| Model | Acc | F1-w | Fit(s) |
|-------|-----|------|--------|
| MajorityClass | 0.3374 | 0.1702 | 0.00 |
| Random | 0.3315 | 0.3248 | 0.00 |
| LR_balanced | 0.4858 | 0.4548 | 71.28 |
| RF_balanced | 0.4867 | 0.4507 | 62.03 |
| HGB_balanced | 0.5619 | 0.5500 | 77.53 |
| LGB_balanced | 0.5670 | 0.5539 | 71.67 |
| XGB_balanced | 0.5824 | 0.5798 | 162.22 |
| LR_SMOTE | 0.4900 | 0.4631 | 100.05 |
| LR_Undersample | 0.4851 | 0.4543 | 49.29 |

Top 10 important features:
- ws25_bar24_Atr14Pct: 0.029111
- ws25_bar11_DayOfWeekSin: 0.009679
- ws25_bar9_DayOfWeekSin: 0.009242
- ws25_bar8_Atr14Pct: 0.007839
- ws25_bar24_HighLowRangePct: 0.007389
- ws25_bar24_RecentPatternEncoded: 0.006800
- ws25_bar0_IsWeekend: 0.006480
- ws25_bar7_DayOfWeekCos: 0.006303
- ws25_bar7_DayOfWeekSin: 0.005699
- ws25_bar0_HourCos: 0.005655

## Notes
- 'balanced' = sklearn class_weight='balanced'.
- SMOTE/undersample results only shown if imbalanced-learn is installed.
- Time-based split prevents look-ahead leakage.
