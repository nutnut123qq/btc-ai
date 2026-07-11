# Baseline Training Report — WindowClassificationDatasets 1h

Generated: 2026-07-11T11:53:40.307895 UTC
Symbol: BTCUSDT, Timeframe: 1h
Split: train on WindowEndMs < 1735689600000 (2025-01-01 UTC), test >= split

## Goal

Verify that the derived window-classification dataset contains a learnable signal 
by training lightweight models per (window_size, horizon) and comparing them to 
random (33.3%) and majority-class baselines.

## Summary by Horizon

| Horizon | Window sizes | Total samples | Best model | Best accuracy | Mean LR acc | Mean GB acc |
|---------|--------------|---------------|------------|---------------|-------------|-------------|
| 1d | 5,10,15,20,25 | 142129 | LR | 0.5082 | 0.5055 | 0.5018 |
| 1h | 5,10,15,20,25 | 142129 | LR | 0.6529 | 0.6482 | 0.6456 |
| 4h | 5,10,15,20,25 | 142129 | LR | 0.6397 | 0.6365 | 0.6347 |

## Detailed Results

### Horizon 1d

#### Window size 5

- Total samples: 28505
- Train samples: 21850, Test samples: 6655
- Label distribution (train): {np.int8(1): 6626, np.int8(0): 9508, np.int8(-1): 5716}
- Label distribution (test):  {np.int8(1): 1638, np.int8(0): 3320, np.int8(-1): 1697}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.4989 | 0.3321 | 0.00 |
| Random | 0.3563 | 0.3627 | 0.00 |
| LogisticRegression | 0.5040 | 0.3720 | 2.96 |
| GradientBoosting | 0.5020 | 0.3918 | 1.68 |

Top 10 features (LogisticRegression mean |coef|):
- ws5_bar4_Ema26Dist: 0.737028
- ws5_bar3_ClosePctChange1: 0.729822
- ws5_bar2_ClosePctChange1: 0.724594
- ws5_bar4_ClosePctChange1: 0.679848
- ws5_bar4_Ema12Dist: 0.666187
- ws5_bar0_Ema26Dist: 0.463544
- ws5_bar1_Ema26Dist: 0.462645
- ws5_bar1_ClosePctChange1: 0.444256
- ws5_bar0_Ema12Dist: 0.337614
- ws5_bar3_Ema12Dist: 0.298889

Top 10 features (GradientBoosting importance):

#### Window size 10

- Total samples: 28467
- Train samples: 21812, Test samples: 6655
- Label distribution (train): {np.int8(0): 9446, np.int8(1): 6637, np.int8(-1): 5729}
- Label distribution (test):  {np.int8(0): 3368, np.int8(1): 1617, np.int8(-1): 1670}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.5061 | 0.3401 | 0.00 |
| Random | 0.3560 | 0.3635 | 0.00 |
| LogisticRegression | 0.5082 | 0.3763 | 3.32 |
| GradientBoosting | 0.5038 | 0.3926 | 0.58 |

Top 10 features (LogisticRegression mean |coef|):
- ws10_bar9_Ema26Dist: 0.925190
- ws10_bar8_ClosePctChange1: 0.778445
- ws10_bar0_Ema26Dist: 0.684320
- ws10_bar3_ClosePctChange1: 0.676667
- ws10_bar7_ClosePctChange1: 0.675485
- ws10_bar9_ClosePctChange1: 0.662348
- ws10_bar4_ClosePctChange1: 0.628602
- ws10_bar6_ClosePctChange1: 0.606175
- ws10_bar5_ClosePctChange1: 0.592762
- ws10_bar2_ClosePctChange1: 0.571355

Top 10 features (GradientBoosting importance):

#### Window size 15

- Total samples: 28425
- Train samples: 21770, Test samples: 6655
- Label distribution (train): {np.int8(0): 9477, np.int8(1): 6589, np.int8(-1): 5704}
- Label distribution (test):  {np.int8(1): 1638, np.int8(0): 3320, np.int8(-1): 1697}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.4989 | 0.3321 | 0.00 |
| Random | 0.3563 | 0.3627 | 0.00 |
| LogisticRegression | 0.5035 | 0.3744 | 5.32 |
| GradientBoosting | 0.5028 | 0.3970 | 0.87 |

Top 10 features (LogisticRegression mean |coef|):
- ws15_bar14_Ema26Dist: 0.979964
- ws15_bar12_ClosePctChange1: 0.830865
- ws15_bar5_ClosePctChange1: 0.804566
- ws15_bar3_ClosePctChange1: 0.763670
- ws15_bar13_ClosePctChange1: 0.746501
- ws15_bar2_ClosePctChange1: 0.718161
- ws15_bar7_ClosePctChange1: 0.692419
- ws15_bar14_ClosePctChange1: 0.673498
- ws15_bar11_ClosePctChange1: 0.662546
- ws15_bar6_ClosePctChange1: 0.646928

Top 10 features (GradientBoosting importance):

#### Window size 20

- Total samples: 28387
- Train samples: 21732, Test samples: 6655
- Label distribution (train): {np.int8(0): 9415, np.int8(1): 6603, np.int8(-1): 5714}
- Label distribution (test):  {np.int8(0): 3368, np.int8(1): 1617, np.int8(-1): 1670}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.5061 | 0.3401 | 0.00 |
| Random | 0.3555 | 0.3630 | 0.00 |
| LogisticRegression | 0.5059 | 0.3796 | 7.04 |
| GradientBoosting | 0.5014 | 0.4009 | 1.70 |

Top 10 features (LogisticRegression mean |coef|):
- ws20_bar10_ClosePctChange1: 0.847364
- ws20_bar19_Ema26Dist: 0.825866
- ws20_bar11_ClosePctChange1: 0.726085
- ws20_bar9_ClosePctChange1: 0.672334
- ws20_bar18_ClosePctChange1: 0.657598
- ws20_bar5_ClosePctChange1: 0.653506
- ws20_bar12_ClosePctChange1: 0.649420
- ws20_bar13_ClosePctChange1: 0.618246
- ws20_bar3_ClosePctChange1: 0.602444
- ws20_bar6_ClosePctChange1: 0.570436

Top 10 features (GradientBoosting importance):

#### Window size 25

- Total samples: 28345
- Train samples: 21690, Test samples: 6655
- Label distribution (train): {np.int8(1): 6553, np.int8(0): 9451, np.int8(-1): 5686}
- Label distribution (test):  {np.int8(1): 1638, np.int8(0): 3320, np.int8(-1): 1697}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.4989 | 0.3321 | 0.00 |
| Random | 0.3563 | 0.3626 | 0.00 |
| LogisticRegression | 0.5056 | 0.3841 | 8.06 |
| GradientBoosting | 0.4987 | 0.3975 | 1.99 |

Top 10 features (LogisticRegression mean |coef|):
- ws25_bar6_ClosePctChange1: 0.846825
- ws25_bar24_Ema26Dist: 0.818854
- ws25_bar12_ClosePctChange1: 0.739444
- ws25_bar13_ClosePctChange1: 0.676712
- ws25_bar5_ClosePctChange1: 0.641046
- ws25_bar15_ClosePctChange1: 0.630095
- ws25_bar22_ClosePctChange1: 0.587722
- ws25_bar7_ClosePctChange1: 0.574387
- ws25_bar11_ClosePctChange1: 0.540318
- ws25_bar24_ClosePctChange1: 0.532193

Top 10 features (GradientBoosting importance):

### Horizon 1h

#### Window size 5

- Total samples: 28505
- Train samples: 21850, Test samples: 6655
- Label distribution (train): {np.int8(0): 12256, np.int8(1): 4949, np.int8(-1): 4645}
- Label distribution (test):  {np.int8(1): 1196, np.int8(-1): 1182, np.int8(0): 4277}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.6427 | 0.5029 | 0.00 |
| Random | 0.4437 | 0.4609 | 0.00 |
| LogisticRegression | 0.6443 | 0.5267 | 2.49 |
| GradientBoosting | 0.6439 | 0.5396 | 0.43 |

Top 10 features (LogisticRegression mean |coef|):
- ws5_bar4_Ema12Dist: 1.348345
- ws5_bar4_ClosePctChange1: 1.072301
- ws5_bar3_ClosePctChange1: 0.789599
- ws5_bar4_Ema26Dist: 0.713225
- ws5_bar2_ClosePctChange1: 0.662732
- ws5_bar0_Ema12Dist: 0.658315
- ws5_bar3_Ema26Dist: 0.550207
- ws5_bar1_MacdHistogramNorm: 0.509905
- ws5_bar1_ClosePctChange1: 0.431577
- ws5_bar2_MacdHistogramNorm: 0.243928

Top 10 features (GradientBoosting importance):

#### Window size 10

- Total samples: 28467
- Train samples: 21812, Test samples: 6655
- Label distribution (train): {np.int8(1): 4922, np.int8(-1): 4656, np.int8(0): 12234}
- Label distribution (test):  {np.int8(-1): 1145, np.int8(0): 4326, np.int8(1): 1184}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.6500 | 0.5122 | 0.00 |
| Random | 0.4377 | 0.4567 | 0.00 |
| LogisticRegression | 0.6529 | 0.5372 | 3.93 |
| GradientBoosting | 0.6502 | 0.5398 | 0.64 |

Top 10 features (LogisticRegression mean |coef|):
- ws10_bar2_ClosePctChange1: 1.058789
- ws10_bar9_Ema12Dist: 0.867572
- ws10_bar3_ClosePctChange1: 0.817491
- ws10_bar6_ClosePctChange1: 0.799777
- ws10_bar5_ClosePctChange1: 0.712382
- ws10_bar7_ClosePctChange1: 0.660258
- ws10_bar9_ClosePctChange1: 0.639391
- ws10_bar8_ClosePctChange1: 0.590848
- ws10_bar0_Ema12Dist: 0.586619
- ws10_bar4_ClosePctChange1: 0.569191

Top 10 features (GradientBoosting importance):

#### Window size 15

- Total samples: 28425
- Train samples: 21770, Test samples: 6655
- Label distribution (train): {np.int8(1): 4930, np.int8(0): 12221, np.int8(-1): 4619}
- Label distribution (test):  {np.int8(1): 1196, np.int8(-1): 1182, np.int8(0): 4277}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.6427 | 0.5029 | 0.00 |
| Random | 0.4439 | 0.4610 | 0.00 |
| LogisticRegression | 0.6452 | 0.5291 | 5.71 |
| GradientBoosting | 0.6416 | 0.5340 | 1.75 |

Top 10 features (LogisticRegression mean |coef|):
- ws15_bar14_Ema12Dist: 1.239861
- ws15_bar14_ClosePctChange1: 0.956616
- ws15_bar4_ClosePctChange1: 0.764398
- ws15_bar13_ClosePctChange1: 0.762694
- ws15_bar12_ClosePctChange1: 0.732621
- ws15_bar11_ClosePctChange1: 0.653638
- ws15_bar3_ClosePctChange1: 0.639253
- ws15_bar7_ClosePctChange1: 0.625424
- ws15_bar5_ClosePctChange1: 0.614538
- ws15_bar13_Ema26Dist: 0.605305

Top 10 features (GradientBoosting importance):

#### Window size 20

- Total samples: 28387
- Train samples: 21732, Test samples: 6655
- Label distribution (train): {np.int8(-1): 4643, np.int8(1): 4896, np.int8(0): 12193}
- Label distribution (test):  {np.int8(-1): 1145, np.int8(0): 4326, np.int8(1): 1184}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.6500 | 0.5122 | 0.00 |
| Random | 0.4382 | 0.4571 | 0.00 |
| LogisticRegression | 0.6523 | 0.5378 | 6.35 |
| GradientBoosting | 0.6508 | 0.5435 | 1.58 |

Top 10 features (LogisticRegression mean |coef|):
- ws20_bar12_ClosePctChange1: 1.023970
- ws20_bar13_ClosePctChange1: 0.763239
- ws20_bar5_ClosePctChange1: 0.720391
- ws20_bar11_ClosePctChange1: 0.690736
- ws20_bar3_ClosePctChange1: 0.645518
- ws20_bar6_ClosePctChange1: 0.644338
- ws20_bar16_ClosePctChange1: 0.639733
- ws20_bar15_ClosePctChange1: 0.586136
- ws20_bar19_Ema12Dist: 0.561723
- ws20_bar4_ClosePctChange1: 0.557661

Top 10 features (GradientBoosting importance):

#### Window size 25

- Total samples: 28345
- Train samples: 21690, Test samples: 6655
- Label distribution (train): {np.int8(0): 12180, np.int8(-1): 4607, np.int8(1): 4903}
- Label distribution (test):  {np.int8(1): 1196, np.int8(-1): 1182, np.int8(0): 4277}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.6427 | 0.5029 | 0.00 |
| Random | 0.4437 | 0.4609 | 0.00 |
| LogisticRegression | 0.6461 | 0.5323 | 8.18 |
| GradientBoosting | 0.6415 | 0.5357 | 2.43 |

Top 10 features (LogisticRegression mean |coef|):
- ws25_bar10_ClosePctChange1: 1.077817
- ws25_bar24_Ema12Dist: 0.903975
- ws25_bar3_ClosePctChange1: 0.844449
- ws25_bar4_ClosePctChange1: 0.800203
- ws25_bar8_ClosePctChange1: 0.799679
- ws25_bar9_ClosePctChange1: 0.790826
- ws25_bar24_ClosePctChange1: 0.781027
- ws25_bar6_ClosePctChange1: 0.706436
- ws25_bar11_ClosePctChange1: 0.686990
- ws25_bar7_ClosePctChange1: 0.682447

Top 10 features (GradientBoosting importance):

### Horizon 4h

#### Window size 5

- Total samples: 28505
- Train samples: 21850, Test samples: 6655
- Label distribution (train): {np.int8(1): 5004, np.int8(0): 12215, np.int8(-1): 4631}
- Label distribution (test):  {np.int8(0): 4216, np.int8(1): 1238, np.int8(-1): 1201}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.6335 | 0.4914 | 0.00 |
| Random | 0.4368 | 0.4520 | 0.00 |
| LogisticRegression | 0.6362 | 0.5123 | 2.30 |
| GradientBoosting | 0.6338 | 0.5152 | 0.46 |

Top 10 features (LogisticRegression mean |coef|):
- ws5_bar4_Ema12Dist: 1.474635
- ws5_bar4_ClosePctChange1: 0.947859
- ws5_bar3_ClosePctChange1: 0.783234
- ws5_bar0_Ema12Dist: 0.746453
- ws5_bar2_ClosePctChange1: 0.713295
- ws5_bar3_Ema26Dist: 0.518402
- ws5_bar1_ClosePctChange1: 0.419910
- ws5_bar0_Ema26Dist: 0.382322
- ws5_bar2_MacdHistogramNorm: 0.333239
- ws5_bar3_Ema12Dist: 0.307169

Top 10 features (GradientBoosting importance):

#### Window size 10

- Total samples: 28467
- Train samples: 21812, Test samples: 6655
- Label distribution (train): {np.int8(-1): 4559, np.int8(1): 4970, np.int8(0): 12283}
- Label distribution (test):  {np.int8(-1): 1199, np.int8(0): 4254, np.int8(1): 1202}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.6392 | 0.4985 | 0.00 |
| Random | 0.4437 | 0.4591 | 0.00 |
| LogisticRegression | 0.6386 | 0.5140 | 3.28 |
| GradientBoosting | 0.6373 | 0.5185 | 0.75 |

Top 10 features (LogisticRegression mean |coef|):
- ws10_bar4_ClosePctChange1: 0.879670
- ws10_bar5_ClosePctChange1: 0.778469
- ws10_bar2_ClosePctChange1: 0.692075
- ws10_bar6_ClosePctChange1: 0.638180
- ws10_bar3_ClosePctChange1: 0.628366
- ws10_bar9_Ema12Dist: 0.544621
- ws10_bar7_ClosePctChange1: 0.505849
- ws10_bar8_ClosePctChange1: 0.486436
- ws10_bar9_ClosePctChange1: 0.482677
- ws10_bar1_Ema12Dist: 0.478245

Top 10 features (GradientBoosting importance):

#### Window size 15

- Total samples: 28425
- Train samples: 21770, Test samples: 6655
- Label distribution (train): {np.int8(0): 12181, np.int8(-1): 4612, np.int8(1): 4977}
- Label distribution (test):  {np.int8(0): 4216, np.int8(1): 1238, np.int8(-1): 1201}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.6335 | 0.4914 | 0.00 |
| Random | 0.4367 | 0.4518 | 0.00 |
| LogisticRegression | 0.6353 | 0.5112 | 4.76 |
| GradientBoosting | 0.6341 | 0.5174 | 1.06 |

Top 10 features (LogisticRegression mean |coef|):
- ws15_bar14_Ema12Dist: 0.882646
- ws15_bar14_ClosePctChange1: 0.600347
- ws15_bar3_ClosePctChange1: 0.592291
- ws15_bar12_ClosePctChange1: 0.547736
- ws15_bar13_ClosePctChange1: 0.538308
- ws15_bar11_ClosePctChange1: 0.499847
- ws15_bar9_ClosePctChange1: 0.461626
- ws15_bar4_ClosePctChange1: 0.458850
- ws15_bar2_ClosePctChange1: 0.453678
- ws15_bar10_ClosePctChange1: 0.431154

Top 10 features (GradientBoosting importance):

#### Window size 20

- Total samples: 28387
- Train samples: 21732, Test samples: 6655
- Label distribution (train): {np.int8(0): 12246, np.int8(1): 4939, np.int8(-1): 4547}
- Label distribution (test):  {np.int8(-1): 1199, np.int8(0): 4254, np.int8(1): 1202}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.6392 | 0.4985 | 0.00 |
| Random | 0.4440 | 0.4594 | 0.00 |
| LogisticRegression | 0.6397 | 0.5194 | 6.86 |
| GradientBoosting | 0.6370 | 0.5194 | 1.38 |

Top 10 features (LogisticRegression mean |coef|):
- ws20_bar0_Ema26Dist: 0.907882
- ws20_bar8_ClosePctChange1: 0.793015
- ws20_bar6_ClosePctChange1: 0.735984
- ws20_bar7_ClosePctChange1: 0.730378
- ws20_bar4_ClosePctChange1: 0.697602
- ws20_bar3_ClosePctChange1: 0.682763
- ws20_bar9_ClosePctChange1: 0.679610
- ws20_bar12_ClosePctChange1: 0.582057
- ws20_bar1_ClosePctChange1: 0.580568
- ws20_bar5_ClosePctChange1: 0.529436

Top 10 features (GradientBoosting importance):

#### Window size 25

- Total samples: 28345
- Train samples: 21690, Test samples: 6655
- Label distribution (train): {np.int8(0): 12140, np.int8(-1): 4597, np.int8(1): 4953}
- Label distribution (test):  {np.int8(0): 4216, np.int8(1): 1238, np.int8(-1): 1201}

| Model | Accuracy | F1-weighted | Fit time (s) |
|-------|----------|-------------|--------------|
| MajorityClass | 0.6335 | 0.4914 | 0.00 |
| Random | 0.4370 | 0.4521 | 0.00 |
| LogisticRegression | 0.6328 | 0.5128 | 8.15 |
| GradientBoosting | 0.6313 | 0.5104 | 2.73 |

Top 10 features (LogisticRegression mean |coef|):
- ws25_bar10_ClosePctChange1: 0.766385
- ws25_bar24_Ema12Dist: 0.721939
- ws25_bar5_ClosePctChange1: 0.720834
- ws25_bar13_ClosePctChange1: 0.717027
- ws25_bar6_ClosePctChange1: 0.696165
- ws25_bar11_ClosePctChange1: 0.689656
- ws25_bar4_ClosePctChange1: 0.689379
- ws25_bar12_ClosePctChange1: 0.685468
- ws25_bar9_ClosePctChange1: 0.684029
- ws25_bar7_ClosePctChange1: 0.631446

Top 10 features (GradientBoosting importance):

## Interpretation

- If LogisticRegression / GradientBoosting consistently beat random (0.333) and majority-class, the dataset has a real predictive signal.
- If models are close to majority-class, the features are not informative beyond class imbalance.
- If models are close to random, labels may be noisy or features may not encode direction well.
