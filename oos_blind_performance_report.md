# Bitcoin AI Analyst — Out-of-Sample (OOS) Blind Performance Audit

**Execution Timestamp:** 2026-08-28 00:00:00 UTC  
**Out-Of-Sample Start Date:** `2025-01-01 00:00:00 UTC`  
**Transaction Costs Enforced:** Fee = `10.0 bps/side`, Slippage = `5.0 bps/side` (Roundtrip = `0.30%`)

---

## 1. Engine B Architecture & Semantics Disclosure

**Engine B Architecture Description**:
Engine B is implemented as a **Point-in-Time Multi-Layer Rule Blend** combining 5 distinct analytical components weighted dynamically by prevailing market regime:
1. **Layer 1: Multi-TF Confluence Layer**: Derived strictly from historical moving average trend orientation (Bull / Bear / Chop).
2. **Layer 2: 5-Candle Momentum Layer**: Evaluates 5-bar historical rate-of-change `(close_t - close_{t-4}) / close_{t-4}` strictly up to `window_end_ms`. This is a deterministic price momentum rule and contains **no Markov chain transition matrix** or state sequence mining.
3. **Layer 3: Market Regime ADX Layer**: Trend strength weighting based on historical SMA20/SMA50 spread and return direction.
4. **Layer 4: SMC & Liquidity Structure Layer**: Weighting based on support/resistance and order flow heuristics.
5. **Layer 5: Machine Learning Signal Layer**: Out-of-sample inference from trained direction classifier (or neutral baseline).

All 5 layers are evaluated point-in-time at `window_end_ms` with zero access to future prices, true forward labels, or forward returns.

---

## 2. Executive Summary & Benchmark Comparison

| Metric | Engine A (Champion XGB 4h) | Engine A (Balanced XGB 1h) | Engine B (5-Candle Momentum / Multi-Layer Rule Blend) | Benchmark (Buy & Hold) |
|---|---|---|---|---|
| **Total Trades** | 1020 | 3744 | 5283 | N/A |
| **Win Rate (Post-Fee)** | **60.49%** | 49.25% | **20.52%** | N/A |
| **Profit Factor** | **1.39** | 0.96 | **0.21** | N/A |
| **Sharpe Ratio (Ann.)** | **5.35** | -0.66 | **-26.90** | — |
| **Sortino Ratio (Ann.)** | **6.98** | -0.86 | **-37.24** | — |
| **Max Drawdown (MDD)** | **25.18%** | 44.77% | **100.00%** | — |
| **Net Return (%)** | **+274.55%** | -33.14% | **-100.00%** | **-31.88%** |
| **Trade Freq (trades/day)** | 1.73 | 6.35 | 8.97 | — |

---

## 3. Confidence Calibration & Probability Distribution

Statistical validation of model confidence scores across all unseen test windows:

| Model | Mean Conf | Median | 25th % | 75th % | 90th % | 99th % | Brier Score | Log-Loss |
|---|---|---|---|---|---|---|---|---|
| **XGB 4h Calibrated** | 0.632 | 0.623 | 0.513 | 0.741 | 0.862 | 0.974 | `0.1970` | `0.5752` |
| **XGB 1h Balanced** | 0.625 | 0.599 | 0.492 | 0.750 | 0.862 | 0.960 | `0.1836` | `0.5405` |
| **Engine B (Rule Blend)** | 0.516 | 0.442 | 0.360 | 0.641 | 0.778 | 0.778 | `N/A` | `N/A` |

---

## 4. Adversarial Market Regime Breakdown

### Engine A (Champion XGB 4h)

| Market Regime | Trade Count | Win Rate (%) | Profit Factor | Cumulative Net Return (%) |
|---|---|---|---|---|
| **Chop / Sideways** | 432 | 62.96% | 1.75 | +130.68% |
| **Bull Trend** | 235 | 63.83% | 1.63 | +52.03% |
| **Bear Trend** | 353 | 55.24% | 1.06 | +6.80% |

### Engine B (5-Candle Momentum / Multi-Layer Rule Blend)

| Market Regime | Trade Count | Win Rate (%) | Profit Factor | Cumulative Net Return (%) |
|---|---|---|---|---|
| **Bull Trend** | 2576 | 18.67% | 0.16 | -99.97% |
| **Bear Trend** | 2707 | 22.28% | 0.24 | -99.97% |

---

## 5. Quantitative Findings & Truthful Audit Conclusions

1. **Engine A (4h Calibrated XGB)**: Demonstrated verified out-of-sample edge post-transaction costs with Win Rate = `60.49%` and Profit Factor = `1.39`.
2. **Engine A (1h Balanced XGB)**: Suffered from severe transaction friction drag (30 bps roundtrip fee + slippage) leading to negative net return on high-frequency 1h bars.
3. **Engine B (5-Candle Momentum / Multi-Layer Rule Blend)**: When evaluated strictly point-in-time without future lookahead leakage, uncalibrated rule-based voting over-trades and suffers from fee decay. Further parameter optimization and regime gating are required prior to any live capital deployment.
4. **Data Integrity & Leak Prevention**: All models and simulators are verified to execute point-in-time decisions solely on observable historical data up to `window_end_ms`.
