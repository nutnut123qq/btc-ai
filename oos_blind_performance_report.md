# Bitcoin AI Analyst — Out-of-Sample (OOS) Blind Performance Audit

**Execution Timestamp:** 2026-08-28 00:00:00 UTC
**Out-Of-Sample Start Date:** `2025-01-01 00:00:00 UTC`
**Transaction Costs Enforced:** Fee = `10.0 bps/side`, Slippage = `5.0 bps/side` (Roundtrip = `0.30%`)

---

## 1. Engine B Architecture & Semantics Disclosure

**Engine B implementation description**:
Engine B is a weighted rule blend containing five scores:

1. A primary same-timeframe regime score from the latest close, SMA20, SMA50, and 20-bar return.
2. A five-bar price-momentum score ending at `window_end_ms`.
3. A second directional weight derived from the same regime classification as score 1; it does not calculate ADX.
4. A symmetric constant selected by whether that regime is trending or sideways; it does not calculate price levels or order flow.
5. Class probabilities from the supplied model, or a regime-dependent fallback when no model is loaded.

This is not a multi-timeframe model, Markov chain, ADX calculation, market-structure model, or liquidity model. The Engine B decision seam ignores dataset labels and target returns, and regression tests verify that permuting or inverting both leaves its decisions unchanged. That test does not independently prove that every upstream feature in the stored vector was constructed point-in-time.

---

## 2. Executive Summary & Benchmark Comparison

| Metric | Engine A (Champion XGB 4h) | Engine A (Balanced XGB 1h) | Engine B (Same-Timeframe Regime / 5-Bar Momentum Rule Blend) | Benchmark (Buy & Hold) |
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

### Engine B (Same-Timeframe Regime / 5-Bar Momentum Rule Blend)

| Market Regime | Trade Count | Win Rate (%) | Profit Factor | Cumulative Net Return (%) |
|---|---|---|---|---|
| **Bull Trend** | 2576 | 18.67% | 0.16 | -99.97% |
| **Bear Trend** | 2707 | 22.28% | 0.24 | -99.97% |

---

## 5. Quantitative Findings & Truthful Audit Conclusions

1. **Engine A (4h Calibrated XGB)**: This replay reported Win Rate = `60.49%` and Profit Factor = `1.39` after its configured transaction costs.
2. **Engine A (1h Balanced XGB)**: Suffered from severe transaction friction drag (30 bps roundtrip fee + slippage) leading to negative net return on high-frequency 1h bars.
3. **Engine B (Same-Timeframe Regime / 5-Bar Momentum Rule Blend)**: This replay over-trades and suffers from fee decay. Its result does not support live capital deployment.
4. **Data-integrity scope**: Production-linked permutation tests cover `backtest_strategy.simulate_trades` and Engine B's extracted decision seam. They establish invariance to dataset labels and target returns in those paths, but do not certify every upstream dataset-building or model-training step.
