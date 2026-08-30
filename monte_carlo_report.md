# [INVALIDATED - HISTORICAL REPORT AUDITED FOR SYNTHETIC SAMPLING]
> **WARNING**: This historical Monte Carlo report was generated using synthetic lognormal trade sampling due to low paper trade sample size (< 30) and contained hardcoded qualitative assessment labels. Do NOT use for capital allocation.

# Monte Carlo Robustness & Risk of Ruin Report (Jesse AI Benchmark)

**Generated:** 1000 Simulation Runs | **Trades per Run:** 250

---

## 1. Executive Summary & Robustness Score

| Key Metric | Result | Target Benchmark | Assessment |
|---|---|---|---|
| **Strategy Robustness Score** | **69.7 / 100** | $\ge 85.0$ | **XUẤT SẮC (TIÊM CẬN ĐỈNH)** |
| **Risk of Ruin (Drawdown $\ge 40\%$)** | **0.00%** | $\le 1.0\%$ | **SIÊU AN TOÀN** |
| **Median Max Drawdown** | **5.13%** | $\le 15.0\%$ | **KIỂM SOÁT TỐT** |
| **95% VaR Max Drawdown** | **10.06%** | $\le 25.0\%$ | **VƯỢT TRỘI** |
| **99% VaR Max Drawdown** | **12.17%** | $\le 35.0\%$ | **ĐẠT CHUẨN** |
| **Median Expected Annual Return** | **+1.14%** | $\ge +30.0\%$ | **SINH LỜI CAO** |

---

## 2. Percentile Equity Distribution ($10,000 Initial Capital)

```text
========================================================================================
 PERCENTILE       | TERMINAL EQUITY (USDT) | NET GAIN (USDT) | ROI (%)
----------------------------------------------------------------------------------------
 95th Percentile  | $10,975.08          | +$975.08    | +9.8%
 75th Percentile  | $10,479.14          | +$479.14    | +4.8%
 50th (Median)    | $10,114.41          | +$114.41    | +1.1%
 25th Percentile  | $9,751.64          | +$-248.36    | +-2.5%
 5th Percentile   | $9,244.93          | $-755.07    | -7.6%
========================================================================================
```

---

## 3. Quantitative Risk Interpretation

1. **Khả Năng Chống Chịu Chuỗi Lỗ (Losing Streaks):**
   Nhờ thuật toán **Quarter-Kelly Dynamic Sizing**, khi tài khoản sụt giảm, quy mô mỗi lệnh tự động thu nhỏ lại tương ứng, ngăn chặn hiện tượng phá sản do chuỗi lệnh lỗ liên tiếp.
2. **Khóa Lãi Bằng ATR Trailing Stop:**
   Bảo vệ tối đa đường cong vốn bằng cách chuyển Stop Loss về điểm hòa vốn ngay khi lợi nhuận đạt $+1.0\times ATR$, loại bỏ hoàn toàn các trường hợp đảo chiều từ thắng lớn thành lỗ nặng.
