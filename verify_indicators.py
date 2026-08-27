import numpy as np
import pandas as pd
from db_config import get_db_connection

def compute_indicators(df):
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    opens = df['Open'].values
    vols = df['Volume'].values
    open_times = df['OpenTimeMs'].values
    n = len(closes)

    # 1. EMA
    def calc_ema(arr, period):
        res = np.full(n, np.nan)
        if n < period:
            return res
        mult = 2.0 / (period + 1)
        sma = np.mean(arr[:period])
        res[period - 1] = sma
        for i in range(period, n):
            res[i] = (arr[i] - res[i - 1]) * mult + res[i - 1]
        return res

    # 2. SMA
    def calc_sma(arr, period):
        res = np.full(n, np.nan)
        if n < period:
            return res
        kernel = np.ones(period) / period
        res[period - 1:] = np.convolve(arr, kernel, mode='valid')
        return res

    # 3. RSI
    def calc_rsi(arr, period=14):
        res = np.full(n, np.nan)
        if n <= period:
            return res
        deltas = np.diff(arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # at index period (bar 14, 0-indexed)
        avg_gain = (avg_gain + gains[period - 1]) / period
        avg_loss = (avg_loss + losses[period - 1]) / period
        if avg_loss == 0:
            res[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            res[period] = 100.0 - 100.0 / (1.0 + rs)
            
        for i in range(period + 1, n):
            gain = gains[i - 1]
            loss = losses[i - 1]
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            if avg_loss == 0:
                res[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                res[i] = 100.0 - 100.0 / (1.0 + rs)
        return res

    # 4. Bollinger Bands (period 20, 2.0 std)
    def calc_bb(arr, period=20, std_mult=2.0):
        upper = np.full(n, np.nan)
        middle = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        if n < period:
            return upper, middle, lower
        for i in range(period - 1, n):
            window = arr[i - period + 1 : i + 1]
            m = np.mean(window)
            s = np.std(window, ddof=0)
            middle[i] = m
            upper[i] = m + std_mult * s
            lower[i] = m - std_mult * s
        return upper, middle, lower

    # 5. ATR 14
    def calc_atr(highs, lows, closes, opens, period=14):
        res = np.full(n, np.nan)
        if n <= period:
            return res
        trs = np.zeros(n)
        trs[0] = highs[0] - lows[0]
        for i in range(1, n):
            prev_c = closes[i - 1]
            trs[i] = max(highs[i] - lows[i], abs(highs[i] - prev_c), abs(lows[i] - prev_c))
        
        # C# lines 428-442:
        # at i == period (14): atr = average(trs[0..15]) (period + 1 elements)
        res[period] = np.mean(trs[:period + 1])
        for i in range(period + 1, n):
            res[i] = (res[i - 1] * (period - 1) + trs[i]) / period
        return res

    # 6. OBV and OBV EMA50
    def calc_obv(closes, vols):
        res = np.zeros(n)
        res[0] = vols[0]
        for i in range(1, n):
            if closes[i] > closes[i - 1]:
                res[i] = res[i - 1] + vols[i]
            elif closes[i] < closes[i - 1]:
                res[i] = res[i - 1] - vols[i]
            else:
                res[i] = res[i - 1]
        return res

    def calc_ema_of_series(arr, period):
        res = np.full(n, np.nan)
        valid_idx = np.where(~np.isnan(arr))[0]
        if len(valid_idx) < period:
            return res
        mult = 2.0 / (period + 1)
        seed_idx = valid_idx[:period]
        sma = np.mean(arr[seed_idx])
        start_i = seed_idx[-1]
        res[start_i] = sma
        for i in range(start_i + 1, n):
            if not np.isnan(arr[i]):
                res[i] = (arr[i] - res[i - 1]) * mult + res[i - 1]
            else:
                res[i] = res[i - 1]
        return res

    # 7. VWAP & RollingVWAP24
    def calc_vwap(highs, lows, closes, vols, open_times):
        vwap = np.full(n, np.nan)
        rolling_vwap = np.full(n, np.nan)
        tps = (highs + lows + closes) / 3.0
        tp_vols = tps * vols

        # Daily vwap
        cur_day = -1
        cum_tp_vol = 0.0
        cum_vol = 0.0
        for i in range(n):
            day = open_times[i] // 86400000
            if day != cur_day:
                cur_day = day
                cum_tp_vol = 0.0
                cum_vol = 0.0
            cum_tp_vol += tp_vols[i]
            cum_vol += vols[i]
            if cum_vol > 0:
                vwap[i] = cum_tp_vol / cum_vol

        # Rolling 24
        period = 24
        rolling_tp_vols = np.zeros(period)
        rolling_vols = np.zeros(period)
        c_tp_vol = 0.0
        c_vol = 0.0
        head = 0
        count = 0
        for i in range(n):
            if count >= period:
                c_tp_vol -= rolling_tp_vols[head]
                c_vol -= rolling_vols[head]
            else:
                count += 1
            rolling_tp_vols[head] = tp_vols[i]
            rolling_vols[head] = vols[i]
            c_tp_vol += tp_vols[i]
            c_vol += vols[i]
            head = (head + 1) % period
            if c_vol > 0:
                rolling_vwap[i] = c_tp_vol / c_vol

        return vwap, rolling_vwap

    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    ema50 = calc_ema(closes, 50)
    ema200 = calc_ema(closes, 200)
    sma50 = calc_sma(closes, 50)
    sma200 = calc_sma(closes, 200)
    rsi14 = calc_rsi(closes, 14)
    bb_u, bb_m, bb_l = calc_bb(closes, 20, 2.0)
    atr14 = calc_atr(highs, lows, closes, opens, 14)
    obv = calc_obv(closes, vols)
    obv_ema50 = calc_ema_of_series(obv, 50)

    # MACD
    macd = np.where(~np.isnan(ema12) & ~np.isnan(ema26), ema12 - ema26, np.nan)
    macd_signal = calc_ema_of_series(macd, 9)
    macd_hist = np.where(~np.isnan(macd) & ~np.isnan(macd_signal), macd - macd_signal, np.nan)
    
    macd_norm = np.where(~np.isnan(macd) & ~np.isnan(atr14) & (atr14 > 0), macd / atr14, np.nan)
    macd_sig_norm = np.where(~np.isnan(macd_signal) & ~np.isnan(atr14) & (atr14 > 0), macd_signal / atr14, np.nan)
    macd_hist_norm = np.where(~np.isnan(macd_hist) & ~np.isnan(atr14) & (atr14 > 0), macd_hist / atr14, np.nan)

    vwap, rolling_vwap = calc_vwap(highs, lows, closes, vols, open_times)

    return {
        'Rsi14': rsi14,
        'Ema12': ema12, 'Ema26': ema26, 'Ema50': ema50, 'Ema200': ema200,
        'Sma50': sma50, 'Sma200': sma200,
        'Macd': macd, 'MacdSignal': macd_signal, 'MacdHistogram': macd_hist,
        'MacdNorm': macd_norm, 'MacdSignalNorm': macd_sig_norm, 'MacdHistogramNorm': macd_hist_norm,
        'BollingerUpper': bb_u, 'BollingerMiddle': bb_m, 'BollingerLower': bb_l,
        'Atr14': atr14, 'Obv': obv, 'ObvEma50': obv_ema50,
        'Vwap': vwap, 'RollingVwap24': rolling_vwap
    }

def verify():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT "OpenTimeMs", "Open", "High", "Low", "Close", "Volume"
        FROM "Klines"
        WHERE "Symbol" = 'BTCUSDT' AND "Timeframe" = '1h'
        ORDER BY "OpenTimeMs" ASC
        LIMIT 500;
    """)
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=['OpenTimeMs', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)

    cur.execute("""
        SELECT "OpenTimeMs", "Rsi14", "Ema12", "Ema26", "Ema50", "Ema200", 
               "Sma50", "Sma200", "Macd", "MacdSignal", "MacdHistogram", 
               "BollingerUpper", "BollingerMiddle", "BollingerLower", 
               "Atr14", "Obv", "ObvEma50", "Vwap", "RollingVwap24",
               "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm"
        FROM "TechnicalIndicators"
        WHERE "Symbol" = 'BTCUSDT' AND "Timeframe" = '1h'
        ORDER BY "OpenTimeMs" ASC
        LIMIT 500;
    """)
    ti_rows = cur.fetchall()
    ti_cols = ["OpenTimeMs", "Rsi14", "Ema12", "Ema26", "Ema50", "Ema200", 
               "Sma50", "Sma200", "Macd", "MacdSignal", "MacdHistogram", 
               "BollingerUpper", "BollingerMiddle", "BollingerLower", 
               "Atr14", "Obv", "ObvEma50", "Vwap", "RollingVwap24",
               "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm"]
    db_ti = pd.DataFrame(ti_rows, columns=ti_cols)

    py_ti = compute_indicators(df)

    # Compare row 250
    print("Comparing Row 250:")
    for k in py_ti.keys():
        py_v = py_ti[k][250]
        db_v = float(db_ti[k].iloc[250]) if db_ti[k].iloc[250] is not None else np.nan
        diff = abs(py_v - db_v) if not (np.isnan(py_v) and np.isnan(db_v)) else 0.0
        print(f"  {k:<18}: Py={py_v:15.6f} | DB={db_v:15.6f} | Diff={diff:10.2e}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    verify()
