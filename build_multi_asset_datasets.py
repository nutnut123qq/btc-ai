#!/usr/bin/env python3
"""
Build ML Feature Stores, Price Targets, and Window Classification Datasets
for multi-asset, multi-timeframe ML model suite.
"""

import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from db_config import get_db_params

INTERVAL_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440
}

HORIZON_MINUTES = {
    "1h": 60, "4h": 240, "1d": 1440, "3d": 4320, "7d": 10080
}

DIRECTION_THRESHOLDS = {
    "1h": 0.3, "4h": 0.6, "1d": 1.2, "3d": 2.0, "7d": 3.0
}


def rolling_zscore(series, window=20):
    s = pd.Series(series)
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std(ddof=1)
    std = std.replace(0, np.nan)
    return (s - mean) / std


def rolling_sma_ratio(series, window=20):
    s = pd.Series(series)
    sma = s.rolling(window, min_periods=window).mean()
    return s / sma


def build_features_and_targets_for_timeframe(conn, symbol, tf):
    print(f"[{symbol} {tf}] Loading Klines and TechnicalIndicators...")
    cur = conn.cursor()
    cur.execute('''
        SELECT k."OpenTimeMs", k."Open", k."High", k."Low", k."Close", k."Volume", k."TakerBuyVolume",
               ti."Rsi14", ti."MacdNorm", ti."MacdSignalNorm", ti."MacdHistogramNorm",
               ti."Ema12", ti."Ema26", ti."Ema50", ti."Ema200",
               ti."Sma50", ti."Sma200", ti."BollingerUpper", ti."BollingerLower", ti."BollingerMiddle",
               ti."Atr14", ti."Vwap", ti."RollingVwap24", ti."Obv", ti."ObvEma50"
        FROM "Klines" k
        LEFT JOIN "TechnicalIndicators" ti
          ON k."Symbol"=ti."Symbol" AND k."Timeframe"=ti."Timeframe" AND k."OpenTimeMs"=ti."OpenTimeMs"
        WHERE k."Symbol"=%s AND k."Timeframe"=%s
        ORDER BY k."OpenTimeMs"
    ''', (symbol, tf))
    rows = cur.fetchall()
    if len(rows) < 50:
        print(f"[{symbol} {tf}] Not enough rows ({len(rows)}), skipping.")
        return

    df = pd.DataFrame(rows, columns=[
        "OpenTimeMs", "Open", "High", "Low", "Close", "Volume", "TakerBuyVolume",
        "Rsi14", "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm",
        "Ema12", "Ema26", "Ema50", "Ema200",
        "Sma50", "Sma200", "BollingerUpper", "BollingerLower", "BollingerMiddle",
        "Atr14", "Vwap", "RollingVwap24", "Obv", "ObvEma50"
    ])

    for col in df.columns:
        if col == "OpenTimeMs":
            df[col] = df[col].astype(np.int64)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    open_p = df["Open"].values
    vol = df["Volume"].values
    taker_vol = df["TakerBuyVolume"].values
    n = len(df)

    tf_min = INTERVAL_MINUTES.get(tf, 60)
    day_bars = max(1, 1440 // tf_min)

    # Rolling calculations
    close_z = rolling_zscore(close, 20).values
    vol_z = rolling_zscore(vol, 20).values
    vol_sma_ratio = rolling_sma_ratio(vol, 20).values

    # Price returns
    ret1 = np.full(n, np.nan)
    ret1[1:] = (close[1:] - close[:-1]) / close[:-1] * 100.0

    ret4 = np.full(n, np.nan)
    if n > 4:
        ret4[4:] = (close[4:] - close[:-4]) / close[:-4] * 100.0

    ret24 = np.full(n, np.nan)
    if n > day_bars:
        ret24[day_bars:] = (close[day_bars:] - close[:-day_bars]) / close[:-day_bars] * 100.0

    hl_range = high - low
    hl_range_pct = np.where(close > 0, hl_range / close * 100.0, np.nan)
    body = np.abs(close - open_p)
    body_pct = np.where(hl_range > 0, body / hl_range, np.nan)
    upper_wick_pct = np.where(hl_range > 0, (high - np.maximum(open_p, close)) / hl_range, np.nan)
    lower_wick_pct = np.where(hl_range > 0, (np.minimum(open_p, close) - low) / hl_range, np.nan)
    taker_ratio = np.where(vol > 0, taker_vol / vol, np.nan)

    # Indicator features
    rsi = df["Rsi14"].values
    rsi_s = pd.Series(rsi)
    rsi_slope = (rsi_s - rsi_s.shift(5)) / 5.0
    rsi_slope = rsi_slope.values

    macd_norm = df["MacdNorm"].values
    macd_sig = df["MacdSignalNorm"].values
    macd_hist = df["MacdHistogramNorm"].values

    ema12_d = (close - df["Ema12"].values) / df["Ema12"].values * 100.0
    ema26_d = (close - df["Ema26"].values) / df["Ema26"].values * 100.0
    ema50_d = (close - df["Ema50"].values) / df["Ema50"].values * 100.0
    ema200_d = (close - df["Ema200"].values) / df["Ema200"].values * 100.0
    sma50_d = (close - df["Sma50"].values) / df["Sma50"].values * 100.0
    sma200_d = (close - df["Sma200"].values) / df["Sma200"].values * 100.0

    bw = (df["BollingerUpper"].values - df["BollingerLower"].values)
    boll_w = np.where(df["BollingerMiddle"].values != 0, bw / df["BollingerMiddle"].values * 100.0, np.nan)
    boll_pos = np.where(bw > 0, (close - df["BollingerLower"].values) / bw, np.nan)

    atr_pct = np.where(close > 0, df["Atr14"].values / close * 100.0, np.nan)
    obv_ema_d = np.where((df["ObvEma50"].values != 0) & (~np.isnan(df["ObvEma50"].values)), (df["Obv"].values - df["ObvEma50"].values) / np.abs(df["ObvEma50"].values) * 100.0, np.nan)
    vwap_d = (close - df["Vwap"].values) / df["Vwap"].values * 100.0
    rvwap_d = (close - df["RollingVwap24"].values) / df["RollingVwap24"].values * 100.0

    # Pattern context
    cur.execute('SELECT "OpenTimeMs", "PatternType" FROM "CandlePatterns" WHERE "Symbol"=%s AND "Timeframe"=%s ORDER BY "OpenTimeMs"', (symbol, tf))
    pats = cur.fetchall()
    pat_dict = {int(p[0]): 1 for p in pats}

    cur.execute('SELECT COUNT(*) FROM "CandleSequenceRules" WHERE "Symbol"=%s AND "Timeframe"=%s AND "IsEnabled"=true', (symbol, tf))
    active_rules = cur.fetchone()[0]

    # Time features
    times_ms = df["OpenTimeMs"].values
    dts = [datetime.fromtimestamp(t / 1000.0, tz=timezone.utc) for t in times_ms]
    hours = np.array([dt.hour for dt in dts])
    dows = np.array([dt.weekday() for dt in dts]) # 0 = Monday, 6 = Sunday

    hour_sin = np.sin(2.0 * math.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * math.pi * hours / 24.0)
    dow_sin = np.sin(2.0 * math.pi * dows / 7.0)
    dow_cos = np.cos(2.0 * math.pi * dows / 7.0)
    is_weekend = np.where((dows == 5) | (dows == 6), 1.0, 0.0)

    # Insert into MlFeatureStores
    print(f"[{symbol} {tf}] Inserting MlFeatureStores ({n} bars)...")
    cur.execute('DELETE FROM "MlFeatureStores" WHERE "Symbol"=%s AND "Timeframe"=%s', (symbol, tf))
    
    now_utc = datetime.now(timezone.utc)
    feature_tuples = []
    for i in range(n):
        t_ms = int(times_ms[i])
        pat_code = pat_dict.get(t_ms, 0)
        
        row_vals = [
            float(close_z[i]) if not np.isnan(close_z[i]) else None,
            float(ret1[i]) if not np.isnan(ret1[i]) else None,
            float(ret4[i]) if not np.isnan(ret4[i]) else None,
            float(ret24[i]) if not np.isnan(ret24[i]) else None,
            float(hl_range_pct[i]) if not np.isnan(hl_range_pct[i]) else None,
            float(body_pct[i]) if not np.isnan(body_pct[i]) else None,
            float(upper_wick_pct[i]) if not np.isnan(upper_wick_pct[i]) else None,
            float(lower_wick_pct[i]) if not np.isnan(lower_wick_pct[i]) else None,
            float(rsi[i]) if not np.isnan(rsi[i]) else None,
            float(rsi_slope[i]) if not np.isnan(rsi_slope[i]) else None,
            float(macd_norm[i]) if not np.isnan(macd_norm[i]) else None,
            float(macd_sig[i]) if not np.isnan(macd_sig[i]) else None,
            float(macd_hist[i]) if not np.isnan(macd_hist[i]) else None,
            float(ema12_d[i]) if not np.isnan(ema12_d[i]) else None,
            float(ema26_d[i]) if not np.isnan(ema26_d[i]) else None,
            float(ema50_d[i]) if not np.isnan(ema50_d[i]) else None,
            float(ema200_d[i]) if not np.isnan(ema200_d[i]) else None,
            float(sma50_d[i]) if not np.isnan(sma50_d[i]) else None,
            float(sma200_d[i]) if not np.isnan(sma200_d[i]) else None,
            float(boll_w[i]) if not np.isnan(boll_w[i]) else None,
            float(boll_pos[i]) if not np.isnan(boll_pos[i]) else None,
            float(atr_pct[i]) if not np.isnan(atr_pct[i]) else None,
            float(obv_ema_d[i]) if not np.isnan(obv_ema_d[i]) else None,
            float(vwap_d[i]) if not np.isnan(vwap_d[i]) else None,
            float(rvwap_d[i]) if not np.isnan(rvwap_d[i]) else None,
            float(vol_z[i]) if not np.isnan(vol_z[i]) else None,
            float(vol_sma_ratio[i]) if not np.isnan(vol_sma_ratio[i]) else None,
            float(taker_ratio[i]) if not np.isnan(taker_ratio[i]) else None,
            pat_code,
            active_rules,
            0.0 # null ratio dummy
        ]
        null_count = sum(1 for v in row_vals[:-1] if v is None)
        null_ratio = null_count / (len(row_vals) - 1)
        row_vals[-1] = float(null_ratio)
        
        feature_tuples.append((
            symbol, tf, t_ms,
            row_vals[0], row_vals[1], row_vals[2], row_vals[3], row_vals[4],
            row_vals[5], row_vals[6], row_vals[7], row_vals[8], row_vals[9],
            row_vals[10], row_vals[11], row_vals[12], row_vals[13], row_vals[14],
            row_vals[15], row_vals[16], row_vals[17], row_vals[18], row_vals[19],
            row_vals[20], row_vals[21], row_vals[22], row_vals[23], row_vals[24],
            row_vals[25], row_vals[26], row_vals[27], row_vals[28], row_vals[29],
            row_vals[30],
            now_utc, now_utc
        ))

    execute_values(cur, '''
        INSERT INTO "MlFeatureStores" (
            "Symbol", "Timeframe", "OpenTimeMs",
            "CloseZscore", "ClosePctChange1", "ClosePctChange4", "ClosePctChange24", "HighLowRangePct",
            "BodyPct", "UpperWickPct", "LowerWickPct", "Rsi14", "Rsi14Slope",
            "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm", "Ema12Dist", "Ema26Dist",
            "Ema50Dist", "Ema200Dist", "Sma50Dist", "Sma200Dist", "BollingerWidth",
            "BollingerPosition", "Atr14Pct", "ObvEmaDist", "VwapDist", "RollingVwapDist",
            "VolumeZscore", "VolumeSma20Ratio", "TakerBuyRatio", "RecentPatternEncoded", "ActiveRuleCount",
            "NullRatio", "CreatedAtUtc", "UpdatedAtUtc"
        ) VALUES %s
    ''', feature_tuples, page_size=2000)
    conn.commit()

    # PriceTargets calculation
    print(f"[{symbol} {tf}] Computing PriceTargets...")
    cur.execute('DELETE FROM "PriceTargets" WHERE "Symbol"=%s AND "Timeframe"=%s', (symbol, tf))
    target_tuples = []

    for i in range(n):
        t_ms = int(times_ms[i])
        c_i = close[i]
        
        # 1h
        b_1h = 60 // tf_min
        ret_1h = float((close[i + b_1h] - c_i) / c_i * 100.0) if b_1h > 0 and i + b_1h < n else None
        dir_1h = int(1 if (ret_1h is not None and ret_1h > 0.3) else (-1 if (ret_1h is not None and ret_1h < -0.3) else (0 if ret_1h is not None else -999)))
        dir_1h = dir_1h if dir_1h != -999 else None
        
        # 4h
        b_4h = 240 // tf_min
        ret_4h = float((close[i + b_4h] - c_i) / c_i * 100.0) if b_4h > 0 and i + b_4h < n else None
        dir_4h = int(1 if (ret_4h is not None and ret_4h > 0.6) else (-1 if (ret_4h is not None and ret_4h < -0.6) else (0 if ret_4h is not None else -999)))
        dir_4h = dir_4h if dir_4h != -999 else None
        
        # 1d
        b_1d = 1440 // tf_min
        ret_1d = float((close[i + b_1d] - c_i) / c_i * 100.0) if b_1d > 0 and i + b_1d < n else None
        dir_1d = int(1 if (ret_1d is not None and ret_1d > 1.2) else (-1 if (ret_1d is not None and ret_1d < -1.2) else (0 if ret_1d is not None else -999)))
        dir_1d = dir_1d if dir_1d != -999 else None
        
        target_tuples.append((
            symbol, tf, t_ms,
            ret_1h, dir_1h,
            ret_4h, dir_4h,
            ret_1d, dir_1d,
            now_utc
        ))

    execute_values(cur, '''
        INSERT INTO "PriceTargets" (
            "Symbol", "Timeframe", "OpenTimeMs",
            "TargetReturn1h", "TargetDirection1h",
            "TargetReturn4h", "TargetDirection4h",
            "TargetReturn1d", "TargetDirection1d",
            "CreatedAtUtc"
        ) VALUES %s
    ''', target_tuples, page_size=2000)
    conn.commit()

    # WindowClassificationDatasets
    print(f"[{symbol} {tf}] Building WindowClassificationDatasets (ws=5,10 x horizons)...")
    for ws in [5, 10]:
        for h in ["1h", "4h", "1d"]:
            h_min = HORIZON_MINUTES[h]
            bars = h_min // tf_min
            if bars <= 0:
                continue
            
            cur.execute('DELETE FROM "WindowClassificationDatasets" WHERE "Symbol"=%s AND "Timeframe"=%s AND "WindowSize"=%s AND "Horizon"=%s', (symbol, tf, ws, h))
            
            window_tuples = []
            for i in range(ws - 1, n):
                if h == "1h":
                    target_ret = (close[i + bars] - close[i]) / close[i] * 100.0 if i + bars < n else None
                    thr = 0.3
                elif h == "4h":
                    target_ret = (close[i + bars] - close[i]) / close[i] * 100.0 if i + bars < n else None
                    thr = 0.6
                else: # 1d
                    target_ret = (close[i + bars] - close[i]) / close[i] * 100.0 if i + bars < n else None
                    thr = 1.2
                
                if target_ret is None:
                    continue
                label = 1 if target_ret > thr else (-1 if target_ret < -thr else 0)
                
                # Build 35 * ws feature vector
                feat_vec = []
                for w_idx in range(i - ws + 1, i + 1):
                    bar_feats = [
                        close_z[w_idx] if not np.isnan(close_z[w_idx]) else 0.0,
                        ret1[w_idx] if not np.isnan(ret1[w_idx]) else 0.0,
                        ret4[w_idx] if not np.isnan(ret4[w_idx]) else 0.0,
                        ret24[w_idx] if not np.isnan(ret24[w_idx]) else 0.0,
                        hl_range_pct[w_idx] if not np.isnan(hl_range_pct[w_idx]) else 0.0,
                        body_pct[w_idx] if not np.isnan(body_pct[w_idx]) else 0.0,
                        upper_wick_pct[w_idx] if not np.isnan(upper_wick_pct[w_idx]) else 0.0,
                        lower_wick_pct[w_idx] if not np.isnan(lower_wick_pct[w_idx]) else 0.0,
                        rsi[w_idx] if not np.isnan(rsi[w_idx]) else 50.0,
                        rsi_slope[w_idx] if not np.isnan(rsi_slope[w_idx]) else 0.0,
                        macd_norm[w_idx] if not np.isnan(macd_norm[w_idx]) else 0.0,
                        macd_sig[w_idx] if not np.isnan(macd_sig[w_idx]) else 0.0,
                        macd_hist[w_idx] if not np.isnan(macd_hist[w_idx]) else 0.0,
                        ema12_d[w_idx] if not np.isnan(ema12_d[w_idx]) else 0.0,
                        ema26_d[w_idx] if not np.isnan(ema26_d[w_idx]) else 0.0,
                        ema50_d[w_idx] if not np.isnan(ema50_d[w_idx]) else 0.0,
                        ema200_d[w_idx] if not np.isnan(ema200_d[w_idx]) else 0.0,
                        sma50_d[w_idx] if not np.isnan(sma50_d[w_idx]) else 0.0,
                        sma200_d[w_idx] if not np.isnan(sma200_d[w_idx]) else 0.0,
                        boll_w[w_idx] if not np.isnan(boll_w[w_idx]) else 0.0,
                        boll_pos[w_idx] if not np.isnan(boll_pos[w_idx]) else 0.5,
                        atr_pct[w_idx] if not np.isnan(atr_pct[w_idx]) else 0.0,
                        obv_ema_d[w_idx] if not np.isnan(obv_ema_d[w_idx]) else 0.0,
                        vwap_d[w_idx] if not np.isnan(vwap_d[w_idx]) else 0.0,
                        rvwap_d[w_idx] if not np.isnan(rvwap_d[w_idx]) else 0.0,
                        vol_z[w_idx] if not np.isnan(vol_z[w_idx]) else 0.0,
                        vol_sma_ratio[w_idx] if not np.isnan(vol_sma_ratio[w_idx]) else 1.0,
                        taker_ratio[w_idx] if not np.isnan(taker_ratio[w_idx]) else 0.5,
                        pat_dict.get(int(times_ms[w_idx]), 0),
                        active_rules,
                        hour_sin[w_idx],
                        hour_cos[w_idx],
                        dow_sin[w_idx],
                        dow_cos[w_idx],
                        is_weekend[w_idx],
                    ]
                    feat_vec.extend([float(x) for x in bar_feats])
                
                window_tuples.append((
                    symbol, tf, ws, h,
                    int(times_ms[i - ws + 1]),
                    int(times_ms[i]),
                    feat_vec,
                    len(feat_vec),
                    int(label),
                    float(target_ret),
                    0.0,
                    now_utc
                ))

            execute_values(cur, '''
                INSERT INTO "WindowClassificationDatasets" (
                    "Symbol", "Timeframe", "WindowSize", "Horizon",
                    "WindowStartMs", "WindowEndMs", "FeatureVector", "FeatureDim",
                    "Label", "TargetReturn", "WindowNullRatio", "CreatedAtUtc"
                ) VALUES %s
            ''', window_tuples, page_size=2000)
            conn.commit()
            print(f"  -> Built {len(window_tuples)} windows for {symbol} {tf} ws={ws} h={h}")

    cur.close()


def main():
    conn = psycopg2.connect(**get_db_params())
    
    tasks = [
        # ETHUSDT
        ("ETHUSDT", "1d"),
        ("ETHUSDT", "4h"),
        ("ETHUSDT", "1h"),
        # SOLUSDT
        ("SOLUSDT", "1d"),
        ("SOLUSDT", "4h"),
        ("SOLUSDT", "1h"),
        # BTCUSDT
        ("BTCUSDT", "1d"),
        ("BTCUSDT", "4h"),
        ("BTCUSDT", "1h"),
    ]

    for sym, tf in tasks:
        print("=" * 70)
        build_features_and_targets_for_timeframe(conn, sym, tf)

    conn.close()
    print("\nDataset building completed for all active assets and timeframes!")

if __name__ == "__main__":
    main()
