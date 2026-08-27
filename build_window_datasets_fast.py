import math
import sys
import time
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values
from db_config import get_db_connection

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

WINDOW_SIZES = [10, 15, 20, 25]
HORIZONS = ["1h", "4h", "1d"]
TIMEFRAMES = ["1h", "4h"]
SYMBOLS = ["ETHUSDT", "SOLUSDT"]

def interval_to_ms(tf):
    if tf == "1h":
        return 3600 * 1000
    if tf == "4h":
        return 4 * 3600 * 1000
    if tf == "1d":
        return 24 * 3600 * 1000
    return 3600 * 1000

def build_windows_for_combo(conn, symbol, timeframe, ws, horizon):
    cur = conn.cursor()
    int_ms = interval_to_ms(timeframe)

    # 1. Fetch ML Features
    cur.execute("""
        SELECT "OpenTimeMs", "CloseZscore", "ClosePctChange1", "ClosePctChange4", "ClosePctChange24",
               "HighLowRangePct", "BodyPct", "UpperWickPct", "LowerWickPct", "Rsi14", "Rsi14Slope",
               "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm", "Ema12Dist", "Ema26Dist", "Ema50Dist",
               "Ema200Dist", "Sma50Dist", "Sma200Dist", "BollingerWidth", "BollingerPosition", "Atr14Pct",
               "ObvEmaDist", "VwapDist", "RollingVwapDist", "VolumeZscore", "VolumeSma20Ratio", "TakerBuyRatio",
               "RecentPatternEncoded", "ActiveRuleCount", "NullRatio"
        FROM "MlFeatureStores"
        WHERE "Symbol" = %s AND "Timeframe" = %s
        ORDER BY "OpenTimeMs" ASC;
    """, (symbol, timeframe))
    f_rows = cur.fetchall()
    if len(f_rows) < ws + 10:
        cur.close()
        return 0

    # 2. Fetch Price Targets
    dir_col = f'"TargetDirection{horizon}"'
    ret_col = f'"TargetReturn{horizon}"'
    cur.execute(f"""
        SELECT "OpenTimeMs", {dir_col}, {ret_col}
        FROM "PriceTargets"
        WHERE "Symbol" = %s AND "Timeframe" = %s;
    """, (symbol, timeframe))
    t_rows = cur.fetchall()
    targets_map = {r[0]: (r[1], r[2]) for r in t_rows}

    # Precompute per-bar 35-feature vectors
    bar_vectors = {}
    null_ratios = {}
    for r in f_rows:
        t_ms = r[0]
        dt = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)
        hour = dt.hour
        dow = dt.weekday() + 1
        if dow == 7:
            dow = 0  # 0 = Sunday
        hour_angle = 2.0 * math.pi * hour / 24.0
        day_angle = 2.0 * math.pi * dow / 7.0
        
        # Fill missing indicator features with 0.0
        vec = [float(v) if v is not None else 0.0 for v in r[1:31]]
        vec.append(math.sin(hour_angle))
        vec.append(math.cos(hour_angle))
        vec.append(math.sin(day_angle))
        vec.append(math.cos(day_angle))
        vec.append(1.0 if dow in (0, 6) else 0.0)
        
        bar_vectors[t_ms] = vec
        null_ratios[t_ms] = float(r[31]) if r[31] is not None else 0.0

    times = [r[0] for r in f_rows]
    n = len(times)
    samples = []
    now = datetime.now(timezone.utc)

    for i in range(n - ws + 1):
        start_t = times[i]
        end_t = times[i + ws - 1]

        # Consecutive check
        if end_t - start_t != (ws - 1) * int_ms:
            continue

        # Target check
        if end_t not in targets_map:
            continue
        label, target_ret = targets_map[end_t]
        if label not in (-1, 0, 1):
            continue

        # Check all bars present
        w_times = times[i:i + ws]
        if any(t not in bar_vectors for t in w_times):
            continue

        avg_null = sum(null_ratios[t] for t in w_times) / ws
        if avg_null > 0.40:
            continue

        full_vec = []
        for t in w_times:
            full_vec.extend(bar_vectors[t])

        samples.append((
            symbol, timeframe, ws, horizon,
            start_t, end_t, full_vec, len(full_vec),
            label, target_ret, avg_null, now
        ))

    # Clean old datasets for this combo
    cur.execute("""
        DELETE FROM "WindowClassificationDatasets"
        WHERE "Symbol" = %s AND "Timeframe" = %s AND "WindowSize" = %s AND "Horizon" = %s;
    """, (symbol, timeframe, ws, horizon))

    insert_sql = """
        INSERT INTO "WindowClassificationDatasets" (
            "Symbol", "Timeframe", "WindowSize", "Horizon",
            "WindowStartMs", "WindowEndMs", "FeatureVector", "FeatureDim",
            "Label", "TargetReturn", "WindowNullRatio", "CreatedAtUtc"
        ) VALUES %s;
    """

    batch_size = 2000
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        execute_values(cur, insert_sql, chunk, page_size=2000)
        conn.commit()

    cur.close()
    print(f"  > [{symbol} - {timeframe} - ws={ws} - h={horizon}] Built {len(samples):,} window datasets.", flush=True)
    return len(samples)

def main():
    conn = get_db_connection()
    total = 0
    t0 = time.time()
    for sym in SYMBOLS:
        print(f"\nBuilding Window Classification Datasets for {sym}...")
        for tf in TIMEFRAMES:
            for ws in WINDOW_SIZES:
                for h in HORIZONS:
                    total += build_windows_for_combo(conn, sym, tf, ws, h)
    conn.close()
    print(f"\nAll Window Datasets built in {time.time() - t0:.2f}s! Total: {total:,} windows.")

if __name__ == "__main__":
    main()
