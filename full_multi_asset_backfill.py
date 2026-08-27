import os
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values
from db_config import get_db_connection
from verify_indicators import compute_indicators

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

INTERVAL_MS = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000
}

# Target configurations
CONFIGS = [
    {
        "symbol": "SOLUSDT",
        "start_ms": 1597125600000, # 2020-08-11 06:00:00 UTC (Listing date)
        "timeframes": ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]
    },
    {
        "symbol": "ETHUSDT",
        "start_ms": 1577836800000, # 2020-01-01 00:00:00 UTC
        "timeframes": ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]
    }
]

def get_http_session():
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3)
    session.mount('https://', adapter)
    return session

def fetch_kline_chunk(session, symbol, timeframe, start_ms, limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&startTime={start_ms}&limit={limit}"
    for attempt in range(6):
        try:
            resp = session.get(url, timeout=12)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "5"))
                print(f" [Rate-Limit 429] Sleeping {retry_after}s...")
                time.sleep(retry_after)
                continue
            if resp.status_code == 200:
                return resp.json()
            time.sleep(0.3 * (attempt + 1))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return []

def backfill_klines(conn, symbol, timeframe, target_start_ms, now_ms):
    intv_ms = INTERVAL_MS[timeframe]
    chunk_span_ms = 1000 * intv_ms

    start_times = list(range(target_start_ms, now_ms, chunk_span_ms))
    total_chunks = len(start_times)
    
    print(f"\n--- [{symbol} - {timeframe}] Backfilling {total_chunks} chunks ({target_start_ms} -> {now_ms}) ---")
    t0 = time.time()
    
    session = get_http_session()
    all_candles = []
    completed_chunks = 0
    
    max_workers = 8 if timeframe in ['1m', '5m'] else 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(fetch_kline_chunk, session, symbol, timeframe, st): st for st in start_times}
        for future in as_completed(future_map):
            res = future.result()
            if res:
                all_candles.extend(res)
            completed_chunks += 1
            if completed_chunks % 500 == 0 or completed_chunks == total_chunks:
                pct = (completed_chunks / total_chunks) * 100
                print(f"  > Fetching {symbol} {timeframe}: {completed_chunks}/{total_chunks} chunks ({pct:.1f}%) - {len(all_candles):,} candles fetched")

    t1 = time.time()
    print(f"  > Fetch completed in {t1 - t0:.2f}s. Deduplicating & sorting {len(all_candles):,} candles...")

    # Deduplicate & sort
    candles_by_time = {}
    for c in all_candles:
        candles_by_time[int(c[0])] = c
    sorted_candles = [candles_by_time[k] for k in sorted(candles_by_time.keys())]

    print(f"  > Unique candles to insert/sync: {len(sorted_candles):,}")

    # Prepare records for Klines
    cur = conn.cursor()
    insert_klines_sql = """
        INSERT INTO "Klines" (
            "Symbol", "Timeframe", "OpenTimeMs", "CloseTimeMs",
            "Open", "High", "Low", "Close", "Volume",
            "QuoteVolume", "TradeCount", "TakerBuyVolume", "TakerBuyQuoteVolume"
        ) VALUES %s
        ON CONFLICT ("Symbol", "Timeframe", "OpenTimeMs") DO NOTHING;
    """
    
    # Stream insert in batches of 10,000
    batch_size = 10000
    records = []
    inserted_count = 0
    t_insert_start = time.time()
    
    for c in sorted_candles:
        records.append((
            symbol, timeframe, int(c[0]), int(c[6]),
            float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]),
            float(c[7]), int(c[8]), float(c[9]), float(c[10])
        ))
        if len(records) >= batch_size:
            execute_values(cur, insert_klines_sql, records, page_size=5000)
            conn.commit()
            inserted_count += len(records)
            records.clear()

    if records:
        execute_values(cur, insert_klines_sql, records, page_size=5000)
        conn.commit()
        inserted_count += len(records)
        records.clear()

    cur.close()
    t_insert_end = time.time()
    print(f"  > DB Klines sync complete in {t_insert_end - t_insert_start:.2f}s ({inserted_count:,} candles processed).")

def compute_and_index_indicators(conn, symbol, timeframe):
    print(f"\n--- [{symbol} - {timeframe}] Computing & Indexing Technical Indicators ---")
    t0 = time.time()
    cur = conn.cursor()

    # 1. Load all klines from DB
    cur.execute(f"""
        SELECT "OpenTimeMs", "Open", "High", "Low", "Close", "Volume"
        FROM "Klines"
        WHERE "Symbol" = %s AND "Timeframe" = %s
        ORDER BY "OpenTimeMs" ASC;
    """, (symbol, timeframe))
    rows = cur.fetchall()
    
    if not rows or len(rows) < 50:
        print(f"  > Not enough klines for {symbol} {timeframe} ({len(rows)} bars). Skipping.")
        cur.close()
        return

    n_bars = len(rows)
    print(f"  > Loaded {n_bars:,} bars from DB in {time.time() - t0:.2f}s. Vectorizing indicators...")

    df = pd.DataFrame(rows, columns=['OpenTimeMs', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)

    t_calc_start = time.time()
    ti_dict = compute_indicators(df)
    t_calc_end = time.time()
    print(f"  > Indicators computed in {t_calc_end - t_calc_start:.2f}s.")

    # 2. Check existing TechnicalIndicators to avoid re-inserting existing
    cur.execute(f"""
        SELECT "OpenTimeMs" FROM "TechnicalIndicators"
        WHERE "Symbol" = %s AND "Timeframe" = %s;
    """, (symbol, timeframe))
    existing_times = set(r[0] for r in cur.fetchall())
    print(f"  > Found {len(existing_times):,} existing indicator rows in DB.")

    def to_float_or_none(val):
        if val is None or np.isnan(val):
            return None
        return float(val)

    # 3. Batch insert missing rows
    insert_ti_sql = """
        INSERT INTO "TechnicalIndicators" (
            "Symbol", "Timeframe", "OpenTimeMs",
            "Rsi14", "Ema12", "Ema26", "Ema50", "Ema200",
            "Sma50", "Sma200",
            "Macd", "MacdSignal", "MacdHistogram",
            "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm",
            "BollingerUpper", "BollingerMiddle", "BollingerLower",
            "Atr14", "Obv", "ObvEma50", "Vwap", "RollingVwap24"
        ) VALUES %s
        ON CONFLICT ("Symbol", "Timeframe", "OpenTimeMs") DO NOTHING;
    """

    t_ins_start = time.time()
    batch_size = 10000
    ti_records = []
    total_added = 0

    for i in range(n_bars):
        ot_ms = int(df['OpenTimeMs'].iloc[i])
        if ot_ms in existing_times:
            continue
            
        ti_records.append((
            symbol, timeframe, ot_ms,
            to_float_or_none(ti_dict['Rsi14'][i]),
            to_float_or_none(ti_dict['Ema12'][i]),
            to_float_or_none(ti_dict['Ema26'][i]),
            to_float_or_none(ti_dict['Ema50'][i]),
            to_float_or_none(ti_dict['Ema200'][i]),
            to_float_or_none(ti_dict['Sma50'][i]),
            to_float_or_none(ti_dict['Sma200'][i]),
            to_float_or_none(ti_dict['Macd'][i]),
            to_float_or_none(ti_dict['MacdSignal'][i]),
            to_float_or_none(ti_dict['MacdHistogram'][i]),
            to_float_or_none(ti_dict['MacdNorm'][i]),
            to_float_or_none(ti_dict['MacdSignalNorm'][i]),
            to_float_or_none(ti_dict['MacdHistogramNorm'][i]),
            to_float_or_none(ti_dict['BollingerUpper'][i]),
            to_float_or_none(ti_dict['BollingerMiddle'][i]),
            to_float_or_none(ti_dict['BollingerLower'][i]),
            to_float_or_none(ti_dict['Atr14'][i]),
            to_float_or_none(ti_dict['Obv'][i]),
            to_float_or_none(ti_dict['ObvEma50'][i]),
            to_float_or_none(ti_dict['Vwap'][i]),
            to_float_or_none(ti_dict['RollingVwap24'][i])
        ))

        if len(ti_records) >= batch_size:
            execute_values(cur, insert_ti_sql, ti_records, page_size=5000)
            conn.commit()
            total_added += len(ti_records)
            ti_records.clear()
            print(f"  > Inserted {total_added:,} indicator rows...")

    if ti_records:
        execute_values(cur, insert_ti_sql, ti_records, page_size=5000)
        conn.commit()
        total_added += len(ti_records)
        ti_records.clear()

    cur.close()
    t_ins_end = time.time()
    print(f"  > Indexing complete in {t_ins_end - t_ins_start:.2f}s ({total_added:,} rows newly added).")

def main():
    total_start = time.time()
    now_ms = int(time.time() * 1000)
    print("=" * 100)
    print("STARTING FULL MULTI-ASSET & MULTI-TIMEFRAME DATA BACKFILL & INDEXING")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 100)

    conn = get_db_connection()

    for cfg in CONFIGS:
        symbol = cfg["symbol"]
        start_ms = cfg["start_ms"]
        timeframes = cfg["timeframes"]

        print(f"\n==================== PROCESSING SYMBOL: {symbol} ====================")
        for tf in timeframes:
            # 1. Backfill Klines
            backfill_klines(conn, symbol, tf, start_ms, now_ms)
            # 2. Compute and Index Technical Indicators
            compute_and_index_indicators(conn, symbol, tf)

    conn.close()

    total_end = time.time()
    print("\n" + "=" * 100)
    print(f"ALL INGESTION & INDEXING TASKS COMPLETED in {(total_end - total_start) / 60:.2f} minutes!")
    print("=" * 100)

if __name__ == "__main__":
    main()
