import time
import sys
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from psycopg2.extras import execute_values
from db_config import get_db_connection
from verify_indicators import compute_indicators

def test_pipeline_1d():
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Fetch 1d ETHUSDT
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
    session.mount('https://', adapter)

    start_ms = 1577836800000 # 2020-01-01
    now_ms = int(time.time() * 1000)
    chunk_span = 1000 * 24 * 3600 * 1000

    start_times = list(range(start_ms, now_ms, chunk_span))
    raw_candles = []
    for st in start_times:
        url = f"https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1d&startTime={st}&limit=1000"
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            raw_candles.extend(resp.json())

    print(f"Fetched {len(raw_candles)} raw candles.")

    # Deduplicate by OpenTimeMs
    candles_dict = {}
    for c in raw_candles:
        candles_dict[int(c[0])] = c
    sorted_candles = [candles_dict[k] for k in sorted(candles_dict.keys())]

    # Prepare Klines rows
    # [OpenTimeMs, Open, High, Low, Close, Volume, CloseTimeMs, QuoteVolume, TradeCount, TakerBuyVolume, TakerBuyQuoteVolume, Ignore]
    kline_records = []
    for c in sorted_candles:
        kline_records.append((
            "ETHUSDT", "1d", int(c[0]), int(c[6]),
            float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]),
            float(c[7]), int(c[8]), float(c[9]), float(c[10])
        ))

    insert_klines_sql = """
        INSERT INTO "Klines" (
            "Symbol", "Timeframe", "OpenTimeMs", "CloseTimeMs",
            "Open", "High", "Low", "Close", "Volume",
            "QuoteVolume", "TradeCount", "TakerBuyVolume", "TakerBuyQuoteVolume"
        ) VALUES %s
        ON CONFLICT ("Symbol", "Timeframe", "OpenTimeMs") DO NOTHING;
    """
    execute_values(cur, insert_klines_sql, kline_records, page_size=2000)
    conn.commit()
    print("Inserted/Verified Klines in DB.")

    # 2. Compute Technical Indicators
    # Load all klines for ETHUSDT 1d from DB
    cur.execute("""
        SELECT "OpenTimeMs", "Open", "High", "Low", "Close", "Volume"
        FROM "Klines"
        WHERE "Symbol" = 'ETHUSDT' AND "Timeframe" = '1d'
        ORDER BY "OpenTimeMs" ASC;
    """)
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=['OpenTimeMs', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)

    ti_dict = compute_indicators(df)
    n = len(df)
    print(f"Computed indicators for {n} bars.")

    def to_float_or_none(val):
        if val is None or np.isnan(val):
            return None
        return float(val)

    ti_records = []
    for i in range(n):
        ti_records.append((
            "ETHUSDT", "1d", int(df['OpenTimeMs'].iloc[i]),
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
    execute_values(cur, insert_ti_sql, ti_records, page_size=2000)
    conn.commit()
    print("Inserted/Verified TechnicalIndicators in DB.")

    cur.execute('SELECT COUNT(*) FROM "Klines" WHERE "Symbol" = \'ETHUSDT\' AND "Timeframe" = \'1d\'')
    k_cnt = cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM "TechnicalIndicators" WHERE "Symbol" = \'ETHUSDT\' AND "Timeframe" = \'1d\'')
    ti_cnt = cur.fetchone()[0]
    print(f"Final Count for ETHUSDT 1d: Klines = {k_cnt}, TechnicalIndicators = {ti_cnt}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    test_pipeline_1d()
