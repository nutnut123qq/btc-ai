import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from psycopg2.extras import execute_values
from db_config import get_db_connection

INTERVAL_MS = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000
}

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3)
session.mount('https://', adapter)

def fetch_chunk(symbol, interval, start_time_ms, limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time_ms}&limit={limit}"
    for attempt in range(5):
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "5"))
                time.sleep(retry_after)
                continue
            if resp.status_code == 200:
                data = resp.json()
                return data
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            time.sleep(0.5 * (attempt + 1))
    return []

def test_fetch_parallel(symbol="ETHUSDT", interval="1d", start_ms=1577836800000):
    now_ms = int(time.time() * 1000)
    intv_ms = INTERVAL_MS[interval]
    chunk_span_ms = 1000 * intv_ms
    
    start_times = list(range(start_ms, now_ms, chunk_span_ms))
    print(f"Testing {symbol} {interval}: {len(start_times)} chunks to fetch.")
    
    t0 = time.time()
    all_candles = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_chunk, symbol, interval, st): st for st in start_times}
        for f in as_completed(futures):
            res = f.result()
            if res:
                all_candles.extend(res)
                
    t1 = time.time()
    print(f"Fetched {len(all_candles)} candles in {t1 - t0:.2f}s.")

if __name__ == "__main__":
    test_fetch_parallel("ETHUSDT", "1d")
