import io
import csv
import sys
import time
import zipfile
import urllib.request
from datetime import date, datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from psycopg2.extras import execute_values
from db_config import get_db_connection

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

DUMP = "https://data.binance.vision/data/futures/um/daily/metrics"

def _f(v):
    return float(v) if v not in ("", None) else None

def download_and_parse_day(symbol, d):
    url = f"{DUMP}/{symbol}/{symbol}-metrics-{d.isoformat()}.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "btc-futures-collector/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            blob = resp.read()
        zf = zipfile.ZipFile(io.BytesIO(blob))
        name = zf.namelist()[0]
        text = zf.read(name).decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        for r in reader:
            ts = int(datetime.strptime(r["create_time"], "%Y-%m-%d %H:%M:%S")
                     .replace(tzinfo=timezone.utc).timestamp() * 1000)
            rows.append((
                symbol, ts,
                _f(r.get("sum_open_interest")),
                _f(r.get("sum_open_interest_value")),
                _f(r.get("count_toptrader_long_short_ratio")),
                _f(r.get("sum_toptrader_long_short_ratio")),
                _f(r.get("count_long_short_ratio")),
                _f(r.get("sum_taker_long_short_vol_ratio"))
            ))
        return rows
    except Exception:
        return []

def backfill_futures_dump_fast(symbol, from_date=date(2021, 1, 1), to_date=None):
    if to_date is None:
        to_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('SELECT "OpenTimeMs" FROM "FuturesMetrics" WHERE "Symbol"=%s AND "OpenInterest" IS NOT NULL', (symbol,))
    have = set()
    for (ms,) in cur.fetchall():
        have.add(datetime.fromtimestamp(ms / 1000, timezone.utc).date())

    days_to_fetch = []
    d = from_date
    while d <= to_date:
        if d not in have:
            days_to_fetch.append(d)
        d += timedelta(days=1)

    print(f"\n[{symbol}] Fast Futures Backfill: {len(days_to_fetch)} daily zips to download ({from_date} -> {to_date})...", flush=True)
    t0 = time.time()

    all_rows = []
    completed = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {executor.submit(download_and_parse_day, symbol, day): day for day in days_to_fetch}
        for f in as_completed(future_map):
            res = f.result()
            if res:
                all_rows.extend(res)
            completed += 1
            if completed % 300 == 0 or completed == len(days_to_fetch):
                print(f"  > Downloaded {completed}/{len(days_to_fetch)} days ({len(all_rows):,} metric records collected)...", flush=True)

    t1 = time.time()
    print(f"  > Downloads completed in {t1 - t0:.2f}s. Deduplicating...", flush=True)

    # Deduplicate by (Symbol, OpenTimeMs)
    rows_dict = {}
    for r in all_rows:
        rows_dict[r[1]] = r
    deduped_rows = [rows_dict[k] for k in sorted(rows_dict.keys())]

    print(f"  > Inserting {len(deduped_rows):,} unique records into Postgres...", flush=True)

    # Upsert with COALESCE
    upsert_sql = """
        INSERT INTO "FuturesMetrics" (
            "Symbol", "OpenTimeMs", "OpenInterest", "OpenInterestValue",
            "TopTraderLsCountRatio", "TopTraderLsSumRatio", "GlobalLsRatio", "TakerBuySellVolRatio"
        ) VALUES %s
        ON CONFLICT ("Symbol", "OpenTimeMs") DO UPDATE SET
            "OpenInterest" = COALESCE(EXCLUDED."OpenInterest", "FuturesMetrics"."OpenInterest"),
            "OpenInterestValue" = COALESCE(EXCLUDED."OpenInterestValue", "FuturesMetrics"."OpenInterestValue"),
            "TopTraderLsCountRatio" = COALESCE(EXCLUDED."TopTraderLsCountRatio", "FuturesMetrics"."TopTraderLsCountRatio"),
            "TopTraderLsSumRatio" = COALESCE(EXCLUDED."TopTraderLsSumRatio", "FuturesMetrics"."TopTraderLsSumRatio"),
            "GlobalLsRatio" = COALESCE(EXCLUDED."GlobalLsRatio", "FuturesMetrics"."GlobalLsRatio"),
            "TakerBuySellVolRatio" = COALESCE(EXCLUDED."TakerBuySellVolRatio", "FuturesMetrics"."TakerBuySellVolRatio");
    """

    batch_size = 10000
    for i in range(0, len(deduped_rows), batch_size):
        chunk = deduped_rows[i:i + batch_size]
        execute_values(cur, upsert_sql, chunk, page_size=5000)
        conn.commit()

    cur.close()
    conn.close()
    print(f"  > [{symbol}] Insert complete in {time.time() - t1:.2f}s.", flush=True)

if __name__ == "__main__":
    for sym in ["SOLUSDT", "ETHUSDT"]:
        backfill_futures_dump_fast(sym)
