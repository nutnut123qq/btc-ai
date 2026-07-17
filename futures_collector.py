#!/usr/bin/env python3
"""
Thu thập dữ liệu Binance USDT-M Futures vào bảng FuturesMetrics (PostgreSQL).

Nguồn dữ liệu (tất cả public, không cần API key):
  - REST polling (chỉ giữ ~30 ngày gần nhất):
      /futures/data/openInterestHist           (5m)
      /futures/data/globalLongShortAccountRatio(5m)
      /futures/data/topLongShortAccountRatio   (5m)
      /futures/data/topLongShortPositionRatio  (5m)
      /futures/data/takerlongshortRatio        (5m)
      /fapi/v1/fundingRate                     (8h, full history)
  - Data dump (history dài, lag ~1 ngày):
      https://data.binance.vision/data/futures/um/daily/metrics/...

Modes:
  poll                Poll REST 1 lần (an toàn chạy lặp lại, upsert idempotent)
  loop --interval N   Poll liên tục mỗi N giây (mặc định 1800)
  backfill-funding    Backfill funding rate từ 2019-09 đến nay (REST phân trang)
  backfill-dump       Backfill OI/L-S/taker từ daily zip dumps (--from, --to)

Bảng FuturesMetrics được tạo bởi script này (raw SQL). EF Core entity/migration
sẽ bổ sung sau khi cần truy cập từ backend (xem roadmap C4).
"""

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone

import psycopg2

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "bitcoin_analyst")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "123456")

FAPI = "https://fapi.binance.com"
DUMP = "https://data.binance.vision/data/futures/um/daily/metrics"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS "FuturesMetrics" (
    "Id" bigserial PRIMARY KEY,
    "Symbol" varchar(32) NOT NULL,
    "OpenTimeMs" bigint NOT NULL,
    "OpenInterest" double precision,
    "OpenInterestValue" double precision,
    "TopTraderLsCountRatio" double precision,
    "TopTraderLsSumRatio" double precision,
    "GlobalLsRatio" double precision,
    "TakerBuySellVolRatio" double precision,
    "FundingRate" double precision,
    "MarkPrice" double precision,
    "CreatedAtUtc" timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS "IX_FuturesMetrics_Symbol_OpenTimeMs"
    ON "FuturesMetrics" ("Symbol", "OpenTimeMs");
"""

# Mỗi nguồn ghi một tập cột; COALESCE để merge nhiều nguồn vào cùng 1 row (5m grid)
UPSERT_SQL = """
INSERT INTO "FuturesMetrics" ("Symbol", "OpenTimeMs", {cols})
VALUES (%s, %s, {placeholders})
ON CONFLICT ("Symbol", "OpenTimeMs") DO UPDATE SET
{updates}
"""


def _f(v):
    """Parse float, chấp nhận chuỗi rỗng -> None."""
    return float(v) if v not in ("", None) else None


def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )


def ensure_table(cur):
    cur.execute(CREATE_TABLE_SQL)


def http_get_json(url, retries=4, timeout=30):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "btc-futures-collector/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            wait = 2 ** attempt * 2
            print(f"    GET failed ({e}), retry in {wait}s: {url[:100]}", flush=True)
            time.sleep(wait)
    return None


def upsert_rows(cur, symbol, rows, col_map):
    """rows: list[dict ms_timestamp -> values]; col_map: {json_key: db_col}"""
    if not rows:
        return 0
    cols = list(col_map.values())
    col_sql = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f'"{c}" = COALESCE(EXCLUDED."{c}", "FuturesMetrics"."{c}")' for c in cols)
    sql = UPSERT_SQL.format(cols=col_sql, placeholders=placeholders, updates=updates)

    # Mỗi row là dữ liệu của 1 nguồn; ON CONFLICT + COALESCE merge vào row chung
    count = 0
    for r in rows:
        cur.execute(sql, (symbol, *r))
        count += 1
    return count


# ---------- REST polling ----------

def poll_once(symbol):
    conn = get_connection()
    cur = conn.cursor()
    ensure_table(cur)
    total = 0

    # Open interest (5m, limit 500 ~ 41h)
    data = http_get_json(f"{FAPI}/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=500")
    if isinstance(data, list):
        rows = [(int(x["timestamp"]), float(x["sumOpenInterest"]), float(x["sumOpenInterestValue"]))
                for x in data]
        n = upsert_rows(cur, symbol, rows, {"a": "OpenInterest", "b": "OpenInterestValue"})
        total += n
        print(f"  openInterestHist: {n} rows", flush=True)

    # Global long/short account ratio
    data = http_get_json(f"{FAPI}/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=500")
    if isinstance(data, list):
        rows = [(int(x["timestamp"]), float(x["longShortRatio"])) for x in data]
        n = upsert_rows(cur, symbol, rows, {"a": "GlobalLsRatio"})
        total += n
        print(f"  globalLongShortAccountRatio: {n} rows", flush=True)

    # Top trader long/short (accounts)
    data = http_get_json(f"{FAPI}/futures/data/topLongShortAccountRatio?symbol={symbol}&period=5m&limit=500")
    if isinstance(data, list):
        rows = [(int(x["timestamp"]), float(x["longShortRatio"])) for x in data]
        n = upsert_rows(cur, symbol, rows, {"a": "TopTraderLsCountRatio"})
        total += n
        print(f"  topLongShortAccountRatio: {n} rows", flush=True)

    # Top trader long/short (positions)
    data = http_get_json(f"{FAPI}/futures/data/topLongShortPositionRatio?symbol={symbol}&period=5m&limit=500")
    if isinstance(data, list):
        rows = [(int(x["timestamp"]), float(x["longShortRatio"])) for x in data]
        n = upsert_rows(cur, symbol, rows, {"a": "TopTraderLsSumRatio"})
        total += n
        print(f"  topLongShortPositionRatio: {n} rows", flush=True)

    # Taker buy/sell volume ratio
    data = http_get_json(f"{FAPI}/futures/data/takerlongshortRatio?symbol={symbol}&period=5m&limit=500")
    if isinstance(data, list):
        rows = [(int(x["timestamp"]), float(x["buySellRatio"])) for x in data]
        n = upsert_rows(cur, symbol, rows, {"a": "TakerBuySellVolRatio"})
        total += n
        print(f"  takerlongshortRatio: {n} rows", flush=True)

    # Funding rate (mới nhất)
    data = http_get_json(f"{FAPI}/fapi/v1/fundingRate?symbol={symbol}&limit=100")
    if isinstance(data, list):
        rows = [(int(x["fundingTime"]), float(x["fundingRate"]), float(x.get("markPrice") or 0) or None)
                for x in data]
        n = upsert_rows(cur, symbol, rows, {"a": "FundingRate", "b": "MarkPrice"})
        total += n
        print(f"  fundingRate: {n} rows", flush=True)

    conn.commit()
    cur.close()
    conn.close()
    print(f"  poll done: {total} upserts", flush=True)
    return total


# ---------- Funding full history (C1) ----------

def backfill_funding(symbol, start_date=date(2019, 9, 1)):
    conn = get_connection()
    cur = conn.cursor()
    ensure_table(cur)
    start_ms = int(datetime(start_date.year, start_date.month, start_date.day,
                            tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    total = 0
    while start_ms < end_ms:
        url = f"{FAPI}/fapi/v1/fundingRate?symbol={symbol}&startTime={start_ms}&limit=1000"
        data = http_get_json(url)
        if not isinstance(data, list) or not data:
            break
        rows = [(int(x["fundingTime"]), float(x["fundingRate"]), float(x.get("markPrice") or 0) or None)
                for x in data]
        total += upsert_rows(cur, symbol, rows, {"a": "FundingRate", "b": "MarkPrice"})
        conn.commit()
        last = int(data[-1]["fundingTime"])
        print(f"  funding ... {datetime.fromtimestamp(last/1000, timezone.utc):%Y-%m-%d} ({total} total)", flush=True)
        if last <= start_ms or len(data) < 1000:
            break
        start_ms = last + 1
        time.sleep(0.3)
    cur.close()
    conn.close()
    print(f"  backfill-funding done: {total} rows", flush=True)
    return total


# ---------- Daily dump backfill ----------

def dump_date_range(cur, symbol, from_d, to_d):
    """Trả về list ngày cần tải (bỏ qua ngày đã có đủ 288 rows)."""
    cur.execute(
        'SELECT "OpenTimeMs" FROM "FuturesMetrics" WHERE "Symbol"=%s AND "OpenInterest" IS NOT NULL',
        (symbol,))
    have = set()
    for (ms,) in cur.fetchall():
        have.add(datetime.fromtimestamp(ms / 1000, timezone.utc).date())
    days = []
    d = from_d
    while d <= to_d:
        if d not in have:
            days.append(d)
        d += timedelta(days=1)
    return days


def backfill_dump(symbol, from_d, to_d):
    conn = get_connection()
    cur = conn.cursor()
    ensure_table(cur)
    days = dump_date_range(cur, symbol, from_d, to_d)
    print(f"  {len(days)} days to backfill ({from_d} -> {to_d})", flush=True)
    total = 0
    for i, d in enumerate(days):
        url = f"{DUMP}/{symbol}/{symbol}-metrics-{d.isoformat()}.zip"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "btc-futures-collector/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                blob = resp.read()
        except Exception as e:
            print(f"  {d}: download failed ({e})", flush=True)
            time.sleep(1)
            continue
        try:
            zf = zipfile.ZipFile(io.BytesIO(blob))
            name = zf.namelist()[0]
            text = zf.read(name).decode("utf-8")
        except Exception as e:
            print(f"  {d}: bad zip ({e})", flush=True)
            continue
        rows = []
        reader = csv.DictReader(io.StringIO(text))
        for r in reader:
            ts = int(datetime.strptime(r["create_time"], "%Y-%m-%d %H:%M:%S")
                     .replace(tzinfo=timezone.utc).timestamp() * 1000)
            rows.append((ts,
                         _f(r["sum_open_interest"]),
                         _f(r["sum_open_interest_value"]),
                         _f(r["count_toptrader_long_short_ratio"]),
                         _f(r["sum_toptrader_long_short_ratio"]),
                         _f(r["count_long_short_ratio"]),
                         _f(r["sum_taker_long_short_vol_ratio"])))
        total += upsert_rows(cur, symbol, rows, {
            "a": "OpenInterest", "b": "OpenInterestValue",
            "c": "TopTraderLsCountRatio", "d": "TopTraderLsSumRatio",
            "e": "GlobalLsRatio", "f": "TakerBuySellVolRatio",
        })
        if (i + 1) % 50 == 0:
            conn.commit()
            print(f"  dump ... {d} ({i+1}/{len(days)} days, {total} rows)", flush=True)
        time.sleep(0.15)
    conn.commit()
    cur.close()
    conn.close()
    print(f"  backfill-dump done: {total} rows", flush=True)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["poll", "loop", "backfill-funding", "backfill-dump"])
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", type=int, default=1800, help="loop interval seconds")
    p.add_argument("--from", dest="from_date", default="2021-01-01")
    p.add_argument("--to", dest="to_date", default=None)
    args = p.parse_args()

    if args.mode == "poll":
        poll_once(args.symbol)
    elif args.mode == "loop":
        print(f"Polling loop every {args.interval}s for {args.symbol}", flush=True)
        while True:
            try:
                print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] poll", flush=True)
                poll_once(args.symbol)
            except Exception as e:
                print(f"  poll error: {e}", flush=True)
            time.sleep(args.interval)
    elif args.mode == "backfill-funding":
        backfill_funding(args.symbol)
    elif args.mode == "backfill-dump":
        from_d = date.fromisoformat(args.from_date)
        to_d = date.fromisoformat(args.to_date) if args.to_date else (
            datetime.now(timezone.utc).date() - timedelta(days=1))
        backfill_dump(args.symbol, from_d, to_d)


if __name__ == "__main__":
    main()
