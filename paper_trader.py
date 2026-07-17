#!/usr/bin/env python3
"""
G1: Paper trading — chạy champion config (calibrated XGB 4h ws5 h4h, thr 0.61)
trên dữ liệu live, ghi nhận mọi quyết định vào bảng PaperTrades.

Semantics giống backtest: tại mỗi 4h bar đóng cửa, nếu signal (label != 0 và
calibrated conf >= threshold) -> mở vị thế tại close của bar đó, đóng tại close
bar kế tiếp (4h sau). Phí 10bps + slippage 5bps mỗi side.

Idempotent: mỗi window (WindowEndMs) chỉ ghi 1 lần; vị thế mở được đóng khi
đến hạn bất kể script chạy lúc nào (lấy giá từ DB).

Chạy định kỳ qua Windows Scheduled Task (mỗi giờ) — script tự kiểm tra có
bar mới đóng cửa hay chưa.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import psycopg2

DB = dict(host=os.getenv("DB_HOST", "localhost"), port=int(os.getenv("DB_PORT", "5432")),
          database=os.getenv("DB_NAME", "bitcoin_analyst"),
          user=os.getenv("DB_USER", "postgres"), password=os.getenv("DB_PASS", "123456"))

SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"
WINDOW_SIZE = 5
TF_MS = 14_400_000
FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0
CONFIDENCE_THRESHOLD = 0.61
MODEL_PATH = Path(__file__).parent / "models" / "BTCUSDT_4h_ws5_h4h_XGB_calibrated.joblib"

FEATURE_COLS = [
    "CloseZscore", "ClosePctChange1", "ClosePctChange4", "ClosePctChange24",
    "HighLowRangePct", "BodyPct", "UpperWickPct", "LowerWickPct",
    "Rsi14", "Rsi14Slope", "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm",
    "Ema12Dist", "Ema26Dist", "Ema50Dist", "Ema200Dist", "Sma50Dist", "Sma200Dist",
    "BollingerWidth", "BollingerPosition", "Atr14Pct", "ObvEmaDist", "VwapDist",
    "RollingVwapDist", "VolumeZscore", "VolumeSma20Ratio", "TakerBuyRatio",
    "RecentPatternEncoded", "ActiveRuleCount",
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS "PaperTrades" (
    "Id" bigserial PRIMARY KEY,
    "Symbol" varchar(32) NOT NULL,
    "Timeframe" varchar(16) NOT NULL,
    "WindowEndMs" bigint NOT NULL,
    "EntryTimeMs" bigint NOT NULL,
    "ExitTimeMs" bigint NOT NULL,
    "Side" varchar(8) NOT NULL,
    "Confidence" double precision,
    "ProbDown" double precision,
    "ProbSideways" double precision,
    "ProbUp" double precision,
    "EntryPrice" double precision,
    "ExitPrice" double precision,
    "NetReturn" double precision,
    "Status" varchar(8) NOT NULL DEFAULT 'open',
    "ModelVersion" varchar(128),
    "CreatedAtUtc" timestamptz NOT NULL DEFAULT now(),
    "ClosedAtUtc" timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS "IX_PaperTrades_Symbol_WindowEndMs"
    ON "PaperTrades" ("Symbol", "WindowEndMs");
"""


def get_conn():
    conn = psycopg2.connect(**DB)
    conn.cursor().execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def time_features(open_ms):
    dt = datetime.fromtimestamp(open_ms / 1000, timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    dow = dt.weekday()
    return [
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
        1.0 if dow >= 5 else 0.0,
    ]


def build_latest_vector(cur):
    """Lấy ws bar MlFeatureStores mới nhất, kiểm tra liên tiếp, build vector ws*35."""
    cols = ", ".join(f'"{c}"' for c in FEATURE_COLS)
    cur.execute(
        f"""SELECT "OpenTimeMs", {cols} FROM "MlFeatureStores"
            WHERE "Symbol"=%s AND "Timeframe"=%s
            ORDER BY "OpenTimeMs" DESC LIMIT %s""",
        (SYMBOL, TIMEFRAME, WINDOW_SIZE))
    rows = cur.fetchall()
    if len(rows) < WINDOW_SIZE:
        return None
    rows = list(reversed(rows))  # oldest -> newest
    for i in range(1, len(rows)):
        if rows[i][0] - rows[i - 1][0] != TF_MS:
            print(f"  gap between bars {rows[i-1][0]} and {rows[i][0]}, skip")
            return None
    vector = []
    for r in rows:
        vals = r[1:]
        if any(v is None for v in vals):
            print(f"  null feature in bar {r[0]}, skip")
            return None
        vector.extend(float(v) for v in vals)
        vector.extend(time_features(r[0]))
    return rows[-1][0], np.array(vector, dtype=np.float32)


def get_close(cur, open_ms):
    cur.execute(
        """SELECT "Close" FROM "Klines" WHERE "Symbol"=%s AND "Timeframe"=%s AND "OpenTimeMs"=%s""",
        (SYMBOL, TIMEFRAME, open_ms))
    row = cur.fetchone()
    return float(row[0]) if row else None


def close_due_trades(conn, cur, now_ms):
    cur.execute("""SELECT "Id", "Side", "EntryPrice", "ExitTimeMs" FROM "PaperTrades"
                   WHERE "Symbol"=%s AND "Status"='open' AND "ExitTimeMs" <= %s""",
                (SYMBOL, now_ms))
    due = cur.fetchall()
    fee = FEE_BPS / 1e4
    slip = SLIPPAGE_BPS / 1e4
    for tid, side, entry_price, exit_ms in due:
        exit_price = get_close(cur, exit_ms)
        if exit_price is None:
            print(f"  trade {tid}: no kline at exit {exit_ms}, keep open")
            continue
        if side == "long":
            gross = (exit_price * (1 - slip) - entry_price * (1 + slip)) / (entry_price * (1 + slip))
        else:
            gross = (entry_price * (1 - slip) - exit_price * (1 + slip)) / (entry_price * (1 + slip))
        net = gross - 2 * fee
        cur.execute("""UPDATE "PaperTrades" SET "ExitPrice"=%s, "NetReturn"=%s,
                       "Status"='closed', "ClosedAtUtc"=now() WHERE "Id"=%s""",
                    (exit_price, net, tid))
        print(f"  closed trade {tid} {side} entry={entry_price:.1f} exit={exit_price:.1f} net={net*100:+.3f}%")
    conn.commit()


def main():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    conn = get_conn()
    cur = conn.cursor()

    # 1. Đóng các vị thế đến hạn
    close_due_trades(conn, cur, now_ms)

    # 2. Build vector từ bar mới nhất (chỉ xét bar đã đóng cửa)
    res = build_latest_vector(cur)
    if res is None:
        print("no usable window")
        conn.close()
        return
    window_end_ms, vector = res
    if window_end_ms + TF_MS > now_ms:
        print(f"latest bar {window_end_ms} still open, nothing to do")
        conn.close()
        return

    # 3. Đã xử lý window này chưa?
    cur.execute('SELECT 1 FROM "PaperTrades" WHERE "Symbol"=%s AND "WindowEndMs"=%s',
                (SYMBOL, window_end_ms))
    if cur.fetchone():
        print(f"window {window_end_ms} already processed")
        conn.close()
        return

    # 4. Predict
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(vector.reshape(1, -1))[0]
    label = int(np.argmax(proba)) - 1  # XGB remap {0,1,2} -> {-1,0,1}
    conf = float(proba.max())
    dt = datetime.fromtimestamp(window_end_ms / 1000, timezone.utc)
    print(f"window {dt:%Y-%m-%d %H:%M}Z: label={label} conf={conf:.3f} "
          f"(down={proba[0]:.3f} side={proba[1]:.3f} up={proba[2]:.3f})")

    # 5. Signal?
    if label == 0:
        print(f"  no signal (predicted sideways, conf={conf:.3f})")
        conn.close()
        return
    if conf < CONFIDENCE_THRESHOLD:
        print(f"  no signal (conf={conf:.3f} < {CONFIDENCE_THRESHOLD})")
        conn.close()
        return

    entry_price = get_close(cur, window_end_ms)
    if entry_price is None:
        print("  no kline at window end, skip")
        conn.close()
        return

    side = "long" if label == 1 else "short"
    exit_ms = window_end_ms + TF_MS
    cur.execute(
        """INSERT INTO "PaperTrades" ("Symbol","Timeframe","WindowEndMs","EntryTimeMs","ExitTimeMs",
           "Side","Confidence","ProbDown","ProbSideways","ProbUp","EntryPrice","Status","ModelVersion")
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'open',%s)
           ON CONFLICT ("Symbol","WindowEndMs") DO NOTHING""",
        (SYMBOL, TIMEFRAME, window_end_ms, window_end_ms, exit_ms,
         side, conf, float(proba[0]), float(proba[1]), float(proba[2]),
         entry_price, "XGB_calibrated"))
    conn.commit()
    print(f"  OPEN {side} @ {entry_price:.1f}, exit due {datetime.fromtimestamp(exit_ms/1000, timezone.utc):%Y-%m-%d %H:%M}Z")
    conn.close()


if __name__ == "__main__":
    main()
