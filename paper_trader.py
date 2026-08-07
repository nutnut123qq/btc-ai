#!/usr/bin/env python3
"""
G1: Paper trading — chạy multi-timeframe configs (4h, 1h, 30m)
trên dữ liệu live, ghi nhận mọi quyết định vào bảng PaperTrades.

Semantics giống backtest: tại mỗi bar đóng cửa, nếu signal (label != 0 và
calibrated conf >= threshold) -> mở vị thế tại close của bar đó, đóng tại close
bar kế tiếp. Phí 10bps + slippage 5bps mỗi side.

Idempotent: mỗi (Symbol, Timeframe, WindowEndMs) chỉ ghi 1 lần.
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

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_NAME", "bitcoin_analyst"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASS", "123456"),
)

SYMBOL = "BTCUSDT"
FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0
MODELS_DIR = Path(__file__).parent / "models"

TIMEFRAME_CONFIGS = {
    "4h": {
        "window_size": 5,
        "tf_ms": 14_400_000,
        "threshold": 0.61,
        "model_file": "BTCUSDT_4h_ws5_h4h_XGB_calibrated.joblib",
    },
    "1h": {
        "window_size": 5,
        "tf_ms": 3_600_000,
        "threshold": 0.58,
        "model_file": "BTCUSDT_1h_ws5_h1h_XGB_balanced.joblib",
    },
    "30m": {
        "window_size": 5,
        "tf_ms": 1_800_000,
        "threshold": 0.56,
        "model_file": "BTCUSDT_30m_ws5_h1h_XGB_balanced.joblib",
    },
}

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
DROP INDEX IF EXISTS "IX_PaperTrades_Symbol_WindowEndMs";
CREATE UNIQUE INDEX IF NOT EXISTS "IX_PaperTrades_Symbol_Timeframe_WindowEndMs"
    ON "PaperTrades" ("Symbol", "Timeframe", "WindowEndMs");
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


def build_latest_vector(cur, timeframe, window_size, tf_ms):
    """Lấy ws bar MlFeatureStores mới nhất cho timeframe, build vector ws*35."""
    cols = ", ".join(f'"{c}"' for c in FEATURE_COLS)
    cur.execute(
        f"""SELECT "OpenTimeMs", {cols} FROM "MlFeatureStores"
            WHERE "Symbol"=%s AND "Timeframe"=%s
            ORDER BY "OpenTimeMs" DESC LIMIT %s""",
        (SYMBOL, timeframe, window_size))
    rows = cur.fetchall()
    if len(rows) < window_size:
        return None
    rows = list(reversed(rows))  # oldest -> newest
    for i in range(1, len(rows)):
        if rows[i][0] - rows[i - 1][0] != tf_ms:
            print(f"  [{timeframe}] gap between bars {rows[i-1][0]} and {rows[i][0]}, skip")
            return None
    vector = []
    for r in rows:
        vals = r[1:]
        if any(v is None for v in vals):
            print(f"  [{timeframe}] null feature in bar {r[0]}, skip")
            return None
        vector.extend(float(v) for v in vals)
        vector.extend(time_features(r[0]))
    return rows[-1][0], np.array(vector, dtype=np.float32)


def get_close(cur, timeframe, open_ms):
    cur.execute(
        """SELECT "Close" FROM "Klines" WHERE "Symbol"=%s AND "Timeframe"=%s AND "OpenTimeMs"=%s""",
        (SYMBOL, timeframe, open_ms))
    row = cur.fetchone()
    return float(row[0]) if row else None


def close_due_trades(conn, cur, now_ms):
    cur.execute("""SELECT "Id", "Timeframe", "Side", "EntryPrice", "ExitTimeMs" FROM "PaperTrades"
                   WHERE "Symbol"=%s AND "Status"='open' AND "ExitTimeMs" <= %s""",
                (SYMBOL, now_ms))
    due = cur.fetchall()
    fee = FEE_BPS / 1e4
    slip = SLIPPAGE_BPS / 1e4
    for tid, tf, side, entry_price, exit_ms in due:
        exit_price = get_close(cur, tf, exit_ms)
        if exit_price is None:
            print(f"  trade {tid} [{tf}]: no kline at exit {exit_ms}, keep open")
            continue
        if side == "long":
            gross = (exit_price * (1 - slip) - entry_price * (1 + slip)) / (entry_price * (1 + slip))
        else:
            gross = (entry_price * (1 - slip) - exit_price * (1 + slip)) / (entry_price * (1 + slip))
        net = gross - 2 * fee
        cur.execute("""UPDATE "PaperTrades" SET "ExitPrice"=%s, "NetReturn"=%s,
                       "Status"='closed', "ClosedAtUtc"=now() WHERE "Id"=%s""",
                    (exit_price, net, tid))
        print(f"  closed trade {tid} [{tf}] {side} entry={entry_price:.1f} exit={exit_price:.1f} net={net*100:+.3f}%")
    conn.commit()


def process_timeframe(conn, cur, tf, cfg, now_ms):
    window_size = cfg["window_size"]
    tf_ms = cfg["tf_ms"]
    threshold = cfg["threshold"]
    model_path = MODELS_DIR / cfg["model_file"]

    if not model_path.exists():
        print(f"[{tf}] Model file {model_path.name} not found, skip")
        return

    res = build_latest_vector(cur, tf, window_size, tf_ms)
    if res is None:
        print(f"[{tf}] no usable window")
        return
    window_end_ms, vector = res
    if window_end_ms + tf_ms > now_ms:
        print(f"[{tf}] latest bar {window_end_ms} still open, skip")
        return

    cur.execute('SELECT 1 FROM "PaperTrades" WHERE "Symbol"=%s AND "Timeframe"=%s AND "WindowEndMs"=%s',
                (SYMBOL, tf, window_end_ms))
    if cur.fetchone():
        print(f"[{tf}] window {window_end_ms} already processed")
        return

    model = joblib.load(model_path)
    proba = model.predict_proba(vector.reshape(1, -1))[0]
    label = int(np.argmax(proba)) - 1  # XGB remap {0,1,2} -> {-1,0,1}
    conf = float(proba.max())
    dt = datetime.fromtimestamp(window_end_ms / 1000, timezone.utc)
    print(f"[{tf}] window {dt:%Y-%m-%d %H:%M}Z: label={label} conf={conf:.3f} "
          f"(down={proba[0]:.3f} side={proba[1]:.3f} up={proba[2]:.3f})")

    if label == 0:
        print(f"  [{tf}] no signal (predicted sideways)")
        return
    if conf < threshold:
        print(f"  [{tf}] no signal (conf={conf:.3f} < {threshold})")
        return

    entry_price = get_close(cur, tf, window_end_ms)
    if entry_price is None:
        print(f"  [{tf}] no kline at window end, skip")
        return

    side = "long" if label == 1 else "short"
    exit_ms = window_end_ms + tf_ms
    cur.execute(
        """INSERT INTO "PaperTrades" ("Symbol","Timeframe","WindowEndMs","EntryTimeMs","ExitTimeMs",
           "Side","Confidence","ProbDown","ProbSideways","ProbUp","EntryPrice","Status","ModelVersion")
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'open',%s)
           ON CONFLICT ("Symbol","Timeframe","WindowEndMs") DO NOTHING""",
        (SYMBOL, tf, window_end_ms, window_end_ms, exit_ms,
         side, conf, float(proba[0]), float(proba[1]), float(proba[2]),
         entry_price, cfg["model_file"]))
    conn.commit()
    print(f"  [{tf}] OPEN {side} @ {entry_price:.1f}, exit due {datetime.fromtimestamp(exit_ms/1000, timezone.utc):%Y-%m-%d %H:%M}Z")


def main():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    conn = get_conn()
    cur = conn.cursor()

    close_due_trades(conn, cur, now_ms)

    for tf, cfg in TIMEFRAME_CONFIGS.items():
        try:
            process_timeframe(conn, cur, tf, cfg, now_ms)
        except Exception as e:
            print(f"[{tf}] Error in paper trader: {e}")

    conn.close()


if __name__ == "__main__":
    main()
