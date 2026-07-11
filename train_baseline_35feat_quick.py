#!/usr/bin/env python3
"""
Quick baseline on the new 35-feature WindowClassificationDatasets (1h, ws=10, h=1h).
Lightweight: LogisticRegression + LightGBM, class weights, time-based split.
"""

import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import lightgbm as lgb

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "bitcoin_analyst")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "123456")

SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
WINDOW_SIZE = 10
HORIZON = "1h"
SPLIT_MS = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

# Match WindowDatasetService.FeatureNames order
FEATURE_NAMES = [
    "CloseZscore", "ClosePctChange1", "ClosePctChange4", "ClosePctChange24",
    "HighLowRangePct", "BodyPct", "UpperWickPct", "LowerWickPct",
    "Rsi14", "Rsi14Slope", "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm",
    "Ema12Dist", "Ema26Dist", "Ema50Dist", "Ema200Dist",
    "Sma50Dist", "Sma200Dist", "BollingerWidth", "BollingerPosition",
    "Atr14Pct", "ObvEmaDist", "VwapDist", "RollingVwapDist",
    "VolumeZscore", "VolumeSma20Ratio", "TakerBuyRatio",
    "RecentPatternEncoded", "ActiveRuleCount",
    "HourSin", "HourCos", "DayOfWeekSin", "DayOfWeekCos", "IsWeekend",
]


def fetch():
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    cur.execute(
        '''
        SELECT "FeatureVector", "Label", "WindowEndMs"
        FROM "WindowClassificationDatasets"
        WHERE "Symbol"=%s AND "Timeframe"=%s AND "WindowSize"=%s AND "Horizon"=%s
        ORDER BY "WindowEndMs"
        ''',
        (SYMBOL, TIMEFRAME, WINDOW_SIZE, HORIZON)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def main():
    print("Fetching data...")
    rows = fetch()
    print(f"Rows: {len(rows)}")

    X = np.vstack([np.array(r[0], dtype=np.float32) for r in rows])
    y = np.array([r[1] for r in rows], dtype=np.int8)
    ends = np.array([r[2] for r in rows], dtype=np.int64)
    print(f"X shape: {X.shape}, features per bar: {X.shape[1] // WINDOW_SIZE}")

    train_mask = ends < SPLIT_MS
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"Labels train: {Counter(y_train)}, test: {Counter(y_test)}")

    maj = Counter(y_train).most_common(1)[0][0]
    maj_pred = np.full_like(y_test, maj)
    print(f"Majority  acc={accuracy_score(y_test, maj_pred):.4f} f1={f1_score(y_test, maj_pred, average='weighted'):.4f}")

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1, random_state=42)),
    ])
    t0 = time.time()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    print(f"LR        acc={accuracy_score(y_test, lr_pred):.4f} f1={f1_score(y_test, lr_pred, average='weighted'):.4f} ({time.time()-t0:.1f}s)")

    lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, class_weight="balanced", n_jobs=-1, random_state=42, verbosity=-1)
    t0 = time.time()
    lgbm.fit(X_train, y_train)
    lgb_pred = lgbm.predict(X_test)
    print(f"LGBM      acc={accuracy_score(y_test, lgb_pred):.4f} f1={f1_score(y_test, lgb_pred, average='weighted'):.4f} ({time.time()-t0:.1f}s)")

    imp = lgbm.feature_importances_
    names = [f"bar{i}_{n}" for i in range(WINDOW_SIZE) for n in FEATURE_NAMES]
    top = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)[:15]
    print("\nTop 15 LGBM features:")
    for n, v in top:
        print(f"  {n}: {v:.0f}")


if __name__ == "__main__":
    main()
