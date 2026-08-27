import psycopg2
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone

from db_config import get_db_params
from paper_trader import build_vector_at, get_model_for_symbol

DB_CONFIG = get_db_params()

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

start_ms = int(datetime(2026, 8, 20, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
end_ms = int(datetime(2026, 8, 24, 4, 0, tzinfo=timezone.utc).timestamp() * 1000)

cur.execute("""
    SELECT DISTINCT "OpenTimeMs" FROM "Klines"
    WHERE "Timeframe"='4h' AND "OpenTimeMs" >= %s AND "OpenTimeMs" <= %s
    ORDER BY "OpenTimeMs" ASC
""", (start_ms, end_ms))
bar_times = [r[0] for r in cur.fetchall()]

for sym in ["ETHUSDT", "SOLUSDT", "BTCUSDT"]:
    model, _ = get_model_for_symbol(sym)
    print(f"\n=== Predictions for {sym} (Threshold=0.61) ===")
    triggered = 0
    for t in bar_times:
        dt = datetime.fromtimestamp(t/1000, timezone.utc)
        res = build_vector_at(cur, sym, "4h", 5, 14400000, t)
        if not res:
            print(f"  {dt:%Y-%m-%d %H:%M} | No continuous vector")
            continue
        _, vec = res
        probas = model.predict_proba(vec.reshape(1, -1))[0]
        pred_idx = int(np.argmax(probas))
        conf = float(probas[pred_idx])
        label = {0: "DOWN (Short)", 1: "SIDEWAYS", 2: "UP (Long)"}[pred_idx]
        is_trig = (pred_idx != 1) and (conf >= 0.61)
        if is_trig:
            triggered += 1
            mark = ">>> TRIGGER"
        else:
            mark = ""
        print(f"  {dt:%Y-%m-%d %H:%M} | {label:15s} | Conf={conf*100:5.1f}% (Down={probas[0]*100:4.1f}%, Side={probas[1]*100:4.1f}%, Up={probas[2]*100:4.1f}%) {mark}")
    print(f"Total triggered for {sym}: {triggered}")

conn.close()
