import psycopg2
import numpy as np
from datetime import datetime, timezone
from db_config import get_db_params
from paper_trader import build_vector_at, get_model_for_symbol

DB_CONFIG = get_db_params()
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

start_ms = int(datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
end_ms = int(datetime(2026, 8, 24, 4, 0, tzinfo=timezone.utc).timestamp() * 1000)

cur.execute("""
    SELECT DISTINCT "OpenTimeMs" FROM "Klines"
    WHERE "Timeframe"='4h' AND "OpenTimeMs" >= %s AND "OpenTimeMs" <= %s
    ORDER BY "OpenTimeMs" ASC
""", (start_ms, end_ms))
bar_times = [r[0] for r in cur.fetchall()]

print(f"Total 4h cycles from Aug 1: {len(bar_times)}")
for sym, thr in [("BTCUSDT", 0.61), ("ETHUSDT", 0.50), ("SOLUSDT", 0.50)]:
    m, _ = get_model_for_symbol(sym)
    trigs = 0
    for t in bar_times:
        res = build_vector_at(cur, sym, "4h", 5, 14400000, t)
        if res:
            p = m.predict_proba(res[1].reshape(1, -1))[0]
            if p.max() >= thr and np.argmax(p) != 1:
                trigs += 1
                dt = datetime.fromtimestamp(t/1000, timezone.utc)
                side = "LONG" if np.argmax(p) == 2 else "SHORT"
                print(f"  [{sym}] {dt:%Y-%m-%d %H:%M} | {side} | Conf={p.max()*100:.1f}%")
    print(f"{sym} (thr={thr}) total triggers: {trigs}\n")

conn.close()
