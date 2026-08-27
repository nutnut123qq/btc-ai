import psycopg2
from db_config import get_db_params

conn = psycopg2.connect(**get_db_params())
cur = conn.cursor()

for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    cur.execute('SELECT COUNT(*), MIN("OpenTimeMs"), MAX("OpenTimeMs") FROM "Klines" WHERE "Symbol"=%s AND "Timeframe"=%s', (sym, "1d"))
    k = cur.fetchone()
    cur.execute('SELECT COUNT(*) FROM "TechnicalIndicators" WHERE "Symbol"=%s AND "Timeframe"=%s', (sym, "1d"))
    ti = cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM "MlFeatureStores" WHERE "Symbol"=%s AND "Timeframe"=%s', (sym, "1d"))
    ml = cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM "WindowClassificationDatasets" WHERE "Symbol"=%s AND "Timeframe"=%s', (sym, "1d"))
    w = cur.fetchone()[0]
    print(f"{sym:<10} 1d -> Klines={k[0]} | TechInd={ti} | MlFeatures={ml} | Windows={w}")

conn.close()
