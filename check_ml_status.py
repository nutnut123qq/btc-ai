import db_config

conn = db_config.get_db_connection()
cur = conn.cursor()
cur.execute('SELECT "Symbol", "Timeframe", count(*) FROM "MlFeatureStores" GROUP BY "Symbol", "Timeframe" ORDER BY "Symbol", "Timeframe";')
for r in cur.fetchall():
    print("MlFeatureStores:", r)

cur.execute('SELECT "Symbol", "Timeframe", count(*) FROM "PriceTargets" GROUP BY "Symbol", "Timeframe" ORDER BY "Symbol", "Timeframe";')
for r in cur.fetchall():
    print("PriceTargets:", r)

cur.execute('SELECT "Symbol", "Timeframe", count(*) FROM "WindowClassificationDatasets" GROUP BY "Symbol", "Timeframe" ORDER BY "Symbol", "Timeframe";')
for r in cur.fetchall():
    print("WindowClassificationDatasets:", r)

cur.close()
conn.close()
