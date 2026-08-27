import db_config

conn = db_config.get_db_connection()
cur = conn.cursor()
cur.execute('SELECT * FROM "MlFeatureStores" WHERE "Symbol"=\'ETHUSDT\' LIMIT 1;')
cols = [d[0] for d in cur.description]
row = cur.fetchone()
for c, v in zip(cols, row):
    print(f"{c}: {v}")

cur.close()
conn.close()
