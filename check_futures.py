import db_config

conn = db_config.get_db_connection()
cur = conn.cursor()
cur.execute("""
    SELECT "Symbol", count(*), min("OpenTimeMs"), max("OpenTimeMs")
    FROM "FuturesMetrics"
    GROUP BY "Symbol";
""")
for r in cur.fetchall():
    print("Futures:", r)

cur.close()
conn.close()
