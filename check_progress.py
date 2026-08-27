import db_config

conn = db_config.get_db_connection()
cur = conn.cursor()

cur.execute("""
    SELECT "Symbol", "Timeframe", count(*)
    FROM "Klines"
    GROUP BY "Symbol", "Timeframe"
    ORDER BY "Symbol", "Timeframe";
""")
k_rows = cur.fetchall()

cur.execute("""
    SELECT "Symbol", "Timeframe", count(*)
    FROM "TechnicalIndicators"
    GROUP BY "Symbol", "Timeframe"
    ORDER BY "Symbol", "Timeframe";
""")
ti_rows = cur.fetchall()

print("=== KLINES ===")
for r in k_rows:
    print(f"  {r[0]:<10} {r[1]:<6} : {r[2]:>12,d}")

print("\n=== TECHNICAL INDICATORS ===")
for r in ti_rows:
    print(f"  {r[0]:<10} {r[1]:<6} : {r[2]:>12,d}")

cur.close()
conn.close()
