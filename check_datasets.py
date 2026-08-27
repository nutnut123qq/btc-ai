import psycopg2
from db_config import get_db_params

conn = psycopg2.connect(**get_db_params())
cur = conn.cursor()

print("--- WindowClassificationDatasets ---")
cur.execute('''
    SELECT "Symbol", "Timeframe", "WindowSize", "Horizon", COUNT(*) 
    FROM "WindowClassificationDatasets" 
    GROUP BY 1,2,3,4 
    ORDER BY 1,2,3,4
''')
for r in cur.fetchall():
    print(f"  {r[0]:<10} | {r[1]:<5} | ws={r[2]:<3} | h={r[3]:<4} | count={r[4]:,}")

print("\n--- MlFeatureStores ---")
cur.execute('''
    SELECT "Symbol", "Timeframe", COUNT(*) 
    FROM "MlFeatureStores" 
    GROUP BY 1,2 
    ORDER BY 1,2
''')
for r in cur.fetchall():
    print(f"  {r[0]:<10} | {r[1]:<5} | count={r[2]:,}")

print("\n--- PriceTargets ---")
cur.execute('''
    SELECT "Symbol", "Timeframe", "Horizon", COUNT(*) 
    FROM "PriceTargets" 
    GROUP BY 1,2,3 
    ORDER BY 1,2,3
''')
for r in cur.fetchall():
    print(f"  {r[0]:<10} | {r[1]:<5} | h={r[2]:<4} | count={r[3]:,}")

conn.close()
