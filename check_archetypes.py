import db_config

conn = db_config.get_db_connection()
cur = conn.cursor()
cur.execute('SELECT "Symbol", "Timeframe", count(*) FROM "CandleArchetypes" GROUP BY "Symbol", "Timeframe";')
for r in cur.fetchall():
    print("CandleArchetypes:", r)

cur.execute('SELECT "Symbol", "Timeframe", count(*) FROM "ArchetypeOccurrences" GROUP BY "Symbol", "Timeframe";')
for r in cur.fetchall():
    print("ArchetypeOccurrences:", r)

cur.execute('SELECT "Symbol", "Timeframe", count(*) FROM "ArchetypeTransitions" GROUP BY "Symbol", "Timeframe";')
for r in cur.fetchall():
    print("ArchetypeTransitions:", r)

cur.close()
conn.close()
