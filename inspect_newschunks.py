import db_config

conn = db_config.get_db_connection()
cur = conn.cursor()

cur.execute("""
    SELECT count(*), count("Embedding")
    FROM "NewsChunks";
""")
row = cur.fetchone()
print(f"NewsChunks total rows: {row[0]}, rows with Embedding: {row[1]}")

cur.execute("""
    SELECT column_name, data_type, udt_name
    FROM information_schema.columns
    WHERE table_name = 'NewsChunks';
""")
print("\nNewsChunks columns:")
for c in cur.fetchall():
    print(" ", c)

cur.close()
conn.close()
