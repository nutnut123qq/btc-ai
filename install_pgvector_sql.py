import db_config

sql_file = "c:/Users/hahuy/Desktop/btc/ai/pgvector_extracted/share/extension/vector--0.8.6.sql"
with open(sql_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

clean_lines = [line for line in lines if not line.strip().startswith("\\")]
clean_sql = "".join(clean_lines)

# Replace MODULE_PATHNAME with explicit path
clean_sql = clean_sql.replace("MODULE_PATHNAME", "D:/PostgreSQL/17/lib/vector")

conn = db_config.get_db_connection()
conn.autocommit = True
cur = conn.cursor()

try:
    cur.execute(clean_sql)
    print("[SUCCESS] pgvector types, operators, and HNSW index access method created successfully in PostgreSQL!")
except Exception as e:
    print("[ERROR] Failed to execute:", e)
finally:
    cur.close()
    conn.close()
