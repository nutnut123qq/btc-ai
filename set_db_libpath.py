import db_config

conn = db_config.get_db_connection()
conn.autocommit = True
cur = conn.cursor()
cur.execute("ALTER DATABASE bitcoin_analyst SET dynamic_library_path = 'D:/PostgreSQL/17/lib;$libdir';")
cur.execute("ALTER SYSTEM SET dynamic_library_path = 'D:/PostgreSQL/17/lib;$libdir';")
cur.execute("SELECT pg_reload_conf();")
cur.close()
conn.close()

conn2 = db_config.get_db_connection()
cur2 = conn2.cursor()
cur2.execute("SHOW dynamic_library_path;")
print("dynamic_library_path is now:", cur2.fetchone())
cur2.close()
conn2.close()
