import db_config

conf_path = "D:/PostgreSQL/17/data/postgresql.conf"
with open(conf_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

new_lines = []
found = False
for line in lines:
    if line.strip().startswith("dynamic_library_path"):
        new_lines.append("dynamic_library_path = 'D:/PostgreSQL/17/lib;$libdir'\n")
        found = True
    else:
        new_lines.append(line)

if not found:
    new_lines.append("\n# Custom library path for pgvector\n")
    new_lines.append("dynamic_library_path = 'D:/PostgreSQL/17/lib;$libdir'\n")

with open(conf_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Updated postgresql.conf with dynamic_library_path")

conn = db_config.get_db_connection()
cur = conn.cursor()
cur.execute("SELECT pg_reload_conf();")
print("Reloaded config:", cur.fetchone())
cur.execute("SHOW dynamic_library_path;")
print("Current dynamic_library_path:", cur.fetchone())
cur.close()
conn.close()
