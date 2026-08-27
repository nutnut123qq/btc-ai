import db_config

conn = db_config.get_db_connection()
conn.autocommit = True
cur = conn.cursor()

# 1. Cosine distance operator <=>
cur.execute("SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector AS cosine_dist;")
print("1. Cosine Distance (<=>):", cur.fetchone()[0])

# 2. L2 distance operator <->
cur.execute("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector AS l2_dist;")
print("2. L2 Distance (<->):", cur.fetchone()[0])

# 3. Inner product operator <#>
cur.execute("SELECT '[1,2,3]'::vector <#> '[1,2,4]'::vector AS inner_prod;")
print("3. Inner Product (<#>):", cur.fetchone()[0])

cur.close()
conn.close()
print("\n[ALL PGVECTOR OPERATORS WORK PERFECTLY!]")
