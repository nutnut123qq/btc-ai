import db_config

conn = db_config.get_db_connection()
conn.autocommit = True
cur = conn.cursor()

print("[1] Ensuring pgvector types & functions exist...")
# Types are already loaded. If extension system catalog table is used, we can also ensure type exists.

print("[2] Adding 'EmbeddingVector' column of type vector(768)...")
cur.execute("""
    ALTER TABLE "NewsChunks" ADD COLUMN IF NOT EXISTS "EmbeddingVector" vector(768);
""")
print("  -> Column 'EmbeddingVector' added or already exists.")

print("[3] Migrating existing embeddings from 'Embedding' (real[]) to 'EmbeddingVector'...")
cur.execute("""
    UPDATE "NewsChunks"
    SET "EmbeddingVector" = "Embedding"::vector
    WHERE "Embedding" IS NOT NULL AND "EmbeddingVector" IS NULL;
""")
print(f"  -> Rows updated: {cur.rowcount}")

print("[4] Creating HNSW index on 'EmbeddingVector' (vector_cosine_ops, m=16, ef_construction=64)...")
cur.execute("""
    CREATE INDEX IF NOT EXISTS "ix_newschunks_hnsw"
    ON "NewsChunks"
    USING hnsw ("EmbeddingVector" vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""")
print("  -> HNSW index 'ix_newschunks_hnsw' created successfully!")

# Verify index and columns
cur.execute("""
    SELECT indexname, indexdef
    FROM pg_indexes
    WHERE tablename = 'NewsChunks' AND indexname = 'ix_newschunks_hnsw';
""")
idx_info = cur.fetchall()
print("\n[+] Verification: Index in pg_indexes:")
for i in idx_info:
    print(f"  Index Name: {i[0]}\n  Definition: {i[1]}")

cur.close()
conn.close()
