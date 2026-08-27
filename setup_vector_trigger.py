import db_config

conn = db_config.get_db_connection()
conn.autocommit = True
cur = conn.cursor()

print("[*] Creating automatic trigger to sync Embedding -> EmbeddingVector...")
cur.execute("""
    CREATE OR REPLACE FUNCTION sync_newschunk_embedding_vector()
    RETURNS TRIGGER AS $$
    BEGIN
        IF NEW."Embedding" IS NOT NULL THEN
            NEW."EmbeddingVector" = NEW."Embedding"::vector;
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    DROP TRIGGER IF EXISTS trg_sync_newschunk_embedding_vector ON "NewsChunks";
    CREATE TRIGGER trg_sync_newschunk_embedding_vector
    BEFORE INSERT OR UPDATE OF "Embedding" ON "NewsChunks"
    FOR EACH ROW
    EXECUTE FUNCTION sync_newschunk_embedding_vector();
""")
print("[+] Trigger created successfully!")

# Test inserting/updating an Embedding and verify EmbeddingVector is automatically populated
cur.execute('SELECT "Id" FROM "NewsChunks" LIMIT 1;')
cid = cur.fetchone()[0]

test_emb = [0.01 * (i % 10) for i in range(768)]
cur.execute("""
    UPDATE "NewsChunks"
    SET "Embedding" = %s
    WHERE "Id" = %s;
""", (test_emb, cid))

cur.execute("""
    SELECT "Embedding" IS NOT NULL, "EmbeddingVector" IS NOT NULL
    FROM "NewsChunks"
    WHERE "Id" = %s;
""", (cid,))
res = cur.fetchone()
print(f"[+] Verification test: Embedding set: {res[0]}, EmbeddingVector auto-synced by trigger: {res[1]}")

cur.close()
conn.close()
