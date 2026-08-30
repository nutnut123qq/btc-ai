import time
import numpy as np
import db_config

def test_hnsw_performance():
    conn = db_config.get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()

    # 1. Generate some synthetic 768-dim embeddings for testing NewsChunks
    print("[*] Generating and inserting synthetic test embeddings into NewsChunks...")
    cur.execute('SELECT "Id" FROM "NewsChunks" LIMIT 50;')
    chunk_ids = [r[0] for r in cur.fetchall()]

    if chunk_ids:
        for cid in chunk_ids:
            # Create random normalized 768-dim vector
            vec = np.random.randn(768).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            cur.execute("""
                UPDATE "NewsChunks"
                SET "EmbeddingVector" = %s::vector
                WHERE "Id" = %s;
            """, (vec_str, cid))
        print(f"  -> Populated {len(chunk_ids)} chunks with 768-dim EmbeddingVector.")

    # 2. Query with random query vector using HNSW index (<=> cosine distance)
    qvec = np.random.randn(768).astype(np.float32)
    qvec = qvec / np.linalg.norm(qvec)
    qvec_str = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"

    # EXPLAIN ANALYZE to verify index scan
    cur.execute(f"""
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT "Id", "Text", "EmbeddingVector" <=> '{qvec_str}'::vector AS cosine_distance
        FROM "NewsChunks"
        WHERE "EmbeddingVector" IS NOT NULL
        ORDER BY "EmbeddingVector" <=> '{qvec_str}'::vector ASC
        LIMIT 5;
    """)
    plan = cur.fetchall()
    print("\n[*] PostgreSQL Query Execution Plan:")
    for p in plan:
        print(" ", p[0])

    # Benchmark query speed
    t0 = time.perf_counter()
    for _ in range(20):
        cur.execute(f"""
            SELECT "Id", "Text", "EmbeddingVector" <=> '{qvec_str}'::vector AS cosine_distance
            FROM "NewsChunks"
            WHERE "EmbeddingVector" IS NOT NULL
            ORDER BY "EmbeddingVector" <=> '{qvec_str}'::vector ASC
            LIMIT 5;
        """)
        results = cur.fetchall()
    elapsed_ms = (time.perf_counter() - t0) * 1000 / 20.0
    print(f"\n[+] Average Query Latency (HNSW Cosine): {elapsed_ms:.2f} ms")
    print(f"[+] Top nearest neighbor cosine distance: {results[0][2]:.4f}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    test_hnsw_performance()
