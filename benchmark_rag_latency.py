import time
import numpy as np
import db_config

def benchmark_rag_vector_query():
    conn = db_config.get_db_connection()
    cur = conn.cursor()
    
    # Generate query vector
    qvec = np.random.randn(768).astype(np.float32)
    qvec = qvec / np.linalg.norm(qvec)
    qvec_str = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
    topK = 8
    
    sql = f"""
        SELECT "Id", "ArticleId", "Text" AS "Content", 1.0 - ("EmbeddingVector" <=> '{qvec_str}'::vector) as "Similarity"
        FROM "NewsChunks"
        WHERE "EmbeddingVector" IS NOT NULL
        ORDER BY "EmbeddingVector" <=> '{qvec_str}'::vector
        LIMIT {topK};
    """
    
    # Warmup
    for _ in range(5):
        cur.execute(sql)
        _ = cur.fetchall()
        
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        cur.execute(sql)
        rows = cur.fetchall()
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        
    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    p95_lat = np.percentile(latencies, 95)
    
    print(f"[*] Benchmark Results on NewsRagService SQL Vector Query:")
    print(f"  - Target Constraint: < 10.0 ms")
    print(f"  - Average Latency:   {avg_lat:.2f} ms")
    print(f"  - Minimum Latency:   {min_lat:.2f} ms")
    print(f"  - P95 Latency:       {p95_lat:.2f} ms")
    print(f"  - Returned Chunks:   {len(rows)} items")
    print(f"  - Status:            {'[PASS - ULTRA FAST]' if p95_lat < 10.0 else '[FAIL]'}")
    
    print("\nSample top-3 retrieved items:")
    for idx, r in enumerate(rows[:3], 1):
        print(f"  [{idx}] ID={r[0]} | Similarity={r[3]:.4f} | Preview={r[2][:60]}...")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    benchmark_rag_vector_query()
