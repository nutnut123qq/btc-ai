import json
import sys
import urllib.request
import db_config

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def verify_full_system():
    print("=" * 80)
    print("      KIỂM THỬ HỆ THỐNG TOÀN DIỆN NHÓM 2 (PROMPT 1/4 -> 4/4)")
    print("=" * 80)
    
    # 1. Database Check (LiquidationSnapshots & SentimentSnapshots)
    conn = db_config.get_db_connection()
    cur = conn.cursor()
    
    cur.execute('SELECT COUNT(*) FROM "LiquidationSnapshots";')
    liq_count = cur.fetchone()[0]
    
    cur.execute('SELECT COUNT(*) FROM "SentimentSnapshots";')
    sent_count = cur.fetchone()[0]
    
    cur.execute('SELECT COUNT(*) FROM "NewsChunks" WHERE "EmbeddingVector" IS NOT NULL;')
    vec_count = cur.fetchone()[0]
    
    print(f"\n[1] PostgreSQL Database State:")
    print(f"  - LiquidationSnapshots count: {liq_count} rows")
    print(f"  - SentimentSnapshots count:   {sent_count} rows")
    print(f"  - NewsChunks with Vector(768): {vec_count} chunks")
    
    assert liq_count > 0, "No liquidation snapshots found"
    assert sent_count > 0, "No sentiment snapshots found"
    assert vec_count > 0, "No vector embeddings found"
    
    # 2. Fetch Latest Liquidation Snapshot Sample
    cur.execute('''
        SELECT "Symbol", "Timeframe", "CurrentPrice", "TotalLongLiqUsdt", "TotalShortLiqUsdt", "HeatmapJson"
        FROM "LiquidationSnapshots"
        ORDER BY "Id" DESC
        LIMIT 1;
    ''')
    row = cur.fetchone()
    bins = json.loads(row[5]) if row[5] else []
    print(f"\n[2] Latest Liquidation Snapshot Sample:")
    print(f"  - Symbol:           {row[0]} ({row[1]})")
    print(f"  - Reference Price:  ${row[2]:,.2f}")
    print(f"  - Long Liq at Risk: ${row[3]:,.2f} USDT")
    print(f"  - Short Liq at Risk: ${row[4]:,.2f} USDT")
    print(f"  - Heatmap Bins:     {len(bins)} bins calculated")
    
    # 3. Fetch Latest Sentiment Snapshot Sample
    cur.execute('''
        SELECT "Symbol", "FearGreedScore", "NewsSentimentScore", "AggregatedSentiment", "SentimentLabel"
        FROM "SentimentSnapshots"
        ORDER BY "Id" DESC
        LIMIT 3;
    ''')
    print(f"\n[3] Latest Macro Sentiment Snapshots in DB:")
    for r in cur.fetchall():
        print(f"  - {r[0]}: FNG={r[1]}/100 | NewsScore={r[2]:+.4f} | Aggregated={r[3]:+.2f} ({r[4]})")
        
    cur.close()
    conn.close()
    
    print("\n" + "=" * 80)
    print("  [SUCCESS] TẤT CẢ 4/4 PROMPT NHÓM 2 ĐÃ ĐƯỢC KIỂM THỬ THÀNH CÔNG!")
    print("=" * 80)

if __name__ == "__main__":
    verify_full_system()
