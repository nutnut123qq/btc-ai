import db_config

def test_sentiment_database():
    conn = db_config.get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT "Id", "Symbol", "FearGreedScore", "NewsSentimentScore",
               "AggregatedSentiment", "SentimentLabel", "CreatedAtUtc"
        FROM "SentimentSnapshots"
        ORDER BY "Id" DESC
        LIMIT 6;
    """)
    rows = cur.fetchall()
    assert len(rows) > 0, "No records in SentimentSnapshots"
    print(f"[PASS] Successfully verified {len(rows)} SentimentSnapshot records in PostgreSQL:")
    for r in rows:
        print(f"  ID={r[0]} | Symbol={r[1]} | FNG={r[2]} | News={r[3]:+.4f} | Aggregated={r[4]:+.2f} | Label={r[5]}")
    cur.close()
    conn.close()

if __name__ == "__main__":
    test_sentiment_database()
