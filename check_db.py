import psycopg2
from datetime import datetime

conn = psycopg2.connect(host='localhost', port=5432, database='bitcoin_analyst', user='postgres', password='123456')
cur = conn.cursor()

# Đếm tổng
cur.execute('SELECT COUNT(*) FROM "CandlePatterns"')
total = cur.fetchone()[0]
print(f'Total rows: {total}')

# Trend distribution
cur.execute('SELECT "TrendDirection", COUNT(*) FROM "CandlePatterns" GROUP BY "TrendDirection" ORDER BY COUNT(*) DESC')
print('\nTrend distribution:')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]}')

# Category distribution
cur.execute('SELECT "PatternCategory", COUNT(*) FROM "CandlePatterns" GROUP BY "PatternCategory" ORDER BY COUNT(*) DESC')
print('\nCategory distribution:')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]}')

# Top patterns
cur.execute('SELECT "PatternType", COUNT(*) FROM "CandlePatterns" GROUP BY "PatternType" ORDER BY COUNT(*) DESC LIMIT 10')
print('\nTop patterns:')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]}')

# Sample 5 rows
print('\nSample rows (newest):')
cur.execute('SELECT "PatternType", "PatternCategory", "TrendDirection", "OpenTimeMs" FROM "CandlePatterns" ORDER BY "OpenTimeMs" DESC LIMIT 5')
for row in cur.fetchall():
    ts = datetime.utcfromtimestamp(row[3]/1000).strftime('%Y-%m-%d %H:%M')
    print(f'  {row[0]} | {row[1]} | {row[2]} | {ts} UTC')

# Sample 5 oldest rows
print('\nSample rows (oldest):')
cur.execute('SELECT "PatternType", "PatternCategory", "TrendDirection", "OpenTimeMs" FROM "CandlePatterns" ORDER BY "OpenTimeMs" ASC LIMIT 5')
for row in cur.fetchall():
    ts = datetime.utcfromtimestamp(row[3]/1000).strftime('%Y-%m-%d %H:%M')
    print(f'  {row[0]} | {row[1]} | {row[2]} | {ts} UTC')

cur.close()
conn.close()
