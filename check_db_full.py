import psycopg2
from datetime import datetime, timezone

conn = psycopg2.connect(host='localhost', port=5432, database='bitcoin_analyst', user='postgres', password='123456')
cur = conn.cursor()

def fmt_ms(ms):
    if ms is None:
        return 'N/A'
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

def table_count(name):
    try:
        cur.execute(f'SELECT COUNT(*) FROM "{name}"')
        return cur.fetchone()[0]
    except Exception as e:
        return f'ERR: {e}'

print('=== DATABASE OVERVIEW: bitcoin_analyst ===\n')

tables = [
    'Klines',
    'CandlePatterns',
    'WindowVectors',
    'TechnicalIndicators',
    'MarketMetrics',
    'NewsArticles',
    'NewsChunks',
    'CandleSequenceRules',
    'CandleSequenceSignals',
    'CandleVolumeStats',
    'MlFeatureStores',
    'PriceTargets',
    'PatternSequences',
    'WindowClassificationDatasets',
    'AppAlerts',
    'PriceAlertSettings',
]

for t in tables:
    print(f'{t}: {table_count(t):,}')

print('\n=== KLINES PER TIMEFRAME ===')
cur.execute('''
    SELECT "Timeframe", COUNT(*), MIN("OpenTimeMs"), MAX("OpenTimeMs")
    FROM "Klines"
    GROUP BY "Timeframe"
    ORDER BY COUNT(*) DESC
''')
for row in cur.fetchall():
    tf, cnt, mn, mx = row
    print(f'  {tf:>4}: {cnt:>7,} rows | {fmt_ms(mn)} -> {fmt_ms(mx)}')

print('\n=== CANDLE PATTERNS PER TIMEFRAME ===')
cur.execute('''
    SELECT "Timeframe", COUNT(*), MIN("OpenTimeMs"), MAX("OpenTimeMs")
    FROM "CandlePatterns"
    GROUP BY "Timeframe"
    ORDER BY COUNT(*) DESC
''')
for row in cur.fetchall():
    tf, cnt, mn, mx = row
    print(f'  {tf:>4}: {cnt:>7,} rows | {fmt_ms(mn)} -> {fmt_ms(mx)}')

print('\n=== CANDLE PATTERNS BY TYPE ===')
cur.execute('''
    SELECT "PatternType", COUNT(*)
    FROM "CandlePatterns"
    GROUP BY "PatternType"
    ORDER BY COUNT(*) DESC
''')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]:,}')

print('\n=== WINDOW VECTORS PER TIMEFRAME / FEATURE ===')
cur.execute('''
    SELECT "Timeframe", "FeatureType", COUNT(*)
    FROM "WindowVectors"
    GROUP BY "Timeframe", "FeatureType"
    ORDER BY "Timeframe", "FeatureType"
''')
for row in cur.fetchall():
    print(f'  {row[0]:>4} / {row[1]:<14}: {row[2]:,}')

print('\n=== TECHNICAL INDICATORS PER TIMEFRAME ===')
cur.execute('''
    SELECT "Timeframe", COUNT(*), MIN("OpenTimeMs"), MAX("OpenTimeMs")
    FROM "TechnicalIndicators"
    GROUP BY "Timeframe"
    ORDER BY COUNT(*) DESC
''')
for row in cur.fetchall():
    tf, cnt, mn, mx = row
    print(f'  {tf:>4}: {cnt:>7,} rows | {fmt_ms(mn)} -> {fmt_ms(mx)}')

print('\n=== MARKET METRICS ===')
cur.execute('''
    SELECT "Timeframe", COUNT(*), MIN("OpenTimeMs"), MAX("OpenTimeMs")
    FROM "MarketMetrics"
    GROUP BY "Timeframe"
    ORDER BY COUNT(*) DESC
''')
for row in cur.fetchall():
    tf, cnt, mn, mx = row
    print(f'  {tf:>4}: {cnt:>7,} rows | {fmt_ms(mn)} -> {fmt_ms(mx)}')

print('\n=== NEWS ===')
cur.execute('SELECT COUNT(*) FROM "NewsArticles"')
articles = cur.fetchone()[0]
cur.execute('SELECT COUNT(*) FROM "NewsChunks"')
chunks = cur.fetchone()[0]
cur.execute('SELECT MIN("PublishedAt"), MAX("PublishedAt") FROM "NewsArticles"')
mn, mx = cur.fetchone()
print(f'  Articles: {articles:,} | Chunks: {chunks:,}')
print(f'  Date range: {mn} -> {mx}')

print('\n=== RULES / SIGNALS / ALERTS ===')
print(f'  CandleSequenceRules: {table_count("CandleSequenceRules")}')
print(f'  CandleSequenceSignals: {table_count("CandleSequenceSignals")}')
print(f'  AppAlerts: {table_count("AppAlerts")}')

cur.close()
conn.close()
