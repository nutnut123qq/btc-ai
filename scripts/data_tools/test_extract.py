import math
import psycopg2
from datetime import datetime, timezone

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'bitcoin_analyst',
    'user': 'postgres',
    'password': '123456'
}

FEATURE_COLS = [
    "CloseZscore", "ClosePctChange1", "ClosePctChange4", "ClosePctChange24",
    "HighLowRangePct", "BodyPct", "UpperWickPct", "LowerWickPct",
    "Rsi14", "Rsi14Slope", "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm",
    "Ema12Dist", "Ema26Dist", "Ema50Dist", "Ema200Dist", "Sma50Dist", "Sma200Dist",
    "BollingerWidth", "BollingerPosition", "Atr14Pct", "ObvEmaDist", "VwapDist",
    "RollingVwapDist", "VolumeZscore", "VolumeSma20Ratio", "TakerBuyRatio",
    "RecentPatternEncoded", "ActiveRuleCount",
]

def test_symbol(sym):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cols = ", ".join(f'"{c}"' for c in FEATURE_COLS)
    cur.execute(f'SELECT "OpenTimeMs", {cols} FROM "MlFeatureStores" WHERE "Symbol"=%s AND "Timeframe"=\'4h\' ORDER BY "OpenTimeMs" ASC', (sym,))
    rows = cur.fetchall()
    
    cur.execute('SELECT "OpenTimeMs", "Close" FROM "Klines" WHERE "Symbol"=%s AND "Timeframe"=\'4h\' ORDER BY "OpenTimeMs" ASC', (sym,))
    kline_rows = cur.fetchall()
    conn.close()
    
    close_dict = {int(r[0]): float(r[1]) for r in kline_rows}
    print(f"[{sym}] Loaded {len(rows)} feature rows and {len(kline_rows)} klines")
    
    valid_windows = 0
    ws = 5
    tf_ms = 14400000
    
    for i in range(ws - 1, len(rows)):
        window = rows[i - ws + 1 : i + 1]
        is_continuous = True
        for j in range(1, len(window)):
            if window[j][0] - window[j-1][0] != tf_ms:
                is_continuous = False
                break
        if not is_continuous:
            continue
            
        has_null = False
        for r in window:
            # check the first 28 features (excluding RecentPatternEncoded & ActiveRuleCount)
            vals = r[1:29]
            if any(v is None for v in vals):
                has_null = True
                break
        if has_null:
            continue
            
        end_open_ms = int(window[-1][0])
        next_open_ms = end_open_ms + tf_ms
        if end_open_ms in close_dict and next_open_ms in close_dict:
            valid_windows += 1
            
    print(f"[{sym}] Extracted valid windows: {valid_windows}")

test_symbol("ETHUSDT")
test_symbol("SOLUSDT")
