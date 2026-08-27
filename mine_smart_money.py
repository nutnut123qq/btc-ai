import sys
import time
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values
from db_config import get_db_connection

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

TIMEFRAMES = ["1d", "4h", "1h", "15m", "5m"]
SYMBOLS = ["ETHUSDT", "SOLUSDT"]

def mine_smc_for_symbol_tf(conn, symbol, timeframe, lookback=20000):
    cur = conn.cursor()
    cur.execute("""
        SELECT "OpenTimeMs", "Open", "High", "Low", "Close", "Volume"
        FROM "Klines"
        WHERE "Symbol" = %s AND "Timeframe" = %s
        ORDER BY "OpenTimeMs" DESC
        LIMIT %s;
    """, (symbol, timeframe, lookback))
    raw = cur.fetchall()
    if len(raw) < 5:
        cur.close()
        return 0

    ordered = sorted(raw, key=lambda x: x[0])
    n = len(ordered)
    now = datetime.now(timezone.utc)

    structures = []
    swing_highs = []
    swing_lows = []

    # 1. 5-bar Pivot Swings
    for i in range(2, n - 2):
        k = ordered[i]
        is_sh = (k[2] > ordered[i - 1][2] and k[2] > ordered[i - 2][2] and
                 k[2] > ordered[i + 1][2] and k[2] > ordered[i + 2][2])
        is_sl = (k[3] < ordered[i - 1][3] and k[3] < ordered[i - 2][3] and
                 k[3] < ordered[i + 1][3] and k[3] < ordered[i + 2][3])

        if is_sh:
            swing_highs.append((i, float(k[2]), int(k[0])))
            structures.append((
                symbol, timeframe, int(k[0]), "SWING_HIGH",
                float(k[2]), None, None, False, "Swing High", now
            ))

        if is_sl:
            swing_lows.append((i, float(k[3]), int(k[0])))
            structures.append((
                symbol, timeframe, int(k[0]), "SWING_LOW",
                float(k[3]), None, None, False, "Swing Low", now
            ))

    # 2. BOS & CHoCH
    current_trend = 0
    last_sh = -1.0
    last_sl = -1.0

    for i in range(n):
        k = ordered[i]
        sh = [x for x in swing_highs if x[0] <= i]
        if sh:
            last_sh = sh[-1][1]
        sl = [x for x in swing_lows if x[0] <= i]
        if sl:
            last_sl = sl[-1][1]

        c = float(k[4])
        t_ms = int(k[0])

        if last_sh > 0 and c > last_sh:
            if current_trend == 1:
                structures.append((
                    symbol, timeframe, t_ms, "BOS_BULL",
                    c, None, None, False, "Bullish BOS", now
                ))
                last_sh = -1.0
            elif current_trend in (-1, 0):
                structures.append((
                    symbol, timeframe, t_ms, "CHOCH_BULL",
                    c, None, None, False, "Bullish CHOCH", now
                ))
                current_trend = 1
                last_sh = -1.0

        if last_sl > 0 and c < last_sl:
            if current_trend == -1:
                structures.append((
                    symbol, timeframe, t_ms, "BOS_BEAR",
                    c, None, None, False, "Bearish BOS", now
                ))
                last_sl = -1.0
            elif current_trend in (1, 0):
                structures.append((
                    symbol, timeframe, t_ms, "CHOCH_BEAR",
                    c, None, None, False, "Bearish CHOCH", now
                ))
                current_trend = -1
                last_sl = -1.0

    # 3. FVG
    for i in range(2, n):
        k0 = ordered[i - 2]
        k1 = ordered[i - 1]
        k2 = ordered[i]

        k0_h = float(k0[2])
        k0_l = float(k0[3])
        k2_h = float(k2[2])
        k2_l = float(k2[3])
        k1_t = int(k1[0])

        # Bullish FVG
        if k0_h < k2_l:
            low_p = k0_h
            high_p = k2_l
            mid_p = (k0_h + k2_l) / 2.0
            is_mit = any(float(ordered[j][3]) <= low_p for j in range(i + 1, min(i + 100, n)))
            structures.append((
                symbol, timeframe, k1_t, "FVG_BULL",
                mid_p, high_p, low_p, is_mit, "Bullish FVG", now
            ))

        # Bearish FVG
        if k0_l > k2_h:
            low_p = k2_h
            high_p = k0_l
            mid_p = (k0_l + k2_h) / 2.0
            is_mit = any(float(ordered[j][2]) >= high_p for j in range(i + 1, min(i + 100, n)))
            structures.append((
                symbol, timeframe, k1_t, "FVG_BEAR",
                mid_p, high_p, low_p, is_mit, "Bearish FVG", now
            ))

    # Clean old structures for this symbol/tf
    cur.execute('DELETE FROM "SmartMoneyStructures" WHERE "Symbol"=%s AND "Timeframe"=%s;', (symbol, timeframe))

    insert_sql = """
        INSERT INTO "SmartMoneyStructures" (
            "Symbol", "Timeframe", "TimeMs", "EventType",
            "Price", "HighPrice", "LowPrice", "IsMitigated", "Description", "CreatedAtUtc"
        ) VALUES %s;
    """
    execute_values(cur, insert_sql, structures, page_size=2000)
    conn.commit()
    cur.close()
    print(f"  > [{symbol} - {timeframe}] Saved {len(structures):,} SMC structures.")
    return len(structures)

def main():
    conn = get_db_connection()
    total = 0
    for sym in SYMBOLS:
        print(f"\nMining Smart Money Concepts for {sym}...")
        for tf in TIMEFRAMES:
            total += mine_smc_for_symbol_tf(conn, sym, tf)
    conn.close()
    print(f"\nAll SMC structures mined! Total: {total:,} structures.")

if __name__ == "__main__":
    main()
