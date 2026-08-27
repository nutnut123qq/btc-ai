import json
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

Epsilon = 1e-7

def detect_trend_at(closes, end_idx):
    if end_idx < 0 or end_idx >= len(closes):
        return "Sideways"
    start = max(0, end_idx - 5)
    if end_idx - start < 3:
        return "Sideways"
    
    up = 0
    down = 0
    for i in range(start + 1, end_idx + 1):
        if closes[i] > closes[i - 1]:
            up += 1
        elif closes[i] < closes[i - 1]:
            down += 1
            
    if up >= 4 and down <= 1:
        return "Uptrend"
    if down >= 4 and up <= 1:
        return "Downtrend"
    return "Sideways"

def recognize_single(o, h, l, c, trend):
    c_range = h - l
    if c_range <= Epsilon:
        return "Doji"
    
    body = abs(c - o)
    is_green = c >= o
    is_red = c < o
    upper = (h - c) if is_green else (h - o)
    lower = (o - l) if is_green else (c - l)
    
    body_ratio = body / c_range
    upper_ratio = upper / c_range
    lower_ratio = lower / c_range
    
    # Doji
    if body_ratio < 0.015:
        if lower_ratio > 0.6 and upper_ratio < 0.1:
            return "DragonflyDoji"
        if upper_ratio > 0.6 and lower_ratio < 0.1:
            return "GravestoneDoji"
        return "Doji"
        
    # Marubozu
    if body_ratio > 0.95:
        return "BullishMarubozu" if is_green else "BearishMarubozu"
        
    # Spinning Top
    if body_ratio < 0.3 and upper_ratio > 0.25 and lower_ratio > 0.25:
        return "SpinningTop"
        
    # Hammer / Hanging Man
    if lower >= body * 2.0 and upper_ratio < 0.15:
        if trend == "Downtrend":
            return "Hammer"
        if trend == "Uptrend":
            return "HangingMan"
            
    # Inverted Hammer / Shooting Star
    if upper >= body * 2.0 and lower_ratio < 0.15:
        if trend == "Downtrend":
            return "InvertedHammer"
        if trend == "Uptrend":
            return "ShootingStar"
            
    return None

def recognize_double(o1, h1, l1, c1, o2, h2, l2, c2, trend):
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    range1 = h1 - l1
    is_green1 = c1 >= o1
    is_red1 = c1 < o1
    is_green2 = c2 >= o2
    is_red2 = c2 < o2
    
    if body1 <= Epsilon or body2 <= Epsilon or range1 <= Epsilon:
        return None
        
    # Engulfing
    if is_red1 and is_green2:
        if o2 <= c1 and c2 >= o1 and body2 > body1:
            return "BullishEngulfing"
    if is_green1 and is_red2:
        if o2 >= c1 and c2 <= o1 and body2 > body1:
            return "BearishEngulfing"
            
    # Piercing Line
    if is_red1 and is_green2 and trend == "Downtrend":
        mid1 = (o1 + c1) / 2.0
        if o2 < c1 and c2 > mid1 and c2 < o1:
            return "PiercingLine"
            
    # Dark Cloud Cover
    if is_green1 and is_red2 and trend == "Uptrend":
        mid1 = (o1 + c1) / 2.0
        if o2 > c1 and c2 < mid1 and c2 > o1:
            return "DarkCloudCover"
            
    # Harami
    if is_red1 and is_green2 and body2 < body1 * 0.6:
        if o2 <= o1 and c2 >= c1:
            return "BullishHarami"
    if is_green1 and is_red2 and body2 < body1 * 0.6:
        if o2 >= o1 and c2 <= c1:
            return "BearishHarami"
            
    # Tweezer
    if trend == "Downtrend" and is_red1 and is_green2 and abs(l1 - l2) / range1 < 0.08:
        return "TweezerBottoms"
    if trend == "Uptrend" and is_green1 and is_red2 and abs(h1 - h2) / range1 < 0.08:
        return "TweezerTops"
        
    return None

def recognize_triple(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3, trend):
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    body3 = abs(c3 - o3)
    range1 = h1 - l1
    range2 = h2 - l2
    is_green1 = c1 >= o1
    is_red1 = c1 < o1
    is_green2 = c2 >= o2
    is_red2 = c2 < o2
    is_green3 = c3 >= o3
    is_red3 = c3 < o3
    
    if range1 <= Epsilon or range2 <= Epsilon:
        return None
        
    # Morning Star
    if is_red1 and is_green3 and trend == "Downtrend":
        if body1 > range1 * 0.5 and body2 < range2 * 0.3 and body3 > body1 * 0.5:
            mid1 = (o1 + c1) / 2.0
            if c3 > mid1:
                return "MorningStar"
                
    # Evening Star
    if is_green1 and is_red3 and trend == "Uptrend":
        if body1 > range1 * 0.5 and body2 < range2 * 0.3 and body3 > body1 * 0.5:
            mid1 = (o1 + c1) / 2.0
            if c3 < mid1:
                return "EveningStar"
                
    # Three White Soldiers
    if is_green1 and is_green2 and is_green3:
        u1 = h1 - c1
        u2 = h2 - c2
        u3 = h3 - c3
        if (body1 > range1 * 0.5 and body2 > range1 * 0.5 and body3 > range1 * 0.5 and
            o2 > o1 and c2 > c1 and o3 > o2 and c3 > c2 and
            u1 < body1 * 0.15 and u2 < body2 * 0.15 and u3 < body3 * 0.15):
            return "ThreeWhiteSoldiers"
            
    # Three Black Crows
    if is_red1 and is_red2 and is_red3:
        l_s1 = c1 - l1
        l_s2 = c2 - l2
        l_s3 = c3 - l3
        if (body1 > range1 * 0.5 and body2 > range1 * 0.5 and body3 > range1 * 0.5 and
            o2 < o1 and c2 < c1 and o3 < o2 and c3 < c2 and
            l_s1 < body1 * 0.15 and l_s2 < body2 * 0.15 and l_s3 < body3 * 0.15):
            return "ThreeBlackCrows"
            
    # Three Inside Up
    if is_red1 and is_green2 and is_green3 and trend == "Downtrend":
        if body2 < body1 * 0.6 and o2 >= c1 and c2 <= o1 and c3 > h1:
            return "ThreeInsideUp"
            
    # Three Inside Down
    if is_green1 and is_red2 and is_red3 and trend == "Uptrend":
        if body2 < body1 * 0.6 and o2 <= c1 and c2 >= o1 and c3 < l1:
            return "ThreeInsideDown"
            
    return None

def process_patterns_and_sequences(conn, symbol, timeframe):
    print(f"\n--- [{symbol} - {timeframe}] Recognizing Candle Patterns & Sequences ---", flush=True)
    t0 = time.time()
    cur = conn.cursor()

    cur.execute("""
        SELECT "OpenTimeMs", "Open", "High", "Low", "Close", "Volume"
        FROM "Klines"
        WHERE "Symbol" = %s AND "Timeframe" = %s
        ORDER BY "OpenTimeMs" ASC;
    """, (symbol, timeframe))
    rows = cur.fetchall()
    n = len(rows)
    if n < 5:
        print("  > Not enough klines. Skipping.")
        cur.close()
        return

    print(f"  > Loaded {n:,} klines in {time.time() - t0:.2f}s. Recognizing patterns...", flush=True)
    
    times = [r[0] for r in rows]
    opens = [float(r[1]) for r in rows]
    highs = [float(r[2]) for r in rows]
    lows = [float(r[3]) for r in rows]
    closes = [float(r[4]) for r in rows]
    vols = [float(r[5]) for r in rows]

    now = datetime.now(timezone.utc)
    patterns = []
    
    # 1. Single Patterns
    for i in range(n):
        trend = detect_trend_at(closes, i)
        sp = recognize_single(opens[i], highs[i], lows[i], closes[i], trend)
        if sp:
            patterns.append((
                symbol, timeframe, times[i], opens[i], highs[i], lows[i], closes[i], vols[i],
                sp, "Single", trend, now
            ))

    # 2. Double Patterns
    for i in range(n - 1):
        trend = detect_trend_at(closes, i)
        dp = recognize_double(opens[i], highs[i], lows[i], closes[i],
                              opens[i+1], highs[i+1], lows[i+1], closes[i+1], trend)
        if dp:
            patterns.append((
                symbol, timeframe, times[i], opens[i], highs[i], lows[i], closes[i], vols[i],
                dp, "Double", trend, now
            ))

    # 3. Triple Patterns
    for i in range(n - 2):
        trend = detect_trend_at(closes, i)
        tp = recognize_triple(opens[i], highs[i], lows[i], closes[i],
                              opens[i+1], highs[i+1], lows[i+1], closes[i+1],
                              opens[i+2], highs[i+2], lows[i+2], closes[i+2], trend)
        if tp:
            patterns.append((
                symbol, timeframe, times[i], opens[i], highs[i], lows[i], closes[i], vols[i],
                tp, "Triple", trend, now
            ))

    print(f"  > Recognized {len(patterns):,} patterns in {time.time() - t0:.2f}s. Inserting...", flush=True)

    # Insert into CandlePatterns
    insert_pattern_sql = """
        INSERT INTO "CandlePatterns" (
            "Symbol", "Timeframe", "OpenTimeMs", "Open", "High", "Low", "Close", "Volume",
            "PatternType", "PatternCategory", "TrendDirection", "CreatedAtUtc"
        ) VALUES %s
        ON CONFLICT DO NOTHING;
    """
    
    batch_size = 10000
    for i in range(0, len(patterns), batch_size):
        chunk = patterns[i:i + batch_size]
        execute_values(cur, insert_pattern_sql, chunk, page_size=5000)
        conn.commit()

    # 4. Pattern Sequences (sliding window 3, 4, 5)
    sorted_patterns = sorted(patterns, key=lambda x: x[2])
    n_p = len(sorted_patterns)
    sequences = []
    for ws in [3, 4, 5]:
        if n_p >= ws:
            for i in range(n_p - ws + 1):
                start_t = sorted_patterns[i][2]
                end_t = sorted_patterns[i + ws - 1][2]
                chain = [sorted_patterns[i + j][8] for j in range(ws)]
                sequences.append((
                    symbol, timeframe, start_t, end_t, ws,
                    json.dumps(chain), 1
                ))

    print(f"  > Building {len(sequences):,} pattern sequence n-grams...", flush=True)
    insert_seq_sql = """
        INSERT INTO "PatternSequences" (
            "Symbol", "Timeframe", "StartTimeMs", "EndTimeMs", "WindowSize",
            "PatternChainJson", "Count"
        ) VALUES %s
        ON CONFLICT DO NOTHING;
    """

    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i:i + batch_size]
        execute_values(cur, insert_seq_sql, chunk, page_size=5000)
        conn.commit()

    cur.close()
    print(f"  > [{symbol} - {timeframe}] Completed in {time.time() - t0:.2f}s ({len(patterns):,} patterns, {len(sequences):,} sequences).", flush=True)

def main():
    conn = get_db_connection()
    tfs = ["1d", "4h", "1h", "30m", "15m", "5m"]
    for sym in ["ETHUSDT", "SOLUSDT"]:
        for tf in tfs:
            process_patterns_and_sequences(conn, sym, tf)
    conn.close()
    print("\nAll Candle Patterns & Sequences completed!")

if __name__ == "__main__":
    main()
