import os
import sys
import gc
import math
import argparse
import time
from datetime import datetime, timezone
from collections import defaultdict

import psycopg2
import psycopg2.extras
import numpy as np

# --- Config ------------------------------------------------------------------
from db_config import get_db_connection

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")

def get_connection():
    return get_db_connection()

def get_interval_ms(timeframe):
    unit = timeframe[-1]
    try:
        val = int(timeframe[:-1])
    except ValueError:
        val = 1
    if unit == 'm':
        return val * 60 * 1000
    elif unit == 'h':
        return val * 60 * 60 * 1000
    elif unit == 'd':
        return val * 24 * 60 * 60 * 1000
    return 60 * 60 * 1000

def fetch_occurrences(conn, symbol, timeframe, window_size):
    cursor_name = f"occ_cursor_{window_size}"
    with conn.cursor(name=cursor_name) as cur:
        cur.itersize = 20000
        cur.execute("""
            SELECT "ArchetypeId", "WindowStartMs", "WindowEndMs", "Label", "TargetReturn"
            FROM "ArchetypeOccurrences"
            WHERE "Symbol" = %s AND "Timeframe" = %s AND "WindowSize" = %s
            ORDER BY "WindowStartMs"
        """, (symbol, timeframe, window_size))
        return cur.fetchall()

def process_combo(conn, symbol, timeframe, window_size, batch_size):
    print(f"[{timeframe}/ws={window_size}] Loading occurrences...")
    occurrences = fetch_occurrences(conn, symbol, timeframe, window_size)
    if not occurrences:
        print(f"[{timeframe}/ws={window_size}] No occurrences found.")
        return
        
    print(f"[{timeframe}/ws={window_size}] Loaded {len(occurrences):,} occurrences...")
    
    interval_ms = get_interval_ms(timeframe)
    
    transitions = defaultdict(lambda: {"count": 0, "returns": [], "gap_bars": []})
    sequences = defaultdict(lambda: {"up": 0, "down": 0, "sideways": 0, "returns": []})
    archetypes = set()
    
    # max gap condition: gap between End(i) and Start(i+1) < 5 * interval_ms
    max_gap_ms = 5 * interval_ms
    
    for i in range(len(occurrences) - 1):
        arch_id, start, end, label, ret = occurrences[i]
        next_arch_id, next_start, next_end, next_label, next_ret = occurrences[i+1]
        
        archetypes.add(arch_id)
        archetypes.add(next_arch_id)
        
        # Check gap between End(i) and Start(i+1)
        gap_ms = next_start - end
        # Also ensure time is moving forward
        shift_ms = next_start - start
        
        if gap_ms < max_gap_ms and shift_ms > 0:
            gap_bars = shift_ms / interval_ms
            trans = transitions[(arch_id, next_arch_id)]
            trans["count"] += 1
            trans["returns"].append(float(next_ret) if next_ret is not None else 0.0)
            trans["gap_bars"].append(gap_bars)
            trans["last_seen"] = next_end
            
            # sequence logic (i, i+1, i+2)
            if i < len(occurrences) - 2:
                third_arch_id, third_start, third_end, third_label, third_ret = occurrences[i+2]
                gap_ms_2 = third_start - next_end
                shift_ms_2 = third_start - next_start
                if gap_ms_2 < max_gap_ms and shift_ms_2 > 0:
                    seq = sequences[(arch_id, next_arch_id, third_arch_id)]
                    if third_label == 1:
                        seq["up"] += 1
                    elif third_label == -1:
                        seq["down"] += 1
                    else:
                        seq["sideways"] += 1
                    seq["returns"].append(float(third_ret) if third_ret is not None else 0.0)

    out_totals = defaultdict(int)
    for (from_id, to_id), stats in transitions.items():
        out_totals[from_id] += stats["count"]
        
    entropies = []
    transition_records = []
    now = datetime.now(timezone.utc)
    
    for (from_id, to_id), stats in transitions.items():
        prob = stats["count"] / out_totals[from_id]
        avg_ret = sum(stats["returns"]) / stats["count"]
        avg_bars = sum(stats["gap_bars"]) / stats["count"]
        
        transition_records.append((
            from_id, to_id, symbol, timeframe, window_size,
            stats["count"], prob, avg_ret, avg_bars, stats.get("last_seen", 0),
            1, now
        ))
        
    for from_id, total in out_totals.items():
        ent = 0
        for to_id in archetypes:
            if (from_id, to_id) in transitions:
                p = transitions[(from_id, to_id)]["count"] / total
                if p > 0:
                    ent -= p * math.log2(p)
        entropies.append(ent)
        
    if entropies:
        print(f"[{timeframe}/ws={window_size}] Entropy range: {min(entropies):.2f} — {max(entropies):.2f} bits (mean: {sum(entropies)/len(entropies):.2f})")
        
    print(f"[{timeframe}/ws={window_size}] Built {len(transition_records):,} first-order transitions from {len(out_totals):,} archetypes")
    
    sequence_records = []
    for (id1, id2, id3), stats in sequences.items():
        total = stats["up"] + stats["down"] + stats["sideways"]
        if total > 0:
            up_rate = stats["up"] / total
            down_rate = stats["down"] / total
            sideways_rate = stats["sideways"] / total
            avg_ret = sum(stats["returns"]) / total
            
            sequence_records.append((
                id1, id2, id3, symbol, timeframe, window_size,
                total, up_rate, down_rate, sideways_rate, avg_ret, now
            ))
            
    print(f"[{timeframe}/ws={window_size}] Built {len(sequence_records):,} second-order sequences")
    
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM "ArchetypeSequences" WHERE "Symbol" = %s AND "Timeframe" = %s AND "WindowSize" = %s;
        """, (symbol, timeframe, window_size))
        cur.execute("""
            DELETE FROM "ArchetypeTransitions" WHERE "Symbol" = %s AND "Timeframe" = %s AND "WindowSize" = %s;
        """, (symbol, timeframe, window_size))
        
        if transition_records:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO "ArchetypeTransitions"
                ("FromArchetypeId", "ToArchetypeId", "Symbol", "Timeframe", "WindowSize",
                 "TransitionCount", "TransitionProbability", "AvgReturnPct", "AvgBarsToTransition",
                 "LastSeenMs", "Version", "CreatedAtUtc")
                VALUES %s""",
                transition_records,
                page_size=batch_size
            )
            
        if sequence_records:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO "ArchetypeSequences"
                ("FirstArchetypeId", "SecondArchetypeId", "ThirdArchetypeId", "Symbol", "Timeframe", "WindowSize",
                 "OccurrenceCount", "OutcomeUpRate", "OutcomeDownRate", "OutcomeSidewaysRate", "AvgReturnPct", "CreatedAtUtc")
                VALUES %s""",
                sequence_records,
                page_size=batch_size
            )
            
    conn.commit()
    print(f"[{timeframe}/ws={window_size}] Inserted to DB OK")
    
    del occurrences, transitions, sequences, transition_records, sequence_records
    gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=SYMBOL, help=f"Trading symbol (default: {SYMBOL})")
    parser.add_argument("--timeframe", help="Specific timeframe")
    parser.add_argument("--window-size", type=int, help="Specific window size")
    parser.add_argument("--all", action="store_true", help="Process all timeframe and window size combos from CandleArchetypes")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for inserts")
    args = parser.parse_args()
    
    conn = get_connection()
    conn.autocommit = False
    
    symbol = args.symbol
    combos = []
    if args.all:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT "Timeframe", "WindowSize" 
                FROM "CandleArchetypes"
                WHERE "Symbol" = %s
            """, (symbol,))
            combos = cur.fetchall()
    else:
        if args.timeframe and args.window_size:
            combos = [(args.timeframe, args.window_size)]
        else:
            print("Please specify --timeframe and --window-size OR --all")
            return
            
    for tf, ws in combos:
        try:
            process_combo(conn, symbol, tf, ws, args.batch_size)
        except Exception as e:
            print(f"Error processing {tf} {ws}: {e}")
            conn.rollback()
            import traceback
            traceback.print_exc()
            
    conn.close()

if __name__ == "__main__":
    main()
