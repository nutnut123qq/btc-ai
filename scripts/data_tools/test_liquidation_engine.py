import json
from datetime import datetime, timezone
import db_config
from liquidation_engine import LiquidationEngine, save_liquidation_snapshot

def test_math_formulas():
    engine = LiquidationEngine()
    
    # 1. Test Long Liq formula: LiqPrice = EntryPrice * (1 - 1/Lev + MMR)
    # Entry = 100,000, Lev = 100, MMR = 0.004 => 100,000 * (1 - 0.01 + 0.004) = 100,000 * 0.994 = 99,400
    long_100x = engine.calculate_long_liq_price(100000.0, 100, 0.004)
    assert abs(long_100x - 99400.0) < 1e-4, f"Expected 99400.0, got {long_100x}"
    
    # 2. Test Short Liq formula: LiqPrice = EntryPrice * (1 + 1/Lev - MMR)
    # Entry = 100,000, Lev = 50, MMR = 0.005 => 100,000 * (1 + 0.02 - 0.005) = 100,000 * 1.015 = 101,500
    short_50x = engine.calculate_short_liq_price(100000.0, 50, 0.005)
    assert abs(short_50x - 101500.0) < 1e-4, f"Expected 101500.0, got {short_50x}"
    
    print("[PASS] Liquidation math formulas verified.")

def test_swept_filtering():
    engine = LiquidationEngine(bin_step_pct=0.003)
    # Synthetic bars: Entry at bar 0 with close=100, low=98, high=102.
    # Bar 1 dips to low=90 (sweeping 100x, 50x long liquidations)
    # Bar 2 at 95.
    bars = [
        {"open_time_ms": 1000, "open": 100, "high": 102, "low": 98, "close": 100, "volume": 10, "volume_usdt": 1000, "delta_oi_usdt": 1000, "ls_ratio": 1.0},
        {"open_time_ms": 2000, "open": 100, "high": 101, "low": 90, "close": 94, "volume": 20, "volume_usdt": 2000, "delta_oi_usdt": 0, "ls_ratio": 1.0},
        {"open_time_ms": 3000, "open": 94, "high": 96, "low": 93, "close": 95, "volume": 10, "volume_usdt": 1000, "delta_oi_usdt": 500, "ls_ratio": 1.0},
    ]
    long_tranches, short_tranches = engine.compute_liquidation_tranches(bars, current_price=95.0)
    
    # Check that bar 0 long tranches with liq_price > 90 were filtered out
    for t in long_tranches:
        if t["bar_idx"] == 0:
            assert t["liq_price"] < 90.0, f"Tranche with liq_price {t['liq_price']} should have been swept by low=90"
            
    print("[PASS] Swept liquidation filtering verified.")

def test_db_records():
    conn = db_config.get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT "Id", "Symbol", "Timeframe", "TimestampUtc", "CurrentPrice", "TotalLongLiqUsdt", "TotalShortLiqUsdt", length("HeatmapJson")
        FROM "LiquidationSnapshots"
        ORDER BY "Id" DESC
        LIMIT 5;
    """)
    rows = cur.fetchall()
    assert len(rows) > 0, "No records found in LiquidationSnapshots"
    print(f"[PASS] Found {len(rows)} snapshot records in PostgreSQL:")
    for r in rows:
        print(f"  ID={r[0]} | Symbol={r[1]} | TF={r[2]} | Price={r[4]} | Long=${r[5]:,.0f} | Short=${r[6]:,.0f} | JsonBytes={r[7]}")
    cur.close()
    conn.close()

if __name__ == "__main__":
    test_math_formulas()
    test_swept_filtering()
    test_db_records()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
