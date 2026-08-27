import numpy as np
import pandas as pd
from db_config import get_db_connection

def test_formula_parity():
    conn = get_db_connection()
    cur = conn.cursor()

    # Load 500 BTCUSDT 1h klines
    cur.execute("""
        SELECT "OpenTimeMs", "Open", "High", "Low", "Close", "Volume"
        FROM "Klines"
        WHERE "Symbol" = 'BTCUSDT' AND "Timeframe" = '1h'
        ORDER BY "OpenTimeMs" ASC
        LIMIT 500;
    """)
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=['OpenTimeMs', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)

    # Load existing TechnicalIndicators from DB for the same 500 bars
    cur.execute("""
        SELECT "OpenTimeMs", "Rsi14", "Ema12", "Ema26", "Ema50", "Ema200", 
               "Sma50", "Sma200", "Macd", "MacdSignal", "MacdHistogram", 
               "BollingerUpper", "BollingerMiddle", "BollingerLower", 
               "Atr14", "Obv", "ObvEma50", "Vwap", "RollingVwap24",
               "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm"
        FROM "TechnicalIndicators"
        WHERE "Symbol" = 'BTCUSDT' AND "Timeframe" = '1h'
        ORDER BY "OpenTimeMs" ASC
        LIMIT 500;
    """)
    ti_rows = cur.fetchall()
    ti_cols = ["OpenTimeMs", "Rsi14", "Ema12", "Ema26", "Ema50", "Ema200", 
               "Sma50", "Sma200", "Macd", "MacdSignal", "MacdHistogram", 
               "BollingerUpper", "BollingerMiddle", "BollingerLower", 
               "Atr14", "Obv", "ObvEma50", "Vwap", "RollingVwap24",
               "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm"]
    db_ti = pd.DataFrame(ti_rows, columns=ti_cols)

    print("Klines count:", len(df), "DB TI count:", len(db_ti))
    print("\nSample DB row 250:")
    print(db_ti.iloc[250].to_dict())

    cur.close()
    conn.close()

if __name__ == "__main__":
    test_formula_parity()
