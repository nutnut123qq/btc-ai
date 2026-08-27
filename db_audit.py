import os
import sys
from datetime import datetime, timezone

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import psycopg2
from db_config import get_db_connection

def format_ts(ms):
    if ms is None:
        return "N/A"
    try:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return str(ms)

def run_deep_audit():
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Total DB Size
    cur.execute("SELECT pg_size_pretty(pg_database_size(current_database())), pg_database_size(current_database());")
    db_size_pretty, db_size_bytes = cur.fetchone()

    # 2. All tables
    cur.execute("""
        SELECT 
            t.tablename,
            pg_total_relation_size('"' || t.tablename || '"') AS total_bytes,
            pg_size_pretty(pg_total_relation_size('"' || t.tablename || '"')) AS total_size,
            pg_relation_size('"' || t.tablename || '"') AS data_bytes,
            pg_size_pretty(pg_relation_size('"' || t.tablename || '"')) AS data_size,
            pg_indexes_size('"' || t.tablename || '"') AS index_bytes,
            pg_size_pretty(pg_indexes_size('"' || t.tablename || '"')) AS index_size
        FROM pg_tables t
        WHERE t.schemaname = 'public'
        ORDER BY total_bytes DESC;
    """)
    tables_meta = cur.fetchall()

    table_stats = []
    total_db_rows = 0
    for row in tables_meta:
        tname = row[0]
        cur.execute(f'SELECT COUNT(*) FROM "{tname}"')
        count = cur.fetchone()[0]
        total_db_rows += count
        table_stats.append({
            "name": tname,
            "total_bytes": row[1],
            "total_size": row[2],
            "data_size": row[4],
            "index_size": row[6],
            "rows": count
        })

    table_desc = {
        "WindowClassificationDatasets": "Dataset phân loại chuỗi nến cửa sổ (ML features)",
        "TechnicalIndicators": "Chỉ báo kỹ thuật (RSI, MACD, EMA, ATR, BB, VWAP...)",
        "PatternSequences": "Chuỗi mẫu nến liên tiếp",
        "MlFeatureStores": "Vector đặc trưng phục vụ Machine Learning per-bar",
        "Klines": "Nến giá OHLCV lịch sử từ Binance",
        "CandleVolumeStats": "Thống kê Volume Anomaly đa khung thời gian",
        "CandlePatterns": "Các mẫu nến đã nhận diện (Hammer, Engulfing...)",
        "WindowVectors": "Vector đặc trưng của cửa sổ nến (Pattern Search)",
        "PriceTargets": "Target nhãn giá tương lai cho ML training",
        "ArchetypeOccurrences": "Các lần xuất hiện của Archetype nến",
        "FuturesMetrics": "Dữ liệu phái sinh (OI, L/S Ratios, Taker Ratio, Funding)",
        "ArchetypeSequences": "Chuỗi Archetype nến liên tiếp",
        "ArchetypeTransitions": "Ma trận chuyển trạng thái Archetype",
        "CandleArchetypes": "Định nghĩa các nhóm Archetype nến",
        "EnsemblePredictionRecords": "Bản ghi dự đoán của mô hình Ensemble",
        "ArchetypeOutcomes": "Kết quả xác suất sau mỗi Archetype",
        "BacktestRuns": "Kết quả các lượt chạy backtest chiến lược",
        "MarketRegimes": "Phân loại chế độ thị trường (Bull, Bear, Sideways)",
        "NewsArticles": "Tin tức thị trường RSS đã thu thập",
        "NewsChunks": "Đoạn văn bản tin tức kèm vector embedding",
        "BacktestTrades": "Chi tiết từng lệnh giao dịch trong backtest",
        "MarketMetrics": "Chỉ số phái sinh thu thập định kỳ",
        "AppAlerts": "Thông báo & cảnh báo hệ thống",
        "CandleSequenceSignals": "Lịch sử tín hiệu kích hoạt quy tắc",
        "RegimeTransitions": "Chuyển tiếp giữa các chế độ thị trường",
        "SmartMoneyStructures": "Cấu trúc thị trường Smart Money (BOS, CHoCH)",
        "CandleSequenceRules": "Quy tắc chuỗi nến phát hiện tự động",
        "PaperTrades": "Sổ lệnh & vị thế Paper Trading",
        "ModelPredictions": "Lịch sử dự đoán của các mô hình ML",
        "ConfluenceSnapshots": "Ảnh chụp điểm hội tụ tín hiệu",
        "VolumeProfileSnapshots": "Ảnh chụp phân phối Volume Profile",
        "PriceAlertSettings": "Cài đặt ngưỡng cảnh báo giá người dùng",
        "SentimentSnapshots": "Ảnh chụp tâm lý thị trường",
        "__EFMigrationsHistory": "Lịch sử EF Core Database Migrations"
    }

    # Generate output
    lines = []
    lines.append("=" * 125)
    lines.append(f"{'POSTGRESQL DATABASE STORAGE & VOLUME AUDIT':^125}")
    lines.append("=" * 125)
    lines.append(f"{'Bảng (Table)':<32} | {'Dung Lượng':<12} | {'Data / Index':<20} | {'Tổng Số Dòng':<16} | {'Mô Tả Dữ Liệu'}")
    lines.append("-" * 125)
    for st in table_stats:
        tname = st["name"]
        sz = st["total_size"]
        sub_sz = f"{st['data_size']} / {st['index_size']}"
        rows_str = f"{st['rows']:,} rows"
        desc = table_desc.get(tname, "Dữ liệu hệ thống")
        lines.append(f"{tname:<32} | {sz:<12} | {sub_sz:<20} | {rows_str:<16} | {desc}")
    lines.append("-" * 125)
    lines.append(f"{'TỔNG CỘNG TẤT CẢ CÁC BẢNG:':<32} | {db_size_pretty:<12} | {'-':<20} | {total_db_rows:,} rows | Database: bitcoin_analyst")
    lines.append("=" * 125)

    # Breakdown Klines
    cur.execute("""
        SELECT "Symbol", "Timeframe", COUNT(*) as cnt, MIN("OpenTimeMs"), MAX("OpenTimeMs")
        FROM "Klines"
        GROUP BY "Symbol", "Timeframe"
        ORDER BY "Symbol", cnt DESC;
    """)
    klines_rows = cur.fetchall()
    lines.append("\n" + "=" * 125)
    lines.append(f"{'PHÂN TÍCH CHI TIẾT BẢNG KLINES (NƠI CHỨA TRIỆU DÒNG DỮ LIỆU)':^125}")
    lines.append("=" * 125)
    lines.append(f"{'Symbol':<12} | {'Khung (TF)':<10} | {'Số Lượng Dòng':<16} | {'Từ Ngày (UTC)':<20} | {'Đến Ngày (UTC)':<20} | {'Ghi Chú'}")
    lines.append("-" * 125)
    for r in klines_rows:
        sym, tf, cnt, min_t, max_t = r
        s_date = format_ts(min_t)
        e_date = format_ts(max_t)
        cnt_str = f"{cnt:,} rows"
        note = ""
        if tf == "1m":
            note = "Khung 1 phút (chiếm đa số ~3.49M)"
        elif tf == "5m":
            note = "Khung 5 phút (~698K)"
        elif tf == "15m":
            note = "Khung 15 phút (~232K)"
        elif tf == "30m":
            note = "Khung 30 phút (~116K)"
        elif tf == "1h":
            note = "Khung 1 giờ (~58K)"
        elif tf == "4h":
            note = "Khung 4 giờ (~14K)"
        elif tf == "1d":
            note = "Khung 1 ngày (~2.4K)"
        print_note = f"{note}" if note else f"Khung {tf}"
        lines.append(f"{sym:<12} | {tf:<10} | {cnt_str:<16} | {s_date:<20} | {e_date:<20} | {print_note}")
    lines.append("=" * 125)

    # TechnicalIndicators breakdown
    cur.execute("""
        SELECT "Symbol", "Timeframe", COUNT(*) as cnt, MIN("OpenTimeMs"), MAX("OpenTimeMs")
        FROM "TechnicalIndicators"
        GROUP BY "Symbol", "Timeframe"
        ORDER BY "Symbol", cnt DESC;
    """)
    ti_rows = cur.fetchall()
    lines.append("\n" + "=" * 125)
    lines.append(f"{'PHÂN TÍCH CHI TIẾT BẢNG TECHNICAL INDICATORS (TÍNH TOÁN THEO TỪNG NẾN)':^125}")
    lines.append("=" * 125)
    lines.append(f"{'Symbol':<12} | {'Khung (TF)':<10} | {'Số Lượng Dòng':<16} | {'Từ Ngày (UTC)':<20} | {'Đến Ngày (UTC)':<20} | {'Ghi Chú'}")
    lines.append("-" * 125)
    for r in ti_rows:
        sym, tf, cnt, min_t, max_t = r
        s_date = format_ts(min_t)
        e_date = format_ts(max_t)
        cnt_str = f"{cnt:,} rows"
        lines.append(f"{sym:<12} | {tf:<10} | {cnt_str:<16} | {s_date:<20} | {e_date:<20} | Full chỉ báo (RSI, EMA, MACD, BB...)")
    lines.append("=" * 125)

    # MlFeatureStores breakdown
    cur.execute("""
        SELECT "Symbol", "Timeframe", COUNT(*) as cnt, MIN("OpenTimeMs"), MAX("OpenTimeMs")
        FROM "MlFeatureStores"
        GROUP BY "Symbol", "Timeframe"
        ORDER BY "Symbol", cnt DESC;
    """)
    ml_rows = cur.fetchall()
    lines.append("\n" + "=" * 125)
    lines.append(f"{'PHÂN TÍCH CHI TIẾT BẢNG ML FEATURE STORES':^125}")
    lines.append("=" * 125)
    lines.append(f"{'Symbol':<12} | {'Khung (TF)':<10} | {'Số Lượng Dòng':<16} | {'Từ Ngày (UTC)':<20} | {'Đến Ngày (UTC)':<20}")
    lines.append("-" * 125)
    for r in ml_rows:
        sym, tf, cnt, min_t, max_t = r
        s_date = format_ts(min_t)
        e_date = format_ts(max_t)
        cnt_str = f"{cnt:,} rows"
        lines.append(f"{sym:<12} | {tf:<10} | {cnt_str:<16} | {s_date:<20} | {e_date:<20}")
    lines.append("=" * 125)

    # WindowClassificationDatasets breakdown
    cur.execute("""
        SELECT "Symbol", "Timeframe", "WindowSize", COUNT(*) as cnt
        FROM "WindowClassificationDatasets"
        GROUP BY "Symbol", "Timeframe", "WindowSize"
        ORDER BY "Symbol", "Timeframe", "WindowSize";
    """)
    wcd_rows = cur.fetchall()
    lines.append("\n" + "=" * 125)
    lines.append(f"{'PHÂN TÍCH CHI TIẾT BẢNG WINDOW CLASSIFICATION DATASETS (BẢNG LỚN NHẤT 7.35 GB)':^125}")
    lines.append("=" * 125)
    lines.append(f"{'Symbol':<12} | {'Khung (TF)':<10} | {'WindowSize':<12} | {'Số Lượng Dòng':<16} | {'Ghi Chú'}")
    lines.append("-" * 125)
    for r in wcd_rows:
        sym, tf, ws, cnt = r
        cnt_str = f"{cnt:,} rows"
        lines.append(f"{sym:<12} | {tf:<10} | {ws:<12} | {cnt_str:<16} | Feature vector flattened")
    lines.append("=" * 125)

    # FuturesMetrics breakdown
    cur.execute("""
        SELECT "Symbol", COUNT(*) as cnt, MIN("OpenTimeMs"), MAX("OpenTimeMs")
        FROM "FuturesMetrics"
        GROUP BY "Symbol"
        ORDER BY cnt DESC;
    """)
    fm_rows = cur.fetchall()
    lines.append("\n" + "=" * 125)
    lines.append(f"{'PHÂN TÍCH CHI TIẾT BẢNG FUTURES METRICS':^125}")
    lines.append("=" * 125)
    lines.append(f"{'Symbol':<12} | {'Số Lượng Dòng':<16} | {'Từ Ngày (UTC)':<20} | {'Đến Ngày (UTC)':<20} | {'Chi Tiết Chỉ Số'}")
    lines.append("-" * 125)
    for r in fm_rows:
        sym, cnt, min_t, max_t = r
        s_date = format_ts(min_t)
        e_date = format_ts(max_t)
        cnt_str = f"{cnt:,} rows"
        lines.append(f"{sym:<12} | {cnt_str:<16} | {s_date:<20} | {e_date:<20} | OI, L/S Ratios, Taker Volume, Funding Rate")
    lines.append("=" * 125)

    # Pattern & Structure tables breakdown
    lines.append("\n" + "=" * 125)
    lines.append(f"{'PHÂN TÍCH CÁC BẢNG PATTERNS, SEQUENCES & MODELS KHÁC':^125}")
    lines.append("=" * 125)
    other_tables = ["CandlePatterns", "CandleVolumeStats", "WindowVectors", "PatternSequences", "PriceTargets", "ArchetypeOccurrences", "ArchetypeSequences", "PaperTrades", "ModelPredictions", "BacktestRuns", "BacktestTrades", "NewsArticles", "NewsChunks"]
    lines.append(f"{'Bảng (Table)':<32} | {'Số Lượng Dòng':<16} | {'Ghi Chú & Thống Kê Nổi Bật'}")
    lines.append("-" * 125)
    for t in other_tables:
        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
        c = cur.fetchone()[0]
        extra = ""
        if t == "CandlePatterns":
            extra = "Hammer, Engulfing, Morning Star, Shooting Star..."
        elif t == "CandleVolumeStats":
            extra = "Thống kê phân phối Volume Z-Score theo nến"
        elif t == "WindowVectors":
            extra = "Dense embedding vector cho Pattern Search"
        elif t == "PatternSequences":
            extra = "Chuỗi hình thành mẫu nến n-gram"
        elif t == "PriceTargets":
            extra = "Nhãn phân loại Up/Down/Sideways cho training"
        elif t == "PaperTrades":
            cur.execute('SELECT "Status", COUNT(*) FROM "PaperTrades" GROUP BY "Status"')
            st_list = [f"{r[0]}:{r[1]}" for r in cur.fetchall()]
            extra = f"Trạng thái: {', '.join(st_list)}" if st_list else "Chưa có lệnh"
        elif t == "ModelPredictions":
            cur.execute('SELECT "ModelVersion", COUNT(*) FROM "ModelPredictions" GROUP BY "ModelVersion"')
            mv_list = [f"{r[0]}:{r[1]}" for r in cur.fetchall()]
            extra = f"Models: {', '.join(mv_list)}" if mv_list else "Dự đoán realtime"
        elif t == "BacktestRuns":
            extra = "Lịch sử backtest các chiến lược ML"
        elif t == "NewsArticles":
            extra = "RSS Articles từ CoinDesk, Cointelegraph, Decrypt..."
        elif t == "NewsChunks":
            extra = "Text chunks kèm 768-dim / 1536-dim vector embedding"
        lines.append(f"{t:<32} | {c:>10,d} rows    | {extra}")
    lines.append("=" * 125)

    report_content = "\n".join(lines)
    with open("db_audit_report.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(report_content)

    cur.close()
    conn.close()

if __name__ == "__main__":
    run_deep_audit()
