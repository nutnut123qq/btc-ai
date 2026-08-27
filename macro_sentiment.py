#!/usr/bin/env python3
"""
Macro Sentiment & Multi-Source Ingestion Engine
================================================
Tổng hợp điểm số tâm lý vĩ mô đa chiều (Macro Sentiment Composite Score)
kết hợp:
  - 40% Phân tích sắc thái tin tức (News NLP Sentiment qua Financial Lexicon).
  - 30% Chỉ số Crypto Fear & Greed Index (Alternative.me API).
  - 30% Tỷ lệ Taker Buy/Sell Volume & Funding Rate phái sinh.

Lưu trữ vào PostgreSQL table `SentimentSnapshots`.
"""

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import psycopg2
from db_config import get_db_connection, get_db_params

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

FNG_API_URL = "https://api.alternative.me/fng/?limit=5"

# Bộ từ điển tài chính & crypto mở rộng (Financial Sentiment Lexicon)
BULLISH_TERMS = {
    "bull": 1.0, "bullish": 1.2, "surge": 1.2, "surging": 1.2, "rally": 1.2, "rallying": 1.2,
    "gain": 0.8, "gains": 0.9, "gaining": 0.8, "breakout": 1.1, "ath": 1.5, "high": 0.6,
    "highest": 1.0, "adoption": 1.0, "inflow": 1.0, "inflows": 1.1, "accumulate": 0.9,
    "accumulation": 1.0, "etf": 0.8, "approval": 1.1, "approved": 1.2, "upgrade": 0.9,
    "partnership": 0.8, "optimistic": 1.0, "profit": 0.8, "profitable": 0.9, "growth": 0.8,
    "jump": 0.8, "jumped": 0.9, "pump": 1.0, "pumping": 1.0, "outperform": 1.1,
    "recovery": 0.9, "rebound": 0.9, "soar": 1.2, "soaring": 1.3, "institutional": 0.8,
    "milestone": 0.9, "record": 1.0, "buying": 0.8, "boost": 0.9, "climb": 0.8
}

BEARISH_TERMS = {
    "bear": 1.0, "bearish": 1.2, "crash": 1.5, "crashing": 1.5, "plunge": 1.3, "plunging": 1.3,
    "dump": 1.2, "dumping": 1.2, "drop": 0.8, "dropped": 0.9, "dropping": 0.8, "fall": 0.8,
    "falling": 0.8, "loss": 0.8, "losses": 1.0, "selloff": 1.2, "sell-off": 1.2,
    "liquidation": 1.1, "liquidated": 1.2, "outflow": 1.0, "outflows": 1.1, "hack": 1.5,
    "hacked": 1.5, "exploit": 1.4, "fraud": 1.4, "scam": 1.4, "lawsuit": 1.1, "sue": 1.0,
    "sued": 1.1, "sec": 0.7, "ban": 1.3, "banned": 1.4, "investigation": 1.0,
    "recession": 1.2, "inflation": 0.8, "hike": 0.7, "default": 1.3, "bankruptcy": 1.5,
    "insolvent": 1.4, "panic": 1.4, "fear": 1.0, "pessimistic": 1.0, "struggle": 0.8,
    "decline": 0.8, "declining": 0.9, "collapse": 1.5, "warning": 0.7, "down": 0.6
}


def fetch_fear_and_greed() -> Tuple[int, str, List[Dict]]:
    """
    Lấy chỉ số Fear & Greed Index từ Alternative.me API.
    Returns: (current_score 0-100, classification_label, history_records)
    """
    try:
        req = urllib.request.Request(FNG_API_URL, headers={"User-Agent": "MacroSentiment/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            records = data.get("data", [])
            if records:
                score = int(records[0].get("value", 50))
                classification = records[0].get("value_classification", "Neutral")
                return score, classification, records
    except Exception as e:
        print(f"  [!] Warning: Failed to fetch Alternative.me FNG API: {e}. Defaulting to 50 (Neutral).")
    return 50, "Neutral", []


def analyze_news_sentiment(conn, lookback_limit: int = 60) -> Tuple[float, int, List[Dict]]:
    """
    Phân tích NLP sắc thái tin tức từ bảng NewsArticles & NewsChunks.
    Returns: (news_sentiment_score [-1.0, +1.0], sample_count, scored_articles)
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT "Title", "Source", "PublishedAt"
        FROM "NewsArticles"
        WHERE "Title" IS NOT NULL AND "Title" != ''
        ORDER BY COALESCE("PublishedAt", "FetchedAt") DESC
        LIMIT %s;
    """, (lookback_limit,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return 0.0, 0, []

    scored_articles = []
    total_score = 0.0
    valid_count = 0

    for title, source, pub_at in rows:
        tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", title.lower())
        bull_score = sum(BULLISH_TERMS[w] for w in tokens if w in BULLISH_TERMS)
        bear_score = sum(BEARISH_TERMS[w] for w in tokens if w in BEARISH_TERMS)

        net = bull_score - bear_score
        tot = bull_score + bear_score
        if tot > 0:
            art_score = net / tot
            total_score += art_score
            valid_count += 1
            scored_articles.append({
                "title": title,
                "source": source or "RSS",
                "score": round(art_score, 3),
                "label": "BULLISH" if art_score > 0.1 else ("BEARISH" if art_score < -0.1 else "NEUTRAL")
            })

    if valid_count > 0:
        avg_score = max(-1.0, min(1.0, total_score / valid_count))
    else:
        avg_score = 0.0

    return round(avg_score, 4), valid_count, scored_articles


def get_derivatives_metrics(conn, symbol: str) -> Dict:
    """
    Lấy các chỉ số phái sinh gần nhất từ FuturesMetrics (Taker Ratio, Funding Rate, L/S Ratio).
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT "OpenTimeMs", "FundingRate", "TakerBuySellVolRatio", "TopTraderLsSumRatio", "GlobalLsRatio"
        FROM "FuturesMetrics"
        WHERE "Symbol" = %s
        ORDER BY "OpenTimeMs" DESC
        LIMIT 1;
    """, (symbol,))
    row = cur.fetchone()
    cur.close()

    if not row:
        return {
            "funding_rate": 0.0001,
            "taker_ratio": 1.0,
            "ls_ratio": 1.0,
            "time_ms": int(time.time() * 1000)
        }

    return {
        "time_ms": row[0],
        "funding_rate": float(row[1]) if row[1] is not None else 0.0001,
        "taker_ratio": float(row[2]) if row[2] is not None else 1.0,
        "ls_ratio": float(row[3]) if row[3] is not None else (float(row[4]) if row[4] is not None else 1.0)
    }


def compute_composite_macro_sentiment(
    fng_score: int,
    news_sentiment: float,
    taker_ratio: float,
    funding_rate: float,
) -> Tuple[float, float, float, str]:
    """
    Tính điểm tâm lý vĩ mô đa chiều:
      S_macro = 0.40 * S_news + 0.30 * S_fng + 0.30 * S_deriv
    
    Returns:
      (composite_score [-1.0, +1.0], fng_norm [-1.0, +1.0], deriv_score [-1.0, +1.0], market_state_label)
    """
    # 1. Fear & Greed normalized (0 -> -1.0, 50 -> 0.0, 100 -> +1.0)
    fng_norm = (fng_score - 50.0) / 50.0

    # 2. Derivatives component: Taker volume ratio + Funding rate
    s_taker = math.tanh(2.0 * (taker_ratio - 1.0))
    s_funding = max(-1.0, min(1.0, funding_rate / 0.0005))
    deriv_score = (0.5 * s_taker) + (0.5 * s_funding)

    # 3. Composite score (40% News + 30% FNG + 30% Derivatives)
    composite = (0.40 * news_sentiment) + (0.30 * fng_norm) + (0.30 * deriv_score)
    composite = max(-1.0, min(1.0, composite))

    # 4. Market state classification
    if composite <= -0.60:
        label = "EXTREME_FEAR"
    elif composite <= -0.15:
        label = "FEAR"
    elif composite <= 0.15:
        label = "NEUTRAL"
    elif composite <= 0.60:
        label = "GREED"
    else:
        label = "EXTREME_GREED"

    return round(composite, 4), round(fng_norm, 4), round(deriv_score, 4), label


def save_sentiment_snapshot_db(
    conn,
    symbol: str,
    time_ms: int,
    fng_score: int,
    funding_rate: float,
    ls_ratio: float,
    news_score: float,
    composite_score: float,
    label: str,
) -> int:
    """
    Lưu snapshot vào bảng SentimentSnapshots trong PostgreSQL.
    """
    cur = conn.cursor()
    now_utc = datetime.now(timezone.utc)
    # AggregatedSentiment is scaled to -100 to +100
    agg_scaled = round(composite_score * 100.0, 2)

    cur.execute("""
        INSERT INTO "SentimentSnapshots" (
            "Symbol", "TimeMs", "FearGreedScore", "FundingRateZScore",
            "LongShortRatio", "NewsSentimentScore", "AggregatedSentiment",
            "SentimentLabel", "CreatedAtUtc"
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING "Id";
    """, (
        symbol,
        time_ms,
        fng_score,
        funding_rate,
        ls_ratio,
        news_score,
        agg_scaled,
        label,
        now_utc,
    ))
    snap_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return snap_id


def run_sentiment_analysis(
    symbols: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    save_db: bool = True,
) -> Dict[str, Dict]:
    conn = get_db_connection()

    print("\n" + "=" * 80)
    print("      KHỞI CHẠY BỘ TỔNG HỢP TÂM LÝ VĨ MÔ (MACRO SENTIMENT COMPOSITE)")
    print("=" * 80)

    # 1. Lấy Fear & Greed Index
    print("\n[*] 1. Đang truy xuất Crypto Fear & Greed Index từ Alternative.me...")
    fng_score, fng_label, fng_history = fetch_fear_and_greed()
    print(f"  -> Điểm số FNG hiện tại: {fng_score}/100 ({fng_label})")

    # 2. Phân tích News NLP Sentiment
    print("\n[*] 2. Đang phân tích sắc thái tin tức vĩ mô gần nhất qua NLP Lexicon...")
    news_score, news_count, sample_articles = analyze_news_sentiment(conn, lookback_limit=50)
    print(f"  -> Tin tức đã phân tích: {news_count} bài")
    print(f"  -> Điểm sắc thái tin tức (News NLP Score): {news_score:+.4f} ([-1.0, +1.0])")

    results = {}
    now_ms = int(time.time() * 1000)

    # 3. Phân tích từng tài sản với dữ liệu phái sinh
    print("\n[*] 3. Đang tổng hợp dữ liệu phái sinh & tính điểm Composite từng tài sản...")
    for sym in symbols:
        deriv = get_derivatives_metrics(conn, sym)
        composite, fng_norm, deriv_score, label = compute_composite_macro_sentiment(
            fng_score=fng_score,
            news_sentiment=news_score,
            taker_ratio=deriv["taker_ratio"],
            funding_rate=deriv["funding_rate"],
        )

        snap_id = None
        if save_db:
            snap_id = save_sentiment_snapshot_db(
                conn=conn,
                symbol=sym,
                time_ms=now_ms,
                fng_score=fng_score,
                funding_rate=deriv["funding_rate"],
                ls_ratio=deriv["ls_ratio"],
                news_score=news_score,
                composite_score=composite,
                label=label,
            )

        results[sym] = {
            "symbol": sym,
            "fng_score": fng_score,
            "fng_label": fng_label,
            "fng_norm": fng_norm,
            "news_sentiment": news_score,
            "taker_ratio": deriv["taker_ratio"],
            "funding_rate": deriv["funding_rate"],
            "deriv_score": deriv_score,
            "composite_score": composite,
            "aggregated_sentiment_100": round(composite * 100.0, 2),
            "sentiment_label": label,
            "snapshot_id": snap_id,
        }

    conn.close()
    return results


def print_acceptance_report(results: Dict[str, Dict]):
    print("\n" + "=" * 90)
    print("         BÁO CÁO NGHIỆM THU: MACRO SENTIMENT & MULTI-SOURCE INGESTION")
    print("=" * 90)

    print("\n### BẢNG TỔNG HỢP ĐIỂM TÂM LÝ VĨ MÔ THỜI GIAN THỰC (REALTIME MACRO SENTIMENT)")
    print("-" * 100)
    print(f"| {'Asset':<8} | {'Fear & Greed':<14} | {'News NLP (40%)':<16} | {'Deriv Score (30%)':<19} | {'Composite Score':<17} | {'Market State':<14} |")
    print("-" * 100)

    for sym, r in results.items():
        fng_str = f"{r['fng_score']}/100 ({r['fng_label']})"
        news_str = f"{r['news_sentiment']:+.4f}"
        deriv_str = f"{r['deriv_score']:+.4f} (T={r['taker_ratio']:.2f})"
        comp_str = f"{r['composite_score']:+.4f} ({r['aggregated_sentiment_100']:+.1f})"
        lbl = r['sentiment_label']
        print(f"| {sym:<8} | {fng_str:<14} | {news_str:<16} | {deriv_str:<19} | {comp_str:<17} | {lbl:<14} |")

    print("-" * 100)
    print("\n* Ghi chú:")
    print("  - S_macro = 0.40 * S_news + 0.30 * S_fng + 0.30 * S_deriv")
    print("  - Phân loại trạng thái: EXTREME_FEAR (<= -0.60), FEAR (<= -0.15), NEUTRAL (±0.15), GREED (<= +0.60), EXTREME_GREED (> +0.60)")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Macro Sentiment & Multi-Source Ingestion Engine")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"], help="Symbols to analyze")
    parser.add_argument("--no-save", action="store_true", help="Do not save snapshots to database")
    parser.add_argument("--report", action="store_true", default=True, help="Print acceptance report")
    args = parser.parse_args()

    results = run_sentiment_analysis(
        symbols=args.symbols,
        save_db=not args.no_save,
    )
    if args.report:
        print_acceptance_report(results)


if __name__ == "__main__":
    main()
