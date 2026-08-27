#!/usr/bin/env python3
"""
Liquidation Levels & Heatmap Engine (Coinglass / Kingfisher Model)
==================================================================
Ước lượng các cụm mức giá thanh lý tiềm năng (Estimated Liquidation Levels)
dựa trên biến động Open Interest (ΔOI), Tỷ lệ Đòn bẩy (25x, 50x, 100x),
Top-Trader Long/Short Ratio và Khối lượng giao dịch.

Thuật toán chính:
1. Mô hình phân bổ đòn bẩy: 25x (40%), 50x (35%), 100x (25%).
2. Công thức giá thanh lý chuẩn Binance Futures kèm Maintenance Margin Rate (MMR):
   - LiqPrice_Long  = EntryPrice * (1 - 1/Leverage + MMR)
   - LiqPrice_Short = EntryPrice * (1 + 1/Leverage - MMR)
3. Phân bổ vị thế từ ΔOI và Top-Trader Long/Short Ratio.
4. Lọc vị thế đã bị quét (Swept Liquidation Filtering) theo biến động giá quá khứ.
5. Gom cụm mật độ thanh lý (Liquidation Density Binning) theo bước 0.25% - 0.5%.
6. Lưu trữ snapshot vào PostgreSQL table `LiquidationSnapshots`.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_values

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

FAPI_BASE = "https://fapi.binance.com"

# Cấu hình đòn bẩy và trọng số phân bổ mặc định
DEFAULT_LEVERAGE_WEIGHTS = {
    100: 0.25,  # 25% volume ở 100x
    50: 0.35,   # 35% volume ở 50x
    25: 0.40,   # 40% volume ở 25x
}

# Tỷ lệ ký quỹ duy trì (Maintenance Margin Rate) theo nấc đòn bẩy trên Binance
DEFAULT_MMR_BY_LEVERAGE = {
    100: 0.004,  # 0.4%
    50: 0.005,   # 0.5%
    25: 0.010,   # 1.0%
    20: 0.010,   # 1.0%
    10: 0.020,   # 2.0%
}


class LiquidationEngine:
    def __init__(
        self,
        leverage_weights: Optional[Dict[int, float]] = None,
        mmr_map: Optional[Dict[int, float]] = None,
        bin_step_pct: float = 0.003,  # 0.3% step (0.25% - 0.5%)
        span_pct: float = 0.15,       # Phạm vi ±15% quanh giá hiện tại
    ):
        self.leverage_weights = leverage_weights or DEFAULT_LEVERAGE_WEIGHTS
        self.mmr_map = mmr_map or DEFAULT_MMR_BY_LEVERAGE
        self.bin_step_pct = bin_step_pct
        self.span_pct = span_pct

    @staticmethod
    def calculate_long_liq_price(entry_price: float, leverage: float, mmr: float) -> float:
        """
        LiqPrice_Long = EntryPrice * (1 - 1/Leverage + MMR)
        """
        return entry_price * (1.0 - (1.0 / leverage) + mmr)

    @staticmethod
    def calculate_short_liq_price(entry_price: float, leverage: float, mmr: float) -> float:
        """
        LiqPrice_Short = EntryPrice * (1 + 1/Leverage - MMR)
        """
        return entry_price * (1.0 + (1.0 / leverage) - mmr)

    def compute_liquidation_tranches(
        self,
        bars: List[Dict],
        current_price: float,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Tính toán các đợt thanh lý từ chuỗi dữ liệu (Klines + FuturesMetrics).
        Lọc bỏ các vị thế đã bị thanh lý (swept) bởi nến sau đó.
        
        Returns:
            active_long_tranches, active_short_tranches
        """
        n = len(bars)
        if n == 0:
            return [], []

        raw_long_tranches = []
        raw_short_tranches = []

        # 1. Tạo các tranches từ từng nến
        for i in range(n):
            b = bars[i]
            entry_price = float(b.get("close") or b.get("Close") or current_price)
            delta_oi = float(b.get("delta_oi_usdt") or 0.0)
            base_vol = float(b.get("volume_usdt") or b.get("volume", 0.0) * entry_price)
            ls_ratio = float(b.get("ls_ratio") or 1.0)

            # Ước lượng tỷ lệ Long / Short
            # Top-Trader Long/Short Ratio R = Long% / Short%  => Long% = R / (1 + R)
            if ls_ratio > 0:
                long_fraction = ls_ratio / (1.0 + ls_ratio)
            else:
                long_fraction = 0.5
            short_fraction = 1.0 - long_fraction

            # Ước lượng quy mô vị thế mở mới
            # Nếu delta_oi > 0: dùng delta_oi; nếu delta_oi <= 0: dùng tỷ lệ nhỏ từ khối lượng giao dịch
            if delta_oi > 0:
                pos_vol = delta_oi
            else:
                # Vị thế luân chuyển từ volume nến
                pos_vol = base_vol * 0.15

            long_vol = pos_vol * long_fraction
            short_vol = pos_vol * short_fraction

            for lev, weight in self.leverage_weights.items():
                mmr = self.mmr_map.get(lev, 0.005)
                
                # Long position liquidation level (nằm dưới entry)
                liq_long = self.calculate_long_liq_price(entry_price, lev, mmr)
                if liq_long > 0:
                    raw_long_tranches.append({
                        "bar_idx": i,
                        "open_time_ms": b.get("open_time_ms") or b.get("OpenTimeMs"),
                        "entry_price": entry_price,
                        "leverage": lev,
                        "liq_price": liq_long,
                        "volume_usdt": long_vol * weight,
                        "side": "LONG",
                    })

                # Short position liquidation level (nằm trên entry)
                liq_short = self.calculate_short_liq_price(entry_price, lev, mmr)
                if liq_short > 0:
                    raw_short_tranches.append({
                        "bar_idx": i,
                        "open_time_ms": b.get("open_time_ms") or b.get("OpenTimeMs"),
                        "entry_price": entry_price,
                        "leverage": lev,
                        "liq_price": liq_short,
                        "volume_usdt": short_vol * weight,
                        "side": "SHORT",
                    })

        # 2. Swept Liquidation Filtering: Lọc bỏ các mức giá thanh lý đã bị chạm
        # Long position: bị thanh lý khi Low_k <= LiqPrice (với k > i)
        # Short position: bị thanh lý khi High_k >= LiqPrice (với k > i)
        active_long_tranches = []
        for t in raw_long_tranches:
            bar_idx = t["bar_idx"]
            liq_p = t["liq_price"]
            swept = False
            # Quét các nến sau thời điểm mở vị thế
            for k in range(bar_idx + 1, n):
                low_k = float(bars[k].get("low") or bars[k].get("Low") or 0.0)
                if low_k > 0 and low_k <= liq_p:
                    swept = True
                    break
            # Nếu giá hiện tại cũng đã xuyên qua liq_p thì đã bị thanh lý
            if current_price <= liq_p:
                swept = True

            if not swept and t["volume_usdt"] > 0:
                active_long_tranches.append(t)

        active_short_tranches = []
        for t in raw_short_tranches:
            bar_idx = t["bar_idx"]
            liq_p = t["liq_price"]
            swept = False
            # Quét các nến sau thời điểm mở vị thế
            for k in range(bar_idx + 1, n):
                high_k = float(bars[k].get("high") or bars[k].get("High") or 0.0)
                if high_k > 0 and high_k >= liq_p:
                    swept = True
                    break
            # Nếu giá hiện tại cũng đã vượt qua liq_p thì đã bị thanh lý
            if current_price >= liq_p:
                swept = True

            if not swept and t["volume_usdt"] > 0:
                active_short_tranches.append(t)

        return active_long_tranches, active_short_tranches

    def bin_liquidation_density(
        self,
        long_tranches: List[Dict],
        short_tranches: List[Dict],
        current_price: float,
    ) -> Tuple[List[Dict], float, float]:
        """
        Gom cụm các tranches thanh lý còn tồn tại theo các bước giá bin_step_pct quanh giá hiện tại.
        
        Returns:
            heatmap_bins, total_long_usdt, total_short_usdt
        """
        step = current_price * self.bin_step_pct
        if step <= 0:
            return [], 0.0, 0.0

        min_price = current_price * (1.0 - self.span_pct)
        max_price = current_price * (1.0 + self.span_pct)

        # Khởi tạo các bins
        bins_dict = {}
        curr_p = min_price
        while curr_p <= max_price + (step * 0.5):
            bin_center = round(curr_p, 2)
            side = "LONG" if bin_center < current_price else "SHORT"
            bins_dict[bin_center] = {
                "price": bin_center,
                "cumulative_vol_usdt": 0.0,
                "side": side,
                "density_pct": 0.0,
                "leverage_breakdown": {f"{k}x": 0.0 for k in self.leverage_weights.keys()},
                "distance_pct": round(((bin_center - current_price) / current_price) * 100, 2),
            }
            curr_p += step

        # Gom Long Tranches
        total_long_usdt = 0.0
        for t in long_tranches:
            p = t["liq_price"]
            v = t["volume_usdt"]
            total_long_usdt += v
            if min_price <= p <= max_price:
                # Tìm bin gần nhất
                best_bin = min(bins_dict.keys(), key=lambda b_p: abs(b_p - p))
                bins_dict[best_bin]["cumulative_vol_usdt"] += v
                lev_key = f"{t['leverage']}x"
                if lev_key in bins_dict[best_bin]["leverage_breakdown"]:
                    bins_dict[best_bin]["leverage_breakdown"][lev_key] += v

        # Gom Short Tranches
        total_short_usdt = 0.0
        for t in short_tranches:
            p = t["liq_price"]
            v = t["volume_usdt"]
            total_short_usdt += v
            if min_price <= p <= max_price:
                # Tìm bin gần nhất
                best_bin = min(bins_dict.keys(), key=lambda b_p: abs(b_p - p))
                bins_dict[best_bin]["cumulative_vol_usdt"] += v
                lev_key = f"{t['leverage']}x"
                if lev_key in bins_dict[best_bin]["leverage_breakdown"]:
                    bins_dict[best_bin]["leverage_breakdown"][lev_key] += v

        # Tính mật độ (density %)
        total_vol = total_long_usdt + total_short_usdt
        heatmap_bins = []
        for b_p in sorted(bins_dict.keys()):
            b_data = bins_dict[b_p]
            b_vol = b_data["cumulative_vol_usdt"]
            if total_vol > 0:
                b_data["density_pct"] = round((b_vol / total_vol) * 100, 3)
            b_data["cumulative_vol_usdt"] = round(b_vol, 2)
            for k in b_data["leverage_breakdown"]:
                b_data["leverage_breakdown"][k] = round(b_data["leverage_breakdown"][k], 2)
            heatmap_bins.append(b_data)

        return heatmap_bins, round(total_long_usdt, 2), round(total_short_usdt, 2)

    def extract_top_targets(
        self,
        heatmap_bins: List[Dict],
        current_price: float,
        top_n: int = 5,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Trích xuất các vùng thanh lý tập trung mạnh nhất:
        - Short Squeeze Targets (nằm trên giá hiện tại)
        - Long Flush Targets (nằm dưới giá hiện tại)
        """
        short_bins = [b for b in heatmap_bins if b["side"] == "SHORT" and b["price"] > current_price]
        long_bins = [b for b in heatmap_bins if b["side"] == "LONG" and b["price"] < current_price]

        # Sắp xếp theo khối lượng thanh lý giảm dần
        top_short_squeeze = sorted(short_bins, key=lambda x: x["cumulative_vol_usdt"], reverse=True)[:top_n]
        top_long_flush = sorted(long_bins, key=lambda x: x["cumulative_vol_usdt"], reverse=True)[:top_n]

        return top_short_squeeze, top_long_flush


# ==============================================================================
# DATA FETCHER & DATABASE PERSISTENCE
# ==============================================================================

def fetch_bars_from_db(
    symbol: str,
    timeframe: str = "1h",
    lookback: int = 500,
) -> Tuple[List[Dict], float]:
    """
    Đọc Klines và FuturesMetrics từ PostgreSQL.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Lấy Klines
    cur.execute("""
        SELECT "OpenTimeMs", "Open", "High", "Low", "Close", "Volume", "QuoteVolume"
        FROM "Klines"
        WHERE "Symbol" = %s AND "Timeframe" = %s
        ORDER BY "OpenTimeMs" DESC
        LIMIT %s;
    """, (symbol, timeframe, lookback))
    kline_rows = cur.fetchall()

    if not kline_rows:
        cur.close()
        conn.close()
        return [], 0.0

    kline_rows = sorted(kline_rows, key=lambda x: x[0])
    current_price = float(kline_rows[-1][4])  # Last Close

    # 2. Lấy FuturesMetrics tương ứng
    min_ms = kline_rows[0][0]
    max_ms = kline_rows[-1][0]
    cur.execute("""
        SELECT "OpenTimeMs", "OpenInterest", "OpenInterestValue", "TopTraderLsSumRatio", "GlobalLsRatio"
        FROM "FuturesMetrics"
        WHERE "Symbol" = %s AND "OpenTimeMs" >= %s AND "OpenTimeMs" <= %s
        ORDER BY "OpenTimeMs" ASC;
    """, (symbol, min_ms, max_ms))
    fm_rows = cur.fetchall()
    fm_map = {r[0]: r for r in fm_rows}

    cur.close()
    conn.close()

    # 3. Ghép nến và tính delta OI
    bars = []
    prev_oi = None
    for k in kline_rows:
        t_ms = k[0]
        fm = fm_map.get(t_ms)

        oi = float(fm[2]) if fm and fm[2] is not None else (float(fm[1]) * float(k[4]) if fm and fm[1] else None)
        ls_ratio = float(fm[3]) if fm and fm[3] is not None else (float(fm[4]) if fm and fm[4] else 1.0)

        delta_oi = 0.0
        if oi is not None:
            if prev_oi is not None:
                delta_oi = oi - prev_oi
            prev_oi = oi

        bars.append({
            "open_time_ms": t_ms,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "volume_usdt": float(k[6]) if k[6] else float(k[5]) * float(k[4]),
            "delta_oi_usdt": delta_oi,
            "ls_ratio": ls_ratio,
        })

    return bars, current_price


def fetch_live_binance_data(symbol: str, limit: int = 200) -> Tuple[List[Dict], float]:
    """
    Fallback: Lấy dữ liệu trực tiếp từ Binance Futures API công khai.
    """
    # 1. Klines
    url_k = f"{FAPI_BASE}/fapi/v1/klines?symbol={symbol}&interval=1h&limit={limit}"
    req = urllib.request.Request(url_k, headers={"User-Agent": "liq-engine/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        k_data = json.loads(resp.read().decode("utf-8"))

    # 2. Open Interest History
    url_oi = f"{FAPI_BASE}/futures/data/openInterestHist?symbol={symbol}&period=1h&limit={limit}"
    req_oi = urllib.request.Request(url_oi, headers={"User-Agent": "liq-engine/1.0"})
    with urllib.request.urlopen(req_oi, timeout=15) as resp:
        oi_data = json.loads(resp.read().decode("utf-8"))
    oi_map = {int(x["timestamp"]): float(x["sumOpenInterestValue"]) for x in oi_data}

    # 3. Top Trader Long/Short Ratio
    url_ls = f"{FAPI_BASE}/futures/data/topLongShortPositionRatio?symbol={symbol}&period=1h&limit={limit}"
    req_ls = urllib.request.Request(url_ls, headers={"User-Agent": "liq-engine/1.0"})
    with urllib.request.urlopen(req_ls, timeout=15) as resp:
        ls_data = json.loads(resp.read().decode("utf-8"))
    ls_map = {int(x["timestamp"]): float(x["longShortRatio"]) for x in ls_data}

    bars = []
    prev_oi = None
    for k in k_data:
        t_ms = int(k[0])
        close_p = float(k[4])
        oi = oi_map.get(t_ms)
        ls = ls_map.get(t_ms, 1.0)

        delta_oi = 0.0
        if oi is not None:
            if prev_oi is not None:
                delta_oi = oi - prev_oi
            prev_oi = oi

        bars.append({
            "open_time_ms": t_ms,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": close_p,
            "volume": float(k[5]),
            "volume_usdt": float(k[7]),
            "delta_oi_usdt": delta_oi,
            "ls_ratio": ls,
        })

    current_price = float(k_data[-1][4])
    return bars, current_price


def save_liquidation_snapshot(
    symbol: str,
    timeframe: str,
    timestamp_utc: datetime,
    current_price: float,
    total_long_usdt: float,
    total_short_usdt: float,
    heatmap_bins: List[Dict],
) -> int:
    """
    Lưu snapshot vào bảng LiquidationSnapshots trong PostgreSQL.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    heatmap_json = json.dumps(heatmap_bins, ensure_ascii=False)
    now_utc = datetime.now(timezone.utc)

    sql = """
        INSERT INTO "LiquidationSnapshots" (
            "Symbol", "Timeframe", "TimestampUtc", "CurrentPrice",
            "TotalLongLiqUsdt", "TotalShortLiqUsdt", "HeatmapJson", "CreatedAtUtc"
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING "Id";
    """
    cur.execute(sql, (
        symbol,
        timeframe,
        timestamp_utc,
        current_price,
        total_long_usdt,
        total_short_usdt,
        heatmap_json,
        now_utc,
    ))
    snap_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return snap_id


# ==============================================================================
# RUNNER & REPORTING
# ==============================================================================

def run_liquidation_analysis(
    symbols: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    timeframe: str = "1h",
    lookback: int = 500,
    bin_step_pct: float = 0.003,
    save_to_db: bool = True,
) -> Dict[str, Dict]:
    """
    Chạy phân tích liquidation heatmap cho danh sách các symbols.
    """
    engine = LiquidationEngine(bin_step_pct=bin_step_pct)
    results = {}

    for sym in symbols:
        print(f"\n==================================================================")
        print(f"[*] Processing Liquidation Heatmap: {sym} (Timeframe: {timeframe})")
        print(f"==================================================================")

        # 1. Đọc dữ liệu
        bars, current_price = fetch_bars_from_db(sym, timeframe, lookback)
        if len(bars) < 10:
            print(f"  [!] DB data for {sym} sparse ({len(bars)} bars). Fetching live Binance Futures...")
            try:
                bars, current_price = fetch_live_binance_data(sym, limit=min(lookback, 500))
            except Exception as ex:
                print(f"  [X] Failed live fetch for {sym}: {ex}")
                continue

        print(f"  -> Total bars loaded: {len(bars)}")
        print(f"  -> Current Price: ${current_price:,.2f}")

        # 2. Tính toán tranches & lọc vị thế đã bị quét
        long_tranches, short_tranches = engine.compute_liquidation_tranches(bars, current_price)
        print(f"  -> Active Long Liquidation Tranches (Unliquidated): {len(long_tranches)}")
        print(f"  -> Active Short Liquidation Tranches (Unliquidated): {len(short_tranches)}")

        # 3. Gom cụm mật độ (Binning)
        heatmap_bins, total_long_usdt, total_short_usdt = engine.bin_liquidation_density(
            long_tranches, short_tranches, current_price
        )
        print(f"  -> Total Active Long Liq Pool:  ${total_long_usdt:,.2f}")
        print(f"  -> Total Active Short Liq Pool: ${total_short_usdt:,.2f}")

        # 4. Trích xuất mục tiêu quét thanh lý
        top_shorts, top_longs = engine.extract_top_targets(heatmap_bins, current_price, top_n=5)

        # 5. Lưu vào Database
        snap_id = None
        if save_to_db:
            ts_utc = datetime.now(timezone.utc)
            snap_id = save_liquidation_snapshot(
                symbol=sym,
                timeframe=timeframe,
                timestamp_utc=ts_utc,
                current_price=current_price,
                total_long_usdt=total_long_usdt,
                total_short_usdt=total_short_usdt,
                heatmap_bins=heatmap_bins,
            )
            print(f"  [+] Saved Snapshot to Database: ID={snap_id}")

        results[sym] = {
            "symbol": sym,
            "timeframe": timeframe,
            "current_price": current_price,
            "total_long_liq_usdt": total_long_usdt,
            "total_short_liq_usdt": total_short_usdt,
            "top_short_squeeze_targets": top_shorts,
            "top_long_flush_targets": top_longs,
            "heatmap_bins_count": len(heatmap_bins),
            "snapshot_id": snap_id,
        }

    return results


def print_acceptance_report(results: Dict[str, Dict]):
    """
    In báo cáo nghiệm thu dạng bảng chuẩn Markdown và Console.
    """
    print("\n" + "=" * 80)
    print("           BÁO CÁO NGHIỆM THU: LIQUIDATION HEATMAP ENGINE")
    print("=" * 80)

    # 1. Bảng Tổng Hợp So Sánh 3 Tài Sản
    print("\n### 1. TỔNG HỢP TÌNH TRẠNG THANH LÝ (LIQUIDATION SUMMARY)")
    print("-" * 88)
    print(f"| {'Asset':<8} | {'Price':<11} | {'Long Liq ($)':<16} | {'Short Liq ($)':<16} | {'Liq Imbalance':<14} | {'Dominant Bias':<10} |")
    print("-" * 88)
    for sym, r in results.items():
        p = f"${r['current_price']:,.2f}"
        l_vol = f"${r['total_long_liq_usdt']:,.0f}"
        s_vol = f"${r['total_short_liq_usdt']:,.0f}"
        
        tot = r['total_long_liq_usdt'] + r['total_short_liq_usdt']
        if tot > 0:
            l_pct = (r['total_long_liq_usdt'] / tot) * 100
            s_pct = (r['total_short_liq_usdt'] / tot) * 100
            imb = f"L {l_pct:.1f}% / S {s_pct:.1f}%"
            bias = "LONG FLUSH" if l_pct > 55 else ("SHORT SQZ" if s_pct > 55 else "NEUTRAL")
        else:
            imb = "N/A"
            bias = "N/A"
        print(f"| {sym:<8} | {p:<11} | {l_vol:<16} | {s_vol:<16} | {imb:<14} | {bias:<10} |")
    print("-" * 88)

    # 2. Chi tiết Top Mục Tiêu Thanh Lý từng tài sản
    for sym, r in results.items():
        p_curr = r['current_price']
        print(f"\n### 2.{list(results.keys()).index(sym) + 1}. MỤC TIÊU THANH LÝ TRỌNG ĐIỂM: {sym} (Current: ${p_curr:,.2f})")
        
        print("\n  [A] TOP SHORT SQUEEZE TARGETS (Vùng gom thanh lý Short phía trên):")
        print("  " + "-" * 76)
        print(f"  | {'Rank':<4} | {'Target Price':<14} | {'Distance (%)':<14} | {'Est. Liq Vol ($)':<18} | {'Density (%)':<11} |")
        print("  " + "-" * 76)
        for idx, t in enumerate(r['top_short_squeeze_targets'], 1):
            dist = f"+{t['distance_pct']:.2f}%" if t['distance_pct'] > 0 else f"{t['distance_pct']:.2f}%"
            print(f"  | {idx:<4} | ${t['price']:<13,.2f} | {dist:<14} | ${t['cumulative_vol_usdt']:<17,.0f} | {t['density_pct']:<10.2f}% |")
        print("  " + "-" * 76)

        print("\n  [B] TOP LONG FLUSH TARGETS (Vùng gom thanh lý Long phía dưới):")
        print("  " + "-" * 76)
        print(f"  | {'Rank':<4} | {'Target Price':<14} | {'Distance (%)':<14} | {'Est. Liq Vol ($)':<18} | {'Density (%)':<11} |")
        print("  " + "-" * 76)
        for idx, t in enumerate(r['top_long_flush_targets'], 1):
            dist = f"{t['distance_pct']:.2f}%"
            print(f"  | {idx:<4} | ${t['price']:<13,.2f} | {dist:<14} | ${t['cumulative_vol_usdt']:<17,.0f} | {t['density_pct']:<10.2f}% |")
        print("  " + "-" * 76)

    print("\n" + "=" * 80 + "\n")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Liquidation Levels & Heatmap Engine")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"], help="Symbols to analyze")
    parser.add_argument("--timeframe", default="1h", choices=["15m", "1h", "4h", "1d"], help="Timeframe")
    parser.add_argument("--lookback", type=int, default=500, help="Lookback candles")
    parser.add_argument("--bin-step", type=float, default=0.003, help="Bin step pct (default 0.003 = 0.3%%)")
    parser.add_argument("--no-save", action="store_true", help="Do not save snapshots to database")
    parser.add_argument("--report", action="store_true", default=True, help="Print acceptance report")
    parser.add_argument("--loop", action="store_true", help="Run in continuous loop mode")
    parser.add_argument("--interval", type=int, default=3600, help="Interval in seconds for loop mode")

    args = parser.parse_args()

    if args.loop:
        print(f"[*] Starting Liquidation Engine loop mode (interval: {args.interval}s)...")
        while True:
            try:
                results = run_liquidation_analysis(
                    symbols=args.symbols,
                    timeframe=args.timeframe,
                    lookback=args.lookback,
                    bin_step_pct=args.bin_step,
                    save_to_db=not args.no_save,
                )
                if args.report:
                    print_acceptance_report(results)
            except Exception as e:
                print(f"[X] Loop iteration error: {e}")
            print(f"[*] Sleeping for {args.interval}s until next candle cycle...")
            time.sleep(args.interval)
    else:
        results = run_liquidation_analysis(
            symbols=args.symbols,
            timeframe=args.timeframe,
            lookback=args.lookback,
            bin_step_pct=args.bin_step,
            save_to_db=not args.no_save,
        )
        if args.report:
            print_acceptance_report(results)


if __name__ == "__main__":
    main()
