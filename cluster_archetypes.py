#!/usr/bin/env python3
"""
Candle Archetype Clustering — Offline Script

Reads WindowClassificationDatasets from PostgreSQL, clusters windows into archetypes
using Mini-Batch K-Means, computes outcome statistics, and writes results to
CandleArchetypes / ArchetypeOutcomes / ArchetypeOccurrences tables.

Usage:
    python cluster_archetypes.py [--timeframe 1h] [--window-sizes 10,15,20,25] [--version 1]

RAM Safety:
    - Streams data via server-side cursor (5000 rows/batch)
    - Processes one (timeframe, windowSize) combo at a time
    - Frees numpy arrays between combos
    - Estimated peak: ~100MB per combo for 50K windows × 350 features
"""

import os
import sys
import gc
import math
import json
import argparse
import time
from datetime import datetime, timezone, timedelta
from collections import Counter

import numpy as np
import psycopg2
import psycopg2.extras

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn is required. Install: pip install scikit-learn")
    sys.exit(1)

from db_config import get_db_connection

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
DEFAULT_TIMEFRAMES = ["1h", "4h"]
DEFAULT_WINDOW_SIZES = [10, 15, 20, 25]
HORIZONS = ["1h", "4h", "1d"]
CURSOR_BATCH = 5000
RECENT_MONTHS = 6

# --- DB helpers --------------------------------------------------------------

def get_connection():
    return get_db_connection()


def fetch_window_data(conn, symbol, timeframe, window_size, horizon):
    """Stream window dataset rows via server-side cursor to avoid OOM."""
    cursor_name = f"arch_cursor_{window_size}_{horizon}"
    with conn.cursor(name=cursor_name) as cur:
        cur.itersize = CURSOR_BATCH
        cur.execute("""
            SELECT "FeatureVector", "Label", "WindowStartMs", "WindowEndMs",
                   "TargetReturn", "FeatureDim"
            FROM "WindowClassificationDatasets"
            WHERE "Symbol" = %s AND "Timeframe" = %s
              AND "WindowSize" = %s AND "Horizon" = %s
            ORDER BY "WindowEndMs"
        """, (symbol, timeframe, window_size, horizon))

        vectors = []
        labels = []
        start_ms_list = []
        end_ms_list = []
        returns_list = []

        for row in cur:
            feat_vec, label, start_ms, end_ms, target_ret, feat_dim = row
            if feat_vec is None or len(feat_vec) == 0:
                continue
            if label not in (-1, 0, 1):
                continue
            vectors.append(np.array(feat_vec, dtype=np.float32))
            labels.append(label)
            start_ms_list.append(start_ms)
            end_ms_list.append(end_ms)
            returns_list.append(target_ret if target_ret is not None else 0.0)

    if len(vectors) == 0:
        return None, None, None, None, None

    # Filter out any legacy vectors with mismatched dimension
    len_counts = Counter(len(v) for v in vectors)
    target_len = len_counts.most_common(1)[0][0]

    valid_indices = [i for i, v in enumerate(vectors) if len(v) == target_len]
    vectors = [vectors[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    start_ms_list = [start_ms_list[i] for i in valid_indices]
    end_ms_list = [end_ms_list[i] for i in valid_indices]
    returns_list = [returns_list[i] for i in valid_indices]

    X = np.vstack(vectors)
    y = np.array(labels, dtype=np.int8)
    starts = np.array(start_ms_list, dtype=np.int64)
    ends = np.array(end_ms_list, dtype=np.int64)
    rets = np.array(returns_list, dtype=np.float64)

    del vectors, labels, start_ms_list, end_ms_list, returns_list
    return X, y, starts, ends, rets


def fetch_representative_ohlc(conn, symbol, timeframe, start_ms, window_size):
    """Fetch OHLC data for the representative window from Klines table."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT "Open", "High", "Low", "Close", "Volume"
            FROM "Klines"
            WHERE "Symbol" = %s AND "Timeframe" = %s AND "OpenTimeMs" >= %s
            ORDER BY "OpenTimeMs"
            LIMIT %s
        """, (symbol, timeframe, start_ms, window_size))
        rows = cur.fetchall()

    if len(rows) < window_size:
        return None

    return [
        {"open": float(r[0]), "high": float(r[1]), "low": float(r[2]),
         "close": float(r[3]), "volume": float(r[4])}
        for r in rows[:window_size]
    ]


# --- Clustering logic --------------------------------------------------------

def compute_k(n_samples):
    """Heuristic: K = min(500, floor(sqrt(N/2)))."""
    k = int(math.floor(math.sqrt(n_samples / 2)))
    return max(5, min(500, k))


def cluster_windows(X, n_clusters, random_state=42):
    """Run Mini-Batch K-Means on feature vectors."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Replace NaN/Inf that may come from scaling zero-variance columns
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(1024, len(X_scaled)),
        random_state=random_state,
        n_init=3,
        max_iter=300,
    )
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Transform centroids back to original space
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_original = np.nan_to_num(centroids_original, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute distances in scaled space
    intra_distances = np.zeros(n_clusters, dtype=np.float32)
    for ci in range(n_clusters):
        mask = cluster_labels == ci
        if mask.sum() == 0:
            continue
        dists = np.linalg.norm(X_scaled[mask] - kmeans.cluster_centers_[ci], axis=1)
        intra_distances[ci] = float(np.mean(dists))

    return cluster_labels, centroids_original, intra_distances, X_scaled, kmeans.cluster_centers_


def find_representative(X_scaled, scaled_centroids, cluster_labels, cluster_id):
    """Find the member closest to the centroid (in scaled space)."""
    mask = cluster_labels == cluster_id
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    members = X_scaled[indices]
    dists = np.linalg.norm(members - scaled_centroids[cluster_id], axis=1)
    best_local = np.argmin(dists)
    return int(indices[best_local])


def compute_cosine_distances(X, centroids, cluster_labels, n_clusters):
    """Compute cosine distance from each member to its cluster centroid."""
    distances = np.zeros(len(X), dtype=np.float32)
    for ci in range(n_clusters):
        mask = cluster_labels == ci
        if mask.sum() == 0:
            continue
        centroid = centroids[ci]
        c_norm = np.linalg.norm(centroid)
        if c_norm < 1e-8:
            distances[mask] = 1.0
            continue
        members = X[mask]
        m_norms = np.linalg.norm(members, axis=1)
        m_norms = np.maximum(m_norms, 1e-8)
        cosine_sim = np.dot(members, centroid) / (m_norms * c_norm)
        distances[mask] = 1.0 - cosine_sim
    return distances


def compute_outcome_stats(labels_arr, returns_arr, ends_arr, recent_cutoff_ms):
    """Compute outcome statistics for a group of windows."""
    n = len(labels_arr)
    if n == 0:
        return None

    up = int(np.sum(labels_arr == 1))
    down = int(np.sum(labels_arr == -1))
    sideways = int(np.sum(labels_arr == 0))

    up_rate = up / n if n > 0 else 0.0
    down_rate = down / n if n > 0 else 0.0
    sideways_rate = sideways / n if n > 0 else 0.0

    avg_ret = float(np.mean(returns_arr)) if n > 0 else 0.0
    median_ret = float(np.median(returns_arr)) if n > 0 else 0.0
    max_ret = float(np.max(returns_arr)) if n > 0 else 0.0
    min_ret = float(np.min(returns_arr)) if n > 0 else 0.0
    std_ret = float(np.std(returns_arr)) if n > 1 else 0.0

    # Recent performance
    recent_mask = ends_arr >= recent_cutoff_ms
    recent_n = int(np.sum(recent_mask))
    recent_up_rate = 0.0
    recent_down_rate = 0.0
    recent_avg_ret = 0.0
    if recent_n > 0:
        recent_labels = labels_arr[recent_mask]
        recent_returns = returns_arr[recent_mask]
        recent_up_rate = float(np.sum(recent_labels == 1) / recent_n)
        recent_down_rate = float(np.sum(recent_labels == -1) / recent_n)
        recent_avg_ret = float(np.mean(recent_returns))

    return {
        "total": n,
        "up": up, "down": down, "sideways": sideways,
        "up_rate": round(up_rate, 4),
        "down_rate": round(down_rate, 4),
        "sideways_rate": round(sideways_rate, 4),
        "avg_return": round(avg_ret, 4),
        "median_return": round(median_ret, 4),
        "max_return": round(max_ret, 4),
        "min_return": round(min_ret, 4),
        "std_return": round(std_ret, 4),
        "recent_n": recent_n,
        "recent_up_rate": round(recent_up_rate, 4),
        "recent_down_rate": round(recent_down_rate, 4),
        "recent_avg_return": round(recent_avg_ret, 4),
    }


# --- DB write ----------------------------------------------------------------

def write_results(conn, symbol, timeframe, window_size, version,
                  centroids, intra_dists, cluster_labels, member_counts,
                  representative_ohlcs, outcome_stats_by_cluster,
                  starts, ends, cosine_dists, labels_all, returns_all,
                  horizons_data):
    """
    Write archetypes, outcomes, and occurrences to PostgreSQL.
    Batch insert to keep memory low.
    """
    n_clusters = len(centroids)

    with conn.cursor() as cur:
        # Clean old data for this combo + version
        cur.execute("""
            DELETE FROM "ArchetypeOccurrences"
            WHERE "ArchetypeId" IN (
                SELECT "Id" FROM "CandleArchetypes"
                WHERE "Symbol" = %s AND "Timeframe" = %s
                  AND "WindowSize" = %s AND "Version" = %s
            )
        """, (symbol, timeframe, window_size, version))

        cur.execute("""
            DELETE FROM "ArchetypeOutcomes"
            WHERE "ArchetypeId" IN (
                SELECT "Id" FROM "CandleArchetypes"
                WHERE "Symbol" = %s AND "Timeframe" = %s
                  AND "WindowSize" = %s AND "Version" = %s
            )
        """, (symbol, timeframe, window_size, version))

        cur.execute("""
            DELETE FROM "CandleArchetypes"
            WHERE "Symbol" = %s AND "Timeframe" = %s
              AND "WindowSize" = %s AND "Version" = %s
        """, (symbol, timeframe, window_size, version))

        conn.commit()

        # Insert archetypes
        now = datetime.now(timezone.utc)
        archetype_ids = {}

        for ci in range(n_clusters):
            if member_counts[ci] == 0:
                continue

            code = f"A-{ci:04d}"
            centroid = centroids[ci]
            c_norm = float(np.linalg.norm(centroid))
            ohlc_json = json.dumps(representative_ohlcs.get(ci)) if ci in representative_ohlcs else None

            cur.execute("""
                INSERT INTO "CandleArchetypes"
                ("Symbol", "Timeframe", "WindowSize", "ClusterId", "ArchetypeCode",
                 "CentroidVector", "CentroidDim", "CentroidNorm",
                 "MemberCount", "IntraClusterDistance", "RepresentativeOhlcJson",
                 "Version", "CreatedAtUtc")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING "Id"
            """, (
                symbol, timeframe, window_size, ci, code,
                centroid.tolist(), len(centroid), c_norm,
                int(member_counts[ci]), float(intra_dists[ci]), ohlc_json,
                version, now,
            ))
            archetype_ids[ci] = cur.fetchone()[0]

        conn.commit()
        print(f"  Inserted {len(archetype_ids)} archetypes")

        # Insert outcomes
        outcome_count = 0
        for ci, arch_id in archetype_ids.items():
            for horizon, stats in outcome_stats_by_cluster.get(ci, {}).items():
                if stats is None:
                    continue
                cur.execute("""
                    INSERT INTO "ArchetypeOutcomes"
                    ("ArchetypeId", "Horizon", "TotalSamples",
                     "UpCount", "DownCount", "SidewaysCount",
                     "UpRate", "DownRate", "SidewaysRate",
                     "AvgReturnPct", "MedianReturnPct", "MaxReturnPct",
                     "MinReturnPct", "StdDevReturnPct",
                     "RecentSamples", "RecentUpRate", "RecentDownRate",
                     "RecentAvgReturnPct", "CreatedAtUtc")
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    arch_id, horizon, stats["total"],
                    stats["up"], stats["down"], stats["sideways"],
                    stats["up_rate"], stats["down_rate"], stats["sideways_rate"],
                    stats["avg_return"], stats["median_return"], stats["max_return"],
                    stats["min_return"], stats["std_return"],
                    stats["recent_n"], stats["recent_up_rate"], stats["recent_down_rate"],
                    stats["recent_avg_return"], now,
                ))
                outcome_count += 1

        conn.commit()
        print(f"  Inserted {outcome_count} outcome records")

        # Insert occurrences in batches (use the primary horizon data for label/return)
        # We use the first horizon in the data as the "primary" for occurrence records
        primary_horizon = HORIZONS[1]  # "4h" as primary
        occ_batch = []
        occ_count = 0

        for idx in range(len(cluster_labels)):
            ci = int(cluster_labels[idx])
            if ci not in archetype_ids:
                continue

            occ_batch.append((
                archetype_ids[ci], symbol, timeframe, window_size,
                int(starts[idx]), int(ends[idx]),
                float(cosine_dists[idx]),
                int(labels_all[idx]),
                float(returns_all[idx]) if returns_all[idx] != 0.0 else None,
                primary_horizon, now,
            ))

            if len(occ_batch) >= 2000:
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO "ArchetypeOccurrences"
                    ("ArchetypeId", "Symbol", "Timeframe", "WindowSize",
                     "WindowStartMs", "WindowEndMs", "DistanceToCentroid",
                     "Label", "TargetReturn", "Horizon", "CreatedAtUtc")
                    VALUES %s""",
                    occ_batch,
                    page_size=2000,
                )
                occ_count += len(occ_batch)
                occ_batch.clear()

        if occ_batch:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO "ArchetypeOccurrences"
                ("ArchetypeId", "Symbol", "Timeframe", "WindowSize",
                 "WindowStartMs", "WindowEndMs", "DistanceToCentroid",
                 "Label", "TargetReturn", "Horizon", "CreatedAtUtc")
                VALUES %s""",
                occ_batch,
                page_size=2000,
            )
            occ_count += len(occ_batch)

        conn.commit()
        print(f"  Inserted {occ_count} occurrence records")


# --- Main pipeline -----------------------------------------------------------

def process_combo(conn, symbol, timeframe, window_size, version):
    """Process one (timeframe, windowSize) combination."""
    print(f"\n{'='*60}")
    print(f"Processing {symbol} {timeframe} ws={window_size}")
    print(f"{'='*60}")

    # Use the primary horizon for clustering (4h for intra-day, 1d for daily)
    primary_horizon = "1d" if timeframe == "1d" else "4h"
    t0 = time.time()

    X, y, starts, ends, rets = fetch_window_data(
        conn, symbol, timeframe, window_size, primary_horizon
    )
    if X is None or len(X) < 50:
        print(f"  Skipping: insufficient data ({0 if X is None else len(X)} rows)")
        return

    n_samples, feat_dim = X.shape
    print(f"  Loaded {n_samples} windows, {feat_dim} features")

    # Determine K
    k = compute_k(n_samples)
    print(f"  Using K={k} clusters")

    # Cluster
    cluster_labels, centroids, intra_dists, X_scaled, scaled_centroids = cluster_windows(X, k)

    # Member counts
    member_counts = np.zeros(k, dtype=np.int32)
    for ci in range(k):
        member_counts[ci] = int(np.sum(cluster_labels == ci))

    # Find representatives and fetch OHLC
    print(f"  Finding representative candles...")
    representative_ohlcs = {}
    for ci in range(k):
        if member_counts[ci] == 0:
            continue
        rep_idx = find_representative(X_scaled, scaled_centroids, cluster_labels, ci)
        if rep_idx is not None:
            ohlc = fetch_representative_ohlc(
                conn, symbol, timeframe, int(starts[rep_idx]), window_size
            )
            if ohlc:
                representative_ohlcs[ci] = ohlc

    print(f"  Found OHLC for {len(representative_ohlcs)}/{k} clusters")

    # Compute cosine distances for occurrences
    cosine_dists = compute_cosine_distances(X, centroids, cluster_labels, k)

    # Compute outcome stats per cluster per horizon
    recent_cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_MONTHS * 30)
    recent_cutoff_ms = int(recent_cutoff.timestamp() * 1000)

    outcome_stats_by_cluster = {}
    horizons_data = {}

    for horizon in HORIZONS:
        print(f"  Computing outcomes for horizon={horizon}...")
        if horizon == primary_horizon:
            h_y, h_rets, h_ends = y, rets, ends
        else:
            h_X, h_y, h_starts, h_ends, h_rets = fetch_window_data(
                conn, symbol, timeframe, window_size, horizon
            )
            if h_X is None:
                print(f"    No data for horizon {horizon}, skipping")
                continue
            # Re-use cluster assignments from primary horizon
            # But we need to match windows by start_ms
            # Build lookup: start_ms → index in primary data
            primary_lookup = {int(s): i for i, s in enumerate(starts)}
            # Filter horizon data to only windows that exist in primary
            mask = np.array([int(s) in primary_lookup for s in h_starts])
            if mask.sum() == 0:
                del h_X, h_y, h_starts, h_ends, h_rets
                gc.collect()
                continue
            h_y = h_y[mask]
            h_rets = h_rets[mask]
            h_ends_filtered = h_ends[mask]
            h_starts_filtered = h_starts[mask]
            # Map to cluster labels
            h_cluster_labels = np.array(
                [cluster_labels[primary_lookup[int(s)]] for s in h_starts_filtered],
                dtype=np.int32
            )
            del h_X, h_starts, h_ends
            gc.collect()

            # Compute per-cluster stats for this horizon
            for ci in range(k):
                if member_counts[ci] == 0:
                    continue
                ci_mask = h_cluster_labels == ci
                if ci_mask.sum() == 0:
                    continue
                stats = compute_outcome_stats(
                    h_y[ci_mask], h_rets[ci_mask],
                    h_ends_filtered[ci_mask], recent_cutoff_ms
                )
                if ci not in outcome_stats_by_cluster:
                    outcome_stats_by_cluster[ci] = {}
                outcome_stats_by_cluster[ci][horizon] = stats

            del h_y, h_rets, h_ends_filtered, h_starts_filtered, h_cluster_labels
            gc.collect()
            continue

        # For primary horizon, compute directly
        for ci in range(k):
            if member_counts[ci] == 0:
                continue
            ci_mask = cluster_labels == ci
            stats = compute_outcome_stats(
                h_y[ci_mask], h_rets[ci_mask],
                h_ends[ci_mask], recent_cutoff_ms
            )
            if ci not in outcome_stats_by_cluster:
                outcome_stats_by_cluster[ci] = {}
            outcome_stats_by_cluster[ci][horizon] = stats

    # Write to DB
    print(f"  Writing results to database...")
    write_results(
        conn, symbol, timeframe, window_size, version,
        centroids, intra_dists, cluster_labels, member_counts,
        representative_ohlcs, outcome_stats_by_cluster,
        starts, ends, cosine_dists, y, rets,
        horizons_data,
    )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {k} archetypes, {n_samples} occurrences")

    # Free memory
    del X, y, starts, ends, rets, X_scaled, cluster_labels, centroids
    del cosine_dists, intra_dists, member_counts, representative_ohlcs
    del outcome_stats_by_cluster, scaled_centroids
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Cluster candle windows into archetypes")
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES),
                        help="Comma-separated timeframes (default: 1h,4h)")
    parser.add_argument("--window-sizes", default=",".join(str(w) for w in DEFAULT_WINDOW_SIZES),
                        help="Comma-separated window sizes (default: 10,15,20,25)")
    parser.add_argument("--version", type=int, default=1,
                        help="Archetype version number (default: 1)")
    parser.add_argument("--symbol", default=SYMBOL,
                        help=f"Trading symbol (default: {SYMBOL})")
    args = parser.parse_args()

    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    window_sizes = [int(w.strip()) for w in args.window_sizes.split(",") if w.strip()]

    print(f"Candle Archetype Clustering")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {timeframes}")
    print(f"Window sizes: {window_sizes}")
    print(f"Version: {args.version}")
    print()

    conn = get_connection()
    conn.autocommit = False

    total_start = time.time()
    combos_done = 0

    for tf in timeframes:
        for ws in window_sizes:
            try:
                process_combo(conn, args.symbol, tf, ws, args.version)
                combos_done += 1
            except Exception as e:
                print(f"\n  ERROR processing {tf} ws={ws}: {e}")
                conn.rollback()
                import traceback
                traceback.print_exc()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All done! {combos_done} combos processed in {total_elapsed:.1f}s")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
