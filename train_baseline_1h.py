#!/usr/bin/env python3
"""
Train lightweight baselines on WindowClassificationDatasets (1h) to verify
that the derived ML data actually contains a learnable signal.

Reads from PostgreSQL, time-based split, trains LogisticRegression and
HistGradientBoostingClassifier per (window_size, horizon), compares against
random & majority baselines. Outputs console report and writes
ai/baseline_1h_report.md.
"""

import os
import sys
import time
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- Config ------------------------------------------------------------------

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "bitcoin_analyst")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "123456")

SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
SPLIT_TIMESTAMP_MS = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

# 8 features per bar, in order produced by WindowDatasetService.BuildFeatureVector
FEATURE_NAMES = [
    "ClosePctChange1",
    "BodyPct",
    "HighLowRangePct",
    "Rsi14",
    "MacdHistogramNorm",
    "Ema12Dist",
    "Ema26Dist",
    "VolumeZscore",
]

LABEL_NAMES = {-1: "Down", 0: "Sideways", 1: "Up"}

REPORT_PATH = Path(__file__).with_name("baseline_1h_report.md")


# --- DB helpers --------------------------------------------------------------

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )


def fetch_data(symbol, timeframe):
    """Stream rows from WindowClassificationDatasets."""
    conn = get_connection()
    cur = conn.cursor(name="baseline_cursor")
    cur.itersize = 5000
    cur.execute(
        """
        SELECT "WindowSize", "Horizon", "WindowStartMs", "WindowEndMs",
               "FeatureVector", "Label", "TargetReturn", "WindowNullRatio"
        FROM "WindowClassificationDatasets"
        WHERE "Symbol" = %s AND "Timeframe" = %s
        ORDER BY "WindowEndMs"
        """,
        (symbol, timeframe),
    )

    rows = []
    total = 0
    start = time.time()
    for row in cur:
        rows.append(row)
        total += 1
        if total % 10000 == 0:
            elapsed = time.time() - start
            print(f"  loaded {total} rows... ({elapsed:.1f}s)")
    cur.close()
    conn.close()
    return rows


# --- Feature engineering -----------------------------------------------------

def build_feature_names(window_size):
    names = []
    for i in range(window_size):
        for f in FEATURE_NAMES:
            names.append(f"ws{window_size}_bar{i}_{f}")
    return names


def rows_to_groups(rows):
    """Group rows by (window_size, horizon)."""
    groups = defaultdict(list)
    for ws, horizon, w_start, w_end, vec, label, target_ret, null_ratio in rows:
        if vec is None or len(vec) == 0:
            continue
        if label not in (-1, 0, 1):
            continue
        groups[(ws, horizon)].append(
            (np.array(vec, dtype=np.float32), int(label), int(w_end))
        )

    result = {}
    for key, samples in groups.items():
        X = np.vstack([s[0] for s in samples])
        y = np.array([s[1] for s in samples], dtype=np.int8)
        ends = np.array([s[2] for s in samples], dtype=np.int64)
        result[key] = {"X": X, "y": y, "ends": ends}
    return result


# --- Models & evaluation -----------------------------------------------------

def time_split(X, y, ends, split_ms):
    train_mask = ends < split_ms
    test_mask = ~train_mask
    return (
        X[train_mask], y[train_mask],
        X[test_mask], y[test_mask],
        int(train_mask.sum()),
        int(test_mask.sum()),
    )


def majority_baseline(y_train, y_test):
    maj = Counter(y_train).most_common(1)[0][0]
    pred = np.full_like(y_test, maj)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="weighted", zero_division=0)


def random_baseline(y_train, y_test):
    classes, counts = np.unique(y_train, return_counts=True)
    probs = counts / counts.sum()
    rng = np.random.RandomState(42)
    pred = rng.choice(classes, size=len(y_test), p=probs)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="weighted", zero_division=0)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[-1, 0, 1])
    return {
        "name": name,
        "accuracy": acc,
        "f1_weighted": f1,
        "fit_time_s": fit_time,
        "confusion_matrix": cm.tolist(),
    }


def top_features_lr(coef, feature_names, n=10):
    mean_abs = np.mean(np.abs(coef), axis=0)
    idx = np.argsort(mean_abs)[::-1][:n]
    return [(feature_names[i], float(mean_abs[i])) for i in idx]


def top_features_gb(model, feature_names, n=10):
    try:
        imp = model.feature_importances_
    except AttributeError:
        # Older sklearn versions may not expose feature_importances_ for HGB.
        return []
    idx = np.argsort(imp)[::-1][:n]
    return [(feature_names[i], float(imp[i])) for i in idx]


# --- Report ------------------------------------------------------------------

def make_report(group_results):
    # Aggregate per horizon
    horizon_groups = defaultdict(list)
    for (ws, horizon), r in group_results.items():
        horizon_groups[horizon].append((ws, r))

    lines = [
        "# Baseline Training Report — WindowClassificationDatasets 1h",
        "",
        f"Generated: {datetime.utcnow().isoformat()} UTC",
        f"Symbol: {SYMBOL}, Timeframe: {TIMEFRAME}",
        f"Split: train on WindowEndMs < {SPLIT_TIMESTAMP_MS} (2025-01-01 UTC), test >= split",
        "",
        "## Goal",
        "",
        "Verify that the derived window-classification dataset contains a learnable signal ",
        "by training lightweight models per (window_size, horizon) and comparing them to ",
        "random (33.3%) and majority-class baselines.",
        "",
        "## Summary by Horizon",
        "",
        "| Horizon | Window sizes | Total samples | Best model | Best accuracy | Mean LR acc | Mean GB acc |",
        "|---------|--------------|---------------|------------|---------------|-------------|-------------|",
    ]

    for horizon in sorted(horizon_groups.keys()):
        items = horizon_groups[horizon]
        total = sum(r["total"] for _, r in items)
        window_sizes = ",".join(str(ws) for ws, _ in items)

        lr_accs = [r["models"][2]["accuracy"] for _, r in items]
        gb_accs = [r["models"][3]["accuracy"] for _, r in items]

        best_per_ws = []
        for _, r in items:
            best_per_ws.append(max(r["models"][2]["accuracy"], r["models"][3]["accuracy"]))
        best_acc = max(best_per_ws)
        best_model_name = "GB" if max(gb_accs) >= max(lr_accs) else "LR"

        lines.append(
            f"| {horizon} | {window_sizes} | {total} | {best_model_name} | {best_acc:.4f} | {np.mean(lr_accs):.4f} | {np.mean(gb_accs):.4f} |"
        )

    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")

    for horizon in sorted(horizon_groups.keys()):
        lines.append(f"### Horizon {horizon}")
        lines.append("")
        for ws, r in sorted(horizon_groups[horizon], key=lambda x: x[0]):
            lines.append(f"#### Window size {ws}")
            lines.append("")
            lines.append(f"- Total samples: {r['total']}")
            lines.append(f"- Train samples: {r['train_count']}, Test samples: {r['test_count']}")
            lines.append(f"- Label distribution (train): {dict(r['label_dist_train'])}")
            lines.append(f"- Label distribution (test):  {dict(r['label_dist_test'])}")
            lines.append("")
            lines.append("| Model | Accuracy | F1-weighted | Fit time (s) |")
            lines.append("|-------|----------|-------------|--------------|")
            for m in r["models"]:
                lines.append(
                    f"| {m['name']} | {m['accuracy']:.4f} | {m['f1_weighted']:.4f} | {m['fit_time_s']:.2f} |"
                )
            lines.append("")
            lines.append("Top 10 features (LogisticRegression mean |coef|):")
            for name, score in r["top_features_lr"]:
                lines.append(f"- {name}: {score:.6f}")
            lines.append("")
            lines.append("Top 10 features (GradientBoosting importance):")
            for name, score in r["top_features_gb"]:
                lines.append(f"- {name}: {score:.6f}")
            lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- If LogisticRegression / GradientBoosting consistently beat random (0.333) and majority-class, "
        "the dataset has a real predictive signal."
    )
    lines.append(
        "- If models are close to majority-class, the features are not informative beyond class imbalance."
    )
    lines.append(
        "- If models are close to random, labels may be noisy or features may not encode direction well."
    )
    lines.append("")

    return "\n".join(lines)


# --- Main --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Baseline training on WindowClassificationDatasets (1h)")
    print("=" * 60)

    print("\n[1/4] Fetching data from PostgreSQL...")
    rows = fetch_data(SYMBOL, TIMEFRAME)
    print(f"Total rows fetched: {len(rows)}")
    if len(rows) == 0:
        print("No data found. Aborting.")
        sys.exit(1)

    print("\n[2/4] Converting to feature arrays...")
    groups = rows_to_groups(rows)
    for (ws, horizon), d in sorted(groups.items()):
        print(f"  (ws={ws}, h={horizon}): X shape={d['X'].shape}, y shape={d['y'].shape}")

    print("\n[3/4] Training & evaluating models...")
    group_results = {}
    for (ws, horizon), d in sorted(groups.items()):
        X, y, ends = d["X"], d["y"], d["ends"]
        feature_names = build_feature_names(ws)

        X_train, y_train, X_test, y_test, n_train, n_test = time_split(X, y, ends, SPLIT_TIMESTAMP_MS)

        # If test set is tiny, fall back to 80/20 time split
        if n_test < max(100, int(0.05 * len(y))):
            split_idx = int(0.8 * len(y))
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_test, y_test = X[split_idx:], y[split_idx:]
            n_train, n_test = len(y_train), len(y_test)
            print(f"  (ws={ws}, h={horizon}): test set too small; used 80/20 time split")

        print(f"\n  (ws={ws}, horizon={horizon}): total={len(y)}, train={n_train}, test={n_test}")
        print(f"    label dist train: {Counter(y_train)}")
        print(f"    label dist test:  {Counter(y_test)}")

        maj_acc, maj_f1 = majority_baseline(y_train, y_test)
        rand_acc, rand_f1 = random_baseline(y_train, y_test)

        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, n_jobs=-1, random_state=42)),
        ])
        lr_res = evaluate_model("LogisticRegression", lr, X_train, y_train, X_test, y_test)
        lr_res["top_features"] = top_features_lr(lr.named_steps["clf"].coef_, feature_names)

        gb = HistGradientBoostingClassifier(random_state=42)
        gb_res = evaluate_model("GradientBoosting", gb, X_train, y_train, X_test, y_test)
        gb_res["top_features"] = top_features_gb(gb, feature_names)

        models = [
            {"name": "MajorityClass", "accuracy": maj_acc, "f1_weighted": maj_f1, "fit_time_s": 0.0, "confusion_matrix": []},
            {"name": "Random", "accuracy": rand_acc, "f1_weighted": rand_f1, "fit_time_s": 0.0, "confusion_matrix": []},
            lr_res,
            gb_res,
        ]

        for m in models:
            print(f"    {m['name']:22s} acc={m['accuracy']:.4f}  f1={m['f1_weighted']:.4f}  time={m['fit_time_s']:.2f}s")

        group_results[(ws, horizon)] = {
            "total": len(y),
            "train_count": n_train,
            "test_count": n_test,
            "label_dist_train": dict(Counter(y_train)),
            "label_dist_test": dict(Counter(y_test)),
            "models": models,
            "top_features_lr": lr_res["top_features"],
            "top_features_gb": gb_res["top_features"],
        }

    print("\n[4/4] Writing report...")
    report = make_report(group_results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report saved to: {REPORT_PATH}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
