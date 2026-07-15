#!/usr/bin/env python3
"""
Advanced baseline training for WindowClassificationDatasets.

Supports multiple models, class imbalance handling, and hyperparameter search.
Reads from PostgreSQL, uses time-based split, evaluates against random/majority.
"""

import os
import sys
import time
import json
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight

# Optional models
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore", category=UserWarning)

# --- Config ------------------------------------------------------------------

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "bitcoin_analyst")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "123456")

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
SPLIT_TIMESTAMP_MS = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
ENABLE_TUNING = os.getenv("ENABLE_TUNING", "0").lower() in ("1", "true", "yes")
WINDOW_SIZES = [int(x) for x in os.getenv("WINDOW_SIZES", "5,10,15,20,25").split(",") if x.strip()]
HORIZONS = [x.strip() for x in os.getenv("HORIZONS", "1h,4h,1d").split(",") if x.strip()]

# Must match WindowDatasetService.FeatureNames in backend/Services/WindowDatasetService.cs
# This list is used when FeatureDim indicates the new 35-feature vector.
FEATURE_NAMES_35 = [
    "CloseZscore", "ClosePctChange1", "ClosePctChange4", "ClosePctChange24",
    "HighLowRangePct", "BodyPct", "UpperWickPct", "LowerWickPct",
    "Rsi14", "Rsi14Slope", "MacdNorm", "MacdSignalNorm", "MacdHistogramNorm",
    "Ema12Dist", "Ema26Dist", "Ema50Dist", "Ema200Dist",
    "Sma50Dist", "Sma200Dist", "BollingerWidth", "BollingerPosition",
    "Atr14Pct", "ObvEmaDist", "VwapDist", "RollingVwapDist",
    "VolumeZscore", "VolumeSma20Ratio", "TakerBuyRatio",
    "RecentPatternEncoded", "ActiveRuleCount",
    "HourSin", "HourCos", "DayOfWeekSin", "DayOfWeekCos", "IsWeekend",
]

# Legacy 8-feature vector (before Phase 1.1/1.2)
FEATURE_NAMES_8 = [
    "ClosePctChange1", "BodyPct", "HighLowRangePct", "Rsi14",
    "MacdHistogramNorm", "Ema12Dist", "Ema26Dist", "VolumeZscore",
]

LABEL_NAMES = {-1: "Down", 0: "Sideways", 1: "Up"}

REPORT_PATH = Path(__file__).with_name("baseline_advanced_report.md")
MODELS_DIR = Path(__file__).with_name("models")
MODELS_DIR.mkdir(exist_ok=True)


# --- DB helpers --------------------------------------------------------------

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )


# --- Feature names -----------------------------------------------------------

def infer_feature_names(window_size, feature_dim):
    per_bar = feature_dim // window_size
    if per_bar == len(FEATURE_NAMES_35):
        return [f"ws{window_size}_bar{i}_{name}" for i in range(window_size) for name in FEATURE_NAMES_35]
    elif per_bar == len(FEATURE_NAMES_8):
        return [f"ws{window_size}_bar{i}_{name}" for i in range(window_size) for name in FEATURE_NAMES_8]
    else:
        return [f"ws{window_size}_bar{i}_f{j}" for i in range(window_size) for j in range(per_bar)]


# --- Data prep ---------------------------------------------------------------

def fetch_group_data(symbol, timeframe, window_size, horizon):
    """Load a single (window_size, horizon) group into memory-efficient arrays."""
    conn = get_connection()
    cur = conn.cursor(name=f"adv_cursor_{window_size}_{horizon}")
    cur.itersize = 5000
    cur.execute(
        """
        SELECT "FeatureVector", "Label", "WindowEndMs", "FeatureDim"
        FROM "WindowClassificationDatasets"
        WHERE "Symbol" = %s AND "Timeframe" = %s
          AND "WindowSize" = %s AND "Horizon" = %s
        ORDER BY "WindowEndMs"
        """,
        (symbol, timeframe, window_size, horizon),
    )
    X_list = []
    y_list = []
    ends_list = []
    feat_dim = None
    total = 0
    start = time.time()
    for vec, label, w_end, fd in cur:
        if vec is None or len(vec) == 0 or label not in (-1, 0, 1):
            continue
        X_list.append(np.array(vec, dtype=np.float32))
        y_list.append(int(label))
        ends_list.append(int(w_end))
        if feat_dim is None:
            feat_dim = int(fd)
        total += 1
        if total % 10000 == 0:
            print(f"    loaded {total} rows... ({time.time()-start:.1f}s)")
    cur.close()
    conn.close()

    if len(X_list) == 0:
        return None

    # Release intermediate arrays quickly by converting once.
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int8)
    ends = np.array(ends_list, dtype=np.int64)
    feature_names = infer_feature_names(window_size, feat_dim)
    return {"X": X, "y": y, "ends": ends, "feature_names": feature_names}


def time_split(X, y, ends, split_ms, fallback_ratio=0.8):
    train_mask = ends < split_ms
    test_mask = ~train_mask
    if int(test_mask.sum()) < max(100, int(0.05 * len(y))):
        split_idx = int(fallback_ratio * len(y))
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:], len(y) - split_idx
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask], int(test_mask.sum())


# --- Baselines & models ------------------------------------------------------

def majority_baseline(y_train, y_test):
    maj = Counter(y_train).most_common(1)[0][0]
    pred = np.full_like(y_test, maj)
    return evaluate("MajorityClass", pred, y_test, fit_time=0.0)


def random_baseline(y_train, y_test):
    classes, counts = np.unique(y_train, return_counts=True)
    probs = counts / counts.sum()
    rng = np.random.RandomState(42)
    pred = rng.choice(classes, size=len(y_test), p=probs)
    return evaluate("Random", pred, y_test, fit_time=0.0)


def evaluate(name, pred, y_test, fit_time=0.0):
    acc = accuracy_score(y_test, pred)
    f1_w = f1_score(y_test, pred, average="weighted", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, labels=[-1, 0, 1], zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[-1, 0, 1])
    return {
        "name": name,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1_per_class": f1.tolist(),
        "fit_time_s": fit_time,
        "confusion_matrix": cm.tolist(),
    }


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    pred = model.predict(X_test)
    return evaluate(name, pred, y_test, fit_time)


def build_models():
    models = []

    models.append(("LR_balanced", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1, random_state=42)),
    ])))

    models.append(("RF_balanced", RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=50,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )))

    models.append(("HGB_balanced", HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42,
    )))

    if HAS_LIGHTGBM:
        models.append(("LGB_balanced", lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            class_weight="balanced", n_jobs=-1, random_state=42, verbosity=-1,
        )))

    if HAS_XGBOOST:
        models.append(("XGB_balanced", xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            eval_metric="mlogloss",
            n_jobs=-1, random_state=42,
        )))

    return models


def remap_labels(y):
    """Map {-1,0,1} -> {0,1,2} for XGBoost and back."""
    mapping = {-1: 0, 0: 1, 1: 2}
    inv_mapping = {v: k for k, v in mapping.items()}
    mapped = np.array([mapping[v] for v in y], dtype=np.int8)
    return mapped, inv_mapping


def maybe_remap_for_model(name, y):
    """Return (y_for_model, inv_mapping_or_none)."""
    if "XGB" in name:
        return remap_labels(y)
    return y, None


def predict_and_map_back(name, model, X_test, inv_mapping):
    """Predict and map XGB labels back to {-1,0,1}."""
    pred = model.predict(X_test)
    if inv_mapping is not None:
        pred = np.array([inv_mapping[p] for p in pred], dtype=np.int8)
    return pred


# --- Imbalance handling ------------------------------------------------------

def apply_smote(X_train, y_train):
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(Counter(y_train).values()) - 1))
        return smote.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"    SMOTE failed: {e}")
        return X_train, y_train


def apply_undersample(X_train, y_train):
    try:
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        return rus.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"    Undersample failed: {e}")
        return X_train, y_train


# --- Hyperparameter search ---------------------------------------------------

def tune_best_model(model_name, model, X_train, y_train, X_test, y_test):
    """Light randomized search for the best model."""
    param_distributions = {}
    if "LGB" in model_name:
        param_distributions = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8, -1],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
        }
    elif "XGB" in model_name:
        param_distributions = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        }
    elif "RF" in model_name:
        param_distributions = {
            "n_estimators": [100, 200, 300],
            "max_depth": [8, 12, 16, None],
            "min_samples_leaf": [10, 50, 100],
        }
    else:
        return None

    search = RandomizedSearchCV(
        model, param_distributions, n_iter=10, scoring="f1_weighted",
        cv=3, random_state=42, n_jobs=-1, verbose=0,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    fit_time = time.time() - t0
    pred = search.predict(X_test)
    result = evaluate(f"{model_name}_tuned", pred, y_test, fit_time)
    result["best_params"] = search.best_params_
    return result


# --- Feature importance ------------------------------------------------------

def extract_importance(model, feature_names):
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.mean(np.abs(model.coef_), axis=0)
        else:
            return []
        idx = np.argsort(imp)[::-1][:15]
        return [(feature_names[i], float(imp[i])) for i in idx]
    except Exception:
        return []


# --- Model persistence -------------------------------------------------------

def save_model_and_metadata(
    symbol: str,
    timeframe: str,
    window_size: int,
    horizon: str,
    model_name: str,
    model,
    feature_names: list[str],
    metrics: dict,
    label_dist_train: dict,
    label_dist_test: dict,
    train_count: int,
    test_count: int,
) -> Path:
    """Persist trained model + metadata to ai/models/."""
    safe_model_name = model_name.replace("/", "_")
    base_name = f"{symbol}_{timeframe}_ws{window_size}_h{horizon}_{safe_model_name}"
    model_path = MODELS_DIR / f"{base_name}.joblib"
    meta_path = MODELS_DIR / f"{base_name}.json"

    joblib.dump(model, model_path)

    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "window_size": window_size,
        "horizon": horizon,
        "model_name": model_name,
        "model_file": model_path.name,
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "metrics": metrics,
        "label_dist_train": {str(k): int(v) for k, v in label_dist_train.items()},
        "label_dist_test": {str(k): int(v) for k, v in label_dist_test.items()},
        "train_count": train_count,
        "test_count": test_count,
        "split_timestamp_ms": SPLIT_TIMESTAMP_MS,
        "label_mapping": {"-1": "Down", "0": "Sideways", "1": "Up"},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


# --- Report ------------------------------------------------------------------

def make_report(results_by_group):
    lines = [
        "# Advanced Baseline Training Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()} UTC",
        f"Symbol: {SYMBOL}, Timeframe: {TIMEFRAME}",
        f"Split: train on WindowEndMs < {SPLIT_TIMESTAMP_MS} (2025-01-01 UTC), test >= split",
        "",
        "## Summary by (WindowSize, Horizon)",
        "",
        "| WS | Horizon | Samples | Best model | Best acc | Best F1 | Majority acc | Majority F1 |",
        "|----|---------|---------|------------|----------|---------|--------------|-------------|",
    ]

    for (ws, horizon), r in sorted(results_by_group.items()):
        models = [m for m in r["models"] if m["name"] not in ("MajorityClass", "Random")]
        if models:
            best = max(models, key=lambda x: x["f1_weighted"])
            lines.append(
                f"| {ws} | {horizon} | {r['total']} | {best['name']} | {best['accuracy']:.4f} | {best['f1_weighted']:.4f} | "
                f"{r['majority_acc']:.4f} | {r['majority_f1']:.4f} |"
            )

    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")

    for (ws, horizon), r in sorted(results_by_group.items()):
        lines.append(f"### WindowSize={ws}, Horizon={horizon}")
        lines.append("")
        lines.append(f"- Total: {r['total']}, Train: {r['train_count']}, Test: {r['test_count']}")
        lines.append(f"- Label distribution (train): {dict(r['label_dist_train'])}")
        lines.append(f"- Label distribution (test):  {dict(r['label_dist_test'])}")
        lines.append("")
        lines.append("| Model | Acc | F1-w | Fit(s) |")
        lines.append("|-------|-----|------|--------|")
        for m in r["models"]:
            lines.append(f"| {m['name']} | {m['accuracy']:.4f} | {m['f1_weighted']:.4f} | {m['fit_time_s']:.2f} |")
        lines.append("")
        lines.append("Top 10 important features:")
        for name, score in r["top_features"][:10]:
            lines.append(f"- {name}: {score:.6f}")
        lines.append("")

    lines.append("## Notes")
    lines.append("- 'balanced' = sklearn class_weight='balanced'.")
    lines.append("- SMOTE/undersample results only shown if imbalanced-learn is installed.")
    lines.append("- Time-based split prevents look-ahead leakage.")
    lines.append("")

    return "\n".join(lines)


# --- Main --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Advanced baseline training")
    print("=" * 60)

    print("\n[1/3] Training & evaluating...")
    results_by_group = {}

    for ws, horizon in sorted([(ws, h) for ws in WINDOW_SIZES for h in HORIZONS]):
        # Skip groups that already have a saved best model
        existing = list(MODELS_DIR.glob(f"{SYMBOL}_{TIMEFRAME}_ws{ws}_h{horizon}_*.joblib"))
        if existing:
            print(f"\n  Skipping (ws={ws}, h={horizon}) - model already exists: {existing[0].name}")
            continue

        print(f"\n  Loading (ws={ws}, h={horizon})...")
        d = fetch_group_data(SYMBOL, TIMEFRAME, ws, horizon)
        if d is None:
            print(f"    No data for (ws={ws}, h={horizon}), skipping.")
            continue
        print(f"    X={d['X'].shape}, features_per_bar={d['X'].shape[1] // ws}")
        X, y, ends = d["X"], d["y"], d["ends"]
        feature_names = d["feature_names"]

        X_train, y_train, X_test, y_test, n_test = time_split(X, y, ends, SPLIT_TIMESTAMP_MS)
        print(f"\n  (ws={ws}, h={horizon}): total={len(y)}, train={len(y_train)}, test={len(y_test)}")
        print(f"    labels train: {dict(Counter(y_train))}, test: {dict(Counter(y_test))}")

        maj_res = majority_baseline(y_train, y_test)
        rand_res = random_baseline(y_train, y_test)

        models = build_models()
        model_results = [maj_res, rand_res]
        top_features = []

        for name, model in models:
            try:
                y_train_m, inv = maybe_remap_for_model(name, y_train)
                y_test_m, _ = maybe_remap_for_model(name, y_test)
                res = train_and_evaluate(name, model, X_train, y_train_m, X_test, y_test_m)
                pred_back = predict_and_map_back(name, model, X_test, inv)
                res = evaluate(name, pred_back, y_test, res["fit_time_s"])
                model_results.append(res)
                print(f"    {name:18s} acc={res['accuracy']:.4f} f1={res['f1_weighted']:.4f} ({res['fit_time_s']:.1f}s)")
                if "Pipeline" in str(type(model)):
                    imp = extract_importance(model.named_steps["clf"], feature_names)
                else:
                    imp = extract_importance(model, feature_names)
                if imp:
                    top_features = imp
            except Exception as e:
                print(f"    {name} failed: {e}")

        # Try SMOTE on the best model candidate
        if HAS_LIGHTGBM or HAS_XGBOOST or True:
            try:
                X_sm, y_sm = apply_smote(X_train, y_train)
                if len(X_sm) != len(X_train):
                    model = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1, random_state=42)
                    scaler = StandardScaler()
                    X_sm_s = scaler.fit_transform(X_sm)
                    X_test_s = scaler.transform(X_test)
                    res = train_and_evaluate("LR_SMOTE", model, X_sm_s, y_sm, X_test_s, y_test)
                    model_results.append(res)
                    print(f"    LR_SMOTE           acc={res['accuracy']:.4f} f1={res['f1_weighted']:.4f}")
            except Exception as e:
                print(f"    SMOTE eval skipped: {e}")

        # Try undersampling
        try:
            X_us, y_us = apply_undersample(X_train, y_train)
            if len(X_us) != len(X_train):
                model = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1, random_state=42)
                scaler = StandardScaler()
                X_us_s = scaler.fit_transform(X_us)
                X_test_s = scaler.transform(X_test)
                res = train_and_evaluate("LR_Undersample", model, X_us_s, y_us, X_test_s, y_test)
                model_results.append(res)
                print(f"    LR_Undersample     acc={res['accuracy']:.4f} f1={res['f1_weighted']:.4f}")
        except Exception as e:
            print(f"    Undersample eval skipped: {e}")

        # Tune best model
        best_name = None
        best_f1 = 0
        for res in model_results:
            if res["name"] in ("MajorityClass", "Random"):
                continue
            if res["f1_weighted"] > best_f1:
                best_f1 = res["f1_weighted"]
                best_name = res["name"]

        if ENABLE_TUNING and best_name:
            # Re-instantiate best model and tune
            try:
                model_for_tune = dict(models).get(best_name)
                if model_for_tune is not None:
                    y_train_t, inv_t = maybe_remap_for_model(best_name, y_train)
                    y_test_t, _ = maybe_remap_for_model(best_name, y_test)
                    tuned_res = tune_best_model(best_name, model_for_tune, X_train, y_train_t, X_test, y_test_t)
                    if tuned_res:
                        pred_back = predict_and_map_back(best_name, model_for_tune, X_test, inv_t)
                        tuned_res = evaluate(tuned_res["name"], pred_back, y_test, tuned_res["fit_time_s"])
                        tuned_res["best_params"] = tuned_res.get("best_params", {})
                        model_results.append(tuned_res)
                        print(f"    {tuned_res['name']:18s} acc={tuned_res['accuracy']:.4f} f1={tuned_res['f1_weighted']:.4f} {tuned_res.get('best_params', {})}")
            except Exception as e:
                print(f"    Tuning skipped: {e}")

        # Persist best model
        best_res = next((m for m in model_results if m["name"] == best_name), None)
        if best_name and best_res:
            try:
                model_to_save = dict(models).get(best_name)
                if model_to_save is not None:
                    # Refit on full train data with original labels (or remapped for XGB)
                    y_train_full, _ = maybe_remap_for_model(best_name, y_train)
                    model_to_save.fit(X_train, y_train_full)
                    model_path = save_model_and_metadata(
                        SYMBOL, TIMEFRAME, ws, horizon, best_name, model_to_save,
                        feature_names, best_res,
                        Counter(y_train), Counter(y_test),
                        len(y_train), len(y_test),
                    )
                    print(f"    Saved best model: {model_path}")
            except Exception as e:
                print(f"    Model save skipped: {e}")

        results_by_group[(ws, horizon)] = {
            "total": len(y),
            "train_count": len(y_train),
            "test_count": len(y_test),
            "label_dist_train": dict(Counter(y_train)),
            "label_dist_test": dict(Counter(y_test)),
            "models": model_results,
            "top_features": top_features,
            "majority_acc": maj_res["accuracy"],
            "majority_f1": maj_res["f1_weighted"],
        }

    print("\n[2/3] Writing report...")
    report = make_report(results_by_group)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report: {REPORT_PATH}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
