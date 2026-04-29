"""
Step 1 - Feature Ranking:
Train LightGBM on all 140 features, rank them via built-in importance + SHAP,
pick top 15, and compare performance (all-140 vs top-15).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from lightgbm import LGBMClassifier, early_stopping

from utils import (
    load_train_data,
    split_data,
    compute_metrics,
    OUTPUT_DIR,
)

random_state = 42
np.random.seed(random_state)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_lightgbm_on_features(X_train, y_train, X_val, y_val, feature_cols):
    """Train LightGBM with default-ish params, return model + predictions on val set."""
    model = LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=random_state,
        verbose=-1,
        device_type="cpu",
        force_col_wise=True,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="logloss",
        callbacks=[early_stopping(stopping_rounds=20, verbose=False)],
    )

    y_pred = model.predict_proba(X_val)[:, 1]
    return model, y_pred


def get_builtin_importance(model, feature_cols):
    """Extract gain and split importance from trained LightGBM."""
    gain = dict(zip(feature_cols, model.booster_.feature_importance(importance_type="gain")))
    split = dict(zip(feature_cols, model.booster_.feature_importance(importance_type="split")))
    return gain, split


def get_shap_importance(model, X_sample, feature_cols):
    """Compute mean |SHAP| values on a sample using TreeExplainer."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # class 0 vs 1; take positive class
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    return dict(zip(feature_cols, mean_abs_shap))


def rank_by_score(score_dict):
    """Return features sorted by descending score."""
    return sorted(score_dict.items(), key=lambda x: x[1], reverse=True)


def get_rank_position(feature, ranked_list):
    """Get 0-indexed rank of feature in ranked list."""
    for i, (name, _) in enumerate(ranked_list):
        if name == feature:
            return i + 1
    return len(ranked_list)


def main():
    print("=" * 60)
    print("Step 1: Feature Ranking")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    X, y = load_train_data()
    all_features = list(X.columns)
    print(f"    Features: {len(all_features)}, Samples: {len(X)}")
    print(f"    Electron fraction: {y.mean():.4f}")

    # Split
    X_train, X_val, y_train, y_val = split_data(X, y)
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}")

    # Train on ALL 140 features
    print("\n[2] Training LightGBM on all 140 features...")
    model_all, preds_all = train_lightgbm_on_features(
        X_train, y_train, X_val, y_val, all_features
    )
    metrics_all = compute_metrics(y_val, preds_all)
    print(f"    LogLoss (all 140): {metrics_all['log_loss']:.6f}")
    print(f"    ROC AUC  (all 140): {metrics_all['roc_auc']:.6f}")

    # Built-in importance
    print("\n[3] Computing feature importance...")
    gain_imp, split_imp = get_builtin_importance(model_all, all_features)
    gain_rank = rank_by_score(gain_imp)
    split_rank = rank_by_score(split_imp)

    # SHAP importance (on a subset for speed)
    print("[4] Computing SHAP values (on 2000 validation samples)...")
    n_shap = min(2000, len(X_val))
    X_shap = X_val.sample(n=n_shap, random_state=random_state)
    shap_imp = get_shap_importance(model_all, X_shap, all_features)
    shap_rank = rank_by_score(shap_imp)

    # Combined ranking: average of 3 rank positions
    print("[5] Combining rankings...")
    combined = {}
    for feat in all_features:
        r_gain = get_rank_position(feat, gain_rank)
        r_split = get_rank_position(feat, split_rank)
        r_shap = get_rank_position(feat, shap_rank)
        combined[feat] = {
            "gain_rank": r_gain,
            "split_rank": r_split,
            "shap_rank": r_shap,
            "avg_rank": (r_gain + r_split + r_shap) / 3.0,
            "gain_val": float(gain_imp.get(feat, 0)),
            "shap_val": float(shap_imp.get(feat, 0)),
        }

    sorted_combined = sorted(combined.items(), key=lambda x: x[1]["avg_rank"])

    # Top 15
    top_15 = [feat for feat, _ in sorted_combined[:15]]
    print(f"\n    Top 15 features (by average rank):")
    for i, (feat, info) in enumerate(sorted_combined[:15]):
        print(
            f"      {i+1:2d}. {feat:40s}  "
            f"avg_rank={info['avg_rank']:.1f}  "
            f"gain={info['gain_val']:.0f}  "
            f"shap={info['shap_val']:.4f}"
        )

    # Save rankings
    ranking_path = OUTPUT_DIR / "feature_ranking.json"
    with open(ranking_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n    Rankings saved to: {ranking_path}")

    # Save top 15 list
    top15_path = OUTPUT_DIR / "top15_features.txt"
    with open(top15_path, "w") as f:
        for feat in top_15:
            f.write(f"{feat}\n")

    # Compare: train on top 15 only
    print("\n[6] Training LightGBM on top 15 features...")
    X_train_top = X_train[top_15]
    X_val_top = X_val[top_15]

    model_top, preds_top = train_lightgbm_on_features(
        X_train_top, y_train, X_val_top, y_val, top_15
    )
    metrics_top = compute_metrics(y_val, preds_top)
    print(f"    LogLoss (top 15): {metrics_top['log_loss']:.6f}")
    print(f"    ROC AUC  (top 15): {metrics_top['roc_auc']:.6f}")

    # Comparison summary
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"  {'':20s} {'All 140':>12s} {'Top 15':>12s}")
    print(f"  {'LogLoss':20s} {metrics_all['log_loss']:12.6f} {metrics_top['log_loss']:12.6f}")
    print(f"  {'ROC AUC':20s} {metrics_all['roc_auc']:12.6f} {metrics_top['roc_auc']:12.6f}")
    print(f"  {'Delta':20s} {'':>12s} {metrics_top['log_loss'] - metrics_all['log_loss']:+12.6f}")

    # Save comparison
    comparison = {
        "all_140": metrics_all,
        "top_15": metrics_top,
        "top_15_features": top_15,
    }
    comparison_path = OUTPUT_DIR / "feature_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to: {comparison_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()
