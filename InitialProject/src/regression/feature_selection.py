"""
Feature selection for regression:
  - Reads best BayesOpt params from lightgbm_tuning_results.json
  - Trains LGBMRegressor on all 140 with tuned params
  - Ranks by gain + split + SHAP → outputs top 20
"""

import json
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
import shap

from config import OUTPUT_DIR, RANDOM_STATE, N_TOP_FEATURES
from preprocessing import load_train_data, split_data, log_transform, compute_metrics


def main():
    print("=" * 60)
    print(f"Feature Selection: rank all 140 → pick {N_TOP_FEATURES}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load best tuning params
    tuning_path = OUTPUT_DIR / "lightgbm_tuning_results.json"
    if tuning_path.exists():
        with open(tuning_path) as f:
            tuning = json.load(f)
        best_params = tuning["bayesopt"]["params"]
        print("[Config] Using tuned params from lightgbm_tuning_results.json")
    else:
        # Default: run tuning first
        print("[Config] WARNING: no tuning results found, using default params.")
        print("         Run: python models/lightgbm_tune.py first")
        best_params = {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 500}

    # Remove n_estimators from params if auto-loaded (we set it explicitly)
    best_params_use = {k: v for k, v in best_params.items() if k != "n_estimators"}
    print(f"    Params: {best_params_use}")

    X, y_raw = load_train_data()
    all_features = list(X.columns)
    y = log_transform(y_raw)
    print(f"\n[1] Data: {len(X)} samples, {len(all_features)} features (after cleaning)")

    X_train, X_val, y_train, y_val = split_data(X, y)

    print("[2] Training LightGBM regressor on all 140 (tuned params)...")
    model = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        n_estimators=500, random_state=RANDOM_STATE,
        verbose=-1, force_col_wise=True,
        **best_params_use,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=20, verbose=False)],
    )
    preds = model.predict(X_val)
    metrics = compute_metrics(log_transform(y_val, inverse=True),
                              log_transform(preds, inverse=True))
    print(f"    RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R²={metrics['r2']:.4f}")

    # Built-in importance
    print("[3] Extracting built-in importance...")
    gain = dict(zip(all_features, model.booster_.feature_importance(importance_type="gain")))
    split = dict(zip(all_features, model.booster_.feature_importance(importance_type="split")))

    # SHAP
    print("[4] Computing SHAP (TreeExplainer, 2000 val samples)...")
    n_shap = min(2000, len(X_val))
    X_shap = X_val.sample(n=n_shap, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    shap_imp = dict(zip(all_features, np.abs(shap_vals).mean(axis=0)))

    # Combined rank
    print("[5] Combining rankings...")
    def rank_pos(feat, ranked):
        for i, (name, _) in enumerate(ranked):
            if name == feat:
                return i + 1
        return len(ranked)

    gain_rank = sorted(gain.items(), key=lambda x: x[1], reverse=True)
    split_rank = sorted(split.items(), key=lambda x: x[1], reverse=True)
    shap_rank = sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)

    combined = {}
    for feat in all_features:
        combined[feat] = {
            "avg_rank": (rank_pos(feat, gain_rank) + rank_pos(feat, split_rank) + rank_pos(feat, shap_rank)) / 3.0,
            "gain": float(gain.get(feat, 0)),
            "split": float(split.get(feat, 0)),
            "shap": float(shap_imp.get(feat, 0)),
        }

    sorted_combined = sorted(combined.items(), key=lambda x: x[1]["avg_rank"])
    top_n = [feat for feat, _ in sorted_combined[:N_TOP_FEATURES]]

    # Train on top N for comparison
    model_top = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        n_estimators=500, random_state=RANDOM_STATE,
        verbose=-1, force_col_wise=True,
        **best_params_use,
    )
    model_top.fit(
        X_train[top_n], y_train,
        eval_set=[(X_val[top_n], y_val)], eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=20, verbose=False)],
    )
    preds_top = model_top.predict(X_val[top_n])
    metrics_top = compute_metrics(log_transform(y_val, inverse=True),
                                  log_transform(preds_top, inverse=True))

    print(f"\n    Top {N_TOP_FEATURES} features (by avg rank):")
    for i, (feat, info) in enumerate(sorted_combined[:N_TOP_FEATURES]):
        print(f"      {i+1:2d}. {feat:45s}  rank={info['avg_rank']:.1f}  gain={info['gain']:.0f}  |SHAP|={info['shap']:.4f}")

    json.dump(combined, open(OUTPUT_DIR / "feature_ranking.json", "w"), indent=2)
    with open(OUTPUT_DIR / "top20_features.txt", "w") as f:
        for feat in top_n:
            f.write(f"{feat}\n")

    comparison = {
        "all_140": metrics,
        f"top_{N_TOP_FEATURES}": metrics_top,
        f"top_{N_TOP_FEATURES}_features": top_n,
    }
    json.dump(comparison, open(OUTPUT_DIR / "feature_comparison.json", "w"), indent=2)

    print(f"\n    All 140: RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R²={metrics['r2']:.4f}")
    print(f"    Top {N_TOP_FEATURES:2d}: RMSE={metrics_top['rmse']:.2f}  MAE={metrics_top['mae']:.2f}  R²={metrics_top['r2']:.4f}")
    print(f"\nSaved: top20_features.txt, feature_ranking.json, feature_comparison.json")
    print("Done.\n")


if __name__ == "__main__":
    main()
