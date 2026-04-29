"""
LightGBM regression tuning on ALL 140 features:
  - Default:  BayesOpt + Optuna hyperparameter search on all 140
              → saves best params to lightgbm_tuning_results.json
  - --final:  Load best params, train final LGBM on top-20 features
              → saves final models for submission
"""

import sys, json, joblib, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.model_selection import cross_val_score, KFold

from config import (
    OUTPUT_DIR, RANDOM_STATE,
    LGBM_CV_FOLDS, LGBM_EARLY_STOP, LGBM_N_ESTIMATORS_CV,
    LGBM_N_ESTIMATORS_FINAL, BAYESOPT_INIT, BAYESOPT_ITER, OPTUNA_TRIALS,
)
from preprocessing import (
    load_train_data, split_data, log_transform,
    compute_metrics, load_top_features,
)

TUNING_RESULT_PATH = OUTPUT_DIR / "lightgbm_tuning_results.json"


def lgbm_cv_score(params, X, y):
    """5-fold CV negated RMSE on log-scale."""
    model = lgb.LGBMRegressor(
        objective="regression", metric="rmse", random_state=RANDOM_STATE,
        verbose=-1, force_col_wise=True,
        **params,
    )
    scores = cross_val_score(
        model, X, y,
        cv=KFold(n_splits=LGBM_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error",
    )
    return scores.mean()


# ================================================================
# BayesOpt
# ================================================================
def tune_bayesopt(X, y):
    from bayes_opt import BayesianOptimization

    pbounds = {
        "num_leaves": (15, 127),
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "min_child_samples": (20, 120),
        "subsample": (0.5, 0.95),
        "colsample_bytree": (0.3, 0.95),
    }

    def objective(num_leaves, max_depth, learning_rate, min_child_samples,
                  subsample, colsample_bytree):
        params = {
            "n_estimators": LGBM_N_ESTIMATORS_CV,
            "num_leaves": int(num_leaves),
            "max_depth": int(max_depth),
            "learning_rate": learning_rate,
            "min_child_samples": int(min_child_samples),
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        return lgbm_cv_score(params, X, y)

    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=RANDOM_STATE)
    optimizer.maximize(init_points=BAYESOPT_INIT, n_iter=BAYESOPT_ITER)

    best = optimizer.max["params"]
    best["num_leaves"] = int(best["num_leaves"])
    best["max_depth"] = int(best["max_depth"])
    best["min_child_samples"] = int(best["min_child_samples"])
    best["n_estimators"] = LGBM_N_ESTIMATORS_FINAL
    return best, optimizer.max["target"]


# ================================================================
# Optuna
# ================================================================
def tune_optuna(X, y):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": LGBM_N_ESTIMATORS_CV,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "subsample": trial.suggest_float("subsample", 0.5, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.95),
        }
        return lgbm_cv_score(params, X, y)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best = study.best_params
    best["n_estimators"] = LGBM_N_ESTIMATORS_FINAL
    return best, study.best_value


# ================================================================
# Train & evaluate on validation set
# ================================================================
def train_and_eval(params, X, y):
    """Train on full X with given params, return val metrics (original units)."""
    X_train, X_val, y_train, y_val = split_data(X, y)
    model = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        random_state=RANDOM_STATE, verbose=-1, force_col_wise=True,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse",
              callbacks=[early_stopping(stopping_rounds=LGBM_EARLY_STOP, verbose=False)])

    val_preds_raw = log_transform(model.predict(X_val), inverse=True)
    y_val_raw = log_transform(y_val, inverse=True)
    return model, compute_metrics(y_val_raw, val_preds_raw)


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="Train final model on top-20 using saved best params")
    args = parser.parse_args()

    # ---- Final mode: just train on top-20 with saved params ----
    if args.final:
        print("=" * 60)
        print("Final LGBM Training (top-20, tuned params)")
        print("=" * 60)

        with open(TUNING_RESULT_PATH) as f:
            tuning_results = json.load(f)

        X_raw, y_raw = load_train_data()
        top_features = load_top_features()
        X_raw = X_raw[top_features]
        y = log_transform(y_raw)
        X = X_raw

        for name in ["bayesopt", "optuna"]:
            print(f"\n[{name}]")
            params = tuning_results[name]["params"]
            model, metrics = train_and_eval(params, X, y)
            print(f"    CV neg_rmse (log) = {tuning_results[name]['cv_score']:.4f}")
            print(f"    Val RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R²={metrics['r2']:.4f}")

            fname = "lightgbm_bayesopt_model.pkl" if name == "bayesopt" else "lightgbm_optuna_model.pkl"
            joblib.dump(model, OUTPUT_DIR / fname)

        print(f"\nSaved: lightgbm_bayesopt_model.pkl, lightgbm_optuna_model.pkl")
        print("Done.\n")
        return

    # ---- Tuning mode: search on ALL 140 features ----
    print("=" * 60)
    print("LightGBM Hyperparameter Tuning on ALL 140 Features")
    print("=" * 60)

    X_raw, y_raw = load_train_data()
    all_features = list(X_raw.columns)
    y = log_transform(y_raw)
    X = X_raw  # Use all 140 features for tuning

    print(f"\n[Data] {len(X)} samples, {len(all_features)} features (after cleaning)")
    print(f"[Target] log1p energy: mean={y.mean():.2f} std={y.std():.2f}")

    # ---- BayesOpt ----
    print(f"\n{'='*40}")
    print("BayesOpt Tuning (all 140 features)")
    print("=" * 40)
    best_params_bo, cv_score_bo = tune_bayesopt(X, y)
    print(f"    Best params: {best_params_bo}")
    print(f"    CV neg_rmse (log) = {cv_score_bo:.4f}")

    model_bo, metrics_bo = train_and_eval(best_params_bo, X, y)
    print(f"    Val RMSE = {metrics_bo['rmse']:.2f}  MAE = {metrics_bo['mae']:.2f}  R² = {metrics_bo['r2']:.4f}")

    # ---- Optuna ----
    print(f"\n{'='*40}")
    print("Optuna Tuning (all 140 features)")
    print("=" * 40)
    best_params_op, cv_score_op = tune_optuna(X, y)
    print(f"    Best params: {best_params_op}")
    print(f"    CV neg_rmse (log) = {cv_score_op:.4f}")

    model_op, metrics_op = train_and_eval(best_params_op, X, y)
    print(f"    Val RMSE = {metrics_op['rmse']:.2f}  MAE = {metrics_op['mae']:.2f}  R² = {metrics_op['r2']:.4f}")

    # Save results
    results = {
        "bayesopt": {"params": best_params_bo, "cv_score": cv_score_bo, "val_metrics": metrics_bo},
        "optuna": {"params": best_params_op, "cv_score": cv_score_op, "val_metrics": metrics_op},
    }
    json.dump(results, open(TUNING_RESULT_PATH, "w"), indent=2)
    print(f"\nSaved tuning results to {TUNING_RESULT_PATH}")
    print("Done.\n")


if __name__ == "__main__":
    main()
