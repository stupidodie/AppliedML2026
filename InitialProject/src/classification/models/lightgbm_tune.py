"""
LightGBM classification tuning on ALL 140 features:
  - Default:  BayesOpt + Optuna hyperparameter search on all 140
              → saves best params to lightgbm_tuning_results.json
  - --final:  Load best params, train final LGBM on top-15 features
              → saves final models for submission
"""

import sys, json, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import cross_val_score
import joblib

from config import (
    OUTPUT_DIR, RANDOM_STATE,
    LGBM_CV_FOLDS, LGBM_EARLY_STOP,
    LGBM_N_ESTIMATORS_CV, LGBM_N_ESTIMATORS_FINAL,
    BAYESOPT_INIT, BAYESOPT_ITER, OPTUNA_TRIALS,
)
from preprocessing import (
    load_train_data, split_data, load_top_features, compute_metrics,
)

TUNING_RESULT_PATH = OUTPUT_DIR / "lightgbm_tuning_results.json"


def cv_score(X, y, **params):
    """5-fold CV neg_log_loss (higher = better)."""
    int_params = ["num_leaves", "max_depth", "min_child_samples"]
    for p in int_params:
        if p in params:
            params[p] = int(round(params[p]))

    model = LGBMClassifier(
        objective="binary", metric="binary_logloss",
        n_estimators=LGBM_N_ESTIMATORS_CV,
        random_state=RANDOM_STATE, verbose=-1, force_col_wise=True,
        **params,
    )
    scores = cross_val_score(
        model, X, y, scoring="neg_log_loss",
        cv=LGBM_CV_FOLDS, n_jobs=-1,
    )
    return scores.mean()


# ================================================================
# BayesOpt
# ================================================================
def tune_bayesopt(X, y):
    from bayes_opt import BayesianOptimization

    pbounds = {
        "num_leaves": (15, 127),
        "max_depth": (3, 25),
        "learning_rate": (0.005, 0.3),
        "min_child_samples": (5, 100),
        "bagging_fraction": (0.5, 1.0),
        "feature_fraction": (0.5, 1.0),
        "scale_pos_weight": (1.0, 5.0),
    }

    def objective(**params):
        return cv_score(X, y, **params)

    optimizer = BayesianOptimization(f=objective, pbounds=pbounds,
                                     random_state=RANDOM_STATE, verbose=0)
    optimizer.maximize(init_points=BAYESOPT_INIT, n_iter=BAYESOPT_ITER)

    best_raw = optimizer.max["params"]
    best = {}
    for k, v in best_raw.items():
        best[k] = int(round(v)) if k in ("num_leaves", "max_depth", "min_child_samples") else v
    return best, optimizer.max["target"]


# ================================================================
# Optuna
# ================================================================
def tune_optuna(X, y):
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        }
        return cv_score(X, y, **params)

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value


# ================================================================
# Train & evaluate on validation
# ================================================================
def train_and_eval(params, X, y):
    X_train, X_val, y_train, y_val = split_data(X, y)
    model = LGBMClassifier(
        objective="binary", metric="binary_logloss",
        n_estimators=LGBM_N_ESTIMATORS_FINAL,
        random_state=RANDOM_STATE, verbose=-1, force_col_wise=True,
        **params,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], eval_metric="logloss",
        callbacks=[early_stopping(stopping_rounds=LGBM_EARLY_STOP, verbose=False)],
    )
    val_probs = model.predict_proba(X_val)[:, 1]
    return model, compute_metrics(y_val, val_probs)


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="Train final model on top-15 using saved best params")
    args = parser.parse_args()

    # ---- Final mode ----
    if args.final:
        print("=" * 60)
        print("Final LGBM Training (top-15, tuned params)")
        print("=" * 60)

        with open(TUNING_RESULT_PATH) as f:
            tuning = json.load(f)

        X, y = load_train_data()
        top_features = load_top_features()
        X = X[top_features]

        for name in ["bayesopt", "optuna"]:
            print(f"\n[{name}]")
            params = tuning[name]["params"]
            model, metrics = train_and_eval(params, X, y)
            print(f"    CV neg_log_loss = {tuning[name]['cv_score']:.6f}")
            print(f"    Val LogLoss={metrics['log_loss']:.6f}  AUC={metrics['roc_auc']:.6f}")

            fname = f"lightgbm_{name}_model.pkl"
            joblib.dump(model, OUTPUT_DIR / fname)

            # Also save individual results
            result = {
                "method": name.capitalize(),
                "best_cv_score": tuning[name]["cv_score"],
                "best_params": params,
                "validation_metrics": metrics,
                "top_features": top_features,
            }
            json.dump(result, open(OUTPUT_DIR / f"{name}_results.json", "w"), indent=2)

        print(f"\nSaved: lightgbm_bayesopt_model.pkl, lightgbm_optuna_model.pkl")
        print("Done.\n")
        return

    # ---- Tuning mode: search on ALL 140 features ----
    print("=" * 60)
    print("LightGBM Hyperparameter Tuning on ALL 140 Features")
    print("=" * 60)

    X, y = load_train_data()
    all_features = list(X.columns)
    print(f"\n[Data] {len(X)} samples, {len(all_features)} features")

    # ---- BayesOpt ----
    print(f"\n{'='*40}")
    print("BayesOpt Tuning (all 140)")
    print("=" * 40)
    bo_params, bo_cv = tune_bayesopt(X, y)
    print(f"    Best params: {json.dumps(bo_params, indent=2)}")
    print(f"    CV neg_log_loss = {bo_cv:.6f}")

    bo_model, bo_metrics = train_and_eval(bo_params, X, y)
    print(f"    Val LogLoss={bo_metrics['log_loss']:.6f}  AUC={bo_metrics['roc_auc']:.6f}")

    # ---- Optuna ----
    print(f"\n{'='*40}")
    print("Optuna Tuning (all 140)")
    print("=" * 40)
    op_params, op_cv = tune_optuna(X, y)
    print(f"    Best params: {json.dumps(op_params, indent=2)}")
    print(f"    CV neg_log_loss = {op_cv:.6f}")

    op_model, op_metrics = train_and_eval(op_params, X, y)
    print(f"    Val LogLoss={op_metrics['log_loss']:.6f}  AUC={op_metrics['roc_auc']:.6f}")

    # Save tuning results
    results = {
        "bayesopt": {"params": bo_params, "cv_score": bo_cv, "val_metrics": bo_metrics},
        "optuna": {"params": op_params, "cv_score": op_cv, "val_metrics": op_metrics},
    }
    json.dump(results, open(TUNING_RESULT_PATH, "w"), indent=2)
    print(f"\nSaved tuning results to {TUNING_RESULT_PATH}")
    print("Done.\n")


if __name__ == "__main__":
    main()
