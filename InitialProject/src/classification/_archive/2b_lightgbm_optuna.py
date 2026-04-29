"""
Step 2b - LightGBM with Optuna:
Hyperparameter tuning on top 15 features using Optuna TPE sampler.
50 trials, 5-fold CV, neg_log_loss scoring.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier, early_stopping

from utils import (
    load_train_data,
    split_data,
    compute_metrics,
    OUTPUT_DIR,
)

random_state = 42
np.random.seed(random_state)


def load_top_features():
    top15_path = OUTPUT_DIR / "top15_features.txt"
    with open(top15_path) as f:
        return [line.strip() for line in f if line.strip()]


def objective(trial, X, y):
    """Optuna objective function."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 300,
        "random_state": random_state,
        "verbose": -1,
        "force_col_wise": True,

        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 25),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
    }

    model = LGBMClassifier(**params)

    scores = cross_val_score(
        model, X, y,
        scoring="neg_log_loss",
        cv=5,
        n_jobs=-1,
    )
    return scores.mean()


def main():
    print("=" * 60)
    print("Step 2b: LightGBM - Optuna Tuning")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data and filtering to top 15 features...")
    X, y = load_train_data()
    top_features = load_top_features()
    print(f"    Top features ({len(top_features)}): {top_features}")
    X = X[top_features]

    X_train, X_val, y_train, y_val = split_data(X, y)
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}")

    # Create Optuna study
    print("\n[2] Running Optuna optimization (50 trials)...")
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    def obj(trial):
        return objective(trial, X_train, y_train)

    study.optimize(obj, n_trials=50, show_progress_bar=True)

    # Best result
    print(f"\n[3] Best Optuna score (neg_log_loss): {study.best_value:.6f}")
    print(f"    Best params: {study.best_params}")

    # Retrain with best params
    print("\n[4] Retraining with best params...")
    best_params = study.best_params.copy()

    model = LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=2000,
        random_state=random_state,
        verbose=-1,
        device_type="cpu",
        force_col_wise=True,
        **best_params,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="logloss",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )

    y_pred_val = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_pred_val)
    print(f"    Validation LogLoss: {metrics['log_loss']:.6f}")
    print(f"    Validation ROC AUC: {metrics['roc_auc']:.6f}")
    print(f"    Best iteration: {model.best_iteration_}")

    # Predict on test set
    print("\n[5] Predicting on test set...")
    from utils import load_test_data, save_submission
    X_test = load_test_data()[top_features]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    save_submission(
        y_pred_test,
        "Classification_GuanranTai_LightGBM_Optuna",
        variable_list=top_features,
    )

    # Save
    import joblib
    model_path = OUTPUT_DIR / "lightgbm_optuna_model.pkl"
    joblib.dump(model, model_path)

    results = {
        "method": "optuna",
        "best_cv_score": float(study.best_value),
        "best_params": study.best_params,
        "validation_metrics": metrics,
        "top_features": top_features,
        "n_trials": 50,
    }
    results_path = OUTPUT_DIR / "optuna_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n    Model saved: {model_path}")
    print(f"    Results saved: {results_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()
