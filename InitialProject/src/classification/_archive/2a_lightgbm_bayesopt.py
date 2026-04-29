"""
Step 2a - LightGBM with Bayesian Optimization (bayes_opt):
Hyperparameter tuning on top 15 features using BayesianOptimization.
50 iterations, neg_log_loss scoring, 5-fold CV.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from bayes_opt import BayesianOptimization
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
    """Load the top 15 features from step 1."""
    top15_path = OUTPUT_DIR / "top15_features.txt"
    with open(top15_path) as f:
        return [line.strip() for line in f if line.strip()]


def crossval_score_wrapper(data, targets, **params):
    """CV score for BayesianOptimization (maximizes neg_log_loss)."""
    int_params = ["num_leaves", "max_depth", "min_child_samples"]
    for p in int_params:
        if p in params:
            params[p] = int(params[p])

    model = LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=300,
        random_state=random_state,
        verbose=-1,
        force_col_wise=True,
        **params,
    )

    scores = cross_val_score(
        model, data, targets,
        scoring="neg_log_loss",
        cv=5,
        n_jobs=-1,
    )
    return scores.mean()


def main():
    print("=" * 60)
    print("Step 2a: LightGBM - Bayesian Optimization (bayes_opt)")
    print("=" * 60)

    # Load data and filter to top 15
    print("\n[1] Loading data and filtering to top 15 features...")
    X, y = load_train_data()
    top_features = load_top_features()
    print(f"    Top features ({len(top_features)}): {top_features}")
    X = X[top_features]

    X_train, X_val, y_train, y_val = split_data(X, y)
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}")

    # Define parameter bounds
    param_bounds = {
        "num_leaves": (15, 127),
        "max_depth": (3, 25),
        "learning_rate": (0.005, 0.3),
        "min_child_samples": (5, 100),
        "bagging_fraction": (0.5, 1.0),
        "feature_fraction": (0.5, 1.0),
        "scale_pos_weight": (1.0, 5.0),
    }

    print("\n[2] Parameter bounds:")
    for k, v in param_bounds.items():
        print(f"    {k}: {v}")

    # Create wrapper function for BayesOpt
    def objective(**params):
        """Wrapper for BayesianOptimization; it maximizes the return value."""
        return crossval_score_wrapper(X_train, y_train, **params)

    # Run Bayesian Optimization
    print("\n[3] Running Bayesian Optimization (50 iterations)...")
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=param_bounds,
        random_state=random_state,
        verbose=2,
    )
    optimizer.maximize(init_points=10, n_iter=40)

    # Best result
    best_params_raw = optimizer.max["params"]
    best_score = optimizer.max["target"]
    print(f"\n[4] Best BayesOpt score (neg_log_loss): {best_score:.6f}")
    print(f"    Best params (raw): {best_params_raw}")

    # Cast integer params
    best_params = {}
    int_params = ["num_leaves", "max_depth", "min_child_samples"]
    for k, v in best_params_raw.items():
        best_params[k] = int(v) if k in int_params else v

    print(f"    Best params (cast): {best_params}")

    # Retrain on full training set with best params
    print("\n[5] Retraining with best params...")
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

    # Evaluate on validation set
    y_pred_val = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_pred_val)
    print(f"    Validation LogLoss: {metrics['log_loss']:.6f}")
    print(f"    Validation ROC AUC: {metrics['roc_auc']:.6f}")
    print(f"    Best iteration: {model.best_iteration_}")

    # Predict on test set
    print("\n[6] Predicting on test set...")
    from utils import load_test_data, save_submission
    X_test = load_test_data()[top_features]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    save_submission(
        y_pred_test,
        "Classification_GuanranTai_LightGBM",
        variable_list=top_features,
    )

    # Save model and params
    import joblib
    model_path = OUTPUT_DIR / "lightgbm_bayesopt_model.pkl"
    joblib.dump(model, model_path)

    results = {
        "method": "bayes_opt",
        "best_cv_score": float(best_score),
        "best_params": {k: v for k, v in best_params.items()},
        "validation_metrics": metrics,
        "top_features": top_features,
        "n_iterations": 50,
    }
    results_path = OUTPUT_DIR / "bayesopt_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n    Model saved: {model_path}")
    print(f"    Results saved: {results_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()
