"""
LightGBM Classification for ATLAS Electron Identification
=========================================================
- Feature selection: LightGBM gain importance + permutation importance (top 15)
- Hyperparameter tuning: Optuna with 5-fold stratified CV, binary logloss
- -999 sentinel values: kept as-is (LightGBM learns splits naturally)
- Output: prediction file + variable list for submission
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent  # InitialProject/
DATA_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "output" / "classification" / "current"

TRAIN_FILE = DATA_DIR / "AppML_InitialProject_train.csv"
TEST_FILE = DATA_DIR / "AppML_InitialProject_test_classification.csv"

TARGET_COL = "p_Truth_isElectron"
ENERGY_COL = "p_Truth_Energy"

MAX_FEATURES = 15
N_CV_FOLDS = 3
OPTUNA_TRIALS = 25
EARLY_STOPPING = 50
RANDOM_SEED = 42

STUDENT_NAME = "GuanranTai"
SOLUTION_NAME = "LightGBM"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

print("=" * 60)
print("[1/6] Loading data ...")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"  Train shape: {train_df.shape}")
print(f"  Test shape:  {test_df.shape}")

# Separate features and target
X = train_df.drop(columns=[TARGET_COL, ENERGY_COL])
y = train_df[TARGET_COL]

print(f"  Features: {X.shape[1]}")
print(f"  Target distribution: 0={int((y == 0).sum())} ({y.mean():.1%} electron)")

# Align test columns to training feature set
test_features = [c for c in X.columns if c in test_df.columns]
X_test = test_df[test_features]
print(f"  Test features matched: {len(test_features)}")

# Split off an independent holdout for feature selection (never seen during tuning)
X_tune, X_fs, y_tune, y_fs = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
)
print(f"  Feature selection set: {X_fs.shape[0]} rows")
print(f"  Tuning/CV set:        {X_tune.shape[0]} rows")

# ---------------------------------------------------------------------------
# 2. Feature selection (LightGBM gain + permutation importance)
#    Performed exclusively on the isolated FS holdout to avoid overfitting.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[2/6] Feature selection (max {} features) on holdout set ...".format(MAX_FEATURES))
print("=" * 60)

# Phase A: LightGBM on FS holdout for gain importance
lgb_quick = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    importance_type="gain",
    is_unbalance=True,
    random_state=RANDOM_SEED,
    verbose=-1,
)
lgb_quick.fit(X_fs, y_fs)

gain_importance = pd.Series(
    lgb_quick.feature_importances_, index=X.columns
).sort_values(ascending=False)

print(f"  Top 30 features (Gain on holdout):")
for i, (feat, imp) in enumerate(gain_importance.head(30).items()):
    print(f"    {i+1:2d}. {feat:<40s} {imp:.4f}")

# Phase B: Permutation importance on top candidates (also on FS holdout)
candidate_count = min(50, len(X.columns))
top_candidates = gain_importance.head(candidate_count).index.tolist()
X_fs_candidates = X_fs[top_candidates]

lgb_candidates = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    importance_type="gain",
    is_unbalance=True,
    random_state=RANDOM_SEED,
    verbose=-1,
)
lgb_candidates.fit(X_fs_candidates, y_fs)

perm_result = permutation_importance(
    lgb_candidates,
    X_fs_candidates,
    y_fs,
    n_repeats=5,
    random_state=RANDOM_SEED,
    scoring="neg_log_loss",
    n_jobs=-1,
)

perm_importance_vals = pd.Series(
    perm_result.importances_mean, index=top_candidates
).sort_values(ascending=False)

# Combine rankings: average rank
gain_ranks = gain_importance[gain_importance.index.isin(top_candidates)].rank(ascending=False)
perm_ranks = perm_importance_vals.rank(ascending=False)
combined_rank = (gain_ranks + perm_ranks) / 2

selected_features = combined_rank.sort_values().head(MAX_FEATURES).index.tolist()

print(f"\n  Selected {len(selected_features)} features (combined rank on holdout):")
for i, feat in enumerate(selected_features):
    print(f"    {i+1:2d}. {feat}")

# Subset to selected features — tuning data and test data
X_tune_selected = X_tune[selected_features]
X_test_selected = X_test[selected_features]

# ---------------------------------------------------------------------------
# 3. Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[3/6] Optuna hyperparameter tuning ...")
print("=" * 60)


def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "is_unbalance": True,
        "num_leaves": trial.suggest_int("num_leaves", 16, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    losses = []

    for train_idx, val_idx in skf.split(X_tune_selected, y_tune):
        X_tr, X_val = X_tune_selected.iloc[train_idx], X_tune_selected.iloc[val_idx]
        y_tr, y_val = y_tune.iloc[train_idx], y_tune.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=1000,
            **params,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )

        y_pred = model.predict_proba(X_val)[:, 1]
        losses.append(log_loss(y_val, y_pred))

    return np.mean(losses)


# Optuna study with SQLite persistence (survives interruptions)
study_db = f"sqlite:///{OUTPUT_DIR}/optuna_study.db"
study = optuna.create_study(
    direction="minimize",
    storage=study_db,
    study_name="lgbm_classification",
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)

def save_best_callback(study, trial):
    """After each trial, save best-so-far predictions to disk."""
    if study.best_trial.number != trial.number:
        return  # not a new best
    best_params = study.best_params
    model = lgb.LGBMClassifier(
        n_estimators=500,
        objective="binary",
        is_unbalance=True,
        random_state=RANDOM_SEED,
        verbose=-1,
        **best_params,
    )
    model.fit(X_tune_selected, y_tune)
    preds = model.predict_proba(X_test_selected)[:, 1]
    preds = np.clip(preds, 1e-7, 1 - 1e-7)
    pred_file = OUTPUT_DIR / f"Classification_{STUDENT_NAME}_{SOLUTION_NAME}.csv"
    with open(pred_file, "w") as f:
        for i, p in enumerate(preds):
            f.write(f"{i},{p:.6f}\n")

study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True,
               callbacks=[save_best_callback])

print(f"\n  Best trial #{study.best_trial.number}")
print(f"  Best CV LogLoss: {study.best_value:.6f}")
print("  Best params:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")

# ---------------------------------------------------------------------------
# 4. Final model training (with validation holdout for early stopping)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[4/6] Training final model ...")
print("=" * 60)

# Hold out 10% of tuning data for early stopping validation
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X_tune_selected, y_tune, test_size=0.1, stratify=y_tune, random_state=RANDOM_SEED
)

best_params = study.best_params.copy()
final_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "is_unbalance": True,
    **best_params,
}

t0 = time.time()
final_model = lgb.LGBMClassifier(n_estimators=5000, **final_params)
final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_holdout, y_holdout)],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
)
train_time = time.time() - t0

train_pred = final_model.predict_proba(X_holdout)[:, 1]
holdout_loss = log_loss(y_holdout, train_pred)
n_estimators_used = final_model.best_iteration_

print(f"  Best iteration: {n_estimators_used}")
print(f"  Holdout LogLoss: {holdout_loss:.6f}")
print(f"  Training time: {train_time:.1f}s")

# ---------------------------------------------------------------------------
# 5. Predict on test set
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[5/6] Predicting on test set ...")
print("=" * 60)

test_preds = final_model.predict_proba(X_test_selected)[:, 1]

# Clamp away from exact 0/1 to stay in ]0, 1[
test_preds = np.clip(test_preds, 1e-7, 1 - 1e-7)

print(f"  Predictions: {len(test_preds)}")
print(f"  Range: [{test_preds.min():.6f}, {test_preds.max():.6f}]")
print(f"  Mean: {test_preds.mean():.6f}")

# ---------------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[6/6] Saving outputs ...")
print("=" * 60)

pred_file = OUTPUT_DIR / f"Classification_{STUDENT_NAME}_{SOLUTION_NAME}.csv"
var_file = OUTPUT_DIR / f"Classification_{STUDENT_NAME}_{SOLUTION_NAME}_VariableList.csv"
summary_file = OUTPUT_DIR / f"Classification_{STUDENT_NAME}_{SOLUTION_NAME}_summary.txt"

# Predictions: index, value (no header, 60000 lines)
with open(pred_file, "w") as f:
    for i, pred in enumerate(test_preds):
        f.write(f"{i},{pred:.6f}\n")
print(f"  Predictions saved: {pred_file}")

# Variable list: one variable per line with trailing comma
with open(var_file, "w") as f:
    for feat in selected_features:
        f.write(f"{feat},\n")
print(f"  Variable list saved: {var_file}")

# Summary
with open(summary_file, "w") as f:
    f.write(f"LightGBM Classification - Guanran Tai\n")
    f.write(f"====================================\n\n")
    f.write(f"Objective: Electron vs non-electron binary classification\n")
    f.write(f"Metric: Binary Cross Entropy (LogLoss)\n\n")
    f.write(f"Selected features ({len(selected_features)}):\n")
    for i, feat in enumerate(selected_features):
        f.write(f"  {i+1}. {feat}\n")
    f.write(f"\nBest hyperparameters:\n")
    for k, v in sorted(best_params.items()):
        f.write(f"  {k}: {v}\n")
    f.write(f"\nBest CV ({N_CV_FOLDS}-fold) LogLoss: {study.best_value:.6f}\n")
    f.write(f"Holdout LogLoss: {holdout_loss:.6f}\n")
    f.write(f"Number of trees (best iteration): {n_estimators_used}\n")
    f.write(f"Training time: {train_time:.1f}s\n")
    f.write(f"Random seed: {RANDOM_SEED}\n")
print(f"  Summary saved: {summary_file}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
