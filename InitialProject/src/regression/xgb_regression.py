"""
XGBoost Regression for ATLAS Electron Energy Estimation
=======================================================
- Only trains on electrons (p_Truth_isElectron == 1, p_Truth_Energy > 0)
- Log-transforms target to optimize for relative MAE
- Fast single-validation Optuna then 5-fold CV validation
- Feature selection: XGBoost gain + permutation importance (top 20)
- -999 sentinel values: kept as-is
- Output: prediction file + variable list for submission
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "output" / "regression" / "current"

TRAIN_FILE = DATA_DIR / "AppML_InitialProject_train.csv"
TEST_FILE = DATA_DIR / "AppML_InitialProject_test_regression.csv"

TARGET_COL = "p_Truth_Energy"
IS_ELECTRON_COL = "p_Truth_isElectron"

MAX_FEATURES = 20
N_CV_FOLDS = 5
OPTUNA_TRIALS = 100
EARLY_STOPPING = 80
RANDOM_SEED = 42

STUDENT_NAME = "GuanranTai"
SOLUTION_NAME = "XGBoost"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data and filter to electrons with valid energy
# ---------------------------------------------------------------------------

print("=" * 60)
print("[1/7] Loading data ...")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"  Train shape: {train_df.shape}")
print(f"  Test shape:  {test_df.shape}")

mask = (train_df[IS_ELECTRON_COL] == 1) & (train_df[TARGET_COL] > 0)
train_elec = train_df[mask].copy()
print(f"  Electrons with valid energy: {len(train_elec)} / {len(train_df)}")

y_raw = train_elec[TARGET_COL].values.astype(np.float64)
y = pd.Series(np.log(y_raw), index=train_elec.index)

print(f"  Energy range: [{y_raw.min():.1f}, {y_raw.max():.1f}] MeV")
print(f"  Log-energy range: [{y.min():.4f}, {y.max():.4f}]")

X = train_elec.drop(columns=[IS_ELECTRON_COL, TARGET_COL])
print(f"  Features: {X.shape[1]}")

test_features = [c for c in X.columns if c in test_df.columns]
X_test = test_df[test_features]
print(f"  Test features matched: {len(test_features)}")

# ---------------------------------------------------------------------------
# 2. Feature selection on dedicated holdout
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[2/7] Feature selection (max {MAX_FEATURES} features) ...")
print("=" * 60)

X_tune, X_fs, y_tune, y_fs = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED
)
print(f"  FS set: {X_fs.shape[0]} rows, Tuning set: {X_tune.shape[0]} rows")

# Quick XGBoost on FS holdout for gain importance
xgb_fs = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cpu",
    importance_type="gain",
    random_state=RANDOM_SEED,
    verbosity=0,
)
xgb_fs.fit(X_fs, y_fs)

gain_importance = pd.Series(
    xgb_fs.feature_importances_, index=X.columns
).sort_values(ascending=False)

print(f"  Top 30 by gain:")
for i, (feat, imp) in enumerate(gain_importance.head(30).items()):
    print(f"    {i+1:2d}. {feat:<40s} {imp:.4f}")

# Permutation importance on top candidates
candidate_count = min(50, len(X.columns))
top_candidates = gain_importance.head(candidate_count).index.tolist()

xgb_perm = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cpu",
    random_state=RANDOM_SEED,
    verbosity=0,
)
xgb_perm.fit(X_fs[top_candidates], y_fs)

perm_result = permutation_importance(
    xgb_perm, X_fs[top_candidates], y_fs,
    n_repeats=5, random_state=RANDOM_SEED,
    scoring="neg_mean_absolute_error", n_jobs=-1,
)
perm_importance_vals = pd.Series(
    perm_result.importances_mean, index=top_candidates
).sort_values(ascending=False)

# Combined rank
gain_ranks = gain_importance[gain_importance.index.isin(top_candidates)].rank(ascending=False)
perm_ranks = perm_importance_vals.rank(ascending=False)
combined_rank = (gain_ranks + perm_ranks) / 2

selected_features = combined_rank.sort_values().head(MAX_FEATURES).index.tolist()

print(f"\n  Selected {len(selected_features)} features:")
for i, feat in enumerate(selected_features):
    print(f"    {i+1:2d}. {feat}")

X_tune_sel = X_tune[selected_features]
X_test_sel = X_test[selected_features]

# ---------------------------------------------------------------------------
# 3. Fast Optuna with single validation split
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[3/7] Fast Optuna tuning (single validation set)...")
print("=" * 60)

X_opt, X_val, y_opt, y_val = train_test_split(
    X_tune_sel, y_tune, test_size=0.20, random_state=RANDOM_SEED
)
print(f"  Optuna train: {X_opt.shape[0]}, val: {X_val.shape[0]}")


def objective(trial):
    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "device": "cpu",
        "verbosity": 0,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "max_depth": trial.suggest_int("max_depth", 3, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 16, 300),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
    }

    model = xgb.XGBRegressor(n_estimators=3000, **params)
    model.fit(
        X_opt, y_opt,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_pred = model.predict(X_val)
    return mean_absolute_error(y_val, y_pred)


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

print(f"\n  Best trial #{study.best_trial.number}")
print(f"  Best single-val MAE (log): {study.best_value:.6f}")
print("  Best params:")
for k, v in sorted(study.best_params.items()):
    print(f"    {k}: {v}")

# ---------------------------------------------------------------------------
# 4. Cross-validate best params
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[4/7] CV validation of best params ...")
print("=" * 60)

best_params = study.best_params.copy()
cv_params = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "device": "cpu",
    "verbosity": 0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    **best_params,
}

kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tune_sel)):
    X_tr = X_tune_sel.iloc[train_idx]
    X_vl = X_tune_sel.iloc[val_idx]
    y_tr = y_tune.iloc[train_idx]
    y_vl = y_tune.iloc[val_idx]

    model = xgb.XGBRegressor(n_estimators=5000, **cv_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=False,
    )
    y_pred = model.predict(X_vl)
    loss = mean_absolute_error(y_vl, y_pred)
    cv_losses.append(loss)
    print(f"  Fold {fold+1}: MAE(log)={loss:.6f}, best_iter={model.best_iteration}")

print(f"  CV MAE (log): {np.mean(cv_losses):.6f} +/- {np.std(cv_losses):.6f}")

# ---------------------------------------------------------------------------
# 5. Train final model
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[5/7] Training final model ...")
print("=" * 60)

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X_tune_sel, y_tune, test_size=0.1, random_state=RANDOM_SEED
)

t0 = time.time()
final_model = xgb.XGBRegressor(n_estimators=5000, **cv_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_holdout, y_holdout)],
    verbose=False,
)
train_time = time.time() - t0

train_pred_log = final_model.predict(X_holdout)
holdout_mae_log = mean_absolute_error(y_holdout, train_pred_log)

y_holdout_raw = np.exp(y_holdout)
train_pred_raw = np.exp(train_pred_log)
holdout_rel_mae = np.mean(np.abs(train_pred_raw - y_holdout_raw) / y_holdout_raw)

n_estimators_used = final_model.best_iteration

print(f"  Best iteration: {n_estimators_used}")
print(f"  Holdout MAE (log):     {holdout_mae_log:.6f}")
print(f"  Holdout relative MAE:  {holdout_rel_mae:.6f}")
print(f"  Training time: {train_time:.1f}s")

# ---------------------------------------------------------------------------
# 6. Predict on test set
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[6/7] Predicting on test set ...")
print("=" * 60)

test_preds_log = final_model.predict(X_test_sel)
test_preds = np.exp(test_preds_log)
test_preds = np.clip(test_preds, 1.0, 1e7)

print(f"  Predictions: {len(test_preds)}")
print(f"  Range: [{test_preds.min():.2f}, {test_preds.max():.2f}] MeV")
print(f"  Mean:  {test_preds.mean():.2f} MeV")

# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[7/7] Saving outputs ...")
print("=" * 60)

pred_file = OUTPUT_DIR / f"Regression_{STUDENT_NAME}_{SOLUTION_NAME}.csv"
var_file = OUTPUT_DIR / f"Regression_{STUDENT_NAME}_{SOLUTION_NAME}_VariableList.csv"
summary_file = OUTPUT_DIR / f"Regression_{STUDENT_NAME}_{SOLUTION_NAME}_summary.txt"

with open(pred_file, "w") as f:
    for i, pred in enumerate(test_preds):
        f.write(f"{i},{pred:.6f}\n")
print(f"  Predictions saved: {pred_file}")

with open(var_file, "w") as f:
    for feat in selected_features:
        f.write(f"{feat},\n")
print(f"  Variable list saved: {var_file}")

with open(summary_file, "w") as f:
    f.write(f"XGBoost Regression - Guanran Tai\n")
    f.write(f"====================================\n\n")
    f.write(f"Objective: Electron energy regression (log-transformed, L1 loss)\n")
    f.write(f"Training samples (electrons only): {len(train_elec)}\n\n")
    f.write(f"Selected features ({len(selected_features)}):\n")
    for i, feat in enumerate(selected_features):
        f.write(f"  {i+1}. {feat}\n")
    f.write(f"\nBest hyperparameters:\n")
    for k, v in sorted(best_params.items()):
        f.write(f"  {k}: {v}\n")
    f.write(f"\nSingle-val MAE (log): {study.best_value:.6f}\n")
    f.write(f"CV ({N_CV_FOLDS}-fold) MAE (log): {np.mean(cv_losses):.6f} +/- {np.std(cv_losses):.6f}\n")
    f.write(f"Holdout MAE (log): {holdout_mae_log:.6f}\n")
    f.write(f"Holdout relative MAE: {holdout_rel_mae:.6f}\n")
    f.write(f"Number of trees (best iteration): {n_estimators_used}\n")
    f.write(f"Training time: {train_time:.1f}s\n")
    f.write(f"Random seed: {RANDOM_SEED}\n")
print(f"  Summary saved: {summary_file}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
