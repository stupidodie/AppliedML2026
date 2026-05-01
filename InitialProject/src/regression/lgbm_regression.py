"""
LightGBM Regression for ATLAS Electron Energy Estimation
========================================================
- Trains on electrons with E > 1000 MeV (removes extreme low-E outliers)
- Log-transformed target + L1 loss (robust to outliers)
- MI + gain combined feature selection (top 20)
- 5-model ensemble for better generalization
- sqrt-inverse-energy sample weights to handle wide energy range
- -999 sentinel values: kept as-is
- Output: prediction file + variable list for submission
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import mutual_info_regression

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
N_ENSEMBLE = 5
N_CV_FOLDS = 5
RANDOM_SEED = 42

STUDENT_NAME = "GuanranTai"
SOLUTION_NAME = "LightGBM"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data and filter
# ---------------------------------------------------------------------------

print("=" * 60)
print("[1/5] Loading data ...")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"  Train shape: {train_df.shape}")
print(f"  Test shape:  {test_df.shape}")

# Only electrons with valid energy (remove extreme low-E outliers < 1000 MeV)
mask = (train_df[IS_ELECTRON_COL] == 1) & (train_df[TARGET_COL] > 1000)
train_elec = train_df[mask].copy()
print(f"  Electrons with E > 1000 MeV: {len(train_elec)} / {len(train_df)}")

y_raw = train_elec[TARGET_COL].values.astype(np.float64)
y = pd.Series(np.log(y_raw), index=train_elec.index)

print(f"  Energy: [{y_raw.min():.1f}, {y_raw.max():.1f}] MeV")
print(f"  Log-energy: [{y.min():.4f}, {y.max():.4f}]")

X = train_elec.drop(columns=[IS_ELECTRON_COL, TARGET_COL])
print(f"  Features: {X.shape[1]}")

test_features = [c for c in X.columns if c in test_df.columns]
X_test = test_df[test_features]
print(f"  Test features matched: {len(test_features)}")

# ---------------------------------------------------------------------------
# 2. Feature selection: MI + gain combined rank on holdout
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[2/5] Feature selection (max {MAX_FEATURES}) ...")
print("=" * 60)

X_tune, X_fs, y_tune, y_fs = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED
)
print(f"  FS set: {X_fs.shape[0]}, Tuning set: {X_tune.shape[0]}")

# Mutual information on FS holdout
mi_vals = mutual_info_regression(X_fs, y_fs, random_state=RANDOM_SEED)
mi_series = pd.Series(mi_vals, index=X.columns).sort_values(ascending=False)

# Gain importance on FS holdout
lgb_fs = lgb.LGBMRegressor(
    n_estimators=500, max_depth=8, num_leaves=63,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, verbose=-1,
)
lgb_fs.fit(X_fs, y_fs)
gain_series = pd.Series(lgb_fs.feature_importances_, index=X.columns).sort_values(ascending=False)

# Combined rank
combined = (mi_series.rank(ascending=False) + gain_series.rank(ascending=False)) / 2
selected_features = combined.sort_values().head(MAX_FEATURES).index.tolist()

print(f"\n  Top {len(selected_features)} by MI+Gain combined:")
for i, feat in enumerate(selected_features):
    print(f"    {i+1:2d}. {feat:<40s} MI={mi_series[feat]:.4f} Gain={gain_series[feat]:.2f}")

X_tune_sel = X_tune[selected_features]
X_test_sel = X_test[selected_features]

# ---------------------------------------------------------------------------
# 3. Train ensemble + 5-fold CV validation
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[3/5] Training {N_ENSEMBLE}-model ensemble + CV ...")
print("=" * 60)

# Build sample weights (sqrt-inverse energy, clipped)
def make_weights(y_vals):
    w = 1.0 / np.sqrt(np.exp(y_vals))
    w = np.clip(w, 0.1, 5.0)
    return w / w.mean()

# Base parameters (tuned for this data)
base_params = {
    "objective": "regression_l1",
    "metric": "l1",
    "boosting_type": "gbdt",
    "num_leaves": 300,
    "max_depth": 14,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.005,
    "reg_lambda": 0.1,
    "min_child_samples": 5,
    "min_child_weight": 0.0001,
    "verbose": -1,
    "n_jobs": -1,
}

# 5-fold CV to estimate performance
kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_losses = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tune_sel)):
    X_tr = X_tune_sel.iloc[tr_idx]
    X_vl = X_tune_sel.iloc[vl_idx]
    y_tr = y_tune.iloc[tr_idx]
    y_vl = y_tune.iloc[vl_idx]

    params = {**base_params, "random_state": RANDOM_SEED + fold}
    model = lgb.LGBMRegressor(n_estimators=5000, **params)
    model.fit(
        X_tr, y_tr, sample_weight=make_weights(y_tr),
        eval_set=[(X_vl, y_vl)], eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    y_pred = model.predict(X_vl)
    loss = mean_absolute_error(y_vl, y_pred)
    cv_losses.append(loss)
    print(f"  Fold {fold+1}: MAE(log)={loss:.6f}, best_iter={model.best_iteration_}")

print(f"  CV MAE (log): {np.mean(cv_losses):.6f} +/- {np.std(cv_losses):.6f}")

# ---------------------------------------------------------------------------
# 4. Final ensemble training on all tuning data
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[4/5] Training final {N_ENSEMBLE}-model ensemble ...")
print("=" * 60)

X_tr, X_ho, y_tr, y_ho = train_test_split(
    X_tune_sel, y_tune, test_size=0.1, random_state=RANDOM_SEED
)

t0 = time.time()
final_models = []

for i in range(N_ENSEMBLE):
    seed = RANDOM_SEED + i * 123
    params = {**base_params, "random_state": seed}
    model = lgb.LGBMRegressor(n_estimators=8000, **params)
    model.fit(
        X_tr, y_tr, sample_weight=make_weights(y_tr),
        eval_set=[(X_ho, y_ho)], eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    final_models.append(model)
    print(f"  Model {i+1}/{N_ENSEMBLE}: best_iter={model.best_iteration_}")

train_time = time.time() - t0

# Evaluate ensemble on holdout
ens_preds_log = np.zeros(len(y_ho))
for model in final_models:
    ens_preds_log += model.predict(X_ho) / N_ENSEMBLE

holdout_mae_log = mean_absolute_error(y_ho, ens_preds_log)
y_ho_raw = np.exp(y_ho)
ens_preds_raw = np.exp(ens_preds_log)
holdout_rel_mae = np.mean(np.abs(ens_preds_raw - y_ho_raw) / y_ho_raw)

print(f"\n  Ensemble holdout MAE (log):    {holdout_mae_log:.6f}")
print(f"  Ensemble holdout relative MAE: {holdout_rel_mae:.6f}")
print(f"  Training time: {train_time:.1f}s")

# ---------------------------------------------------------------------------
# 5. Predict on test set and save
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[5/5] Predicting on test set + saving outputs ...")
print("=" * 60)

# Ensemble prediction on test
test_preds_log = np.zeros(len(X_test_sel))
for model in final_models:
    test_preds_log += model.predict(X_test_sel) / N_ENSEMBLE
test_preds = np.exp(test_preds_log)
test_preds = np.clip(test_preds, 1.0, 1e7)

print(f"  Predictions: {len(test_preds)}")
print(f"  Range: [{test_preds.min():.2f}, {test_preds.max():.2f}] MeV")
print(f"  Mean:  {test_preds.mean():.2f} MeV")

# Save
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
    f.write(f"LightGBM Regression (Ensemble) - Guanran Tai\n")
    f.write(f"=============================================\n\n")
    f.write(f"Objective: Electron energy regression (log-transformed, L1 loss)\n")
    f.write(f"Ensemble: {N_ENSEMBLE} models with different random seeds\n")
    f.write(f"Training samples: {len(train_elec)} electrons (E > 1000 MeV)\n")
    f.write(f"Features: {len(selected_features)} (MI + gain combined selection)\n\n")
    f.write(f"Selected features:\n")
    for i, feat in enumerate(selected_features):
        f.write(f"  {i+1}. {feat}\n")
    f.write(f"\nModel hyperparameters:\n")
    for k, v in sorted(base_params.items()):
        if k not in ("verbose", "n_jobs", "random_state"):
            f.write(f"  {k}: {v}\n")
    f.write(f"\nCV ({N_CV_FOLDS}-fold) MAE (log): {np.mean(cv_losses):.6f} +/- {np.std(cv_losses):.6f}\n")
    f.write(f"Holdout MAE (log): {holdout_mae_log:.6f}\n")
    f.write(f"Holdout relative MAE: {holdout_rel_mae:.6f}\n")
    f.write(f"Training time: {train_time:.1f}s\n")
    f.write(f"Random seed: {RANDOM_SEED}\n")
print(f"  Summary saved: {summary_file}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
