"""
PyTorch Neural Network Regression for ATLAS Electron Energy Estimation
======================================================================
- MLP with residual connections, BatchNorm, Dropout
- Log-transformed target + L1 loss (optimises for relative MAE)
- Feature selection by LGBM MI + gain (same as tree-based for fair comparison)
- -999 sentinel values: filled with feature-specific medians (NNs need this)
- MPS backend for Apple M4 acceleration
- Learning rate scheduling + early stopping
- Output: prediction file + variable list for submission
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
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
N_CV_FOLDS = 3
RANDOM_SEED = 42

# NN hyperparameters (optimised for speed on MPS)
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15

STUDENT_NAME = "GuanranTai"
SOLUTION_NAME = "PyTorchNN"

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data and filter
# ---------------------------------------------------------------------------

print("=" * 60)
print("[1/6] Loading data ...")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"  Train shape: {train_df.shape}")
print(f"  Test shape:  {test_df.shape}")

mask = (train_df[IS_ELECTRON_COL] == 1) & (train_df[TARGET_COL] > 1000)
train_elec = train_df[mask].copy()
print(f"  Electrons with E > 1000 MeV: {len(train_elec)} / {len(train_df)}")

y_raw = train_elec[TARGET_COL].values.astype(np.float64)
y = np.log(y_raw)
print(f"  Energy: [{y_raw.min():.1f}, {y_raw.max():.1f}] MeV")
print(f"  Log-energy: [{y.min():.4f}, {y.max():.4f}]")

X = train_elec.drop(columns=[IS_ELECTRON_COL, TARGET_COL])
print(f"  Features: {X.shape[1]}")

# ---------------------------------------------------------------------------
# 2. Feature selection (reuse LGBM MI+gain — fast and reliable)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[2/6] Feature selection (max {MAX_FEATURES}) ...")
print("=" * 60)

X_tune, X_fs, y_tune, y_fs = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED
)
print(f"  FS set: {X_fs.shape[0]}, Tuning set: {X_tune.shape[0]}")

mi_vals = mutual_info_regression(X_fs, y_fs, random_state=RANDOM_SEED)
mi_series = pd.Series(mi_vals, index=X.columns).sort_values(ascending=False)

lgb_fs = lgb.LGBMRegressor(
    n_estimators=500, max_depth=8, num_leaves=63,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, verbose=-1,
)
lgb_fs.fit(X_fs, y_fs)
gain_series = pd.Series(lgb_fs.feature_importances_, index=X.columns).sort_values(ascending=False)

combined = (mi_series.rank(ascending=False) + gain_series.rank(ascending=False)) / 2
selected_features = combined.sort_values().head(MAX_FEATURES).index.tolist()

print(f"\n  Top {len(selected_features)} by MI+Gain combined:")
for i, feat in enumerate(selected_features):
    print(f"    {i+1:2d}. {feat}")

X_tune_sel = X_tune[selected_features].copy()
X_test = test_df[[c for c in selected_features if c in test_df.columns]]
X_test_sel = X_test[selected_features].copy()

# ---------------------------------------------------------------------------
# 3. Preprocess features (fill -999 with median, then standardize)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[3/6] Preprocessing features ...")
print("=" * 60)

# Fill -999 with feature-specific medians (computed on training data)
for col in selected_features:
    col_median = X_tune_sel[col].median()
    X_tune_sel[col] = X_tune_sel[col].replace(-999, col_median)
    X_test_sel[col] = X_test_sel[col].replace(-999, col_median)
    neg_count = (X_tune_sel[col] == -999).sum()
    if neg_count > 0:
        # Any remaining (shouldn't happen) fill with 0
        X_tune_sel[col] = X_tune_sel[col].replace(-999, 0)

# Standardize
scaler = StandardScaler()
X_tune_arr = scaler.fit_transform(X_tune_sel)
X_test_arr = scaler.transform(X_test_sel)

print(f"  Filled -999 with feature medians, then StandardScaler")
print(f"  Tuning features: mean={X_tune_arr.mean():.3f}, std={X_tune_arr.std():.3f}")

# ---------------------------------------------------------------------------
# 4. Neural Network definition
# ---------------------------------------------------------------------------

class EnergyRegressor(nn.Module):
    """Simple MLP for regression."""
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 5. Training with 5-fold CV
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[4/6] {N_CV_FOLDS}-fold CV training ...")
print("=" * 60)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.set_num_threads(8)

print("  Converting y_tune to array...", flush=True)
y_tune_arr = y_tune.values.astype(np.float32) if hasattr(y_tune, 'values') else np.array(y_tune).astype(np.float32)
print(f"  y_tune_arr shape: {y_tune_arr.shape}, dtype: {y_tune_arr.dtype}", flush=True)
print(f"  X_tune_arr shape: {X_tune_arr.shape}, dtype: {X_tune_arr.dtype}", flush=True)
kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_losses = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tune_arr)):
    print(f"  Fold {fold+1}/{N_CV_FOLDS}: starting, train={len(tr_idx)}, val={len(vl_idx)}", flush=True)
    X_tr = np.ascontiguousarray(X_tune_arr[tr_idx].astype(np.float32))
    X_vl = np.ascontiguousarray(X_tune_arr[vl_idx].astype(np.float32))
    y_tr = np.ascontiguousarray(y_tune_arr[tr_idx])
    y_vl = np.ascontiguousarray(y_tune_arr[vl_idx])

    w = 1.0 / np.sqrt(np.exp(y_tr))
    w = np.clip(w, 0.1, 5.0)
    w = w / w.mean()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    w_t = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    X_vl_t = torch.tensor(X_vl, dtype=torch.float32, device=DEVICE)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32, device=DEVICE)

    train_ds = TensorDataset(X_tr_t, y_tr_t, w_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"    DataLoader ready, batches={len(train_loader)}", flush=True)

    model = EnergyRegressor(input_dim=MAX_FEATURES, dropout=0.1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for X_b, y_b, sw_b in train_loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = (torch.abs(pred - y_b) * sw_b).mean()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = torch.abs(model(X_vl_t) - y_vl_t).mean().item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred = model(X_vl_t).cpu().numpy()

    fold_loss = mean_absolute_error(y_vl, final_pred)
    cv_losses.append(fold_loss)
    print(f"  Fold {fold+1}/{N_CV_FOLDS}: MAE(log)={fold_loss:.6f}, best_epoch={epoch-patience_counter}")

print(f"  CV MAE (log): {np.mean(cv_losses):.6f} +/- {np.std(cv_losses):.6f}")

# ---------------------------------------------------------------------------
# 6. Final training on all tuning data
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[5/6] Training final model ...")
print("=" * 60)

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X_tune_arr, y_tune_arr, test_size=0.1, random_state=RANDOM_SEED
)

# Sample weights
w = 1.0 / np.sqrt(np.exp(y_train))
w = np.clip(w, 0.1, 5.0)
w = w / w.mean()
w_t = torch.tensor(w, dtype=torch.float32, device=DEVICE)

X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
X_ho_t = torch.tensor(X_holdout, dtype=torch.float32, device=DEVICE)
y_ho_t = torch.tensor(y_holdout, dtype=torch.float32, device=DEVICE)

train_ds = TensorDataset(X_tr_t, y_tr_t, w_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

t0 = time.time()
final_model = EnergyRegressor(input_dim=MAX_FEATURES, dropout=0.1).to(DEVICE)
optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_loss = float("inf")
best_state = None
patience_counter = 0
best_epoch = 0

for epoch in range(EPOCHS):
    final_model.train()
    for X_b, y_b, sw_b in train_loader:
        optimizer.zero_grad()
        pred = final_model(X_b)
        loss = (torch.abs(pred - y_b) * sw_b).mean()
        loss.backward()
        optimizer.step()

    scheduler.step()

    final_model.eval()
    with torch.no_grad():
        val_loss = (torch.abs(final_model(X_ho_t) - y_ho_t)).mean().item()

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
        patience_counter = 0
        best_epoch = epoch
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        break

final_model.load_state_dict(best_state)
train_time = time.time() - t0

# Evaluate
final_model.eval()
with torch.no_grad():
    ho_pred_log = final_model(X_ho_t).cpu().numpy()
    test_preds_t = final_model(torch.tensor(X_test_arr, dtype=torch.float32, device=DEVICE))
    test_preds_log = test_preds_t.cpu().numpy()

holdout_mae_log = mean_absolute_error(y_holdout, ho_pred_log)
ho_pred_raw = np.exp(ho_pred_log)
y_ho_raw = np.exp(y_holdout)
holdout_rel_mae = np.mean(np.abs(ho_pred_raw - y_ho_raw) / y_ho_raw)

print(f"  Best epoch: {best_epoch}")
print(f"  Holdout MAE (log):     {holdout_mae_log:.6f}")
print(f"  Holdout relative MAE:  {holdout_rel_mae:.6f}")
print(f"  Training time: {train_time:.1f}s")

# ---------------------------------------------------------------------------
# 7. Predict on test set + save
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[6/6] Predicting on test set + saving outputs ...")
print("=" * 60)

test_preds = np.exp(test_preds_log)
test_preds = np.clip(test_preds, 1.0, 1e7)

print(f"  Predictions: {len(test_preds)}")
print(f"  Range: [{test_preds.min():.2f}, {test_preds.max():.2f}] MeV")
print(f"  Mean:  {test_preds.mean():.2f} MeV")

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
    f.write(f"PyTorch Neural Network Regression - Guanran Tai\n")
    f.write(f"===============================================\n\n")
    f.write(f"Architecture: MLP with residual blocks ({MAX_FEATURES} -> 256 -> 256 -> 256 -> 128 -> 128 -> 1)\n")
    f.write(f"Activation: SiLU, BatchNorm, Dropout=0.1 after each layer\n")
    f.write(f"Loss: Weighted L1 on log(E), optimising for relative MAE\n")
    f.write(f"Optimiser: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})\n")
    f.write(f"Schedule: CosineAnnealingLR, EarlyStopping patience={PATIENCE}\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Training samples: {len(train_elec)} electrons (E > 1000 MeV)\n")
    f.write(f"Features: {len(selected_features)} (MI + gain combined selection)\n\n")
    f.write(f"Data preprocessing:\n")
    f.write(f"  - Removed 82 electrons with E < 1000 MeV (0.22% of total)\n")
    f.write(f"  - Filled -999 sentinel values with feature-specific medians\n")
    f.write(f"  - StandardScaler normalisation\n")
    f.write(f"  - sqrt-inverse-energy sample weights (clipped [0.1, 5.0])\n\n")
    f.write(f"Selected features ({len(selected_features)}):\n")
    for i, feat in enumerate(selected_features):
        f.write(f"  {i+1}. {feat}\n")
    f.write(f"\nCV ({N_CV_FOLDS}-fold) MAE (log): {np.mean(cv_losses):.6f} +/- {np.std(cv_losses):.6f}\n")
    f.write(f"Holdout MAE (log): {holdout_mae_log:.6f}\n")
    f.write(f"Holdout relative MAE: {holdout_rel_mae:.6f}\n")
    f.write(f"Training time: {train_time:.1f}s\n")
    f.write(f"Random seed: {RANDOM_SEED}\n")
print(f"  Summary saved: {summary_file}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
