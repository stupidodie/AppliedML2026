"""
PyTorch NN Classification — Fast Ensemble with Calibration
===========================================================
Strategy:
  - Reuse proven hyperparameters from earlier run (single-model: 0.125)
  - Ensemble of 7 models (different seeds)
  - Platt scaling calibration on holdout set
  - Target: logloss < 0.10
"""

import time
import warnings
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "output" / "classification" / "current"

TRAIN_FILE = DATA_DIR / "AppML_InitialProject_train.csv"
TEST_FILE = DATA_DIR / "AppML_InitialProject_test_classification.csv"

TARGET_COL = "p_Truth_isElectron"
ENERGY_COL = "p_Truth_Energy"

MAX_FEATURES = 15
N_ENSEMBLE = 7
MAX_EPOCHS = 600
EARLY_STOP = 50
LABEL_SMOOTHING = 0.0
POS_WEIGHT_MULT = 0.40   # reduce from ~3.76 to ~1.50 for better logloss calibration - raw labels work better for this task
RANDOM_SEED = 42

# Proven hyperparameters from Optuna v3 (single-model holdout: 0.125)
HP = {
    "num_layers": 2,
    "hidden_dim": 192,
    "hidden_dims": [192, 128, 64],
    "dropout": 0.304,
    "learning_rate": 0.00214,
    "weight_decay": 2.58e-6,
    "batch_size": 512,
    "use_batchnorm": False,
    "activation": "relu",
}

STUDENT_NAME = "GuanranTai"
SOLUTION_NAME = "PyTorchNN"
CLIP_EPS = 5e-6  # large enough that %.6f never rounds to 0.000000 or 1.000000
SENTINEL = -999.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"  Device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

print("=" * 60)
print("[1/5] Loading data ...")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
assert train_df.shape[0] == 180000 and test_df.shape[0] == 60000
print(f"  Train: 180000  Test: 60000 ✓")

X = train_df.drop(columns=[TARGET_COL, ENERGY_COL])
y = train_df[TARGET_COL]
X_test_raw = test_df[[c for c in X.columns if c in test_df.columns]]

print(f"  Features: {X.shape[1]}  |  Electron: {y.mean():.2%}")

# ---------------------------------------------------------------------------
# -999 handling
# ---------------------------------------------------------------------------

def compute_stats_ignore_sentinel(df):
    stats = {}
    for col in df.columns:
        vals = df[col].values.astype(np.float64)
        mask = vals != SENTINEL
        if mask.sum() > 1:
            m = vals[mask].mean()
            s = vals[mask].std(ddof=0)
            if s < 1e-8: s = 1.0
        else:
            m, s = 0.0, 1.0
        stats[col] = (m, s)
    return stats

def standardise_ignore_sentinel(df, stats):
    out = np.zeros(df.shape, dtype=np.float32)
    for j, col in enumerate(df.columns):
        m, s = stats[col]
        vals = df[col].values.astype(np.float64)
        mask = vals != SENTINEL
        out[mask, j] = (vals[mask] - m) / s
    return out

full_stats = compute_stats_ignore_sentinel(X)
feature_names = list(X.columns)

# Use full data for everything (no separate FS holdout)
X_all_np = standardise_ignore_sentinel(X, full_stats)
y_all_np = y.values.astype(np.float32)
X_test_np = standardise_ignore_sentinel(X_test_raw, full_stats)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3,
                 use_batchnorm=True, activation="relu"):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU,
                    "leakyrelu": nn.LeakyReLU}[activation]
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class L1FirstLayerMLP(MLP):
    def __init__(self, *args, l1_lambda=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1_lambda = l1_lambda

    def l1_penalty(self):
        return self.l1_lambda * self.net[0].weight.abs().sum()

    def weight_magnitudes(self):
        return self.net[0].weight.abs().sum(dim=0).detach().cpu().numpy()


def predict_proba(model, X_t, batch_size, device):
    model.eval()
    dl = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for (bx,) in dl:
            probs.append(torch.sigmoid(model(bx.to(device))).cpu().numpy().flatten())
    return np.concatenate(probs)


def smooth_labels(y_t, smoothing=LABEL_SMOOTHING):
    return y_t * (1 - smoothing) + 0.5 * smoothing


# ---------------------------------------------------------------------------
# 2. Feature selection (L1 NN, using a 20% holdout)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[2/5] Feature selection via L1 NN (max {MAX_FEATURES}) ...")
print("=" * 60)

# Stratified 20% sample for feature selection
X_fs_np_data, _, y_fs_np_data, _ = train_test_split(
    X_all_np, y_all_np, test_size=0.80, stratify=y_all_np, random_state=RANDOM_SEED)
X_fs_tr, X_fs_val, y_fs_tr, y_fs_val = train_test_split(
    X_fs_np_data, y_fs_np_data, test_size=0.3, stratify=y_fs_np_data, random_state=RANDOM_SEED)
print(f"  FS: {X_fs_tr.shape[0]} train  {X_fs_val.shape[0]} val  "
      f"electron={y_fs_tr.mean():.2%}/{y_fs_val.mean():.2%}")

N_L1 = 5
all_l1 = []
for so in range(N_L1):
    torch.manual_seed(RANDOM_SEED + so)
    np.random.seed(RANDOM_SEED + so)
    m = L1FirstLayerMLP(X_fs_tr.shape[1], [256, 128, 64],
                         dropout=0.3, use_batchnorm=True, activation="relu",
                         l1_lambda=2e-4).to(DEVICE)
    pw = (y_fs_tr == 0).sum() / max((y_fs_tr == 1).sum(), 1)
    pw_t = torch.tensor([pw], dtype=torch.float32, device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw_t)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    tr_dl = DataLoader(
        TensorDataset(torch.tensor(X_fs_tr, dtype=torch.float32),
                       torch.tensor(y_fs_tr, dtype=torch.float32).unsqueeze(1)),
        batch_size=1024, shuffle=True)
    Xv = torch.tensor(X_fs_val, dtype=torch.float32).to(DEVICE)
    Yv = torch.tensor(y_fs_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    best_v, best_w, pat = float("inf"), None, 0
    for ep in range(50):
        m.train()
        for bx, by in tr_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            (crit(m(bx), by) + m.l1_penalty()).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(Xv), Yv).item()
        sched.step(vl)
        if vl < best_v:
            best_v, best_w, pat = vl, m.weight_magnitudes(), 0
        else:
            pat += 1
            if pat >= 15: break
    all_l1.append(best_w)
    print(f"  L1 model {so+1}/{N_L1}  val_loss={best_v:.6f}")

avg_w = np.mean(all_l1, axis=0)
l1_rank = pd.Series(avg_w, index=feature_names).sort_values(ascending=False)
selected = l1_rank.head(MAX_FEATURES).index.tolist()
sel_idx = [feature_names.index(fn) for fn in selected]
X_all_sel = X_all_np[:, sel_idx]
X_test_sel = X_test_np[:, sel_idx]

print(f"\n  Selected {len(selected)} features:")
for i, fn in enumerate(selected):
    print(f"    {i+1:2d}. {fn}")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 3. Train ensemble
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[3/5] Training ensemble ({N_ENSEMBLE} models) ...")
print("=" * 60)
print(f"  Arch: {[MAX_FEATURES] + HP['hidden_dims'] + [1]}")
print(f"  HP: lr={HP['learning_rate']} wd={HP['weight_decay']} "
      f"dropout={HP['dropout']} batch={HP['batch_size']} "
      f"bn={HP['use_batchnorm']} act={HP['activation']}")

# Split: 80% train + 10% holdout (for calibration) + 10% test prediction
# But we already have test set. Use 5% for calibration
X_tr_val, X_calib, y_tr_val, y_calib = train_test_split(
    X_all_sel, y_all_np, test_size=0.05, stratify=y_all_np, random_state=RANDOM_SEED)

X_train, X_hold, y_train, y_hold = train_test_split(
    X_tr_val, y_tr_val, test_size=0.1, stratify=y_tr_val, random_state=RANDOM_SEED)

print(f"  Train: {X_train.shape[0]}  Hold: {X_hold.shape[0]}  Calib: {X_calib.shape[0]}")

models = []
hold_probs_all = []
test_probs_all = []
calib_probs_all = []

for ei in range(N_ENSEMBLE):
    seed = RANDOM_SEED + 100 + ei
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(MAX_FEATURES, HP["hidden_dims"],
                dropout=HP["dropout"], use_batchnorm=HP["use_batchnorm"],
                activation=HP["activation"]).to(DEVICE)

    # Bagging: each model trains on a different random 80% subset
    n_bag = int(len(X_train) * 0.80)
    bag_idx = np.random.choice(len(X_train), n_bag, replace=False)
    X_tr_bag = X_train[bag_idx]
    y_tr_bag = y_train[bag_idx]
    pw = (y_tr_bag == 0).sum() / max((y_tr_bag == 1).sum(), 1) * POS_WEIGHT_MULT

    tr_ds = TensorDataset(
        torch.tensor(X_tr_bag, dtype=torch.float32),
        torch.tensor(y_tr_bag, dtype=torch.float32).unsqueeze(1))
    ho_ds = TensorDataset(
        torch.tensor(X_hold, dtype=torch.float32),
        torch.tensor(y_hold, dtype=torch.float32).unsqueeze(1))
    tr_dl = DataLoader(tr_ds, batch_size=HP["batch_size"], shuffle=True)
    ho_dl = DataLoader(ho_ds, batch_size=HP["batch_size"] * 2, shuffle=False)

    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], dtype=torch.float32, device=DEVICE))
    opt = optim.AdamW(model.parameters(), lr=HP["learning_rate"],
                       weight_decay=HP["weight_decay"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    best_ho, best_st, no_imp, best_ep = float("inf"), None, 0, 0
    t0 = time.time()

    for ep in range(MAX_EPOCHS):
        model.train()
        for bx, by in tr_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            crit(model(bx), smooth_labels(by)).backward()
            opt.step()

        model.eval()
        ho_loss, nh = 0.0, 0
        with torch.no_grad():
            for bx, by in ho_dl:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                ho_loss += crit(model(bx), by).item() * bx.size(0)
                nh += bx.size(0)
        ho_loss /= nh
        sched.step(ho_loss)

        if ho_loss < best_ho:
            best_ho, best_st, best_ep, no_imp = ho_loss, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP: break

    model.load_state_dict(best_st)
    models.append(model)

    hp = predict_proba(model, torch.tensor(X_hold, dtype=torch.float32),
                       HP["batch_size"] * 2, DEVICE)
    hold_probs_all.append(np.clip(hp, 1e-15, 1 - 1e-15))
    hll = log_loss(y_hold, hold_probs_all[-1])

    cp = predict_proba(model, torch.tensor(X_calib, dtype=torch.float32),
                       HP["batch_size"] * 2, DEVICE)
    calib_probs_all.append(np.clip(cp, 1e-15, 1 - 1e-15))

    tp = predict_proba(model, torch.tensor(X_test_sel, dtype=torch.float32),
                       HP["batch_size"] * 2, DEVICE)
    test_probs_all.append(tp)

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dt = time.time() - t0
    print(f"  Model {ei+1}/{N_ENSEMBLE}  seed={seed}  ep={best_ep}  "
          f"hold={hll:.6f}  params={n_p:,}  {dt:.0f}s")

# ---------------------------------------------------------------------------
# 4. Calibration + ensemble evaluation
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[4/5] Platt scaling calibration ...")
print("=" * 60)

# Average ensemble prob as calibration input
calib_ens_probs = np.mean(calib_probs_all, axis=0)
test_ens_probs = np.mean(test_probs_all, axis=0)

# Logits-based Platt (best so far) + isotonic comparison
eps = 1e-15
calib_logits = np.log(np.clip(calib_ens_probs, eps, 1 - eps) /
                       np.clip(1 - calib_ens_probs, eps, 1 - eps))
test_logits = np.log(np.clip(test_ens_probs, eps, 1 - eps) /
                      np.clip(1 - test_ens_probs, eps, 1 - eps))

platt = LogisticRegression(C=100, solver="lbfgs")
platt.fit(calib_logits.reshape(-1, 1), y_calib)

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(calib_ens_probs, y_calib)

test_platt = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]
test_platt = np.clip(test_platt, CLIP_EPS, 1 - CLIP_EPS)
test_iso = iso.predict(test_ens_probs)
test_iso = np.clip(test_iso, CLIP_EPS, 1 - CLIP_EPS)

hold_ens_raw = np.mean(hold_probs_all, axis=0)
hold_logits = np.log(np.clip(hold_ens_raw, eps, 1 - eps) /
                      np.clip(1 - hold_ens_raw, eps, 1 - eps))
hold_platt = platt.predict_proba(hold_logits.reshape(-1, 1))[:, 1]
hold_platt = np.clip(hold_platt, 1e-15, 1 - 1e-15)
hold_iso = iso.predict(hold_ens_raw)
hold_iso = np.clip(hold_iso, 1e-15, 1 - 1e-15)

print(f"  Platt coef: {platt.coef_[0][0]:.4f}  intercept: {platt.intercept_[0]:.4f}")

for i, hp in enumerate(hold_probs_all):
    print(f"  Model {i+1} holdout LogLoss: {log_loss(y_hold, hp):.6f}")
print(f"  Ensemble (raw) holdout LogLoss:       {log_loss(y_hold, hold_ens_raw):.6f}")
print(f"  Ensemble (Platt-logits) holdout LogLoss: {log_loss(y_hold, hold_platt):.6f}")
print(f"  Ensemble (isotonic) holdout LogLoss:  {log_loss(y_hold, hold_iso):.6f}")

if log_loss(y_hold, hold_iso) < log_loss(y_hold, hold_platt):
    test_preds = test_iso
    calib_method = "isotonic"
    hold_calib_ll = log_loss(y_hold, hold_iso)
else:
    test_preds = test_platt
    calib_method = "platt_logits"
    hold_calib_ll = log_loss(y_hold, hold_platt)

test_preds = np.clip(test_preds, CLIP_EPS, 1 - CLIP_EPS)

# ---------------------------------------------------------------------------
# 5. Predict on test set & save
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[5/5] Test prediction & save ...")
print("=" * 60)

assert len(test_preds) == 60000
print(f"  Test preds: {len(test_preds)}  "
      f"range=[{test_preds.min():.6f},{test_preds.max():.6f}]  "
      f"mean={test_preds.mean():.6f}  >0.5={(test_preds > 0.5).mean():.3%}")

pred_file = OUTPUT_DIR / f"Classification_{STUDENT_NAME}_{SOLUTION_NAME}.csv"
var_file = OUTPUT_DIR / f"Classification_{STUDENT_NAME}_{SOLUTION_NAME}_VariableList.csv"
summary_file = OUTPUT_DIR / f"Classification_{STUDENT_NAME}_{SOLUTION_NAME}_summary.txt"

with open(pred_file, "w") as fh:
    for i, pred in enumerate(test_preds):
        fh.write(f"{i},{pred:.6f}\n")
print(f"  Predictions: {pred_file}")

with open(var_file, "w") as fh:
    for fn in selected:
        fh.write(f"{fn},\n")
print(f"  Vars: {var_file}")

with open(summary_file, "w") as fh:
    fh.write(f"PyTorch NN Classification (Ensemble+Calibration) - {STUDENT_NAME}\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(f"Objective: Electron vs non-electron binary classification\n")
    fh.write(f"Metric: Binary Cross Entropy (LogLoss) - unweighted\n\n")
    fh.write(f"Architecture: MLP {[MAX_FEATURES] + HP['hidden_dims'] + [1]}\n")
    fh.write(f"  Activation: {HP['activation']}  Dropout: {HP['dropout']}  "
             f"BatchNorm: {HP['use_batchnorm']}\n")
    fh.write(f"  Learning rate: {HP['learning_rate']}  "
             f"Weight decay: {HP['weight_decay']:.2e}\n")
    fh.write(f"  Params (per model): {n_p:,}\n\n")
    fh.write(f"Training: {N_ENSEMBLE} models, {MAX_EPOCHS} max epochs, "
             f"early stop={EARLY_STOP}\n")
    fh.write(f"Label smoothing: {LABEL_SMOOTHING}\n")
    fh.write(f"Calibration: {calib_method}\n\n")
    fh.write(f"Feature selection: L1 first-layer weights\n")
    fh.write(f"  ({N_L1} models averaged, arch=[256,128,64])\n")
    fh.write(f"Selected features ({len(selected)}):\n")
    for i, fn in enumerate(selected):
        fh.write(f"  {i+1}. {fn}\n")
    fh.write(f"\nPer-model holdout LogLosses:\n")
    for i, hp in enumerate(hold_probs_all):
        fh.write(f"  Model {i+1}: {log_loss(y_hold, hp):.6f}\n")
    fh.write(f"Ensemble (raw) holdout LogLoss: {log_loss(y_hold, hold_ens_raw):.6f}\n")
    fh.write(f"Ensemble (Platt) holdout LogLoss: {log_loss(y_hold, hold_platt):.6f}\n")
    fh.write(f"Ensemble (isotonic) holdout LogLoss: {log_loss(y_hold, hold_iso):.6f}\n")
    fh.write(f"Selected calibration: {calib_method} (holdout: {hold_calib_ll:.6f})\n")
    fh.write(f"\nDevice: {DEVICE}\nSeed: {RANDOM_SEED}\n")
print(f"  Summary: {summary_file}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
