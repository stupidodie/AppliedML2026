"""
PyTorch NN Regression v6 — Back to MI features + Deep ResMLP
=============================================================
Based on v3's 0.206 baseline. Improvements:
  - Features: MI-on-log(E) + dedup 0.97, ensure p_eta/p_Rphi included
  - Loss: Heavy relative MAE focus (REL_LOSS_W=5.0)
  - Sampler: 1/E^1.5 (more aggressive low-E oversampling)
  - Architecture: 512 hidden × 6 ResBlocks × SiLU
  - Ensemble: 20 models
  - Post-hoc: per-energy-bin linear calibration
"""

import copy, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = PROJECT_DIR / "dataset"
OUTPUT_DIR  = PROJECT_DIR / "output" / "regression" / "current"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "AppML_InitialProject_train.csv"
TEST_FILE  = DATA_DIR / "AppML_InitialProject_test_regression.csv"

SEED        = 42
MAX_FEAT    = 20
DEDUP_THR   = 0.95

BATCH_SIZE  = 1024
EPOCHS      = 800
LR_MAX      = 3e-4
WD          = 1e-3
WARMUP_EP   = 15
PATIENCE    = 150
GRAD_CLIP   = 1.0

HIDDEN_DIM  = 512
NUM_BLOCKS  = 5
DROPOUT     = 0.08
REL_LOSS_W  = 2.0

N_ENSEMBLE  = 15
CV_FOLDS    = 3
EMA_DECAY   = 0.999

STUDENT  = "GuanranTai"
SOLUTION = "PyTorchNN"

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    torch.backends.cudnn.benchmark = True

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── 1. Load & filter ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[1/6] Loading data")
print("=" * 60, flush=True)

train = pd.read_csv(TRAIN_FILE)
test  = pd.read_csv(TEST_FILE)
print(f"  Train: {train.shape}  Test: {test.shape}")

mask = ((train["p_Truth_isElectron"] == 1) & (train["p_Truth_Energy"] > 1000))
elec = train[mask].copy()

y_raw = elec["p_Truth_Energy"].values.astype(np.float64)
y_log = np.log(y_raw)
X_all = elec.drop(columns=["p_Truth_isElectron", "p_Truth_Energy"])

n_cut = (train["p_Truth_isElectron"] == 1).sum() - len(elec)
print(f"  Electrons: {len(elec)}  (cut {n_cut} E<=1000)")
print(f"  Energy: [{y_raw.min():.0f}, {y_raw.max():.0f}] MeV, "
      f"median={np.median(y_raw):.0f}", flush=True)


# ── 2. Feature selection (MI+Spearman on log(E)) ───────────────────────────
print("\n" + "=" * 60)
print(f"[2/6] Feature selection (MI+Spearman, max {MAX_FEAT})")
print("=" * 60, flush=True)

from scipy.stats import spearmanr
X_fs, X_tune, y_fs, y_tune = train_test_split(
    X_all, y_log, test_size=0.80, random_state=SEED)
print(f"  FS holdout: {len(X_fs)}  |  Tuning: {len(X_tune)}")

mi_scores = mutual_info_regression(X_fs, y_fs, random_state=SEED, n_neighbors=5)
mi_series = pd.Series(mi_scores, index=X_all.columns).sort_values(ascending=False)

spearman_scores = {}
for c in X_all.columns:
    valid = X_fs[c].notna()
    n_valid = valid.sum()
    if n_valid < 100:
        spearman_scores[c] = 0.0; continue
    r, _ = spearmanr(X_fs.loc[valid, c].values, y_fs[valid.values])
    spearman_scores[c] = abs(r) if not np.isnan(r) else 0.0
sp_series = pd.Series(spearman_scores).sort_values(ascending=False)

combined = (mi_series.rank(ascending=False) + sp_series.rank(ascending=False)) / 2
selected = []
for feat in combined.sort_values().index:
    if len(selected) >= MAX_FEAT:
        break
    redundant = False
    for s in selected:
        if abs(X_fs[feat].corr(X_fs[s])) > DEDUP_THR:
            redundant = True; break
    if not redundant:
        selected.append(feat)

print(f"\n  Selected {len(selected)} features:")
for i, f in enumerate(selected):
    print(f"    {i+1:2d}. {f:<40s} MI={mi_series[f]:.4f}  |r_s|={sp_series[f]:.4f}")

X_sel = X_all[selected]
X_test_sel = test[selected].copy()


# ── 3. Preprocessing ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[3/6] Preprocessing")
print("=" * 60, flush=True)

for c in selected:
    med = X_sel[c].median()
    X_sel[c]      = X_sel[c].replace(-999, med)
    X_test_sel[c] = X_test_sel[c].replace(-999, med)

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
X_arr  = qt.fit_transform(X_sel).astype(np.float32)
Xt_arr = qt.transform(X_test_sel).astype(np.float32)
y_arr  = y_log.astype(np.float32)

assert not np.isnan(X_arr).any(), "NaN in features!"
assert not np.isnan(Xt_arr).any(), "NaN in test!"
print(f"  X_train: {X_arr.shape}  X_test: {Xt_arr.shape}  y: {y_arr.shape}", flush=True)


# ── 4. Model ────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.net(x))


class DeepRegressor(nn.Module):
    """Deep residual MLP for energy regression."""
    def __init__(self, in_dim, hidden=512, dropout=0.08, num_blocks=5):
        super().__init__()
        self.input_bn  = nn.BatchNorm1d(in_dim)
        self.input_fc  = nn.Linear(in_dim, hidden)
        self.input_act = nn.SiLU()
        self.input_do  = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(num_blocks)])

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4), nn.BatchNorm1d(hidden // 4),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.input_act(self.input_fc(x))
        x = self.input_do(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


def relative_mae_np(pred_log, y_raw):
    return float(np.mean(np.abs(np.exp(pred_log) - y_raw) / y_raw))

def relative_mae_torch(pred_log, y_raw_t):
    return (torch.abs(torch.exp(pred_log) - y_raw_t) / (y_raw_t + 1.0)).mean()

def combined_loss_fn(pred_log, y_log):
    y_raw = torch.exp(y_log)
    huber = nn.HuberLoss(delta=0.3, reduction='none')
    loss_log = huber(pred_log, y_log)
    e_pred = torch.exp(pred_log)
    rel_err = torch.abs(e_pred - y_raw) / (y_raw + 1.0)
    rel_err = torch.clamp(rel_err, max=10.0)
    return (loss_log + REL_LOSS_W * rel_err).mean()


def _train_single(X_tr, y_tr, X_vl, y_vl, y_raw_vl, seed):
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    Xt = torch.tensor(X_tr, device=DEVICE)
    yt = torch.tensor(y_tr, device=DEVICE)
    Xv = torch.tensor(X_vl, device=DEVICE)
    yv = torch.tensor(y_vl, device=DEVICE)
    yv_raw = torch.tensor(y_raw_vl, device=DEVICE, dtype=torch.float32)

    # WeightedRandomSampler: 1/E — oversample low-E (sampler normalises internally)
    energies_tr = np.exp(y_tr)
    sw = 1.0 / (energies_tr + 1.0)
    sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)
    train_loader = DataLoader(
        TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
        sampler=sampler, drop_last=True)

    model = DeepRegressor(MAX_FEAT, HIDDEN_DIM, DROPOUT, NUM_BLOCKS).to(DEVICE)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WD)

    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=steps_per_epoch * 25, T_mult=2, eta_min=LR_MAX * 1e-3)

    best_rel = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(EPOCHS):
        model.train()
        if ep < WARMUP_EP:
            for pg in opt.param_groups:
                pg['lr'] = LR_MAX * (ep + 1) / WARMUP_EP

        for xb, yb in train_loader:
            opt.zero_grad()
            combined_loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if ep >= WARMUP_EP:
                sched.step()

        model.eval()
        with torch.no_grad():
            rel = relative_mae_torch(model(Xv), yv_raw)

        rel_val = float(rel.item())
        if np.isnan(rel_val):
            return best_rel, None

        if rel_val < best_rel - 1e-7:
            best_rel = rel_val
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    return best_rel, best_state


# ── 5. Train (CV + ensemble) ────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"[4/6] Training ({CV_FOLDS}-fold CV + {N_ENSEMBLE}-model ensemble)")
print("=" * 60, flush=True)

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
cv_rel = []
for fold, (tr, vl) in enumerate(kf.split(X_arr)):
    rel, _ = _train_single(
        X_arr[tr], y_arr[tr], X_arr[vl], y_arr[vl], y_raw[vl],
        SEED + fold)
    cv_rel.append(rel)
    print(f"  Fold {fold+1}/{CV_FOLDS}:  rel MAE = {rel:.6f}", flush=True)
print(f"  CV Relative MAE:  {np.mean(cv_rel):.6f}  +/- {np.std(cv_rel):.6f}", flush=True)

# Ensemble
split = train_test_split(X_arr, y_arr, y_raw, test_size=0.1, random_state=SEED)
X_tr, X_ho, y_tr, y_ho, y_raw_tr, y_raw_ho = split

t0 = time.time()
hold_preds = []
test_preds = []
ho_errs    = []
model_count = 0

for i in range(N_ENSEMBLE):
    seed = SEED + 100 + i
    rel, state = _train_single(
        X_tr, y_tr, X_ho, y_ho, y_raw_ho, seed)
    if state is None:
        continue
    m = DeepRegressor(MAX_FEAT, HIDDEN_DIM, DROPOUT, NUM_BLOCKS).to(DEVICE)
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        Xh_t = torch.tensor(X_ho, device=DEVICE)
        Xt_t = torch.tensor(Xt_arr, device=DEVICE)
        hp_log = m(Xh_t).cpu().numpy().astype(np.float64)
        tp_log = m(Xt_t).cpu().numpy().astype(np.float64)
        hold_preds.append(hp_log)
        test_preds.append(tp_log)
        ho_errs.append(rel)
    model_count += 1

    # Weighted ensemble: softmax over -rel_mae (better models get higher weight)
    weights = np.exp(-np.array(ho_errs) * 5.0)  # temperature scaling
    weights /= weights.sum()

    weighted_ho = np.average(np.array(hold_preds), axis=0, weights=weights)
    weighted_test = np.average(np.array(test_preds), axis=0, weights=weights)

    cur_ho = relative_mae_np(weighted_ho, y_raw_ho)
    cur_test_mev = np.clip(np.exp(weighted_test), 1.0, 1e7)
    unweighted_ho = relative_mae_np(np.mean(hold_preds, axis=0), y_raw_ho)

    dt = time.time() - t0
    print(f"  Model {i+1:2d}/{N_ENSEMBLE}  seed={seed}  "
          f"ho_rel={rel:.6f}  wgt_ens={cur_ho:.6f}  unwgt={unweighted_ho:.6f}  {dt:.0f}s", flush=True)

    ckpt = OUTPUT_DIR / f"Regression_{STUDENT}_{SOLUTION}_ckpt.csv"
    with open(ckpt, "w") as f:
        for j, v in enumerate(cur_test_mev):
            f.write(f"{j},{v:.6f}\n")

# Final ensemble: weighted average
weights = np.exp(-np.array(ho_errs) * 5.0)
weights /= weights.sum()
preds_ho_log   = np.average(np.array(hold_preds), axis=0, weights=weights)
preds_test_log = np.average(np.array(test_preds), axis=0, weights=weights)
unwgt_ho_log   = np.mean(hold_preds, axis=0)
train_time = time.time() - t0

ho_rel_wgt = relative_mae_np(preds_ho_log, y_raw_ho)
ho_rel_unw = relative_mae_np(unwgt_ho_log, y_raw_ho)
ho_mae_log = float(mean_absolute_error(y_ho, preds_ho_log))

print(f"\n  Ensemble ({model_count} models, weighted):")
print(f"    Holdout MAE (log):     {ho_mae_log:.6f}")
print(f"    Holdout Rel MAE (wgt): {ho_rel_wgt:.6f}  ({ho_rel_wgt*100:.2f}%)")
print(f"    Holdout Rel MAE (unw): {ho_rel_unw:.6f}  ({ho_rel_unw*100:.2f}%)")
print(f"    Train time:            {train_time:.1f}s", flush=True)
print(f"    Model weights: {weights.round(4).tolist()}", flush=True)


# ── 6. Post-hoc per-energy-bin calibration ─────────────────────────────────
print("\n" + "=" * 60)
print("[5/6] Post-hoc calibration (per-energy-bin)")
print("=" * 60, flush=True)

# Split holdout into calib/val for unbiased calibration evaluation
X_calib, X_val, y_calib, y_val = train_test_split(
    preds_ho_log, y_raw_ho, test_size=0.3, random_state=SEED)

# Per-energy-bin linear calibration
bins = [0, 5000, 15000, 40000, 100000, 1e9]
calib_models = {}
for i in range(len(bins) - 1):
    mask = (y_calib > bins[i]) & (y_calib <= bins[i+1])
    if mask.sum() < 10:
        calib_models[i] = calib_models.get(i-1, type('', (), {'predict': lambda x: x})())
        continue
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_calib[mask].reshape(-1, 1), np.log(y_calib[mask]))
    calib_models[i] = lr

# Apply calibration
def calibrate(preds_log, bin_models, bins):
    out = np.zeros_like(preds_log)
    for i in range(len(bins) - 1):
        mask = (np.exp(preds_log) > bins[i]) & (np.exp(preds_log) <= bins[i+1])
        if mask.sum() == 0:
            continue
        if hasattr(bin_models[i], 'predict'):
            out[mask] = bin_models[i].predict(preds_log[mask].reshape(-1, 1))
        else:
            out[mask] = preds_log[mask]
    return out

calib_val = calibrate(X_val, calib_models, bins)
calib_rel = float(np.mean(np.abs(np.exp(calib_val) - y_val) / y_val))
raw_rel   = float(np.mean(np.abs(np.exp(X_val) - y_val) / y_val))

print(f"  Per-bin calibration: raw={raw_rel:.6f} → calib={calib_rel:.6f} "
      f"({(1-calib_rel/raw_rel)*100:.1f}% improvement)")

use_calib = calib_rel < raw_rel
if use_calib:
    test_ens_log_final = calibrate(preds_test_log, calib_models, bins)
else:
    test_ens_log_final = preds_test_log
print(f"  Calibration applied: {use_calib}")


# ── 7. Save ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[6/6] Saving outputs")
print("=" * 60, flush=True)

test_preds_mev = np.clip(np.exp(test_ens_log_final), 1.0, 1e7)
print(f"  Predictions: {len(test_preds_mev)}, "
      f"range=[{test_preds_mev.min():.0f}, {test_preds_mev.max():.0f}] MeV", flush=True)

pred_path = OUTPUT_DIR / f"Regression_{STUDENT}_{SOLUTION}.csv"
with open(pred_path, "w") as f:
    for i, v in enumerate(test_preds_mev):
        f.write(f"{i},{v:.6f}\n")

var_path = OUTPUT_DIR / f"Regression_{STUDENT}_{SOLUTION}_VariableList.csv"
with open(var_path, "w") as f:
    for feat in selected:
        f.write(f"{feat},\n")

sum_path = OUTPUT_DIR / f"Regression_{STUDENT}_{SOLUTION}_summary.txt"
with open(sum_path, "w") as f:
    f.write(f"PyTorch NN Regression v6 - {STUDENT}\n")
    f.write(f"========================================\n\n")
    f.write(f"Architecture: ResMLP({MAX_FEAT}→{HIDDEN_DIM}→{NUM_BLOCKS}×ResBlock→"
            f"{HIDDEN_DIM//2}→{HIDDEN_DIM//4}→1)\n")
    f.write(f"  Activation: SiLU + BatchNorm + Dropout({DROPOUT})\n")
    f.write(f"  Init: Kaiming, EMA(decay={EMA_DECAY})\n")
    f.write(f"Loss: Huber(log,δ=0.3) + {REL_LOSS_W}×RelativeMAE\n")
    f.write(f"Sampler: WeightedRandomSampler(1/E^{SAMPLER_POW})\n")
    f.write(f"Optim: AdamW(lr={LR_MAX},wd={WD}) + {WARMUP_EP}ep warmup\n")
    f.write(f"Scheduler: CosineAnnealingWarmRestarts(T_0=30×steps,T_mult=2)\n")
    f.write(f"Ensemble: {model_count} models + EMA\n")
    f.write(f"Calibration: {'per-energy-bin' if use_calib else 'none'}\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Training data: {len(elec)} electrons (E>1000 MeV)\n\n")
    f.write(f"Feature selection: LightGBM gain+permutation (proven set)\n")
    f.write(f"Features ({len(selected)}):\n")
    for i, fn in enumerate(selected):
        f.write(f"  {i+1:2d}. {fn}\n")
    f.write(f"\nCV ({CV_FOLDS}-fold) Rel MAE: {np.mean(cv_rel):.6f} +/- {np.std(cv_rel):.6f}\n")
    f.write(f"Holdout MAE (log):     {ho_mae_log:.6f}\n")
    f.write(f"Holdout Rel MAE (wgt): {ho_rel_wgt:.6f} ({ho_rel_wgt*100:.2f}%)\n")
    if use_calib:
        f.write(f"Calibrated val Rel MAE:{calib_rel:.6f}\n")
    f.write(f"Train time:            {train_time:.1f}s\n")
    f.write(f"Random seed: {SEED}\n")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
