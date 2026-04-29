"""
PyTorch NN regression: same architecture, linear output (no sigmoid),
Huber loss, log-transformed target.
"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

from config import (
    OUTPUT_DIR, RANDOM_STATE, BATCH_SIZE,
    NN_EPOCHS, NN_PATIENCE, NN_LR, NN_WEIGHT_DECAY,
)
from preprocessing import (
    load_train_data, load_test_data, split_data,
    scale_features, log_transform, load_top_features, compute_metrics,
)


np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class EnergyRegressor(nn.Module):
    """N → 256 → 128 → 64 → 1  GELU + LayerNorm + Dropout. Linear output."""
    def __init__(self, input_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader, device=DEVICE,
                lr=NN_LR, wd=NN_WEIGHT_DECAY, epochs=NN_EPOCHS,
                patience=NN_PATIENCE):
    """Train with Huber loss (robust to outliers)."""
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    best_val_loss, best_weights, no_improve = float("inf"), None, 0
    epochs_trained = 0

    for epoch in range(epochs):
        model.train()
        train_loss, n = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            n += len(yb)
        train_loss /= n
        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                loss = criterion(model(Xb).squeeze(-1), yb)
                val_loss_sum += loss.item() * len(yb)
        val_loss = val_loss_sum / len(val_loader.dataset)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                epochs_trained = epoch + 1
                break

    model.load_state_dict(best_weights)
    return model, epochs_trained


def predict(model, X_np, batch_size=4096, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            batch = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
            preds = model(batch).squeeze(-1).cpu().numpy()
            all_preds.extend(preds.tolist())
    return np.array(all_preds)


def compute_shap(model, X_sample, feature_names):
    import shap
    X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(DEVICE)
    explainer = shap.GradientExplainer(model, X_tensor[:200])
    shap_vals = explainer.shap_values(X_tensor)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.squeeze(shap_vals)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)
    return shap_vals


def main():
    print("=" * 60)
    print("PyTorch NN Regression")
    print("=" * 60)

    X_raw, y_raw = load_train_data()
    top_features = load_top_features()
    X_raw = X_raw[top_features]

    # Log transform target
    y = log_transform(y_raw)
    print(f"[Target] log1p energy: mean={y.mean():.2f} std={y.std():.2f}")

    X_train, X_val, y_train, y_val = split_data(X_raw, y)
    X_test = load_test_data()[top_features]
    X_test = X_test

    scaler, X_train_s, X_val_s, X_test_s = scale_features(X_train, X_val, X_test)

    y_train_t = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train,
                             dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val,
                           dtype=torch.float32)

    train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), y_train_t)
    val_ds = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), y_val_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_feat = len(top_features)
    print(f"\n[Data] Train={len(y_train)}  Val={len(y_val)}  Test={len(X_test)}")
    print(f"[Model] {n_feat} → 256 → 128 → 64 → 1  GELU+LayerNorm  HuberLoss")

    model = EnergyRegressor(input_dim=n_feat).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params] {n_params:,}")
    print(f"[Device] {DEVICE}")
    print(f"[Training] lr={NN_LR}, wd={NN_WEIGHT_DECAY}")

    model, epochs_done = train_model(model, train_loader, val_loader)

    # Predict (log scale)
    val_preds_log = predict(model, X_val_s)
    # Convert to original units
    val_preds = log_transform(val_preds_log, inverse=True)
    y_val_orig = log_transform(y_val.values if hasattr(y_val, 'values') else y_val,
                               inverse=True)

    metrics = compute_metrics(y_val_orig, val_preds)
    print(f"[Result] RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R²={metrics['r2']:.4f}  epochs={epochs_done}")

    # Test predictions (inverse log for original units)
    test_preds_log = predict(model, X_test_s)
    test_preds = log_transform(test_preds_log, inverse=True)

    # SHAP
    print("\n[SHAP] Computing on 500 validation samples...")
    try:
        n_shap = min(500, len(X_val_s))
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_val_s), n_shap, replace=False)
        shap_vals = compute_shap(model, X_val_s[idx], top_features)
        np.save(OUTPUT_DIR / "pytorch_shap.npy", shap_vals)
    except Exception as e:
        print(f"    SHAP skipped: {e}")

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": n_feat,
        "top_features": top_features,
    }, OUTPUT_DIR / "pytorch_nn_best.pt")
    joblib.dump(scaler, OUTPUT_DIR / "pytorch_scaler_best.pkl")
    json.dump({
        "architecture": f"{n_feat}→256→128→64→1 GELU+LayerNorm HuberLoss",
        "params": n_params, "validation_metrics": metrics,
        "top_features": top_features, "epochs": epochs_done,
    }, open(OUTPUT_DIR / "pytorch_best_results.json", "w"), indent=2)

    print(f"\nSaved: pytorch_nn_best.pt, pytorch_scaler_best.pkl")
    print("Done.\n")


if __name__ == "__main__":
    main()
