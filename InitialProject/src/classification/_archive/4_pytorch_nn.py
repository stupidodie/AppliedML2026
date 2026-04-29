"""
Step 4 - PyTorch Neural Network for Electron Classification.
Feed-forward NN with BatchNorm, Dropout, class weights, early stopping.
Outputs: submission CSVs + training curves plot.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score

from utils import (
    load_train_data,
    load_test_data,
    split_data,
    save_submission,
    OUTPUT_DIR,
)

random_state = 42
np.random.seed(random_state)
torch.manual_seed(random_state)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[device] Using: {DEVICE}")

BATCH_SIZE = 512
EPOCHS_MAX = 200
EARLY_STOP_PATIENCE = 25
LR_INIT = 1e-3


def load_top_features():
    top15_path = OUTPUT_DIR / "top15_features.txt"
    with open(top15_path) as f:
        return [line.strip() for line in f if line.strip()]


class ElectronClassifier(nn.Module):
    """Feed-forward NN: 15 → 128 → 64 → 32 → 16 → 1 (sigmoid)."""

    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_n = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        total_n += len(y_batch)
    return total_loss / total_n if total_n > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        total_n += len(y_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(y_batch.cpu().numpy().tolist())
    avg_loss = total_loss / total_n if total_n > 0 else 0.0
    return avg_loss, np.array(all_probs), np.array(all_labels)


def train_model(model, train_loader, val_loader, device,
                epochs=EPOCHS_MAX, lr=LR_INIT, patience=EARLY_STOP_PATIENCE):
    pos_count = (train_loader.dataset.tensors[1] == 1).sum().item()
    neg_count = len(train_loader.dataset.tensors[1]) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    history = defaultdict(list)
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        lr_current = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(lr_current)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"val_auc={val_auc:.6f}  lr={lr_current:.2e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)
    return model, history


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train", color="#1f77b4")
    axes[0].plot(history["val_loss"], label="Val", color="#ff7f0e")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_auc"], color="#2ca02c")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_title("Val AUC")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["lr"], color="#d62728")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale("log")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Training curves saved: {save_path}")


def compute_metrics(y_true, y_probs):
    bce = log_loss(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    return {"log_loss": bce, "roc_auc": auc}


def main():
    print("=" * 60)
    print("Step 4: PyTorch Neural Network Classification")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    X, y = load_train_data()
    top_features = load_top_features()
    print(f"    Top features ({len(top_features)}): {top_features}")
    X = X[top_features]

    X_train, X_val, y_train, y_val = split_data(X, y)
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"    Pos fraction: {y.mean():.4f}")

    # Preprocessing: scale features
    print("\n[2] Preprocessing (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print(f"    X_train mean range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

    # PyTorch datasets
    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    print("\n[3] Building model...")
    model = ElectronClassifier(input_dim=len(top_features), dropout=0.3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Params: {n_params:,}")
    print(f"    Layers: 15 → 128 → 64 → 32 → 16 → 1")
    print(f"    BatchNorm + ReLU + Dropout(0.3) each hidden layer")
    print(f"    Output: BCEWithLogitsLoss with pos_weight={len(y_train[y_train==0])/len(y_train[y_train==1]):.2f}")

    # Train
    print("\n[4] Training...")
    model, history = train_model(model, train_loader, val_loader, DEVICE)

    # Evaluate
    print("\n[5] Evaluating on validation set...")
    _, val_probs, val_labels = evaluate(model, val_loader, nn.BCEWithLogitsLoss(), DEVICE)
    metrics = compute_metrics(val_labels, val_probs)
    print(f"    Validation LogLoss: {metrics['log_loss']:.6f}")
    print(f"    Validation ROC AUC: {metrics['roc_auc']:.6f}")

    # Plot training curves
    plot_training_history(history, OUTPUT_DIR / "pytorch_training_history.png")

    # SHAP with GradientExplainer
    print("\n[6] Computing SHAP (GradientExplainer, on 500 val samples)...")
    n_shap = min(500, len(X_val))
    X_shap = torch.tensor(X_val_scaled[:n_shap], dtype=torch.float32).to(DEVICE)
    background = torch.tensor(X_train_scaled[:200], dtype=torch.float32).to(DEVICE)

    model.eval()
    def predict_fn(x_tensor):
        with torch.no_grad():
            return torch.sigmoid(model(x_tensor)).cpu().numpy()

    try:
        import shap
        # Use GradientExplainer since model is small
        explainer_nn = shap.GradientExplainer(model, background)
        shap_vals_nn = explainer_nn.shap_values(X_shap)
        if isinstance(shap_vals_nn, list):
            shap_vals_nn = shap_vals_nn[0]

        fig_shap = plt.figure(figsize=(10, 8))
        X_shap_np = X_shap.cpu().numpy()
        shap.summary_plot(shap_vals_nn, X_shap_np,
                          feature_names=top_features,
                          plot_type="violin",
                          max_display=15,
                          show=False)
        plt.tight_layout()
        shap_path = OUTPUT_DIR / "shap_summary_pytorch.png"
        fig_shap.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close(fig_shap)
        print(f"[plot] SHAP summary (PyTorch) saved: {shap_path}")
    except Exception as e:
        print(f"    SHAP failed for PyTorch model: {e}")

    # Predict on test set for submission
    print("\n[7] Predicting on test set...")
    X_test = load_test_data()[top_features]
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

    model.eval()
    with torch.no_grad():
        y_test_probs = torch.sigmoid(model(X_test_tensor)).squeeze(-1).cpu().numpy()

    save_submission(
        y_test_probs,
        "Classification_GuanranTai_PyTorchNN",
        variable_list=top_features,
    )

    # Save model and scaler
    import joblib
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": len(top_features),
        "top_features": top_features,
    }, OUTPUT_DIR / "pytorch_nn_model.pt")
    joblib.dump(scaler, OUTPUT_DIR / "pytorch_scaler.pkl")

    # Save results
    results = {
        "method": "pytorch_nn",
        "architecture": "15→128→64→32→16→1",
        "params_count": n_params,
        "validation_metrics": metrics,
        "top_features": top_features,
        "hyperparams": {
            "batch_size": BATCH_SIZE,
            "lr_init": LR_INIT,
            "epochs_trained": len(history["train_loss"]),
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "dropout": 0.3,
            "class_weight_pos": float(len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
        },
    }
    with open(OUTPUT_DIR / "pytorch_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n    Model saved: {OUTPUT_DIR / 'pytorch_nn_model.pt'}")
    print(f"    Results saved: {OUTPUT_DIR / 'pytorch_results.json'}")
    print("Done.\n")


if __name__ == "__main__":
    main()
