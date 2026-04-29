"""
Step 4b - PyTorch NN Architecture Experiments.
Tries multiple architectures, picks the best one.
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


def load_top_features():
    top15_path = OUTPUT_DIR / "top15_features.txt"
    with open(top15_path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Architecture variants
# ---------------------------------------------------------------------------

class WideDeep(nn.Module):
    """15→256→128→64→32→16→1  GELU + LayerNorm + Dropout"""
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.LayerNorm(16), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
    def forward(self, x): return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.block(x)


class DeepResidual(nn.Module):
    """15→256→[ResBlock×3]→128→64→1  with residual skip connections"""
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
        )
        self.res1 = ResidualBlock(256, dropout)
        self.res2 = ResidualBlock(256, dropout)
        self.res3 = ResidualBlock(256, dropout)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.head(x)


class WideDeepBN(nn.Module):
    """15→256→128→64→32→1  BatchNorm pre-activation pattern"""
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

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
    return total_loss / total_n


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
    return total_loss / total_n, np.array(all_probs), np.array(all_labels)


def train_with_cosine(model, train_loader, val_loader, device,
                      lr_init=1e-3, weight_decay=1e-4,
                      epochs=EPOCHS_MAX, patience=EARLY_STOP_PATIENCE):
    pos_count = (train_loader.dataset.tensors[1] == 1).sum().item()
    neg_count = len(train_loader.dataset.tensors[1]) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    history = defaultdict(list)
    best_val_loss, best_weights = float("inf"), None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        lr_current = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(lr_current)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}  train={train_loss:.6f}  val={val_loss:.6f}  auc={val_auc:.6f}  lr={lr_current:.2e}")

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
    return model, history, best_val_loss


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Step 4b: PyTorch NN Architecture Experiments")
    print("=" * 60)

    print("\n[1] Loading data...")
    X, y = load_train_data()
    top_features = load_top_features()
    X = X[top_features]
    X_train, X_val, y_train, y_val = split_data(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

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

    # -----------------------------------------------------------------------
    # Define experiments
    # -----------------------------------------------------------------------
    experiments = [
        {
            "name": "baseline",
            "arch": lambda: nn.Sequential(
                nn.Linear(15, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(16, 1),
            ),
            "lr": 1e-3, "wd": 1e-5, "desc": "15→128→64→32→16→1  ReLU+BN",
        },
        {
            "name": "wide_deep",
            "arch": lambda: WideDeep(15, dropout=0.3),
            "lr": 1e-3, "wd": 1e-4, "desc": "15→256→128→64→32→16→1  GELU+LayerNorm",
        },
        {
            "name": "deep_residual",
            "arch": lambda: DeepResidual(15, dropout=0.2),
            "lr": 1e-3, "wd": 1e-4, "desc": "15→256→3xResBlock→128→64→1",
        },
        {
            "name": "wide_bn",
            "arch": lambda: WideDeepBN(15, dropout=0.3),
            "lr": 1e-3, "wd": 1e-4, "desc": "15→256→128→64→32→1  GELU+BatchNorm",
        },
        {
            "name": "wide_deep_lowlr",
            "arch": lambda: WideDeep(15, dropout=0.2),
            "lr": 5e-4, "wd": 1e-4, "desc": "Same as wide_deep, lower lr=5e-4",
        },
        {
            "name": "wide_deep_nodrop",
            "arch": lambda: WideDeep(15, dropout=0.1),
            "lr": 1e-3, "wd": 1e-4, "desc": "wide_deep with dropout=0.1",
        },
    ]

    results = []
    best_model_state = None
    best_scaler = scaler
    best_val_loss = float("inf")
    best_name = ""

    print(f"\n[2] Running {len(experiments)} experiments...")
    for exp in experiments:
        name, arch_fn, lr, wd, desc = exp["name"], exp["arch"], exp["lr"], exp["wd"], exp["desc"]
        print(f"\n{'─'*50}")
        print(f"  {name}: {desc}")
        print(f"  lr={lr}, weight_decay={wd}")

        torch.manual_seed(random_state)
        model = arch_fn().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {n_params:,}")

        model, history, val_loss = train_with_cosine(
            model, train_loader, val_loader, DEVICE,
            lr_init=lr, weight_decay=wd,
        )

        _, val_probs, val_labels = evaluate(
            model, val_loader,
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([len(y_train[y_train==0])/len(y_train[y_train==1])],
                                                          device=DEVICE)),
            DEVICE,
        )
        metrics = {"log_loss": log_loss(val_labels, val_probs), "roc_auc": roc_auc_score(val_labels, val_probs)}

        print(f"  >> LogLoss={metrics['log_loss']:.6f}  AUC={metrics['roc_auc']:.6f}")

        results.append({
            "name": name, "desc": desc,
            "log_loss": metrics["log_loss"], "roc_auc": metrics["roc_auc"],
            "params": n_params, "lr": lr, "wd": wd,
            "epochs": len(history["train_loss"]),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_name = name

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Architecture Experiment Results")
    print("=" * 60)
    print(f"  {'Name':20s}  {'LogLoss':>10s}  {'AUC':>10s}  {'Params':>8s}  {'Epochs':>7s}")
    print(f"  {'-'*63}")
    for r in sorted(results, key=lambda x: x["log_loss"]):
        print(f"  {r['name']:20s}  {r['log_loss']:10.6f}  {r['roc_auc']:10.6f}  {r['params']:>8,d}  {r['epochs']:>7d}")

    print(f"\n  Best: {best_name}")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    names = [r["name"] for r in results]
    loglosses = [r["log_loss"] for r in results]
    aucs = [r["roc_auc"] for r in results]
    colors = ["#1f77b4"] * len(results)
    best_idx = next(i for i, r in enumerate(results) if r["name"] == best_name)
    colors[best_idx] = "#2ca02c"

    ax1.barh(names, loglosses, color=colors, edgecolor="black")
    ax1.set_xlabel("LogLoss")
    ax1.set_title("LogLoss by Architecture")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(loglosses):
        ax1.text(v + 0.001, i, f"{v:.5f}", va="center", fontsize=9)

    ax2.barh(names, aucs, color=colors, edgecolor="black")
    ax2.set_xlabel("ROC AUC")
    ax2.set_title("ROC AUC by Architecture")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(aucs):
        ax2.text(v + 0.0003, i, f"{v:.5f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "nn_architecture_comparison.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Save best model + predict on test
    # -----------------------------------------------------------------------
    print(f"\n[3] Saving best model ({best_name}) and predicting...")
    best_arch = next(exp["arch"] for exp in experiments if exp["name"] == best_name)
    best_model = best_arch().to(DEVICE)
    best_model.load_state_dict(best_model_state)

    X_test = load_test_data()[top_features]
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

    best_model.eval()
    with torch.no_grad():
        y_test_probs = torch.sigmoid(best_model(X_test_tensor)).squeeze(-1).cpu().numpy()

    sub_name = f"Classification_GuanranTai_PyTorchNN_{best_name}"
    save_submission(y_test_probs, sub_name, variable_list=top_features)

    import joblib
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "input_dim": len(top_features),
        "top_features": top_features,
        "arch_name": best_name,
    }, OUTPUT_DIR / f"pytorch_nn_{best_name}.pt")
    joblib.dump(scaler, OUTPUT_DIR / "pytorch_scaler_best.pkl")

    with open(OUTPUT_DIR / "nn_arch_experiments.json", "w") as f:
        json.dump({"results": results, "best": best_name}, f, indent=2)

    print(f"\nAll experiments saved.")
    print("Done.\n")


if __name__ == "__main__":
    main()
