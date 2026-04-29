"""
Quick focused run: compare top architectures + the ones that timed out.
"""

import json, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_train_data, split_data, OUTPUT_DIR

random_state = 42
np.random.seed(random_state)
torch.manual_seed(random_state)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 512

def load_top_features():
    with open(OUTPUT_DIR / "top15_features.txt") as f:
        return [line.strip() for line in f if line.strip()]


class WideDeep_tuned(nn.Module):
    """15→256→128→64→1  GELU+LayerNorm, optimized depth"""
    def __init__(self, input_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
        )
    def forward(self, x): return x + self.block(x)


class DeepResidualLight(nn.Module):
    """15→128→[ResBlock×2]→64→32→1  lighter residual"""
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
        )
        self.res1 = ResidualBlock(128, dropout)
        self.res2 = ResidualBlock(128, dropout)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)


class WideDeepBN_tuned(nn.Module):
    """15→256→128→64→1  GELU+BatchNorm"""
    def __init__(self, input_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)


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


def train_quick(model, train_loader, val_loader, device, lr=1e-3, wd=1e-4):
    pos_count = (train_loader.dataset.tensors[1] == 1).sum().item()
    neg_count = len(train_loader.dataset.tensors[1]) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    best_val_loss = float("inf")
    best_weights = None
    patience, no_improve = 25, 0

    for epoch in range(200):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_weights)
    _, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
    return {
        "log_loss": log_loss(val_labels, val_probs),
        "roc_auc": roc_auc_score(val_labels, val_probs),
        "epochs": epoch + 1,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def main():
    print("Quick Architectures Comparison (4 configs)")
    print("=" * 60)

    X, y = load_train_data()
    top_features = load_top_features()
    X = X[top_features]
    X_train, X_val, y_train, y_val = split_data(X, y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    configs = [
        ("wide_deep", WideDeep_tuned(15).to(DEVICE), "15→256→128→64→1  GELU+LayerNorm"),
        ("deep_res_light", DeepResidualLight(15).to(DEVICE), "15→128→2xResBlock→64→32→1"),
        ("wide_bn", WideDeepBN_tuned(15).to(DEVICE), "15→256→128→64→1  GELU+BatchNorm"),
    ]

    results = {}
    for name, model, desc in configs:
        torch.manual_seed(random_state)
        print(f"\n--- {name}: {desc}")
        m = train_quick(model, train_loader, val_loader, DEVICE)
        results[name] = m
        print(f"    LogLoss={m['log_loss']:.6f}  AUC={m['roc_auc']:.6f}  params={m['params']:,}  epochs={m['epochs']}")

    print("\n" + "=" * 60)
    print("Summary (including previous run's best)")
    print("=" * 60)

    # Include results from earlier failed batches (from the console output we saw)
    prior = {
        "original_v1": {"log_loss": 0.125855, "roc_auc": 0.990142, "params": 13409},
        "baseline_rerun": {"log_loss": 0.131985, "roc_auc": 0.989600, "params": 13409},
        "wide_deep_big": {"log_loss": 0.118769, "roc_auc": 0.990712, "params": 48865},
        "deep_residual_v1": {"log_loss": 0.116720, "roc_auc": 0.991194, "params": 444033},
    }
    all_results = {**prior, **results}

    print(f"  {'Name':25s}  {'LogLoss':>10s}  {'AUC':>10s}  {'Params':>8s}")
    print(f"  {'-'*58}")
    for name, m in sorted(all_results.items(), key=lambda x: x[1]["log_loss"]):
        if "epochs" in m:
            print(f"  {name:25s}  {m['log_loss']:10.6f}  {m['roc_auc']:10.6f}  {m['params']:>8,d}")
        else:
            print(f"  {name:25s}  {m['log_loss']:10.6f}  {m['roc_auc']:10.6f}  {m['params']:>8,d}")

    with open(OUTPUT_DIR / "nn_arch_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_DIR / 'nn_arch_results.json'}")


if __name__ == "__main__":
    main()
