"""
Feature ranking from NN perspective: train NN on all 140 features,
rank by SHAP values, compare top 15 with LightGBM's top 15.
"""

import json, sys, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from InitialProject.src.classification.utils import load_train_data, split_data, OUTPUT_DIR

random_state = 42
np.random.seed(random_state); torch.manual_seed(random_state)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[device] {DEVICE}")


class BestNN(nn.Module):
    def __init__(self, input_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)


def train_best_nn(X_train_s, y_train, X_val_s, y_val, input_dim,
                  lr=1e-3, wd=1e-4, epochs=200, patience=30):
    train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32),
                             torch.tensor(y_train.values, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32),
                           torch.tensor(y_val.values, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    torch.manual_seed(random_state)
    model = BestNN(input_dim).to(DEVICE)
    pos_count = int((y_train == 1).sum())
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0
    epochs_trained = 0

    for epoch in range(epochs):
        model.train()
        train_loss, n = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            n += len(yb)
        train_loss /= n
        scheduler.step()

        model.eval()
        val_loss_sum, all_probs, all_labels = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb).squeeze(-1)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * len(yb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(yb.cpu().numpy().tolist())
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
    return model, best_val_loss, epochs_trained, criterion


def compute_nn_shap_ranking(model, X_sample, feature_names):
    """Rank features by mean |SHAP| using GradientExplainer."""
    import shap
    X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(DEVICE)
    background = X_tensor[:200]

    explainer = shap.GradientExplainer(model, background)
    shap_vals = explainer.shap_values(X_tensor)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    # Handle shapes: (n_samples, n_features) is ideal
    #   (n_samples, n_features, 1) or (1, ...) needs squeeze
    shap_vals = np.squeeze(shap_vals)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_abs = np.asarray(mean_abs).ravel()
    assert len(mean_abs) == len(feature_names), f"shape mismatch: {len(mean_abs)} vs {len(feature_names)}"

    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    return ranked, shap_vals


def evaluate(model, X_val_s, y_val):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_val_s, dtype=torch.float32).to(DEVICE)).squeeze(-1).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
    return log_loss(y_val, probs), roc_auc_score(y_val, probs)


def main():
    print("=" * 60)
    print("NN Feature Ranking on ALL 140 features")
    print("=" * 60)

    # Load all 140 features
    X, y = load_train_data()
    all_features = list(X.columns)
    print(f"\n[1] Data: {X.shape[1]} features, {len(X)} samples")

    X_train, X_val, y_train, y_val = split_data(X, y)

    # Scale all 140 features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    # Train NN on all 140
    print(f"\n[2] Training NN on all {len(all_features)} features...")
    model, val_loss, epochs, criterion_obj = train_best_nn(
        X_train_s, y_train, X_val_s, y_val, input_dim=len(all_features))
    logloss, auc = evaluate(model, X_val_s, y_val)
    print(f"    LogLoss={logloss:.6f}  AUC={auc:.6f}  epochs={epochs}")

    # SHAP ranking on all 140
    print(f"\n[3] Computing SHAP (GradientExplainer, 500 samples)...")
    n_shap = min(500, len(X_val_s))
    np.random.seed(random_state)
    idx = np.random.choice(len(X_val_s), n_shap, replace=False)
    X_shap = X_val_s[idx]
    ranked, _ = compute_nn_shap_ranking(model, X_shap, all_features)

    nn_top15 = [f for f, _ in ranked[:15]]
    print(f"\n    NN Top 15 features:")
    for i, (f, v) in enumerate(ranked[:15]):
        val = float(v)
        print(f"      {i+1:2d}. {f:45s}  |SHAP|={val:.6f}")

    # Load LGBM top 15
    lgbm_top15_path = OUTPUT_DIR / "top15_features.txt"
    with open(lgbm_top15_path) as f:
        lgbm_top15 = [l.strip() for l in f if l.strip()]

    print(f"\n[4] Comparing feature sets...")
    common = set(nn_top15) & set(lgbm_top15)
    nn_only = set(nn_top15) - set(lgbm_top15)
    lgbm_only = set(lgbm_top15) - set(nn_top15)
    print(f"    Common: {len(common)} features")
    print(f"    NN only: {len(nn_only)} features -> {sorted(nn_only)}")
    print(f"    LGBM only: {len(lgbm_only)} features -> {sorted(lgbm_only)}")

    # Train NN on each feature set
    print(f"\n[5] Comparing NN performance on different feature sets...")
    results = {}
    for name, feat_list in [("LGBM Top 15", lgbm_top15), ("NN Top 15", nn_top15)]:
        print(f"\n--- Training on {name}...")
        X_train_sub = X_train[feat_list].values.astype(np.float32)
        X_val_sub = X_val[feat_list].values.astype(np.float32)

        scaler_sub = StandardScaler()
        X_train_sub_s = scaler_sub.fit_transform(X_train_sub)
        X_val_sub_s = scaler_sub.transform(X_val_sub)

        model_sub, _, epochs_sub, _ = train_best_nn(
            X_train_sub_s, y_train, X_val_sub_s, y_val, input_dim=len(feat_list))
        logloss_sub, auc_sub = evaluate(model_sub, X_val_sub_s, y_val)
        results[name] = {"log_loss": logloss_sub, "roc_auc": auc_sub, "epochs": epochs_sub}
        print(f"    LogLoss={logloss_sub:.6f}  AUC={auc_sub:.6f}")

    # Print final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"  {'Feature Set':20s}  {'LogLoss':>10s}  {'AUC':>10s}")
    print(f"  {'-'*46}")
    for name, m in results.items():
        print(f"  {name:20s}  {m['log_loss']:10.6f}  {m['roc_auc']:10.6f}")
    print(f"  {'All 140 (NN)':20s}  {logloss:10.6f}  {auc:10.6f}")
    print(f"  {'LightGBM (ref)':20s}  {'0.078023':>10s}  {'0.993545':>10s}")

    # Save results
    output = {
        "all_140_nn": {"log_loss": logloss, "roc_auc": auc},
        "results_by_featureset": results,
        "lgbm_top15": lgbm_top15,
        "nn_top15": nn_top15,
        "overlap": sorted(list(common)),
        "nn_only": sorted(list(nn_only)),
        "lgbm_only": sorted(list(lgbm_only)),
    }
    with open(OUTPUT_DIR / "nn_vs_lgbm_features.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'nn_vs_lgbm_features.json'}")


if __name__ == "__main__":
    main()
