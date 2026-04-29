"""
Model-agnostic feature selection:
  - tree : LightGBM gain + split + SHAP ranking (fast)
  - nn   : PyTorch NN GradientExplainer SHAP ranking
  - both : run both and compare (LGBM + NN on each top-15)
"""

import sys, json, argparse
import numpy as np
from pathlib import Path

# Allow importing from sibling 'models/' subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from lightgbm import early_stopping
import shap

from config import OUTPUT_DIR, RANDOM_STATE, BATCH_SIZE
from preprocessing import load_train_data, split_data, compute_metrics


np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Lightweight NN (same architecture, used for feature ranking)
# ---------------------------------------------------------------------------

class ElectronClassifier(nn.Module):
    """N → 256 → 128 → 64 → 1  GELU + LayerNorm + Dropout."""
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


def train_nn(model, train_loader, val_loader, epochs=200, patience=30,
             lr=1e-3, wd=1e-4, device=DEVICE, verbose=False):
    """Train NN with class-weighted BCE + CosineAnnealingWarmRestarts."""
    pos_count = (train_loader.dataset.tensors[1] == 1).sum().item()
    neg_count = len(train_loader.dataset.tensors[1]) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
        val_loss_sum, all_probs, all_labels = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
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
    if verbose:
        ll = log_loss(all_labels, all_probs)
        auc = roc_auc_score(all_labels, all_probs)
        print(f"    Val: LogLoss={ll:.6f}  AUC={auc:.6f}  epochs={epochs_trained}")
    return model, epochs_trained

from sklearn.metrics import log_loss, roc_auc_score


# ---------------------------------------------------------------------------
# Tree-based ranking (existing)
# ---------------------------------------------------------------------------

def rank_by_tree(X, y, feature_names):
    """LightGBM → gain + split + SHAP → combined avg-rank. Uses tuned params if available."""
    print("[tree] Training LightGBM on all features...")
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Load tuned params if available
    tuning_path = OUTPUT_DIR / "lightgbm_tuning_results.json"
    if tuning_path.exists():
        with open(tuning_path) as f:
            tuning = json.load(f)
        params = {k: v for k, v in tuning["bayesopt"]["params"].items()
                  if k != "n_estimators"}
        print("    Using tuned params from lightgbm_tuning_results.json")
    else:
        params = {"num_leaves": 31, "learning_rate": 0.05}

    model = lgb.LGBMClassifier(
        objective="binary", metric="binary_logloss",
        n_estimators=500, random_state=RANDOM_STATE, verbose=-1, force_col_wise=True,
        **params,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], eval_metric="logloss",
        callbacks=[early_stopping(stopping_rounds=20, verbose=False)],
    )
    preds = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, preds)
    print(f"    LogLoss={metrics['log_loss']:.6f}  AUC={metrics['roc_auc']:.6f}")

    # Built-in importance
    gain = dict(zip(feature_names, model.booster_.feature_importance(importance_type="gain")))
    split = dict(zip(feature_names, model.booster_.feature_importance(importance_type="split")))

    # SHAP
    print("[tree] Computing SHAP (TreeExplainer, 2000 val samples)...")
    n_shap = min(2000, len(X_val))
    X_shap = X_val.sample(n=n_shap, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_imp = dict(zip(feature_names, np.abs(shap_vals).mean(axis=0)))

    # Combined rank
    def rank_pos(feat, ranked):
        for i, (name, _) in enumerate(ranked):
            if name == feat:
                return i + 1
        return len(ranked)

    gain_rank = sorted(gain.items(), key=lambda x: x[1], reverse=True)
    split_rank = sorted(split.items(), key=lambda x: x[1], reverse=True)
    shap_rank = sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)

    combined = {}
    for feat in feature_names:
        combined[feat] = {
            "avg_rank": (rank_pos(feat, gain_rank) + rank_pos(feat, split_rank) + rank_pos(feat, shap_rank)) / 3.0,
            "gain": float(gain.get(feat, 0)),
            "split": float(split.get(feat, 0)),
            "shap": float(shap_imp.get(feat, 0)),
        }
    sorted_combined = sorted(combined.items(), key=lambda x: x[1]["avg_rank"])
    return sorted_combined, metrics


# ---------------------------------------------------------------------------
# NN-based ranking
# ---------------------------------------------------------------------------

def rank_by_nn(X, y, feature_names, n_shap=500):
    """PyTorch NN on all features → GradientExplainer SHAP → ranking."""
    from sklearn.preprocessing import StandardScaler

    print("[nn] Splitting & scaling...")
    X_train, X_val, y_train, y_val = split_data(X, y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    train_ds = TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train.values, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_s), torch.tensor(y_val.values, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[nn] Training on all {len(feature_names)} features...")
    torch.manual_seed(RANDOM_STATE)
    model = ElectronClassifier(input_dim=len(feature_names)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Params: {n_params:,}  Device: {DEVICE}")
    model, epochs = train_nn(model, train_loader, val_loader, verbose=True)

    # SHAP
    print(f"[nn] Computing SHAP (GradientExplainer, {n_shap} val samples)...")
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_val_s), n_shap, replace=False)
    X_sample = X_val_s[idx]

    X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(DEVICE)
    explainer = shap.GradientExplainer(model, X_tensor[:200])
    shap_vals_nn = explainer.shap_values(X_tensor)
    if isinstance(shap_vals_nn, list):
        shap_vals_nn = shap_vals_nn[0]
    shap_vals_nn = np.squeeze(shap_vals_nn)
    if shap_vals_nn.ndim == 1:
        shap_vals_nn = shap_vals_nn.reshape(1, -1)

    mean_abs = np.abs(shap_vals_nn).mean(axis=0)
    mean_abs = np.asarray(mean_abs).ravel()

    assert len(mean_abs) == len(feature_names), \
        f"SHAP shape mismatch: {mean_abs.shape} vs {len(feature_names)}"

    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    combined = {feat: {"shap": float(val)} for feat, val in ranked}
    return ranked, combined


# ---------------------------------------------------------------------------
# Comparison: train LGBM on each top-15
# ---------------------------------------------------------------------------

def compare_feature_sets(X, y, tree_top15, nn_top15):
    """Train LightGBM on each top-15 and compare LogLoss."""
    print("\n[compare] Training LightGBM on each top-15...")
    X_train, X_val, y_train, y_val = split_data(X, y)

    results = {}
    for name, feats in [("LGBM Top 15", tree_top15), ("NN Top 15", nn_top15)]:
        model = lgb.LGBMClassifier(
            objective="binary", metric="binary_logloss",
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            random_state=RANDOM_STATE, verbose=-1, force_col_wise=True,
        )
        model.fit(
            X_train[feats], y_train,
            eval_set=[(X_val[feats], y_val)],
            eval_metric="logloss",
            callbacks=[early_stopping(stopping_rounds=20, verbose=False)],
        )
        preds = model.predict_proba(X_val[feats])[:, 1]
        m = compute_metrics(y_val, preds)
        results[name] = m
        print(f"    {name:15s}  LogLoss={m['log_loss']:.6f}  AUC={m['roc_auc']:.6f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature selection: tree / nn / both")
    parser.add_argument("--method", choices=["tree", "nn", "both"], default="both",
                        help="Ranking method (default: both)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Feature Selection (method={args.method})")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_train_data()
    all_features = list(X.columns)
    print(f"\nData: {len(X)} samples, {len(all_features)} features\n")

    # --- Tree ranking ---
    if args.method in ("tree", "both"):
        tree_ranked, tree_metrics_all = rank_by_tree(X, y, all_features)
        tree_top15 = [feat for feat, _ in tree_ranked[:15]]

        print(f"\n[tree] Top 15 features (avg rank):")
        for i, (feat, info) in enumerate(tree_ranked[:15]):
            print(f"  {i+1:2d}. {feat:45s}  rank={info['avg_rank']:.1f}  gain={info['gain']:.0f}  |SHAP|={info['shap']:.4f}")

        # Save
        tree_combined = {feat: info for feat, info in tree_ranked}
        json.dump(tree_combined, open(OUTPUT_DIR / "feature_ranking.json", "w"), indent=2)
        with open(OUTPUT_DIR / "top15_features.txt", "w") as f:
            for feat in tree_top15:
                f.write(f"{feat}\n")
        print(f"\nSaved: top15_features.txt, feature_ranking.json")

    # --- NN ranking ---
    if args.method in ("nn", "both"):
        print()
        nn_ranked, nn_combined = rank_by_nn(X, y, all_features)
        nn_top15 = [feat for feat, _ in nn_ranked[:15]]

        print(f"\n[nn] Top 15 features (|SHAP|):")
        for i, (feat, val) in enumerate(nn_ranked[:15]):
            print(f"  {i+1:2d}. {feat:45s}  |SHAP|={val:.6f}")

        json.dump(nn_combined, open(OUTPUT_DIR / "nn_feature_ranking.json", "w"), indent=2)
        with open(OUTPUT_DIR / "nn_top15_features.txt", "w") as f:
            for feat in nn_top15:
                f.write(f"{feat}\n")
        print(f"\nSaved: nn_top15_features.txt, nn_feature_ranking.json")

    # --- Comparison ---
    if args.method == "both":
        common = set(tree_top15) & set(nn_top15)
        tree_only = set(tree_top15) - set(nn_top15)
        nn_only = set(nn_top15) - set(tree_top15)
        print(f"\n{'='*60}")
        print("FEATURE SET COMPARISON")
        print(f"{'='*60}")
        print(f"  Common:   {len(common)} features → {sorted(common)}")
        print(f"  Tree only: {len(tree_only)} features → {sorted(tree_only)}")
        print(f"  NN only:   {len(nn_only)} features → {sorted(nn_only)}")

        # LGBM on each top-15
        lgbm_results = compare_feature_sets(X, y, tree_top15, nn_top15)

        # NN on each top-15
        print("\n[compare] Training PyTorch NN on each top-15...")
        from sklearn.preprocessing import StandardScaler
        X_train, X_val, y_train, y_val = split_data(X, y)

        nn_results = {}
        for name, feats in [("LGBM Top 15", tree_top15), ("NN Top 15", nn_top15)]:
            scaler_sub = StandardScaler()
            X_tr_s = scaler_sub.fit_transform(X_train[feats]).astype(np.float32)
            X_val_s = scaler_sub.transform(X_val[feats]).astype(np.float32)

            train_ds = TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_train.values, dtype=torch.float32))
            val_ds = TensorDataset(torch.tensor(X_val_s), torch.tensor(y_val.values, dtype=torch.float32))
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

            torch.manual_seed(RANDOM_STATE)
            m_nn = ElectronClassifier(input_dim=len(feats)).to(DEVICE)
            m_nn, _ = train_nn(m_nn, train_loader, val_loader)

            m_nn.eval()
            with torch.no_grad():
                logits = m_nn(torch.tensor(X_val_s).to(DEVICE)).squeeze(-1).cpu().numpy()
                probs = 1 / (1 + np.exp(-logits))
            nn_results[name] = {"log_loss": float(log_loss(y_val, probs)),
                                "roc_auc": float(roc_auc_score(y_val, probs))}
            print(f"    {name:15s}  LogLoss={nn_results[name]['log_loss']:.6f}  AUC={nn_results[name]['roc_auc']:.6f}")

        # Summary table
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Evaluator':10s}  {'Feature Set':15s}  {'LogLoss':>10s}  {'AUC':>10s}")
        print(f"  {'-'*52}")
        print(f"  {'LGBM':10s}  {'LGBM Top 15':15s}  {lgbm_results['LGBM Top 15']['log_loss']:10.6f}  {lgbm_results['LGBM Top 15']['roc_auc']:10.6f}")
        print(f"  {'LGBM':10s}  {'NN Top 15':15s}    {lgbm_results['NN Top 15']['log_loss']:10.6f}  {lgbm_results['NN Top 15']['roc_auc']:10.6f}")
        print(f"  {'NN':10s}    {'LGBM Top 15':15s}  {nn_results['LGBM Top 15']['log_loss']:10.6f}  {nn_results['LGBM Top 15']['roc_auc']:10.6f}")
        print(f"  {'NN':10s}    {'NN Top 15':15s}    {nn_results['NN Top 15']['log_loss']:10.6f}  {nn_results['NN Top 15']['roc_auc']:10.6f}")

        # Save full comparison
        output = {
            "tree_top15": tree_top15,
            "nn_top15": nn_top15,
            "overlap": sorted(list(common)),
            "tree_only": sorted(list(tree_only)),
            "nn_only": sorted(list(nn_only)),
            "lgbm_on_each": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                                 for kk, vv in v.items()}
                            for k, v in lgbm_results.items()},
            "nn_on_each": nn_results,
        }
        json.dump(output, open(OUTPUT_DIR / "nn_vs_lgbm_features.json", "w"), indent=2)
        print(f"\nSaved: nn_vs_lgbm_features.json")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
