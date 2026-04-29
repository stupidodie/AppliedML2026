"""
Step 5 - Final Evaluation:
Add PyTorch NN to ROC curves and metric comparison.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from utils import (
    load_train_data,
    split_data,
    OUTPUT_DIR,
)

random_state = 42
np.random.seed(random_state)


def load_top_features():
    top15_path = OUTPUT_DIR / "top15_features.txt"
    with open(top15_path) as f:
        return [line.strip() for line in f if line.strip()]


class ElectronClassifierSimple(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128), torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64), torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 32), torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 16), torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x)


def load_nn_predictions(X_val_df):
    import torch
    import joblib

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(
        OUTPUT_DIR / "pytorch_nn_model.pt",
        map_location=DEVICE,
        weights_only=False,
    )
    scaler = joblib.load(OUTPUT_DIR / "pytorch_scaler.pkl")

    model = ElectronClassifierSimple(input_dim=len(load_top_features()))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    X_scaled = scaler.transform(X_val_df)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(X_tensor)).squeeze(-1).cpu().numpy()
    return probs


def main():
    print("=" * 60)
    print("Step 5: Final Evaluation (PyTorch NN added)")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data and models...")
    X, y = load_train_data()
    top_features = load_top_features()
    X = X[top_features]
    _, X_val, _, y_val = split_data(X, y)

    import joblib
    model_bayes = joblib.load(OUTPUT_DIR / "lightgbm_bayesopt_model.pkl")
    model_optuna = joblib.load(OUTPUT_DIR / "lightgbm_optuna_model.pkl")

    preds_bayes = model_bayes.predict_proba(X_val)[:, 1]
    preds_optuna = model_optuna.predict_proba(X_val)[:, 1]

    # PyTorch NN predictions
    print("[2] Computing PyTorch NN predictions...")
    preds_nn = load_nn_predictions(X_val)

    # Metrics
    metrics = {
        "BayesOpt LightGBM": {"log_loss": log_loss(y_val, preds_bayes), "roc_auc": roc_auc_score(y_val, preds_bayes)},
        "Optuna LightGBM": {"log_loss": log_loss(y_val, preds_optuna), "roc_auc": roc_auc_score(y_val, preds_optuna)},
        "PyTorch NN": {"log_loss": log_loss(y_val, preds_nn), "roc_auc": roc_auc_score(y_val, preds_nn)},
    }
    for name, m in metrics.items():
        print(f"    {name:20s}  LogLoss={m['log_loss']:.6f}  AUC={m['roc_auc']:.6f}")

    # ROC curves with all 3 models
    print("\n[3] Plotting ROC curves (all 3 models)...")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_roc = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    pred_map = {
        "BayesOpt LightGBM": preds_bayes,
        "Optuna LightGBM": preds_optuna,
        "PyTorch NN": preds_nn,
    }
    for name, color, (preds_name, preds) in zip(metrics.keys(), colors_roc, pred_map.items()):
        fpr, tpr, _ = roc_curve(y_val, preds)
        ax.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['roc_auc']:.4f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Electron Classification")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves_all.png", dpi=150)
    plt.close(fig)
    print(f"[plot] ROC curves saved: {OUTPUT_DIR / 'roc_curves_all.png'}")

    # Metric comparison bar
    print("\n[4] Plotting metric comparison (all 3)...")
    try:
        with open(OUTPUT_DIR / "feature_comparison.json") as f:
            fc = json.load(f)
        all_140 = fc["all_140"]
    except:
        all_140 = {"log_loss": 0, "roc_auc": 0}

    configs = ["All 140", "BayesOpt\nLightGBM", "Optuna\nLightGBM", "PyTorch NN"]
    loglosses = [
        all_140.get("log_loss", 0),
        metrics["BayesOpt LightGBM"]["log_loss"],
        metrics["Optuna LightGBM"]["log_loss"],
        metrics["PyTorch NN"]["log_loss"],
    ]
    aucs = [
        all_140.get("roc_auc", 0),
        metrics["BayesOpt LightGBM"]["roc_auc"],
        metrics["Optuna LightGBM"]["roc_auc"],
        metrics["PyTorch NN"]["roc_auc"],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors_bar = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    ax1.bar(range(len(configs)), loglosses, color=colors_bar, alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Binary Cross Entropy (LogLoss)")
    ax1.set_title("LogLoss Comparison")
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs)
    for i, v in enumerate(loglosses):
        ax1.text(i, v + 0.002, f"{v:.5f}", ha="center", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(range(len(configs)), aucs, color=colors_bar, alpha=0.85, edgecolor="black")
    ax2.set_ylabel("ROC AUC")
    ax2.set_title("ROC AUC Comparison")
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs)
    for i, v in enumerate(aucs):
        ax2.text(i, v + 0.002, f"{v:.5f}", ha="center", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0.985, 1.0)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "metric_comparison_all.png", dpi=150)
    plt.close(fig)
    print(f"[plot] Metric comparison saved: {OUTPUT_DIR / 'metric_comparison_all.png'}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"  {'Model':20s} {'LogLoss':>10s}  {'ROC AUC':>10s}")
    print(f"  {'-'*42}")
    for name, m in metrics.items():
        print(f"  {name:20s} {m['log_loss']:10.6f}  {m['roc_auc']:10.6f}")
    print(f"  {'All 140 (reference)':20s} {all_140.get('log_loss', 0):10.6f}  {all_140.get('roc_auc', 0):10.6f}")

    summary = {
        "models": {name: m for name, m in metrics.items()},
        "all_140_reference": all_140,
    }
    with open(OUTPUT_DIR / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFull summary saved: {OUTPUT_DIR / 'final_summary.json'}")
    print("Done.\n")


if __name__ == "__main__":
    main()
