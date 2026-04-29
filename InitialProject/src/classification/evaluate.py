"""
Evaluation: ROC curves, SHAP, metric comparison plots (all models).
"""

import sys, json, joblib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import torch

from config import OUTPUT_DIR, RANDOM_STATE
from preprocessing import (
    load_train_data, split_data, scale_features,
    load_top_features, compute_metrics,
)

plt.rcParams.update({"font.size": 12, "figure.dpi": 100})
np.random.seed(RANDOM_STATE)


def plot_roc_curves(y_val, preds_dict, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, preds), color in zip(preds_dict.items(), colors):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_val, preds)
        auc = compute_metrics(y_val, preds)["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves - Electron Classification", xlim=[0, 1], ylim=[0, 1])
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[plot] {save_path.name}")


def plot_shap_summary(shap_vals, features, save_path):
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, features, plot_type="violin",
                      max_display=15, show=False)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {save_path.name}")


def plot_shap_bar(shap_vals, features, save_path):
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, features, plot_type="bar",
                      max_display=15, show=False)
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {save_path.name}")


def plot_metric_comparison(all_metrics, save_path):
    configs = list(all_metrics.keys())
    loglosses = [all_metrics[m]["log_loss"] for m in configs]
    aucs = [all_metrics[m]["roc_auc"] for m in configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    ax1.bar(range(len(configs)), loglosses, color=colors[:len(configs)],
            alpha=0.85, edgecolor="black")
    ax1.set_ylabel("LogLoss"); ax1.set_title("LogLoss Comparison")
    ax1.set_xticks(range(len(configs))); ax1.set_xticklabels(configs, rotation=15, ha="right")
    for i, v in enumerate(loglosses):
        ax1.text(i, v + 0.002, f"{v:.5f}", ha="center", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(range(len(configs)), aucs, color=colors[:len(configs)],
            alpha=0.85, edgecolor="black")
    ax2.set_ylabel("ROC AUC"); ax2.set_title("ROC AUC Comparison")
    ax2.set_xticks(range(len(configs))); ax2.set_xticklabels(configs, rotation=15, ha="right")
    for i, v in enumerate(aucs):
        ax2.text(i, v + 0.001, f"{v:.5f}", ha="center", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    if min(aucs) > 0.98:
        ax2.set_ylim(min(aucs) - 0.005, 1.0)

    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[plot] {save_path.name}")


def plot_feature_set_comparison(save_path):
    """Bar chart: LGBM/NN evaluated on LGBM Top-15 vs NN Top-15."""
    import json
    data_path = OUTPUT_DIR / "nn_vs_lgbm_features.json"
    if not data_path.exists():
        print("[plot] Skipped (no nn_vs_lgbm_features.json)")
        return

    with open(data_path) as f:
        comp = json.load(f)

    rows_data = []
    for eval_name, results_key in [("LGBM", "lgbm_on_each"), ("NN", "nn_on_each")]:
        for feat_name in ["LGBM Top 15", "NN Top 15"]:
            val = comp[results_key][feat_name]
            rows_data.append({
                "label": f"{eval_name} → {feat_name}",
                "log_loss": val["log_loss"],
                "roc_auc": val["roc_auc"],
            })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(rows_data))
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]
    labels = [r["label"] for r in rows_data]
    ll_vals = [r["log_loss"] for r in rows_data]
    auc_vals = [r["roc_auc"] for r in rows_data]

    ax1.bar(x, ll_vals, color=colors, alpha=0.85, edgecolor="black")
    for i, v in enumerate(ll_vals):
        ax1.text(i, v + 0.002, f"{v:.5f}", ha="center", fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("LogLoss"); ax1.set_title("LogLoss by Model + Feature Set")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, auc_vals, color=colors, alpha=0.85, edgecolor="black")
    for i, v in enumerate(auc_vals):
        ax2.text(i, v + 0.001, f"{v:.5f}", ha="center", fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("ROC AUC"); ax2.set_title("AUC by Model + Feature Set")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(min(auc_vals) - 0.005, 1.0)

    plt.tight_layout(); fig.savefig(save_path, dpi=150);
    plt.close(fig)
    print(f"[plot] {save_path.name}")


def main():
    print("=" * 60)
    print("Evaluation: ROC, SHAP, Metrics")
    print("=" * 60)

    X, y = load_train_data()
    top_features = load_top_features()
    X = X[top_features]
    _, X_val, _, y_val = split_data(X, y)

    # Load models
    print("\n[1] Loading models...")
    _, X_val, _, y_val = split_data(X, y)
    scaler = joblib.load(OUTPUT_DIR / "pytorch_scaler_best.pkl")
    X_val_s = scaler.transform(X_val).astype(np.float32)
    model_bayes = joblib.load(OUTPUT_DIR / "lightgbm_bayesopt_model.pkl")
    model_optuna = joblib.load(OUTPUT_DIR / "lightgbm_optuna_model.pkl")

    # Predict LGBM
    preds_bayes = model_bayes.predict_proba(X_val)[:, 1]
    preds_optuna = model_optuna.predict_proba(X_val)[:, 1]

    # Predict PyTorch
    checkpoint = torch.load(OUTPUT_DIR / "pytorch_nn_best.pt", map_location="cpu", weights_only=False)
    from models.pytorch_nn import ElectronClassifier, predict
    model_nn = ElectronClassifier(input_dim=len(top_features))
    model_nn.load_state_dict(checkpoint["model_state_dict"])
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_nn = model_nn.to(DEVICE)
    preds_nn = predict(model_nn, X_val_s)

    preds_dict = {
        "BayesOpt LGBM": preds_bayes,
        "Optuna LGBM": preds_optuna,
        "PyTorch NN": preds_nn,
    }
    for name, p in preds_dict.items():
        m = compute_metrics(y_val, p)
        print(f"    {name:20s}  LogLoss={m['log_loss']:.6f}  AUC={m['roc_auc']:.6f}")

    # ROC
    print("\n[2] ROC curves...")
    plot_roc_curves(y_val, preds_dict, OUTPUT_DIR / "roc_curves.png")

    # SHAP - LGBM
    print("\n[3] SHAP (LightGBM)...")
    n_shap = min(1000, len(X_val))
    X_shap = X_val.sample(n=n_shap, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model_bayes)
    shap_vals_lgbm = explainer.shap_values(X_shap)
    if isinstance(shap_vals_lgbm, list):
        shap_vals_lgbm = shap_vals_lgbm[1]
    plot_shap_summary(shap_vals_lgbm, X_shap, OUTPUT_DIR / "shap_summary_lgbm.png")
    plot_shap_bar(shap_vals_lgbm, X_shap, OUTPUT_DIR / "shap_bar_lgbm.png")

    # SHAP - PyTorch
    print("[4] SHAP (PyTorch NN)...")
    try:
        shap_path = OUTPUT_DIR / "pytorch_shap.npy"
        if shap_path.exists():
            shap_vals_nn = np.load(shap_path)
            model_nn.to("cpu")
            n = min(len(shap_vals_nn), len(X_val_s))
            X_shap_np = X_val_s[:n]
            X_shap_df = pd.DataFrame(X_shap_np, columns=top_features)
            plot_shap_summary(shap_vals_nn, X_shap_df,
                              OUTPUT_DIR / "shap_summary_pytorch.png")
    except Exception as e:
        print(f"    SHAP skipped: {e}")

    # Metric comparison
    print("\n[5] Metric comparison...")
    metrics = {name: compute_metrics(y_val, p) for name, p in preds_dict.items()}

    try:
        with open(OUTPUT_DIR / "feature_comparison.json") as f:
            fc = json.load(f)
        if "all_140" in fc:
            metrics["All 140"] = fc["all_140"]
    except FileNotFoundError:
        pass

    plot_metric_comparison(metrics, OUTPUT_DIR / "metric_comparison.png")

    # Feature selection comparison
    print("\n[6] Feature set comparison...")
    plot_feature_set_comparison(OUTPUT_DIR / "feature_set_comparison.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Model':20s}  {'LogLoss':>10s}  {'ROC AUC':>10s}")
    print(f"  {'-'*44}")
    for name, m in metrics.items():
        print(f"  {name:20s}  {m['log_loss']:10.6f}  {m['roc_auc']:10.6f}")

    json.dump(metrics, open(OUTPUT_DIR / "evaluation_summary.json", "w"), indent=2)
    print(f"\nAll plots saved to {OUTPUT_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
