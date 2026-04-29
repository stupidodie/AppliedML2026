"""
Evaluation: residual plots, SHAP, metric comparison (all regression models).
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
    load_train_data, split_data, log_transform,
    load_top_features, compute_metrics,
)

plt.rcParams.update({"font.size": 12, "figure.dpi": 100})
np.random.seed(RANDOM_STATE)


def plot_residuals(y_true, preds_dict, save_path):
    """Histogram + scatter of residuals for each model."""
    n_models = len(preds_dict)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 8))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for j, (name, preds) in enumerate(preds_dict.items()):
        residuals = y_true - preds

        # Histogram
        ax = axes[0, j]
        ax.hist(residuals, bins=80, color=colors[j % len(colors)],
                alpha=0.75, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Residual (true − pred)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}")
        ax.grid(True, alpha=0.3)

        # Scatter
        ax = axes[1, j]
        ax.scatter(y_true, preds, s=1, alpha=0.3, color=colors[j % len(colors)])
        lims = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
        ax.plot(lims, lims, "r--", alpha=0.7)
        ax.set_xlabel("True Energy")
        ax.set_ylabel("Predicted Energy")
        ax.set_title(f"{name}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {save_path.name}")


def plot_shap_summary(shap_vals, features, save_path):
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, features, plot_type="violin",
                      max_display=15, show=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {save_path.name}")


def plot_shap_bar(shap_vals, features, save_path):
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, features, plot_type="bar",
                      max_display=15, show=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {save_path.name}")


def plot_metric_comparison(all_metrics, save_path):
    configs = list(all_metrics.keys())
    rmses = [all_metrics[m]["rmse"] for m in configs]
    maes = [all_metrics[m]["mae"] for m in configs]
    r2s = [all_metrics[m]["r2"] for m in configs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    for ax, vals, title, fmt in [
        (axes[0], rmses, "RMSE", ".0f"),
        (axes[1], maes, "MAE", ".0f"),
        (axes[2], r2s, "R²", ".4f"),
    ]:
        ax.bar(range(len(configs)), vals, color=colors[:len(configs)],
               alpha=0.85, edgecolor="black")
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:{fmt}}", ha="center", fontsize=9)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=15, ha="right")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {save_path.name}")


def main():
    print("=" * 60)
    print("Evaluation: Residuals, SHAP, Metrics")
    print("=" * 60)

    X_raw, y_raw = load_train_data()
    top_features = load_top_features()
    X_raw = X_raw[top_features]

    # Split and get original-scale y_val
    _, X_val, _, y_val_raw = split_data(X_raw, y_raw)

    # Load models
    print("\n[1] Loading models...")
    scaler = joblib.load(OUTPUT_DIR / "pytorch_scaler_best.pkl")
    X_val_s = scaler.transform(X_val).astype(np.float32)

    model_bayes = joblib.load(OUTPUT_DIR / "lightgbm_bayesopt_model.pkl")
    model_optuna = joblib.load(OUTPUT_DIR / "lightgbm_optuna_model.pkl")

    # LGBM: trained on log target, so predict returns log values
    preds_bayes_log = model_bayes.predict(X_val)
    preds_optuna_log = model_optuna.predict(X_val)
    preds_bayes = log_transform(preds_bayes_log, inverse=True)
    preds_optuna = log_transform(preds_optuna_log, inverse=True)

    # PyTorch NN
    checkpoint = torch.load(OUTPUT_DIR / "pytorch_nn_best.pt",
                            map_location="cpu", weights_only=False)
    from models.pytorch_nn import EnergyRegressor, predict
    model_nn = EnergyRegressor(input_dim=len(top_features))
    model_nn.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_nn = model_nn.to(device)
    preds_nn_log = predict(model_nn, X_val_s)
    preds_nn = log_transform(preds_nn_log, inverse=True)

    y_val_orig = y_val_raw.values if hasattr(y_val_raw, 'values') else y_val_raw

    preds_dict = {
        "BayesOpt LGBM": preds_bayes,
        "Optuna LGBM": preds_optuna,
        "PyTorch NN": preds_nn,
    }
    for name, p in preds_dict.items():
        m = compute_metrics(y_val_orig, p)
        print(f"    {name:20s}  RMSE={m['rmse']:>10.2f}  MAE={m['mae']:>10.2f}  R²={m['r2']:>.4f}")

    # Residual plots
    print("\n[2] Residual plots...")
    plot_residuals(y_val_orig, preds_dict, OUTPUT_DIR / "residuals.png")

    # SHAP - LGBM
    print("[3] SHAP (LightGBM)...")
    n_shap = min(1000, len(X_val))
    X_shap = X_val.sample(n=n_shap, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model_bayes)
    shap_vals_lgbm = explainer.shap_values(X_shap)
    if isinstance(shap_vals_lgbm, list):
        shap_vals_lgbm = shap_vals_lgbm[1] if len(shap_vals_lgbm) > 1 else shap_vals_lgbm[0]
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
    print("[5] Metric comparison...")
    metrics = {name: compute_metrics(y_val_orig, p) for name, p in preds_dict.items()}

    try:
        with open(OUTPUT_DIR / "feature_comparison.json") as f:
            fc = json.load(f)
        if "all_140" in fc:
            metrics["All 140"] = fc["all_140"]
    except FileNotFoundError:
        pass

    plot_metric_comparison(metrics, OUTPUT_DIR / "metric_comparison.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Model':20s}  {'RMSE':>10s}  {'MAE':>10s}  {'R²':>10s}")
    print(f"  {'-'*56}")
    for name, m in metrics.items():
        print(f"  {name:20s}  {m['rmse']:10.2f}  {m['mae']:10.2f}  {m['r2']:10.4f}")

    json.dump(metrics, open(OUTPUT_DIR / "evaluation_summary.json", "w"), indent=2)
    print(f"\nAll plots saved to {OUTPUT_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
