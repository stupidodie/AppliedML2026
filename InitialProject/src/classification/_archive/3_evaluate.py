"""
Step 3 - Evaluation:
ROC curves, SHAP analysis, and comparison plots for both models.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import shap
from lightgbm import LGBMClassifier

from utils import (
    load_train_data,
    split_data,
    compute_metrics,
    compute_roc_curve,
    OUTPUT_DIR,
)

random_state = 42
np.random.seed(random_state)

matplotlib.rcParams.update({"font.size": 12, "figure.dpi": 100})


def load_top_features():
    top15_path = OUTPUT_DIR / "top15_features.txt"
    with open(top15_path) as f:
        return [line.strip() for line in f if line.strip()]


def plot_roc_curves(y_val, preds_dict, save_path):
    """Plot ROC curves for multiple models on one plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"bayesopt": "#1f77b4", "optuna": "#ff7f0e"}
    labels = {"bayesopt": "LightGBM (BayesOpt)", "optuna": "LightGBM (Optuna)"}

    for name, preds in preds_dict.items():
        fpr, tpr, _ = compute_roc_curve(y_val, preds)
        auc = compute_metrics(y_val, preds)["roc_auc"]
        ax.plot(fpr, tpr, label=f"{labels.get(name, name)} (AUC={auc:.4f})",
                color=colors.get(name, "#333"), linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Electron Classification")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] ROC curves saved: {save_path}")


def plot_shap_summary(model, X_val, top_features, save_path):
    """Generate SHAP summary plot (violin) for a trained model."""
    n_shap = min(1000, len(X_val))
    X_shap = X_val.sample(n=n_shap, random_state=random_state)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_shap,
                      feature_names=top_features,
                      plot_type="violin",
                      max_display=15,
                      show=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] SHAP summary saved: {save_path}")


def plot_shap_bar(model, X_val, top_features, save_path):
    """Generate SHAP bar plot of mean |SHAP| values."""
    n_shap = min(1000, len(X_val))
    X_shap = X_val.sample(n=n_shap, random_state=random_state)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_shap,
                      feature_names=top_features,
                      plot_type="bar",
                      max_display=15,
                      show=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] SHAP bar saved: {save_path}")


def plot_feature_importance_comparison(bayesopt_model, optuna_model, top_features, save_path):
    """Side-by-side bar plot comparing gain importance across both models."""
    gain_bayes = bayesopt_model.booster_.feature_importance(importance_type="gain")
    gain_optuna = optuna_model.booster_.feature_importance(importance_type="gain")

    # Normalize to fractions
    gain_bayes_norm = np.array(gain_bayes) / np.sum(gain_bayes)
    gain_optuna_norm = np.array(gain_optuna) / np.sum(gain_optuna)

    # Sort by bayesopt order
    sorted_idx = np.argsort(gain_bayes_norm)[::-1]
    features_sorted = [top_features[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(features_sorted))
    bar_height = 0.35

    ax.barh(y_pos + bar_height / 2, gain_bayes_norm[sorted_idx], bar_height,
            label="BayesOpt (gain)", color="#1f77b4", alpha=0.8)
    ax.barh(y_pos - bar_height / 2, gain_optuna_norm[sorted_idx], bar_height,
            label="Optuna (gain)", color="#ff7f0e", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Gain Importance")
    ax.set_title("Feature Importance Comparison (Gain)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Feature importance comparison saved: {save_path}")


def plot_metric_comparison(metrics_bayes, metrics_optuna, all_140_metrics, save_path):
    """Bar chart comparing LogLoss across configurations."""
    configs = ["All 140", "Top 15 (BayesOpt)", "Top 15 (Optuna)"]
    loglosses = [
        all_140_metrics.get("log_loss", 0),
        metrics_bayes.get("log_loss", 0),
        metrics_optuna.get("log_loss", 0),
    ]
    aucs = [
        all_140_metrics.get("roc_auc", 0),
        metrics_bayes.get("roc_auc", 0),
        metrics_optuna.get("roc_auc", 0),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    ax1.bar(configs, loglosses, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Binary Cross Entropy (LogLoss)")
    ax1.set_title("LogLoss Comparison")
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=15, ha="right")
    for i, v in enumerate(loglosses):
        ax1.text(i, v + 0.0005, f"{v:.5f}", ha="center", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(configs, aucs, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_ylabel("ROC AUC")
    ax2.set_title("ROC AUC Comparison")
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=15, ha="right")
    for i, v in enumerate(aucs):
        ax2.text(i, v + 0.001, f"{v:.5f}", ha="center", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Metric comparison saved: {save_path}")


def main():
    print("=" * 60)
    print("Step 3: Evaluation & Plots")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    X, y = load_train_data()
    top_features = load_top_features()
    print(f"    Top features: {top_features}")

    X = X[top_features]
    _, X_val, _, y_val = split_data(X, y)
    print(f"    Val set: {len(X_val)}")

    # Load models
    import joblib
    print("\n[2] Loading models...")
    model_bayes = joblib.load(OUTPUT_DIR / "lightgbm_bayesopt_model.pkl")
    model_optuna = joblib.load(OUTPUT_DIR / "lightgbm_optuna_model.pkl")

    # Predictions
    preds_bayes = model_bayes.predict_proba(X_val)[:, 1]
    preds_optuna = model_optuna.predict_proba(X_val)[:, 1]

    metrics_bayes = compute_metrics(y_val, preds_bayes)
    metrics_optuna = compute_metrics(y_val, preds_optuna)
    print(f"    BayesOpt - LogLoss: {metrics_bayes['log_loss']:.6f}, AUC: {metrics_bayes['roc_auc']:.6f}")
    print(f"    Optuna   - LogLoss: {metrics_optuna['log_loss']:.6f}, AUC: {metrics_optuna['roc_auc']:.6f}")

    # Load all-140 metrics for comparison
    try:
        with open(OUTPUT_DIR / "feature_comparison.json") as f:
            fc = json.load(f)
        all_140_metrics = fc["all_140"]
    except FileNotFoundError:
        all_140_metrics = {"log_loss": 0, "roc_auc": 0}

    # 1. ROC curves
    print("\n[3] Plotting ROC curves...")
    plot_roc_curves(
        y_val,
        {"bayesopt": preds_bayes, "optuna": preds_optuna},
        OUTPUT_DIR / "roc_curves.png",
    )

    # 2. SHAP summary (violin) - BayesOpt model
    print("[4] SHAP summary plot (BayesOpt)...")
    plot_shap_summary(
        model_bayes, X_val, top_features,
        OUTPUT_DIR / "shap_summary_bayesopt.png",
    )

    # 3. SHAP bar - BayesOpt model
    print("[5] SHAP bar plot (BayesOpt)...")
    plot_shap_bar(
        model_bayes, X_val, top_features,
        OUTPUT_DIR / "shap_bar_bayesopt.png",
    )

    # 4. Feature importance comparison
    print("[6] Feature importance comparison...")
    plot_feature_importance_comparison(
        model_bayes, model_optuna, top_features,
        OUTPUT_DIR / "feature_importance_comparison.png",
    )

    # 5. Metric comparison
    print("[7] Metric comparison bar chart...")
    plot_metric_comparison(
        metrics_bayes, metrics_optuna, all_140_metrics,
        OUTPUT_DIR / "metric_comparison.png",
    )

    # Save evaluation summary
    summary = {
        "bayesopt": metrics_bayes,
        "optuna": metrics_optuna,
        "all_140": all_140_metrics,
    }
    with open(OUTPUT_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
