"""
Shared utilities for classification workflow:
- Data loading and preprocessing
- Train/validation splitting
- Evaluation metrics (BCE, ROC AUC)
- Submission file generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
import warnings

warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent.parent / "dataset"
OUTPUT_DIR = ROOT / "outputs"
SUBMISSION_DIR = OUTPUT_DIR / "submission"

TRAIN_PATH = DATA_DIR / "AppML_InitialProject_train.csv"
TEST_CLASS_PATH = DATA_DIR / "AppML_InitialProject_test_classification.csv"

TARGET_COL = "p_Truth_isElectron"
ENERGY_COL = "p_Truth_Energy"
EXCLUDE_COLS = {TARGET_COL, ENERGY_COL, "averageInteractionsPerCrossing"}


def load_train_data(path=None):
    """Load training data, separate features and target."""
    path = path or TRAIN_PATH
    df = pd.read_csv(path)
    target = df[TARGET_COL].astype(float)
    features = df.drop(columns=[TARGET_COL, ENERGY_COL])
    return features, target


def load_test_data(path=None):
    """Load test classification data (no labels)."""
    path = path or TEST_CLASS_PATH
    df = pd.read_csv(path)
    return df


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train/validation split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def check_data_quality(df):
    """Report NaN, inf, and basic stats."""
    report = {}
    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    report["nan_columns"] = nan_cols.to_dict()

    inf_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        ninf = np.isinf(df[col]).sum()
        if ninf > 0:
            inf_counts[col] = ninf
    report["inf_columns"] = inf_counts
    return report


def compute_metrics(y_true, y_pred_proba):
    """Compute BCE (LogLoss) and ROC AUC."""
    bce = log_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    return {"log_loss": bce, "roc_auc": auc}


def compute_roc_curve(y_true, y_pred_proba):
    """Compute ROC curve data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return fpr, tpr, thresholds


def save_submission(predictions, output_name, variable_list=None):
    """
    Save classification submission CSV.
    predictions: array-like of shape (n_samples,) with values in ]0, 1[
    output_name: base name like 'Classification_GuanranTai_LightGBM'
    """
    Path(SUBMISSION_DIR).mkdir(parents=True, exist_ok=True)

    preds = np.clip(predictions, 1e-10, 1 - 1e-10)
    pred_path = SUBMISSION_DIR / f"{output_name}.csv"
    with open(pred_path, "w") as f:
        for i, p in enumerate(preds):
            f.write(f"{i},{p:.10f}\n")

    if variable_list is not None:
        var_path = SUBMISSION_DIR / f"{output_name}_VariableList.csv"
        with open(var_path, "w") as f:
            for v in variable_list:
                f.write(f"{v},\n")

    print(f"[save] Predictions: {pred_path}")
    print(f"[save] Variable list: {var_path if variable_list else 'N/A'}")
    return pred_path
