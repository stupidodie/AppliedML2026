"""
Data loading, scaling, train/val split, metrics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from config import (
    TRAIN_PATH, TEST_CLASS_PATH, TARGET_COL, ENERGY_COL,
    RANDOM_STATE, TEST_SIZE, OUTPUT_DIR,
)

import warnings
warnings.filterwarnings("ignore")


def load_train_data(path=None):
    path = path or TRAIN_PATH
    df = pd.read_csv(path)
    target = df[TARGET_COL].astype(float)
    features = df.drop(columns=[TARGET_COL, ENERGY_COL])
    return features, target


def load_test_data(path=None):
    path = path or TEST_CLASS_PATH
    return pd.read_csv(path)


def split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def scale_features(X_train, X_val, X_test=None):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32) if X_val is not None else None
    X_test_s = scaler.transform(X_test).astype(np.float32) if X_test is not None else None
    return scaler, X_train_s, X_val_s, X_test_s


def compute_metrics(y_true, y_pred_proba):
    return {
        "log_loss": log_loss(y_true, y_pred_proba),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }


def check_data_quality(df):
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


def load_top_features():
    from config import TOP15_PATH
    with open(TOP15_PATH) as f:
        return [line.strip() for line in f if line.strip()]
