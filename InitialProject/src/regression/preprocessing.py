"""
Data loading, cleaning (-999 sentinel), log1p transform, scaling, metrics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import (
    TRAIN_PATH, TEST_REG_PATH, TARGET_COL, ELECTRON_COL,
    SENTINEL_ENERGY, RANDOM_STATE, TEST_SIZE,
)

import warnings
warnings.filterwarnings("ignore")


def load_train_data(path=None):
    """Load training data, filter -999 sentinel, return X and raw y."""
    path = path or TRAIN_PATH
    df = pd.read_csv(path)

    # Filter out sentinel energy values
    before = len(df)
    df = df[df[TARGET_COL] != SENTINEL_ENERGY].copy()
    n_dropped = before - len(df)
    if n_dropped > 0:
        print(f"    Dropped {n_dropped} rows with energy={SENTINEL_ENERGY}")

    features = df.drop(columns=[TARGET_COL, ELECTRON_COL])
    target = df[TARGET_COL].astype(float)
    return features, target


def load_test_data(path=None):
    path = path or TEST_REG_PATH
    return pd.read_csv(path)


def split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def log_transform(y, inverse=False):
    """log1p or expm1."""
    if inverse:
        return np.expm1(y)
    return np.log1p(y)


def scale_features(X_train, X_val, X_test=None):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32) if X_val is not None else None
    X_test_s = scaler.transform(X_test).astype(np.float32) if X_test is not None else None
    return scaler, X_train_s, X_val_s, X_test_s


def compute_metrics(y_true, y_pred):
    """Regression metrics (on original scale, not log)."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_top_features():
    from config import TOP20_PATH
    with open(TOP20_PATH) as f:
        return [line.strip() for line in f if line.strip()]
