"""
Generate regression submission CSVs: predict on 40k test set,
inverse log-transform, output index,prediction.
"""

import sys, joblib, torch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, SUBMISSION_DIR
from preprocessing import (
    load_train_data, load_test_data, log_transform,
    load_top_features, scale_features,
)


def main():
    print("=" * 60)
    print("Generate Regression Submission CSVs")
    print("=" * 60)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data (only features, 40k rows)
    X_test = load_test_data()
    top_features = load_top_features()
    X_test = X_test[top_features]

    # Fit scaler on full training data (for consistency)
    X_raw, _ = load_train_data()
    X_raw = X_raw[top_features]
    scaler = joblib.load(OUTPUT_DIR / "pytorch_scaler_best.pkl")
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # ---- LightGBM BayesOpt ----
    print("[1] LightGBM BayesOpt...")
    model_bo = joblib.load(OUTPUT_DIR / "lightgbm_bayesopt_model.pkl")
    preds_bo_log = model_bo.predict(X_test)
    preds_bo = log_transform(preds_bo_log, inverse=True)
    write_csv(preds_bo, SUBMISSION_DIR / "Regression_GuanranTai_LightGBM.csv")

    # ---- LightGBM Optuna ----
    print("[2] LightGBM Optuna...")
    model_op = joblib.load(OUTPUT_DIR / "lightgbm_optuna_model.pkl")
    preds_op_log = model_op.predict(X_test)
    preds_op = log_transform(preds_op_log, inverse=True)
    write_csv(preds_op, SUBMISSION_DIR / "Regression_GuanranTai_LightGBM_Optuna.csv")

    # ---- PyTorch NN ----
    print("[3] PyTorch NN...")
    checkpoint = torch.load(OUTPUT_DIR / "pytorch_nn_best.pt",
                            map_location="cpu", weights_only=False)
    from models.pytorch_nn import EnergyRegressor, predict
    model_nn = EnergyRegressor(input_dim=len(top_features))
    model_nn.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_nn = model_nn.to(device)
    preds_nn_log = predict(model_nn, X_test_s)
    preds_nn = log_transform(preds_nn_log, inverse=True)
    write_csv(preds_nn, SUBMISSION_DIR / "Regression_GuanranTai_PyTorchNN.csv")

    # ---- Verify ----
    print("\n[4] Verifying submissions...")
    print(f"  {'Name':50s}  {'Lines':>8s}  {'Status'}")
    print(f"  {'-'*68}")
    for csv_path in sorted(SUBMISSION_DIR.glob("*.csv")):
        if "_VariableList" in csv_path.name:
            continue
        with open(csv_path) as f:
            lines = sum(1 for _ in f) - 1  # subtract header
        status = "✓" if lines == 40000 else f"✗ ({lines} lines, expected 40000)"
        print(f"  {csv_path.name:50s}  {lines:>8d}     {status}")

    # Variable list
    varlist_path = SUBMISSION_DIR / "Regression_GuanranTai_LightGBM_VariableList.csv"
    with open(varlist_path, "w") as f:
        for feat in top_features:
            f.write(f"{feat}\n")

    print(f"\nAll submission files in: {SUBMISSION_DIR}")
    print("Done.\n")


def write_csv(predictions, path):
    with open(path, "w") as f:
        f.write("index,prediction\n")
        for i, val in enumerate(predictions):
            f.write(f"{i},{float(val)}\n")


if __name__ == "__main__":
    main()
