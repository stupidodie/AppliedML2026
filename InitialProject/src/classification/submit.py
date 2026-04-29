"""
Generate final submission CSV files from trained models.
"""

import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib, torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, SUBMISSION_DIR
from preprocessing import (
    load_test_data, scale_features, load_top_features,
)
from models.pytorch_nn import ElectronClassifier, predict as nn_predict


def save_submission(predictions, output_name, variable_list=None):
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    preds = np.clip(predictions, 1e-10, 1 - 1e-10)
    pred_path = SUBMISSION_DIR / f"{output_name}.csv"
    with open(pred_path, "w") as f:
        for i, p in enumerate(preds):
            f.write(f"{i},{p:.10f}\n")

    if variable_list:
        var_path = SUBMISSION_DIR / f"{output_name}_VariableList.csv"
        with open(var_path, "w") as f:
            for v in variable_list:
                f.write(f"{v},\n")
    return pred_path


def verify_submission(path):
    lines = open(path).read().strip().split("\n")
    n = len(lines)
    first = lines[0].split(",")
    assert len(first) == 2, f"Expected 2 columns, got {len(first)}"
    assert first[0].strip().isdigit(), f"Index not integer: {first[0]}"
    prob = float(first[1].strip())
    assert 0 < prob < 1, f"Probability not in ]0,1[: {prob}"
    return n


def main():
    print("=" * 60)
    print("Generate Submission CSVs")
    print("=" * 60)

    top_features = load_top_features()
    X_test = load_test_data()[top_features]
    _, _, _, X_test_s = scale_features(X_test, None, X_test)

    submissions = []

    # LightGBM BayesOpt
    print("\n[1] LightGBM BayesOpt...")
    model = joblib.load(OUTPUT_DIR / "lightgbm_bayesopt_model.pkl")
    test_probs = model.predict_proba(X_test)[:, 1]
    save_submission(test_probs, "Classification_GuanranTai_LightGBM",
                    variable_list=top_features)
    submissions.append(("Classification_GuanranTai_LightGBM", "LightGBM (BayesOpt)"))

    # LightGBM Optuna
    print("[2] LightGBM Optuna...")
    model = joblib.load(OUTPUT_DIR / "lightgbm_optuna_model.pkl")
    test_probs = model.predict_proba(X_test)[:, 1]
    save_submission(test_probs, "Classification_GuanranTai_LightGBM_Optuna",
                    variable_list=top_features)
    submissions.append(("Classification_GuanranTai_LightGBM_Optuna", "LightGBM (Optuna)"))

    # PyTorch NN
    print("[3] PyTorch NN...")
    checkpoint = torch.load(OUTPUT_DIR / "pytorch_nn_best.pt",
                            map_location="cpu", weights_only=False)
    model_nn = ElectronClassifier(input_dim=len(top_features))
    model_nn.load_state_dict(checkpoint["model_state_dict"])
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_nn = model_nn.to(DEVICE)
    test_probs = nn_predict(model_nn, X_test_s)
    save_submission(test_probs, "Classification_GuanranTai_PyTorchNN_Best",
                    variable_list=top_features)
    submissions.append(("Classification_GuanranTai_PyTorchNN_Best", "PyTorch NN"))

    # Verify
    print("\n[4] Verifying submissions...")
    print(f"  {'Name':50s}  {'Lines':>7s}  {'Status':>10s}")
    print(f"  {'-'*70}")
    for fname, label in submissions:
        path = SUBMISSION_DIR / f"{fname}.csv"
        try:
            n = verify_submission(path)
            var_path = SUBMISSION_DIR / f"{fname}_VariableList.csv"
            v_lines = len(open(var_path).read().strip().split("\n"))
            ok = "✓" if (n == 60000 and v_lines == 15) else "✗"
            print(f"  {fname:50s}  {n:>7d}  {ok:>10s}")
        except Exception as e:
            print(f"  {fname:50s}  {'ERROR':>7s}  {str(e):>10s}")

    print(f"\nAll submission files in: {SUBMISSION_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
