"""
Shared configuration for regression task: paths, constants, seed.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent.parent / "dataset"
OUTPUT_DIR = ROOT / "outputs"
SUBMISSION_DIR = OUTPUT_DIR / "submission"

TRAIN_PATH = DATA_DIR / "AppML_InitialProject_train.csv"
TEST_REG_PATH = DATA_DIR / "AppML_InitialProject_test_regression.csv"

TARGET_COL = "p_Truth_Energy"
ELECTRON_COL = "p_Truth_isElectron"
SENTINEL_ENERGY = -999

N_TOP_FEATURES = 20

TOP20_PATH = OUTPUT_DIR / "top20_features.txt"
RANKING_PATH = OUTPUT_DIR / "feature_ranking.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

BATCH_SIZE = 512
NN_EPOCHS = 200
NN_PATIENCE = 30
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4

LGBM_CV_FOLDS = 5
LGBM_EARLY_STOP = 50
LGBM_N_ESTIMATORS_CV = 300
LGBM_N_ESTIMATORS_FINAL = 2000

BAYESOPT_INIT = 10
BAYESOPT_ITER = 40
OPTUNA_TRIALS = 50
