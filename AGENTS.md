# AppliedML 2026 — Initial Project

## Project Overview

Two tasks on the same dataset (electron identification in particle physics):

| | Classification | Regression |
|---|---|---|
| Target | `p_Truth_isElectron` (0/1) | `p_Truth_Energy` (continuous) |
| Train rows | 180,000 | 180,000 |
| Test rows | 60,000 | 40,000 |
| Max features | 15 | 20 |
| Output col | `probability` ∈ [0,1] | `prediction` ∈ (−∞,∞) |
| Metric | LogLoss, AUC | RMSE, MAE, R² |
| LGBM objective | `binary` | `regression` |
| NN loss | BCEWithLogitsLoss (weighted) | HuberLoss |
| Submission name | `Classification_GuanranTai_*` | `Regression_GuanranTai_*` |

Source code: `InitialProject/src/classification/` and `InitialProject/src/regression/`
Data: `InitialProject/dataset/`
Python: `.venv/bin/python` (Python 3.12, packages: torch, sklearn, lightgbm, shap, bayes-opt, optuna)
Device: MPS (Apple Metal) for PyTorch

---

## Correct Pipeline Flow (important!)

```
① tune hyperparams on ALL features   →  python models/lightgbm_tune.py
② rank & select features (tuned params) →  python feature_selection.py --method tree
③ train final LGBM on top-N         →  python models/lightgbm_tune.py --final
④ train PyTorch NN on top-N         →  python models/pytorch_nn.py
⑤ evaluate & plot                   →  python evaluate.py
⑥ generate submission CSVs          →  python submit.py
```

**Why this order?** Hyperparameter tuning should use full information (all features). Then features are selected using the best-found params. Then final models are trained on just the selected features with those optimal params.

---

## Classification — Status: Done (but wrong flow)

The classification pipeline was completed with the **old/wrong flow** (feature selection with default params → tune on selected features). It should be re-run with the correct flow for best results.

### Current results (old flow):
| Model | LogLoss | AUC |
|-------|---------|-----|
| BayesOpt LGBM | 0.078883 | 0.993383 |
| Optuna LGBM | 0.078848 | 0.993344 |
| PyTorch NN | 0.112204 | 0.991481 |
| All 140 features (ref) | 0.066115 | 0.995474 |

### Files:
```
src/classification/
├── config.py                     # Paths, RANDOM_STATE=42, TEST_SIZE=0.2
├── preprocessing.py              # load/split/scale, LogLoss+AUC metrics
├── feature_selection.py          # LGBM gain+split+SHAP → top15; also NN SHAP with --method both
├── models/
│   ├── lightgbm_tune.py          # Hyperparameter tuning + --final mode
│   └── pytorch_nn.py             # 15→256→128→64→1 GELU+LayerNorm, class-weighted BCE
├── evaluate.py                   # ROC, SHAP, metric comparison
├── submit.py                     # Generate CSV files
└── outputs/                      # Models, plots, submissions
    ├── submission/               # 8 CSV files (3 submissions × 2 varlists + extras)
    ├── top15_features.txt        # Selected features
    ├── lightgbm_bayesopt_model.pkl
    ├── lightgbm_optuna_model.pkl
    ├── pytorch_nn_best.pt
    ├── pytorch_scaler_best.pkl
    └── *.png, *.json
```

### Feature selection comparison (tree vs NN):
Only 6/15 overlap. Both LGBM and NN perform better on LGBM's top-15 than NN's own top-15.
→ Confirmed: use LGBM ranking for both models.

### To redo with correct flow:
```
cd InitialProject/src/classification
python models/lightgbm_tune.py              # tune on ALL 140
python feature_selection.py --method tree   # rank with tuned params
python models/lightgbm_tune.py --final      # train final on top-15
python models/pytorch_nn.py                 # train NN on top-15
python evaluate.py
python submit.py
```

---

## Regression — Status: In Progress (flow being corrected)

### Data cleaning:
- 10,325 rows have `p_Truth_Energy = -999` (sentinel for non-electron), all are `isElectron=0`
- Filtered out → 169,675 training rows remain
- Applied `log1p` transform to target → ~N(10.15, 1.05) — very well-behaved

### Key differences from classification:
- Target: `log1p(Energy)`, inverse for predictions
- LGBM: `LGBMRegressor` with `objective="regression"`, `metric="rmse"`
- NN: Linear output (no sigmoid), `HuberLoss(delta=1.0)` — robust to outliers
- Metrics: RMSE, MAE, R² (on original scale, not log)
- Feature cap: 20 (vs 15 for classification)

### Feature ranking (from first run with default params, RMSE/R² on original scale):
```
All 140: RMSE=14193  R²=0.8936
Top 20:   RMSE=15017  R²=0.8809  (only 1.3% R² drop)
```

Top 20 features (by avg rank): p_pt_track, p_eta, pX_ecore, pX_ptconecoreTrackPtrCorrection, p_numberOfTRTXenonHits, p_ptPU30, p_sigmad0, pX_E_Lr2_HiG, pX_deltaPhi2, pX_deltaPhi0, pX_maxEcell_z, pX_MultiLepton, pX_core57cellsEnergyCorrection, p_d0, p_dPOverP, p_numberOfSCTHits, pX_topoetcone40ptCorrection, p_ptcone40, pX_deltaPhiFromLastMeasurement, pX_neflowisolcoreConeEnergyCorrection

Note: Very different from classification top-15! These are energy/track-momentum oriented.

### Files:
```
src/regression/
├── config.py                     # Adapted for regression: TARGET_COL, SENTINEL_ENERGY, N_TOP_FEATURES=20
├── preprocessing.py              # Filters -999, log1p_transform, RMSE/MAE/R² metrics
├── feature_selection.py          # Reads tuned params from lightgbm_tuning_results.json
├── models/
│   ├── lightgbm_tune.py          # Tune on all 140, --final for top-20
│   └── pytorch_nn.py             # 20→256→128→64→1, HuberLoss, log target
├── evaluate.py                   # Residual plots, SHAP, metric comparison
├── submit.py                     # 40k predictions, inverse log
└── outputs/
    ├── top20_features.txt        # (from first run, needs regeneration with tuned params)
    └── pytorch_nn_best.pt        # (from first run on old feature set)
```

### NN results (first run, before flow correction):
RMSE=15015  MAE=4888  R²=0.8809  epochs=170

### To complete (correct flow):
```
cd InitialProject/src/regression
python models/lightgbm_tune.py              # tune on ALL 140 (BayesOpt + Optuna, ~15-20 min each)
python feature_selection.py                 # rank with tuned params, pick top 20
python models/lightgbm_tune.py --final      # train final LGBM on top-20
python models/pytorch_nn.py                 # train NN on top-20
python evaluate.py
python submit.py
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| LightGBM | Tree model (primary, dominates on tabular data) |
| PyTorch (MPS) | Neural network |
| SHAP | TreeExplainer for LGBM, GradientExplainer for NN |
| BayesianOptimization | LGBM hyperparameter search |
| Optuna | Alternative hyperparameter search |
| scikit-learn | Preprocessing, CV, metrics |
| matplotlib | All plots |

## Key Constants (config.py)

```python
RANDOM_STATE = 42
TEST_SIZE = 0.2
BATCH_SIZE = 512
NN_EPOCHS = 200, NN_PATIENCE = 30, NN_LR = 1e-3, NN_WD = 1e-4
LGBM_CV_FOLDS = 5
BAYESOPT_INIT = 10, BAYESOPT_ITER = 40
OPTUNA_TRIALS = 50
```

## NN Architecture (shared across tasks)

```
input → 256 → LayerNorm → GELU → Dropout(0.25)
      → 128 → LayerNorm → GELU → Dropout(0.25)
      → 64  → LayerNorm → GELU → Dropout(0.25)
      → 1
Optimizer: AdamW
Scheduler: CosineAnnealingWarmRestarts(T_0=20, T_mult=2)
```
