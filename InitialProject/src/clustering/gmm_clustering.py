"""
Gaussian Mixture Model Clustering for SDSS-Gaia Stellar Data
=============================================================
- Feature selection: different from K-Means (correlation + BIC-based scoring)
- Scaling: StandardScaler
- Optimal components: BIC (Bayesian Information Criterion)
- Covariance type comparison: full, tied, diag, spherical
- Output: integer cluster labels (0..n)
"""

import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "output" / "clustering"

DATA_FILE = DATA_DIR / "SDSS-Gaia_5950stars.csv"

MAX_FEATURES = 6
K_MIN, K_MAX = 4, 40
RANDOM_SEED = 42

STUDENT_NAME = "GuanranTai"
SOLUTION_NAME = "GMM"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

print("=" * 60)
print("[1/5] Loading data ...")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
df = df.drop(columns=["Unnamed: 0"])
feature_names = list(df.columns)
n_samples = df.shape[0]

print(f"  Shape: {df.shape}")
print(f"  Features: {feature_names}")

# ---------------------------------------------------------------------------
# 2. Feature selection (different strategy from K-Means)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[2/5] Feature selection (max {MAX_FEATURES}) ...")
print("=" * 60)

# GMM works well with features that follow roughly Gaussian distributions.
# Prefer features with lower skewness/kurtosis.

scaler_fs = StandardScaler()
X_scaled_full = scaler_fs.fit_transform(df)

# Calculate skewness for each feature
skewness = df.skew().abs().sort_values()
print(f"  Feature skewness (lower = more Gaussian):")
for feat, val in skewness.items():
    print(f"    {feat:<10s}: {val:.4f}")

# Build a candidate pool excluding the most skewed features and J/K (keep H from JHK group)
# J, H, K highly correlated — keep only H
# Exclude highly correlated: J, K dropped (corr > 0.99 with H)

corr = df.corr().abs()
# From J/H/K group, keep H
exclude_from_corr = {"J", "K"}
# Also check other high correlations
high_corr_pairs = set()
for i, f1 in enumerate(feature_names):
    for f2 in feature_names[i + 1:]:
        if corr.loc[f1, f2] > 0.8 and f1 not in exclude_from_corr and f2 not in exclude_from_corr:
            high_corr_pairs.add(frozenset({f1, f2}))

print(f"\n  High correlation pairs (|r| > 0.8):")
for pair in high_corr_pairs:
    p = list(pair)
    print(f"    {p[0]} <-> {p[1]}: {corr.loc[p[0], p[1]]:.4f}")

# Candidate pool: drop J, K (correlated with H), keep H
candidate_pool = [f for f in feature_names if f not in exclude_from_corr]
print(f"\n  Candidate pool ({len(candidate_pool)}): {candidate_pool}")

# Try combinations and score with GMM BIC at K=8
print(f"\n  Scoring candidate feature combinations via GMM BIC ...")

# Ensure diversity: 1 photometric + mix of kinematics + abundances
phot_candidates = ["H"]
kin_candidates = ["E", "Energy", "Lz"]
abu_candidates = [f for f in candidate_pool if f not in phot_candidates and f not in kin_candidates]

all_combos = []

# Strategy: 1 photo + 1 kin + 4 abu
for kin in kin_candidates:
    for abu_quad in combinations(abu_candidates, 4):
        combo = list(phot_candidates) + [kin] + list(abu_quad)
        all_combos.append(combo)

# Strategy: 1 photo + 2 kin + 3 abu
for kin_pair in combinations(kin_candidates, 2):
    for abu_trip in combinations(abu_candidates, 3):
        combo = list(phot_candidates) + list(kin_pair) + list(abu_trip)
        all_combos.append(combo)

# Deduplicate
all_combos = [list(x) for x in set(tuple(sorted(c)) for c in all_combos)]

# Cap for speed
np.random.seed(RANDOM_SEED)
if len(all_combos) > 150:
    all_combos = [all_combos[i] for i in
                  np.random.choice(len(all_combos), 150, replace=False)]

print(f"  Trying {len(all_combos)} candidate combinations ...")

combo_scores = []
for combo in all_combos:
    combo_indices = [feature_names.index(c) for c in combo]
    X_sub = X_scaled_full[:, combo_indices]
    try:
        gmm = GaussianMixture(n_components=8, covariance_type="full",
                              random_state=RANDOM_SEED, max_iter=200)
        gmm.fit(X_sub)
        bic = gmm.bic(X_sub)
        combo_scores.append((bic, combo))
    except Exception:
        continue

combo_scores.sort(key=lambda x: x[0])  # lower BIC is better

print(f"\n  Top 10 candidate combos (BIC @ K=8, full cov):")
for i, (bic, c) in enumerate(combo_scores[:10]):
    print(f"    {i+1:2d}. BIC={bic:.1f}  |  {', '.join(c)}")

# Select best combo
best_bic, selected_features = combo_scores[0]
print(f"\n  Selected features: {selected_features}")

# ---------------------------------------------------------------------------
# 3. Model selection: covariance type + number of components
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[3/5] Model selection (K={K_MIN}-{K_MAX}, covariance types) ...")
print("=" * 60)

X_selected = df[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

cov_types = ["full", "tied", "diag", "spherical"]
best_overall_bic = np.inf
best_overall_k = None
best_overall_cov = None
best_overall_model = None

for cov_type in cov_types:
    bics = []
    aics = []
    for k in range(K_MIN, K_MAX + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type=cov_type,
                                  random_state=RANDOM_SEED, max_iter=200,
                                  n_init=3, reg_covar=1e-6)
            gmm.fit(X_scaled)
            bics.append(gmm.bic(X_scaled))
            aics.append(gmm.aic(X_scaled))
        except Exception:
            bics.append(np.inf)
            aics.append(np.inf)

    # Find best K for this covariance type
    valid_bics = [(b, i) for i, b in enumerate(bics) if b < np.inf]
    if valid_bics:
        best_b, best_i = min(valid_bics)
        best_k = K_MIN + best_i
        print(f"  {cov_type:<10s}: best K={best_k}, BIC={best_b:.1f}, AIC={aics[best_i]:.1f}")
        if best_b < best_overall_bic:
            best_overall_bic = best_b
            best_overall_k = best_k
            best_overall_cov = cov_type

print(f"\n  Best model: K={best_overall_k}, cov_type={best_overall_cov}, BIC={best_overall_bic:.1f}")

# ---------------------------------------------------------------------------
# 4. Final GMM training
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[4/5] Training final GMM ...")
print("=" * 60)

t0 = time.time()
final_gmm = GaussianMixture(n_components=best_overall_k,
                            covariance_type=best_overall_cov,
                            random_state=RANDOM_SEED,
                            max_iter=500,
                            n_init=10,
                            reg_covar=1e-6,
                            init_params="kmeans")
final_gmm.fit(X_scaled)
final_labels = final_gmm.predict(X_scaled)
train_time = time.time() - t0

# Remap labels by cluster size (descending)
cluster_sizes = np.bincount(final_labels)
sorted_clusters = np.argsort(-cluster_sizes)
remap = {old: new for new, old in enumerate(sorted_clusters)}
final_labels_remapped = np.array([remap[l] for l in final_labels])

n_clusters = len(np.unique(final_labels_remapped))
print(f"  Clusters: {n_clusters}")
print(f"  Training time: {train_time:.1f}s")
print(f"  Cluster sizes: {np.bincount(final_labels_remapped)}")

# Metrics
final_sil = silhouette_score(X_scaled, final_labels_remapped)
final_ch = calinski_harabasz_score(X_scaled, final_labels_remapped)
final_db = davies_bouldin_score(X_scaled, final_labels_remapped)
print(f"  Final Silhouette:  {final_sil:.4f}")
print(f"  Final Calinski-H:  {final_ch:.2f}")
print(f"  Final Davies-B:    {final_db:.4f}")
print(f"  Final BIC:         {final_gmm.bic(X_scaled):.1f}")

# ---------------------------------------------------------------------------
# 5. Save outputs
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[5/5] Saving outputs ...")
print("=" * 60)

pred_file = OUTPUT_DIR / f"Clustering_{STUDENT_NAME}_{SOLUTION_NAME}.csv"
var_file = OUTPUT_DIR / f"Clustering_{STUDENT_NAME}_{SOLUTION_NAME}_VariableList.csv"
summary_file = OUTPUT_DIR / f"Clustering_{STUDENT_NAME}_{SOLUTION_NAME}_summary.txt"

with open(pred_file, "w") as f:
    for i, label in enumerate(final_labels_remapped):
        f.write(f"{i},{label}\n")
print(f"  Predictions saved: {pred_file}")

with open(var_file, "w") as f:
    for feat in selected_features:
        f.write(f"{feat},\n")
print(f"  Variable list saved: {var_file}")

with open(summary_file, "w") as f:
    f.write(f"GMM Clustering - Guanran Tai\n")
    f.write(f"============================\n\n")
    f.write(f"Objective: Unsupervised clustering of SDSS-Gaia stars\n")
    f.write(f"Algorithm: Gaussian Mixture Model (scikit-learn)\n\n")
    f.write(f"Selected features ({len(selected_features)}):\n")
    for i, feat in enumerate(selected_features):
        f.write(f"  {i+1}. {feat}\n")
    f.write(f"\nNumber of components: {n_clusters}\n")
    f.write(f"Covariance type: {best_overall_cov}\n")
    f.write(f"\nInternal metrics:\n")
    f.write(f"  Silhouette Score:  {final_sil:.4f}\n")
    f.write(f"  Calinski-Harabasz: {final_ch:.2f}\n")
    f.write(f"  Davies-Bouldin:    {final_db:.4f}\n")
    f.write(f"  BIC:               {final_gmm.bic(X_scaled):.1f}\n")
    f.write(f"  AIC:               {final_gmm.aic(X_scaled):.1f}\n")
    f.write(f"\nTraining time: {train_time:.1f}s\n")
    f.write(f"Random seed: {RANDOM_SEED}\n")
print(f"  Summary saved: {summary_file}")

print("\n" + "=" * 60)
print("GMM Clustering DONE")
print("=" * 60)
