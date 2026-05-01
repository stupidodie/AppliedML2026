"""
DBSCAN Clustering for SDSS-Gaia Stellar Data
=============================================
- Feature selection: density-sensitive features (low correlation, local structure)
- Scaling: StandardScaler
- Parameter tuning: eps via k-distance plot, min_samples via grid search
- Noise handling: assign noise points to nearest non-noise cluster
- Output: integer cluster labels (0..n)
"""

import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

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
SOLUTION_NAME = "DBSCAN"

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
# 2. Feature selection
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[2/5] Feature selection (max {MAX_FEATURES}) ...")
print("=" * 60)

# DBSCAN works best with features that capture local density structure.
# Strategy: use features with high local variance, avoid redundant ones.

# Drop J, K (too correlated with H), keep H
# Pick diverse features from different physical groups
corr = df.corr().abs()

# Identify candidate pool: drop one from each highly correlated pair
drop = {"J", "K"}  # drop J, K (correlate with H)
candidate_pool = [f for f in feature_names if f not in drop]

# Build combos ensuring mix of: photometry + kinematic + abundances
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
if len(all_combos) > 200:
    all_combos = [all_combos[i] for i in
                  np.random.choice(len(all_combos), 200, replace=False)]

print(f"  Trying {len(all_combos)} candidate combinations ...")

# Score combos using DBSCAN with fixed reasonable params
scaler_fs = StandardScaler()
X_scaled_full = scaler_fs.fit_transform(df)

combo_scores = []
for combo in all_combos:
    combo_indices = [feature_names.index(c) for c in combo]
    X_sub = X_scaled_full[:, combo_indices]
    # Quick DBSCAN with default-ish params
    try:
        db = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1)
        labels = db.fit_predict(X_sub)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.sum(labels == -1) / len(labels)
        # We want reasonable cluster count and low noise
        if K_MIN <= n_clusters <= K_MAX and noise_ratio < 0.3:
            # Score: positive = good
            non_noise = labels != -1
            if np.sum(non_noise) > 1 and len(set(labels[non_noise])) > 1:
                sil = silhouette_score(X_sub[non_noise], labels[non_noise])
                score = sil - noise_ratio * 0.5  # penalize noise
                combo_scores.append((score, combo, n_clusters, noise_ratio))
    except Exception:
        pass

combo_scores.sort(key=lambda x: x[0], reverse=True)

print(f"\n  Top 10 candidate combos:")
for i, (s, c, nc, nr) in enumerate(combo_scores[:10]):
    print(f"    {i+1:2d}. Score={s:.4f} K={nc:2d} Noise={nr:.2%} | {', '.join(c)}")

if combo_scores:
    best_score, selected_features, _, _ = combo_scores[0]
else:
    # Fallback: use a diverse default set
    selected_features = ["H", "FE_H", "MG_FE", "SI_FE", "E", "Energy"]
    print(f"  WARNING: No combos passed filter, using default: {selected_features}")

print(f"\n  Selected features: {selected_features}")

# ---------------------------------------------------------------------------
# 3. DBSCAN parameter tuning
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[3/5] Tuning DBSCAN parameters (eps, min_samples) ...")
print("=" * 60)

X_selected = df[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Determine eps via k-distance plot (k = min_samples)
# Use k=2*MAX_FEATURES as recommended (2*dim)
k_neighbors = 2 * MAX_FEATURES
nn = NearestNeighbors(n_neighbors=k_neighbors + 1, n_jobs=-1)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_distances = np.sort(distances[:, k_neighbors])

# Find elbow in k-distance curve
# Use the point of maximum curvature
grad = np.gradient(k_distances)
grad2 = np.gradient(grad)
elbow_idx = np.argmax(grad2[:len(grad2) // 2])  # look in first half
eps_auto = k_distances[elbow_idx]

print(f"  Auto-detected eps (k={k_neighbors}-dist elbow): {eps_auto:.4f}")

# Grid search around auto eps
eps_values = np.linspace(eps_auto * 0.5, eps_auto * 2.0, 15)
min_samples_values = [3, 5, 10, 15, 20, 30]

best_result = None
best_score_db = -1

for eps in eps_values:
    for ms in min_samples_values:
        try:
            db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)

            if K_MIN <= n_clusters <= K_MAX and noise_ratio < 0.5:
                non_noise = labels != -1
                if np.sum(non_noise) > 1 and len(set(labels[non_noise])) > 1:
                    try:
                        sil = silhouette_score(X_scaled[non_noise], labels[non_noise])
                        score = sil * (1 - noise_ratio)
                        if score > best_score_db:
                            best_score_db = score
                            best_result = (eps, ms, labels.copy(), n_clusters, noise_ratio, sil)
                    except Exception:
                        pass
        except Exception:
            pass

if best_result is None:
    # Fallback: default params
    eps_best, ms_best = 0.5, 10
    db = DBSCAN(eps=eps_best, min_samples=ms_best, n_jobs=-1)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.sum(labels == -1) / len(labels)
    print(f"  WARNING: Grid search found no valid result, using defaults")
else:
    eps_best, ms_best, labels, n_clusters, noise_ratio, sil = best_result
    print(f"  Best params: eps={eps_best:.4f}, min_samples={ms_best}")
    print(f"  Clusters: {n_clusters}, Noise: {noise_ratio:.2%}, Silhouette: {sil:.4f}")

# ---------------------------------------------------------------------------
# 4. Handle noise points + final labels
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[4/5] Handling noise points & finalizing ...")
print("=" * 60)

t0 = time.time()

# Re-fit with best params
db_final = DBSCAN(eps=eps_best, min_samples=ms_best, n_jobs=-1)
raw_labels = db_final.fit_predict(X_scaled)

# Assign noise points (-1) to nearest non-noise cluster via centroid distance
noise_mask = raw_labels == -1
valid_labels = raw_labels[~noise_mask]
unique_labels = np.unique(valid_labels)

if np.sum(noise_mask) > 0 and len(unique_labels) > 0:
    # Compute centroids of non-noise clusters
    centroids = {}
    for lbl in unique_labels:
        centroids[lbl] = X_scaled[raw_labels == lbl].mean(axis=0)

    # Assign each noise point to nearest centroid
    noise_indices = np.where(noise_mask)[0]
    for idx in noise_indices:
        dists = {lbl: np.linalg.norm(X_scaled[idx] - cent) for lbl, cent in centroids.items()}
        raw_labels[idx] = min(dists, key=dists.get)

    noise_reassigned = np.sum(noise_mask)
    print(f"  Re-assigned {noise_reassigned} noise points to nearest clusters")

# Relabel by cluster size (descending)
final_labels_set = np.unique(raw_labels)
cluster_sizes = {lbl: np.sum(raw_labels == lbl) for lbl in final_labels_set}
sorted_clusters = sorted(cluster_sizes.keys(), key=lambda x: -cluster_sizes[x])
remap = {old: new for new, old in enumerate(sorted_clusters)}
final_labels = np.array([remap[l] for l in raw_labels])

final_n_clusters = len(sorted_clusters)
train_time = time.time() - t0

print(f"  Final clusters: {final_n_clusters}")
print(f"  Training time: {train_time:.1f}s")
print(f"  Cluster sizes: {np.bincount(final_labels)}")

# Metrics
final_sil = silhouette_score(X_scaled, final_labels)
final_ch = calinski_harabasz_score(X_scaled, final_labels)
final_db = davies_bouldin_score(X_scaled, final_labels)
print(f"  Final Silhouette:  {final_sil:.4f}")
print(f"  Final Calinski-H:  {final_ch:.2f}")
print(f"  Final Davies-B:    {final_db:.4f}")

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
    for i, label in enumerate(final_labels):
        f.write(f"{i},{label}\n")
print(f"  Predictions saved: {pred_file}")

with open(var_file, "w") as f:
    for feat in selected_features:
        f.write(f"{feat},\n")
print(f"  Variable list saved: {var_file}")

with open(summary_file, "w") as f:
    f.write(f"DBSCAN Clustering - Guanran Tai\n")
    f.write(f"===============================\n\n")
    f.write(f"Objective: Unsupervised clustering of SDSS-Gaia stars\n")
    f.write(f"Algorithm: DBSCAN (scikit-learn)\n\n")
    f.write(f"Selected features ({len(selected_features)}):\n")
    for i, feat in enumerate(selected_features):
        f.write(f"  {i+1}. {feat}\n")
    f.write(f"\nParameters:\n")
    f.write(f"  eps: {eps_best:.4f}\n")
    f.write(f"  min_samples: {ms_best}\n")
    f.write(f"  k-neighbors for eps detection: {k_neighbors}\n")
    f.write(f"\nNumber of clusters: {final_n_clusters}\n")
    f.write(f"Noise points reassigned to nearest clusters\n")
    f.write(f"\nInternal metrics:\n")
    f.write(f"  Silhouette Score:  {final_sil:.4f}\n")
    f.write(f"  Calinski-Harabasz: {final_ch:.2f}\n")
    f.write(f"  Davies-Bouldin:    {final_db:.4f}\n")
    f.write(f"\nTraining time: {train_time:.1f}s\n")
    f.write(f"Random seed: {RANDOM_SEED}\n")
print(f"  Summary saved: {summary_file}")

print("\n" + "=" * 60)
print("DBSCAN Clustering DONE")
print("=" * 60)
