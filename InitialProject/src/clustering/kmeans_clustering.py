"""
K-Means Clustering for SDSS-Gaia Stellar Data
==============================================
- Feature selection: correlation-based grouping + silhouette score on candidate combos
- Scaling: StandardScaler
- Optimal K: elbow method + silhouette score (K in [4, 40])
- Output: integer cluster labels (0..n)
"""

import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent  # InitialProject/
DATA_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "output" / "clustering"

DATA_FILE = DATA_DIR / "SDSS-Gaia_5950stars.csv"

MAX_FEATURES = 6
K_MIN, K_MAX = 4, 40
RANDOM_SEED = 42

STUDENT_NAME = "GuanranTai"
SOLUTION_NAME = "KMeans"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

print("=" * 60)
print("[1/5] Loading data ...")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
df = df.drop(columns=["Unnamed: 0"])
n_samples = df.shape[0]
feature_names = list(df.columns)

print(f"  Shape: {df.shape}")
print(f"  Features ({len(feature_names)}): {feature_names}")

# ---------------------------------------------------------------------------
# 2. Feature selection
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[2/5] Feature selection (max {MAX_FEATURES}) ...")
print("=" * 60)

# --- Correlation grouping ---
corr = df.corr().abs()
# Build groups of highly correlated features (|r| > 0.85)
threshold = 0.85
correlated_groups = []
seen = set()

for i, f1 in enumerate(feature_names):
    if f1 in seen:
        continue
    group = {f1}
    for f2 in feature_names[i + 1:]:
        if f2 not in seen and corr.loc[f1, f2] > threshold:
            group.add(f2)
    if len(group) > 1:
        correlated_groups.append(sorted(group))
        seen.update(group)

print(f"  Highly correlated groups (|r| > {threshold}):")
for g in correlated_groups:
    print(f"    {g}")

# Pick representatives: J/H/K → pick H; other groups pick first
rep_map = {}
for g in correlated_groups:
    if "H" in g:
        rep_map["H"] = g
    else:
        rep_map[g[0]] = g

# Create a reduced feature pool (keep one per correlated group + independent features)
correlated_pool = list(rep_map.keys())
independent_pool = [f for f in feature_names if f not in seen]
reduced_pool = correlated_pool + independent_pool
print(f"\n  Reduced pool ({len(reduced_pool)}): {reduced_pool}")

# Score candidate 6-var combos using quick K-Means silhouette
# Limit combo count for speed: from the reduced pool, try diverse sets
print(f"\n  Scoring candidate feature combinations via silhouette ...")

scaler_fs = StandardScaler()
X_scaled_full = scaler_fs.fit_transform(df)

# Try combos that ensure diversity: pick from photometry, abundances, kinematics
phot = ["H"]
kin = ["E", "Energy", "Lz"]
# Abundance features (not in correlated groups' representatives for J/H/K)
abu_rep = [f for f in reduced_pool if f not in phot and f not in kin]
abu_all = [f for f in feature_names if f not in phot and f not in kin and f not in {"J", "K"}]

best_score = -1
best_combo = None
best_combo_k = None
combo_scores = []

# Strategy: fix 1 photometry + 1 kinematic + 4 abundances (or vary)
all_combos_to_try = []

# Try several strategies for diverse combinations
# Strategy A: 1 photo + 2 kin + 3 abu
for kin_pair in combinations(kin, 2):
    for abu_trip in combinations(abu_rep, 3):
        combo = list(phot) + list(kin_pair) + list(abu_trip)
        if len(combo) == MAX_FEATURES:
            all_combos_to_try.append(combo)

# Strategy B: 1 photo + 1 kin + 4 abu (from full abundance list)
for k in kin:
    for abu_quad in combinations(abu_all, 4):
        combo = list(phot) + [k] + list(abu_quad)
        if len(combo) == MAX_FEATURES:
            all_combos_to_try.append(combo)

# Deduplicate
all_combos_to_try = [list(x) for x in set(tuple(sorted(c)) for c in all_combos_to_try)]

# Cap combos for speed
np.random.seed(RANDOM_SEED)
if len(all_combos_to_try) > 200:
    all_combos_to_try = [all_combos_to_try[i] for i in
                         np.random.choice(len(all_combos_to_try), 200, replace=False)]

print(f"  Trying {len(all_combos_to_try)} candidate combinations ...")

for combo in all_combos_to_try:
    combo_indices = [feature_names.index(c) for c in combo]
    X_sub = X_scaled_full[:, combo_indices]
    # Quick K-Means with k=8 (common reasonable starting point)
    km = KMeans(n_clusters=8, random_state=RANDOM_SEED, n_init=10, max_iter=300)
    labels = km.fit_predict(X_sub)
    sil = silhouette_score(X_sub, labels)
    combo_scores.append((sil, combo, 8))

# Also evaluate each combo at different K values for top candidates
top_n = min(20, len(combo_scores))
combo_scores.sort(key=lambda x: x[0], reverse=True)
top_combos = combo_scores[:top_n]

print(f"\n  Top {top_n} candidate combos (silhouette @ K=8):")
for i, (s, c, k) in enumerate(top_combos):
    print(f"    {i+1:2d}. Sil={s:.4f}  |  {', '.join(c)}")

# For top 5 combos, test multiple K values to find best overall
top_for_detailed = top_combos[:5]
best_overall_score = -1
best_overall_combo = None
best_overall_k = None

for _, combo, _ in top_for_detailed:
    combo_indices = [feature_names.index(c) for c in combo]
    X_sub = X_scaled_full[:, combo_indices]
    for k_test in range(5, 31, 5):
        km = KMeans(n_clusters=k_test, random_state=RANDOM_SEED, n_init=10, max_iter=300)
        labels = km.fit_predict(X_sub)
        sil = silhouette_score(X_sub, labels)
        if sil > best_overall_score:
            best_overall_score = sil
            best_overall_combo = combo
            best_overall_k = k_test

selected_features = best_overall_combo
print(f"\n  Selected {len(selected_features)} features: {selected_features}")
print(f"  Best silhouette in search: {best_overall_score:.4f} @ K={best_overall_k}")

# ---------------------------------------------------------------------------
# 3. Determine optimal K
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"[3/5] Determining optimal K ({K_MIN}-{K_MAX}) ...")
print("=" * 60)

X_selected = df[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

inertias = []
silhouettes = []
ch_scores = []
db_scores = []

k_range = range(K_MIN, K_MAX + 1)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, max_iter=500)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# Find elbow using kneedle-like approach: max curvature in inertia curve
inertias_arr = np.array(inertias)
# Second derivative approximation
inertia_deltas = np.diff(inertias_arr)
inertia_deltas2 = np.diff(inertia_deltas)
elbow_idx = np.argmax(inertia_deltas2) + 2  # +2 for diff offsets
elbow_k = K_MIN + elbow_idx

# Best silhouette
sil_arr = np.array(silhouettes)
sil_best_k = K_MIN + np.argmax(sil_arr)
sil_best_val = sil_arr.max()

# Best CH
ch_arr = np.array(ch_scores)
ch_best_k = K_MIN + np.argmax(ch_arr)

# Best DB (lower is better)
db_arr = np.array(db_scores)
db_best_k = K_MIN + np.argmin(db_arr)

print(f"\n  Elbow K:           {elbow_k}")
print(f"  Best Silhouette:   K={sil_best_k} ({sil_best_val:.4f})")
print(f"  Best Calinski-H:   K={ch_best_k} ({ch_arr.max():.2f})")
print(f"  Best Davies-B:     K={db_best_k} ({db_arr.min():.4f})")

# Select K: prefer silhouette, but validate with elbow
if abs(sil_best_k - elbow_k) <= 10:
    optimal_k = sil_best_k
else:
    # Average of the two best methods
    optimal_k = int(round((sil_best_k + ch_best_k + elbow_k) / 3))
    optimal_k = max(K_MIN, min(K_MAX, optimal_k))

print(f"\n  Selected K: {optimal_k}")

# ---------------------------------------------------------------------------
# 4. Final clustering
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[4/5] Running final K-Means ...")
print("=" * 60)

t0 = time.time()
final_km = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=20, max_iter=1000)
final_labels = final_km.fit_predict(X_scaled)
train_time = time.time() - t0

# Relabel clusters by size (descending) for consistency
cluster_sizes = np.bincount(final_labels)
sorted_clusters = np.argsort(-cluster_sizes)
remap = {old: new for new, old in enumerate(sorted_clusters)}
final_labels_remapped = np.array([remap[l] for l in final_labels])

print(f"  Clusters: {optimal_k}")
print(f"  Training time: {train_time:.1f}s")
print(f"  Cluster sizes: {np.bincount(final_labels_remapped)}")

# Final metrics
final_sil = silhouette_score(X_scaled, final_labels_remapped)
final_ch = calinski_harabasz_score(X_scaled, final_labels_remapped)
final_db = davies_bouldin_score(X_scaled, final_labels_remapped)
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

# Predictions: index, cluster_label
with open(pred_file, "w") as f:
    for i, label in enumerate(final_labels_remapped):
        f.write(f"{i},{label}\n")
print(f"  Predictions saved: {pred_file}")

# Variable list
with open(var_file, "w") as f:
    for feat in selected_features:
        f.write(f"{feat},\n")
print(f"  Variable list saved: {var_file}")

# Summary
with open(summary_file, "w") as f:
    f.write(f"K-Means Clustering - Guanran Tai\n")
    f.write(f"================================\n\n")
    f.write(f"Objective: Unsupervised clustering of SDSS-Gaia stars\n")
    f.write(f"Algorithm: K-Means (scikit-learn)\n\n")
    f.write(f"Selected features ({len(selected_features)}):\n")
    for i, feat in enumerate(selected_features):
        f.write(f"  {i+1}. {feat}\n")
    f.write(f"\nNumber of clusters (K): {optimal_k}\n")
    f.write(f"\nInternal metrics:\n")
    f.write(f"  Silhouette Score:  {final_sil:.4f}\n")
    f.write(f"  Calinski-Harabasz: {final_ch:.2f}\n")
    f.write(f"  Davies-Bouldin:    {final_db:.4f}\n")
    f.write(f"\nTraining time: {train_time:.1f}s\n")
    f.write(f"Random seed: {RANDOM_SEED}\n")
print(f"  Summary saved: {summary_file}")

print("\n" + "=" * 60)
print("K-Means Clustering DONE")
print("=" * 60)
