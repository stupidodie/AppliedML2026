"""
Autoencoder + K-Means Clustering for SDSS-Gaia Stellar Data
=============================================================
- Feature selection: same as K-Means for comparison, or correlation-based
- Scaling: StandardScaler
- Autoencoder: PyTorch neural network, 6 -> 32 -> 16 -> 2 -> 16 -> 32 -> 6
- K-Means on 2D latent space
- Output: integer cluster labels (0..n)
"""

import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
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
LATENT_DIM = 2
K_MIN, K_MAX = 4, 40
RANDOM_SEED = 42

# Autoencoder hyperparams
HIDDEN_DIMS = [32, 16]
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

STUDENT_NAME = "GuanranTai"
SOLUTION_NAME = "AutoencoderKMeans"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"  Device: {DEVICE}")

# Set seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

print("=" * 60)
print("[1/6] Loading data ...")
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
print(f"[2/6] Feature selection (max {MAX_FEATURES}) ...")
print("=" * 60)

# Use correlation-based grouping + silhouette scoring
corr = df.corr().abs()

# Drop J, K (correlate ~1.0 with H)
drop = {"J", "K"}
candidate_pool = [f for f in feature_names if f not in drop]

phot_candidates = ["H"]
kin_candidates = ["E", "Energy", "Lz"]
abu_candidates = [f for f in candidate_pool if f not in phot_candidates and f not in kin_candidates]

all_combos = []

# Strategy: 1 photo + 2 kin + 3 abu
for kin_pair in combinations(kin_candidates, 2):
    for abu_trip in combinations(abu_candidates, 3):
        combo = list(phot_candidates) + list(kin_pair) + list(abu_trip)
        all_combos.append(combo)

# Strategy: 1 photo + 1 kin + 4 abu
for kin in kin_candidates:
    for abu_quad in combinations(abu_candidates, 4):
        combo = list(phot_candidates) + [kin] + list(abu_quad)
        all_combos.append(combo)

all_combos = [list(x) for x in set(tuple(sorted(c)) for c in all_combos)]

# Cap
np.random.seed(RANDOM_SEED)
if len(all_combos) > 150:
    all_combos = [all_combos[i] for i in
                  np.random.choice(len(all_combos), 150, replace=False)]

print(f"  Trying {len(all_combos)} candidate combinations ...")

# Score using quick K-Means silhouette on scaled data
scaler_fs = StandardScaler()
X_scaled_full = scaler_fs.fit_transform(df)

combo_scores = []
for combo in all_combos:
    combo_indices = [feature_names.index(c) for c in combo]
    X_sub = X_scaled_full[:, combo_indices]
    km = KMeans(n_clusters=8, random_state=RANDOM_SEED, n_init=10, max_iter=300)
    labels = km.fit_predict(X_sub)
    sil = silhouette_score(X_sub, labels)
    combo_scores.append((sil, combo))

combo_scores.sort(key=lambda x: x[0], reverse=True)

print(f"\n  Top 10 candidate combos:")
for i, (s, c) in enumerate(combo_scores[:10]):
    print(f"    {i+1:2d}. Sil={s:.4f}  |  {', '.join(c)}")

best_sil, selected_features = combo_scores[0]
print(f"\n  Selected features: {selected_features}")

# ---------------------------------------------------------------------------
# 3. Prepare data
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[3/6] Preparing data & building autoencoder ...")
print("=" * 60)

X_selected = df[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
dataset = TensorDataset(X_tensor, X_tensor)  # input = target for autoencoder
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

input_dim = MAX_FEATURES


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        # Encoder
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Decoder
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


model = Autoencoder(input_dim, HIDDEN_DIMS, LATENT_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Architecture: {input_dim} -> {HIDDEN_DIMS} -> {LATENT_DIM} -> {HIDDEN_DIMS[::-1]} -> {input_dim}")
print(f"  Parameters: {n_params}")

# ---------------------------------------------------------------------------
# 4. Train autoencoder
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[4/6] Training autoencoder ...")
print("=" * 60)

t0 = time.time()

model.train()
best_loss = float("inf")
patience = 30
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for batch_x, _ in dataloader:
        optimizer.zero_grad()
        reconstructed = model(batch_x)
        loss = criterion(reconstructed, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)

    epoch_loss /= n_samples

    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  Loss: {epoch_loss:.6f}")

    # Early stopping
    if epoch_loss < best_loss - 1e-6:
        best_loss = epoch_loss
        patience_counter = 0
        # Save best model state
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"  Early stopping at epoch {epoch}")
        model.load_state_dict(best_state)
        break

train_time = time.time() - t0
print(f"  Final reconstruction loss: {best_loss:.6f}")
print(f"  Training time: {train_time:.1f}s")

# ---------------------------------------------------------------------------
# 5. Encode + K-Means on latent space
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[5/6] K-Means on latent space ...")
print("=" * 60)

model.eval()
with torch.no_grad():
    latent = model.encode(X_tensor).cpu().numpy()

# Determine optimal K on latent space
print(f"  Searching K in [{K_MIN}, {K_MAX}] on {LATENT_DIM}D latent space ...")
inertias = []
silhouettes = []
ch_scores = []

for k in range(K_MIN, K_MAX + 1):
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, max_iter=500)
    labels = km.fit_predict(latent)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(latent, labels))
    ch_scores.append(calinski_harabasz_score(latent, labels))

# Select K: best silhouette
sil_arr = np.array(silhouettes)
best_sil_idx = np.argmax(sil_arr)
best_k = K_MIN + best_sil_idx
best_sil_val = sil_arr.max()

# Also check elbow
inertias_arr = np.array(inertias)
inertia_deltas2 = np.diff(np.diff(inertias_arr))
if len(inertia_deltas2) > 0:
    elbow_idx = np.argmax(inertia_deltas2) + 2
    elbow_k = K_MIN + elbow_idx
else:
    elbow_k = best_k

ch_arr = np.array(ch_scores)
ch_best_k = K_MIN + np.argmax(ch_arr)

print(f"  Elbow K:       {elbow_k}")
print(f"  Best Silhouette: K={best_k} ({best_sil_val:.4f})")
print(f"  Best CH:       K={ch_best_k} ({ch_arr.max():.2f})")

# Use silhouette-priority selection
optimal_k = best_k
print(f"  Selected K: {optimal_k}")

# Final clustering on latent space
final_km = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=20, max_iter=1000)
final_labels = final_km.fit_predict(latent)

# Remap by cluster size
cluster_sizes = np.bincount(final_labels)
sorted_clusters = np.argsort(-cluster_sizes)
remap = {old: new for new, old in enumerate(sorted_clusters)}
final_labels_remapped = np.array([remap[l] for l in final_labels])

print(f"  Clusters: {optimal_k}")
print(f"  Cluster sizes: {np.bincount(final_labels_remapped)}")

# Evaluate on original scaled space (for fair comparison)
final_sil = silhouette_score(X_scaled, final_labels_remapped)
final_ch = calinski_harabasz_score(X_scaled, final_labels_remapped)
final_db = davies_bouldin_score(X_scaled, final_labels_remapped)
print(f"  Final Silhouette (original space): {final_sil:.4f}")
print(f"  Final Calinski-H:  {final_ch:.2f}")
print(f"  Final Davies-B:    {final_db:.4f}")

# ---------------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("[6/6] Saving outputs ...")
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
    f.write(f"Autoencoder + K-Means Clustering - Guanran Tai\n")
    f.write(f"===============================================\n\n")
    f.write(f"Objective: Unsupervised clustering of SDSS-Gaia stars\n")
    f.write(f"Algorithm: Autoencoder (PyTorch NN) + K-Means on latent space\n\n")
    f.write(f"Autoencoder architecture:\n")
    f.write(f"  Input: {input_dim} features\n")
    f.write(f"  Encoder: {input_dim} -> {HIDDEN_DIMS} -> {LATENT_DIM}\n")
    f.write(f"  Decoder: {LATENT_DIM} -> {HIDDEN_DIMS[::-1]} -> {input_dim}\n")
    f.write(f"  Total parameters: {n_params}\n")
    f.write(f"  Activation: ReLU, BatchNorm\n")
    f.write(f"\nTraining:\n")
    f.write(f"  Loss: MSE\n")
    f.write(f"  Optimizer: Adam (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})\n")
    f.write(f"  Epochs: {EPOCHS} (early stopping patience={patience})\n")
    f.write(f"  Batch size: {BATCH_SIZE}\n")
    f.write(f"  Final reconstruction loss: {best_loss:.6f}\n")
    f.write(f"  Training time: {train_time:.1f}s\n")
    f.write(f"  Device: {DEVICE}\n")
    f.write(f"\nSelected features ({len(selected_features)}):\n")
    for i, feat in enumerate(selected_features):
        f.write(f"  {i+1}. {feat}\n")
    f.write(f"\nK-Means on {LATENT_DIM}D latent space:\n")
    f.write(f"  Number of clusters: {optimal_k}\n")
    f.write(f"\nInternal metrics (original feature space):\n")
    f.write(f"  Silhouette Score:  {final_sil:.4f}\n")
    f.write(f"  Calinski-Harabasz: {final_ch:.2f}\n")
    f.write(f"  Davies-Bouldin:    {final_db:.4f}\n")
    f.write(f"\nRandom seed: {RANDOM_SEED}\n")
print(f"  Summary saved: {summary_file}")

print("\n" + "=" * 60)
print("Autoencoder + K-Means Clustering DONE")
print("=" * 60)
