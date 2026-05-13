import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))

# ============================================================
# Imports
# ============================================================

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.financial_dataset import FinancialDataset
from models.temporal_vae import TemporalVAE

# ============================================================
# Paths
# ============================================================

PROCESSED_FOLDER = BASE_DIR / "processed"
CHECKPOINT_FOLDER = BASE_DIR / "checkpoints"

# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}\n")

# ============================================================
# Hyperparameters
# ============================================================

BATCH_SIZE = 256

HIDDEN_DIM = 128
LATENT_DIM = 32

# ============================================================
# Load Dataset
# ============================================================

dataset = FinancialDataset()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# Load Model
# ============================================================

input_dim = dataset.X.shape[-1]

model = TemporalVAE(
    input_dim=input_dim,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM
)

checkpoint_path = (
    CHECKPOINT_FOLDER /
    "temporal_vae.pt"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("\nTemporalVAE loaded successfully.\n")

# ============================================================
# Extract Latent Means
# ============================================================

all_latents = []

with torch.no_grad():

    for batch in loader:

        x = batch["sequence"].to(device)

        # ----------------------------------------------------
        # Encode
        # ----------------------------------------------------

        mu, logvar = model.encode(x)

        # ----------------------------------------------------
        # Store latent means
        # ----------------------------------------------------

        mu = mu.cpu().numpy()

        all_latents.append(mu)

# ============================================================
# Combine All Latents
# ============================================================

latent_array = np.concatenate(
    all_latents,
    axis=0
)

# ============================================================
# Save Latents
# ============================================================

output_path = (
    PROCESSED_FOLDER /
    "latent_mu.npy"
)

np.save(
    output_path,
    latent_array
)

# ============================================================
# Summary
# ============================================================

print("\nLatent extraction complete.\n")

print(f"Latent shape: {latent_array.shape}")

print("\nSaved:")
print(output_path)