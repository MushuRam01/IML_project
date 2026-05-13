import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))

# ============================================================
# Imports
# ============================================================

import numpy as np
import pandas as pd
import pickle

import torch

from models.temporal_vae import TemporalVAE
from models.predictor import LatentPredictor

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

MAX_SEQ_LEN = 12

HIDDEN_DIM = 128
LATENT_DIM = 32

# ============================================================
# Load Metadata
# ============================================================

with open(
    PROCESSED_FOLDER / "metadata.pkl",
    "rb"
) as f:

    metadata = pickle.load(f)

feature_columns = metadata["feature_columns"]

num_features = metadata["num_features"]

# ============================================================
# Load Models
# ============================================================

# ------------------------------------------------------------
# Temporal VAE
# ------------------------------------------------------------

vae = TemporalVAE(
    input_dim=num_features,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM
)

vae.load_state_dict(
    torch.load(
        CHECKPOINT_FOLDER / "temporal_vae.pt",
        map_location=device
    )
)

vae = vae.to(device)

vae.eval()

# ------------------------------------------------------------
# Predictor
# ------------------------------------------------------------

predictor = LatentPredictor(
    latent_dim=LATENT_DIM,
    hidden_dim=HIDDEN_DIM
)

predictor.load_state_dict(
    torch.load(
        CHECKPOINT_FOLDER / "latent_predictor.pt",
        map_location=device
    )
)

predictor = predictor.to(device)

predictor.eval()

print("\nModels loaded successfully.\n")

# ============================================================
# Load Company File
# ============================================================

# CHANGE THIS PATH
COMPANY_FILE = (
    BASE_DIR /
    "company_data" /
    "241.csv"
)

df = pd.read_csv(COMPANY_FILE)

# ============================================================
# Preprocessing
# ============================================================

df["Month"] = pd.to_datetime(df["Month"])

df = df.sort_values("Month")

# ------------------------------------------------------------
# Add Missing Columns If Needed
# ------------------------------------------------------------

for col in feature_columns:

    if col not in df.columns:
        df[col] = 0

# ------------------------------------------------------------
# Keep Correct Feature Order
# ------------------------------------------------------------

df_features = df[feature_columns]

# ------------------------------------------------------------
# Convert To Numpy
# ------------------------------------------------------------

feature_array = df_features.values.astype(np.float32)

# ============================================================
# Build Sequence
# ============================================================

sequence = feature_array[-MAX_SEQ_LEN:]

seq_len = len(sequence)

# ------------------------------------------------------------
# Padding
# ------------------------------------------------------------

padded_sequence = np.zeros(
    (MAX_SEQ_LEN, num_features),
    dtype=np.float32
)

mask = np.zeros(
    MAX_SEQ_LEN,
    dtype=np.float32
)

padded_sequence[-seq_len:] = sequence

mask[-seq_len:] = 1.0

# ============================================================
# Convert To Torch
# ============================================================

x = torch.tensor(
    padded_sequence,
    dtype=torch.float32
).unsqueeze(0)

x = x.to(device)

# ============================================================
# Latent Extraction
# ============================================================

with torch.no_grad():

    mu, logvar = vae.encode(x)

    prediction = predictor(mu)

prediction = prediction.item()

# ============================================================
# Output
# ============================================================

print("\nInference Complete\n")

print(f"Company File: {COMPANY_FILE.name}")

print(
    f"Predicted Next-Month Gross Return: "
    f"{prediction:.6f}"
)