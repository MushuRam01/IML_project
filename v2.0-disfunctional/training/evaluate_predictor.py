import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))

# ============================================================
# Imports
# ============================================================

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    Subset
)

from datasets.latent_dataset import LatentDataset
from models.predictor import LatentPredictor

# ============================================================
# Paths
# ============================================================

PROCESSED_FOLDER = BASE_DIR / "processed"
CHECKPOINT_FOLDER = BASE_DIR / "checkpoints"
RESULTS_FOLDER = BASE_DIR / "results"

RESULTS_FOLDER.mkdir(exist_ok=True)

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

LATENT_DIM = 32
HIDDEN_DIM = 128

# ============================================================
# Dataset
# ============================================================

dataset = LatentDataset()

# ============================================================
# Test Split
# ============================================================

test_indices = np.load(
    PROCESSED_FOLDER / "test_indices.npy"
)

test_dataset = Subset(
    dataset,
    test_indices
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# Load Model
# ============================================================

model = LatentPredictor(
    latent_dim=LATENT_DIM,
    hidden_dim=HIDDEN_DIM
)

checkpoint_path = (
    CHECKPOINT_FOLDER /
    "latent_predictor.pt"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("\nPredictor loaded successfully.\n")

# ============================================================
# Evaluation
# ============================================================

all_predictions = []
all_targets = []

with torch.no_grad():

    for batch in test_loader:

        z = batch["latent"].to(device)

        y = batch["target"].to(device)

        predictions = model(z)

        predictions = predictions.cpu().numpy()

        y = y.cpu().numpy()

        all_predictions.append(predictions)

        all_targets.append(y)

# ============================================================
# Combine
# ============================================================

predictions = np.concatenate(
    all_predictions,
    axis=0
)

targets = np.concatenate(
    all_targets,
    axis=0
)

# ============================================================
# Metrics
# ============================================================

mse = np.mean(
    (predictions - targets) ** 2
)

mae = np.mean(
    np.abs(predictions - targets)
)

directional_accuracy = np.mean(
    np.sign(predictions) ==
    np.sign(targets)
)

correlation = np.corrcoef(
    predictions,
    targets
)[0,1]

# ============================================================
# Print Metrics
# ============================================================

print("\nEvaluation Metrics\n")

print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(
    f"Directional Accuracy: "
    f"{directional_accuracy:.6f}"
)

print(f"Correlation: {correlation:.6f}")

# ============================================================
# Save Predictions
# ============================================================

results_df = pd.DataFrame({
    "prediction": predictions,
    "target": targets
})

results_path = (
    RESULTS_FOLDER /
    "predictor_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nSaved results:\n")
print(results_path)