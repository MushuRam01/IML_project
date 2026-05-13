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

from torch.utils.data import (
    DataLoader,
    Subset
)

from datasets.latent_dataset import LatentDataset
from models.latent_transformer import LatentTransformer

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
# Dataset
# ============================================================

dataset = LatentDataset()

test_indices = np.load(
    PROCESSED_FOLDER / "test_indices.npy"
)

test_dataset = Subset(
    dataset,
    test_indices
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)

# ============================================================
# Model
# ============================================================

model = LatentTransformer()

model.load_state_dict(
    torch.load(
        CHECKPOINT_FOLDER /
        "latent_transformer.pt",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("\nTransformer loaded.\n")

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

print("\nTransformer Metrics\n")

print(f"MSE: {mse:.6f}")

print(f"MAE: {mae:.6f}")

print(
    f"Directional Accuracy: "
    f"{directional_accuracy:.6f}"
)

print(
    f"Correlation: "
    f"{correlation:.6f}"
)

# ============================================================
# Save Results
# ============================================================

results_df = pd.DataFrame({

    "prediction": predictions,

    "target": targets
})

results_path = (
    RESULTS_FOLDER /
    "transformer_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nSaved results:\n")

print(results_path)