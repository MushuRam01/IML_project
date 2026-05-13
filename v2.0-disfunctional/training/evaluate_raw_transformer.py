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

from datasets.financial_dataset import FinancialDataset

from models.raw_transformer import (
    RawFinancialTransformer
)

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

print(device)

# ============================================================
# Dataset
# ============================================================

dataset = FinancialDataset()

test_indices = np.load(
    PROCESSED_FOLDER / "test_indices.npy"
)

test_dataset = Subset(
    dataset,
    test_indices
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

# ============================================================
# Model
# ============================================================

model = RawFinancialTransformer()

model.load_state_dict(
    torch.load(
        CHECKPOINT_FOLDER /
        "raw_transformer.pt",
        map_location=device
    )
)

model = model.to(device)

model.eval()

# ============================================================
# Evaluation
# ============================================================

all_predictions = []
all_targets = []

with torch.no_grad():

    for batch in test_loader:

        x = batch["sequence"].to(device)

        y = batch["target"].to(device)

        predictions = model(x)

        predictions = predictions.cpu().numpy()

        y = y.cpu().numpy()

        all_predictions.append(predictions)

        all_targets.append(y)

predictions = np.concatenate(all_predictions)

targets = np.concatenate(all_targets)

# ============================================================
# Metrics
# ============================================================

mse = np.mean(
    (predictions - targets) ** 2
)

mae = np.mean(
    np.abs(predictions - targets)
)

correlation = np.corrcoef(
    predictions,
    targets
)[0,1]

prediction_std = np.std(predictions)

# ============================================================
# Print
# ============================================================

print("\nRaw Transformer Metrics\n")

print("MSE:", mse)

print("MAE:", mae)

print("Correlation:", correlation)

print("Prediction Std:", prediction_std)

# ============================================================
# Save Results
# ============================================================

results_df = pd.DataFrame({

    "prediction": predictions,

    "target": targets
})

results_path = (
    RESULTS_FOLDER /
    "raw_transformer_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(results_path)
