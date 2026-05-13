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
import matplotlib.pyplot as plt

import torch

from models.temporal_vae import TemporalVAE
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
# Company File
# ============================================================

# CHANGE THIS
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

df = df.sort_values("Month").reset_index(drop=True)

# ------------------------------------------------------------
# Add Missing Columns
# ------------------------------------------------------------

for col in feature_columns:

    if col not in df.columns:
        df[col] = 0

# ------------------------------------------------------------
# Feature Ordering
# ------------------------------------------------------------

df_features = df[feature_columns]

feature_array = df_features.values.astype(np.float32)

targets = df["monthly_gross_return"].values

months = df["Month"].values

# ============================================================
# Rolling Inference
# ============================================================

predictions = []
actuals = []
prediction_months = []

with torch.no_grad():

    for target_index in range(1, len(df)):

        # ----------------------------------------------------
        # Build Sequence
        # ----------------------------------------------------

        start_index = max(
            0,
            target_index - MAX_SEQ_LEN
        )

        sequence = feature_array[
            start_index:target_index
        ]

        actual_return = targets[target_index]

        prediction_month = months[target_index]

        seq_len = len(sequence)

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        padded_sequence = np.zeros(
            (MAX_SEQ_LEN, num_features),
            dtype=np.float32
        )

        padded_sequence[-seq_len:] = sequence

        # ----------------------------------------------------
        # Torch Tensor
        # ----------------------------------------------------

        x = torch.tensor(
            padded_sequence,
            dtype=torch.float32
        ).unsqueeze(0)

        x = x.to(device)

        # ----------------------------------------------------
        # Encode
        # ----------------------------------------------------

        mu, logvar = vae.encode(x)

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predicted_return = predictor(mu)

        predicted_return = (
            predicted_return
            .cpu()
            .item()
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        predictions.append(predicted_return)

        actuals.append(actual_return)

        prediction_months.append(prediction_month)

# ============================================================
# Convert Arrays
# ============================================================

predictions = np.array(predictions)

actuals = np.array(actuals)

# ============================================================
# Metrics
# ============================================================

mse = np.mean(
    (predictions - actuals) ** 2
)

mae = np.mean(
    np.abs(predictions - actuals)
)

directional_accuracy = np.mean(
    np.sign(predictions) ==
    np.sign(actuals)
)

correlation = np.corrcoef(
    predictions,
    actuals
)[0,1]

# ============================================================
# Print Metrics
# ============================================================

print("\nDebug Inference Metrics\n")

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
# Prediction Table
# ============================================================

results_df = pd.DataFrame({

    "Month": prediction_months,

    "Prediction": predictions,

    "Actual": actuals,

    "Absolute_Error":
        np.abs(predictions - actuals)
})

print("\nSample Predictions:\n")

print(results_df.head(10))

# ============================================================
# Save CSV
# ============================================================

csv_path = (
    RESULTS_FOLDER /
    "debug_inference_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

print("\nSaved results:\n")
print(csv_path)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(14,6))

plt.plot(
    prediction_months,
    actuals,
    label="Actual"
)

plt.plot(
    prediction_months,
    predictions,
    label="Predicted"
)

plt.xlabel("Month")

plt.ylabel("Monthly Gross Return")

plt.title(
    f"Prediction vs Actual: "
    f"{COMPANY_FILE.name}"
)

plt.legend()

plt.tight_layout()

plot_path = (
    RESULTS_FOLDER /
    "debug_inference_plot.png"
)

plt.savefig(plot_path)

print("\nSaved plot:\n")
print(plot_path)

plt.show()