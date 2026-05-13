from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_FOLDER = BASE_DIR / "processed"

RESULTS_FOLDER = BASE_DIR / "results"

CHECKPOINT_FOLDER = BASE_DIR / "checkpoints"

SAVE_FOLDER = (
    RESULTS_FOLDER /
    "residual_transformer"
)

PLOT_FOLDER = (
    SAVE_FOLDER /
    "company_plots"
)

os.makedirs(
    SAVE_FOLDER,
    exist_ok=True
)

os.makedirs(
    PLOT_FOLDER,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_FOLDER,
    exist_ok=True
)

# ============================================================
# Device
# ============================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

print("\nUsing Device:\n")

print(DEVICE)

# ============================================================
# Hyperparameters
# ============================================================

BATCH_SIZE = 128

EPOCHS = 20

LEARNING_RATE = 1e-4

EMBED_DIM = 64

NUM_HEADS = 4

NUM_LAYERS = 2

DROPOUT = 0.1

# ============================================================
# Load Residual Sequences
# ============================================================

X_train = np.load(

    PROCESSED_FOLDER /
    "residual_X_train.npy"
)

y_train = np.load(

    PROCESSED_FOLDER /
    "residual_y_train.npy"
)

X_test = np.load(

    PROCESSED_FOLDER /
    "residual_X_test.npy"
)

y_test = np.load(

    PROCESSED_FOLDER /
    "residual_y_test.npy"
)

catboost_test = np.load(

    PROCESSED_FOLDER /
    "residual_catboost_test.npy"
)

actual_test = np.load(

    PROCESSED_FOLDER /
    "residual_actual_test.npy"
)

months_test = np.load(

    PROCESSED_FOLDER /
    "residual_months_test.npy"
)

companies_test = np.load(

    PROCESSED_FOLDER /
    "residual_companies_test.npy"
)

print("\nLoaded Residual Sequences\n")

print(X_train.shape)

print(X_test.shape)

# ============================================================
# Normalize Features
# ============================================================

mean = X_train.mean(
    axis=(0,1),
    keepdims=True
)

std = X_train.std(
    axis=(0,1),
    keepdims=True
)

std[std < 1e-6] = 1.0

X_train = (
    X_train - mean
) / std

X_test = (
    X_test - mean
) / std

# ============================================================
# Replace Invalid Values
# ============================================================

X_train = np.nan_to_num(
    X_train
)

X_test = np.nan_to_num(
    X_test
)

y_train = np.nan_to_num(
    y_train
)

y_test = np.nan_to_num(
    y_test
)

# ============================================================
# Dataset
# ============================================================

class ResidualDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(
        self,
        idx
    ):

        return (
            self.X[idx],
            self.y[idx]
        )

# ============================================================
# Transformer Model
# ============================================================

class ResidualTransformer(nn.Module):

    def __init__(
        self,
        num_features
    ):

        super().__init__()

        self.embedding = nn.Linear(
            num_features,
            EMBED_DIM
        )

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=EMBED_DIM,

            nhead=NUM_HEADS,

            dim_feedforward=128,

            dropout=DROPOUT,

            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=NUM_LAYERS
        )

        self.head = nn.Sequential(

            nn.Linear(
                EMBED_DIM,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            )
        )

    def forward(
        self,
        x
    ):

        x = self.embedding(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.head(x)

        return x.squeeze()

# ============================================================
# DataLoader
# ============================================================

train_dataset = ResidualDataset(

    X_train,

    y_train
)

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True
)

# ============================================================
# Model
# ============================================================

model = ResidualTransformer(

    X_train.shape[2]
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)

# ============================================================
# Training
# ============================================================

print("\n==============================")
print("TRAINING TRANSFORMER")
print("==============================")

best_loss = np.inf

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)

        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(
            batch_X
        )

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0
        )

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = (
        epoch_loss /
        len(train_loader)
    )

    print(

        f"Epoch {epoch+1} | "
        f"Loss: {avg_loss:.6f}"
    )

    # --------------------------------------------------------
    # Save Best
    # --------------------------------------------------------

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(

            model.state_dict(),

            CHECKPOINT_FOLDER /
            "best_residual_transformer.pt"
        )

# ============================================================
# Load Best Model
# ============================================================

model.load_state_dict(

    torch.load(

        CHECKPOINT_FOLDER /
        "best_residual_transformer.pt",

        map_location=DEVICE
    )
)

# ============================================================
# Inference
# ============================================================

print("\nRunning Inference\n")

model.eval()

with torch.no_grad():

    X_tensor = torch.tensor(

        X_test,

        dtype=torch.float32
    ).to(DEVICE)

    predicted_residuals = model(
        X_tensor
    ).cpu().numpy()

# ============================================================
# Correct Predictions
# ============================================================

corrected_predictions = (

    catboost_test
    +
    predicted_residuals
)

# ============================================================
# Metrics
# ============================================================

mae = mean_absolute_error(

    actual_test,

    corrected_predictions
)

rmse = np.sqrt(

    mean_squared_error(

        actual_test,

        corrected_predictions
    )
)

directional_accuracy = np.mean(

    np.sign(corrected_predictions)

    ==

    np.sign(actual_test)
)

r2 = r2_score(

    actual_test,

    corrected_predictions
)

print("\n==============================")
print("TRANSFORMER RESULTS")
print("==============================")

print("\nMAE:")

print(mae)

print("\nRMSE:")

print(rmse)

print("\nDirectional Accuracy:")

print(directional_accuracy)

print("\nR2:")

print(r2)

# ============================================================
# Scatter Plot
# ============================================================

plt.figure(figsize=(10,10))

plt.scatter(

    actual_test,

    corrected_predictions,

    alpha=0.2
)

min_val = min(

    actual_test.min(),

    corrected_predictions.min()
)

max_val = max(

    actual_test.max(),

    corrected_predictions.max()
)

plt.plot(

    [min_val, max_val],

    [min_val, max_val]
)

plt.xlabel(
    "Actual"
)

plt.ylabel(
    "Prediction"
)

plt.title(
    "CatBoost + Residual Transformer"
)

plt.savefig(

    SAVE_FOLDER /
    "scatter.png",

    dpi=300,

    bbox_inches="tight"
)

plt.close()

# ============================================================
# Save Results CSV
# ============================================================

results_df = pd.DataFrame({

    "month":
        months_test,

    "company":
        companies_test,

    "actual":
        actual_test,

    "catboost":
        catboost_test,

    "predicted_residual":
        predicted_residuals,

    "corrected":
        corrected_predictions
})

results_df.to_csv(

    SAVE_FOLDER /
    "results.csv",

    index=False
)

# ============================================================
# Random Company Plots
# ============================================================

unique_companies = np.unique(
    companies_test
)

selected_companies = random.sample(

    list(unique_companies),

    min(50, len(unique_companies))
)

for company in selected_companies:

    mask = companies_test == company

    if np.sum(mask) < 2:
        continue

    plt.figure(figsize=(14,5))

    plt.plot(

        actual_test[mask],

        label="Actual"
    )

    plt.plot(

        catboost_test[mask],

        label="CatBoost"
    )

    plt.plot(

        corrected_predictions[mask],

        label="Corrected"
    )

    plt.title(
        f"Company {company}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        PLOT_FOLDER /
        f"{company}.png",

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

print("\nSaved Results\n")