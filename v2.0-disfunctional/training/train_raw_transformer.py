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

from torch.optim import Adam

from datasets.financial_dataset import FinancialDataset

from models.raw_transformer import (
    RawFinancialTransformer
)

# ============================================================
# Paths
# ============================================================

PROCESSED_FOLDER = BASE_DIR / "processed"

CHECKPOINT_FOLDER = BASE_DIR / "checkpoints"

CHECKPOINT_FOLDER.mkdir(exist_ok=True)

# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

# ============================================================
# Hyperparameters
# ============================================================

BATCH_SIZE = 256

EPOCHS = 10

LEARNING_RATE = 1e-2

# ============================================================
# Dataset
# ============================================================

dataset = FinancialDataset()

train_indices = np.load(
    PROCESSED_FOLDER / "train_indices.npy"
)

val_indices = np.load(
    PROCESSED_FOLDER / "val_indices.npy"
)

train_dataset = Subset(
    dataset,
    train_indices
)

val_dataset = Subset(
    dataset,
    val_indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# Model
# ============================================================

model = RawFinancialTransformer()

model = model.to(device)

# ============================================================
# Optimizer
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.HuberLoss()

# ============================================================
# Logging
# ============================================================

training_logs = []

# ============================================================
# Training Loop
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.train()

    train_loss_total = 0

    for batch in train_loader:

        x = batch["sequence"].to(device)

        y = batch["target"].to(device)

        optimizer.zero_grad()

        predictions = model(x)

        loss = criterion(
            predictions,
            y
        )

        loss.backward()

        optimizer.step()

        train_loss_total += loss.item()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0

    with torch.no_grad():

        for batch in val_loader:

            x = batch["sequence"].to(device)

            y = batch["target"].to(device)

            predictions = model(x)

            loss = criterion(
                predictions,
                y
            )

            val_loss_total += loss.item()

    avg_train_loss = (
        train_loss_total /
        len(train_loader)
    )

    avg_val_loss = (
        val_loss_total /
        len(val_loader)
    )

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Val Loss: {avg_val_loss:.6f}"
    )

    training_logs.append({

        "epoch": epoch + 1,

        "train_loss": avg_train_loss,

        "val_loss": avg_val_loss
    })

# ============================================================
# Save Model
# ============================================================

checkpoint_path = (
    CHECKPOINT_FOLDER /
    "raw_transformer.pt"
)

torch.save(
    model.state_dict(),
    checkpoint_path
)

print("\nSaved Transformer\n")

print(checkpoint_path)

# ============================================================
# Save Logs
# ============================================================

logs_df = pd.DataFrame(training_logs)

logs_path = (
    CHECKPOINT_FOLDER /
    "raw_transformer_logs.csv"
)

logs_df.to_csv(
    logs_path,
    index=False
)

print(logs_path)
