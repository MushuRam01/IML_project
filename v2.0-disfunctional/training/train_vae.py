import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))



from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
import numpy as np

# ============================================================
# Imports
# ============================================================

from datasets.financial_dataset import FinancialDataset
from models.temporal_vae import TemporalVAE

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_FOLDER = BASE_DIR / "processed"
CHECKPOINT_FOLDER = BASE_DIR / "checkpoints"

CHECKPOINT_FOLDER.mkdir(exist_ok=True)

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

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3

HIDDEN_DIM = 128
LATENT_DIM = 32

BETA = 0.001

# ============================================================
# Dataset
# ============================================================

dataset = FinancialDataset()

# ============================================================
# Load Splits
# ============================================================

train_indices = np.load(
    PROCESSED_FOLDER / "train_indices.npy"
)

val_indices = np.load(
    PROCESSED_FOLDER / "val_indices.npy"
)

# ============================================================
# Subsets
# ============================================================

train_dataset = Subset(
    dataset,
    train_indices
)

val_dataset = Subset(
    dataset,
    val_indices
)

# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# Model
# ============================================================

input_dim = dataset.X.shape[-1]

model = TemporalVAE(
    input_dim=input_dim,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM
)

model = model.to(device)

# ============================================================
# Optimizer
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# Loss Functions
# ============================================================

mse_loss = nn.MSELoss(reduction="none")

# ============================================================
# Training Loop
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss_total = 0

    for batch in train_loader:

        x = batch["sequence"].to(device)

        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        output = model(x)

        reconstruction = output["reconstruction"]

        mu = output["mu"]

        logvar = output["logvar"]

        # ----------------------------------------------------
        # Reconstruction Loss
        # ----------------------------------------------------

        reconstruction_loss = mse_loss(
            reconstruction,
            x
        )

        # Apply mask
        reconstruction_loss = (
            reconstruction_loss *
            mask.unsqueeze(-1)
        )

        reconstruction_loss = reconstruction_loss.mean()

        # ----------------------------------------------------
        # KL Divergence
        # ----------------------------------------------------

        kl_loss = -0.5 * torch.mean(
            1 +
            logvar -
            mu.pow(2) -
            logvar.exp()
        )

        # ----------------------------------------------------
        # Total Loss
        # ----------------------------------------------------

        loss = (
            reconstruction_loss +
            BETA * kl_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

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

            mask = batch["mask"].to(device)

            output = model(x)

            reconstruction = output["reconstruction"]

            mu = output["mu"]

            logvar = output["logvar"]

            reconstruction_loss = mse_loss(
                reconstruction,
                x
            )

            reconstruction_loss = (
                reconstruction_loss *
                mask.unsqueeze(-1)
            )

            reconstruction_loss = reconstruction_loss.mean()

            kl_loss = -0.5 * torch.mean(
                1 +
                logvar -
                mu.pow(2) -
                logvar.exp()
            )

            loss = (
                reconstruction_loss +
                BETA * kl_loss
            )

            val_loss_total += loss.item()

    # --------------------------------------------------------
    # Epoch Summary
    # --------------------------------------------------------

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

# ============================================================
# Save Model
# ============================================================

checkpoint_path = (
    CHECKPOINT_FOLDER /
    "temporal_vae.pt"
)

torch.save(
    model.state_dict(),
    checkpoint_path
)

print("\nModel saved.\n")
print(checkpoint_path)