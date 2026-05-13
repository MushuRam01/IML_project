from pathlib import Path
import numpy as np
import os

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_FOLDER = BASE_DIR / "processed"

# ============================================================
# Load Metadata
# ============================================================

months = np.load(
    PROCESSED_FOLDER / "months.npy",
    allow_pickle=True
)

# ============================================================
# Convert to Datetime
# ============================================================

months = months.astype("datetime64[M]")

# ============================================================
# Define Splits
# ============================================================

train_end = np.datetime64("2019-01")
val_end = np.datetime64("2022-01")

# ============================================================
# Create Indices
# ============================================================

train_indices = np.where(months < train_end)[0]

val_indices = np.where(
    (months >= train_end) &
    (months < val_end)
)[0]

test_indices = np.where(months >= val_end)[0]

# ============================================================
# Save Splits
# ============================================================

np.save(
    PROCESSED_FOLDER / "train_indices.npy",
    train_indices
)

np.save(
    PROCESSED_FOLDER / "val_indices.npy",
    val_indices
)

np.save(
    PROCESSED_FOLDER / "test_indices.npy",
    test_indices
)

# ============================================================
# Summary
# ============================================================

print("\nChronological splits created.\n")

print(f"Train samples: {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")
print(f"Test samples: {len(test_indices)}")