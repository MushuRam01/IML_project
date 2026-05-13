from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FOLDER = BASE_DIR / "company_data"
OUTPUT_FOLDER = BASE_DIR / "processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# Configuration
# ============================================================

MAX_SEQ_LEN = 12
TARGET_COLUMN = "monthly_gross_return"

# ============================================================
# Feature Selection
# ============================================================

EXCLUDE_COLUMNS = [
    "co_code",
    "Month",
    "Year",
    "Corrected_Year",
    "Corrected_Month",
    "Size_Label",
    "BM_Label",
    "OpProf_Label",
    "Inv_Label",
    "Mom_Label"
]

# ============================================================
# Containers
# ============================================================

all_sequences = []
all_masks = []
all_targets = []

all_months = []
all_companies = []

feature_columns = None

# ============================================================
# Process Company Files
# ============================================================

for file_name in os.listdir(INPUT_FOLDER):

    if not file_name.endswith(".csv"):
        continue

    file_path = INPUT_FOLDER / file_name

    print(f"Processing: {file_name}")

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(file_path)

    # Ensure sorted by time
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Month").reset_index(drop=True)

    # --------------------------------------------------------
    # Select Features
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_columns_current = [
        col for col in numeric_columns
        if col not in EXCLUDE_COLUMNS
        and col != TARGET_COLUMN
    ]

    # Store feature columns once
    if feature_columns is None:
        feature_columns = feature_columns_current

    # Ensure same feature ordering everywhere
    #df_features = df[feature_columns]

    # Add missing columns dynamically
    for col in feature_columns:

        if col not in df.columns:
            df[col] = 0

    # Ensure consistent ordering
    df_features = df[feature_columns]

    # Convert to numpy
    feature_array = df_features.values.astype(np.float32)

    # Targets
    targets = df[TARGET_COLUMN].values.astype(np.float32)

    # --------------------------------------------------------
    # Build Sequences
    # --------------------------------------------------------

    num_rows = len(df)

    for target_index in range(1, num_rows):

        # Historical window
        start_index = max(0, target_index - MAX_SEQ_LEN)

        sequence = feature_array[start_index:target_index]

        # Target is next timestep return
        target = np.log(targets[target_index])
        target_month = df.iloc[target_index]["Month"]
        company_id = file_name.replace(".csv", "")

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        seq_len = len(sequence)

        padded_sequence = np.zeros(
            (MAX_SEQ_LEN, len(feature_columns)),
            dtype=np.float32
        )

        mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)

        # Left padding
        padded_sequence[-seq_len:] = sequence
        mask[-seq_len:] = 1.0

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_sequences.append(padded_sequence)
        all_masks.append(mask)
        all_targets.append(target)
        all_months.append(str(target_month))
        all_companies.append(company_id)

# ============================================================
# Convert to Arrays
# ============================================================

X = np.array(all_sequences, dtype=np.float32)
M = np.array(all_masks, dtype=np.float32)
y = np.array(all_targets, dtype=np.float32)
months = np.array(all_months)
companies = np.array(all_companies)

# ============================================================
# Save Arrays
# ============================================================

np.save(OUTPUT_FOLDER / "X.npy", X)
np.save(OUTPUT_FOLDER / "M.npy", M)
np.save(OUTPUT_FOLDER / "y.npy", y)
np.save(OUTPUT_FOLDER / "months.npy", months)
np.save(OUTPUT_FOLDER / "companies.npy", companies)

# ============================================================
# Save Metadata
# ============================================================

metadata = {
    "feature_columns": feature_columns,
    "max_seq_len": MAX_SEQ_LEN,
    "num_features": len(feature_columns)
}

with open(OUTPUT_FOLDER / "metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# ============================================================
# Summary
# ============================================================

print("\nFinished building sequences.\n")

print(f"X shape: {X.shape}")
print(f"M shape: {M.shape}")
print(f"y shape: {y.shape}")

print("\nSaved files:")
print("processed/X.npy")
print("processed/M.npy")
print("processed/y.npy")
print("processed/metadata.pkl")