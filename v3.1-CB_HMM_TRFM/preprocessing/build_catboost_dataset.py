from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FOLDER = BASE_DIR / "cleaned_company_data"

OUTPUT_FOLDER = BASE_DIR / "processed"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# Config
# ============================================================

WINDOW_SIZE = 12

TARGET_COLUMN = "monthly_gross_return"

# ============================================================
# Ignore Columns
# ============================================================

IGNORE_COLUMNS = [

    "Month",

    "co_code"
]

# ============================================================
# Containers
# ============================================================

all_sequences = []

all_flattened = []

all_targets = []

all_months = []

all_companies = []

all_history_lengths = []

feature_columns = None

# ============================================================
# Process Files
# ============================================================

for file_name in os.listdir(INPUT_FOLDER):

    if not file_name.endswith(".csv"):
        continue

    print(f"Processing: {file_name}")

    file_path = INPUT_FOLDER / file_name

    df = pd.read_csv(file_path)

    # --------------------------------------------------------
    # Time Sorting
    # --------------------------------------------------------

    df["Month"] = pd.to_datetime(
        df["Month"]
    )

    df = df.sort_values(
        "Month"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Initial Numeric Feature Discovery
    # --------------------------------------------------------

    feature_columns_current = []

    for col in df.columns:

        if col in IGNORE_COLUMNS:
            continue

        if col == TARGET_COLUMN:
            continue

        # ----------------------------------------------------
        # Try Numeric Conversion
        # ----------------------------------------------------

        converted = pd.to_numeric(

            df[col],

            errors="coerce"
        )

        # ----------------------------------------------------
        # Keep If At Least Some Numeric Values Exist
        # ----------------------------------------------------

        if converted.notna().sum() > 0:

            feature_columns_current.append(
                col
            )

    # --------------------------------------------------------
    # Global Feature Ordering
    # --------------------------------------------------------

    if feature_columns is None:

        feature_columns = feature_columns_current

    # --------------------------------------------------------
    # Ensure Consistent Columns
    # --------------------------------------------------------

    for col in feature_columns:

        if col not in df.columns:

            df[col] = 0.0

    # --------------------------------------------------------
    # Extract Features
    # --------------------------------------------------------

    df_features = df[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Force Numeric Conversion
    # --------------------------------------------------------

    for col in feature_columns:

        df_features[col] = pd.to_numeric(

            df_features[col],

            errors="coerce"
        )

    # --------------------------------------------------------
    # Replace Invalid Values
    # --------------------------------------------------------

    df_features = df_features.replace(

        [np.inf, -np.inf],

        np.nan
    )

    # --------------------------------------------------------
    # Fill Missing Values
    # --------------------------------------------------------

    df_features = df_features.fillna(
        0.0
    )

    # --------------------------------------------------------
    # Final Float Conversion
    # --------------------------------------------------------

    feature_array = df_features.values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    targets = pd.to_numeric(

        df[TARGET_COLUMN],

        errors="coerce"
    ).values.astype(np.float32)

    # --------------------------------------------------------
    # Rolling Windows
    # --------------------------------------------------------

    for target_index in range(1, len(df)):

        start_index = max(

            0,

            target_index - WINDOW_SIZE
        )

        window = feature_array[
            start_index:target_index
        ]

        history_length = len(window)

        # ----------------------------------------------------
        # Left Padding
        # ----------------------------------------------------

        padded_window = np.zeros(

            (
                WINDOW_SIZE,
                len(feature_columns)
            ),

            dtype=np.float32
        )

        padded_window[
            -history_length:
        ] = window

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        gross_return = targets[
            target_index
        ]

        if np.isnan(gross_return):
            continue

        if gross_return <= 0:
            continue

        target = np.log(
            gross_return
        )

        # ----------------------------------------------------
        # Flatten For CatBoost
        # ----------------------------------------------------

        flattened_window = padded_window.flatten()

        flattened_window = np.concatenate([

            flattened_window,

            np.array(
                [history_length],
                dtype=np.float32
            )
        ])

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_sequences.append(
            padded_window
        )

        all_flattened.append(
            flattened_window
        )

        all_targets.append(
            target
        )

        all_months.append(
            str(df.iloc[target_index]["Month"])
        )

        all_companies.append(
            file_name.replace(".csv", "")
        )

        all_history_lengths.append(
            history_length
        )

# ============================================================
# Convert To Arrays
# ============================================================

X_sequences = np.array(
    all_sequences,
    dtype=np.float32
)

X_flattened = np.array(
    all_flattened,
    dtype=np.float32
)

y = np.array(
    all_targets,
    dtype=np.float32
)

months = np.array(
    all_months
)

companies = np.array(
    all_companies
)

history_lengths = np.array(
    all_history_lengths
)

# ============================================================
# Final Safety Cleaning
# ============================================================

X_sequences = np.nan_to_num(

    X_sequences,

    nan=0.0,

    posinf=0.0,

    neginf=0.0
)

X_flattened = np.nan_to_num(

    X_flattened,

    nan=0.0,

    posinf=0.0,

    neginf=0.0
)

y = np.nan_to_num(

    y,

    nan=0.0,

    posinf=0.0,

    neginf=0.0
)

# ============================================================
# Save Arrays
# ============================================================

np.save(
    OUTPUT_FOLDER / "X_sequences.npy",
    X_sequences
)

np.save(
    OUTPUT_FOLDER / "X_flattened.npy",
    X_flattened
)

np.save(
    OUTPUT_FOLDER / "y.npy",
    y
)

np.save(
    OUTPUT_FOLDER / "months.npy",
    months
)

np.save(
    OUTPUT_FOLDER / "companies.npy",
    companies
)

np.save(
    OUTPUT_FOLDER / "history_lengths.npy",
    history_lengths
)

# ============================================================
# Metadata
# ============================================================

metadata = {

    "feature_columns":
        feature_columns,

    "window_size":
        WINDOW_SIZE,

    "num_features":
        len(feature_columns)
}

with open(
    OUTPUT_FOLDER / "metadata.pkl",
    "wb"
) as f:

    pickle.dump(
        metadata,
        f
    )

# ============================================================
# Summary
# ============================================================

print("\n==============================")
print("DATASET BUILD COMPLETE")
print("==============================")

print(
    "Sequence Shape:",
    X_sequences.shape
)

print(
    "Flattened Shape:",
    X_flattened.shape
)

print(
    "Target Shape:",
    y.shape
)

print(
    "NaNs in Sequences:",
    np.isnan(X_sequences).sum()
)

print(
    "NaNs in Flattened:",
    np.isnan(X_flattened).sum()
)

print(
    "NaNs in Targets:",
    np.isnan(y).sum()
)

print("\nSaved To:\n")

print(OUTPUT_FOLDER)