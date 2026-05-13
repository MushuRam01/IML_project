from pathlib import Path
import pandas as pd
import numpy as np
import os

from tqdm import tqdm

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

TARGET_COLUMN = "monthly_gross_return"

LAG_STEPS = [1, 2, 3, 6]

ROLLING_WINDOWS = [3, 6]

IGNORE_COLUMNS = [

    "Month",

    "co_code"
]

# ============================================================
# Containers
# ============================================================

all_dataframes = []

# ============================================================
# Company File List
# ============================================================

company_files = [

    file_name

    for file_name in os.listdir(INPUT_FOLDER)

    if file_name.endswith(".csv")
]

print("\n==============================")
print("FEATURE ENGINEERING")
print("==============================")

print(f"\nTotal Companies: {len(company_files)}")

# ============================================================
# Process Files
# ============================================================

for file_name in tqdm(

    company_files,

    desc="Engineering Features",

    ncols=100
):

    file_path = INPUT_FOLDER / file_name

    df = pd.read_csv(file_path)

    # ========================================================
    # Time Sorting
    # ========================================================

    df["Month"] = pd.to_datetime(
        df["Month"]
    )

    df = df.sort_values(
        "Month"
    ).reset_index(drop=True)

    company_id = file_name.replace(
        ".csv",
        ""
    )

    # ========================================================
    # Numeric Feature Discovery
    # ========================================================

    numeric_columns = []

    for col in df.columns:

        if col in IGNORE_COLUMNS:
            continue

        if col == TARGET_COLUMN:
            continue

        converted = pd.to_numeric(

            df[col],

            errors="coerce"
        )

        # ----------------------------------------------------
        # Keep Only Columns With Numeric Signal
        # ----------------------------------------------------

        if converted.notna().sum() > 0:

            df[col] = converted

            numeric_columns.append(col)

    # ========================================================
    # Engineered Feature Container
    # ========================================================

    engineered_features = {}

    # ========================================================
    # Lag Features
    # ========================================================

    for col in numeric_columns:

        series = df[col]

        for lag in LAG_STEPS:

            engineered_features[
                f"{col}_lag_{lag}"
            ] = series.shift(lag)

    # ========================================================
    # Rolling Features
    # ========================================================

    for col in numeric_columns:

        series = df[col]

        for window in ROLLING_WINDOWS:

            rolling = series.rolling(window)

            engineered_features[
                f"{col}_roll_mean_{window}"
            ] = rolling.mean()

            engineered_features[
                f"{col}_roll_std_{window}"
            ] = rolling.std()

    # ========================================================
    # Target Momentum Features
    # ========================================================

    if TARGET_COLUMN in df.columns:

        target_series = df[TARGET_COLUMN]

        engineered_features[
            "return_momentum_3"
        ] = (

            target_series
            .rolling(3)
            .mean()
        )

        engineered_features[
            "return_momentum_6"
        ] = (

            target_series
            .rolling(6)
            .mean()
        )

        engineered_features[
            "return_volatility_3"
        ] = (

            target_series
            .rolling(3)
            .std()
        )

        engineered_features[
            "return_volatility_6"
        ] = (

            target_series
            .rolling(6)
            .std()
        )

        engineered_features[
            "return_diff_1"
        ] = target_series.diff(1)

        engineered_features[
            "return_diff_3"
        ] = target_series.diff(3)

    # ========================================================
    # Merge Engineered Features
    # ========================================================

    engineered_df = pd.DataFrame(
        engineered_features
    )

    df = pd.concat(

        [df, engineered_df],

        axis=1
    )

    # ========================================================
    # Replace Invalid Values
    # ========================================================

    df = df.replace(

        [np.inf, -np.inf],

        np.nan
    )

    # ========================================================
    # Fill Missing Values
    # ========================================================

    df = df.fillna(0.0)

    # ========================================================
    # Add Company ID
    # ========================================================

    df["company_id"] = company_id

    # ========================================================
    # Append
    # ========================================================

    all_dataframes.append(df)

# ============================================================
# Merge All Companies
# ============================================================

print("\nMerging DataFrames\n")

final_df = pd.concat(

    all_dataframes,

    ignore_index=True
)

# ============================================================
# Global Chronological Ordering
# ============================================================

final_df["Month"] = pd.to_datetime(
    final_df["Month"]
)

final_df = final_df.sort_values(

    "Month"

).reset_index(drop=True)

# ============================================================
# Final Cleanup
# ============================================================

final_df = final_df.replace(

    [np.inf, -np.inf],

    np.nan
)

final_df = final_df.fillna(0.0)

# ============================================================
# Defragment Memory
# ============================================================

final_df = final_df.copy()

# ============================================================
# Save
# ============================================================

save_path = (
    OUTPUT_FOLDER /
    "engineered_features.csv"
)

print("\nSaving Engineered Features\n")

final_df.to_csv(

    save_path,

    index=False
)

# ============================================================
# Summary
# ============================================================

print("\n==============================")
print("FEATURE ENGINEERING COMPLETE")
print("==============================")

print("\nFinal Shape:\n")

print(final_df.shape)

print("\nTotal Columns:\n")

print(len(final_df.columns))

print("\nSaved To:\n")

print(save_path)