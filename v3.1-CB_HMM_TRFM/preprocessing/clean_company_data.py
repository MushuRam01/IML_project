from pathlib import Path
import pandas as pd
import numpy as np
import os

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FOLDER = BASE_DIR / "company_data"

OUTPUT_FOLDER = BASE_DIR / "cleaned_company_data"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# Columns To Ignore
# ============================================================

IGNORE_COLUMNS = [

    "Month",

    "co_code"
]

# ============================================================
# Summary Counters
# ============================================================

total_files = 0

total_nan_before = 0

total_nan_after = 0

# ============================================================
# Process Files
# ============================================================

for file_name in os.listdir(INPUT_FOLDER):

    if not file_name.endswith(".csv"):
        continue

    total_files += 1

    print(f"\nCleaning: {file_name}")

    file_path = INPUT_FOLDER / file_name

    df = pd.read_csv(file_path)

    # --------------------------------------------------------
    # Replace Inf With NaN
    # --------------------------------------------------------

    df = df.replace(

        [np.inf, -np.inf],

        np.nan
    )

    # --------------------------------------------------------
    # Count NaNs Before
    # --------------------------------------------------------

    nan_before = df.isna().sum().sum()

    total_nan_before += nan_before

    print(
        f"NaNs Before: {nan_before}"
    )

    # --------------------------------------------------------
    # Numeric Columns
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(

        include=[np.number]
    ).columns.tolist()

    # --------------------------------------------------------
    # Remove Ignored Columns
    # --------------------------------------------------------

    numeric_columns = [

        col for col in numeric_columns

        if col not in IGNORE_COLUMNS
    ]

    # --------------------------------------------------------
    # Missing Indicators
    # --------------------------------------------------------

    for col in numeric_columns:

        missing_mask = df[col].isna()

        df[f"{col}_missing"] = (
            missing_mask.astype(np.int8)
        )

    # --------------------------------------------------------
    # Median Imputation
    # --------------------------------------------------------

    for col in numeric_columns:

        median_value = df[col].median()

        # Handle all-NaN columns
        if pd.isna(median_value):

            median_value = 0.0

        df[col] = df[col].fillna(
            median_value
        )

    # --------------------------------------------------------
    # Final NaN Count
    # --------------------------------------------------------

    nan_after = df.isna().sum().sum()

    total_nan_after += nan_after

    print(
        f"NaNs After: {nan_after}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_FOLDER / file_name
    )

    df.to_csv(
        output_path,
        index=False
    )

# ============================================================
# Final Summary
# ============================================================

print("\n==============================")
print("CLEANING COMPLETE")
print("==============================")

print(f"Files Processed: {total_files}")

print(
    f"Total NaNs Before: "
    f"{total_nan_before}"
)

print(
    f"Total NaNs After: "
    f"{total_nan_after}"
)

print(
    "\nCleaned files saved to:\n"
)

print(OUTPUT_FOLDER)
