from pathlib import Path
import pandas as pd
import numpy as np
import os

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Company data folder
INPUT_FOLDER = BASE_DIR / "company_data"
# Process every CSV file
for file_name in os.listdir(INPUT_FOLDER):

    if not file_name.endswith(".csv"):
        continue

    file_path = os.path.join(INPUT_FOLDER, file_name)

    # Read CSV
    df = pd.read_csv(file_path)

    # Sort by time just to be safe
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Month")

    # Select numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Skip identifier-like columns
    exclude_columns = [
        "co_code",
        "Year",
        "Corrected_Year",
        "Corrected_Month"
    ]

    numeric_columns = [
        col for col in numeric_columns
        if col not in exclude_columns
    ]

    # Create missingness indicator columns
    for col in numeric_columns:

        missing_col_name = f"{col}_missing"

        df[missing_col_name] = (
            df[col]
            .isna()
            .astype(int)
        )

    # Forward fill numeric columns
    df[numeric_columns] = df[numeric_columns].ffill()

    # Replace remaining NaNs with 0
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Save back to same file
    df.to_csv(file_path, index=False)

    print(f"Processed: {file_name}")

print("All company files processed.")