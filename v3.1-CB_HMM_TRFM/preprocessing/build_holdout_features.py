# ============================================================
# preprocessing/build_holdout_features.py
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import os

from tqdm import tqdm

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

HOLDOUT_FOLDER = BASE_DIR / "holdout_set"

OUTPUT_FOLDER = (
    BASE_DIR /
    "processed" /
    "holdout_engineered"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# Config
# ============================================================

LAGS = [1, 2, 3, 6]

ROLLING_WINDOWS = [3, 6, 12]

TARGET_COLUMN = "monthly_gross_return"

# ============================================================
# Files
# ============================================================

csv_files = sorted(
    HOLDOUT_FOLDER.glob("*.csv")
)

print("\n==============================")
print("BUILD HOLDOUT FEATURES")
print("==============================")

print("\nFiles Found:\n")

print(len(csv_files))

# ============================================================
# Process Each Company
# ============================================================

for csv_path in tqdm(

    csv_files,

    desc="Holdout Companies",

    ncols=100
):

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_path
    )

    # --------------------------------------------------------
    # Skip Empty
    # --------------------------------------------------------

    if len(df) == 0:
        continue

    # --------------------------------------------------------
    # Company ID
    # --------------------------------------------------------

    company_id = (
        csv_path.stem
    )

    df["company_id"] = str(
        company_id
    )

    # --------------------------------------------------------
    # Datetime
    # --------------------------------------------------------

    df["Month"] = pd.to_datetime(
        df["Month"]
    )

    df = df.sort_values(
        "Month"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Date Features
    # --------------------------------------------------------

    date_features = pd.DataFrame({

        "year":
            df["Month"].dt.year,

        "month_num":
            df["Month"].dt.month,

        "quarter":
            (
                "Q"
                +
                df["Month"]
                .dt.quarter.astype(str)
            )
    })

    df = pd.concat(

        [df, date_features],

        axis=1
    )

    # --------------------------------------------------------
    # Numeric Columns
    # --------------------------------------------------------

    numeric_columns = []

    for col in df.columns:

        if col in [

            "Month",

            "company_id",

            "quarter",

            "Size_Label",

            "BM_Label",

            "OpProf_Label",

            "Inv_Label",

            "Mom_Label"
        ]:
            continue

        try:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            numeric_columns.append(
                col
            )

        except:
            pass

    # --------------------------------------------------------
    # Lag Features
    # --------------------------------------------------------

    lag_feature_dict = {}

    for col in numeric_columns:

        for lag in LAGS:

            lag_feature_dict[
                f"{col}_lag_{lag}"
            ] = df[col].shift(lag)

    lag_df = pd.DataFrame(
        lag_feature_dict
    )

    # --------------------------------------------------------
    # Rolling Features
    # --------------------------------------------------------

    rolling_feature_dict = {}

    for col in numeric_columns:

        for window in ROLLING_WINDOWS:

            rolling_feature_dict[
                f"{col}_roll_mean_{window}"
            ] = (

                df[col]
                .rolling(window)
                .mean()
            )

            rolling_feature_dict[
                f"{col}_roll_std_{window}"
            ] = (

                df[col]
                .rolling(window)
                .std()
            )

    rolling_df = pd.DataFrame(
        rolling_feature_dict
    )

    # --------------------------------------------------------
    # Momentum Features
    # --------------------------------------------------------

    momentum_feature_dict = {}

    for lag in LAGS:

        momentum_feature_dict[
            f"{TARGET_COLUMN}_momentum_{lag}"
        ] = (

            df[TARGET_COLUMN]
            -
            df[TARGET_COLUMN].shift(lag)
        )

    momentum_df = pd.DataFrame(
        momentum_feature_dict
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    df = pd.concat(

        [

            df,

            lag_df,

            rolling_df,

            momentum_df
        ],

        axis=1
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = df.replace(

        [np.inf, -np.inf],

        np.nan
    )

    df = df.fillna(0.0)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (

        OUTPUT_FOLDER /
        f"{company_id}.csv"
    )

    df.to_csv(

        output_path,

        index=False
    )

print("\n==============================")
print("HOLDOUT FEATURES COMPLETE")
print("==============================")