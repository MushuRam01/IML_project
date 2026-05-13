# ============================================================
# preprocessing/build_residual_sequences.py
# FAST + FAIR VERSION
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

PROCESSED_FOLDER = BASE_DIR / "processed"

RESULTS_FOLDER = BASE_DIR / "results"

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)

# ============================================================
# Config
# ============================================================

WINDOW_SIZE = 6

STRIDE = 3

TARGET_COLUMN = "monthly_gross_return"

# ============================================================
# Load Data
# ============================================================

features_df = pd.read_csv(

    PROCESSED_FOLDER /
    "engineered_features.csv",

    low_memory=False
)

validation_df = pd.read_csv(

    RESULTS_FOLDER /
    "validation_residuals.csv"
)

test_df = pd.read_csv(

    RESULTS_FOLDER /
    "test_predictions.csv"
)

# ============================================================
# Datetime
# ============================================================

features_df["Month"] = pd.to_datetime(
    features_df["Month"]
)

validation_df["Month"] = pd.to_datetime(
    validation_df["Month"]
)

test_df["Month"] = pd.to_datetime(
    test_df["Month"]
)

# ============================================================
# REMOVE LEAKAGE FEATURES
# ============================================================

REMOVE_COLUMNS = [

    TARGET_COLUMN,

    "Month",

    "company_id"
]

feature_columns = [

    col

    for col in features_df.columns

    if col not in REMOVE_COLUMNS
]

print("\n==============================")
print("FEATURE COUNT")
print("==============================")

print(len(feature_columns))

# ============================================================
# Merge
# ============================================================

validation_merged = validation_df.merge(

    features_df,

    on=[

        "Month",

        "company_id"
    ],

    how="left"
)

test_merged = test_df.merge(

    features_df,

    on=[

        "Month",

        "company_id"
    ],

    how="left"
)

# ============================================================
# GLOBAL NUMERIC CONVERSION
# ============================================================

print("\nConverting Features To Numeric\n")

for col in tqdm(

    feature_columns,

    desc="Numeric Conversion",

    ncols=100
):

    validation_merged[col] = pd.to_numeric(

        validation_merged[col],

        errors="coerce"
    )

    test_merged[col] = pd.to_numeric(

        test_merged[col],

        errors="coerce"
    )

validation_merged = validation_merged.fillna(0.0)

test_merged = test_merged.fillna(0.0)

# ============================================================
# Containers
# ============================================================

X_train = []

y_train = []

X_test = []

y_test = []

catboost_test_predictions = []

actual_test_targets = []

months_test = []

companies_test = []

# ============================================================
# BUILD TRAIN WINDOWS
# ============================================================

print("\n==============================")
print("BUILDING TRAIN WINDOWS")
print("==============================")

train_companies = validation_merged[
    "company_id"
].unique()

for company_id in tqdm(

    train_companies,

    desc="Train Windows",

    ncols=100
):

    company_df = validation_merged[

        validation_merged[
            "company_id"
        ] == company_id

    ].copy()

    company_df = company_df.sort_values(
        "Month"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Skip Short Histories
    # --------------------------------------------------------

    if len(company_df) < WINDOW_SIZE + 1:
        continue

    # --------------------------------------------------------
    # Pre-Extract Numeric Matrix
    # --------------------------------------------------------

    feature_matrix = company_df[
        feature_columns
    ].values.astype(np.float32)

    residuals = company_df[
        "residual"
    ].values.astype(np.float32)

    # --------------------------------------------------------
    # Sliding Windows
    # --------------------------------------------------------

    for end_idx in range(

        WINDOW_SIZE,

        len(company_df),

        STRIDE
    ):

        start_idx = (
            end_idx - WINDOW_SIZE
        )

        sequence = feature_matrix[
            start_idx:end_idx
        ]

        residual_target = residuals[
            end_idx
        ]

        X_train.append(sequence)

        y_train.append(
            residual_target
        )

# ============================================================
# BUILD TEST WINDOWS
# ============================================================

print("\n==============================")
print("BUILDING TEST WINDOWS")
print("==============================")

test_companies = test_merged[
    "company_id"
].unique()

for company_id in tqdm(

    test_companies,

    desc="Test Windows",

    ncols=100
):

    company_df = test_merged[

        test_merged[
            "company_id"
        ] == company_id

    ].copy()

    company_df = company_df.sort_values(
        "Month"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Skip Short Histories
    # --------------------------------------------------------

    if len(company_df) < WINDOW_SIZE + 1:
        continue

    # --------------------------------------------------------
    # Pre-Extract Arrays
    # --------------------------------------------------------

    feature_matrix = company_df[
        feature_columns
    ].values.astype(np.float32)

    residuals = company_df[
        "residual"
    ].values.astype(np.float32)

    predictions = company_df[
        "prediction"
    ].values.astype(np.float32)

    targets = company_df[
        "target"
    ].values.astype(np.float32)

    months = company_df[
        "Month"
    ].astype(str).values

    # --------------------------------------------------------
    # Sliding Windows
    # --------------------------------------------------------

    for end_idx in range(

        WINDOW_SIZE,

        len(company_df),

        STRIDE
    ):

        start_idx = (
            end_idx - WINDOW_SIZE
        )

        sequence = feature_matrix[
            start_idx:end_idx
        ]

        residual_target = residuals[
            end_idx
        ]

        X_test.append(sequence)

        y_test.append(
            residual_target
        )

        catboost_test_predictions.append(

            predictions[end_idx]
        )

        actual_test_targets.append(

            targets[end_idx]
        )

        months_test.append(
            months[end_idx]
        )

        companies_test.append(
            company_id
        )

# ============================================================
# Convert To Arrays
# ============================================================

print("\nConverting To Arrays\n")

X_train = np.array(
    X_train,
    dtype=np.float32
)

y_train = np.array(
    y_train,
    dtype=np.float32
)

X_test = np.array(
    X_test,
    dtype=np.float32
)

y_test = np.array(
    y_test,
    dtype=np.float32
)

catboost_test_predictions = np.array(
    catboost_test_predictions,
    dtype=np.float32
)

actual_test_targets = np.array(
    actual_test_targets,
    dtype=np.float32
)

months_test = np.array(
    months_test
)

companies_test = np.array(
    companies_test
)

# ============================================================
# Final Safety Cleaning
# ============================================================

X_train = np.nan_to_num(

    X_train,

    nan=0.0,

    posinf=0.0,

    neginf=0.0
)

X_test = np.nan_to_num(

    X_test,

    nan=0.0,

    posinf=0.0,

    neginf=0.0
)

y_train = np.nan_to_num(

    y_train,

    nan=0.0,

    posinf=0.0,

    neginf=0.0
)

y_test = np.nan_to_num(

    y_test,

    nan=0.0,

    posinf=0.0,

    neginf=0.0
)

# ============================================================
# Save
# ============================================================

print("\nSaving Residual Sequences\n")

np.save(

    PROCESSED_FOLDER /
    "residual_X_train.npy",

    X_train
)

np.save(

    PROCESSED_FOLDER /
    "residual_y_train.npy",

    y_train
)

np.save(

    PROCESSED_FOLDER /
    "residual_X_test.npy",

    X_test
)

np.save(

    PROCESSED_FOLDER /
    "residual_y_test.npy",

    y_test
)

np.save(

    PROCESSED_FOLDER /
    "residual_catboost_test.npy",

    catboost_test_predictions
)

np.save(

    PROCESSED_FOLDER /
    "residual_actual_test.npy",

    actual_test_targets
)

np.save(

    PROCESSED_FOLDER /
    "residual_months_test.npy",

    months_test
)

np.save(

    PROCESSED_FOLDER /
    "residual_companies_test.npy",

    companies_test
)

# ============================================================
# Summary
# ============================================================

print("\n==============================")
print("RESIDUAL DATASET COMPLETE")
print("==============================")

print("\nTrain Shape:")

print(X_train.shape)

print("\nTest Shape:")

print(X_test.shape)

print("\nTrain Windows:")

print(len(X_train))

print("\nTest Windows:")

print(len(X_test))

print("\nUnique Test Companies:")

print(
    len(np.unique(companies_test))
)

print("\nNaN Checks:")

print(
    "Train:",
    np.isnan(X_train).sum()
)

print(
    "Test:",
    np.isnan(X_test).sum()
)