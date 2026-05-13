# ============================================================
# training/train_catboost.py
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import os

from catboost import (
    CatBoostRegressor,
    Pool
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_FOLDER = BASE_DIR / "processed"

CHECKPOINT_FOLDER = BASE_DIR / "checkpoints"

RESULTS_FOLDER = BASE_DIR / "results"

os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ============================================================
# Load Features
# ============================================================

df = pd.read_csv(

    PROCESSED_FOLDER /
    "engineered_features.csv",

    low_memory=False
)

print("\nLoaded Engineered Features\n")

print(df.shape)

# ============================================================
# Datetime
# ============================================================

df["Month"] = pd.to_datetime(
    df["Month"]
)

df = df.sort_values(
    "Month"
).reset_index(drop=True)

# ============================================================
# Company ID String
# ============================================================

df["company_id"] = df[
    "company_id"
].astype(str)

# ============================================================
# Date Features
# ============================================================

df["year"] = (
    df["Month"].dt.year
)

df["month_num"] = (
    df["Month"].dt.month
)

df["quarter"] = (
    "Q"
    +
    df["Month"].dt.quarter.astype(str)
)

# ============================================================
# Target
# ============================================================

TARGET_COLUMN = "monthly_gross_return"

# ============================================================
# Feature Columns
# ============================================================

IGNORE_COLUMNS = [

    TARGET_COLUMN,

    "Month"
]

feature_columns = [

    col

    for col in df.columns

    if col not in IGNORE_COLUMNS
]

# ============================================================
# STRICT TEMPORAL SPLIT
# ============================================================

train_df = df[
    df["Month"] < "2021-01-01"
].copy()

validation_df = df[

    (df["Month"] >= "2021-01-01")

    &

    (df["Month"] < "2022-01-01")
].copy()

# ------------------------------------------------------------
# GAP
# ------------------------------------------------------------

gap_df = df[

    (df["Month"] >= "2022-01-01")

    &

    (df["Month"] < "2022-07-01")
].copy()

test_df = df[
    df["Month"] >= "2022-07-01"
].copy()

print("\n==============================")
print("STRICT TEMPORAL SPLIT")
print("==============================")

print("\nTrain Range:")

print(
    train_df["Month"].min(),
    "->",
    train_df["Month"].max()
)

print("\nValidation Range:")

print(
    validation_df["Month"].min(),
    "->",
    validation_df["Month"].max()
)

print("\nGap Range:")

print(
    gap_df["Month"].min(),
    "->",
    gap_df["Month"].max()
)

print("\nTest Range:")

print(
    test_df["Month"].min(),
    "->",
    test_df["Month"].max()
)

print("\nTrain Shape:")

print(train_df.shape)

print("\nValidation Shape:")

print(validation_df.shape)

print("\nTest Shape:")

print(test_df.shape)

# ============================================================
# X / y
# ============================================================

X_train = train_df[
    feature_columns
].copy()

y_train = train_df[
    TARGET_COLUMN
]

X_val = validation_df[
    feature_columns
].copy()

y_val = validation_df[
    TARGET_COLUMN
]

X_test = test_df[
    feature_columns
].copy()

y_test = test_df[
    TARGET_COLUMN
]

# ============================================================
# Categorical Columns
# ============================================================

categorical_columns = [

    "company_id",

    "Size_Label",

    "BM_Label",

    "OpProf_Label",

    "Inv_Label",

    "Mom_Label",

    "quarter"
]

categorical_columns = [

    col

    for col in categorical_columns

    if col in X_train.columns
]

print("\nCategorical Columns:\n")

print(categorical_columns)

# ============================================================
# Numeric Conversion
# ============================================================

numeric_columns = [

    col

    for col in X_train.columns

    if col not in categorical_columns
]

for col in numeric_columns:

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_val[col] = pd.to_numeric(
        X_val[col],
        errors="coerce"
    )

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )

# ============================================================
# Fill Missing
# ============================================================

X_train = X_train.fillna(0.0)

X_val = X_val.fillna(0.0)

X_test = X_test.fillna(0.0)

# ============================================================
# Pools
# ============================================================

train_pool = Pool(

    X_train,

    y_train,

    cat_features=categorical_columns
)

val_pool = Pool(

    X_val,

    y_val,

    cat_features=categorical_columns
)

test_pool = Pool(

    X_test,

    y_test,

    cat_features=categorical_columns
)

# ============================================================
# Model
# ============================================================

model = CatBoostRegressor(

    iterations=3000,

    learning_rate=0.01,

    depth=8,

    loss_function="MAE",

    eval_metric="MAE",

    random_seed=42,

    verbose=100,

    od_type="Iter",

    od_wait=100
)

# ============================================================
# Train
# ============================================================

print("\nTraining CatBoost\n")

model.fit(

    train_pool,

    eval_set=val_pool,

    use_best_model=True
)

# ============================================================
# Save
# ============================================================

model.save_model(

    CHECKPOINT_FOLDER /
    "catboost_model.cbm"
)

# ============================================================
# Predictions
# ============================================================

val_predictions = model.predict(
    val_pool
)

test_predictions = model.predict(
    test_pool
)

# ============================================================
# Residuals
# ============================================================

val_residuals = (
    y_val.values
    -
    val_predictions
)

test_residuals = (
    y_test.values
    -
    test_predictions
)

# ============================================================
# Metrics
# ============================================================

print("\n==============================")
print("CATBOOST RESULTS")
print("==============================")

print("\nValidation MAE:")

print(
    mean_absolute_error(
        y_val,
        val_predictions
    )
)

print("\nTest MAE:")

print(
    mean_absolute_error(
        y_test,
        test_predictions
    )
)

print("\nValidation RMSE:")

print(
    np.sqrt(
        mean_squared_error(
            y_val,
            val_predictions
        )
    )
)

print("\nTest RMSE:")

print(
    np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )
)

print("\nDirectional Accuracy:")

print(
    np.mean(
        np.sign(test_predictions)
        ==
        np.sign(y_test.values)
    )
)

print("\nR2:")

print(
    r2_score(
        y_test,
        test_predictions
    )
)

# ============================================================
# Save Validation Residuals
# ============================================================

validation_results = pd.DataFrame({

    "Month":
        validation_df["Month"],

    "company_id":
        validation_df["company_id"],

    "target":
        y_val,

    "prediction":
        val_predictions,

    "residual":
        val_residuals
})

validation_results.to_csv(

    RESULTS_FOLDER /
    "validation_residuals.csv",

    index=False
)

# ============================================================
# Save Test Predictions
# ============================================================

test_results = pd.DataFrame({

    "Month":
        test_df["Month"],

    "company_id":
        test_df["company_id"],

    "target":
        y_test,

    "prediction":
        test_predictions,

    "residual":
        test_residuals
})

test_results.to_csv(

    RESULTS_FOLDER /
    "test_predictions.csv",

    index=False
)

print("\nSaved Results\n")