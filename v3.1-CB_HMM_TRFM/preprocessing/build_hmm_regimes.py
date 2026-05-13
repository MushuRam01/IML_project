# ============================================================
# preprocessing/build_hmm_regimes.py
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import os
import json

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_FOLDER = BASE_DIR / "processed"
CHECKPOINT_FOLDER = BASE_DIR / "checkpoints"

os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

# ============================================================
# Load Engineered Features
# ============================================================

print("\nLoading Engineered Features...")
df = pd.read_csv(
    PROCESSED_FOLDER / "engineered_features.csv",
    low_memory=False
)

print(f"Original shape: {df.shape}")

# ============================================================
# Sort Chronologically
# ============================================================

df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

# ============================================================
# Programmatic Feature Selection
# ============================================================
# We robustly search for features matching the concepts of
# target return, volatility, and momentum.
# ============================================================

def find_column(df, keywords):
    """Find the first column that contains all keywords."""
    for col in df.columns:
        if all(kw in col for kw in keywords):
            return col
    return None

HMM_FEATURES = []

# 1. Target Return
target_col = find_column(df, ["monthly_gross_return"])
if target_col and target_col not in HMM_FEATURES:
    HMM_FEATURES.append(target_col)

# 2. Volatility (prefer 6-month, fallback to 3-month)
vol_col = find_column(df, ["return_volatility_6"])
if not vol_col:
    vol_col = find_column(df, ["return_volatility_3"])
if vol_col and vol_col not in HMM_FEATURES:
    HMM_FEATURES.append(vol_col)

# 3. Momentum (prefer 6-month, fallback to 3-month)
mom_col = find_column(df, ["return_momentum_6"])
if not mom_col:
    mom_col = find_column(df, ["return_momentum_3"])
if mom_col and mom_col not in HMM_FEATURES:
    HMM_FEATURES.append(mom_col)

print(f"\nDiscovered HMM Features:\n{HMM_FEATURES}")

if len(HMM_FEATURES) < 3:
    raise ValueError("Could not find all required HMM feature concepts.")

# Save feature list for holdout evaluation
with open(CHECKPOINT_FOLDER / "hmm_features.json", "w") as f:
    json.dump(HMM_FEATURES, f)

# ============================================================
# Build Matrix
# ============================================================

X_hmm_df = df[HMM_FEATURES].copy()
X_hmm_df = X_hmm_df.replace([np.inf, -np.inf], np.nan)
X_hmm_df = X_hmm_df.fillna(0.0)

# ============================================================
# Strict Temporal Split (Train Only for Fitting)
# ============================================================
# MUST match train_catboost.py:
# Train: < 2021-01-01
# ============================================================

train_mask = df["Month"] < "2021-01-01"

X_train_df = X_hmm_df[train_mask]
X_train = X_train_df.values
X_all = X_hmm_df.values

print(f"\nTrain shape for HMM fitting: {X_train.shape}")

# ============================================================
# Normalize
# ============================================================

scaler = StandardScaler()

# Fit scaler ONLY on train data to prevent leakage
X_train_scaled = scaler.fit_transform(X_train)

# Transform ALL data using train scaler
X_all_scaled = scaler.transform(X_all)

# ============================================================
# Train HMM
# ============================================================

print("\nTraining HMM...")

hmm_model = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=200,
    random_state=42,
    verbose=False
)

# Fit ONLY on train data
hmm_model.fit(X_train_scaled)

# ============================================================
# Infer Hidden States
# ============================================================

hidden_states = hmm_model.predict(X_all_scaled)

# Add to dataframe as a categorical string
df["hmm_state"] = hidden_states.astype(str)

# ============================================================
# State Diagnostics
# ============================================================

print("\n==============================")
print("HMM STATE DISTRIBUTION (ALL DATA)")
print("==============================")
print(df["hmm_state"].value_counts().sort_index())

print("\n==============================")
print("HMM TRANSITION MATRIX")
print("==============================")
print(np.round(hmm_model.transmat_, 3))

print("\n==============================")
print("MEAN FEATURES PER REGIME (TRAIN SCALE)")
print("==============================")
for i in range(hmm_model.n_components):
    print(f"\nState {i}:")
    for j, feature in enumerate(HMM_FEATURES):
        print(f"  {feature}: {hmm_model.means_[i, j]:.4f}")

# ============================================================
# Save Updated Features
# ============================================================

output_path = PROCESSED_FOLDER / "engineered_features_hmm.csv"

print(f"\nSaving augmented features to: {output_path.name}")
df.to_csv(output_path, index=False)

# ============================================================
# Save HMM + Scaler
# ============================================================

joblib.dump(hmm_model, CHECKPOINT_FOLDER / "hmm_model.pkl")
joblib.dump(scaler, CHECKPOINT_FOLDER / "hmm_scaler.pkl")

print("\nSaved HMM Model, Scaler, and Features schema successfully.\n")