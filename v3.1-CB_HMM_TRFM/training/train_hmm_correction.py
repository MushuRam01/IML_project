# ============================================================
# training/train_hmm_correction.py
# ============================================================
# Trains a Gaussian Hidden Markov Model (HMM) on the features 
# from the Validation split. It then groups the CatBoost errors
# (residuals) by the inferred HMM regime and calculates the mean
# error for each state. This provides a "regime-aware" baseline
# residual correction.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import os
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_FOLDER = BASE_DIR / "processed"
RESULTS_FOLDER = BASE_DIR / "results"
CHECKPOINT_FOLDER = BASE_DIR / "checkpoints"

os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

# ============================================================
# Load Data
# ============================================================

print("\nLoading Validation Residuals...")
val_res_df = pd.read_csv(RESULTS_FOLDER / "validation_residuals.csv")
val_res_df["Month"] = pd.to_datetime(val_res_df["Month"])

print("Loading Engineered Features (for regime indicators)...")
features_df = pd.read_csv(
    PROCESSED_FOLDER / "engineered_features.csv",
    usecols=["Month", "company_id", "return_volatility_6", "return_momentum_6", "return_volatility_3", "return_momentum_3"],
    low_memory=False
)
features_df["Month"] = pd.to_datetime(features_df["Month"])

# ============================================================
# Merge Data
# ============================================================

df = val_res_df.merge(features_df, on=["Month", "company_id"], how="left")

# Select best available HMM features
HMM_FEATURES = []
if "return_volatility_6" in df.columns:
    HMM_FEATURES.append("return_volatility_6")
else:
    HMM_FEATURES.append("return_volatility_3")

if "return_momentum_6" in df.columns:
    HMM_FEATURES.append("return_momentum_6")
else:
    HMM_FEATURES.append("return_momentum_3")

print(f"\nUsing HMM Features: {HMM_FEATURES}")

# Drop NaNs for training the HMM mapping safely
df = df.dropna(subset=HMM_FEATURES + ["residual"]).copy()
X_features = df[HMM_FEATURES].values

# ============================================================
# Fit Scaler and HMM on Validation Split
# ============================================================

print(f"\nFitting StandardScaler and GaussianHMM on Validation Split (N={len(df)})...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

hmm_model = GaussianHMM(
    n_components=7,
    covariance_type="full",
    n_iter=200,
    random_state=42,
    verbose=False
)
hmm_model.fit(X_scaled)

# Infer the states for the validation set
df["hmm_state"] = hmm_model.predict(X_scaled)

# ============================================================
# Calculate State -> Residual Mappings
# ============================================================

print("\n==============================")
print("HMM STATE RESIDUAL MAPPING")
print("==============================")

state_mapping = {}

for state in range(hmm_model.n_components):
    state_mask = df["hmm_state"] == state
    state_residuals = df.loc[state_mask, "residual"]
    
    mean_res = float(state_residuals.mean())
    std_res = float(state_residuals.std())
    count = int(state_residuals.count())
    
    state_mapping[str(state)] = mean_res
    
    print(f"State {state}:")
    print(f"  Count         : {count}")
    print(f"  Mean Residual : {mean_res:.6f}")
    print(f"  Std Dev       : {std_res:.6f}")

# ============================================================
# Save Models and Mapping
# ============================================================

# Save the scaler, model, feature list, and mapping dictionary
joblib.dump(scaler, CHECKPOINT_FOLDER / "hmm_corrector_scaler.pkl")
joblib.dump(hmm_model, CHECKPOINT_FOLDER / "hmm_corrector_model.pkl")

with open(CHECKPOINT_FOLDER / "hmm_corrector_features.json", "w") as f:
    json.dump(HMM_FEATURES, f)

with open(CHECKPOINT_FOLDER / "hmm_corrector_mapping.json", "w") as f:
    json.dump(state_mapping, f)

print("\nSaved HMM Corrector assets successfully.")
