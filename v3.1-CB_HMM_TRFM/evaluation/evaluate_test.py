# ============================================================
# evaluation/evaluate_test.py
# ============================================================
# Evaluates all residual architectures on the Test split and
# saves consolidated predictions and RMSE summaries.
# ============================================================

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_FOLDER = BASE_DIR / "processed"
CHECKPOINT_FOLDER = BASE_DIR / "checkpoints"
RESULTS_FOLDER = BASE_DIR / "results" / "rmse" / "test"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ============================================================
# Constants
# ============================================================

TARGET_COLUMN = "monthly_gross_return"
WINDOW_SIZE = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "catboost": {"label": "CatBoost", "color": "#1f77b4", "linestyle": "--"},
    "catboost_hmm": {"label": "CatBoost + HMM", "color": "#2ca02c", "linestyle": "-."},
    "catboost_transformer": {"label": "CatBoost + Transformer", "color": "#ff7f0e", "linestyle": ":"},
    "catboost_mlp": {"label": "CatBoost + MLP", "color": "#8c564b", "linestyle": "-"},
    "catboost_cnn": {"label": "CatBoost + CNN", "color": "#9467bd", "linestyle": "-"},
    "catboost_ridge": {"label": "CatBoost + Ridge", "color": "#17becf", "linestyle": "-"},
    "catboost_elasticnet": {"label": "CatBoost + ElasticNet", "color": "#bcbd22", "linestyle": "-"},
    "catboost_residual": {"label": "CatBoost + Residual CatBoost", "color": "#e377c2", "linestyle": "-"},
}

# ============================================================
# Helpers
# ============================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def predict_torch(model, X, batch_size=1024):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(DEVICE)
            pred = model(batch).cpu().numpy().reshape(-1)
            preds.append(pred)
    if len(preds) == 0:
        return np.array([])
    return np.concatenate(preds).reshape(-1)

# ============================================================
# Model Definitions (must match training)
# ============================================================

class ResidualTransformer(nn.Module):
    def __init__(self, num_features, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(num_features, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x.squeeze(-1)


class ResidualMLP(nn.Module):
    def __init__(self, num_features, window_size):
        super().__init__()
        input_dim = num_features * window_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.net(x)
        return x.squeeze(-1)


class ResidualCNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.squeeze(-1)
        x = self.head(x)
        return x.squeeze(-1)

# ============================================================
# Load Residual Sequences
# ============================================================

X_test = np.load(PROCESSED_FOLDER / "residual_X_test.npy")
catboost_test = np.load(PROCESSED_FOLDER / "residual_catboost_test.npy")
actual_test = np.load(PROCESSED_FOLDER / "residual_actual_test.npy")
months_test = np.load(PROCESSED_FOLDER / "residual_months_test.npy")
companies_test = np.load(PROCESSED_FOLDER / "residual_companies_test.npy")

# ============================================================
# Normalize Sequences
# ============================================================

stats = np.load(CHECKPOINT_FOLDER / "residual_norm_stats.npz")
mean = stats["mean"]
std = stats["std"]

X_test = (X_test - mean) / std
X_test = np.nan_to_num(X_test)

# ============================================================
# Load Models
# ============================================================

num_features = X_test.shape[2]
window_size = X_test.shape[1]

transformer = ResidualTransformer(num_features).to(DEVICE)
transformer.load_state_dict(torch.load(CHECKPOINT_FOLDER / "best_residual_transformer.pt", map_location=DEVICE))
transformer.eval()

mlp = ResidualMLP(num_features, window_size).to(DEVICE)
mlp.load_state_dict(torch.load(CHECKPOINT_FOLDER / "residual_mlp.pt", map_location=DEVICE))
mlp.eval()

cnn = ResidualCNN(num_features).to(DEVICE)
cnn.load_state_dict(torch.load(CHECKPOINT_FOLDER / "residual_cnn.pt", map_location=DEVICE))
cnn.eval()

ridge_model = joblib.load(CHECKPOINT_FOLDER / "residual_ridge.pkl")
elastic_model = joblib.load(CHECKPOINT_FOLDER / "residual_elasticnet.pkl")

residual_catboost = CatBoostRegressor()
residual_catboost.load_model(CHECKPOINT_FOLDER / "residual_catboost.cbm")

# ============================================================
# Predict Residuals (Test)
# ============================================================

X_test_flat = X_test.reshape(len(X_test), -1)

res_tr = predict_torch(transformer, X_test)
res_mlp = predict_torch(mlp, X_test)
res_cnn = predict_torch(cnn, X_test)
res_ridge = ridge_model.predict(X_test_flat)
res_elastic = elastic_model.predict(X_test_flat)
res_cb = residual_catboost.predict(X_test_flat)

# ============================================================
# HMM Corrections (Test)
# ============================================================

features_df = pd.read_csv(
    PROCESSED_FOLDER / "engineered_features.csv",
    usecols=["Month", "company_id", "return_volatility_6", "return_momentum_6", "return_volatility_3", "return_momentum_3"],
    low_memory=False,
)
features_df.rename(columns={"Month": "month", "company_id": "company"}, inplace=True)
features_df["month"] = pd.to_datetime(features_df["month"])
features_df = features_df.drop_duplicates(subset=["month", "company"], keep="first").reset_index(drop=True)

with open(CHECKPOINT_FOLDER / "hmm_corrector_features.json", "r") as f:
    hmm_features = json.load(f)
with open(CHECKPOINT_FOLDER / "hmm_corrector_mapping.json", "r") as f:
    hmm_mapping = json.load(f)

hmm_scaler = joblib.load(CHECKPOINT_FOLDER / "hmm_corrector_scaler.pkl")
hmm_model = joblib.load(CHECKPOINT_FOLDER / "hmm_corrector_model.pkl")

base_df = pd.DataFrame({
    "month": pd.to_datetime(months_test),
    "company": companies_test,
    "actual": actual_test,
    "catboost": catboost_test,
})

base_df = base_df.merge(features_df, on=["month", "company"], how="left")
if len(base_df) != len(months_test):
    raise ValueError(
        f"Feature merge expanded rows: base={len(months_test)} merged={len(base_df)}"
    )

for col in hmm_features:
    if col not in base_df.columns:
        base_df[col] = 0.0

X_hmm = base_df[hmm_features].fillna(0.0)
hmm_scaled = hmm_scaler.transform(X_hmm.values)

inferred_states = hmm_model.predict(hmm_scaled)
hmm_corrections = np.array([hmm_mapping[str(s)] for s in inferred_states])

# ============================================================
# Build Predictions DataFrame
# ============================================================

results_df = pd.DataFrame({
    "month": base_df["month"].dt.strftime("%Y-%m-%d"),
    "company": base_df["company"].astype(str),
    "actual": base_df["actual"].values,
    "catboost": base_df["catboost"].values,
    "catboost_hmm": base_df["catboost"].values + hmm_corrections,
    "catboost_transformer": base_df["catboost"].values + res_tr,
    "catboost_mlp": base_df["catboost"].values + res_mlp,
    "catboost_cnn": base_df["catboost"].values + res_cnn,
    "catboost_ridge": base_df["catboost"].values + res_ridge,
    "catboost_elasticnet": base_df["catboost"].values + res_elastic,
    "catboost_residual": base_df["catboost"].values + res_cb,
})

results_path = RESULTS_FOLDER / "predictions.csv"
results_df.to_csv(results_path, index=False)
print(f"Saved test predictions to {results_path}")

# ============================================================
# RMSE Summaries
# ============================================================

model_cols = [k for k in MODEL_CONFIG.keys() if k in results_df.columns]

company_rows = []
for company, group in results_df.groupby("company"):
    entry = {"company": company, "n_months": len(group)}
    for col in model_cols:
        entry[f"{col}_rmse"] = rmse(group["actual"].values, group[col].values)
    company_rows.append(entry)

company_df = pd.DataFrame(company_rows)
company_path = RESULTS_FOLDER / "company_rmse.csv"
company_df.to_csv(company_path, index=False)
print(f"Saved company RMSE to {company_path}")

summary_rows = []
for col in model_cols:
    summary_rows.append({
        "model": col,
        "global_rmse": rmse(results_df["actual"].values, results_df[col].values),
        "min_company_rmse": company_df[f"{col}_rmse"].min(),
        "max_company_rmse": company_df[f"{col}_rmse"].max(),
        "mean_company_rmse": company_df[f"{col}_rmse"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = RESULTS_FOLDER / "rmse_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved RMSE summary to {summary_path}")
