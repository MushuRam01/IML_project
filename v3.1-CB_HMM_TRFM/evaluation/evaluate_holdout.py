# ============================================================
# evaluation/evaluate_holdout.py
# ============================================================
# Evaluates all residual architectures on Holdout companies and
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
HOLDOUT_ENGINEERED = PROCESSED_FOLDER / "holdout_engineered"
RESULTS_FOLDER = BASE_DIR / "results" / "rmse" / "holdout"

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

CATEGORICAL_COLUMNS = [
    "company_id",
    "Size_Label",
    "BM_Label",
    "OpProf_Label",
    "Inv_Label",
    "Mom_Label",
    "quarter",
]

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


def build_feature_matrix(df, feature_columns):
    X = df.drop(columns=[TARGET_COLUMN, "Month", "company_id"], errors="ignore").copy()
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        X = pd.concat([X, pd.DataFrame(0.0, index=X.index, columns=missing)], axis=1)
    X = X[feature_columns].copy()
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)
    return X.values.astype(np.float32)


def build_sequences(feature_matrix, window_size):
    sequences = []
    indices = []
    for end_idx in range(window_size, len(feature_matrix)):
        start_idx = end_idx - window_size
        sequences.append(feature_matrix[start_idx:end_idx])
        indices.append(end_idx)
    if len(sequences) == 0:
        return np.array([]), []
    return np.array(sequences, dtype=np.float32), indices

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
# Load Base CatBoost & Feature Schema
# ============================================================

catboost_model = CatBoostRegressor()
catboost_model.load_model(CHECKPOINT_FOLDER / "catboost_model.cbm")
catboost_feature_columns = list(catboost_model.feature_names_)

schema_df = pd.read_csv(PROCESSED_FOLDER / "engineered_features.csv", nrows=1, low_memory=False)
feature_columns = [c for c in schema_df.columns if c not in [TARGET_COLUMN, "Month", "company_id"]]

# ============================================================
# Load HMM Assets
# ============================================================

hmm_scaler = joblib.load(CHECKPOINT_FOLDER / "hmm_corrector_scaler.pkl")
hmm_model = joblib.load(CHECKPOINT_FOLDER / "hmm_corrector_model.pkl")

with open(CHECKPOINT_FOLDER / "hmm_corrector_features.json", "r") as f:
    hmm_features = json.load(f)
with open(CHECKPOINT_FOLDER / "hmm_corrector_mapping.json", "r") as f:
    hmm_mapping = json.load(f)

# ============================================================
# Load Residual Models
# ============================================================

stats = np.load(CHECKPOINT_FOLDER / "residual_norm_stats.npz")
mean = stats["mean"]
std = stats["std"]

num_features = len(feature_columns)
window_size = WINDOW_SIZE

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
# Evaluate Holdout
# ============================================================

holdout_files = sorted(HOLDOUT_ENGINEERED.glob("*.csv"))

all_actuals = []
all_catboost = []
all_hmm = []
all_tr = []
all_mlp = []
all_cnn = []
all_ridge = []
all_elastic = []
all_cb_res = []
all_months = []
all_companies = []

for csv_path in holdout_files:
    company_id = csv_path.stem
    df = pd.read_csv(csv_path, low_memory=False)

    if len(df) < WINDOW_SIZE + 1:
        continue

    df["Month"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Month").reset_index(drop=True)

    df["year"] = df["Month"].dt.year
    df["month_num"] = df["Month"].dt.month
    df["quarter"] = "Q" + df["Month"].dt.quarter.astype(str)

    # CatBoost baseline
    X_cb = df.drop(columns=[TARGET_COLUMN, "Month"], errors="ignore").copy()
    missing_cb = [c for c in catboost_feature_columns if c not in X_cb.columns]
    if missing_cb:
        X_cb = pd.concat([X_cb, pd.DataFrame(0.0, index=X_cb.index, columns=missing_cb)], axis=1)

    X_cb = X_cb[catboost_feature_columns].copy()
    cat_in_X = [c for c in CATEGORICAL_COLUMNS if c in X_cb.columns]
    num_in_X = [c for c in X_cb.columns if c not in cat_in_X]

    for col in num_in_X:
        X_cb[col] = pd.to_numeric(X_cb[col], errors="coerce")
    X_cb = X_cb.fillna(0.0)
    for col in cat_in_X:
        X_cb[col] = X_cb[col].fillna("").astype(str)

    catboost_predictions = catboost_model.predict(X_cb)

    # HMM corrections
    X_hmm = df.copy()
    for col in hmm_features:
        if col not in X_hmm.columns:
            X_hmm[col] = 0.0
    X_hmm = X_hmm[hmm_features].fillna(0.0)
    hmm_scaled = hmm_scaler.transform(X_hmm.values)
    inferred_states = hmm_model.predict(hmm_scaled)
    hmm_corrections = np.array([hmm_mapping[str(s)] for s in inferred_states])
    hmm_corrected = catboost_predictions + hmm_corrections

    # Residual model features
    feature_matrix = build_feature_matrix(df, feature_columns)

    mean_vec = mean[0, 0, :]
    std_vec = std[0, 0, :]
    feature_matrix = (feature_matrix - mean_vec) / std_vec
    feature_matrix = np.nan_to_num(feature_matrix)

    sequences, indices = build_sequences(feature_matrix, WINDOW_SIZE)
    if len(indices) == 0:
        continue

    X_flat = sequences.reshape(len(sequences), -1)

    res_tr = predict_torch(transformer, sequences)
    res_mlp = predict_torch(mlp, sequences)
    res_cnn = predict_torch(cnn, sequences)
    res_ridge = ridge_model.predict(X_flat)
    res_elastic = elastic_model.predict(X_flat)
    res_cb = residual_catboost.predict(X_flat)

    actuals = df[TARGET_COLUMN].values[indices]
    months = df["Month"].dt.strftime("%Y-%m-%d").values[indices]

    catboost_subset = catboost_predictions[indices]
    hmm_subset = hmm_corrected[indices]

    all_actuals.extend(actuals)
    all_catboost.extend(catboost_subset)
    all_hmm.extend(hmm_subset)
    all_tr.extend(catboost_subset + res_tr)
    all_mlp.extend(catboost_subset + res_mlp)
    all_cnn.extend(catboost_subset + res_cnn)
    all_ridge.extend(catboost_subset + res_ridge)
    all_elastic.extend(catboost_subset + res_elastic)
    all_cb_res.extend(catboost_subset + res_cb)
    all_months.extend(months)
    all_companies.extend([company_id] * len(actuals))

# ============================================================
# Save Predictions
# ============================================================

results_df = pd.DataFrame({
    "month": all_months,
    "company": [str(c) for c in all_companies],
    "actual": all_actuals,
    "catboost": all_catboost,
    "catboost_hmm": all_hmm,
    "catboost_transformer": all_tr,
    "catboost_mlp": all_mlp,
    "catboost_cnn": all_cnn,
    "catboost_ridge": all_ridge,
    "catboost_elasticnet": all_elastic,
    "catboost_residual": all_cb_res,
})

results_path = RESULTS_FOLDER / "predictions.csv"
results_df.to_csv(results_path, index=False)
print(f"Saved holdout predictions to {results_path}")

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
