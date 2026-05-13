# ============================================================
# training/train_residual_models.py
# ============================================================
# Trains multiple residual learners to correct CatBoost errors.
# Models: Transformer, MLP, CNN, Ridge, ElasticNet, CatBoost.
# ============================================================

from pathlib import Path
import os
import numpy as np
import joblib

from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_FOLDER = BASE_DIR / "processed"
CHECKPOINT_FOLDER = BASE_DIR / "checkpoints"
RESULTS_FOLDER = BASE_DIR / "results" / "residual_models"

os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ============================================================
# Device and Seeds
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# Hyperparameters
# ============================================================

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-4
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# ============================================================
# Load Residual Sequences
# ============================================================

X_train = np.load(PROCESSED_FOLDER / "residual_X_train.npy")
y_train = np.load(PROCESSED_FOLDER / "residual_y_train.npy")
X_test = np.load(PROCESSED_FOLDER / "residual_X_test.npy")
y_test = np.load(PROCESSED_FOLDER / "residual_y_test.npy")

catboost_test = np.load(PROCESSED_FOLDER / "residual_catboost_test.npy")
actual_test = np.load(PROCESSED_FOLDER / "residual_actual_test.npy")

print("\nLoaded Residual Sequences")
print(X_train.shape, X_test.shape)

# ============================================================
# Normalize Features (based on train set)
# ============================================================

mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True)
std[std < 1e-6] = 1.0

# Save stats for inference
np.savez(CHECKPOINT_FOLDER / "residual_norm_stats.npz", mean=mean, std=std)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)

# ============================================================
# Dataset
# ============================================================

class ResidualDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# Models
# ============================================================

class ResidualTransformer(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.embedding = nn.Linear(num_features, EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=128,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM, 32),
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
# Training Helpers
# ============================================================

def train_torch_model(model, train_loader, epochs, lr, name):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = np.inf

    print(f"\nTraining {name}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1:02d} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CHECKPOINT_FOLDER / f"{name}.pt")

    model.load_state_dict(torch.load(CHECKPOINT_FOLDER / f"{name}.pt", map_location=DEVICE))
    model.eval()
    return model


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


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ============================================================
# Train Torch Models
# ============================================================

num_features = X_train.shape[2]
window_size = X_train.shape[1]

train_dataset = ResidualDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

transformer = ResidualTransformer(num_features)
transformer = train_torch_model(transformer, train_loader, EPOCHS, LEARNING_RATE, "best_residual_transformer")

mlp = ResidualMLP(num_features, window_size)
mlp = train_torch_model(mlp, train_loader, EPOCHS, LEARNING_RATE, "residual_mlp")

cnn = ResidualCNN(num_features)
cnn = train_torch_model(cnn, train_loader, EPOCHS, LEARNING_RATE, "residual_cnn")

# ============================================================
# Train Linear Models
# ============================================================

X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0, random_state=42)),
])

elastic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42)),
])

print("\nTraining Ridge Regression...")
ridge_model.fit(X_train_flat, y_train)
joblib.dump(ridge_model, CHECKPOINT_FOLDER / "residual_ridge.pkl")

print("\nTraining ElasticNet Regression...")
elastic_model.fit(X_train_flat, y_train)
joblib.dump(elastic_model, CHECKPOINT_FOLDER / "residual_elasticnet.pkl")

# ============================================================
# Train Residual CatBoost
# ============================================================

print("\nTraining Residual CatBoost...")
residual_catboost = CatBoostRegressor(
    iterations=800,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    random_seed=42,
    verbose=100,
)
residual_catboost.fit(X_train_flat, y_train)
residual_catboost.save_model(CHECKPOINT_FOLDER / "residual_catboost.cbm")

# ============================================================
# Quick Test Metrics
# ============================================================

print("\n==============================")
print("RESIDUAL MODEL TEST RMSE")
print("==============================")

res_tr = predict_torch(transformer, X_test)
res_mlp = predict_torch(mlp, X_test)
res_cnn = predict_torch(cnn, X_test)
res_ridge = ridge_model.predict(X_test_flat)
res_elastic = elastic_model.predict(X_test_flat)
res_cb = residual_catboost.predict(X_test_flat)

models = {
    "Transformer": res_tr,
    "MLP": res_mlp,
    "CNN": res_cnn,
    "Ridge": res_ridge,
    "ElasticNet": res_elastic,
    "CatBoost": res_cb,
}

for name, residuals in models.items():
    corrected = catboost_test + residuals
    score = rmse(actual_test, corrected)
    print(f"{name:<12}: RMSE {score:.6f}")

print("\nSaved residual model checkpoints.")
