# CatBoost + CNN Residual Learning Pipeline (GPU-Optimized, Notebook-Oriented)

This document defines a complete, leakage-safe, GPU-aware training pipeline for predicting monthly returns using CatBoost followed by a CNN trained on residuals. Every concept is defined as it appears.

---

# 0. Core Objective

We learn a function over panel data:

y(j,t) = return of company j at time t

We approximate:

y(j,t) ≈ f_CB(x(j,t)) + f_CNN(sequence(j,t))

where:

* f_CB = CatBoost model (tabular, static)
* f_CNN = temporal residual corrector

---

# 1. Definitions

### 1.1 Feature Function

x(j,t) = φ(raw data up to time t)

Definition: φ is a deterministic transformation that uses only past and current information. It must never use future values.

---

### 1.2 Residual

e(j,t) = y(j,t) − f_CB(x(j,t))

Definition: residual is the error of the first model. The second model learns this.

---

### 1.3 Sequence Tensor

X_seq(j,t) = [x(j,t−k), ..., x(j,t)]

Definition: a fixed-length window of past feature vectors. This introduces temporal structure required by CNN.

---

# 2. Notebook Structure (IPYNB)

Organize notebook into strict sections:

1. Imports + Config
2. Data Load
3. Feature Construction
4. Split Verification
5. CatBoost Training
6. Residual Computation
7. CNN Dataset Construction
8. CNN Training
9. Evaluation + Metrics
10. Inference Pipeline

---

# 3. Data Loading

* Load train, validation, test CSVs
* Concatenate temporarily for feature generation
* Sort by (co_code, Month)

Convert:

* Month → datetime
* categorical columns → string

---

# 4. Feature Construction (φ)

### 4.1 Lag Features

For each company:

* lag_ret_1, lag_ret_3, lag_ret_6

Definition: lag means shifting values backward in time.

---

### 4.2 Rolling Features

Compute per company:

* rolling_mean_k
* rolling_std_k

Definition: rolling statistic = statistic over past k observations.

---

### 4.3 Missing Indicators

For each feature f:

is_missing_f = 1 if f is NaN else 0

---

### 4.4 Categorical Features

Explicitly mark:

* Size_Label
* BM_Label
* OpProf_Label
* Inv_Label
* Mom_Label

Definition: categorical feature = discrete label with no numeric meaning.

---

### 4.5 Final Step

After feature construction:

* Slice back into train / val / test

---

# 5. CatBoost Training (GPU)

### 5.1 Model Definition

Use:

* task_type = "GPU"
* loss_function = RMSE
* depth = 6–10
* learning_rate = 0.03–0.1
* iterations = large (1000+)

Definition: gradient boosting = iterative model where each step fits residual errors.

---

### 5.2 Training

Train only on TRAIN set.

Validate on VAL set.

Log:

* RMSE
* MAE

---

### 5.3 Prediction

Compute:

y_hat_CB(j,t) for:

* validation
* test

---

# 6. Residual Computation

Compute ONLY on validation:

e(j,t) = y(j,t) − y_hat_CB(j,t)

Store residuals aligned with rows.

---

# 7. CNN Dataset Construction

### 7.1 Sequence Length

Choose k = 6 or 12

---

### 7.2 Tensor Construction

For each (j,t):

X_seq(j,t) = stack of past k feature vectors

Shape:

(batch_size, channels=d, length=k)

---

### 7.3 Optional Augmentation

Append CatBoost predictions:

channel += y_hat_CB history

---

# 8. CNN Model (GPU via PyTorch)

### 8.1 Architecture

* Conv1D (channels → 64)
* ReLU
* Conv1D (64 → 32)
* ReLU
* Global Average Pooling
* Linear → 1 output

Definition: convolution = local weighted aggregation over neighboring time steps.

---

### 8.2 Training

Target:

e(j,t)

Loss:

MSE

Optimizer:

Adam (GPU)

---

# 9. Metrics (EXTREMELY IMPORTANT)

Compute across TRAIN / VAL / TEST:

---

## 9.1 Regression Metrics

* RMSE
* MAE
* R²

Definition: R² measures proportion of variance explained.

---

## 9.2 Finance-Specific Metrics

### (a) Rank Correlation (Spearman)

Measures ordering correctness across companies at time t.

---

### (b) Information Coefficient (IC)

IC(t) = corr(rank(y_true), rank(y_pred)) per month

Track:

* mean IC
* IC std

---

### (c) Directional Accuracy

fraction of correct signs:

sign(y_true) == sign(y_pred)

---

### (d) Top-K Performance

For each month:

* pick top K predicted returns
* compute actual return

---

### (e) Sharpe Ratio (proxy)

mean(predicted strategy returns) / std

---

## 9.3 Residual Diagnostics

Check:

* autocorrelation of residuals
* variance reduction after CNN

---

# 10. Training Logs

Log per epoch / iteration:

CatBoost:

* train loss
* val loss

CNN:

* train loss
* val loss

Also log:

* IC over time
* RMSE over time

---

# 11. Inference Pipeline

Given new data at time t:

1. Build x(j,t)
2. Predict:

y_hat_CB

1. Build sequence:

X_seq(j,t)

1. Predict residual:

e_hat

1. Final:

y_hat = y_hat_CB + e_hat

---

# 12. GPU Utilization

* CatBoost: task_type="GPU"
* PyTorch: model.to("cuda")
* DataLoader with pin_memory=True

---

# 13. Failure Checks

Verify:

* No future leakage in φ
* Residuals computed out-of-sample
* CNN input strictly causal
* Stable validation loss

---

# 14. Final System Interpretation

Return(j,t) = structured tabular signal + temporal correction

CatBoost captures:

* cross-sectional structure
* nonlinear feature interactions

CNN captures:

* temporal patterns
* systematic residual errors

---

# 15. Minimal Execution Flow

Load → Build Features → Train CB → Predict → Compute Residuals → Build Sequences → Train CNN → Evaluate → Deploy

---
