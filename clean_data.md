# CatBoost + Neural Network Pipeline for Monthly Return Prediction

## 1. Data Understanding

Each row represents a (company, month) observation.

Key columns:

* Target:

  * `monthly_gross_return`

* Time:

  * `Month`, `Year`, `Corrected_Year`, `Corrected_Month`

* Entity:

  * `co_code` (company ID)

* Numerical features:

  * `Momentum`, `log_ret`, `lag_ret`, `mktcap`, `price`, `BM_sep`, `OpProf`, `Inv`, etc.

* Lagged / derived features:

  * `lag_mv`, `lagged_assets`, `lagged_book_equity`, etc.

* Categorical features:

  * `Size_Label`, `BM_Label`, `OpProf_Label`, `Inv_Label`, `Mom_Label`

---

## 2. Data Cleaning

### 2.1 Sort and Index

Sort strictly by:

* `co_code`, then `Month`

Convert `Month` to datetime.

---

### 2.2 Handle Missing Values

Do NOT globally impute.

Instead:

* Keep missing values as NaN (CatBoost handles them)
* Add binary indicators:

  Example:

  * `is_missing_mktcap`
  * `is_missing_BM_sep`

This captures missingness as signal.

---

### 2.3 Fix Mixed Types

Column with mixed dtype (warning seen):

* Convert explicitly:

  * numeric → float
  * categorical → string

---

### 2.4 Remove Leakage Columns

Drop or avoid:

* Any feature computed using future data
* Any forward-looking aggregation

Keep only:

* past values
* current month factors

---

## 3. Feature Engineering

For each (j, t):

### 3.1 Lag Features

Ensure:

* `lag_ret` is valid
* Add more lags if needed:

  * `ret_t-1`, `ret_t-3`, `ret_t-6`

---

### 3.2 Rolling Features

Per company:

* rolling mean of returns
* rolling volatility

Example:

* 3-month mean
* 6-month std

---

### 3.3 Cross-sectional Features

Keep:

* `mktcap`
* `BM_sep`
* factor labels

These help model relative positioning.

---

### 3.4 Categorical Handling

Mark explicitly:

* `Size_Label`
* `BM_Label`
* `OpProf_Label`
* `Inv_Label`
* `Mom_Label`

These will be passed as categorical to CatBoost.

---

## 4. Train / Validation / Test Split

### 4.1 Time-based Split (MANDATORY)

Do NOT shuffle.

Example:

* Train: up to 2019
* Validation: 2020–2021
* Test: 2022+

Formally:

Train: t ≤ T1
Val: T1 < t ≤ T2
Test: t > T2

---

### 4.2 Cross-sectional Integrity

All companies can appear in all splits, but:

* Only future months go into validation/test

---

## 5. CatBoost Training

### 5.1 Input

X = all features except:

* `monthly_gross_return`
* identifiers (`co_code`, raw time if not encoded)

y = `monthly_gross_return`

---

### 5.2 Categorical Features

Pass explicitly:

* label columns (Size_Label, etc.)

---

### 5.3 Training Objective

Regression:

* loss = RMSE or MAE

---

### 5.4 Output
generate a latex report for this file
Model produces:

ŷ_CB(j,t)

---

## 6. Residual Construction (Critical Step)

Compute ONLY on validation (or OOS):

e(j,t) = y(j,t) − ŷ_CB(j,t)

Do NOT compute residuals on training set predictions.

---

## 7. Neural Network Training

### 7.1 Target

Train NN on:

e(j,t)

---

### 7.2 Inputs

Option A (simple):

* same X as CatBoost

Option B (better):

* X + temporal sequences (past returns window)

---

### 7.3 Architecture Suggestion

* MLP (baseline), or
* LSTM (if modeling temporal structure)

---

### 7.4 Output

NN predicts:

ê(j,t)

---

## 8. Final Prediction

Combine:

ŷ(j,t) = ŷ_CB(j,t) + ê(j,t)

---

## 9. Evaluation

Evaluate ONLY on test set:

* RMSE
* Rank correlation (important in finance)
* Directional accuracy

---

## 10. Common Failure Points

Avoid:

* Random train-test split
* Using future data in features
* Training NN on in-sample residuals
* Overfitting small residual signal

---

## 11. Recommended Extensions

* Add sector encoding (if available)
* Normalize features per month (cross-sectional z-score)
* Add interaction features (CatBoost can also learn them)

---

## 12. Mental Model

The system is:

Return = structured tabular signal (CatBoost) + residual signal (NN)

CatBoost handles:

* nonlinear tabular structure
* missing values
* categorical effects

NN handles:

* leftover temporal or latent structure

---
