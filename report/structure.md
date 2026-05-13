# structure.md

## Title Page

- Project title
- Team members
- Institution / course
- Date
- Repository link (if applicable)

Keep the title page minimal and readable.

---

# Abstract

Length: 1 short paragraph.

Goal:
- State the prediction task clearly.
- Mention the core modeling approach.
- Mention the main engineering decisions.
- Mention the best-performing result very briefly.
- Mention the major limitation or challenge.

Do not:
- Add equations.
- Add implementation details.
- Add literature review.

The abstract should allow an evaluator to understand the entire project in under 30 seconds.

---

# 1. Introduction

## Purpose

Explain:
- What the prediction problem is.
- Why predicting monthly gross return matters.
- Why the problem is difficult.

Keep this practical rather than theoretical.

## Problem Formulation

Define:
- Input:
  - company-month spectral and financial features
  - limited historical window (up to 6 months)
- Output:
  - next monthly gross return

Mention:
- panel structure
- temporal nature
- sparsity
- distribution shift
- leakage risks

Do not yet discuss model architecture in detail.

---

# 2. Design Philosophy and High-Level Pipeline

This section is extremely important.

The evaluator should understand:
- why the pipeline exists,
- why each stage exists,
- why simpler approaches were insufficient.

## Pipeline Overview

Include:
- a single clean end-to-end pipeline diagram

Recommended flow:

Raw panel data  
→ cleaning and canonicalization  
→ missingness handling  
→ feature engineering  
→ CatBoost baseline  
→ residual modeling  
→ evaluation and diagnostics

## Design Goals

Explain the motivations behind:
- chronological integrity
- explicit leakage prevention
- interpretability
- robustness to sparse financial data
- modular experimentation
- reproducibility

This section should read like engineering reasoning rather than marketing.

---

# 3. Dataset and Data Engineering

## Dataset Description

Describe:
- company-month structure
- number of rows
- number of companies
- temporal range
- target variable

Avoid excessive dataset statistics.

## Data Cleaning

Explain:
- duplicate handling
- canonicalization
- missing values
- infinity suppression
- historical median imputation

Explain *why* each step exists.

## Missingness as Signal

Very important section.

Explain:
- why missing values may contain information
- why missingness flags were added
- why this matters in financial panels

## Target Transformation

Explain:
- why log-transforming returns stabilizes variance
- why this improves optimization and modeling stability

Keep this intuitive.

---

# 4. Feature Engineering

This should be one of the clearest sections in the report.

## Feature Families

Organize into subsections:

### Lag Features
Explain delayed market effects.

### Rolling Statistics
Explain local trend and volatility estimation.

### Momentum and Volatility Signals
Explain regime-sensitive dynamics.

### Categorical Encodings
Explain:
- deciles
- bins
- ordered target statistics
- why CatBoost benefits from them

## Feature Traceability

Emphasize:
- every feature is deterministic
- all features depend only on past information
- reproducibility and auditability

This section should convince the evaluator that the feature pipeline is principled.

---

# 5. Modeling Approach

Structure this section from simplest to most advanced.

---

## 5.1 Baseline: CatBoost

Explain:
- why CatBoost was chosen
- why tree ensembles work well for sparse tabular data
- handling of:
  - missing values
  - nonlinear interactions
  - categorical variables

Do not over-explain boosting mathematics.

Only include minimal equations if necessary.

Focus on practical justification.

---

## 5.2 Residual Learning Strategy

This section should explain the core project insight.

Explain:
- why predicting residuals can help
- how residual models correct systematic baseline errors
- why residual correction is safer than replacing the baseline entirely

Then explain:
- validation gating
- chronological residual training
- leakage prevention

---

## 5.3 Regime-Aware and Sequence-Aware Extensions

Split into two subsections.

### HMM Residual Corrector

Explain:
- market regimes
- latent states
- volatility-conditioned corrections

Focus on intuition.

### Transformer Residual Corrector

Explain:
- sequence modeling
- temporal context
- trailing windows
- attention over historical patterns

Be honest about limitations and instability if results were weak.

---

# 6. Experimental Protocol

This section must be extremely clean and easy to skim.

## Chronological Splits

Use tables.

Clearly show:
- train
- validation
- holdout
- test
- external/OOD evaluation

Explain:
- why chronological isolation matters
- why random splitting would leak future information

## Experimental Variants

Clearly separate:
- v3.0 pipeline
- v3.1 pipeline

Explain:
- differences in residual training
- differences in evaluation logic
- why results are not directly comparable

This avoids evaluator confusion.

---

# 7. Evaluation Metrics

Keep concise.

Define:
- RMSE
- MAE
- R²
- directional accuracy
- rank correlation / IC

Briefly explain:
- what each metric captures
- why multiple metrics are necessary

Do not spend more than 1 page here.

---

# 8. Results

This section should prioritize readability over volume.

The evaluator should immediately see:
- what worked,
- what failed,
- what improved performance.

---

## 8.1 Internal Evaluation

Include:
- compact summary table
- baseline vs corrected models
- key improvements

Then include:
- 1–2 important plots only

Do not overload with redundant graphs.

---

## 8.2 External / OOD Evaluation

This section is critical.

Explain:
- distribution shift
- degradation outside training distribution
- why external testing matters

Be transparent.

Strong honesty improves credibility.

---

## 8.3 Company-Level Examples

Use:
- best-case example
- typical example
- failure example

Explain:
- what the model captures correctly
- where it breaks

Keep discussion visual and intuitive.

---

# 9. Error Analysis

This section should feel analytical rather than defensive.

Focus on:
- abrupt regime shifts
- sparse sequence limitations
- external generalization failure
- instability during rare events
- bias toward mean behavior

Mention:
- what was learned from failures
- how this informed later design decisions

---

# 10. Implementation and Repository Structure

This section should help evaluators navigate quickly.

Include:
- preprocessing scripts
- training scripts
- evaluation scripts
- dashboard generation
- pipeline entry points

Use compact directory trees or tables.

Do not dump full file structures.

---

# 11. Limitations

Be explicit and honest.

Possible topics:
- limited sequence depth
- insufficient external robustness
- sparse labels
- temporal instability
- lack of causal structure
- computational tradeoffs

Avoid overstating success.

---

# 12. Future Work

Focus on:
- domain adaptation
- regime-conditioned recalibration
- richer sequence modeling
- positional encoding
- multimodal integration
- uncertainty estimation
- factor-neutral evaluation

Keep realistic.

---

# 13. Conclusion

Length: short.

Summarize:
- the pipeline design
- the major engineering insight
- the strongest result
- the key limitation

End with:
- what the project demonstrated technically,
- not exaggerated claims about finance or prediction.

---

# Writing Guidelines

## Tone

The report should:
- sound technical and deliberate
- prioritize clarity over sophistication
- avoid exaggerated claims
- avoid buzzwords
- avoid storytelling language

The voice should feel:
- reproducible
- engineering-oriented
- experimentally grounded

---

## Readability Rules

The evaluator may read dozens of reports.

Therefore:
- keep sections visually clean
- use short paragraphs
- prefer tables over dense prose
- avoid unnecessary equations
- every figure must have a purpose
- every section must answer:
  - why was this done?
  - what did it achieve?
  - what limitation remained?

---

## Figure Rules

Every figure should:
- support a claim
- have readable titles
- have readable axes
- be referenced in nearby text

Avoid:
- decorative plots
- redundant plots
- tiny unreadable dashboards

---

## Results Presentation Rules

Always:
- compare against a baseline
- discuss both strengths and failures
- separate internal vs external performance
- explain whether improvements are statistically meaningful or merely cosmetic

Do not:
- hide weak results
- cherry-pick examples only

---

## Final Report Goal

The evaluator should finish the report understanding:

1. What problem was solved.
2. Why the pipeline was designed this way.
3. How leakage and temporal integrity were handled.
4. Why CatBoost was effective.
5. Why residual correction was explored.
6. What worked and what failed.
7. What technical lessons were learned.

The report should optimize for:
- clarity,
- structure,
- reproducibility,
- and engineering reasoning.
