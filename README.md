# Mechanisms of Action (MoA) — Final Report

## Data Insights

### Distribution of Gene (g-) and Cell (c-) Features

* Both **g-** and **c-features** follow approximately **normal, bell-shaped distributions**.
  → Indicates biological data were pre-normalized but require minor scaling for stable model convergence.

### Train vs Test Comparisons

* **cp_dose:** Balanced across datasets → no stratification needed.
* **cp_time:** All three intervals (24h, 48h, 72h) appear in similar proportions; **48h slightly dominates (~8500 samples)** — a realistic biological condition.
* **cp_type:** Control samples (`ctl_vehicle`) comprise **<10%** of total and have **no active MoA targets** → excluded from training.

### Drug Frequency Distribution

* Heavy **imbalance**: few drugs occur >1500 times, most appear only a few dozen.
  → **Use `GroupKFold(by drug_id)`** to prevent leakage between folds.

### Feature Distributions

* Histograms for individual **g-** and **c-features** show near-normal shapes but some skewness.
  → **Scaling (QuantileTransformer or StandardScaler)** required.

### Correlation Heatmaps

* **g-features:** strong internal clusters.
* **c-features:** moderate correlation internally, weakly linked to g-features.
  → **PCA justified** to decorrelate features and reduce dimensionality.

### Target Imbalance

* Frequent MoAs (e.g., `nfkb_inhibitor`, `proteasome_inhibitor`) dominate loss.
* Many MoAs have very few positives → **`MultilabelStratifiedKFold`** needed to balance rare labels.

### PCA & UMAP Visualizations

* **PCA:** 24h samples form distinct clusters; 48h & 72h partially overlap → shows **time-dependent biological gradients**.
* **UMAP:** D1 & D2 doses overlap, but D2 (higher dose) clusters with lower cell viability.
  → **`cp_time`** and **`cp_dose`** are biologically meaningful and must be included.


## Feature Engineering

### Exclusion of Control Samples

* Removed all entries with `cp_type == ctl_vehicle` to avoid noise.

### Scaling

* Differences in value ranges between **g-** and **c-features** handled with:

  * `QuantileTransformer` or
  * `StandardScaler`

### Categorical Encoding

* **cp_time** and **cp_dose** → one-hot encoded to retain discrete biological stages.

### Dimensionality Reduction

* **PCA applied separately:**

  * 600 components for g-features
  * 60 components for c-features
    → Combined total: **662 features**

### Cross-Validation Scheme

* Combined approach:

  * `GroupKFold(by drug_id)` → avoids leakage
  * `MultilabelStratifiedKFold` → preserves rare-label balance

## Modeling Approaches

### 1. Multilayer Perceptron (MLP)

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Optimizer     | Adam                            |
| Learning rate | 1e-3                            |
| Batch size    | 256                             |
| Epochs        | 50                              |
| Dropout       | 0.2–0.4                         |
| Weight decay  | 1e-5                            |
| Loss          | BCEWithLogitsLoss               |
| Activation    | ReLU (hidden), Sigmoid (output) |

**Architecture:**

Input (662 features)
→ Dense(1024) + BatchNorm + FeatureGate + ReLU + Dropout(0.4)
→ Dense(512)  + BatchNorm + FeatureGate + ReLU + Dropout(0.3)
→ Dense(256)  + BatchNorm + ReLU + Dropout(0.2)
→ Output(206) + Sigmoid
```

**Results:**

| Model             | Private LogLoss | Public LogLoss |
| ----------------- | --------------- | -------------- |
| MLP (FeatureGate) | **0.01704**     | **0.01947**    |

→ Minimal public-private gap (~0.0024) confirms strong generalization.


### 2. XGBoost

**Features:** Same as MLP (Quantile-scaled + PCA).

```python
xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    tree_method='gpu_hist'
)
```

**Results:**

| Model   | Private LogLoss | Public LogLoss |
| ------- | --------------- | -------------- |
| XGBoost | 0.0208          | 0.0221         |

→ Slightly weaker on rare labels due to imbalance.


### 3. Blending (Level 1 Ensemble)

**Method:** Weighted average
`final_preds = 0.7 * mlp_preds + 0.3 * xgb_preds`

| Model                 | Private LogLoss | Public LogLoss |
| --------------------- | --------------- | -------------- |
| MLP (FeatureGate)     | 0.0170          | 0.0195         |
| XGBoost               | 0.0208          | 0.0221         |
| **Blend (MLP + XGB)** | **0.01841**     | **0.02060**    |

→ Blending smooths fold variance and reduces LogLoss.


### 4. Stacking (Level 2 Meta-Model)

**Composition:**

* MLP (FeatureGate)
* XGBoost
* Blending layer (MLP + XGB)

Their **OOF predictions** → input for **Logistic Regression meta-learner**.

| Model                           | Private LogLoss | Public LogLoss |
| ------------------------------- | --------------- | -------------- |
| Blend (MLP + XGB)               | 0.01841         | 0.02060        |
| **Stacking (MLP + XGB + LGBM)** | **0.01713**     | **0.01930**    |

→ Stacking improved validation stability and reduced LogLoss by ~7%.


## ⚖️ Adversarial Validation

* ROC AUC: **0.57** → mild train-test difference.
* Indicates **slight feature distribution shift**, potentially from biological or batch effects.
* Adjusting sampling or rebalancing may further improve model generalization.
