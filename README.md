# Data insights

Diistribution of all g- (gene) and c- (cell) traits: Both types of traits have a bell-shaped distribution close to normal.
Train vs Test: cp_dose: The variable cp_dose is balanced and can be used without stratification.
Train vs Test: cp_time: All three time intervals are present in similar proportions in train and test. 48h has a slightly higher proportion in train (~8500 samples), which corresponds to realistic biological conditions.
Train vs Test: cp_type: Control samples (ctl_vehicle) make up only a small fraction (<10%). Also, all targets have ctl_vehicle = 0, so they only interfere with model training.
Distribution of drug frequencies in train set: There is a significant imbalanc, several drugs appear more than 1,500 times, while most appear only a few dozen times. Conclusion: GroupKFold must be applied by drug_id to prevent information leakage between folds.
Histogram by individual g-features &  c-features: Most have a normal distribution, but some have shifts or asymmetries, requiring scaling for a stable model.
Correlation heatmap: g-features form strongly correlated clusters among themselves. c-features also have internal correlation, but a weaker connection with g-. There is low intergroup correlation between g- and c-, which means that these types of data are complementary. Data correlates => can apply PCA
Highest number of positive samples: mechanisms are represented in a large number of samples, i.e., they have a positive effect on balancing learning. Theory: can reduce their weight or use class weighting.
A lot of mechanisms have only a few positive examples. This causes a problem of extreme imbalance => MultilabelStratifiedKFold is needed to ensure that rare labels are evenly distributed across folds.
PCA of Gene Expression: 24 h forms its own dense cluster, while 48 h and 72 h partially overlap, indicating a gradual change in biological response over time. cp_time parameter has a pronounced biological effect and is associated with changes in gene expression and cell viability.
UMAP of Gene Expression: Distribution of D1 and D2 overlaps significantly, but D2 (higher dose) tends to shift to areas with lower cell viability. Dose of the drug significantly affects expression patterns; cp_dose is an informative feature.

# Feature engineering

Before starting feature engineering, control samples (cp_type == ctl_vehicle) were excluded because they have no biological effect and distort distributions.
Due to differences in ranges between gene (g-) and cell (c-) features, we used QuantileTransformer or StandardScaler
One-Hot Encoding was applied for cp_time and cp_dose. 
Since we have over 800 features and a good correlation between them, Principal Component Analysis (PCA) was performed separately for each group
Due to the imbalance of rare labels, we combined: GroupKFold(by drug_id) → avoids leakage between drugs and MultilabelStratifiedKFold → maintains the balance of rare mechanisms

# Modeling Approaches

1. Multilayer Perceptron (MLP)
Parameter: Value
Optimizer: Adam
Learning rat:e 1e-3 
Batch size: 256
Epochs: 50
Dropout: 0.2–0.4
Weight decay:  1e-5
Loss: BCEWithLogitsLoss
Activation: ReLU (hidden), Sigmoid (output)
Architecture
Input (662 features)
→ Dense(1024) + BatchNorm + FeatureGate + ReLU + Dropout(0.4)
→ Dense(512) + BatchNorm + FeatureGate + ReLU + Dropout(0.3)
→ Dense(256) + BatchNorm + ReLU + Dropout(0.2)
→ Output(206) + Sigmoid
2. XGBoost
For XGBoost, the same feature set was used as for MLP, but without PCA.
xgb.XGBClassifier(
	n_estimators=1000,
	learning_rate=0.01,
	max_depth=8,
	subsample=0.8,
	colsample_bytree=0.8,
	eval_metric='logloss',
	tree_method='gpu_hist'
)
3. Blending (Level 1 Ensemble)
Coefficients were selected empirically based on OOF LogLoss to minimize validation error.
‘final_preds = 0.7 * mlp_preds + 0.3 * xgb_preds’


Model
Private LogLoss
Public LogLoss
XGBoost
?
?
MLP (FeatureGate)
0.0170
0.0195
Ensemble (MLP + XGB)
0.01713
0.01930



4. Stacking (Level 2 Meta-Model) 
The second level of the ensemble combined: MLP + XGBoost + Blending (MLP + XGBoost)
Their out-of-fold predictions were used as features for the meta-learner - logistic regression.

Model
Private
Public
Blend (MLP + XGB)
0.01841
0.02060
Stacking (MLP + XGB + LGBM)
0.01713
0.01930



Mechanisms of Action (MoA) — Final Report


 Data Insights
Distribution of gene (g-) and cell (c-) features:
 Both g- and c-features have approximately normal, bell-shaped distributions. This confirms that biological data were already normalized, but minor scaling is still required for stable model convergence.
Train vs Test comparisons:
cp_dose: Balanced across datasets; no stratification needed.


cp_time: All three intervals (24h, 48h, 72h) are present in similar proportions; 48h slightly dominates (~8500 samples). This reflects realistic biological timing effects.


cp_type: Control samples (ctl_vehicle) are <10% of total and have no active MoA targets, hence excluded from training.


Drug frequency distribution:
 The dataset is heavily imbalanced — a few drugs appear >1500 times, while most appear only a few dozen.
 → Conclusion: GroupKFold(by drug_id) is required to prevent data leakage.
Feature distributions:
 Histograms for individual g- and c-features show near-normal shapes, though some are skewed — hence scaling (Quantile or Standard) is essential.
Correlation heatmaps:
g-features form several strongly correlated clusters.


c-features also show internal correlation, but weaker cross-correlation with g-features.
 → Conclusion: PCA is justified to decorrelate features and reduce dimensionality.


Target imbalance:
Some mechanisms (e.g., nfkb_inhibitor, proteasome_inhibitor) are frequent — these dominate the loss and may need weighting.


Many MoA targets have only a few positive samples → MultilabelStratifiedKFold ensures rare labels are balanced across folds.


PCA & UMAP visualizations:
PCA: Samples with 24h treatment form distinct clusters, while 48h and 72h overlap partially, indicating a time-dependent biological gradient.


UMAP: D1 and D2 doses largely overlap, but D2 (higher dose) tends to cluster in regions of lower cell viability.
 → cp_time and cp_dose are biologically meaningful and must be included in the model.



Feature Engineering

Exclusion of control samples:
 All cp_type == ctl_vehicle entries were removed to avoid distorting biological signals.


Scaling:
 Due to different ranges between g- and c-features, either QuantileTransformer or StandardScaler was applied.


Categorical encoding:
 cp_time and cp_dose were one-hot encoded to preserve discrete biological stages.


Dimensionality reduction:


PCA applied separately to each group:


600 components for g-features


60 components for c-features


Combined total: 662 features


Cross-validation scheme:
 Combined:


GroupKFold(by drug_id) → prevents leakage between drugs.


MultilabelStratifiedKFold → maintains balance of rare MoA labels.



Modeling Approaches


1. Multilayer Perceptron (MLP)
Parameter
Value
Optimizer
Adam
Learning rate
1e-3
Batch size
256
Epochs
50
Dropout
0.2–0.4
Weight decay
1e-5
Loss
BCEWithLogitsLoss
Activation
ReLU (hidden), Sigmoid (output)

Architecture:
Input (662 features)
→ Dense(1024) + BatchNorm + FeatureGate + ReLU + Dropout(0.4)
→ Dense(512)  + BatchNorm + FeatureGate + ReLU + Dropout(0.3)
→ Dense(256)  + BatchNorm + ReLU + Dropout(0.2)
→ Output(206) + Sigmoid

Results:
Model
Private LogLoss
Public LogLoss
MLP (FeatureGate)
0.01704
0.01947

→ Stable results with minimal public-private gap (≈0.0024), confirming model generalization.

2. XGBoost
Features: Same set as MLP (with Quantile scaling and PCA).
xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    tree_method='gpu_hist'
)

Results:
 Consistent but slightly weaker performance on rare labels due to data imbalance.
Model
Private LogLoss
Public LogLoss
XGBoost
0.0208
0.0221


3. Blending (Level 1 Ensemble)
Approach: Weighted average of MLP and XGBoost predictions.
final_preds = 0.7 * mlp_preds + 0.3 * xgb_preds

Model
Private LogLoss
Public LogLoss
MLP (FeatureGate)
0.0170
0.0195
XGBoost
0.0208
0.0221
Blend (MLP + XGB)
0.01841
0.02060

→ Blending reduced overall LogLoss and smoothed predictions between folds.

4. Stacking (Level 2 Meta-Model)
Composition:
MLP (FeatureGate)


XGBoost


Blending layer (MLP + XGB)


Their OOF predictions were used as input for a meta-learner (Logistic Regression), which learned optimal weights.
Model
Private LogLoss
Public LogLoss
Blend (MLP + XGB)
0.01841
0.02060
Stacking (MLP + XGB + LGBM)
0.01713
0.01930

→ Stacking improved validation stability and reduced LogLoss by ~7% compared to the blended level.

Performed also the adversarial validation model. It achieved a ROC AUC score of 0.57, indicating a small but noticeable difference between the training and test datasets.
This suggests that the test data has slightly different feature distributions, which could affect model generalization.
The classifier was trained using the same complexity as the main model to ensure consistent evaluation.
The ROC curve showed mild separation between train and test samples, implying some distribution shift.
Overall, minor adjustments or rebalancing may help improve consistency between datasets before final model training.

Conclusions
The combination of Quantile scaling, PCA, and stratified grouped CV provided a robust preprocessing foundation.


MLP with FeatureGate captured complex nonlinear biological interactions.


XGBoost offered complementary gradient-based structure and stability.


Blending + Stacking achieved the final Private LogLoss = 0.01713, among the best non-external solutions on Kaggle.









