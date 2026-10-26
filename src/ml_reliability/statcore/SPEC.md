# StatCore Models – Specification for Implementation (VS Code Copilot Ready)

## Overall Contract (applies to every technique)

| Item | Requirement |
|------|-------------|
| Base model | Clean, strong, pure-NumPy implementation – sovereignty layer |
| Publishable variants | Multiple distinct methodological ideas that can each support a paper, focused on **ML reliability** |
| Selectable at call time | `variant="base" \| "variant_name_1" \| ...` |
| Evaluation metrics | Every model must implement `.evaluate(...)` that returns a dict of appropriate metrics |
| Meta-learning version | Each family must be usable inside the MetaLearner with multiple meta-variants |

---

## Evaluation Metrics (mandatory)

All models must expose an `.evaluate` method that returns a dictionary of metrics.

- Classification → accuracy, precision_macro, recall_macro, f1_macro, mcc
- Regression → mse, mae, r2
- Reconstruction (PCA etc.) → reconstruction_mse, reconstruction_mae, relative_frobenius
- Clustering → inertia, n_clusters_found

Shared implementations live in `evaluation/metrics.py`.

---

## Meta-Learning Variants (mandatory options)

The `MetaLearner` class (in `meta/meta_learning.py`) must support these selectable styles:

| meta_variant | Meaning |
|--------------|---------|
| `performance_aware` | Select or weight base models by validation performance |
| `metrics_based` | Generate metrics from base models and **learn from those metrics** as meta-features |
| `oof_stacking` | Classical out-of-fold probability / prediction stacking |
| `statistical_meta` | Statistical meta-classifier (LDA + clustering + logistic style) |
| `calibrated_stack` | Stacking followed by calibration |

Usage example:
```python
meta = MetaLearner(
    base_estimators=[("lda", LDA()), ("mlp", MLPClassifier())],
    meta_variant="metrics_based",   # or any of the above
    n_folds=5,
)
meta.fit(X, y)
```

---

## Model Families

### 1. PCA
- File: `pca/pca.py`
- Variants: `base`, `supervised_row_aware`, `robust`, `sparse`, `probabilistic`
- Must implement: `fit`, `transform`, `fit_transform`, `inverse_transform`, `evaluate`

### 2. LDA
- File: `lda/lda.py`
- Variants: `base`, `shift_corrected`, `shrinkage`, `sparse`, `heteroscedastic`
- Must implement: `fit`, `transform`, `predict`, `evaluate`

### 3. KMeans
- File: `kmeans/kmeans.py`
- Variants: `base`, `topology_aware`, `stat_model_selection`, `robust`, `better_init`
- Must implement: `fit`, `predict`, `evaluate`

### 4. Linear / Logistic
- File: `linear/linear.py`
- Classes: `LogisticRegression`, `RidgeRegression`
- Variants: `base`, `risk_calibrated`, `stat_regularised`, `robust_loss`
- Must implement: `fit`, `predict`, `predict_proba` (for classifier), `evaluate`

### 5. KNN
- File: `knn/knn.py`
- Classes: `KNNClassifier`, `KNNRegressor`
- Variants: `base`, `prototype_compressed`, `learnable_metric`, `stat_k_selection`
- Must implement: `fit`, `predict`, `evaluate`

### 6. MLP
- File: `mlp/mlp.py`
- Classes: `MLPClassifier`, `MLPRegressor`
- Variants: `base`, `stat_regularised`, `reliability_head`, `uncertainty_head`
- Must implement: `fit`, `predict`, `predict_proba` (classifier), `evaluate`
- Base MLP must be strong (proper initialisation, weight decay, etc.)

---

## Dependency Policy
- Default: Level 0 (numpy + pandas + minimal scipy)
- No mandatory scikit-learn in the critical path
- Heavier libraries only in optional higher levels

---

## Integration with ReliableML
All classes should be importable as:
```python
from ml_reliability.statcore import PCA, LDA, KMeans, LogisticRegression, MLPClassifier, MetaLearner
```
and usable inside existing ReliableML pipelines and meta-learning code.
