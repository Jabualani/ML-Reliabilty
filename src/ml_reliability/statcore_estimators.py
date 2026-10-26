"""scikit-learn compatible adapters for the pure-NumPy StatCore classifiers.

The adapters follow the same standard as :mod:`ml_reliability.meta_learning`:
``BaseEstimator``/``ClassifierMixin`` parameter handling (``get_params``,
``clone``, grid search), validated input, ``feature_names_in_``, ``classes_``,
aligned ``predict_proba`` and the raw-feature importance mixin. The numerical
work is done entirely by :mod:`ml_reliability.statcore` (NumPy/SciPy only).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .meta_learning import _RawFeatureImportanceMixin, _input_feature_names
from .statcore import LDA, KNNClassifier, MetaLearner, MLPClassifier
from .statcore import LogisticRegression as _Logistic

# Own (non-base) variants that can be fitted from (X, y) alone.
STATCORE_VARIANTS: dict[str, tuple[str, ...]] = {
    "lda": ("shrinkage", "sparse", "heteroscedastic"),
    "knn": ("prototype_compressed", "learnable_metric", "stat_k_selection"),
    "logistic": ("risk_calibrated", "stat_regularised", "robust_loss"),
    "mlp": ("stat_regularised", "reliability_head", "uncertainty_head"),
}


class _StatCoreClassifier(
    _RawFeatureImportanceMixin, ClassifierMixin, BaseEstimator
):
    """Common fit/predict plumbing; subclasses define ``_build``."""

    def _build(self) -> Any:
        raise NotImplementedError

    def fit(self, X: Any, y: Any):
        self.feature_names_in_ = _input_feature_names(X)
        values, labels = check_X_y(X, y, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.classes_ = np.unique(labels)
        if len(self.classes_) < 2:
            raise ValueError("StatCore classifiers require at least two classes.")
        self.model_ = self._build().fit(values, labels)
        self.is_fitted_ = True
        return self

    def _checked(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classes_")
        values = check_array(X, dtype=float, ensure_2d=True)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {values.shape[1]} features; expected {self.n_features_in_}."
            )
        return values

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classes_")
        raw = np.asarray(self.model_.predict_proba(self._checked(X)), dtype=float)
        raw = np.clip(raw, 0.0, None)
        return raw / raw.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classes_")
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class StatCoreLogisticClassifier(_StatCoreClassifier):
    """Multinomial logistic regression (variants: base, risk_calibrated,
    stat_regularised, robust_loss)."""

    def __init__(self, variant: str = "stat_regularised", alpha: float = 1e-2,
                 cv: int = 5, max_iter: int = 300, random_state: int = 0) -> None:
        self.variant = variant
        self.alpha = alpha
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state

    def _build(self) -> Any:
        return _Logistic(variant=self.variant, alpha=self.alpha, cv=self.cv,
                         max_iter=self.max_iter, random_state=self.random_state)


class StatCoreLDAClassifier(_StatCoreClassifier):
    """Discriminant analysis (variants: base, shrinkage, sparse, heteroscedastic)."""

    def __init__(self, variant: str = "shrinkage", shrinkage: float | None = None,
                 reg: float = 1e-6, threshold: float = 1.0, reg_param: float = 0.3,
                 random_state: int = 0) -> None:
        self.variant = variant
        self.shrinkage = shrinkage
        self.reg = reg
        self.threshold = threshold
        self.reg_param = reg_param
        self.random_state = random_state

    def _build(self) -> Any:
        return LDA(variant=self.variant, shrinkage=self.shrinkage, reg=self.reg,
                   threshold=self.threshold, reg_param=self.reg_param,
                   random_state=self.random_state)


class StatCoreKNNClassifier(_StatCoreClassifier):
    """k-NN (variants: base, prototype_compressed, learnable_metric, stat_k_selection)."""

    def __init__(self, variant: str = "stat_k_selection", n_neighbors: int = 5,
                 weights: str = "uniform", compression: float = 0.2,
                 metric_iters: int = 60, metric_lr: float = 0.1,
                 random_state: int = 0) -> None:
        self.variant = variant
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.compression = compression
        self.metric_iters = metric_iters
        self.metric_lr = metric_lr
        self.random_state = random_state

    def _build(self) -> Any:
        return KNNClassifier(
            n_neighbors=self.n_neighbors, variant=self.variant, weights=self.weights,
            compression=self.compression, metric_iters=self.metric_iters,
            metric_lr=self.metric_lr, random_state=self.random_state)


class StatCoreMLPClassifier(_StatCoreClassifier):
    """MLP (variants: base, stat_regularised, reliability_head, uncertainty_head)."""

    def __init__(self, variant: str = "stat_regularised",
                 hidden_layer_sizes: Sequence[int] = (32, 32),
                 learning_rate: float = 0.01, weight_decay: float = 1e-2,
                 max_epochs: int = 150, batch_size: int = 64, patience: int = 15,
                 dropout: float = 0.1, random_state: int = 0) -> None:
        self.variant = variant
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout
        self.random_state = random_state

    def _build(self) -> Any:
        return MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes), variant=self.variant,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            max_epochs=self.max_epochs, batch_size=self.batch_size,
            patience=self.patience, dropout=self.dropout,
            random_state=self.random_state)


def default_statcore_base_estimators(
    random_state: int = 0,
) -> list[tuple[str, _StatCoreClassifier]]:
    """Own-variant base models used by :class:`StatCoreMetaClassifier`."""
    return [
        ("sc_logistic", StatCoreLogisticClassifier(random_state=random_state)),
        ("sc_lda", StatCoreLDAClassifier(random_state=random_state)),
        ("sc_knn", StatCoreKNNClassifier(random_state=random_state)),
    ]


class StatCoreMetaClassifier(_StatCoreClassifier):
    """Leakage-free StatCore meta-learner (OOF stacking and relatives).

    ``meta_variant`` is one of ``performance_aware``, ``metrics_based``,
    ``oof_stacking``, ``statistical_meta`` or ``calibrated_stack``. With
    ``base_estimators=None`` the StatCore own-variant logistic, LDA and k-NN
    models are stacked.
    """

    def __init__(self, base_estimators: Sequence[tuple[str, Any]] | None = None,
                 meta_variant: str = "oof_stacking", n_folds: int = 5,
                 temperature: float = 0.1, random_state: int = 0) -> None:
        self.base_estimators = base_estimators
        self.meta_variant = meta_variant
        self.n_folds = n_folds
        self.temperature = temperature
        self.random_state = random_state

    def _build(self) -> Any:
        bases = (default_statcore_base_estimators(self.random_state)
                 if self.base_estimators is None else list(self.base_estimators))
        return MetaLearner(bases, meta_variant=self.meta_variant,
                           n_folds=self.n_folds, temperature=self.temperature,
                           random_state=self.random_state)


__all__ = [
    "STATCORE_VARIANTS",
    "StatCoreKNNClassifier",
    "StatCoreLDAClassifier",
    "StatCoreLogisticClassifier",
    "StatCoreMLPClassifier",
    "StatCoreMetaClassifier",
    "default_statcore_base_estimators",
]
