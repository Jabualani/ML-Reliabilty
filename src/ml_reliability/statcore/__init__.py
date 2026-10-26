"""
StatCore - pure-NumPy statistical and machine-learning models for ReliableML.

Every model supports a ``variant`` argument, ``evaluate(...)`` returning a metric dict
(see ``evaluation.metrics``) and a ``random_state`` for determinism. See ``SPEC.md``.
"""

from ._common import NotFittedError
from .evaluation.metrics import (
    calibration_metrics,
    classification_metrics,
    clustering_metrics,
    reconstruction_metrics,
    regression_metrics,
)
from .kmeans.kmeans import KMeans
from .knn.knn import KNNClassifier, KNNRegressor
from .lda.lda import LDA
from .linear.linear import LogisticRegression, RidgeRegression
from .meta.meta_learning import MetaLearner
from .mlp.mlp import MLPClassifier, MLPRegressor
from .pca.pca import PCA

__all__ = [
    "PCA", "LDA", "KMeans", "LogisticRegression", "RidgeRegression",
    "KNNClassifier", "KNNRegressor", "MLPClassifier", "MLPRegressor", "MetaLearner",
    "NotFittedError", "classification_metrics", "regression_metrics",
    "reconstruction_metrics", "clustering_metrics", "calibration_metrics",
]
