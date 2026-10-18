from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


_DEFAULT_BASE_ESTIMATORS = (
    ("logistic_regression", LogisticRegression(max_iter=500)),
    ("linear_discriminant_analysis", LinearDiscriminantAnalysis()),
    ("decision_tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
)


def _named_estimators(
    estimators: Sequence[tuple[str, Any]] | None,
) -> list[tuple[str, Any]]:
    result = list(_DEFAULT_BASE_ESTIMATORS if estimators is None else estimators)
    if not result:
        raise ValueError("At least one base estimator must be provided.")
    names = [name for name, _ in result]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Base estimator names must be non-empty strings.")
    if len(names) != len(set(names)):
        raise ValueError("Base estimator names must be unique.")
    for name, estimator in result:
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError(
                f"Base estimator {name!r} must implement fit and predict."
            )
    return result


def _validate_folds(y: np.ndarray, cv: int, name: str) -> None:
    if not isinstance(cv, int) or isinstance(cv, bool) or cv < 2:
        raise ValueError(f"{name} must be an integer of at least 2.")
    counts = np.unique(y, return_counts=True)[1]
    if len(counts) < 2:
        raise ValueError("Meta-classifiers require at least two target classes.")
    if int(counts.min()) < cv:
        raise ValueError(
            f"cv={cv} exceeds the smallest class count ({int(counts.min())}); "
            "reduce cv or supply more examples per class."
        )


def _fit_predict_proba(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    X_predict: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    estimator.fit(X, y)
    return _aligned_probabilities(estimator, X_predict, classes)


def _aligned_probabilities(
    estimator: Any, X: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    if callable(getattr(estimator, "predict_proba", None)):
        raw = np.asarray(estimator.predict_proba(X), dtype=float)
        source_classes = np.asarray(estimator.classes_)
        if raw.ndim != 2 or raw.shape[1] != len(source_classes):
            raise ValueError("Estimator returned an invalid predict_proba matrix.")
        if (
            not np.isfinite(raw).all()
            or np.any(raw < 0)
            or np.any(raw > 1)
            or not np.allclose(raw.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError("Estimator returned invalid class probabilities.")
        result = np.zeros((len(X), len(classes)), dtype=float)
        for source_index, source_class in enumerate(source_classes):
            matches = np.flatnonzero(classes == source_class)
            if len(matches):
                result[:, int(matches[0])] = raw[:, source_index]
        row_sums = result.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError(
                "Estimator probabilities do not cover the fitted target classes."
            )
        return result / row_sums

    decision_function = getattr(estimator, "decision_function", None)
    if callable(decision_function):
        scores = np.asarray(decision_function(X), dtype=float)
        if scores.ndim == 1 and len(estimator.classes_) == 2:
            positive = expit(scores)
            raw = np.column_stack([1.0 - positive, positive])
        elif scores.ndim == 2 and scores.shape[1] == len(estimator.classes_):
            raw = softmax(scores, axis=1)
        else:
            raise ValueError("Estimator returned an invalid decision_function result.")
        if not np.isfinite(raw).all():
            raise ValueError("Estimator returned non-finite decision scores.")
        result = np.zeros((len(X), len(classes)), dtype=float)
        for source_index, source_class in enumerate(estimator.classes_):
            matches = np.flatnonzero(classes == source_class)
            if len(matches):
                result[:, int(matches[0])] = raw[:, source_index]
        return result / result.sum(axis=1, keepdims=True)

    predictions = np.asarray(estimator.predict(X))
    result = np.zeros((len(X), len(classes)), dtype=float)
    for index, label in enumerate(classes):
        result[:, index] = predictions == label
    sums = result.sum(axis=1, keepdims=True)
    if np.any(sums == 0):
        raise ValueError("Estimator predicted labels outside the fitted classes.")
    return result / sums


def _oof_meta_features(
    estimators: Sequence[tuple[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    cv: int,
    random_state: int,
) -> tuple[np.ndarray, list[Any]]:
    _validate_folds(y, cv, "cv")
    splitter = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )
    folds = list(splitter.split(X, y))
    columns = []
    fitted = []
    for _, prototype in estimators:
        oof = np.zeros((len(X), len(classes)), dtype=float)
        for train_indices, valid_indices in folds:
            fold_model = clone(prototype)
            oof[valid_indices] = _fit_predict_proba(
                fold_model,
                X[train_indices],
                y[train_indices],
                X[valid_indices],
                classes,
            )
        full_model = clone(prototype).fit(X, y)
        columns.append(oof)
        fitted.append(full_model)
    return np.column_stack(columns), fitted


class _RawFeatureImportanceMixin:
    def local_feature_importances(
        self, X: Any, top_n: int | None = None
    ) -> list[dict[str, Any]]:
        """Estimate row-level raw-feature importance by baseline replacement."""
        check_is_fitted(self, "classes_")
        matrix = self._importance_matrix(X)
        names = self._importance_names(X)
        count = _top_count(len(names), top_n)
        baseline = self._importance_baseline(matrix)
        original = np.asarray(self.predict_proba(X), dtype=float)
        predicted_indices = original.argmax(axis=1)
        rows = []
        for row_index in range(len(matrix)):
            impacts = []
            for feature_index, feature in enumerate(names):
                changed = matrix[row_index : row_index + 1].copy()
                changed[0, feature_index] = baseline[feature_index]
                changed_input = self._restore_importance_input(
                    X, changed, names
                )
                changed_probability = np.asarray(
                    self.predict_proba(changed_input), dtype=float
                )[0, predicted_indices[row_index]]
                base_probability = original[row_index, predicted_indices[row_index]]
                impacts.append(
                    {
                        "feature": feature,
                        "signed_probability_change": float(
                            changed_probability - base_probability
                        ),
                        "importance": float(abs(changed_probability - base_probability)),
                    }
                )
            impacts.sort(key=lambda item: item["importance"], reverse=True)
            rows.append(
                {
                    "row_index": row_index,
                    "predicted_class": _native_scalar(
                        self.classes_[predicted_indices[row_index]]
                    ),
                    "top_features": impacts[:count],
                    "feature_importances": impacts,
                }
            )
        return rows

    def feature_importances(
        self, X: Any, top_n: int | None = None
    ) -> dict[str, Any]:
        """Summarize raw-feature importance and local top-feature frequency."""
        local = self.local_feature_importances(X, top_n=top_n)
        names = self._importance_names(X)
        sums = {name: 0.0 for name in names}
        top_frequency = {name: 0 for name in names}
        for row in local:
            for item in row["feature_importances"]:
                sums[item["feature"]] += item["importance"]
            for item in row["top_features"]:
                top_frequency[item["feature"]] += 1
        row_count = len(local)
        results = [
            {
                "feature": name,
                "mean_absolute_probability_change": sums[name] / row_count,
                "top_feature_frequency": top_frequency[name],
                "top_feature_frequency_fraction": top_frequency[name] / row_count,
            }
            for name in names
        ]
        results.sort(
            key=lambda item: (
                item["top_feature_frequency"],
                item["mean_absolute_probability_change"],
            ),
            reverse=True,
        )
        total = sum(item["mean_absolute_probability_change"] for item in results)
        for item in results:
            item["normalized_importance"] = (
                item["mean_absolute_probability_change"] / total
                if total > 0
                else 0.0
            )
        return {
            "method": (
                "raw-feature baseline replacement; local absolute probability "
                "change with top-feature frequency across rows"
            ),
            "sample_count": row_count,
            "features": results[: _top_count(len(names), top_n)],
        }

    def _importance_matrix(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classes_")
        names = self._importance_names(X)
        if hasattr(X, "columns"):
            received = tuple(str(name) for name in X.columns)
            if received != names:
                raise ValueError("X columns must match the fitted feature order.")
        matrix = check_array(X, dtype=float, ensure_2d=True)
        if matrix.shape[1] != len(names):
            raise ValueError("X feature count differs from the fitted model.")
        return matrix

    def _importance_names(self, X: Any) -> tuple[str, ...]:
        if hasattr(self, "feature_names_in_"):
            return tuple(str(name) for name in self.feature_names_in_)
        if hasattr(X, "columns"):
            return tuple(str(name) for name in X.columns)
        return tuple(
            f"feature_{index}" for index in range(1, self.n_features_in_ + 1)
        )

    def _importance_baseline(self, matrix: np.ndarray) -> np.ndarray:
        return np.median(matrix, axis=0)

    def _restore_importance_input(
        self, template: Any, matrix: np.ndarray, names: tuple[str, ...]
    ) -> Any:
        if hasattr(template, "columns"):
            return template.__class__(matrix, columns=list(names))
        return matrix


def _top_count(feature_count: int, top_n: int | None) -> int:
    if top_n is None:
        return min(5, max(1, math.ceil(feature_count * 0.1)))
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    return min(top_n, feature_count)


def _native_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


class PerfMetaSelect(_RawFeatureImportanceMixin, ClassifierMixin, BaseEstimator):
    """Select and refit the strongest candidate by stratified CV score."""

    def __init__(
        self,
        estimators: Sequence[tuple[str, Any]] | None = None,
        cv: int = 5,
        scoring: str = "roc_auc",
        random_state: int = 42,
    ) -> None:
        self.estimators = estimators
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> PerfMetaSelect:
        self.feature_names_in_ = _input_feature_names(X)
        values, labels = check_X_y(X, y, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.classes_ = np.unique(labels)
        estimators = _named_estimators(self.estimators)
        _validate_folds(labels, self.cv, "cv")
        splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        scorer = get_scorer(self.scoring)
        scores: dict[str, float] = {}
        for name, prototype in estimators:
            fold_scores = []
            for train_indices, valid_indices in splitter.split(values, labels):
                candidate = clone(prototype).fit(
                    values[train_indices], labels[train_indices]
                )
                fold_scores.append(
                    float(
                        scorer(
                            candidate, values[valid_indices], labels[valid_indices]
                        )
                    )
                )
            scores[name] = float(np.mean(fold_scores))
        self.cv_scores_ = scores
        self.best_name_ = max(scores, key=scores.get)
        self.best_estimator_ = clone(dict(estimators)[self.best_name_]).fit(
            values, labels
        )
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        values = self._importance_matrix(X)
        return _aligned_probabilities(self.best_estimator_, values, self.classes_)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class MetaFeatureTransformer(TransformerMixin, BaseEstimator):
    """Create class-aligned out-of-fold probabilities from base estimators."""

    def __init__(
        self,
        estimators: Sequence[tuple[str, Any]] | None = None,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.estimators = estimators
        self.cv = cv
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> MetaFeatureTransformer:
        self.fit_transform(X, y)
        return self

    def fit_transform(
        self, X: Any, y: Any = None, **fit_params: Any
    ) -> np.ndarray:
        if fit_params:
            raise ValueError(
                "MetaFeatureTransformer does not support fit metadata parameters."
            )
        if y is None:
            raise ValueError("MetaFeatureTransformer.fit_transform requires y.")
        values, labels = self._validate_training_data(X, y)
        self.feature_names_in_ = _input_feature_names(X)
        self.n_features_in_ = values.shape[1]
        self.classes_ = np.unique(labels)
        self.estimators_ = _named_estimators(self.estimators)
        features, self.fitted_estimators_ = _oof_meta_features(
            self.estimators_,
            values,
            labels,
            self.classes_,
            self.cv,
            self.random_state,
        )
        self.meta_feature_names_out_ = tuple(
            f"{name}__class_{label}"
            for name, _ in self.estimators_
            for label in self.classes_
        )
        self.is_fitted_ = True
        return features

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        values = self._validate_input(X)
        return np.column_stack(
            [
                _aligned_probabilities(model, values, self.classes_)
                for model in self.fitted_estimators_
            ]
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "meta_feature_names_out_")
        _ = input_features
        return np.asarray(self.meta_feature_names_out_, dtype=object)

    def _validate_training_data(
        self, X: Any, y: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        values, labels = check_X_y(X, y, dtype=float)
        return values, labels

    def _validate_input(self, X: Any) -> np.ndarray:
        if hasattr(X, "columns") and tuple(str(v) for v in X.columns) != tuple(
            self.feature_names_in_
        ):
            raise ValueError("X columns must match the fitted feature order.")
        values = check_array(X, dtype=float, ensure_2d=True)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("X feature count differs from the fitted transformer.")
        return values


class MetaPipelineClassifier(
    _RawFeatureImportanceMixin, ClassifierMixin, BaseEstimator
):
    """Turn arbitrary classifiers into an OOF-trained meta-classifier."""

    def __init__(
        self,
        base_estimators: Sequence[tuple[str, Any]] | None = None,
        meta_classifier: Any | None = None,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.base_estimators = base_estimators
        self.meta_classifier = meta_classifier
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> MetaPipelineClassifier:
        self.feature_names_in_ = _input_feature_names(X)
        values, labels = check_X_y(X, y, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.classes_ = np.unique(labels)
        self.base_estimators_ = _named_estimators(self.base_estimators)
        self.meta_features_, self.fitted_base_estimators_ = _oof_meta_features(
            self.base_estimators_,
            values,
            labels,
            self.classes_,
            self.n_folds,
            self.random_state,
        )
        prototype = (
            LogisticRegression(max_iter=500)
            if self.meta_classifier is None
            else self.meta_classifier
        )
        self.meta_classifier_ = clone(prototype).fit(self.meta_features_, labels)
        self.is_fitted_ = True
        return self

    def transform_meta_features(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        values = self._importance_matrix(X)
        return np.column_stack(
            [
                _aligned_probabilities(model, values, self.classes_)
                for model in self.fitted_base_estimators_
            ]
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        meta = self.transform_meta_features(X)
        return _aligned_probabilities(self.meta_classifier_, meta, self.classes_)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class CalibratedStackedClassifier(
    _RawFeatureImportanceMixin, ClassifierMixin, BaseEstimator
):
    """Stack OOF base probabilities and calibrate the final classifier."""

    def __init__(
        self,
        base_estimators: Sequence[tuple[str, Any]] | None = None,
        meta_estimator: Any | None = None,
        n_folds: int = 5,
        calibration_method: str = "sigmoid",
        random_state: int = 42,
    ) -> None:
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.n_folds = n_folds
        self.calibration_method = calibration_method
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> CalibratedStackedClassifier:
        self.feature_names_in_ = _input_feature_names(X)
        values, labels = check_X_y(X, y, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.classes_ = np.unique(labels)
        self.base_estimators_ = _named_estimators(self.base_estimators)
        self.meta_features_, self.fitted_base_estimators_ = _oof_meta_features(
            self.base_estimators_,
            values,
            labels,
            self.classes_,
            self.n_folds,
            self.random_state,
        )
        _validate_folds(labels, self.n_folds, "n_folds")
        calibration_cv = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        meta = (
            LogisticRegression(max_iter=500)
            if self.meta_estimator is None
            else clone(self.meta_estimator)
        )
        self.meta_estimator_ = CalibratedClassifierCV(
            estimator=meta,
            method=self.calibration_method,
            cv=calibration_cv,
        ).fit(self.meta_features_, labels)
        self.is_fitted_ = True
        return self

    def transform_meta_features(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        values = self._importance_matrix(X)
        return np.column_stack(
            [
                _aligned_probabilities(model, values, self.classes_)
                for model in self.fitted_base_estimators_
            ]
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        meta = self.transform_meta_features(X)
        return _aligned_probabilities(self.meta_estimator_, meta, self.classes_)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class _StatisticalMetaBase(_RawFeatureImportanceMixin, ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        is_nonlinear: bool,
        n_lda_components: int | None,
        n_clusters: int,
        gamma: float,
        use_tree_meta: bool,
        tree_max_depth: int,
        use_polynomial_interactions: bool,
        estimation_method: str,
        C: float,
        l1_ratio: float,
        use_monte_carlo: bool,
        n_mc_sims: int,
        cv: int,
        random_state: int,
    ) -> None:
        self.is_nonlinear = is_nonlinear
        self.n_lda_components = n_lda_components
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.use_tree_meta = use_tree_meta
        self.tree_max_depth = tree_max_depth
        self.use_polynomial_interactions = use_polynomial_interactions
        self.estimation_method = estimation_method
        self.C = C
        self.l1_ratio = l1_ratio
        self.use_monte_carlo = use_monte_carlo
        self.n_mc_sims = n_mc_sims
        self.cv = cv
        self.random_state = random_state
        self.adaptive_gamma = False

    def fit(self, X: Any, y: Any) -> _StatisticalMetaBase:
        self.feature_names_in_ = _input_feature_names(X)
        values, labels = check_X_y(X, y, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.classes_ = np.unique(labels)
        if len(self.classes_) != 2:
            raise ValueError(
                f"{type(self).__name__} currently supports binary classification."
            )
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        _validate_folds(labels, self.cv, "cv")
        if self.n_mc_sims < 1:
            raise ValueError("n_mc_sims must be a positive integer.")

        self.scaler_ = StandardScaler().fit(values)
        scaled = self.scaler_.transform(values)
        lda_components = min(
            self.n_lda_components or 1, values.shape[1], len(self.classes_) - 1
        )
        self.lda_ = LinearDiscriminantAnalysis(
            n_components=max(1, lda_components)
        ).fit(scaled, labels)
        lda_oof = np.zeros((len(values), len(self.classes_)), dtype=float)
        tree_oof = np.zeros_like(lda_oof)
        splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        for train_indices, valid_indices in splitter.split(scaled, labels):
            fold_lda = LinearDiscriminantAnalysis(
                n_components=max(1, lda_components)
            ).fit(scaled[train_indices], labels[train_indices])
            lda_oof[valid_indices] = _aligned_probabilities(
                fold_lda, scaled[valid_indices], self.classes_
            )
            if self.use_tree_meta:
                fold_tree = DecisionTreeClassifier(
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state,
                ).fit(scaled[train_indices], labels[train_indices])
                tree_oof[valid_indices] = _aligned_probabilities(
                    fold_tree, scaled[valid_indices], self.classes_
                )
        if self.use_tree_meta:
            self.tree_ = DecisionTreeClassifier(
                max_depth=self.tree_max_depth,
                random_state=self.random_state,
            ).fit(scaled, labels)

        self.kmeans_ = MiniBatchKMeans(
            n_clusters=min(self.n_clusters, len(scaled)),
            random_state=self.random_state,
            n_init=3,
        ).fit(scaled)
        distances = self.kmeans_.transform(scaled)
        if self.is_nonlinear and self.gamma <= 0:
            raise ValueError("gamma must be positive for nonlinear clustering.")
        if self.is_nonlinear:
            if self.adaptive_gamma:
                assignments = self.kmeans_.labels_
                variances = np.ones(self.kmeans_.n_clusters, dtype=float)
                for cluster in range(self.kmeans_.n_clusters):
                    within = distances[assignments == cluster, cluster]
                    if len(within) > 1 and float(np.var(within)) > 1e-8:
                        variances[cluster] = float(np.var(within))
                self.gammas_ = 1.0 / (2.0 * variances)
            else:
                self.gammas_ = np.full(self.kmeans_.n_clusters, self.gamma)
            cluster_meta = np.exp(-self.gammas_[None, :] * distances**2)
        else:
            self.gammas_ = np.full(self.kmeans_.n_clusters, self.gamma)
            cluster_meta = distances

        lda_projection = self.lda_.transform(scaled)
        lda_score = lda_oof[:, 1:2]
        if not self.is_nonlinear:
            lda_score = np.abs(0.5 - lda_score)
        parts = [lda_projection, lda_score]
        if self.use_tree_meta:
            parts.append(tree_oof[:, 1:2])
        parts.append(cluster_meta)
        base_meta = np.column_stack(parts)
        if self.is_nonlinear and self.use_polynomial_interactions:
            self.poly_ = PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False
            ).fit(base_meta)
            transformed = self.poly_.transform(base_meta)
        else:
            transformed = base_meta
        self.meta_feature_count_ = transformed.shape[1]
        self.beta_ = self._fit_coefficients(transformed, labels)
        self.is_fitted_ = True
        return self

    def _fit_coefficients(
        self, meta: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        design = np.column_stack([np.ones(len(meta)), meta])
        binary_y = (labels == self.classes_[1]).astype(float)
        if self.use_monte_carlo:
            rng = np.random.default_rng(self.random_state)
            samples = []
            positive = np.flatnonzero(binary_y == 1)
            negative = np.flatnonzero(binary_y == 0)
            for _ in range(self.n_mc_sims):
                indices = np.concatenate(
                    [
                        rng.choice(negative, size=len(negative), replace=True),
                        rng.choice(positive, size=len(positive), replace=True),
                    ]
                )
                rng.shuffle(indices)
                samples.append(
                    self._estimate_parameters(design[indices], binary_y[indices])
                )
            return np.mean(samples, axis=0)
        return self._estimate_parameters(design, binary_y)

    def _estimate_parameters(self, design: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.estimation_method == "elasticnet":
            solver = LogisticRegression(
                penalty="elasticnet",
                l1_ratio=self.l1_ratio,
                C=self.C,
                solver="saga",
                max_iter=2000,
                fit_intercept=False,
                random_state=self.random_state,
            ).fit(design, y.astype(int))
            return solver.coef_[0]
        if self.estimation_method not in {"mle", "irls"}:
            raise ValueError("estimation_method must be 'mle', 'irls', or 'elasticnet'.")
        if self.estimation_method == "irls":
            beta = np.zeros(design.shape[1], dtype=float)
            for _ in range(100):
                probability = np.clip(expit(design @ beta), 1e-8, 1 - 1e-8)
                weights = probability * (1 - probability)
                hessian = (design.T * weights) @ design + 1e-8 * np.eye(
                    design.shape[1]
                )
                gradient = design.T @ (y - probability)
                update = np.linalg.solve(hessian, gradient)
                beta_next = beta + update
                if np.linalg.norm(beta_next - beta) < 1e-7:
                    return beta_next
                beta = beta_next
            return beta

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            logits = design @ beta
            probability = expit(logits)
            loss = np.logaddexp(0.0, logits).sum() - float(y @ logits)
            loss += 5e-9 * float(beta[1:] @ beta[1:])
            gradient = design.T @ (probability - y)
            gradient[1:] += 1e-8 * beta[1:]
            return float(loss), gradient

        result = minimize(
            objective,
            np.zeros(design.shape[1], dtype=float),
            method="L-BFGS-B",
            jac=True,
        )
        if not np.isfinite(result.x).all():
            raise RuntimeError("Statistical logistic MLE produced non-finite coefficients.")
        return result.x

    def _transform_meta(self, X: Any) -> np.ndarray:
        values = self._importance_matrix(X)
        scaled = self.scaler_.transform(values)
        lda_projection = self.lda_.transform(scaled)
        lda_probability = _aligned_probabilities(
            self.lda_, scaled, self.classes_
        )[:, 1:2]
        lda_score = (
            lda_probability
            if self.is_nonlinear
            else np.abs(0.5 - lda_probability)
        )
        parts = [lda_projection, lda_score]
        if self.use_tree_meta:
            parts.append(
                _aligned_probabilities(self.tree_, scaled, self.classes_)[:, 1:2]
            )
        distances = self.kmeans_.transform(scaled)
        if self.is_nonlinear:
            cluster_meta = np.exp(-self.gammas_[None, :] * distances**2)
        else:
            cluster_meta = distances
        parts.append(cluster_meta)
        base_meta = np.column_stack(parts)
        if self.is_nonlinear and self.use_polynomial_interactions:
            return self.poly_.transform(base_meta)
        return base_meta

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        meta = self._transform_meta(X)
        logits = np.column_stack([np.ones(len(meta)), meta]) @ self.beta_
        positive = expit(logits)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    @property
    def meta_coefficients_(self) -> np.ndarray:
        check_is_fitted(self, "beta_")
        return self.beta_[1:].copy()


class StatisticalMetaClassifier(_StatisticalMetaBase):
    """Statistical logistic meta-classifier with optional nonlinear features."""

    def __init__(
        self,
        is_nonlinear: bool = False,
        n_lda_components: int | None = None,
        n_clusters: int = 4,
        gamma: float = 0.5,
        use_tree_meta: bool = False,
        tree_max_depth: int = 4,
        use_polynomial_interactions: bool = True,
        estimation_method: str = "mle",
        C: float = 1.0,
        l1_ratio: float = 0.5,
        use_monte_carlo: bool = True,
        n_mc_sims: int = 20,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            is_nonlinear=is_nonlinear,
            n_lda_components=n_lda_components,
            n_clusters=n_clusters,
            gamma=gamma,
            use_tree_meta=use_tree_meta,
            tree_max_depth=tree_max_depth,
            use_polynomial_interactions=use_polynomial_interactions,
            estimation_method=estimation_method,
            C=C,
            l1_ratio=l1_ratio,
            use_monte_carlo=use_monte_carlo,
            n_mc_sims=n_mc_sims,
            cv=cv,
            random_state=random_state,
        )


class StatisticalMetaLDAClassifier(StatisticalMetaClassifier):
    """Linear statistical meta-classifier using LDA and logistic estimation."""

    def __init__(
        self,
        n_lda_components: int | None = None,
        n_clusters: int = 4,
        estimation_method: str = "mle",
        C: float = 1.0,
        l1_ratio: float = 0.5,
        use_monte_carlo: bool = True,
        n_mc_sims: int = 20,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            is_nonlinear=False,
            n_lda_components=n_lda_components,
            n_clusters=n_clusters,
            estimation_method=estimation_method,
            C=C,
            l1_ratio=l1_ratio,
            use_monte_carlo=use_monte_carlo,
            n_mc_sims=n_mc_sims,
            cv=cv,
            random_state=random_state,
        )


class StatisticalMetaDTreeClassifier(StatisticalMetaClassifier):
    """Nonlinear statistical meta-classifier with tree probabilities."""

    def __init__(
        self,
        n_lda_components: int | None = None,
        n_clusters: int = 12,
        use_tree_meta: bool = True,
        tree_max_depth: int = 4,
        use_polynomial_interactions: bool = True,
        estimation_method: str = "elasticnet",
        C: float = 5.0,
        l1_ratio: float = 0.1,
        use_monte_carlo: bool = True,
        n_mc_sims: int = 20,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            is_nonlinear=True,
            n_lda_components=n_lda_components,
            n_clusters=n_clusters,
            use_tree_meta=use_tree_meta,
            tree_max_depth=tree_max_depth,
            use_polynomial_interactions=use_polynomial_interactions,
            estimation_method=estimation_method,
            C=C,
            l1_ratio=l1_ratio,
            use_monte_carlo=use_monte_carlo,
            n_mc_sims=n_mc_sims,
            cv=cv,
            random_state=random_state,
        )
        self.adaptive_gamma = True


StatisticalMetaTreeClassifier = StatisticalMetaDTreeClassifier


class _NumpyDropoutNetwork:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout_rate: float,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
        random_state: int,
    ) -> None:
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        dimensions = [input_dim, *hidden_dims, 1]
        rng = np.random.default_rng(random_state)
        self.weights_ = [
            rng.normal(
                0.0,
                math.sqrt(2.0 / (input_size + output_size)),
                size=(input_size, output_size),
            )
            for input_size, output_size in zip(dimensions, dimensions[1:])
        ]
        self.biases_ = [
            np.zeros((1, output_size), dtype=float)
            for output_size in dimensions[1:]
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> _NumpyDropoutNetwork:
        rng = np.random.default_rng(self.random_state)
        first_moments = [
            np.zeros_like(parameter)
            for parameter in [*self.weights_, *self.biases_]
        ]
        second_moments = [np.zeros_like(value) for value in first_moments]
        step = 0
        targets = y.reshape(-1, 1)
        for _ in range(self.epochs):
            epoch_indices = rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                indices = epoch_indices[start : start + self.batch_size]
                batch = X[indices]
                batch_y = targets[indices]
                logits, cache = self._forward(batch, training=True, rng=rng)
                delta = (_sigmoid_array(logits) - batch_y) / len(batch)
                weight_gradients = [np.empty_like(value) for value in self.weights_]
                bias_gradients = [np.empty_like(value) for value in self.biases_]
                for layer in reversed(range(len(self.weights_))):
                    layer_input, pre_activation, _dropout_mask = cache[layer]
                    weight_gradients[layer] = layer_input.T @ delta
                    bias_gradients[layer] = np.sum(delta, axis=0, keepdims=True)
                    if layer:
                        delta = delta @ self.weights_[layer].T
                        previous_activation = cache[layer - 1]
                        if previous_activation[2] is not None:
                            delta *= previous_activation[2]
                        previous_pre_activation = previous_activation[1]
                        derivative = _sigmoid_array(previous_pre_activation) * (
                            1.0
                            + previous_pre_activation
                            * (1.0 - _sigmoid_array(previous_pre_activation))
                        )
                        delta *= derivative
                step += 1
                gradients = [
                    *weight_gradients,
                    *bias_gradients,
                ]
                parameters = [*self.weights_, *self.biases_]
                for index, (parameter, gradient) in enumerate(
                    zip(parameters, gradients)
                ):
                    if index < len(self.weights_):
                        gradient = gradient + self.weight_decay * parameter
                    first_moments[index] = (
                        0.9 * first_moments[index] + 0.1 * gradient
                    )
                    second_moments[index] = (
                        0.999 * second_moments[index] + 0.001 * gradient**2
                    )
                    corrected_first = first_moments[index] / (1.0 - 0.9**step)
                    corrected_second = second_moments[index] / (1.0 - 0.999**step)
                    parameter -= self.learning_rate * corrected_first / (
                        np.sqrt(corrected_second) + 1e-8
                    )
        return self

    def predict_positive(
        self, X: np.ndarray, stochastic: bool, n_draws: int
    ) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        repetitions = n_draws if stochastic and self.dropout_rate > 0 else 1
        probabilities = [
            _sigmoid_array(
                self._forward(X, training=stochastic, rng=rng)[0]
            ).ravel()
            for _ in range(repetitions)
        ]
        return np.mean(probabilities, axis=0)

    def _forward(
        self, X: np.ndarray, training: bool, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]]:
        activation = X
        cache = []
        for index, (weight, bias) in enumerate(zip(self.weights_, self.biases_)):
            pre_activation = activation @ weight + bias
            if index == len(self.weights_) - 1:
                cache.append((activation, pre_activation, None))
                return pre_activation, cache
            activated = pre_activation * _sigmoid_array(pre_activation)
            dropout_mask = None
            if training and self.dropout_rate > 0:
                keep_probability = 1.0 - self.dropout_rate
                dropout_mask = (
                    rng.random(activated.shape) < keep_probability
                ).astype(float) / keep_probability
                activated = activated * dropout_mask
            cache.append((activation, pre_activation, dropout_mask))
            activation = activated
        raise RuntimeError("Neural network has no output layer.")


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class StatisticalMetaNNClassifier(_StatisticalMetaBase):
    """Neural meta-network over LDA, tree, and cluster features."""

    def __init__(
        self,
        n_clusters: int = 12,
        use_tree_meta: bool = True,
        tree_max_depth: int = 4,
        hidden_dims: Sequence[int] = (64, 32),
        dropout_rate: float = 0.2,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 64,
        use_mc_dropout: bool = True,
        n_mc_sims: int = 50,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            is_nonlinear=True,
            n_lda_components=None,
            n_clusters=n_clusters,
            gamma=0.5,
            use_tree_meta=use_tree_meta,
            tree_max_depth=tree_max_depth,
            use_polynomial_interactions=False,
            estimation_method="mle",
            C=1.0,
            l1_ratio=0.5,
            use_monte_carlo=False,
            n_mc_sims=n_mc_sims,
            cv=cv,
            random_state=random_state,
        )
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_mc_dropout = use_mc_dropout

    def fit(self, X: Any, y: Any) -> StatisticalMetaNNClassifier:
        self.feature_names_in_ = _input_feature_names(X)
        values, labels = check_X_y(X, y, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.classes_ = np.unique(labels)
        if len(self.classes_) != 2:
            raise ValueError("StatisticalMetaNNClassifier supports binary classification.")
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0:
            raise ValueError("epochs, batch_size, or lr is outside its valid range.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1).")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if self.n_mc_sims < 1:
            raise ValueError("n_mc_sims must be a positive integer.")
        _validate_folds(labels, self.cv, "cv")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 1 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")

        self.scaler_ = StandardScaler().fit(values)
        meta = self._transform_scaled_training(values, labels)
        self.nn_model_ = _NumpyDropoutNetwork(
            input_dim=meta.shape[1],
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            batch_size=self.batch_size,
            random_state=self.random_state,
        ).fit(meta, (labels == self.classes_[1]).astype(float))
        self.is_fitted_ = True
        return self

    def _transform_scaled_training(
        self, values: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        scaled = self.scaler_.transform(values)
        lda_oof = np.zeros(len(values), dtype=float)
        tree_oof = np.zeros(len(values), dtype=float)
        splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        for train_indices, valid_indices in splitter.split(scaled, labels):
            lda = LinearDiscriminantAnalysis().fit(
                scaled[train_indices], labels[train_indices]
            )
            lda_oof[valid_indices] = _aligned_probabilities(
                lda, scaled[valid_indices], self.classes_
            )[:, 1]
            if self.use_tree_meta:
                tree = DecisionTreeClassifier(
                    max_depth=self.tree_max_depth, random_state=self.random_state
                ).fit(scaled[train_indices], labels[train_indices])
                tree_oof[valid_indices] = _aligned_probabilities(
                    tree, scaled[valid_indices], self.classes_
                )[:, 1]
        self.lda_ = LinearDiscriminantAnalysis().fit(scaled, labels)
        if self.use_tree_meta:
            self.tree_ = DecisionTreeClassifier(
                max_depth=self.tree_max_depth, random_state=self.random_state
            ).fit(scaled, labels)
        self.kmeans_ = MiniBatchKMeans(
            n_clusters=min(self.n_clusters, len(scaled)),
            random_state=self.random_state,
            n_init=3,
        ).fit(scaled)
        distances = self.kmeans_.transform(scaled)
        assigned = self.kmeans_.labels_
        variances = np.ones(self.kmeans_.n_clusters, dtype=float)
        for cluster in range(self.kmeans_.n_clusters):
            within = distances[assigned == cluster, cluster]
            if len(within) > 1 and float(np.var(within)) > 1e-8:
                variances[cluster] = float(np.var(within))
        self.gammas_ = 1.0 / (2.0 * variances)
        parts = [
            self.lda_.transform(scaled),
            lda_oof.reshape(-1, 1),
        ]
        if self.use_tree_meta:
            parts.append(tree_oof.reshape(-1, 1))
        parts.append(np.exp(-self.gammas_[None, :] * distances**2))
        return np.column_stack(parts)

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        values = self._importance_matrix(X)
        scaled = self.scaler_.transform(values)
        lda_probability = _aligned_probabilities(
            self.lda_, scaled, self.classes_
        )[:, 1:2]
        parts = [self.lda_.transform(scaled), lda_probability]
        if self.use_tree_meta:
            parts.append(
                _aligned_probabilities(self.tree_, scaled, self.classes_)[:, 1:2]
            )
        distances = self.kmeans_.transform(scaled)
        parts.append(np.exp(-self.gammas_[None, :] * distances**2))
        positive = self.nn_model_.predict_positive(
            np.column_stack(parts),
            stochastic=self.use_mc_dropout,
            n_draws=self.n_mc_sims,
        )
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def halving_grid_search(
    estimator: Any,
    param_grid: Mapping[str, Sequence[Any]],
    **kwargs: Any,
) -> Any:
    """Construct sklearn's successive-halving grid search for an estimator."""
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV

    return HalvingGridSearchCV(estimator, param_grid, **kwargs)


def halving_random_search(
    estimator: Any,
    param_distributions: Mapping[str, Any],
    **kwargs: Any,
) -> Any:
    """Construct sklearn's successive-halving randomized search."""
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV

    return HalvingRandomSearchCV(estimator, param_distributions, **kwargs)


def _input_feature_names(X: Any) -> np.ndarray:
    if hasattr(X, "columns"):
        names = np.asarray([str(name) for name in X.columns], dtype=object)
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique.")
        return names
    values = check_array(X, dtype=float, ensure_2d=True)
    return np.asarray(
        [f"feature_{index}" for index in range(1, values.shape[1] + 1)],
        dtype=object,
    )
