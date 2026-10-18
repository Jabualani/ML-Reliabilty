from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .drift import _as_columns, _category_key
from .explainability import (
    ExplanationReport,
    _numeric_matrix,
    _probability_adapter,
    _restore_input,
    explain_model,
)


@dataclass(frozen=True)
class ReasonAuditReport:
    positive_label: Any
    sample_count: int
    predictive_validity: Mapping[str, Any]
    feature_reliance: tuple[Mapping[str, Any], ...]
    stability: Mapping[str, Any]
    counterfactual_sensitivity: Mapping[str, Any]
    logistic_association: Mapping[str, Any]
    explanations: Mapping[str, Any]
    explanation_alignment: Mapping[str, Any]
    supplied_importance_alignment: Mapping[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_label": _native_scalar(self.positive_label),
            "sample_count": self.sample_count,
            "predictive_validity": dict(self.predictive_validity),
            "feature_reliance": [dict(item) for item in self.feature_reliance],
            "stability": dict(self.stability),
            "counterfactual_sensitivity": dict(self.counterfactual_sensitivity),
            "logistic_association": dict(self.logistic_association),
            "explanations": dict(self.explanations),
            "explanation_alignment": dict(self.explanation_alignment),
            "supplied_importance_alignment": dict(
                self.supplied_importance_alignment
            ),
            "warnings": list(self.warnings),
            "interpretation": (
                "This is a statistical diagnostic for predictive validity, "
                "observed feature reliance, stability, sensitivity, and "
                "associational consistency. It cannot establish that a model "
                "used causally correct reasons or that no spurious correlation exists."
            ),
        }


def audit_model_reasons(
    estimator: Any,
    X: Any,
    y_true: Sequence[Any],
    *,
    feature_names: Sequence[str] | None = None,
    background_data: Any | None = None,
    positive_label: Any | None = None,
    feature_importance: Mapping[str, Real] | None = None,
    conditional_on: Sequence[str] | None = None,
    conditional_neighbors: int = 10,
    groups: Mapping[str, Sequence[Any]] | None = None,
    time_periods: Sequence[Any] | None = None,
    counterfactuals: Mapping[str, Sequence[Any]] | None = None,
    negative_control_features: Sequence[str] = (),
    unavailable_features: Sequence[str] = (),
    bootstrap_samples: int = 300,
    confidence_level: float = 0.95,
    explanation_samples: int = 200,
    max_explained_instances: int = 100,
    random_seed: int = 42,
) -> ReasonAuditReport:
    if not all(
        hasattr(estimator, attribute)
        for attribute in ("predict", "predict_proba", "classes_")
    ):
        raise TypeError(
            "A fitted classification estimator with predict, predict_proba, and "
            "classes_ is required."
        )
    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples < 100
    ):
        raise ValueError("bootstrap_samples must be an integer of at least 100.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if (
        not isinstance(conditional_neighbors, int)
        or isinstance(conditional_neighbors, bool)
        or conditional_neighbors < 1
    ):
        raise ValueError("conditional_neighbors must be a positive integer.")

    columns = _as_columns(X)
    names = tuple(columns) if feature_names is None else tuple(feature_names)
    if set(columns) != set(names) or len(names) != len(set(names)):
        raise ValueError("feature_names must uniquely match the columns in X.")
    if feature_importance is not None:
        _validate_named_features(set(feature_importance), names)
    matrix = _numeric_matrix(X, names)
    actual = list(y_true)
    if len(actual) != len(matrix) or not actual:
        raise ValueError("y_true must have one label for each non-empty row of X.")
    if any(_is_missing_label(value) for value in actual):
        raise ValueError("y_true must not contain missing or non-finite labels.")
    classes = list(estimator.classes_)
    class_keys = [_value_key(value) for value in classes]
    observed = {_value_key(value) for value in actual}
    if not observed <= set(class_keys):
        raise ValueError("y_true contains labels absent from estimator.classes_.")
    if len(classes) == 2:
        if positive_label is None:
            positive_label = classes[1]
    elif positive_label is None:
        raise ValueError("positive_label is required for multiclass reason auditing.")
    if _value_key(positive_label) not in class_keys:
        raise ValueError("positive_label must be one of estimator.classes_.")
    positive_index = class_keys.index(_value_key(positive_label))

    predicted = list(estimator.predict(X))
    if len(predicted) != len(actual):
        raise ValueError("Estimator predictions must match the row count of X.")
    predict_probabilities = _probability_adapter(estimator, X, names)
    probabilities = predict_probabilities(matrix)
    if probabilities.shape != (len(actual), len(classes)):
        raise ValueError("predict_proba output shape does not match labels and rows.")

    reference = X if background_data is None else background_data
    reference_matrix = _numeric_matrix(reference, names)
    if len(reference_matrix) == 0:
        raise ValueError("background_data must contain at least one row.")
    rng = random.Random(random_seed)
    warnings = [
        "Association, feature reliance, and explanation agreement do not establish causality or prove absence of spurious correlations.",
        "This implementation requires finite numeric features and a fitted classifier with predict_proba and classes_.",
    ]
    if background_data is None:
        warnings.append(
            "No training/reference background was supplied; X itself was used as the explanation background."
        )

    predictive_validity = _predictive_validity(
        actual,
        predicted,
        probabilities,
        classes,
        positive_index,
        bootstrap_samples,
        confidence_level,
        rng,
    )
    if conditional_on:
        unknown_conditioners = set(conditional_on) - set(names)
        if unknown_conditioners:
            raise ValueError(
                f"conditional_on contains unknown features: {sorted(unknown_conditioners)}."
            )
        if set(conditional_on) == set(names):
            raise ValueError(
                "conditional_on must leave at least one feature unconditioned."
            )
        if len(matrix) < 2:
            raise ValueError(
                "Conditional perturbation requires at least two holdout rows."
            )
        donor_pools = {}
        conditioning_features = {}
        for feature in names:
            applicable = [name for name in conditional_on if name != feature]
            conditioning_features[feature] = applicable
            if applicable:
                conditioning_matrix = matrix[
                    :, [names.index(name) for name in applicable]
                ]
                donor_pools[feature] = _nearest_donors(
                    conditioning_matrix,
                    min(conditional_neighbors, len(matrix) - 1),
                )
            else:
                donor_pools[feature] = _marginal_donors(len(matrix))
    else:
        donor_pools = {
            feature: _marginal_donors(len(matrix)) for feature in names
        }
        conditioning_features = {feature: [] for feature in names}
        warnings.append(
            "Feature-reliance estimates use marginal donor perturbations. "
            "Correlated features may yield unrealistic combinations; set "
            "conditional_on to reduce, but not eliminate, that limitation."
        )

    feature_reliance = _feature_reliance(
        estimator,
        X,
        matrix,
        actual,
        classes,
        names,
        donor_pools,
        conditioning_features,
        bootstrap_samples,
        confidence_level,
        rng,
    )
    stability = _stability_slices(
        estimator,
        X,
        matrix,
        actual,
        classes,
        names,
        donor_pools,
        groups,
        time_periods,
        rng,
        bootstrap_samples,
        confidence_level,
    )
    if groups is None and time_periods is None:
        warnings.append(
            "No group or time-period variables were supplied; subgroup/temporal stability was not assessed."
        )
    sensitivity = _counterfactual_sensitivity(
        estimator,
        X,
        matrix,
        probabilities,
        predicted,
        names,
        counterfactuals,
        positive_index,
        bootstrap_samples,
        confidence_level,
        random_seed,
    )
    if not counterfactuals:
        warnings.append(
            "No domain-valid counterfactuals were supplied; realistic intervention sensitivity was not assessed."
        )

    logistic = _logistic_associations(matrix, actual, names, positive_label)
    try:
        explain_background = reference if background_data is not None else matrix
        explanations = explain_model(
            estimator=estimator,
            X=X,
            feature_names=names,
            background_data=explain_background,
            class_label=positive_label,
            max_instances=max_explained_instances,
            num_samples=explanation_samples,
            random_seed=random_seed,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
        )
        explanation_alignment = _logistic_explanation_alignment(
            explanations, matrix, names, logistic
        )
        if explanation_alignment.get("status") != "ok":
            warnings.append(str(explanation_alignment.get("reason")))
        explanation_output = explanations.to_dict()
    except (ValueError, TypeError) as error:
        explanations = None
        explanation_output = {
            "status": "unavailable",
            "reason": str(error),
        }
        explanation_alignment = {
            "status": "unavailable",
            "reason": str(error),
        }
        warnings.append(
            "SHAP/LIME-to-logistic row-level alignment was unavailable; see its reason."
        )

    supplied_alignment = _supplied_importance_alignment(
        feature_importance, names, explanations
    )
    if negative_control_features or unavailable_features:
        _validate_named_features(
            set(negative_control_features) | set(unavailable_features), names
        )
    feature_by_name = {item["feature"]: item for item in feature_reliance}
    for name in negative_control_features:
        item = feature_by_name[name]
        item["negative_control_warning"] = (
            item["loss_increase_confidence_interval"][0] > 0
        )
    for name in unavailable_features:
        item = feature_by_name[name]
        item["unavailable_at_prediction_time_warning"] = (
            item["loss_increase"] > 0
        )
    if unavailable_features:
        warnings.append(
            "Unavailable-at-prediction-time features with positive reliance are potential leakage/shortcut indicators, not proof of leakage."
        )
    return ReasonAuditReport(
        positive_label=positive_label,
        sample_count=len(actual),
        predictive_validity=predictive_validity,
        feature_reliance=tuple(feature_reliance),
        stability=stability,
        counterfactual_sensitivity=sensitivity,
        logistic_association=logistic,
        explanations=explanation_output,
        explanation_alignment=explanation_alignment,
        supplied_importance_alignment=supplied_alignment,
        warnings=tuple(warnings),
    )


def _predictive_validity(
    actual: list[Any],
    predicted: list[Any],
    probabilities: np.ndarray,
    classes: list[Any],
    positive_index: int,
    bootstrap_samples: int,
    confidence_level: float,
    rng: random.Random,
) -> dict[str, Any]:
    class_indices = {_value_key(label): index for index, label in enumerate(classes)}
    actual_indices = [class_indices[_value_key(label)] for label in actual]
    correctness = [
        _value_key(label) == _value_key(prediction)
        for label, prediction in zip(actual, predicted)
    ]
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(actual)), actual_indices] = 1.0
    brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    confidences = np.max(probabilities, axis=1)
    confidence_correctness = np.asarray(correctness, dtype=float)
    ece = _expected_calibration_error(confidences, confidence_correctness)
    boot_accuracy = []
    boot_brier = []
    boot_ece = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(actual)) for _ in actual]
        boot_accuracy.append(float(np.mean([correctness[index] for index in indices])))
        boot_brier.append(float(np.mean(np.sum(
            (probabilities[indices] - one_hot[indices]) ** 2, axis=1
        ))))
        boot_ece.append(
            _expected_calibration_error(
                confidences[indices], confidence_correctness[indices]
            )
        )
    return {
        "accuracy": float(np.mean(correctness)),
        "accuracy_confidence_interval": _bootstrap_interval(
            boot_accuracy, confidence_level
        ),
        "brier_score": float(brier),
        "brier_score_confidence_interval": _bootstrap_interval(
            boot_brier, confidence_level
        ),
        "top_label_expected_calibration_error": float(ece),
        "expected_calibration_error_confidence_interval": _bootstrap_interval(
            boot_ece, confidence_level
        ),
        "calibration_bins": 10,
        "positive_label": _native_scalar(classes[positive_index]),
        "positive_class_brier_score": float(
            np.mean(
                (
                    probabilities[:, positive_index]
                    - np.asarray(
                        [
                            _value_key(label) == _value_key(classes[positive_index])
                            for label in actual
                        ],
                        dtype=float,
                    )
                )
                ** 2
            )
        ),
        "interpretation": (
            "Intervals are row-bootstrap sampling intervals for this labeled batch; "
            "representativeness and independent rows are assumed."
        ),
    }


def _expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (confidences >= edges[index]) & (
                confidences <= edges[index + 1]
            )
        else:
            selected = (confidences >= edges[index]) & (
                confidences < edges[index + 1]
            )
        if np.any(selected):
            score += (
                np.mean(selected)
                * abs(float(np.mean(confidences[selected])) - float(np.mean(correctness[selected])))
            )
    return float(score)


def _feature_reliance(
    estimator: Any,
    template: Any,
    matrix: np.ndarray,
    actual: list[Any],
    classes: list[Any],
    names: tuple[str, ...],
    donor_pools: Mapping[str, list[list[int]]],
    conditioning_features: Mapping[str, list[str]],
    bootstrap_samples: int,
    confidence_level: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    base_loss = _loss(estimator, template, matrix, actual, classes, names)
    output = []
    for feature_index, feature in enumerate(names):
        feature_donors = donor_pools[feature]
        boot_differences = []
        for _ in range(bootstrap_samples):
            indices = [rng.randrange(len(matrix)) for _ in matrix]
            sample = matrix[indices].copy()
            sample_y = [actual[index] for index in indices]
            perturbed = sample.copy()
            for row_position, source_index in enumerate(indices):
                candidates = feature_donors[source_index]
                donor = rng.choice(candidates)
                perturbed[row_position, feature_index] = matrix[donor, feature_index]
            delta = _loss(
                estimator, template, perturbed, sample_y, classes, names
            )["loss"] - _loss(
                estimator, template, sample, sample_y, classes, names
            )["loss"]
            boot_differences.append(float(delta))
        mean_delta = float(np.mean(boot_differences))
        output.append(
            {
                "feature": feature,
                "loss_increase": mean_delta,
                "loss_increase_confidence_interval": _bootstrap_interval(
                    boot_differences, confidence_level
                ),
                "positive_lower_bound": _bootstrap_interval(
                    boot_differences, confidence_level
                )[0]
                > 0,
                "metric": base_loss["metric"],
                "baseline_loss": base_loss["loss"],
                "perturbation_method": (
                    "nearest-neighbor conditional donor resampling"
                    if conditioning_features[feature]
                    else "marginal row donor resampling"
                ),
                "conditioned_on_features": conditioning_features[feature],
                "bootstrap_samples": bootstrap_samples,
            }
        )
    return output


def _loss(
    estimator: Any,
    template: Any,
    matrix: np.ndarray,
    actual: list[Any],
    classes: list[Any],
    names: tuple[str, ...],
) -> dict[str, Any]:
    model_input = _restore_input(template, matrix, names)
    predictions = list(estimator.predict(model_input))
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(model_input), dtype=float)
        positions = {_value_key(label): index for index, label in enumerate(classes)}
        target = np.zeros_like(probabilities)
        target[np.arange(len(actual)), [positions[_value_key(label)] for label in actual]] = 1
        loss = float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))
        return {"metric": "multiclass_brier_loss", "loss": loss}
    loss = float(
        np.mean(
            [
                _value_key(prediction) != _value_key(label)
                for prediction, label in zip(predictions, actual)
            ]
        )
    )
    return {"metric": "classification_error", "loss": loss}


def _nearest_donors(conditioning: np.ndarray, neighbors: int) -> list[list[int]]:
    scale = np.std(conditioning, axis=0)
    scale[scale == 0] = 1
    standardized = conditioning / scale
    distances = np.sum(
        (standardized[:, None, :] - standardized[None, :, :]) ** 2, axis=2
    )
    output = []
    for index, row in enumerate(distances):
        order = np.argsort(row)
        candidates = [int(value) for value in order if value != index][:neighbors]
        output.append(candidates or [index])
    return output


def _marginal_donors(row_count: int) -> list[list[int]]:
    return [
        [donor for donor in range(row_count) if donor != source] or [source]
        for source in range(row_count)
    ]


def _stability_slices(
    estimator: Any,
    template: Any,
    matrix: np.ndarray,
    actual: list[Any],
    classes: list[Any],
    names: tuple[str, ...],
    donor_pools: Mapping[str, list[list[int]]],
    groups: Mapping[str, Sequence[Any]] | None,
    time_periods: Sequence[Any] | None,
    rng: random.Random,
    bootstrap_samples: int,
    confidence_level: float,
) -> dict[str, Any]:
    reports: dict[str, Any] = {"groups": {}, "time_periods": {}}
    slice_definitions = []
    feature_estimates: dict[str, list[float]] = {name: [] for name in names}
    if groups is not None:
        for group_name, values in groups.items():
            if len(values) != len(actual):
                raise ValueError(
                    f"Group values for {group_name!r} must align with X."
                )
            if any(_is_missing_label(value) for value in values):
                raise ValueError(f"Group values for {group_name!r} contain missing values.")
            slice_definitions.append(("groups", group_name, list(values)))
    if time_periods is not None:
        if len(time_periods) != len(actual):
            raise ValueError("time_periods must contain one value per X row.")
        if any(_is_missing_label(value) for value in time_periods):
            raise ValueError("time_periods contains missing values.")
        slice_definitions.append(("time_periods", "time_period", list(time_periods)))

    for report_key, variable, values in slice_definitions:
        category_keys: dict[str, Any] = {}
        for value in values:
            category_keys.setdefault(_value_key(value), value)
        slice_output = []
        for key, category in category_keys.items():
            selected = [
                index
                for index, value in enumerate(values)
                if _value_key(value) == key
            ]
            if len(selected) < 2:
                continue
            batch_matrix = matrix[selected]
            batch_y = [actual[index] for index in selected]
            reliance = {}
            for feature_index, name in enumerate(names):
                differences = []
                for _ in range(bootstrap_samples):
                    local_indices = [
                        rng.randrange(len(selected)) for _ in selected
                    ]
                    sampled = batch_matrix[local_indices].copy()
                    sampled_y = [batch_y[index] for index in local_indices]
                    permuted = sampled.copy()
                    for local_index, sample_index in enumerate(local_indices):
                        source_index = selected[sample_index]
                        allowed = [
                            donor
                            for donor in donor_pools[name][source_index]
                            if donor in selected and donor != source_index
                        ]
                        if not allowed:
                            allowed = [
                                donor for donor in selected if donor != source_index
                            ]
                        donor = rng.choice(allowed)
                        permuted[local_index, feature_index] = batch_matrix[
                            selected.index(donor), feature_index
                        ]
                    differences.append(
                        _loss(
                            estimator, template, permuted, sampled_y, classes, names
                        )["loss"]
                        - _loss(
                            estimator, template, sampled, sampled_y, classes, names
                        )["loss"]
                    )
                estimate = float(np.mean(differences))
                feature_estimates[name].append(estimate)
                reliance[name] = {
                    "loss_increase": estimate,
                    "loss_increase_confidence_interval": _bootstrap_interval(
                        differences, confidence_level
                    ),
                    "status": "ok",
                }
            slice_output.append(
                {
                    "value": _native_scalar(category),
                    "sample_count": len(selected),
                    "feature_reliance": reliance,
                    "perturbation_note": (
                        "Conditioned donor resampling is restricted to this slice "
                        "when suitable donor rows are available; otherwise donor "
                        "resampling falls back to within-slice marginal shuffling."
                    ),
                }
            )
        reports[report_key][str(variable)] = slice_output

    reports["features"] = {
        name: {
            "slice_count": len(estimates),
            "between_slice_standard_deviation": (
                float(np.std(estimates, ddof=1)) if len(estimates) > 1 else None
            ),
            "slice_loss_increases": estimates,
            "interpretation": (
                "Descriptive heterogeneity across supplied strata/periods; "
                "small slices are noisy and are not inferentially adjusted."
            ),
        }
        for name, estimates in feature_estimates.items()
    } if slice_definitions else {}
    return reports


def _counterfactual_sensitivity(
    estimator: Any,
    template: Any,
    matrix: np.ndarray,
    base_probabilities: np.ndarray,
    base_predictions: list[Any],
    names: tuple[str, ...],
    counterfactuals: Mapping[str, Sequence[Any]] | None,
    positive_index: int,
    bootstrap_samples: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, Any]:
    output = {}
    rng = random.Random(random_seed)
    for feature, replacements in (counterfactuals or {}).items():
        if feature not in names:
            raise ValueError(f"Counterfactual feature {feature!r} is not in X.")
        if len(replacements) != len(matrix):
            raise ValueError(
                f"Counterfactual values for {feature!r} must align with X rows."
            )
        if any(not isinstance(value, Real) or not math.isfinite(float(value)) for value in replacements):
            raise ValueError(
                f"Counterfactual values for {feature!r} must be finite numeric values."
            )
        index = names.index(feature)
        changed = matrix.copy()
        changed[:, index] = [float(value) for value in replacements]
        changed_predictions = list(estimator.predict(_restore_input(template, changed, names)))
        changed_probabilities = np.asarray(
            estimator.predict_proba(_restore_input(template, changed, names)),
            dtype=float,
        )
        flips = np.asarray(
            [
                _value_key(before) != _value_key(after)
                for before, after in zip(base_predictions, changed_predictions)
            ],
            dtype=float,
        )
        probability_changes = np.abs(
            base_probabilities[:, positive_index]
            - changed_probabilities[:, positive_index]
        )
        flip_bootstrap = []
        probability_bootstrap = []
        for _ in range(bootstrap_samples):
            indices = [rng.randrange(len(matrix)) for _ in matrix]
            flip_bootstrap.append(float(np.mean(flips[indices])))
            probability_bootstrap.append(float(np.mean(probability_changes[indices])))
        output[feature] = {
            "prediction_flip_rate": float(np.mean(flips)),
            "prediction_flip_rate_confidence_interval": _bootstrap_interval(
                flip_bootstrap, confidence_level
            ),
            "mean_absolute_positive_probability_change": float(
                np.mean(probability_changes)
            ),
            "mean_absolute_probability_change_confidence_interval": _bootstrap_interval(
                probability_bootstrap, confidence_level
            ),
            "sample_count": len(matrix),
            "status": "ok",
        }
    return {
        "features": output,
        "interpretation": (
            "Only user-supplied replacement values are tested. Their domain "
            "validity is the caller's responsibility."
        ),
    }


def _logistic_associations(
    matrix: np.ndarray,
    actual: list[Any],
    names: tuple[str, ...],
    positive_label: Any,
) -> dict[str, Any]:
    response = np.asarray(
        [_value_key(value) == _value_key(positive_label) for value in actual],
        dtype=int,
    )
    if len(np.unique(response)) != 2:
        return {
            "status": "unavailable",
            "reason": "Both positive and negative labels are needed to fit logistic regression.",
        }
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=1)
    active = np.flatnonzero(scales > 0)
    if len(active) == 0:
        return {
            "status": "unavailable",
            "reason": "No non-constant numeric features are available.",
        }
    standardized = (matrix[:, active] - means[active]) / scales[active]
    design = sm.add_constant(standardized, has_constant="add")
    try:
        result = sm.Logit(response, design).fit(disp=False)
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as error:
        return {
            "status": "unavailable",
            "reason": f"Logistic regression could not estimate stable coefficients: {error}",
        }
    if not result.mle_retvals.get("converged", False) or not np.isfinite(
        result.params
    ).all():
        return {
            "status": "unavailable",
            "reason": "Logistic regression did not converge to finite coefficients.",
        }
    intervals = result.conf_int()
    by_feature = {}
    for position, feature_index in enumerate(active, start=1):
        coefficient = float(result.params[position])
        lower = float(intervals[position, 0])
        upper = float(intervals[position, 1])
        if max(abs(coefficient), abs(lower), abs(upper)) > 700:
            return {
                "status": "unavailable",
                "reason": (
                    "Logistic regression coefficients are too large for stable "
                    "odds-ratio confidence intervals; possible separation."
                ),
            }
        by_feature[names[feature_index]] = {
            "odds_ratio_per_one_standard_deviation": math.exp(coefficient),
            "odds_ratio_confidence_interval": [
                math.exp(lower),
                math.exp(upper),
            ],
            "coefficient": coefficient,
            "p_value": float(result.pvalues[position]),
            "standard_error": float(result.bse[position]),
        }
    adjusted = multipletests(
        [item["p_value"] for item in by_feature.values()],
        method="fdr_bh",
    )[1]
    for item, adjusted_p_value in zip(by_feature.values(), adjusted):
        item["fdr_bh_adjusted_p_value"] = float(adjusted_p_value)
    return {
        "status": "ok",
        "method": "unregularized logistic regression with standardized numeric features",
        "sample_count": len(actual),
        "pseudo_r_squared_mcfadden": float(result.prsquared),
        "odds_ratios": by_feature,
        "standardization": {
            "features": [names[index] for index in active],
            "means": [float(means[index]) for index in active],
            "standard_deviations": [float(scales[index]) for index in active],
        },
        "interpretation": (
            "Odds ratios are conditional associations with the observed label, "
            "not causal effects or proof that the fitted model used those reasons."
        ),
    }


def _logistic_explanation_alignment(
    explanations: ExplanationReport,
    matrix: np.ndarray,
    names: tuple[str, ...],
    logistic: Mapping[str, Any],
) -> dict[str, Any]:
    if logistic.get("status") != "ok":
        return {
            "status": "unavailable",
            "reason": "Logistic association model was unavailable.",
        }
    means = np.asarray(logistic["standardization"]["means"])
    scales = np.asarray(logistic["standardization"]["standard_deviations"])
    coefficients = logistic["odds_ratios"]
    active_names = list(coefficients)
    scales_by_name = dict(zip(active_names, scales))
    means_by_name = dict(zip(active_names, means))
    correlations = []
    direction_agreements = []
    local_results = []
    for explanation in explanations.local_explanations:
        row_index = explanation["row_index"]
        shap_values = explanation["shap_values"]
        logistic_contributions = {
            name: coefficients[name]["coefficient"]
            * (matrix[row_index, names.index(name)] - means_by_name[name])
            / scales_by_name[name]
            for name in active_names
        }
        common = list(active_names)
        correlation = _spearman(
            [abs(float(shap_values[name])) for name in common],
            [abs(logistic_contributions[name]) for name in common],
        )
        if correlation is not None:
            correlations.append(correlation)
        directions = [
            math.copysign(1, float(shap_values[name]))
            == math.copysign(1, logistic_contributions[name])
            for name in common
            if float(shap_values[name]) != 0 and logistic_contributions[name] != 0
        ]
        agreement = float(np.mean(directions)) if directions else None
        if agreement is not None:
            direction_agreements.append(agreement)
        local_results.append(
            {
                "row_index": row_index,
                "absolute_contribution_rank_spearman": correlation,
                "direction_agreement": agreement,
                "logistic_log_odds_contributions": logistic_contributions,
            }
        )
    return {
        "status": "ok",
        "per_row": local_results,
        "mean_absolute_contribution_rank_spearman": (
            float(np.mean(correlations)) if correlations else None
        ),
        "mean_direction_agreement": (
            float(np.mean(direction_agreements)) if direction_agreements else None
        ),
        "compared_feature_count": len(active_names),
        "interpretation": (
            "Logistic associations on observed labels are compared with fitted-model "
            "Shapley contributions. Agreement is consistency evidence only, not "
            "correctness, causal justification, or evidence against confounding."
        ),
    }


def _supplied_importance_alignment(
    supplied: Mapping[str, Real] | None,
    names: tuple[str, ...],
    explanations: ExplanationReport | None,
) -> dict[str, Any]:
    if supplied is None:
        return {"status": "not_provided"}
    unknown = set(supplied) - set(names)
    if unknown:
        raise ValueError(
            f"feature_importance contains unknown features: {sorted(unknown)}."
        )
    if not supplied:
        raise ValueError("feature_importance must not be empty.")
    for feature, value in supplied.items():
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(
                f"Importance score for {feature!r} must be finite and numeric."
            )
    if explanations is None:
        return {
            "status": "unavailable",
            "reason": "SHAP/Shapley explanations were unavailable.",
        }
    shap_values = {
        item["feature"]: item["mean_absolute_shapley_value"]
        for item in explanations.global_shap_importance
    }
    common = sorted(set(supplied) & set(shap_values))
    correlation = _spearman(
        [abs(float(supplied[name])) for name in common],
        [shap_values[name] for name in common],
    )
    return {
        "status": "ok" if correlation is not None else "unavailable",
        "source": "user-supplied; source/method not independently validated",
        "compared_features": common,
        "absolute_importance_spearman": correlation,
        "reason": (
            None
            if correlation is not None
            else "At least two non-constant shared feature scores are required."
        ),
        "interpretation": (
            "Ranking agreement does not establish that either importance estimate "
            "is correct, causal, or free from spurious correlations."
        ),
    }


def _validate_named_features(features: set[str], names: tuple[str, ...]) -> None:
    unknown = features - set(names)
    if unknown:
        raise ValueError(f"Unknown feature names: {sorted(unknown)}.")


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    result = spearmanr(left, right)
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def _bootstrap_interval(
    values: Sequence[float], confidence_level: float
) -> list[float]:
    ordered = sorted(float(value) for value in values)
    tail = (1 - confidence_level) / 2
    return [_quantile(ordered, tail), _quantile(ordered, 1 - tail)]


def _quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _value_key(value: Any) -> str:
    return _category_key(_native_scalar(value))


def _native_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _is_missing_label(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, Real):
        return not math.isfinite(float(value))
    return False
