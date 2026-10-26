from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from sklearn.metrics import log_loss
from statsmodels.multivariate.manova import MANOVA

from .drift import (
    _FeatureProfile,
    _as_columns,
    _category_key,
    _is_missing,
    _validate_columns,
)
from .fairness import assess_fairness


@dataclass(frozen=True)
class EvaluationReport:
    sample_count: int
    classification: Mapping[str, Any]
    statistical_tests: Mapping[str, Any]
    bias_variance: Mapping[str, Any] | None
    fairness: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "classification": dict(self.classification),
            "statistical_tests": dict(self.statistical_tests),
            "bias_variance": (
                None if self.bias_variance is None else dict(self.bias_variance)
            ),
            "fairness": dict(self.fairness),
            "limitations": [
                "Performance metrics require true labels for the evaluated batch.",
                "Statistical test p-values are assumption-dependent and are not evidence of causal effects.",
                "Bias/variance values are empirical repeated-CV Brier-error decomposition, not population-level irreducible-noise estimates.",
            ],
        }


def evaluate_predictions(
    estimator: Any,
    profiles: tuple[_FeatureProfile, ...],
    X_test: Any,
    y_true: Any,
    y_pred: Any,
    folds: int,
    repeats: int,
    random_seed: int,
    sensitive_features: Mapping[str, Any] | None = None,
    positive_label: Any | None = None,
    bootstrap_samples: int = 500,
    confidence_level: float = 0.95,
) -> EvaluationReport:
    columns = _as_columns(X_test)
    _validate_columns(columns, profiles)
    actual = list(y_true)
    predictions = list(y_pred)
    sample_count = len(next(iter(columns.values())))
    if len(actual) != sample_count or len(predictions) != sample_count:
        raise ValueError("X_test, y_true, and predictions must have equal row counts.")
    if sample_count == 0:
        raise ValueError("Evaluation requires at least one labeled test row.")
    if any(_is_missing(label) for label in actual):
        raise ValueError("y_true must not contain missing labels.")
    if any(_is_missing(label) for label in predictions):
        raise ValueError("Predictions must not contain missing labels.")

    classification = _classification_metrics(actual, predictions)
    classification["probability_calibration"] = _probability_calibration(
        estimator,
        X_test,
        actual,
    )
    target_tests = _feature_target_tests(columns, profiles, actual)
    cohort_tests = _train_test_tests(columns, profiles)
    bias_variance = _repeated_cv_bias_variance(
        estimator,
        X_test,
        actual,
        folds,
        repeats,
        random_seed,
    )
    observed_labels = {
        _value_key(label): label for label in [*actual, *predictions]
    }
    fairness_labels = (
        list(estimator.classes_)
        if hasattr(estimator, "classes_")
        else list(observed_labels.values())
    )
    fairness = (
        assess_fairness(
            actual,
            predictions,
            sensitive_features,
            positive_label,
            fairness_labels,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        if sensitive_features is not None
        else {"status": "not_requested"}
    )
    return EvaluationReport(
        sample_count=sample_count,
        classification=classification,
        statistical_tests={
            "feature_target_associations": target_tests,
            "training_vs_test_cohorts": cohort_tests,
        },
        bias_variance=bias_variance,
        fairness=fairness,
    )


def _classification_metrics(y_true: list[Any], y_pred: list[Any]) -> dict[str, Any]:
    labels_by_key: dict[str, Any] = {}
    for label in [*y_true, *y_pred]:
        labels_by_key.setdefault(_value_key(label), label)
    keys = sorted(labels_by_key)
    index = {key: position for position, key in enumerate(keys)}
    matrix = [[0 for _ in keys] for _ in keys]
    for actual, predicted in zip(y_true, y_pred):
        matrix[index[_value_key(actual)]][index[_value_key(predicted)]] += 1

    total = len(y_true)
    correct = sum(matrix[position][position] for position in range(len(keys)))
    per_class = []
    for class_index, key in enumerate(keys):
        true_positive = matrix[class_index][class_index]
        false_positive = sum(
            matrix[row][class_index]
            for row in range(len(keys))
            if row != class_index
        )
        false_negative = sum(
            matrix[class_index][column]
            for column in range(len(keys))
            if column != class_index
        )
        true_negative = total - true_positive - false_positive - false_negative
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        specificity = _ratio(true_negative, true_negative + false_positive)
        f1 = _ratio(2 * precision * recall, precision + recall)
        support = sum(matrix[class_index])
        per_class.append(
            {
                "label": str(labels_by_key[key]),
                "support": support,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
            }
        )

    support_total = sum(item["support"] for item in per_class)
    macro = {
        metric: sum(item[metric] for item in per_class) / len(per_class)
        for metric in ("precision", "recall", "specificity", "f1")
    }
    weighted = {
        metric: sum(item[metric] * item["support"] for item in per_class)
        / support_total
        for metric in ("precision", "recall", "specificity", "f1")
    }
    true_positive_sum = correct
    actual_sums = [sum(row) for row in matrix]
    predicted_sums = [
        sum(matrix[row][column] for row in range(len(keys)))
        for column in range(len(keys))
    ]
    numerator = true_positive_sum * total - sum(
        actual * predicted
        for actual, predicted in zip(actual_sums, predicted_sums)
    )
    denominator = math.sqrt(
        (total * total - sum(value * value for value in predicted_sums))
        * (total * total - sum(value * value for value in actual_sums))
    )
    return {
        "labels": [str(labels_by_key[key]) for key in keys],
        "confusion_matrix": matrix,
        "accuracy": correct / total,
        "balanced_accuracy": macro["recall"],
        "matthews_correlation_coefficient": (
            numerator / denominator if denominator else 0.0
        ),
        "macro_average": macro,
        "weighted_average": weighted,
        "per_class": per_class,
    }


def _probability_calibration(
    estimator: Any,
    X: Any,
    y_true: list[Any],
    confidence_bins: int = 10,
) -> dict[str, Any]:
    predict_proba = getattr(estimator, "predict_proba", None)
    classes = getattr(estimator, "classes_", None)
    if not callable(predict_proba):
        return {
            "status": "unavailable",
            "reason": "The fitted estimator does not implement predict_proba.",
        }
    if classes is None:
        raise ValueError(
            "The fitted estimator implements predict_proba but has no classes_ attribute."
        )

    class_labels = list(classes)
    class_keys = [_value_key(label) for label in class_labels]
    if len(class_labels) < 2 or len(set(class_keys)) != len(class_labels):
        raise ValueError("estimator.classes_ must contain at least two unique labels.")
    probabilities = np.asarray(predict_proba(X), dtype=float)
    if probabilities.shape != (len(y_true), len(class_labels)):
        raise ValueError(
            "predict_proba output must have one row per evaluated sample and "
            "one column per estimator class."
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(
        probabilities > 1
    ):
        raise ValueError("predict_proba output must contain finite values in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError("predict_proba rows must sum to 1.")

    class_indices = {key: index for index, key in enumerate(class_keys)}
    try:
        actual_indices = np.asarray(
            [class_indices[_value_key(label)] for label in y_true],
            dtype=int,
        )
    except KeyError as error:
        raise ValueError(
            "y_true contains a label that is absent from estimator.classes_."
        ) from error
    one_hot = np.eye(len(class_labels), dtype=float)[actual_indices]
    mean_classwise_brier = float(np.mean((probabilities - one_hot) ** 2))
    multiclass_log_loss = float(
        log_loss(actual_indices, probabilities, labels=np.arange(len(class_labels)))
    )

    predicted_indices = probabilities.argmax(axis=1)
    confidences = probabilities[np.arange(len(y_true)), predicted_indices]
    correct = predicted_indices == actual_indices
    bin_indices = np.minimum(
        (confidences * confidence_bins).astype(int),
        confidence_bins - 1,
    )
    bins = []
    expected_calibration_error = 0.0
    for bin_index in range(confidence_bins):
        members = bin_indices == bin_index
        count = int(members.sum())
        if not count:
            continue
        mean_confidence = float(confidences[members].mean())
        accuracy = float(correct[members].mean())
        expected_calibration_error += (
            count / len(y_true) * abs(mean_confidence - accuracy)
        )
        bins.append(
            {
                "lower_bound": bin_index / confidence_bins,
                "upper_bound": (bin_index + 1) / confidence_bins,
                "sample_count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    return {
        "status": "ok",
        "mean_classwise_brier_score": mean_classwise_brier,
        "log_loss": multiclass_log_loss,
        "top_label_expected_calibration_error": float(
            expected_calibration_error
        ),
        "confidence_bins": bins,
        "interpretation": (
            "Lower scores indicate better probability calibration on this "
            "labeled evaluation batch. The expected calibration error uses "
            "equal-width bins and top-label confidence; results depend on the "
            "evaluation sample and bin count."
        ),
    }


def _feature_target_tests(
    columns: Mapping[str, list[Any]],
    profiles: tuple[_FeatureProfile, ...],
    labels: list[Any],
) -> dict[str, Any]:
    numeric = []
    categorical = []
    for profile in profiles:
        values = columns[profile.name]
        if profile.numeric:
            result = _anova(values, labels)
            numeric.append({"feature": profile.name, **result})
        else:
            result = _categorical_association(values, labels)
            categorical.append({"feature": profile.name, **result})
    return {
        "anova_numeric_features": numeric,
        "manova_numeric_features": _manova(
            {profile.name: columns[profile.name] for profile in profiles if profile.numeric},
            labels,
            "target_class",
        ),
        "chi_square_categorical_features": categorical,
        "interpretation": (
            "Tests whether numeric feature means or categorical feature "
            "frequencies differ across observed target classes."
        ),
    }


def _train_test_tests(
    columns: Mapping[str, list[Any]],
    profiles: tuple[_FeatureProfile, ...],
) -> dict[str, Any]:
    numeric = []
    categorical = []
    for profile in profiles:
        incoming = columns[profile.name]
        reference = list(profile.reference_observations)
        values = reference + incoming
        cohorts = ["training"] * len(reference) + ["test"] * len(incoming)
        if profile.numeric:
            numeric.append(
                {"feature": profile.name, **_anova(values, cohorts)}
            )
        else:
            categorical.append(
                {
                    "feature": profile.name,
                    **_categorical_association(values, cohorts),
                }
            )
    combined_numeric = {
        profile.name: list(profile.reference_observations) + columns[profile.name]
        for profile in profiles
        if profile.numeric
    }
    combined_cohorts = [
        *["training"] * len(profiles[0].reference_observations),
        *["test"] * len(next(iter(columns.values()))),
    ]
    return {
        "anova_numeric_features": numeric,
        "manova_numeric_features": _manova(
            combined_numeric, combined_cohorts, "cohort"
        ),
        "chi_square_categorical_features": categorical,
        "interpretation": (
            "Tests train-versus-test cohort differences; it complements the "
            "per-feature drift statistics and does not measure performance."
        ),
    }


def _anova(values: list[Any], groups: list[Any]) -> dict[str, Any]:
    keyed: dict[str, list[float]] = {}
    for value, group in zip(values, groups):
        if _is_missing(value) or not isinstance(value, Real):
            continue
        keyed.setdefault(_value_key(group), []).append(float(value))
    valid = [group for group in keyed.values() if group]
    if len(valid) < 2:
        return _unavailable("ANOVA requires at least two non-empty groups.")
    if any(len(group) < 2 for group in valid):
        return _unavailable(
            "ANOVA needs at least two numeric observations in each group."
        )
    result = f_oneway(*valid)
    included_count = sum(map(len, valid))
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "degrees_of_freedom_between": len(valid) - 1,
        "degrees_of_freedom_within": included_count - len(valid),
        "sample_count": included_count,
        "excluded_count": len(values) - included_count,
        "group_sizes": [len(group) for group in valid],
        "assumptions_not_tested": [
            "Observations are independent.",
            "Residuals are approximately normal within each group.",
            "Group variances are approximately equal.",
        ],
        "status": "ok",
    }


def _categorical_association(
    values: list[Any], groups: list[Any]
) -> dict[str, Any]:
    valid = [
        (value, group)
        for value, group in zip(values, groups)
        if not _is_missing(value) and not _is_missing(group)
    ]
    value_keys = sorted({_value_key(value) for value, _ in valid})
    group_keys = sorted({_value_key(group) for _, group in valid})
    if len(value_keys) < 2 or len(group_keys) < 2:
        return _unavailable(
            "Chi-square association requires at least two categories per variable."
        )
    value_index = {key: index for index, key in enumerate(value_keys)}
    group_index = {key: index for index, key in enumerate(group_keys)}
    table = [[0 for _ in group_keys] for _ in value_keys]
    for value, group in valid:
        table[value_index[_value_key(value)]][group_index[_value_key(group)]] += 1
    statistic, p_value, degrees, expected = chi2_contingency(table, correction=False)
    expected_values = [item for row in expected for item in row]
    low_expected_count = sum(value < 5 for value in expected_values)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees),
        "contingency_table": table,
        "expected_counts": expected.tolist(),
        "sample_count": len(valid),
        "expected_cells_below_5": int(low_expected_count),
        "assumptions_not_tested": ["Observations are independent."],
        "assumption_warning": (
            "Some expected cell counts are below 5."
            if low_expected_count
            else None
        ),
        "status": "ok",
    }


def _manova(
    numeric_features: Mapping[str, list[Any]],
    groups: list[Any],
    group_name: str,
) -> dict[str, Any]:
    import pandas as pd

    valid_rows = []
    feature_names = list(numeric_features)
    for row_index, group in enumerate(groups):
        if _is_missing(group):
            continue
        values = [numeric_features[name][row_index] for name in feature_names]
        if any(
            _is_missing(value) or not isinstance(value, Real)
            for value in values
        ):
            continue
        valid_rows.append(
            (_value_key(group), [float(value) for value in values])
        )
    group_values = sorted({group for group, _ in valid_rows})
    if len(feature_names) < 2:
        return _unavailable("MANOVA requires at least two numeric features.")
    if len(group_values) < 2:
        return _unavailable("MANOVA requires at least two comparison groups.")
    if len(valid_rows) <= len(feature_names) + len(group_values):
        return _unavailable("Insufficient complete observations for MANOVA.")

    grouped_values: dict[str, list[list[float]]] = {}
    for group, values in valid_rows:
        grouped_values.setdefault(group, []).append(values)
    residual_rows = []
    for values in grouped_values.values():
        group_matrix = np.asarray(values, dtype=float)
        residual_rows.extend(group_matrix - group_matrix.mean(axis=0))
    residual_matrix = np.asarray(residual_rows, dtype=float)
    if np.linalg.matrix_rank(residual_matrix) < len(feature_names):
        return _unavailable(
            "MANOVA residual covariance is singular; dependent features or "
            "insufficient within-group variation prevent a stable multivariate test."
        )

    frame = pd.DataFrame(
        [
            [*values, group]
            for group, values in valid_rows
        ],
        columns=[
            *[f"feature_{index}" for index in range(len(feature_names))],
            "group",
        ],
    )
    response_formula = " + ".join(
        f"feature_{index}" for index in range(len(feature_names))
    )
    try:
        result = MANOVA.from_formula(
            f"{response_formula} ~ C(group)", data=frame
        ).mv_test()
    except (np.linalg.LinAlgError, ValueError) as error:
        return _unavailable(f"MANOVA could not estimate a stable test: {error}")
    term = "C(group)"
    if term not in result.results:
        return _unavailable("MANOVA could not construct the cohort/class effect.")
    table = result.results[term]["stat"]
    return {
        "dependent_features": feature_names,
        "group": group_name,
        "group_count": len(group_values),
        "sample_count": len(valid_rows),
        "wilks_lambda": float(table.loc["Wilks' lambda", "Value"]),
        "f_statistic": float(table.loc["Wilks' lambda", "F Value"]),
        "p_value": float(table.loc["Wilks' lambda", "Pr > F"]),
        "numerator_df": float(table.loc["Wilks' lambda", "Num DF"]),
        "denominator_df": float(table.loc["Wilks' lambda", "Den DF"]),
        "assumptions_not_tested": [
            "Observations are independent.",
            "Multivariate residuals are approximately normal within groups.",
            "Group covariance matrices are approximately equal.",
        ],
        "status": "ok",
    }


def _repeated_cv_bias_variance(
    estimator: Any,
    X: Any,
    labels: list[Any],
    folds: int,
    repeats: int,
    random_seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(_value_key(label), []).append(index)
    minimum_class_size = min(map(len, grouped.values()))
    actual_folds = min(folds, minimum_class_size)
    if actual_folds < 2:
        return {
            "status": "unavailable",
            "reason": "Repeated stratified CV requires at least two rows per class.",
            "folds": actual_folds,
            "repeats": repeats,
        }

    rng = random.Random(random_seed)
    predictions_by_row: list[list[list[float]]] = [[] for _ in labels]
    label_keys = sorted(grouped)
    for _ in range(repeats):
        fold_indices: list[list[int]] = [[] for _ in range(actual_folds)]
        for indices in grouped.values():
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            for offset, row_index in enumerate(shuffled):
                fold_indices[offset % actual_folds].append(row_index)
        for test_indices in fold_indices:
            test_set = set(test_indices)
            training_indices = [
                index for index in range(len(labels)) if index not in test_set
            ]
            model = copy.deepcopy(estimator)
            model.fit(_take_rows(X, training_indices), _take_values(labels, training_indices))
            test_X = _take_rows(X, test_indices)
            hard_predictions = list(model.predict(test_X))
            probabilities = _probabilities(model, test_X, hard_predictions, label_keys)
            for row_index, probability in zip(test_indices, probabilities):
                predictions_by_row[row_index].append(probability)

    bias_squared_total = 0.0
    variance_total = 0.0
    error_total = 0.0
    for row_index, repeated_predictions in enumerate(predictions_by_row):
        mean_prediction = [
            sum(prediction[index] for prediction in repeated_predictions)
            / len(repeated_predictions)
            for index in range(len(label_keys))
        ]
        target_key = _value_key(labels[row_index])
        target = [float(key == target_key) for key in label_keys]
        bias_squared_total += sum(
            (prediction - expected) ** 2
            for prediction, expected in zip(mean_prediction, target)
        )
        variance_total += sum(
            sum(
                (prediction[index] - mean_prediction[index]) ** 2
                for index in range(len(label_keys))
            )
            / len(repeated_predictions)
            for prediction in repeated_predictions
        )
        error_total += sum(
            sum(
                (prediction[index] - target[index]) ** 2
                for index in range(len(label_keys))
            )
            for prediction in repeated_predictions
        ) / len(repeated_predictions)

    sample_count = len(labels)
    return {
        "status": "ok",
        "method": "repeated_stratified_cv_brier_decomposition",
        "folds": actual_folds,
        "repeats": repeats,
        "estimator_fits": actual_folds * repeats,
        "bias_squared": bias_squared_total / sample_count,
        "prediction_variance": variance_total / sample_count,
        "mean_squared_probability_error": error_total / sample_count,
        "interpretation": (
            "Empirical one-hot/Brier error decomposition from out-of-fold "
            "predictions; not a population irreducible-noise decomposition."
        ),
        "probabilities": (
            "Estimator predict_proba is used when available; otherwise hard "
            "predictions are one-hot encoded."
        ),
    }


def _probabilities(
    estimator: Any,
    X: Any,
    predictions: list[Any],
    label_keys: list[str],
) -> list[list[float]]:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        classes = list(estimator.classes_)
        class_keys = [_value_key(value) for value in classes]
        return [
            [
                float(row[class_keys.index(key)]) if key in class_keys else 0.0
                for key in label_keys
            ]
            for row in probabilities
        ]
    return [
        [
            float(_value_key(predicted) == key)
            for key in label_keys
        ]
        for predicted in predictions
    ]


def _take_rows(X: Any, indices: list[int]) -> Any:
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    if isinstance(X, Mapping):
        return {
            name: _take_values(values, indices) for name, values in X.items()
        }
    return [X[index] for index in indices]


def _take_values(values: Any, indices: list[int]) -> list[Any]:
    return [values[index] for index in indices]


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _value_key(value: Any) -> str:
    scalar = value.item() if callable(getattr(value, "item", None)) else value
    return _category_key(scalar)


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}
