from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from .drift import _category_key, _is_missing


def _value_key(value: Any) -> str:
    scalar = value.item() if callable(getattr(value, "item", None)) else value
    return _category_key(scalar)


def assess_fairness(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    sensitive_features: Mapping[str, Sequence[Any]],
    positive_label: Any | None,
    labels: Sequence[Any],
    bootstrap_samples: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, Any]:
    actual = list(y_true)
    predicted = list(y_pred)
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have equal row counts.")
    if not isinstance(sensitive_features, Mapping) or not sensitive_features:
        raise ValueError("sensitive_features must be a non-empty mapping.")
    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples < 100
    ):
        raise ValueError("bootstrap_samples must be an integer of at least 100.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    for name, values in sensitive_features.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Sensitive feature names must be non-empty strings.")
        if isinstance(values, (str, bytes)):
            raise ValueError(f"Sensitive values for {name!r} must be a sequence.")
        if len(values) != len(actual):
            raise ValueError(
                f"Sensitive feature {name!r} must have one value per evaluated row."
            )
        if any(_is_missing(value) for value in values):
            raise ValueError(f"Sensitive feature {name!r} contains missing values.")

    label_keys = {_value_key(label) for label in labels}
    binary = len(label_keys) == 2
    if binary:
        if positive_label is None:
            raise ValueError(
                "positive_label is required for binary fairness evaluation."
            )
        if _value_key(positive_label) not in label_keys:
            raise ValueError("positive_label must be one of the observed true labels.")
        evaluated_labels = [positive_label]
    else:
        if positive_label is not None:
            raise ValueError(
                "positive_label is not used for multiclass fairness; "
                "all classes are assessed one-vs-rest."
            )
        evaluated_labels = list(labels)

    reports = {}
    for name, raw_groups in sensitive_features.items():
        group_keys: dict[str, Any] = {}
        for group in raw_groups:
            group_keys.setdefault(_value_key(group), group)
        if len(group_keys) < 2:
            reports[name] = {
                "status": "unavailable",
                "reason": "Fairness comparison requires at least two protected groups.",
            }
            continue
        group_indices = {
            key: [
                index
                for index, value in enumerate(raw_groups)
                if _value_key(value) == key
            ]
            for key in sorted(group_keys)
        }
        class_reports = {}
        for label in evaluated_labels:
            label_key = _value_key(label)
            class_reports[str(label)] = _class_fairness(
                actual,
                predicted,
                group_indices,
                group_keys,
                label_key,
                label,
                bootstrap_samples,
                confidence_level,
                random_seed,
            )
        reports[name] = {
            "status": "ok",
            "method": "binary" if binary else "one_vs_rest_per_class",
            "classes": class_reports,
        }
    return reports


def _class_fairness(
    actual: list[Any],
    predicted: list[Any],
    group_indices: Mapping[str, list[int]],
    group_keys: Mapping[str, Any],
    positive_key: str,
    positive_label: Any,
    bootstrap_samples: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, Any]:
    per_group = []
    group_rates: dict[str, dict[str, float | None]] = {}
    for group_key, indices in group_indices.items():
        positives = [
            index for index in indices if _value_key(actual[index]) == positive_key
        ]
        negatives = [
            index for index in indices if _value_key(actual[index]) != positive_key
        ]
        selected = [
            index for index in indices if _value_key(predicted[index]) == positive_key
        ]
        true_positives = [
            index for index in positives if _value_key(predicted[index]) == positive_key
        ]
        false_positives = [
            index for index in negatives if _value_key(predicted[index]) == positive_key
        ]
        values = {
            "selection_rate": _rate(len(selected), len(indices)),
            "true_positive_rate": _rate(len(true_positives), len(positives)),
            "false_positive_rate": _rate(len(false_positives), len(negatives)),
            "positive_predictive_value": _rate(len(true_positives), len(selected)),
        }
        group_rates[group_key] = values
        per_group.append(
            {
                "group": str(group_keys[group_key]),
                "sample_count": len(indices),
                "actual_positive_count": len(positives),
                "actual_negative_count": len(negatives),
                "predicted_positive_count": len(selected),
                **values,
            }
        )

    rng = random.Random(random_seed)
    selection_gap = _gap_summary(
        group_rates, "selection_rate", actual, predicted, group_indices,
        positive_key, "selection", bootstrap_samples, confidence_level, rng
    )
    tpr_gap = _gap_summary(
        group_rates, "true_positive_rate", actual, predicted, group_indices,
        positive_key, "tpr", bootstrap_samples, confidence_level, rng
    )
    fpr_gap = _gap_summary(
        group_rates, "false_positive_rate", actual, predicted, group_indices,
        positive_key, "fpr", bootstrap_samples, confidence_level, rng
    )
    ppv_gap = _gap_summary(
        group_rates, "positive_predictive_value", actual, predicted, group_indices,
        positive_key, "ppv", bootstrap_samples, confidence_level, rng
    )
    equalized_odds = _equalized_odds_summary(
        actual,
        predicted,
        group_indices,
        positive_key,
        bootstrap_samples,
        confidence_level,
        random.Random(random_seed),
    )
    return {
        "positive_label": str(positive_label),
        "per_group": per_group,
        "statistical_parity": selection_gap,
        "equalized_odds": {
            "true_positive_rate_gap": tpr_gap,
            "false_positive_rate_gap": fpr_gap,
            **equalized_odds,
        },
        "predictive_parity": ppv_gap,
        "interpretation": (
            "Gaps are max-minus-min across the supplied groups; zero indicates "
            "equal observed rates. Confidence intervals use within-group row bootstrap."
        ),
    }


def _equalized_odds_summary(
    actual: list[Any],
    predicted: list[Any],
    group_indices: Mapping[str, list[int]],
    positive_key: str,
    bootstrap_samples: int,
    confidence_level: float,
    rng: random.Random,
) -> dict[str, Any]:
    estimates = []
    for indices in group_indices.values():
        actual_positive = sum(
            _value_key(actual[index]) == positive_key for index in indices
        )
        actual_negative = len(indices) - actual_positive
        true_positive = sum(
            _value_key(actual[index]) == positive_key
            and _value_key(predicted[index]) == positive_key
            for index in indices
        )
        false_positive = sum(
            _value_key(actual[index]) != positive_key
            and _value_key(predicted[index]) == positive_key
            for index in indices
        )
        if actual_positive and actual_negative:
            estimates.append(
                (true_positive / actual_positive, false_positive / actual_negative)
            )
    if len(estimates) < 2:
        return {
            "maximum_gap": None,
            "maximum_gap_confidence_interval": None,
            "status": "unavailable",
            "reason": "At least two groups need both actual positive and negative rows.",
        }
    observed_tpr_gap = max(value[0] for value in estimates) - min(
        value[0] for value in estimates
    )
    observed_fpr_gap = max(value[1] for value in estimates) - min(
        value[1] for value in estimates
    )
    bootstrap_max_gaps = []
    for _ in range(bootstrap_samples):
        tpr_rates = []
        fpr_rates = []
        for indices in group_indices.values():
            sample = [rng.choice(indices) for _ in indices]
            positive_rows = [
                index for index in sample if _value_key(actual[index]) == positive_key
            ]
            negative_rows = [
                index for index in sample if _value_key(actual[index]) != positive_key
            ]
            if not positive_rows or not negative_rows:
                continue
            tpr_rates.append(
                sum(
                    _value_key(predicted[index]) == positive_key
                    for index in positive_rows
                )
                / len(positive_rows)
            )
            fpr_rates.append(
                sum(
                    _value_key(predicted[index]) == positive_key
                    for index in negative_rows
                )
                / len(negative_rows)
            )
        if len(tpr_rates) > 1 and len(fpr_rates) > 1:
            bootstrap_max_gaps.append(
                max(
                    max(tpr_rates) - min(tpr_rates),
                    max(fpr_rates) - min(fpr_rates),
                )
            )
    if not bootstrap_max_gaps:
        return {
            "maximum_gap": max(observed_tpr_gap, observed_fpr_gap),
            "maximum_gap_confidence_interval": None,
            "status": "unavailable",
            "reason": "Bootstrap samples did not yield defined rates for two groups.",
        }
    tail = (1 - confidence_level) / 2
    return {
        "maximum_gap": max(observed_tpr_gap, observed_fpr_gap),
        "maximum_gap_confidence_interval": [
            _quantile(bootstrap_max_gaps, tail),
            _quantile(bootstrap_max_gaps, 1 - tail),
        ],
        "status": "ok",
    }


def _gap_summary(
    group_rates: Mapping[str, Mapping[str, float | None]],
    metric: str,
    actual: list[Any],
    predicted: list[Any],
    group_indices: Mapping[str, list[int]],
    positive_key: str,
    metric_kind: str,
    bootstrap_samples: int,
    confidence_level: float,
    rng: random.Random,
) -> dict[str, Any]:
    available = [
        value[metric] for value in group_rates.values() if value[metric] is not None
    ]
    if len(available) < 2:
        return {
            "gap": None,
            "confidence_interval": None,
            "status": "unavailable",
            "reason": "Fewer than two groups have a defined rate for this metric.",
        }
    gap = max(available) - min(available)
    bootstrap_gaps = []
    for _ in range(bootstrap_samples):
        rates = []
        for indices in group_indices.values():
            sample = [rng.choice(indices) for _ in indices]
            selected = [
                index for index in sample if _value_key(predicted[index]) == positive_key
            ]
            positive = [
                index for index in sample if _value_key(actual[index]) == positive_key
            ]
            negative = [
                index for index in sample if _value_key(actual[index]) != positive_key
            ]
            true_positive = [
                index
                for index in positive
                if _value_key(predicted[index]) == positive_key
            ]
            false_positive = [
                index
                for index in negative
                if _value_key(predicted[index]) == positive_key
            ]
            if metric_kind == "selection":
                numerator, denominator = len(selected), len(sample)
            elif metric_kind == "tpr":
                numerator, denominator = len(true_positive), len(positive)
            elif metric_kind == "fpr":
                numerator, denominator = len(false_positive), len(negative)
            else:
                numerator, denominator = len(true_positive), len(selected)
            if denominator:
                rates.append(numerator / denominator)
        if len(rates) > 1:
            bootstrap_gaps.append(max(rates) - min(rates))
    if not bootstrap_gaps:
        return {
            "gap": float(gap),
            "confidence_interval": None,
            "status": "ok",
            "reason": "Bootstrap samples did not yield two defined group rates.",
        }
    tail = (1 - confidence_level) / 2
    return {
        "gap": float(gap),
        "confidence_interval": [
            _quantile(bootstrap_gaps, tail),
            _quantile(bootstrap_gaps, 1 - tail),
        ],
        "status": "ok",
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )
