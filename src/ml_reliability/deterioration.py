from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from typing import Any

from scipy.stats import linregress


@dataclass(frozen=True)
class DeteriorationReport:
    observations: tuple[Mapping[str, Any], ...]
    accuracy_slope: float
    slope_confidence_interval: tuple[float, float]
    p_value: float
    confidence_level: float
    bootstrap_samples: int
    deterioration_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [dict(observation) for observation in self.observations],
            "accuracy_slope": self.accuracy_slope,
            "slope_confidence_interval": list(self.slope_confidence_interval),
            "p_value": self.p_value,
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.bootstrap_samples,
            "deterioration_detected": self.deterioration_detected,
            "interpretation": (
                "Deterioration is flagged when the bootstrap confidence interval "
                "for the linear accuracy trend is entirely below zero."
            ),
            "limitations": [
                "Each dated batch contributes one equally weighted accuracy observation.",
                "The trend is associational and may reflect changing case mix or evaluation conditions.",
                "At least three distinct dates are required; temporal dependence is not modeled.",
            ],
        }


def assess_deterioration(
    performance_history: Sequence[Mapping[str, Any]],
    confidence_level: float = 0.95,
    bootstrap_samples: int = 1000,
    random_seed: int = 42,
) -> DeteriorationReport:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples < 100
    ):
        raise ValueError("bootstrap_samples must be an integer of at least 100.")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer.")
    if len(performance_history) < 3:
        raise ValueError("At least three dated performance batches are required.")

    parsed: list[tuple[float, dict[str, Any], list[Any], list[Any]]] = []
    time_kind: type | None = None
    for index, record in enumerate(performance_history):
        if not isinstance(record, Mapping):
            raise ValueError(f"Performance history item {index} must be a mapping.")
        if not {"timestamp", "y_true", "y_pred"} <= set(record):
            raise ValueError(
                f"Performance history item {index} needs timestamp, y_true, and y_pred."
            )
        true_values = list(record["y_true"])
        predicted_values = list(record["y_pred"])
        if not true_values or len(true_values) != len(predicted_values):
            raise ValueError(
                f"Performance history item {index} must have equal, non-empty label arrays."
            )
        if any(_missing(value) for value in true_values + predicted_values):
            raise ValueError(f"Performance history item {index} contains missing labels.")
        timestamp = record["timestamp"]
        if callable(getattr(timestamp, "item", None)):
            timestamp = timestamp.item()
        numeric_timestamp = _numeric_timestamp(timestamp)
        current_kind = float if numeric_timestamp is not None else str
        if time_kind is None:
            time_kind = current_kind
        elif time_kind is not current_kind:
            raise ValueError("Timestamps must all be numeric or all ISO date/time strings.")
        if numeric_timestamp is None:
            numeric_timestamp = _datetime_timestamp(timestamp)
        accuracy = sum(
            actual == predicted
            for actual, predicted in zip(true_values, predicted_values)
        ) / len(true_values)
        parsed.append(
            (
                numeric_timestamp,
                {
                    "timestamp": (
                        float(timestamp)
                        if isinstance(timestamp, Real)
                        else timestamp.isoformat()
                        if isinstance(timestamp, (date, datetime))
                        else timestamp
                    ),
                    "sample_count": len(true_values),
                    "accuracy": accuracy,
                },
                true_values,
                predicted_values,
            )
        )

    parsed.sort(key=lambda entry: entry[0])
    time_values = [entry[0] for entry in parsed]
    if len(set(time_values)) < 3:
        raise ValueError("At least three distinct timestamps are required.")
    accuracies = [entry[1]["accuracy"] for entry in parsed]
    regression = linregress(time_values, accuracies)

    rng = random.Random(random_seed)
    bootstrap_slopes: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_accuracies = []
        for _, _, true_values, predicted_values in parsed:
            sampled_indices = [
                rng.randrange(len(true_values)) for _ in range(len(true_values))
            ]
            sampled_accuracies.append(
                sum(
                    true_values[row] == predicted_values[row]
                    for row in sampled_indices
                )
                / len(sampled_indices)
            )
        bootstrap_slopes.append(
            float(linregress(time_values, sampled_accuracies).slope)
        )
    tail = (1 - confidence_level) / 2
    bounds = (
        _quantile(bootstrap_slopes, tail),
        _quantile(bootstrap_slopes, 1 - tail),
    )
    return DeteriorationReport(
        observations=tuple(entry[1] for entry in parsed),
        accuracy_slope=float(regression.slope),
        slope_confidence_interval=bounds,
        p_value=float(regression.pvalue),
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        deterioration_detected=bounds[1] < 0,
    )


def _numeric_timestamp(value: Any) -> float | None:
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        raise ValueError("Numeric timestamps must be finite.")
    return None


def _datetime_timestamp(value: Any) -> float:
    if isinstance(value, datetime):
        return value.timestamp() / 86400
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).timestamp() / 86400
    if isinstance(value, str):
        try:
            return (
                datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                / 86400
            )
        except ValueError as error:
            raise ValueError(
                f"Timestamp {value!r} is not a valid ISO date/time string."
            ) from error
    raise ValueError("Timestamps must be finite numbers, dates, or ISO date/time strings.")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
