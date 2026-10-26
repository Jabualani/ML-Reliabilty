from __future__ import annotations

import csv
import json
import math
import pickle
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from os import PathLike
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

from .drift import _as_columns, _category_key


@dataclass(frozen=True)
class ExplanationReport:
    class_label: Any
    feature_names: tuple[str, ...]
    global_shap_importance: tuple[Mapping[str, Any], ...]
    local_explanations: tuple[Mapping[str, Any], ...]
    reason_alignment: Mapping[str, Any]
    supplied_importance_alignment: Mapping[str, Any] | None = None
    label_evaluation: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    export_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_label": (
                self.class_label.item()
                if callable(getattr(self.class_label, "item", None))
                else self.class_label
            ),
            "methods": {
                "global": "Monte Carlo interventional SHAP (random permutation estimator)",
                "local": "LIME-style weighted linear surrogate",
            },
            "feature_names": list(self.feature_names),
            "global_shap_importance": [
                dict(item) for item in self.global_shap_importance
            ],
            "local_explanations": [dict(item) for item in self.local_explanations],
            "reason_alignment": dict(self.reason_alignment),
            "supplied_importance_alignment": (
                dict(self.supplied_importance_alignment)
                if self.supplied_importance_alignment is not None
                else None
            ),
            "label_evaluation": (
                dict(self.label_evaluation)
                if self.label_evaluation is not None
                else None
            ),
            "warnings": list(self.warnings),
            "export_path": self.export_path,
            "limitations": [
                "SHAP and LIME explain model behavior, not whether its decisions are correct or causally justified.",
                "Attribution agreement is a consistency measure, not a probability of correctness.",
                "Only finite numeric input features are supported by this model-agnostic tabular explainer.",
                "LIME is a local surrogate whose fidelity depends on its perturbation distribution and sample budget.",
            ],
        }


def explain_model(
    estimator: Any,
    X: Any,
    feature_names: tuple[str, ...],
    background_data: Any,
    class_label: Any | None,
    max_instances: int,
    num_samples: int,
    random_seed: int,
    confidence_level: float,
    bootstrap_samples: int,
) -> ExplanationReport:
    if not isinstance(max_instances, int) or isinstance(max_instances, bool) or max_instances < 1:
        raise ValueError("max_instances must be a positive integer.")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 100:
        raise ValueError("num_samples must be an integer of at least 100.")
    if not all(
        hasattr(estimator, attribute)
        for attribute in ("predict", "predict_proba", "classes_")
    ):
        raise TypeError(
            "Explanation requires a fitted estimator with predict, predict_proba, and classes_."
        )
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100.")

    X_matrix = _numeric_matrix(X, feature_names)
    if not len(X_matrix):
        raise ValueError("At least one instance is required for explanation.")
    selected_matrix = X_matrix[:max_instances]
    if len(feature_names) < 2:
        raise ValueError("At least two features are required for alignment statistics.")
    background_matrix = _numeric_matrix(background_data, feature_names)
    if not len(background_matrix):
        raise ValueError("The fitted reference data is empty.")

    classes = list(estimator.classes_)
    if class_label is None:
        if len(classes) != 2:
            raise ValueError(
                "class_label is required for multiclass explanations."
            )
        class_label = classes[1]
    class_keys = [_value_key(label) for label in classes]
    if _value_key(class_label) not in class_keys:
        raise ValueError("class_label must be present in the fitted estimator classes_.")
    class_index = class_keys.index(_value_key(class_label))

    predict_probabilities = _probability_adapter(estimator, X, feature_names)
    target_probability = lambda values: predict_probabilities(values)[:, class_index]
    predicted_labels = list(
        estimator.predict(_restore_input(X, selected_matrix, feature_names))
    )
    if len(predicted_labels) != len(selected_matrix):
        raise ValueError("Estimator predictions must match the explained row count.")
    shap_matrix = _permutation_shapley(
        selected_matrix,
        background_matrix,
        target_probability,
        num_samples,
        random_seed,
    )
    background_mean = background_matrix.mean(axis=0)
    background_scale = background_matrix.std(axis=0)
    background_scale[background_scale == 0] = 1.0
    global_scores = [
        float(
            sum(abs(row[index]) for row in shap_matrix) / len(shap_matrix)
        )
        for index in range(len(feature_names))
    ]
    alignment_indices = sorted(
        range(len(feature_names)),
        key=lambda index: global_scores[index],
        reverse=True,
    )[: min(10, len(feature_names))]
    local_explanations = []
    local_correlations: list[float] = []
    sign_agreements: list[float] = []
    top_overlaps: list[float] = []
    for row_index, row in enumerate(selected_matrix):
        lime_weights, local_fidelity = _lime_local_surrogate(
            row,
            background_mean,
            background_scale,
            target_probability,
            num_samples,
            random_seed + row_index,
            alignment_indices,
        )
        shap_weights = [float(value) for value in shap_matrix[row_index]]
        correlation = _spearman(
            [abs(shap_weights[index]) for index in alignment_indices],
            [abs(lime_weights[index]) for index in alignment_indices],
        )
        if correlation is not None:
            local_correlations.append(correlation)
        shared_nonzero = [
            index
            for index, (shap_value, lime_value) in enumerate(
                zip(shap_weights, lime_weights)
            )
            if shap_value != 0 and lime_value != 0
        ]
        if shared_nonzero:
            sign_agreements.append(
                sum(
                    math.copysign(1, shap_weights[index])
                    == math.copysign(1, lime_weights[index])
                    for index in shared_nonzero
                )
                / len(shared_nonzero)
            )
        top_k = max(1, math.ceil(len(alignment_indices) / 3))
        shap_top = set(
            sorted(
                alignment_indices,
                key=lambda index: abs(shap_weights[index]),
                reverse=True,
            )[:top_k]
        )
        lime_top = set(
            sorted(
                alignment_indices,
                key=lambda index: abs(lime_weights[index]),
                reverse=True,
            )[:top_k]
        )
        top_overlaps.append(len(shap_top & lime_top) / top_k)
        report_top_count = min(
            5,
            max(1, math.ceil(0.1 * len(feature_names))),
        )
        top_features = sorted(
            [
                {
                    "feature": name,
                    "shap_value": shap_weights[index],
                    "absolute_importance": abs(shap_weights[index]),
                }
                for index, name in enumerate(feature_names)
            ],
            key=lambda item: item["absolute_importance"],
            reverse=True,
        )[:report_top_count]
        local_explanations.append(
            {
                "row_index": row_index,
                "prediction": _native_export_value(predicted_labels[row_index]),
                "predicted_probability_for_class": float(
                    target_probability(selected_matrix[row_index : row_index + 1])[0]
                ),
                "shap_values": {
                    name: value for name, value in zip(feature_names, shap_weights)
                },
                "top_features": top_features,
                "lime_weights": {
                    name: value for name, value in zip(feature_names, lime_weights)
                },
                "alignment_features": [
                    feature_names[index] for index in alignment_indices
                ],
                "lime_local_fidelity_r_squared": local_fidelity,
                "absolute_importance_spearman": correlation,
                "top_feature_overlap": top_overlaps[-1],
                "sign_agreement": sign_agreements[-1] if shared_nonzero else None,
            }
        )

    global_importance = [
        {
            "feature": name,
            "mean_absolute_shapley_value": global_scores[index],
        }
        for index, name in enumerate(feature_names)
    ]
    global_importance.sort(
        key=lambda item: item["mean_absolute_shapley_value"], reverse=True
    )
    alignment = _alignment_summary(
        local_correlations,
        sign_agreements,
        top_overlaps,
        confidence_level,
        bootstrap_samples,
        random_seed,
    )
    alignment["alignment_features"] = [
        feature_names[index] for index in alignment_indices
    ]
    return ExplanationReport(
        class_label=class_label,
        feature_names=feature_names,
        global_shap_importance=tuple(global_importance),
        local_explanations=tuple(local_explanations),
        reason_alignment=alignment,
    )


def explain_estimator(
    estimator: Any,
    X: Any,
    *,
    y_true: Any | None = None,
    background_data: Any | None = None,
    class_label: Any | None = None,
    feature_importance: Mapping[str, Real] | None = None,
    max_instances: int | None = None,
    num_samples: int = 500,
    random_seed: int = 42,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 1000,
    export_path: str | PathLike[str] | None = None,
    export_format: str | None = None,
) -> ExplanationReport:
    """Explain an already-fitted classifier, optionally exporting row results.

    Supplying ``export_path`` explicitly opts in to writing input rows, model
    predictions, optional labels, and per-row top features to disk.
    """
    if not hasattr(estimator, "predict"):
        raise TypeError("estimator must be a fitted model implementing predict.")
    columns = _as_columns(X)
    if not columns:
        raise ValueError("X must contain at least one feature.")
    feature_names = tuple(columns)
    row_count = len(next(iter(columns.values())))
    if row_count == 0:
        raise ValueError("X must contain at least one row.")
    if max_instances is None:
        max_instances = (
            row_count if export_path is not None else min(50, row_count)
        )
    if export_path is not None and max_instances < row_count:
        raise ValueError(
            "max_instances must cover every X row when exporting row-level explanations."
        )
    if export_path is None and export_format is not None:
        raise ValueError("export_path is required when export_format is specified.")
    format_name = "csv"
    if export_path is not None:
        reserved_names = {"prediction", "top_features"}
        if y_true is not None:
            reserved_names.add("y_true")
        conflicts = reserved_names & set(feature_names)
        if conflicts:
            raise ValueError(
                "X feature names conflict with export columns: "
                f"{sorted(conflicts)}."
            )
        format_name = _resolve_export_format(Path(export_path), export_format)

    reference = X if background_data is None else background_data
    report = explain_model(
        estimator=estimator,
        X=X,
        feature_names=feature_names,
        background_data=reference,
        class_label=class_label,
        max_instances=max_instances,
        num_samples=num_samples,
        random_seed=random_seed,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
    )
    if background_data is None:
        report = replace(
            report,
            warnings=(
                "No background_data supplied; X was used as the explanation reference. "
                "Use representative training/reference data when available.",
            ),
        )
    if feature_importance is not None:
        report = replace(
            report,
            supplied_importance_alignment=_compare_supplied_importance(
                feature_importance, report, feature_names
            ),
        )

    predictions = list(estimator.predict(X))
    if len(predictions) != row_count:
        raise ValueError("Estimator predictions must match the row count of X.")
    labels = None if y_true is None else list(y_true)
    if labels is not None:
        if len(labels) != row_count:
            raise ValueError("y_true must contain one label per row of X.")
        correct = [
            _value_key(label) == _value_key(prediction)
            for label, prediction in zip(labels, predictions)
        ]
        report = replace(
            report,
            label_evaluation={
                "sample_count": row_count,
                "accuracy": sum(correct) / row_count,
                "interpretation": (
                    "Descriptive accuracy on the supplied rows; it is not a "
                    "representative-holdout guarantee and has no confidence interval."
                ),
            },
        )
    if export_path is not None:
        destination = Path(export_path)
        _export_explanations(
            destination,
            format_name,
            columns,
            feature_names,
            predictions,
            labels,
            report.local_explanations,
        )
        report = replace(report, export_path=str(destination))
    return report


def _compare_supplied_importance(
    supplied: Mapping[str, Real],
    report: ExplanationReport,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    if not supplied:
        raise ValueError("feature_importance must not be empty.")
    unknown = set(supplied) - set(feature_names)
    if unknown:
        raise ValueError(
            f"feature_importance contains unknown features: {sorted(unknown)}."
        )
    if any(
        not isinstance(score, Real) or not math.isfinite(float(score))
        for score in supplied.values()
    ):
        raise ValueError("feature_importance values must be finite numeric scores.")

    common = sorted(supplied)
    supplied_scores = [abs(float(supplied[name])) for name in common]
    global_scores = {
        item["feature"]: float(item["mean_absolute_shapley_value"])
        for item in report.global_shap_importance
    }
    global_correlation = _spearman(
        supplied_scores, [global_scores[name] for name in common]
    )
    per_row = []
    for explanation in report.local_explanations:
        local_values = explanation["shap_values"]
        local_correlation = _spearman(
            supplied_scores,
            [abs(float(local_values[name])) for name in common],
        )
        top_count = min(5, max(1, math.ceil(0.1 * len(feature_names))))
        supplied_top = set(
            sorted(common, key=lambda name: abs(float(supplied[name])), reverse=True)[
                :top_count
            ]
        )
        local_top = set(
            sorted(
                feature_names,
                key=lambda name: abs(float(local_values[name])),
                reverse=True,
            )[:top_count]
        )
        per_row.append(
            {
                "row_index": explanation["row_index"],
                "absolute_importance_spearman": local_correlation,
                "top_feature_overlap": len(supplied_top & local_top) / top_count,
            }
        )
    return {
        "status": (
            "ok"
            if global_correlation is not None
            and all(item["absolute_importance_spearman"] is not None for item in per_row)
            else "partial"
        ),
        "source": "user-supplied; source/method not independently validated",
        "compared_features": common,
        "global_absolute_importance_spearman": global_correlation,
        "per_row": per_row,
        "interpretation": (
            "Agreement compares user-provided scores with model attributions; "
            "it does not show that either is correct or causal."
        ),
    }


def _resolve_export_format(destination: Path, requested: str | None) -> str:
    aliases = {"pkl": "pickle", "pickle": "pickle", "joblib": "joblib", "csv": "csv"}
    if requested is not None:
        result = aliases.get(requested.lower())
        if result is None:
            raise ValueError("export_format must be 'csv', 'pickle', or 'joblib'.")
        return result
    suffix = destination.suffix.lower().lstrip(".")
    return aliases.get(suffix, "csv")


def _export_explanations(
    destination: Path,
    export_format: str,
    columns: Mapping[str, Any],
    feature_names: tuple[str, ...],
    predictions: list[Any],
    labels: list[Any] | None,
    explanations: tuple[Mapping[str, Any], ...],
) -> None:
    if len(explanations) != len(predictions):
        raise ValueError("Every exported row must have a local explanation.")
    rows = []
    for index, explanation in enumerate(explanations):
        row = {
            name: _native_export_value(columns[name][index])
            for name in feature_names
        }
        if labels is not None:
            row["y_true"] = _native_export_value(labels[index])
        row["prediction"] = _native_export_value(predictions[index])
        row["top_features"] = explanation["top_features"]
        rows.append(row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if export_format == "csv":
        fieldnames = [
            *feature_names,
            *(["y_true"] if labels is not None else []),
            "prediction",
            "top_features",
        ]
        with destination.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                csv_row = dict(row)
                csv_row["top_features"] = json.dumps(
                    csv_row["top_features"], ensure_ascii=False
                )
                writer.writerow(csv_row)
    elif export_format == "pickle":
        with destination.open("wb") as file:
            pickle.dump(rows, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            import joblib
        except ImportError as error:
            raise ImportError(
                "joblib export requires the optional 'joblib' package."
            ) from error
        joblib.dump(rows, destination)


def _native_export_value(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _numeric_matrix(data: Any, names: tuple[str, ...]) -> Any:
    import numpy as np

    columns = _as_columns(data)
    if set(columns) != set(names):
        raise ValueError("Explanation features must match the fitted feature names.")
    rows = []
    for row_index in range(len(next(iter(columns.values())))):
        row = [columns[name][row_index] for name in names]
        row = [
            value.item() if callable(getattr(value, "item", None)) else value
            for value in row
        ]
        if any(not isinstance(value, Real) or not math.isfinite(float(value)) for value in row):
            raise ValueError(
                "SHAP/LIME comparison currently requires finite numeric features."
            )
        rows.append([float(value) for value in row])
    return np.asarray(rows, dtype=float)


def _probability_adapter(
    estimator: Any,
    template: Any,
    feature_names: tuple[str, ...],
) -> Any:
    import numpy as np

    def predict_probabilities(matrix: Any) -> Any:
        model_input = _restore_input(template, matrix, feature_names)
        probabilities = np.asarray(estimator.predict_proba(model_input), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(estimator.classes_):
            raise ValueError("predict_proba returned an invalid probability matrix.")
        if not np.isfinite(probabilities).all():
            raise ValueError("predict_proba returned non-finite probabilities.")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("predict_proba values must lie between 0 and 1.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("predict_proba rows must sum to 1.")
        return probabilities

    return predict_probabilities


def _restore_input(template: Any, matrix: Any, names: tuple[str, ...]) -> Any:
    if hasattr(template, "iloc") and hasattr(template, "columns"):
        return template.__class__(matrix, columns=list(names))
    if isinstance(template, list) and template and isinstance(template[0], Mapping):
        return [
            {name: row[index] for index, name in enumerate(names)}
            for row in matrix
        ]
    if isinstance(template, Mapping):
        return {
            name: [row[index] for row in matrix]
            for index, name in enumerate(names)
        }
    import numpy as np

    return np.asarray(matrix, dtype=float)


def _permutation_shapley(
    instances: Any,
    background: Any,
    predict_target: Any,
    permutation_count: int,
    random_seed: int,
) -> Any:
    import numpy as np

    rng = np.random.default_rng(random_seed)
    feature_count = instances.shape[1]
    results = []
    for instance in instances:
        row_values = np.zeros(feature_count, dtype=float)
        chunk_size = max(1, min(64, 1_000_000 // (feature_count**2)))
        remaining = permutation_count
        while remaining:
            current_count = min(chunk_size, remaining)
            background_indices = rng.integers(
                0, len(background), size=current_count
            )
            background_rows = background[background_indices]
            orders = np.asarray(
                [rng.permutation(feature_count) for _ in range(current_count)]
            )
            states = np.empty(
                (current_count, feature_count, feature_count), dtype=float
            )
            for sample_index, order in enumerate(orders):
                state = background_rows[sample_index].copy()
                for step, feature_index in enumerate(order):
                    state[feature_index] = instance[feature_index]
                    states[sample_index, step] = state
            flattened = states.reshape(-1, feature_count)
            after_values = np.asarray(
                [
                    probability
                    for start in range(0, len(flattened), 1024)
                    for probability in predict_target(flattened[start : start + 1024])
                ],
                dtype=float,
            ).reshape(current_count, feature_count)
            before_values = np.asarray(predict_target(background_rows), dtype=float)
            before_values = np.column_stack(
                [before_values, after_values[:, :-1]]
            )
            marginal_effects = after_values - before_values
            for sample_index, order in enumerate(orders):
                row_values[order] += marginal_effects[sample_index]
            remaining -= current_count
        results.append(row_values / permutation_count)
    return np.asarray(results)


def _lime_local_surrogate(
    instance: Any,
    background_mean: Any,
    background_scale: Any,
    predict_target: Any,
    sample_count: int,
    random_seed: int,
    feature_indices: list[int],
) -> tuple[list[float], float]:
    import numpy as np

    rng = np.random.default_rng(random_seed)
    selected_mean = background_mean[feature_indices]
    selected_scale = background_scale[feature_indices]
    perturbations = np.tile(instance, (sample_count, 1))
    perturbations[:, feature_indices] = rng.normal(
        loc=selected_mean,
        scale=selected_scale,
        size=(sample_count, len(feature_indices)),
    )
    perturbations[0] = instance
    standardized = (
        perturbations[:, feature_indices] - selected_mean
    ) / selected_scale
    instance_standardized = (instance[feature_indices] - selected_mean) / selected_scale
    distances = np.sqrt(
        np.sum(
            (
                (perturbations[:, feature_indices] - instance[feature_indices])
                / selected_scale
            )
            ** 2,
            axis=1,
        )
    )
    kernel_width = 0.75 * math.sqrt(len(instance))
    weights = np.exp(-((distances / kernel_width) ** 2))
    design = np.column_stack([np.ones(sample_count), standardized])
    outcomes = np.asarray(predict_target(perturbations), dtype=float)
    weighted_design = design * weights[:, None]
    gram = design.T @ weighted_design
    penalty = np.eye(gram.shape[0]) * 1e-6
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(gram + penalty, design.T @ (weights * outcomes))
    fitted = design @ coefficients
    weighted_mean = float(np.average(outcomes, weights=weights))
    residual_sum = float(np.sum(weights * (outcomes - fitted) ** 2))
    total_sum = float(np.sum(weights * (outcomes - weighted_mean) ** 2))
    fidelity = 1 - residual_sum / total_sum if total_sum else 1.0
    attributions = np.zeros(len(instance), dtype=float)
    attributions[feature_indices] = coefficients[1:] * instance_standardized
    return [float(value) for value in attributions], float(fidelity)


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    result = spearmanr(left, right)
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def _alignment_summary(
    correlations: list[float],
    sign_agreements: list[float],
    top_overlaps: list[float],
    confidence_level: float,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    if correlations:
        mean = sum(correlations) / len(correlations)
        rng = random.Random(random_seed)
        means = [
            sum(rng.choice(correlations) for _ in correlations) / len(correlations)
            for _ in range(bootstrap_samples)
        ]
        tail = (1 - confidence_level) / 2
        interval = [_quantile(means, tail), _quantile(means, 1 - tail)]
        score = (mean + 1) / 2
    else:
        mean = score = None
        interval = None
    return {
        "status": "ok" if correlations else "unavailable",
        "mean_absolute_importance_spearman": mean,
        "alignment_score_0_to_1": score,
        "mean_spearman_confidence_interval": interval,
        "mean_top_feature_overlap": (
            sum(top_overlaps) / len(top_overlaps) if top_overlaps else None
        ),
        "mean_sign_agreement": (
            sum(sign_agreements) / len(sign_agreements) if sign_agreements else None
        ),
        "explained_instance_count": len(top_overlaps),
        "confidence_level": confidence_level,
        "bootstrap_samples": bootstrap_samples,
        "interpretation": (
            "Absolute-importance rank agreement compares global SHAP with local "
            "LIME; it is not model confidence, correctness evidence, or causal validation."
        ),
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _value_key(value: Any) -> str:
    scalar = value.item() if callable(getattr(value, "item", None)) else value
    return _category_key(scalar)
