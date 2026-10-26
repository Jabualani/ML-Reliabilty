from __future__ import annotations

import bisect
import random
import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FeatureDrift:
    name: str
    psi: float
    drifted: bool
    ks_statistic: float | None = None
    kl_divergence: float | None = None
    wasserstein_distance: float | None = None
    dsi: float | None = None
    chi_square_statistic: float | None = None
    chi_square_p_value: float | None = None
    chi_square_drifted: bool | None = None
    dsi_drifted: bool | None = None
    confidence_intervals: Mapping[str, tuple[float, float]] | None = None


@dataclass(frozen=True)
class ReliabilityReport:
    sample_count: int
    similarity_score: float
    mean_psi: float
    drift_threshold: float
    drifted_features: tuple[str, ...]
    features: tuple[FeatureDrift, ...]
    covariate_shift_detected: bool
    chi_square_alpha: float
    confidence_level: float
    confidence_intervals: Mapping[str, tuple[float, float]]
    bootstrap_samples: int
    evaluation: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "similarity_score": self.similarity_score,
            "mean_psi": self.mean_psi,
            "drift_threshold": self.drift_threshold,
            "chi_square_alpha": self.chi_square_alpha,
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.bootstrap_samples,
            "confidence_intervals": _intervals_to_dict(self.confidence_intervals),
            "evaluation": self.evaluation,
            "covariate_shift_detected": self.covariate_shift_detected,
            "drifted_features": list(self.drifted_features),
            "features": [
                {
                    "name": feature.name,
                    "ks_statistic": feature.ks_statistic,
                    "kl_divergence": feature.kl_divergence,
                    "psi": feature.psi,
                    "wasserstein_distance": feature.wasserstein_distance,
                    "dsi": feature.dsi,
                    "chi_square_statistic": feature.chi_square_statistic,
                    "chi_square_p_value": feature.chi_square_p_value,
                    "chi_square_drifted": feature.chi_square_drifted,
                    "dsi_drifted": feature.dsi_drifted,
                    "confidence_intervals": _intervals_to_dict(
                        feature.confidence_intervals or {}
                    ),
                    "drifted": feature.drifted,
                }
                for feature in self.features
            ],
        }


@dataclass(frozen=True)
class FeatureConceptShift:
    name: str
    js_divergence: float | None
    kl_divergence: float | None
    psi: float | None
    shifted: bool


@dataclass(frozen=True)
class ConceptShiftReport:
    sample_count: int
    mean_js_divergence: float | None
    shift_threshold: float
    shifted_features: tuple[str, ...]
    features: tuple[FeatureConceptShift, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mean_js_divergence": self.mean_js_divergence,
            "shift_threshold": self.shift_threshold,
            "shifted_features": list(self.shifted_features),
            "features": [
                {
                    "name": feature.name,
                    "js_divergence": feature.js_divergence,
                    "kl_divergence": feature.kl_divergence,
                    "psi": feature.psi,
                    "shifted": feature.shifted,
                }
                for feature in self.features
            ],
        }


@dataclass(frozen=True)
class _FeatureProfile:
    name: str
    numeric: bool
    edges: tuple[float, ...]
    categories: tuple[str, ...]
    reference_counts: tuple[int, ...]
    minimum: float | None = None
    maximum: float | None = None
    reference_values: tuple[float, ...] = ()
    reference_moments: tuple[float, float, float, float] | None = None
    reference_bin_ids: tuple[int, ...] = ()
    reference_sample_count: int = 0
    reference_observations: tuple[Any, ...] = ()


class DriftMonitor:
    """Compare incoming feature distributions with a fitted reference dataset.

    Numeric features are compared using quantile bins and categorical features
    using their fitted categories. The reported score is a distribution-
    similarity indicator derived from mean PSI; it is not a probability or a
    measure of prediction correctness.
    """

    def __init__(
        self,
        n_bins: int = 10,
        drift_threshold: float = 0.2,
        chi_square_alpha: float = 0.05,
        dsi_threshold: float = 1.0,
        concept_shift_threshold: float = 0.1,
        confidence_level: float = 0.95,
        bootstrap_samples: int = 200,
        random_seed: int = 42,
    ) -> None:
        if not isinstance(n_bins, int) or isinstance(n_bins, bool):
            raise ValueError("n_bins must be an integer.")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if not math.isfinite(drift_threshold) or drift_threshold <= 0:
            raise ValueError("drift_threshold must be a finite positive number.")
        if not 0 < chi_square_alpha < 1:
            raise ValueError("chi_square_alpha must be between 0 and 1.")
        if not math.isfinite(dsi_threshold) or dsi_threshold <= 0:
            raise ValueError("dsi_threshold must be a finite positive number.")
        if (
            not math.isfinite(concept_shift_threshold)
            or concept_shift_threshold <= 0
        ):
            raise ValueError(
                "concept_shift_threshold must be a finite positive number."
            )
        if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        if (
            not isinstance(bootstrap_samples, int)
            or isinstance(bootstrap_samples, bool)
            or bootstrap_samples < 20
        ):
            raise ValueError("bootstrap_samples must be an integer of at least 20.")
        if not isinstance(random_seed, int) or isinstance(random_seed, bool):
            raise ValueError("random_seed must be an integer.")
        self.n_bins = n_bins
        self.drift_threshold = drift_threshold
        self.chi_square_alpha = chi_square_alpha
        self.dsi_threshold = dsi_threshold
        self.concept_shift_threshold = concept_shift_threshold
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed
        self._profiles: tuple[_FeatureProfile, ...] | None = None
        self._reference_labels: tuple[str, ...] | None = None

    def fit(self, X: Any, y: Any = None) -> DriftMonitor:
        columns = _as_columns(X)
        if not columns or not next(iter(columns.values())):
            raise ValueError("X must contain at least one feature and one row.")
        labels = None if y is None else _as_values(y, "y")
        if labels is not None:
            if len(labels) != len(next(iter(columns.values()))):
                raise ValueError("X and y must contain the same number of rows.")
            if any(_is_missing(label) for label in labels):
                raise ValueError("y must not contain missing labels.")
            labels = [_category_key(label) for label in labels]

        profiles = []
        for name, values in columns.items():
            nonmissing = [value for value in values if not _is_missing(value)]
            numeric = bool(nonmissing) and all(
                isinstance(value, numbers.Real) for value in nonmissing
            )

            if numeric:
                sorted_values = sorted(float(value) for value in nonmissing)
                edges = _quantile_edges(sorted_values, self.n_bins)
                categories: tuple[str, ...] = ()
                minimum = sorted_values[0]
                maximum = sorted_values[-1]
                reference_values = tuple(sorted_values)
                reference_moments = _moments(sorted_values)
            else:
                edges = ()
                categories = tuple(
                    sorted({_category_key(value) for value in nonmissing})
                )
                minimum = None
                maximum = None
                reference_values = ()
                reference_moments = None

            profile = _FeatureProfile(
                name=name,
                numeric=numeric,
                edges=edges,
                categories=categories,
                reference_counts=(),
                minimum=minimum,
                maximum=maximum,
                reference_values=reference_values,
                reference_moments=reference_moments,
            )
            reference_bin_ids = tuple(_bin_ids(values, profile))
            profiles.append(
                _FeatureProfile(
                    name=name,
                    numeric=numeric,
                    edges=edges,
                    categories=categories,
                    reference_counts=tuple(_counts(values, profile)),
                    minimum=minimum,
                    maximum=maximum,
                    reference_values=reference_values,
                    reference_moments=reference_moments,
                    reference_bin_ids=reference_bin_ids,
                    reference_sample_count=len(values),
                    reference_observations=tuple(values),
                )
            )

        self._profiles = tuple(profiles)
        self._reference_labels = None if labels is None else tuple(labels)
        return self

    def compare(self, X: Any) -> ReliabilityReport:
        if self._profiles is None:
            raise RuntimeError("DriftMonitor must be fitted before comparing data.")

        columns = _as_columns(X)
        _validate_columns(columns, self._profiles)
        row_counts = {len(values) for values in columns.values()}
        if len(row_counts) != 1 or not row_counts or next(iter(row_counts)) == 0:
            raise ValueError(
                "X must contain at least one row with consistent feature lengths."
            )
        sample_count = next(iter(row_counts))

        rng = random.Random(self.random_seed)
        results = []
        bootstrap_similarity: list[float] = []
        bootstrap_mean_psi: list[float] = []
        bootstrap_metrics: dict[str, list[dict[str, float]]] = {}
        for _ in range(self.bootstrap_samples):
            reference_indices = rng.choices(
                range(len(self._profiles[0].reference_observations)),
                k=len(self._profiles[0].reference_observations),
            )
            current_indices = rng.choices(
                range(sample_count),
                k=sample_count,
            )
            replicate_psi = []
            for profile in self._profiles:
                sampled_reference = [
                    profile.reference_observations[index]
                    for index in reference_indices
                ]
                sampled_current = [
                    columns[profile.name][index] for index in current_indices
                ]
                metrics = _measure_feature(
                    profile, sampled_reference, sampled_current
                )
                replicate_psi.append(metrics["psi"])
                bootstrap_metrics.setdefault(profile.name, []).append(metrics)
            replicate_mean_psi = sum(replicate_psi) / len(replicate_psi)
            bootstrap_mean_psi.append(replicate_mean_psi)
            bootstrap_similarity.append(math.exp(-replicate_mean_psi))

        for profile in self._profiles:
            values = columns[profile.name]
            metrics = _measure_feature(
                profile, profile.reference_observations, values
            )
            intervals = {}
            for metric_name in metrics:
                if metric_name == "chi_square_p_value":
                    continue
                replicates = [
                    replicate[metric_name]
                    for replicate in bootstrap_metrics[profile.name]
                    if metric_name in replicate
                ]
                if replicates:
                    intervals[metric_name] = _bootstrap_interval(
                        replicates,
                        metrics[metric_name],
                        self.confidence_level,
                        lower_bound=0.0,
                    )
            psi = metrics["psi"]
            kl_divergence = metrics["kl_divergence"]
            ks_statistic = metrics.get("ks_statistic")
            wasserstein_distance = metrics.get("wasserstein_distance")
            dsi = metrics.get("dsi")
            chi_square_statistic = metrics.get("chi_square_statistic")
            chi_square_p_value = metrics.get("chi_square_p_value")
            chi_square_drifted = (
                None
                if chi_square_p_value is None
                else chi_square_p_value < self.chi_square_alpha
            )
            dsi_drifted = None if dsi is None else dsi >= self.dsi_threshold
            results.append(
                FeatureDrift(
                    name=profile.name,
                    psi=psi,
                    drifted=psi >= self.drift_threshold,
                    ks_statistic=ks_statistic,
                    kl_divergence=kl_divergence,
                    wasserstein_distance=wasserstein_distance,
                    dsi=dsi,
                    chi_square_statistic=chi_square_statistic,
                    chi_square_p_value=chi_square_p_value,
                    chi_square_drifted=chi_square_drifted,
                    dsi_drifted=dsi_drifted,
                    confidence_intervals=intervals,
                )
            )

        mean_psi = sum(feature.psi for feature in results) / len(results)
        return ReliabilityReport(
            sample_count=sample_count,
            similarity_score=math.exp(-mean_psi),
            mean_psi=mean_psi,
            drift_threshold=self.drift_threshold,
            drifted_features=tuple(
                feature.name for feature in results if feature.drifted
            ),
            features=tuple(results),
            covariate_shift_detected=any(
                feature.drifted
                or feature.dsi_drifted is True
                or feature.chi_square_drifted is True
                for feature in results
            ),
            chi_square_alpha=self.chi_square_alpha,
            confidence_level=self.confidence_level,
            confidence_intervals={
                "similarity_score": _bootstrap_interval(
                    bootstrap_similarity,
                    math.exp(-mean_psi),
                    self.confidence_level,
                    lower_bound=0.0,
                    upper_bound=1.0,
                ),
                "mean_psi": _bootstrap_interval(
                    bootstrap_mean_psi,
                    mean_psi,
                    self.confidence_level,
                    lower_bound=0.0,
                ),
            },
            bootstrap_samples=self.bootstrap_samples,
        )

    def assess_covariate_shift(self, X: Any) -> ReliabilityReport:
        """Assess input-distribution shift without using target labels."""
        return self.compare(X)

    def plot_distributions(
        self,
        X: Any,
        feature_names: list[str] | tuple[str, ...] | None = None,
        report: ReliabilityReport | None = None,
    ) -> tuple[Any, Any]:
        """Plot reference and incoming distributions for selected features.

        Requires the optional plotting dependency: ``pip install
        ml-reliability[plots]``. Returns the Matplotlib figure and axes array.
        """
        from .visualization import plot_distributions

        return plot_distributions(self, X, feature_names, report)

    def plot_reference_distributions(
        self,
        feature_names: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[Any, Any]:
        """Plot the fitted training distributions with bootstrap CIs."""
        from .visualization import plot_reference_distributions

        return plot_reference_distributions(self, feature_names)

    def assess_concept_shift(self, X: Any, y: Any) -> ConceptShiftReport:
        """Compare per-feature conditional target distributions.

        Requires labels for both the fitted reference batch and incoming batch.
        It is a univariate-bin approximation to changes in P(y | x), and does
        not detect interactions that appear only across multiple features.
        """
        if self._profiles is None:
            raise RuntimeError("DriftMonitor must be fitted before comparison.")
        if self._reference_labels is None:
            raise RuntimeError(
                "Concept-shift assessment requires fitting DriftMonitor with y."
            )

        columns = _as_columns(X)
        _validate_columns(columns, self._profiles)
        row_count = len(next(iter(columns.values())))
        if row_count == 0:
            raise ValueError("X must contain at least one row.")
        labels = _as_values(y, "y")
        if len(labels) != row_count:
            raise ValueError("X and y must contain the same number of rows.")
        if any(_is_missing(label) for label in labels):
            raise ValueError("y must not contain missing labels.")

        current_labels = [_category_key(label) for label in labels]
        label_keys = tuple(
            sorted(set(self._reference_labels).union(current_labels))
        )
        feature_results = []
        for profile in self._profiles:
            current_bins = _bin_ids(columns[profile.name], profile)
            divergences = _conditional_divergences(
                profile.reference_bin_ids,
                self._reference_labels,
                current_bins,
                current_labels,
                label_keys,
            )
            js_divergence, kl_divergence, psi = (
                divergences if divergences is not None else (None, None, None)
            )
            feature_results.append(
                FeatureConceptShift(
                    name=profile.name,
                    js_divergence=js_divergence,
                    kl_divergence=kl_divergence,
                    psi=psi,
                    shifted=(
                        js_divergence is not None
                        and js_divergence >= self.concept_shift_threshold
                    ),
                )
            )

        scores = [
            feature.js_divergence
            for feature in feature_results
            if feature.js_divergence is not None
        ]
        mean_js = sum(scores) / len(scores) if scores else None
        return ConceptShiftReport(
            sample_count=row_count,
            mean_js_divergence=mean_js,
            shift_threshold=self.concept_shift_threshold,
            shifted_features=tuple(
                feature.name for feature in feature_results if feature.shifted
            ),
            features=tuple(feature_results),
        )


def _as_columns(X: Any) -> dict[str, list[Any]]:
    if isinstance(X, Mapping):
        columns = {
            str(name): _as_values(values, str(name)) for name, values in X.items()
        }
    elif hasattr(X, "columns") and hasattr(X, "__getitem__"):
        columns = {
            str(name): _as_values(X[name], str(name)) for name in X.columns
        }
    else:
        try:
            rows = list(X)
        except TypeError as error:
            raise ValueError(
                "X must be a 2D array or a mapping of feature columns."
            ) from error
        if not rows:
            raise ValueError("X must contain at least one row.")

        if all(isinstance(row, Mapping) for row in rows):
            names = list(rows[0])
            if any(set(row) != set(names) for row in rows):
                raise ValueError("Every row must contain the same feature names.")
            columns = {
                str(name): [row[name] for row in rows] for name in names
            }
        else:
            matrix = []
            for row in rows:
                if isinstance(row, (str, bytes)):
                    raise ValueError("X must be 2D; rows cannot be strings.")
                try:
                    matrix.append(list(row))
                except TypeError as error:
                    raise ValueError(
                        "X must be 2D; every row must be iterable."
                    ) from error
            width = len(matrix[0])
            if width == 0 or any(len(row) != width for row in matrix):
                raise ValueError(
                    "X must have a consistent, non-zero number of features."
                )
            columns = {
                f"feature_{index}": [row[index] for row in matrix]
                for index in range(width)
            }

    names = list(columns)
    if len(names) != len(set(names)):
        raise ValueError("Feature names must be unique after conversion to strings.")
    lengths = {len(values) for values in columns.values()}
    if len(lengths) > 1:
        raise ValueError("All feature columns must have the same number of rows.")
    return columns


def _as_values(values: Any, name: str) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"Feature {name!r} must contain a sequence of values.")
    try:
        return list(values)
    except TypeError as error:
        raise ValueError(
            f"Feature {name!r} must contain a sequence of values."
        ) from error


def _validate_columns(
    columns: Mapping[str, list[Any]], profiles: tuple[_FeatureProfile, ...]
) -> None:
    expected = {profile.name for profile in profiles}
    actual = set(columns)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Feature columns do not match the reference data; "
            f"missing={missing}, unexpected={unexpected}."
        )
    row_counts = {len(values) for values in columns.values()}
    if len(row_counts) != 1 or not row_counts:
        raise ValueError("All feature columns must have the same number of rows.")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, numbers.Real):
        return not math.isfinite(float(value))
    return False


def _category_key(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _quantile_edges(sorted_values: list[float], n_bins: int) -> tuple[float, ...]:
    if len(sorted_values) < 2:
        return ()
    edges = set()
    for index in range(1, n_bins):
        position = (index * len(sorted_values)) // n_bins
        if 0 < position < len(sorted_values):
            edge = sorted_values[position]
            if edge < sorted_values[-1]:
                edges.add(edge)
    return tuple(sorted(edges))


def _counts(values: list[Any], profile: _FeatureProfile) -> list[int]:
    if profile.numeric:
        regular_count = len(profile.edges) + 1
        underflow_index = regular_count
        overflow_index = regular_count + 1
        missing_index = regular_count + 2
        invalid_index = regular_count + 3
        counts = [0] * (regular_count + 4)
        if profile.minimum is None or profile.maximum is None:
            raise RuntimeError("Numeric drift profile is missing its reference range.")
        for value in values:
            if _is_missing(value):
                counts[missing_index] += 1
            elif isinstance(value, numbers.Real):
                numeric_value = float(value)
                if numeric_value < profile.minimum:
                    counts[underflow_index] += 1
                elif numeric_value > profile.maximum:
                    counts[overflow_index] += 1
                else:
                    counts[bisect.bisect_right(profile.edges, numeric_value)] += 1
            else:
                counts[invalid_index] += 1
        return counts

    categories = {category: index for index, category in enumerate(profile.categories)}
    counts = [0] * (len(profile.categories) + 2)
    other_index = len(profile.categories)
    missing_index = len(profile.categories) + 1
    for value in values:
        if _is_missing(value):
            counts[missing_index] += 1
        else:
            counts[categories.get(_category_key(value), other_index)] += 1
    return counts


def _bin_ids(values: list[Any], profile: _FeatureProfile) -> list[int]:
    if profile.numeric:
        regular_count = len(profile.edges) + 1
        underflow_index = regular_count
        overflow_index = regular_count + 1
        missing_index = regular_count + 2
        invalid_index = regular_count + 3
        if profile.minimum is None or profile.maximum is None:
            raise RuntimeError("Numeric drift profile is missing its reference range.")
        bins = []
        for value in values:
            if _is_missing(value):
                bins.append(missing_index)
            elif isinstance(value, numbers.Real):
                numeric_value = float(value)
                if numeric_value < profile.minimum:
                    bins.append(underflow_index)
                elif numeric_value > profile.maximum:
                    bins.append(overflow_index)
                else:
                    bins.append(bisect.bisect_right(profile.edges, numeric_value))
            else:
                bins.append(invalid_index)
        return bins

    categories = {category: index for index, category in enumerate(profile.categories)}
    other_index = len(profile.categories)
    missing_index = other_index + 1
    return [
        missing_index
        if _is_missing(value)
        else categories.get(_category_key(value), other_index)
        for value in values
    ]


def _measure_feature(
    profile: _FeatureProfile,
    reference_values: list[Any] | tuple[Any, ...],
    current_values: list[Any] | tuple[Any, ...],
) -> dict[str, float]:
    reference_counts = _counts(list(reference_values), profile)
    current_counts = _counts(list(current_values), profile)
    measurements = {
        "psi": _population_stability_index(reference_counts, current_counts),
        "kl_divergence": _kullback_leibler_divergence(
            reference_counts, current_counts
        ),
    }
    if profile.numeric:
        reference_numeric = tuple(
            sorted(
                float(value)
                for value in reference_values
                if isinstance(value, numbers.Real) and not _is_missing(value)
            )
        )
        current_numeric = tuple(
            sorted(
                float(value)
                for value in current_values
                if isinstance(value, numbers.Real) and not _is_missing(value)
            )
        )
        if reference_numeric and current_numeric:
            measurements["ks_statistic"] = _two_sample_ks_statistic(
                reference_numeric, current_numeric
            )
            measurements["wasserstein_distance"] = _wasserstein_1d(
                reference_numeric, current_numeric
            )
            measurements["dsi"] = _moment_drift_score(
                _moments(reference_numeric), _moments(current_numeric)
            )
    else:
        statistic, p_value = _chi_square_homogeneity(
            reference_counts, current_counts
        )
        if statistic is not None and p_value is not None:
            measurements["chi_square_statistic"] = statistic
            measurements["chi_square_p_value"] = p_value
    return measurements


def _bootstrap_interval(
    values: list[float],
    estimate: float,
    confidence_level: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float]:
    if not values:
        raise ValueError("At least one bootstrap value is required.")
    ordered = sorted(values)
    tail = (1 - confidence_level) / 2
    if lower_bound is not None and estimate <= lower_bound:
        return lower_bound, max(
            lower_bound, _linear_percentile(ordered, confidence_level)
        )
    if upper_bound is not None and estimate >= upper_bound:
        return min(
            upper_bound, _linear_percentile(ordered, 1 - confidence_level)
        ), upper_bound

    low = 2 * estimate - _linear_percentile(ordered, 1 - tail)
    high = 2 * estimate - _linear_percentile(ordered, tail)
    low = min(estimate, low)
    high = max(estimate, high)
    if lower_bound is not None:
        low = max(lower_bound, low)
    if upper_bound is not None:
        high = min(upper_bound, high)
    return low, high


def _linear_percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _intervals_to_dict(
    intervals: Mapping[str, tuple[float, float]],
) -> dict[str, list[float]]:
    return {
        name: [lower, upper]
        for name, (lower, upper) in intervals.items()
    }


def _population_stability_index(
    reference_counts: tuple[int, ...], current_counts: list[int]
) -> float:
    reference, current = _smoothed_distributions(reference_counts, current_counts)
    return sum(
        (actual - expected) * math.log(actual / expected)
        for expected, actual in zip(reference, current)
    )


def _kullback_leibler_divergence(
    reference_counts: tuple[int, ...], current_counts: list[int]
) -> float:
    reference, current = _smoothed_distributions(reference_counts, current_counts)
    return sum(
        expected * math.log(expected / actual)
        for expected, actual in zip(reference, current)
    )


def _smoothed_distributions(
    reference_counts: tuple[int, ...], current_counts: list[int]
) -> tuple[list[float], list[float]]:
    epsilon = 1e-6
    reference_total = sum(reference_counts)
    current_total = sum(current_counts)
    bucket_count = len(reference_counts)
    reference_distribution = [
        (count + epsilon) / (reference_total + epsilon * bucket_count)
        for count in reference_counts
    ]
    current_distribution = [
        (count + epsilon) / (current_total + epsilon * bucket_count)
        for count in current_counts
    ]
    return reference_distribution, current_distribution


def _two_sample_ks_statistic(
    reference_values: tuple[float, ...], current_values: tuple[float, ...]
) -> float:
    support = sorted(set(reference_values).union(current_values))
    return max(
        abs(
            bisect.bisect_right(reference_values, value) / len(reference_values)
            - bisect.bisect_right(current_values, value) / len(current_values)
        )
        for value in support
    )


def _wasserstein_1d(
    reference_values: tuple[float, ...], current_values: tuple[float, ...]
) -> float:
    support = sorted(set(reference_values).union(current_values))
    distance = 0.0
    for index in range(len(support) - 1):
        value = support[index]
        reference_cdf = bisect.bisect_right(reference_values, value) / len(
            reference_values
        )
        current_cdf = bisect.bisect_right(current_values, value) / len(current_values)
        distance += abs(reference_cdf - current_cdf) * (
            support[index + 1] - value
        )
    return distance


def _moments(values: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    mean = sum(values) / len(values)
    central = [sum((value - mean) ** order for value in values) / len(values)
               for order in (2, 3, 4)]
    variance, third, fourth = central
    standard_deviation = math.sqrt(variance)
    if variance == 0:
        return mean, standard_deviation, 0.0, 0.0
    return (
        mean,
        standard_deviation,
        third / variance**1.5,
        fourth / variance**2 - 3.0,
    )


def _moment_drift_score(
    reference: tuple[float, float, float, float] | None,
    current: tuple[float, float, float, float],
) -> float:
    if reference is None:
        raise RuntimeError("Numeric drift profile is missing reference moments.")
    reference_mean, reference_std, reference_skew, reference_kurtosis = reference
    current_mean, current_std, current_skew, current_kurtosis = current
    scale = max(reference_std, 1e-12 * max(1.0, abs(reference_mean)))
    standardized_differences = (
        (current_mean - reference_mean) / scale,
        (current_std - reference_std) / scale,
        current_skew - reference_skew,
        current_kurtosis - reference_kurtosis,
    )
    return math.sqrt(sum(value * value for value in standardized_differences))


def _chi_square_homogeneity(
    reference_counts: tuple[int, ...], current_counts: list[int]
) -> tuple[float | None, float | None]:
    categories = [
        (reference, current)
        for reference, current in zip(reference_counts, current_counts)
        if reference + current > 0
    ]
    if len(categories) < 2:
        return 0.0, 1.0

    reference_total = sum(reference for reference, _ in categories)
    current_total = sum(current for _, current in categories)
    grand_total = reference_total + current_total
    statistic = 0.0
    for reference, current in categories:
        category_total = reference + current
        expected_reference = reference_total * category_total / grand_total
        expected_current = current_total * category_total / grand_total
        if expected_reference:
            statistic += (reference - expected_reference) ** 2 / expected_reference
        if expected_current:
            statistic += (current - expected_current) ** 2 / expected_current

    degrees_of_freedom = len(categories) - 1
    p_value = _regularized_gamma_q(degrees_of_freedom / 2, statistic / 2)
    return statistic, p_value


def _regularized_gamma_q(a: float, x: float) -> float:
    if x == 0:
        return 1.0
    if x < 0 or a <= 0:
        raise ValueError("Regularized gamma parameters are outside their domain.")
    if x < a + 1:
        term = total = 1.0 / a
        denominator = a
        for _ in range(1000):
            denominator += 1.0
            term *= x / denominator
            total += term
            if abs(term) <= abs(total) * 3e-14:
                break
        p_value = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return min(1.0, max(0.0, 1.0 - p_value))

    tiny = 1e-300
    b = x + 1 - a
    c = 1.0 / tiny
    d = 1.0 / b
    result = d
    for index in range(1, 1001):
        coefficient = -index * (index - a)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= 3e-14:
            break
    return min(
        1.0,
        max(0.0, result * math.exp(-x + a * math.log(x) - math.lgamma(a))),
    )


def _conditional_divergences(
    reference_bins: tuple[int, ...],
    reference_labels: tuple[str, ...],
    current_bins: list[int],
    current_labels: list[str],
    label_keys: tuple[str, ...],
) -> tuple[float, float, float] | None:
    reference_counts: dict[int, dict[str, int]] = {}
    current_counts: dict[int, dict[str, int]] = {}
    for feature_bin, label in zip(reference_bins, reference_labels):
        counts = reference_counts.setdefault(feature_bin, {})
        counts[label] = counts.get(label, 0) + 1
    for feature_bin, label in zip(current_bins, current_labels):
        counts = current_counts.setdefault(feature_bin, {})
        counts[label] = counts.get(label, 0) + 1

    shared_bins = set(reference_counts).intersection(current_counts)
    if not shared_bins:
        return None
    reference_total = len(reference_bins)
    current_total = len(current_bins)
    weighted_js = 0.0
    weighted_kl = 0.0
    weighted_psi = 0.0
    total_weight = 0.0
    for feature_bin in shared_bins:
        reference_n = sum(reference_counts[feature_bin].values())
        current_n = sum(current_counts[feature_bin].values())
        reference_distribution = _smoothed_label_distribution(
            reference_counts[feature_bin], label_keys
        )
        current_distribution = _smoothed_label_distribution(
            current_counts[feature_bin], label_keys
        )
        midpoint = [
            (left + right) / 2
            for left, right in zip(reference_distribution, current_distribution)
        ]
        divergence = 0.5 * sum(
            left * math.log(left / middle)
            for left, middle in zip(reference_distribution, midpoint)
            if left > 0
        ) + 0.5 * sum(
            right * math.log(right / middle)
            for right, middle in zip(current_distribution, midpoint)
            if right > 0
        )
        kl_divergence = sum(
            reference * math.log(reference / current)
            for reference, current in zip(
                reference_distribution, current_distribution
            )
        )
        psi = sum(
            (current - reference) * math.log(current / reference)
            for reference, current in zip(
                reference_distribution, current_distribution
            )
        )
        weight = 0.5 * (
            reference_n / reference_total + current_n / current_total
        )
        weighted_js += weight * divergence
        weighted_kl += weight * kl_divergence
        weighted_psi += weight * psi
        total_weight += weight
    if not total_weight:
        return None
    return (
        weighted_js / total_weight,
        weighted_kl / total_weight,
        weighted_psi / total_weight,
    )


def _smoothed_label_distribution(
    counts: Mapping[str, int], label_keys: tuple[str, ...]
) -> list[float]:
    total = sum(counts.values())
    epsilon = 1e-6
    denominator = total + epsilon * len(label_keys)
    return [
        (counts.get(label, 0) + epsilon) / denominator
        for label in label_keys
    ]
