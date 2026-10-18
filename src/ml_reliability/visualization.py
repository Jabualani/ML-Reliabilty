from __future__ import annotations

import math
import numbers
import textwrap
from typing import Any, TYPE_CHECKING

from .drift import (
    ReliabilityReport,
    _FeatureProfile,
    _as_columns,
    _category_key,
    _counts,
    _is_missing,
    _validate_columns,
)

if TYPE_CHECKING:
    from .drift import DriftMonitor


def plot_distributions(
    monitor: DriftMonitor,
    X: Any,
    feature_names: list[str] | tuple[str, ...] | None = None,
    report: ReliabilityReport | None = None,
) -> tuple[Any, Any]:
    if monitor._profiles is None:
        raise RuntimeError("DriftMonitor must be fitted before plotting.")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    columns = _as_columns(X)
    _validate_columns(columns, monitor._profiles)
    profiles = _select_profiles(monitor._profiles, feature_names)
    if not profiles:
        raise ValueError("At least one feature must be selected for plotting.")
    report = report or monitor.compare(X)

    ncols = min(2, len(profiles))
    nrows = math.ceil(len(profiles) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    results = {result.name: result for result in report.features}
    for index, profile in enumerate(profiles):
        axis = axes[index // ncols][index % ncols]
        current_values = columns[profile.name]
        if profile.numeric:
            _plot_numeric(
                axis,
                profile,
                current_values,
                monitor.n_bins,
                monitor.confidence_level,
                monitor.bootstrap_samples,
                monitor.random_seed,
                pd,
                sns,
            )
        else:
            _plot_categorical(
                axis,
                profile,
                current_values,
                monitor.confidence_level,
                monitor.random_seed,
                pd,
                sns,
            )
        metric = results[profile.name]
        metric_text = _metric_title(metric)
        axis.set_title(
            f"{profile.name} — {metric_text}",
            fontsize=8,
            linespacing=1.25,
        )

    for index in range(len(profiles), nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)
    interval = report.confidence_intervals["similarity_score"]
    psi_interval = report.confidence_intervals["mean_psi"]
    figure.suptitle(
        f"Train vs unseen | Similarity {report.similarity_score:.2e} "
        f"(CI {interval[0]:.1e}–{interval[1]:.1e}) | "
        f"Mean PSI {report.mean_psi:.2g} "
        f"(CI {psi_interval[0]:.2g}–{psi_interval[1]:.2g}), "
        f"{report.confidence_level:.0%} bootstrap",
        fontsize=10,
    )
    return figure, axes


def plot_reference_distributions(
    monitor: DriftMonitor,
    feature_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[Any, Any]:
    if monitor._profiles is None:
        raise RuntimeError("DriftMonitor must be fitted before plotting.")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    profiles = _select_profiles(monitor._profiles, feature_names)
    if not profiles:
        raise ValueError("At least one feature must be selected for plotting.")
    ncols = min(2, len(profiles))
    nrows = math.ceil(len(profiles) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 4.8 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for index, profile in enumerate(profiles):
        axis = axes[index // ncols][index % ncols]
        if profile.numeric:
            _plot_reference_numeric(
                axis,
                profile,
                monitor.confidence_level,
                monitor.bootstrap_samples,
                monitor.random_seed,
                pd,
                sns,
            )
        else:
            _plot_reference_categorical(
                axis,
                profile,
                monitor.confidence_level,
                monitor.random_seed,
                pd,
                sns,
            )
        axis.set_title(f"Training reference: {profile.name}")
        axis.set_xlabel(profile.name)
    for index in range(len(profiles), nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)
    figure.suptitle(
        f"Training distributions with {monitor.confidence_level:.0%} "
        "bootstrap confidence intervals"
    )
    return figure, axes


def _select_profiles(
    profiles: tuple[_FeatureProfile, ...],
    feature_names: list[str] | tuple[str, ...] | None,
) -> tuple[_FeatureProfile, ...]:
    if feature_names is None:
        return profiles
    requested = tuple(feature_names)
    if len(set(requested)) != len(requested):
        raise ValueError("feature_names must not contain duplicates.")
    available = {profile.name: profile for profile in profiles}
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Unknown feature names: {missing}.")
    return tuple(available[name] for name in requested)


def _plot_numeric(
    axis: Any,
    profile: _FeatureProfile,
    current_values: list[Any],
    n_bins: int,
    confidence_level: float,
    bootstrap_samples: int,
    random_seed: int,
    pd: Any,
    sns: Any,
) -> None:
    reference = list(profile.reference_values)
    incoming = [
        float(value)
        for value in current_values
        if isinstance(value, numbers.Real) and not _is_missing(value)
    ]
    combined = reference + incoming
    if not combined:
        axis.text(0.5, 0.5, "No finite numeric values", ha="center", va="center")
        return
    lower = min(combined)
    upper = max(combined)
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.5)
        lower -= padding
        upper += padding
    bin_edges = [
        lower + (upper - lower) * index / n_bins for index in range(n_bins + 1)
    ]
    frame = pd.DataFrame(
        [
            {"value": value, "dataset": dataset}
            for dataset, values in (("Training", reference), ("Unseen", incoming))
            for value in values
        ]
    )
    sns.histplot(
        data=frame,
        x="value",
        hue="dataset",
        bins=bin_edges,
        stat="density",
        common_norm=False,
        common_bins=True,
        element="step",
        fill=True,
        alpha=0.24,
        ax=axis,
    )
    axis.set_ylabel("Density")
    cdf_axis = axis.twinx()
    if reference:
        sns.ecdfplot(
            x=reference,
            ax=cdf_axis,
            color=sns.color_palette("colorblind")[0],
            linestyle="--",
            label="Training ECDF",
        )
    if incoming:
        sns.ecdfplot(
            x=incoming,
            ax=cdf_axis,
            color=sns.color_palette("colorblind")[1],
            linestyle="--",
            label="Unseen ECDF",
        )
    cdf_axis.set_ylabel("Empirical CDF")
    cdf_axis.set_ylim(0, 1)
    cdf_axis.text(
        0.98,
        0.03,
        "Dashed lines: ECDF (right axis)",
        transform=cdf_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize="x-small",
    )
    _annotate_non_numeric(
        axis, "Training", profile.reference_sample_count, len(reference)
    )
    _annotate_non_numeric(axis, "Unseen", len(current_values), len(incoming))
    if reference:
        mean_ci = _bootstrap_mean_interval(
            reference, confidence_level, bootstrap_samples, random_seed
        )
        axis.axvline(
            sum(reference) / len(reference),
            color=sns.color_palette("colorblind")[0],
            linewidth=1.5,
            label=f"Training mean {mean_ci[0]:.3g}–{mean_ci[1]:.3g} CI",
        )
    handles, labels = axis.get_legend_handles_labels()
    cdf_handles, cdf_labels = cdf_axis.get_legend_handles_labels()
    axis.legend(handles + cdf_handles, labels + cdf_labels, loc="best", fontsize="small")
    cdf_axis.set_zorder(axis.get_zorder() - 1)
    cdf_axis.patch.set_visible(False)
    axis.set_xlabel(profile.name)
    axis.legend()


def _plot_categorical(
    axis: Any,
    profile: _FeatureProfile,
    current_values: list[Any],
    confidence_level: float,
    random_seed: int,
    pd: Any,
    sns: Any,
) -> None:
    frame = _categorical_frame(profile, current_values)
    sns.barplot(
        data=frame,
        x="category",
        y="observed",
        hue="dataset",
        order=_category_order(profile),
        errorbar=("ci", confidence_level * 100),
        n_boot=1000,
        seed=random_seed,
        capsize=0.08,
        ax=axis,
    )
    axis.set_ylabel("Category proportion")
    axis.set_xlabel(profile.name)
    axis.tick_params(axis="x", rotation=30)
    axis.legend(title="")


def _plot_reference_numeric(
    axis: Any,
    profile: _FeatureProfile,
    confidence_level: float,
    bootstrap_samples: int,
    random_seed: int,
    pd: Any,
    sns: Any,
) -> None:
    values = list(profile.reference_values)
    if not values:
        axis.text(0.5, 0.5, "No finite numeric values", ha="center", va="center")
        return
    sns.histplot(
        x=values,
        stat="density",
        bins=min(20, max(5, round(math.sqrt(len(values))))),
        color=sns.color_palette("colorblind")[0],
        alpha=0.3,
        ax=axis,
    )
    if len(set(values)) > 1:
        sns.kdeplot(x=values, color=sns.color_palette("colorblind")[0], ax=axis)
    lower, upper = _bootstrap_mean_interval(
        values, confidence_level, bootstrap_samples, random_seed
    )
    axis.axvline(
        sum(values) / len(values),
        color=sns.color_palette("colorblind")[3],
        linestyle="--",
        label=(
            f"Mean {confidence_level:.0%} CI: "
            f"[{lower:.3g}, {upper:.3g}]"
        ),
    )
    _annotate_non_numeric(
        axis, "Training", profile.reference_sample_count, len(values)
    )
    axis.set_ylabel("Density")
    axis.legend()


def _plot_reference_categorical(
    axis: Any,
    profile: _FeatureProfile,
    confidence_level: float,
    random_seed: int,
    pd: Any,
    sns: Any,
) -> None:
    frame = _categorical_frame(profile, [])
    sns.barplot(
        data=frame[frame["dataset"] == "Training"],
        x="category",
        y="observed",
        order=_category_order(profile),
        errorbar=("ci", confidence_level * 100),
        n_boot=1000,
        seed=random_seed,
        capsize=0.08,
        color=sns.color_palette("colorblind")[0],
        ax=axis,
    )
    axis.set_ylabel("Category proportion")
    axis.tick_params(axis="x", rotation=30)


def _categorical_frame(
    profile: _FeatureProfile, current_values: list[Any]
) -> Any:
    import pandas as pd

    rows = []
    for dataset, values in (
        ("Training", profile.reference_observations),
        ("Unseen", tuple(current_values)),
    ):
        categories = _category_order(profile)
        for value in values:
            observed_category = _display_category(value, profile)
            rows.extend(
                {
                    "dataset": dataset,
                    "category": category,
                    "observed": float(category == observed_category),
                }
                for category in categories
            )
    return pd.DataFrame(rows, columns=["dataset", "category", "observed"])


def _display_category(value: Any, profile: _FeatureProfile) -> str:
    if _is_missing(value):
        return "<missing>"
    if _category_key(value) not in profile.categories:
        return "<unseen category>"
    return str(value)


def _category_order(profile: _FeatureProfile) -> list[str]:
    labels = {
        _category_key(value): str(value)
        for value in profile.reference_observations
        if not _is_missing(value)
    }
    return [
        *[labels.get(key, key) for key in profile.categories],
        "<unseen category>",
        "<missing>",
    ]


def _metric_title(metric: Any) -> str:
    parts = []
    for name, label in (
        ("psi", "PSI"),
        ("kl_divergence", "KL"),
        ("ks_statistic", "KS"),
        ("wasserstein_distance", "W"),
        ("dsi", "DSI"),
    ):
        value = getattr(metric, name)
        if value is None:
            continue
        interval = (metric.confidence_intervals or {}).get(name)
        if interval:
            parts.append(
                f"{label}={value:.3g} "
                f"[{interval[0]:.3g}, {interval[1]:.3g}]"
            )
        else:
            parts.append(f"{label}={value:.3g}")
    if metric.chi_square_p_value is not None:
        parts.append(f"χ² p={metric.chi_square_p_value:.3g}")
    return "\n".join(textwrap.wrap(" | ".join(parts), width=58))


def _bootstrap_mean_interval(
    values: list[float],
    confidence_level: float,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[float, float]:
    import random

    rng = random.Random(random_seed)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(bootstrap_samples)
    )
    tail = (1 - confidence_level) / 2
    return (
        _percentile(means, tail),
        _percentile(means, 1 - tail),
    )


def _percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _annotate_non_numeric(
    axis: Any, dataset_name: str, sample_count: int, numeric_count: int
) -> None:
    non_numeric_count = sample_count - numeric_count
    if non_numeric_count:
        axis.text(
            0.02 if dataset_name == "Training" else 0.98,
            0.98,
            f"{dataset_name}: {non_numeric_count}/{sample_count} missing or non-numeric",
            transform=axis.transAxes,
            va="top",
            ha="left" if dataset_name == "Training" else "right",
            fontsize="small",
        )
