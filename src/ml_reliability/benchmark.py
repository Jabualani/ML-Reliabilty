from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon


@dataclass(frozen=True)
class MethodComparisonReport:
    """Multi-dataset comparison where each dataset is one observation."""

    methods: tuple[str, ...]
    dataset_count: int
    alpha: float
    higher_is_better: bool
    average_ranks: Mapping[str, float]
    mean_scores: Mapping[str, float]
    friedman: Mapping[str, Any]
    nemenyi: Mapping[str, Any]
    pairwise: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "methods": list(self.methods),
            "dataset_count": self.dataset_count,
            "alpha": self.alpha,
            "higher_is_better": self.higher_is_better,
            "average_ranks": dict(self.average_ranks),
            "mean_scores": dict(self.mean_scores),
            "friedman": dict(self.friedman),
            "nemenyi": dict(self.nemenyi),
            "pairwise": [dict(row) for row in self.pairwise],
            "warnings": list(self.warnings),
            "interpretation": (
                "Each dataset contributes one aggregated score per method. "
                "Rank 1 is best. Folds, repeats, and seeds within a dataset "
                "are not independent observations and must be aggregated "
                "before this analysis."
            ),
        }


def summarize_repeats(
    results: pd.DataFrame,
    *,
    dataset: str = "dataset",
    method: str = "method",
    score: str = "score",
    aggregate: str = "mean",
) -> pd.DataFrame:
    """Aggregate repeated results into a datasets-by-methods score table."""
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")
    missing = {dataset, method, score} - set(results.columns)
    if missing:
        raise ValueError(f"results is missing columns: {sorted(missing)!r}.")
    if aggregate not in {"mean", "median"}:
        raise ValueError("aggregate must be 'mean' or 'median'.")
    return results.pivot_table(
        index=dataset,
        columns=method,
        values=score,
        aggfunc=aggregate,
    )


def compare_methods(
    scores: pd.DataFrame,
    *,
    higher_is_better: bool = True,
    alpha: float = 0.05,
    bootstrap_samples: int = 2000,
    random_seed: int = 42,
) -> MethodComparisonReport:
    """Compare methods across datasets with Friedman, Nemenyi, and Wilcoxon-Holm.

    ``scores`` has one row per dataset and one column per method. Each cell
    must already summarize repeats, folds, or seeds for that dataset (see
    :func:`summarize_repeats`). Datasets are the independent units.
    """
    table = _validate_scores(scores, alpha, bootstrap_samples, random_seed)
    methods = tuple(str(name) for name in table.columns)
    dataset_count, method_count = table.shape
    values = table.to_numpy(dtype=float)

    oriented = values if higher_is_better else -values
    ranks = np.vstack([rankdata(-row, method="average") for row in oriented])
    average_ranks = ranks.mean(axis=0)
    rank_by_method = {name: float(rank) for name, rank in zip(methods, average_ranks)}

    warnings: list[str] = []
    if dataset_count < 10:
        warnings.append(
            "Fewer than 10 datasets: Friedman/Nemenyi and Wilcoxon results "
            "have limited power and approximate p-values; treat with caution."
        )

    friedman = _friedman(values, methods)
    nemenyi = _nemenyi(average_ranks, methods, dataset_count, alpha)
    pairwise = _pairwise(
        table,
        higher_is_better,
        alpha,
        bootstrap_samples,
        random_seed,
        nemenyi["pairwise_p_values"],
    )

    return MethodComparisonReport(
        methods=methods,
        dataset_count=dataset_count,
        alpha=alpha,
        higher_is_better=higher_is_better,
        average_ranks=rank_by_method,
        mean_scores={
            name: float(value) for name, value in zip(methods, values.mean(axis=0))
        },
        friedman=friedman,
        nemenyi=nemenyi,
        pairwise=tuple(pairwise),
        warnings=tuple(warnings),
    )


def plot_critical_difference(report: MethodComparisonReport) -> Any:
    """Draw a critical-difference diagram; returns ``(figure, axes)``."""
    import matplotlib.pyplot as plt

    critical_difference = report.nemenyi["critical_difference"]
    if critical_difference is None:
        raise ValueError("A critical difference is unavailable for this report.")
    ordered = sorted(report.average_ranks.items(), key=lambda item: item[1])
    count = len(ordered)
    figure, axes = plt.subplots(figsize=(max(6.0, 1.2 * count), 2.0 + 0.3 * count))
    axes.set_xlim(0.5, count + 0.5)
    axes.set_ylim(0, count + 2)
    axes.invert_xaxis()
    axes.yaxis.set_visible(False)
    for side in ("left", "right", "top"):
        axes.spines[side].set_visible(False)
    axes.set_xlabel("Average rank (1 = best)")

    for position, (name, rank) in enumerate(ordered):
        height = count + 0.5 - position * 0.8
        axes.plot([rank, rank], [height, count + 1.2], color="0.4", linewidth=0.8)
        axes.text(rank, height - 0.1, f" {name} ({rank:.2f})", va="top", ha="right")

    axes.plot(
        [1, 1 + critical_difference],
        [count + 1.7, count + 1.7],
        color="black",
        linewidth=2,
    )
    axes.text(1 + critical_difference / 2, count + 1.85, f"CD = {critical_difference:.2f}", ha="center")

    bar_height = 0.2
    drawn: list[tuple[int, int]] = []
    for start in range(count):
        end = start
        while end + 1 < count and ordered[end + 1][1] - ordered[start][1] < critical_difference:
            end += 1
        if end > start and not any(s <= start and end <= e for s, e in drawn):
            drawn.append((start, end))
            axes.plot(
                [ordered[start][1], ordered[end][1]],
                [bar_height, bar_height],
                color="black",
                linewidth=3,
            )
            bar_height += 0.35
    return figure, axes


def _validate_scores(
    scores: pd.DataFrame,
    alpha: float,
    bootstrap_samples: int,
    random_seed: int,
) -> pd.DataFrame:
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a pandas DataFrame (datasets x methods).")
    if scores.shape[1] < 2:
        raise ValueError("At least two methods (columns) are required.")
    if scores.shape[0] < 2:
        raise ValueError("At least two datasets (rows) are required.")
    if scores.columns.has_duplicates:
        raise ValueError("Method names must be unique.")
    if scores.isna().to_numpy().any():
        raise ValueError(
            "scores must not contain missing values; every method needs a "
            "score on every dataset."
        )
    try:
        table = scores.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("scores must be numeric.") from error
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("scores must be finite.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples < 100
    ):
        raise ValueError("bootstrap_samples must be an integer of at least 100.")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer.")
    return table


def _friedman(values: np.ndarray, methods: Sequence[str]) -> dict[str, Any]:
    if len(methods) < 3:
        return {
            "status": "unavailable",
            "reason": "The Friedman test requires at least three methods.",
        }
    if np.all(values == values[:, [0]]):
        return {"status": "ok", "statistic": 0.0, "p_value": 1.0}
    statistic, p_value = friedmanchisquare(*[values[:, i] for i in range(values.shape[1])])
    return {"status": "ok", "statistic": float(statistic), "p_value": float(p_value)}


def _nemenyi(
    average_ranks: np.ndarray,
    methods: Sequence[str],
    dataset_count: int,
    alpha: float,
) -> dict[str, Any]:
    method_count = len(methods)
    if method_count < 3:
        return {
            "status": "unavailable",
            "reason": "The Nemenyi post-hoc test requires at least three methods.",
            "critical_difference": None,
            "pairwise_p_values": {},
        }
    standard_error = math.sqrt(method_count * (method_count + 1) / (12 * dataset_count))
    critical_q = float(studentized_range.ppf(1 - alpha, method_count, np.inf))
    p_values: dict[tuple[str, str], float] = {}
    for (i, first), (j, second) in itertools.combinations(enumerate(methods), 2):
        q_value = abs(average_ranks[i] - average_ranks[j]) / standard_error
        p_values[(first, second)] = float(studentized_range.sf(q_value, method_count, np.inf))
    return {
        "status": "ok",
        "critical_difference": critical_q * standard_error,
        "pairwise_p_values": p_values,
    }


def _pairwise(
    table: pd.DataFrame,
    higher_is_better: bool,
    alpha: float,
    bootstrap_samples: int,
    random_seed: int,
    nemenyi_p_values: Mapping[tuple[str, str], float],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(table.columns, 2):
        a = table[first].to_numpy(dtype=float)
        b = table[second].to_numpy(dtype=float)
        differences = (a - b) if higher_is_better else (b - a)
        if np.all(differences == 0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(differences, zero_method="wilcox").pvalue)
        resampled = rng.choice(differences, size=(bootstrap_samples, len(differences)))
        low, high = np.quantile(resampled.mean(axis=1), [alpha / 2, 1 - alpha / 2])
        rows.append(
            {
                "method_a": str(first),
                "method_b": str(second),
                "wins_a": int((differences > 0).sum()),
                "ties": int((differences == 0).sum()),
                "wins_b": int((differences < 0).sum()),
                "mean_difference_a_minus_b": float(differences.mean()),
                "mean_difference_confidence_interval": (float(low), float(high)),
                "cliffs_delta": _cliffs_delta(a, b, higher_is_better),
                "wilcoxon_p_value": p_value,
                "nemenyi_p_value": nemenyi_p_values.get((str(first), str(second))),
            }
        )
    adjusted = _holm([row["wilcoxon_p_value"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["wilcoxon_holm_p_value"] = value
        row["significant_after_holm"] = value < alpha
    return rows


def _cliffs_delta(a: np.ndarray, b: np.ndarray, higher_is_better: bool) -> float:
    # Paired dominance per dataset: dataset difficulty varies, so pooled
    # unpaired comparisons would be misleading.
    delta = float(((a > b).sum() - (a < b).sum()) / len(a))
    return delta if higher_is_better else -delta


def _holm(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (len(p_values) - position) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted
