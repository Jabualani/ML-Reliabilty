from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class StatisticalDataCleaner(BaseEstimator, TransformerMixin):
    """Fit-on-training-data imputation and configurable IQR outlier handling.

    ``light`` mode imputes missing values and records IQR outliers without
    capping them. Stronger modes cap numeric values using fit-time bounds.
    The transformer preserves the input columns so later pipeline stages can
    encode and scale them without leaking fitted state across validation folds.
    """

    def __init__(self, cleaning_intensity: str = "light") -> None:
        self.cleaning_intensity = cleaning_intensity

    def fit(self, X: pd.DataFrame, y: Any = None) -> StatisticalDataCleaner:
        _ = y
        if self.cleaning_intensity not in {"light", "moderate", "heavy"}:
            raise ValueError(
                "cleaning_intensity must be 'light', 'moderate', or 'heavy'."
            )
        self._validate_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.numeric_columns_ = list(X.select_dtypes(include=[np.number]).columns)
        self.fill_values_: dict[str, Any] = {}
        self.outlier_bounds_: dict[str, tuple[float, float]] = {}
        self.missing_counts_: dict[str, int] = {}
        self.outlier_flags_: dict[str, bool] = {}
        self.uninformative_features_: list[str] = []

        for column in X.columns:
            series = X[column]
            self.missing_counts_[column] = int(series.isna().sum())
            observed = series.dropna()
            if column in self.numeric_columns_:
                values = observed.to_numpy(dtype=float)
                if not len(values):
                    warnings.warn(
                        f"Numeric feature {column!r} is entirely missing; "
                        "it will be filled with 0 and should be reviewed.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.uninformative_features_.append(column)
                    self.fill_values_[column] = 0.0
                    self.outlier_bounds_[column] = (0.0, 0.0)
                    self.outlier_flags_[column] = False
                    continue
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Numeric feature {column!r} contains infinite values."
                    )
                if observed.nunique() <= 2:
                    fill_value = observed.mode().iloc[0]
                elif self._approximately_normal(observed):
                    fill_value = float(observed.mean())
                else:
                    fill_value = float(observed.median())
                q1, q3 = np.quantile(values, [0.25, 0.75])
                multiplier = 1.0 if self.cleaning_intensity == "heavy" else 1.5
                spread = float(q3 - q1)
                lower = float(q1 - multiplier * spread)
                upper = float(q3 + multiplier * spread)
                self.fill_values_[column] = fill_value
                self.outlier_bounds_[column] = (lower, upper)
                self.outlier_flags_[column] = bool(
                    np.any((values < lower) | (values > upper))
                )
            else:
                modes = observed.mode()
                self.fill_values_[column] = (
                    modes.iloc[0] if not modes.empty else "__reliableml_missing__"
                )
                self.outlier_flags_[column] = False

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        self._validate_frame(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                "Input feature names and order must match the fitted cleaner."
            )
        cleaned = X.copy()
        for column, fill_value in self.fill_values_.items():
            cleaned[column] = cleaned[column].fillna(fill_value)
        if self.cleaning_intensity != "light":
            for column, (lower, upper) in self.outlier_bounds_.items():
                cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
        return cleaned

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and list(input_features) != list(
            self.feature_names_in_
        ):
            raise ValueError("input_features must match the fitted feature names.")
        return self.feature_names_in_.copy()

    def get_cleaning_report(self) -> dict[str, Any]:
        check_is_fitted(self, "feature_names_in_")
        return {
            "cleaning_intensity": self.cleaning_intensity,
            "missing_values_at_fit": dict(self.missing_counts_),
            "uninformative_features_filled_with_zero": list(
                self.uninformative_features_
            ),
            "numeric_outliers_detected": dict(self.outlier_flags_),
            "outlier_capping_applied": self.cleaning_intensity != "light",
        }

    @staticmethod
    def _approximately_normal(series: pd.Series) -> bool:
        if len(series) < 10:
            return False
        standard_deviation = float(series.std(ddof=0))
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            return False
        skewness = float(series.skew())
        excess_kurtosis = float(series.kurtosis())
        return (
            abs(float(series.mean()) - float(series.median()))
            <= 0.1 * standard_deviation
            and np.isfinite(skewness)
            and np.isfinite(excess_kurtosis)
            and abs(skewness) < 0.5
            and abs(excess_kurtosis) < 1.5
        )

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if X.empty or X.columns.has_duplicates:
            raise ValueError("X must be non-empty and have unique column names.")
        if any(not isinstance(column, str) for column in X.columns):
            raise ValueError("All feature names must be strings.")
        if X.select_dtypes(include=["datetime", "datetimetz"]).shape[1]:
            raise ValueError("Datetime features must be converted before cleaning.")


def clean_statistical_data(
    X: pd.DataFrame,
    *,
    cleaning_intensity: str = "light",
    return_cleaner: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, StatisticalDataCleaner]:
    """Clean a DataFrame once, using light non-capping mode by default."""
    cleaner = StatisticalDataCleaner(cleaning_intensity=cleaning_intensity)
    cleaned = cleaner.fit_transform(X)
    if return_cleaner:
        return cleaned, cleaner
    return cleaned
