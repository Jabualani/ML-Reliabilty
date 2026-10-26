from __future__ import annotations

import json
import pickle
from dataclasses import replace
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any

from .deterioration import DeteriorationReport, assess_deterioration
from .drift import (
    ConceptShiftReport,
    DriftMonitor,
    ReliabilityReport,
    _as_columns,
)
from .explainability import ExplanationReport, explain_estimator
from .evaluation import EvaluationReport, evaluate_predictions
from .reason_audit import ReasonAuditReport, audit_model_reasons


class ReliablePipeline:
    """Pair a fitted predictor with a reference-data drift monitor.

    The estimator may be any object implementing ``fit(X, y)`` and
    ``predict(X)``, including a scikit-learn Pipeline.
    """

    def __init__(
        self,
        estimator: Any,
        n_bins: int = 10,
        drift_threshold: float = 0.2,
        chi_square_alpha: float = 0.05,
        dsi_threshold: float = 1.0,
        concept_shift_threshold: float = 0.1,
        plot_dir: str | PathLike[str] = "reliability_plots",
        confidence_level: float = 0.95,
        bootstrap_samples: int = 200,
        random_seed: int = 42,
        reliability_required: bool = True,
        bias_variance_folds: int = 5,
        bias_variance_repeats: int = 5,
    ) -> None:
        self.estimator = estimator
        self.n_bins = n_bins
        self.drift_threshold = drift_threshold
        self.chi_square_alpha = chi_square_alpha
        self.dsi_threshold = dsi_threshold
        self.concept_shift_threshold = concept_shift_threshold
        self.plot_dir = Path(plot_dir)
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed
        if not isinstance(reliability_required, bool):
            raise ValueError("reliability_required must be a boolean.")
        self.reliability_required = reliability_required
        if (
            not isinstance(bias_variance_folds, int)
            or isinstance(bias_variance_folds, bool)
            or bias_variance_folds < 2
        ):
            raise ValueError("bias_variance_folds must be an integer of at least 2.")
        if (
            not isinstance(bias_variance_repeats, int)
            or isinstance(bias_variance_repeats, bool)
            or bias_variance_repeats < 1
        ):
            raise ValueError("bias_variance_repeats must be a positive integer.")
        self.bias_variance_folds = bias_variance_folds
        self.bias_variance_repeats = bias_variance_repeats
        self._monitor: DriftMonitor | None = None
        self._estimator_fitted = False
        self.last_report: ReliabilityReport | None = None
        self.last_plot_path: Path | None = None
        self.last_report_path: Path | None = None

    def fit(self, X: Any, y: Any) -> ReliablePipeline:
        if not self.reliability_required:
            self.estimator.fit(X, y)
            self._estimator_fitted = True
            self._monitor = None
            self.last_report = None
            self.last_plot_path = None
            self.last_report_path = None
            return self

        monitor = DriftMonitor(
            n_bins=self.n_bins,
            drift_threshold=self.drift_threshold,
            chi_square_alpha=self.chi_square_alpha,
            dsi_threshold=self.dsi_threshold,
            concept_shift_threshold=self.concept_shift_threshold,
            confidence_level=self.confidence_level,
            bootstrap_samples=self.bootstrap_samples,
            random_seed=self.random_seed,
        ).fit(X, y)
        self.estimator.fit(X, y)
        self._estimator_fitted = True
        self._monitor = monitor
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        figure, _ = monitor.plot_reference_distributions()
        try:
            figure.savefig(self.plot_dir / "training-distributions.png", dpi=160)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        return self

    def predict(self, X: Any) -> Any:
        self._ensure_estimator_fitted()
        predictions = self.estimator.predict(X)
        if not self.reliability_required:
            self.last_report = None
            self.last_plot_path = None
            self.last_report_path = None
            return predictions

        self._ensure_fitted()
        self.last_report = self._monitor.compare(X)
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
        figure, _ = self._monitor.plot_distributions(X, report=self.last_report)
        self.last_plot_path = self.plot_dir / f"prediction-{timestamp}.png"
        try:
            figure.savefig(self.last_plot_path, dpi=160)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        self.last_report_path = self.plot_dir / f"prediction-{timestamp}.json"
        with self.last_report_path.open("w", encoding="utf-8") as file:
            payload = self.last_report.to_dict()
            payload["predictions"] = _json_compatible(predictions)
            payload["output_type"] = "predictions_and_reliability_report"
            json.dump(_json_compatible(payload), file, indent=2)
        return predictions

    def assess_reliability(self, X: Any) -> ReliabilityReport:
        self._ensure_reliability_enabled()
        return self._monitor.compare(X)

    def assess_covariate_shift(self, X: Any) -> ReliabilityReport:
        self._ensure_reliability_enabled()
        return self._monitor.assess_covariate_shift(X)

    def assess_concept_shift(self, X: Any, y: Any) -> ConceptShiftReport:
        self._ensure_reliability_enabled()
        return self._monitor.assess_concept_shift(X, y)

    def evaluate(
        self,
        X: Any,
        y_true: Any,
        sensitive_features: dict[str, Any] | None = None,
        positive_label: Any | None = None,
    ) -> EvaluationReport:
        """Evaluate labeled predictions, statistical tests, and CV bias/variance.

        This performs ordinary prediction (including the configured reliability
        report/plot saves), then adds label-dependent analyses to the report.
        Repeated stratified CV defaults to five folds and five repeats.
        """
        self._ensure_reliability_enabled()
        predictions = self.predict(X)
        evaluation = evaluate_predictions(
            estimator=self.estimator,
            profiles=self._monitor._profiles,
            X_test=X,
            y_true=y_true,
            y_pred=predictions,
            folds=self.bias_variance_folds,
            repeats=self.bias_variance_repeats,
            random_seed=self.random_seed,
            sensitive_features=sensitive_features,
            positive_label=positive_label,
            bootstrap_samples=self.bootstrap_samples,
            confidence_level=self.confidence_level,
        )
        if self.last_report is None or self.last_report_path is None:
            raise RuntimeError("Prediction completed without a saved reliability report.")
        self.last_report = replace(
            self.last_report,
            evaluation=evaluation.to_dict(),
        )
        with self.last_report_path.open("w", encoding="utf-8") as file:
            payload = self.last_report.to_dict()
            payload["predictions"] = _json_compatible(predictions)
            payload["output_type"] = "predictions_and_reliability_report"
            json.dump(_json_compatible(payload), file, indent=2)
        return evaluation

    def assess_deterioration(
        self,
        performance_history: Any,
        bootstrap_samples: int = 1000,
    ) -> DeteriorationReport:
        """Analyze dated labeled batches for a downward accuracy trend."""
        self._ensure_estimator_fitted()
        return assess_deterioration(
            performance_history,
            confidence_level=self.confidence_level,
            bootstrap_samples=bootstrap_samples,
            random_seed=self.random_seed,
        )

    def explain(
        self,
        X: Any,
        class_label: Any | None = None,
        background_data: Any | None = None,
        max_instances: int | None = None,
        num_samples: int = 500,
        *,
        y_true: Any | None = None,
        feature_importance: dict[str, float] | None = None,
        export_path: str | PathLike[str] | None = None,
        export_format: str | None = None,
    ) -> ExplanationReport:
        """Explain model predictions and optionally export row-level results."""
        self._ensure_estimator_fitted()
        if self.reliability_required:
            self._ensure_reliability_enabled()
            profiles = self._monitor._profiles
            if profiles is None:
                raise RuntimeError("ReliablePipeline has no fitted feature profile.")
            if background_data is None:
                background_data = {
                    profile.name: profile.reference_observations
                    for profile in profiles
                }
        else:
            if background_data is None:
                raise ValueError(
                    "background_data is required when reliability monitoring is disabled."
                )
        return explain_estimator(
            estimator=self.estimator,
            X=X,
            background_data=background_data,
            class_label=class_label,
            max_instances=max_instances,
            num_samples=num_samples,
            random_seed=self.random_seed,
            confidence_level=self.confidence_level,
            bootstrap_samples=max(1000, self.bootstrap_samples),
            y_true=y_true,
            feature_importance=feature_importance,
            export_path=export_path,
            export_format=export_format,
        )

    def audit_reasons(
        self,
        X: Any,
        y_true: Any,
        *,
        background_data: Any | None = None,
        positive_label: Any | None = None,
        feature_importance: dict[str, float] | None = None,
        conditional_on: tuple[str, ...] | list[str] | None = None,
        conditional_neighbors: int = 10,
        groups: dict[str, Any] | None = None,
        time_periods: Any | None = None,
        counterfactuals: dict[str, Any] | None = None,
        negative_control_features: tuple[str, ...] | list[str] = (),
        unavailable_features: tuple[str, ...] | list[str] = (),
        bootstrap_samples: int = 300,
        explanation_samples: int = 200,
        max_explained_instances: int = 100,
    ) -> ReasonAuditReport:
        """Audit a fitted classifier on labeled holdout data.

        This produces statistical diagnostics, not a guarantee of correct or
        causally justified reasoning.
        """
        self._ensure_estimator_fitted()
        columns = _as_columns(X)
        if self.reliability_required:
            self._ensure_reliability_enabled()
            profiles = self._monitor._profiles
            if profiles is None:
                raise RuntimeError("ReliablePipeline has no fitted feature profile.")
            feature_names = tuple(profile.name for profile in profiles)
            if background_data is None:
                background_data = {
                    profile.name: profile.reference_observations
                    for profile in profiles
                }
        else:
            feature_names = tuple(columns)
            if background_data is None:
                raise ValueError(
                    "background_data is required for reason auditing when "
                    "reliability monitoring is disabled."
                )
        return audit_model_reasons(
            estimator=self.estimator,
            X=X,
            y_true=y_true,
            feature_names=feature_names,
            background_data=background_data,
            positive_label=positive_label,
            feature_importance=feature_importance,
            conditional_on=conditional_on,
            conditional_neighbors=conditional_neighbors,
            groups=groups,
            time_periods=time_periods,
            counterfactuals=counterfactuals,
            negative_control_features=negative_control_features,
            unavailable_features=unavailable_features,
            bootstrap_samples=bootstrap_samples,
            confidence_level=self.confidence_level,
            explanation_samples=explanation_samples,
            max_explained_instances=max_explained_instances,
            random_seed=self.random_seed,
        )

    def plot_distributions(
        self,
        X: Any,
        feature_names: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[Any, Any]:
        self._ensure_reliability_enabled()
        return self._monitor.plot_distributions(X, feature_names)

    def plot_reference_distributions(
        self,
        feature_names: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[Any, Any]:
        self._ensure_reliability_enabled()
        return self._monitor.plot_reference_distributions(feature_names)

    def predict_with_report(self, X: Any) -> tuple[Any, ReliabilityReport]:
        predictions = self.predict(X)
        if self.last_report is None:
            raise RuntimeError(
                "predict_with_report requires reliability_required=True."
            )
        return predictions, self.last_report

    def save(self, path: str | PathLike[str]) -> None:
        self._ensure_estimator_fitted()
        with Path(path).open("wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> ReliablePipeline:
        with Path(path).open("rb") as file:
            pipeline = pickle.load(file)
        if not isinstance(pipeline, cls):
            raise TypeError(f"Artifact does not contain a {cls.__name__}.")
        return pipeline

    def _ensure_estimator_fitted(self) -> None:
        if not self._estimator_fitted:
            raise RuntimeError("ReliablePipeline must be fitted before use.")

    def _ensure_reliability_enabled(self) -> None:
        if not self.reliability_required:
            raise RuntimeError(
                "Reliability assessment is disabled; create the pipeline with "
                "reliability_required=True to enable reports and plots."
            )
        if self._monitor is None:
            raise RuntimeError("ReliablePipeline must be fitted before use.")

    def _ensure_fitted(self) -> None:
        self._ensure_reliability_enabled()


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return _json_compatible(value.tolist())
    if hasattr(value, "item"):
        return _json_compatible(value.item())
    raise TypeError(
        f"Prediction value of type {type(value).__name__} cannot be written to JSON."
    )
