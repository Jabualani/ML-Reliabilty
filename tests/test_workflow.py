import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_reliability import (
    MetaPipelineClassifier,
    ReliablePipeline,
    StatisticalDataCleaner,
)


class RecommendedWorkflowTests(unittest.TestCase):
    def test_dirty_data_modeling_evaluation_and_monitoring_workflow(self):
        rng = np.random.default_rng(27)
        values = rng.normal(size=(120, 3))
        labels = ((values[:, 0] + 0.4 * values[:, 1]) > 0).astype(int)
        features = pd.DataFrame(values, columns=["signal", "second", "third"])
        missing_rows = rng.choice(len(features), size=18, replace=False)
        features.loc[missing_rows, "second"] = np.nan

        train_indices, test_indices = train_test_split(
            np.arange(len(features)),
            test_size=0.25,
            random_state=14,
            stratify=labels,
        )
        X_train, X_test = features.iloc[train_indices], features.iloc[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        groups = {
            "cohort": np.where(test_indices % 2 == 0, "A", "B"),
        }

        estimator = Pipeline(
            [
                ("cleaner", StatisticalDataCleaner()),
                ("scaler", StandardScaler()),
                (
                    "meta",
                    MetaPipelineClassifier(
                        base_estimators=[
                            ("logistic", LogisticRegression(max_iter=300)),
                        ],
                        n_folds=3,
                        random_state=14,
                    ),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            model = ReliablePipeline(
                estimator,
                plot_dir=Path(directory) / "plots",
                bootstrap_samples=100,
                bias_variance_folds=2,
                bias_variance_repeats=1,
            ).fit(X_train, y_train)

            evaluation = model.evaluate(
                X_test,
                y_test,
                sensitive_features=groups,
                positive_label=1,
            )
            self.assertEqual(evaluation.sample_count, len(X_test))
            self.assertEqual(evaluation.fairness["cohort"]["status"], "ok")

            shifted = X_test.copy()
            shifted["signal"] = shifted["signal"].fillna(0.0) + 5.0
            drift_report = model.assess_covariate_shift(shifted)
            self.assertEqual(drift_report.sample_count, len(X_test))
            concept_report = model.assess_concept_shift(X_test, y_test)
            self.assertEqual(concept_report.sample_count, len(X_test))

            cleaner = model.estimator.named_steps["cleaner"]
            clean_train = cleaner.transform(X_train)
            clean_test = cleaner.transform(X_test)
            explanation = model.explain(
                clean_test,
                background_data=clean_train,
                class_label=1,
                max_instances=2,
                num_samples=100,
            )
            self.assertEqual(len(explanation.local_explanations), 2)

            audited = model.audit_reasons(
                clean_test,
                y_test,
                background_data=clean_train,
                positive_label=1,
                bootstrap_samples=100,
                explanation_samples=100,
                max_explained_instances=2,
            )
            self.assertEqual(audited.sample_count, len(X_test))

            predictions = model.estimator.predict(X_test)
            history = []
            for index, timestamp in enumerate(
                ("2026-01-01", "2026-02-01", "2026-03-01")
            ):
                degraded = predictions.copy()
                degraded[: index * (len(degraded) // 3)] = (
                    1 - degraded[: index * (len(degraded) // 3)]
                )
                history.append(
                    {
                        "timestamp": timestamp,
                        "y_true": y_test,
                        "y_pred": degraded,
                    }
                )
            deterioration = model.assess_deterioration(
                history,
                bootstrap_samples=100,
            )
            self.assertEqual(len(deterioration.observations), 3)

            figure, _ = model.plot_distributions(X_test)
            import matplotlib.pyplot as plt

            plt.close(figure)
            predictions, report = model.predict_with_report(X_test)
            self.assertEqual(len(predictions), len(X_test))
            self.assertEqual(report.sample_count, len(X_test))
            self.assertTrue(model.last_plot_path.exists())
            self.assertTrue(model.last_report_path.exists())
