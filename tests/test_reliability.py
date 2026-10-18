import math
import importlib.util
import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from ml_reliability import (
    DriftMonitor,
    ReliablePipeline,
    audit_model_reasons,
    explain_estimator,
)
from ml_reliability.deterioration import assess_deterioration
from ml_reliability.evaluation import _classification_metrics, _manova
from ml_reliability.fairness import assess_fairness


class MeanThresholdClassifier:
    def fit(self, X, y):
        self.threshold = sum(y) / len(y)
        return self

    def predict(self, X):
        return [int(value["signal"] >= 0.5) for value in X]


class ProbabilityClassifier:
    def fit(self, X, y):
        self.classes_ = [0, 1]
        return self

    def predict_proba(self, X):
        import math

        probabilities = []
        for row in X:
            score = row["signal"] * 2.0 - row["second"]
            positive = 1.0 / (1.0 + math.exp(-score))
            probabilities.append([1.0 - positive, positive])
        return probabilities

    def predict(self, X):
        return [int(probability[1] >= 0.5) for probability in self.predict_proba(X)]


class DriftMonitorTests(unittest.TestCase):
    def setUp(self):
        self.reference = [
            {"signal": 0.0, "group": "a"},
            {"signal": 0.2, "group": "a"},
            {"signal": 0.4, "group": "b"},
            {"signal": 0.6, "group": "b"},
            {"signal": 0.8, "group": "c"},
            {"signal": 1.0, "group": "c"},
        ]

    def test_reference_like_data_has_high_similarity(self):
        monitor = DriftMonitor().fit(self.reference)

        report = monitor.compare(self.reference)

        self.assertEqual(report.sample_count, len(self.reference))
        self.assertAlmostEqual(report.similarity_score, 1.0)
        self.assertEqual(report.drifted_features, ())
        self.assertEqual(report.confidence_level, 0.95)
        self.assertGreaterEqual(
            report.confidence_intervals["similarity_score"][0], 0.0
        )
        self.assertLessEqual(
            report.confidence_intervals["similarity_score"][1], 1.0
        )
        self.assertLessEqual(
            report.confidence_intervals["similarity_score"][0],
            report.similarity_score,
        )
        self.assertGreaterEqual(
            report.confidence_intervals["similarity_score"][1],
            report.similarity_score,
        )
        self.assertEqual(
            report.to_dict()["confidence_intervals"]["similarity_score"],
            list(report.confidence_intervals["similarity_score"]),
        )
        self.assertIn("mean_psi", report.confidence_intervals)
        for feature in report.features:
            self.assertEqual(feature.kl_divergence, 0.0)
            self.assertEqual(feature.psi, 0.0)
            self.assertIn("psi", feature.confidence_intervals)
            self.assertLessEqual(feature.confidence_intervals["psi"][0], feature.psi)
            self.assertGreaterEqual(
                feature.confidence_intervals["psi"][1], feature.psi
            )
            if feature.name == "signal":
                self.assertEqual(feature.ks_statistic, 0.0)
                self.assertEqual(feature.wasserstein_distance, 0.0)
            else:
                self.assertIsNone(feature.ks_statistic)
                self.assertIsNone(feature.wasserstein_distance)

    def test_shifted_numeric_and_categorical_features_are_reported(self):
        monitor = DriftMonitor(drift_threshold=0.2).fit(self.reference)
        shifted = [
            {"signal": 10.0, "group": "new"},
            {"signal": 11.0, "group": "new"},
            {"signal": 12.0, "group": "new"},
        ]

        report = monitor.compare(shifted)

        self.assertEqual(report.drifted_features, ("signal", "group"))
        self.assertLess(report.similarity_score, 0.1)
        self.assertEqual(report.to_dict()["drifted_features"], ["signal", "group"])
        numeric = report.features[0]
        categorical = report.features[1]
        self.assertEqual(numeric.ks_statistic, 1.0)
        self.assertGreater(numeric.kl_divergence, 0.0)
        self.assertGreater(numeric.psi, 0.0)
        self.assertGreater(numeric.wasserstein_distance, 0.0)
        self.assertIsNone(categorical.ks_statistic)
        self.assertGreater(categorical.kl_divergence, 0.0)
        self.assertGreater(categorical.psi, 0.0)
        self.assertIsNone(categorical.wasserstein_distance)
        self.assertIn("wasserstein_distance", report.to_dict()["features"][0])
        self.assertTrue(report.covariate_shift_detected)

    def test_ks_and_wasserstein_match_known_empirical_values(self):
        monitor = DriftMonitor().fit([{"x": 0.0}, {"x": 1.0}])

        feature = monitor.compare([{"x": 2.0}, {"x": 3.0}]).features[0]

        self.assertEqual(feature.ks_statistic, 1.0)
        self.assertAlmostEqual(feature.wasserstein_distance, 2.0)

    def test_dsi_detects_changes_in_numeric_moments(self):
        monitor = DriftMonitor().fit(
            [{"measurement": value} for value in [0, 1, 2, 3, 4, 5]]
        )

        feature = monitor.compare(
            [{"measurement": value} for value in [10, 11, 12, 13, 14, 15]]
        ).features[0]

        self.assertGreater(feature.dsi, 1.0)
        self.assertTrue(feature.dsi_drifted)

    def test_categorical_chi_square_detects_distribution_change(self):
        reference = [{"group": group} for group in ["a", "b", "c"] * 20]
        incoming = [{"group": "new"} for _ in range(60)]
        monitor = DriftMonitor().fit(reference)

        feature = monitor.assess_covariate_shift(incoming).features[0]

        self.assertGreater(feature.chi_square_statistic, 0.0)
        self.assertLess(feature.chi_square_p_value, 0.05)
        self.assertTrue(feature.chi_square_drifted)

    def test_chi_square_p_value_matches_known_two_category_case(self):
        monitor = DriftMonitor().fit(
            [{"group": group} for group in ["a"] * 50 + ["b"] * 50]
        )

        feature = monitor.assess_covariate_shift(
            [{"group": group} for group in ["a"] * 40 + ["b"] * 60]
        ).features[0]

        self.assertAlmostEqual(feature.chi_square_statistic, 2.0202, delta=0.001)
        self.assertAlmostEqual(feature.chi_square_p_value, 0.155, delta=0.002)

    def test_concept_shift_detects_changed_feature_target_relationship(self):
        reference = [
            {"segment": segment}
            for segment in ["a", "a", "a", "b", "b", "b"] * 5
        ]
        reference_y = [0, 0, 0, 1, 1, 1] * 5
        incoming_y = [1, 1, 1, 0, 0, 0] * 5
        monitor = DriftMonitor().fit(reference, reference_y)

        report = monitor.assess_concept_shift(reference, incoming_y)

        self.assertEqual(report.shifted_features, ("segment",))
        self.assertAlmostEqual(report.mean_js_divergence, math.log(2), places=5)
        self.assertGreater(report.features[0].kl_divergence, 0)
        self.assertGreater(report.features[0].psi, 0)
        self.assertEqual(
            report.to_dict()["shifted_features"], ["segment"]
        )

    def test_concept_shift_requires_training_labels(self):
        monitor = DriftMonitor().fit(self.reference)

        with self.assertRaisesRegex(RuntimeError, "requires fitting.*with y"):
            monitor.assess_concept_shift(self.reference, [0, 0, 0, 1, 1, 1])

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib")
        and importlib.util.find_spec("seaborn"),
        "Install the visualization dependencies to run plotting tests.",
    )
    def test_distribution_plot_handles_numeric_and_categorical_features(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        monitor = DriftMonitor().fit(self.reference)

        figure, axes = monitor.plot_distributions(
            [{"signal": 1.2, "group": "new"}, {"signal": 1.4, "group": "a"}]
        )

        self.assertEqual(axes.shape, (1, 2))
        self.assertIn("signal", axes[0, 0].get_title())
        self.assertGreater(len(axes[0, 0].collections), 0)
        self.assertIn("PSI=", axes[0, 0].get_title())
        self.assertIn("[", axes[0, 0].get_title())
        self.assertIn("<unseen category>", [
            tick.get_text() for tick in axes[0, 1].get_xticklabels()
        ])
        plt.close(figure)

    def test_plotting_unfitted_monitor_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            DriftMonitor().plot_distributions(self.reference)

    def test_constant_numeric_reference_flags_values_outside_its_range(self):
        monitor = DriftMonitor().fit([{"constant": 1.0}, {"constant": 1.0}])

        report = monitor.compare([{"constant": 2.0}, {"constant": 3.0}])

        self.assertEqual(report.drifted_features, ("constant",))

    def test_column_mismatch_is_explicit(self):
        monitor = DriftMonitor().fit(self.reference)

        with self.assertRaisesRegex(ValueError, "Feature columns do not match"):
            monitor.compare([{"signal": 0.3, "other": "a"}])

    def test_pipeline_predicts_reports_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "reliability_plots"
            pipeline = ReliablePipeline(
                MeanThresholdClassifier(),
                plot_dir=output_directory,
                bootstrap_samples=40,
            ).fit(self.reference, [0, 0, 0, 1, 1, 1])
            incoming = [{"signal": 0.2, "group": "a"}]
            self.assertTrue((output_directory / "training-distributions.png").exists())
            predictions, report = pipeline.predict_with_report(incoming)

            self.assertEqual(predictions, [0])
            self.assertEqual(report.sample_count, 1)
            self.assertIs(pipeline.last_report, report)
            self.assertTrue(pipeline.last_plot_path.is_file())
            self.assertTrue(pipeline.last_report_path.is_file())
            with pipeline.last_report_path.open(encoding="utf-8") as file:
                saved_report = json.load(file)
            self.assertIn("similarity_score", saved_report["confidence_intervals"])
            self.assertEqual(saved_report["predictions"], predictions)
            self.assertEqual(
                saved_report["output_type"],
                "predictions_and_reliability_report",
            )
            concept_report = pipeline.assess_concept_shift(incoming, [0])
            self.assertEqual(concept_report.sample_count, 1)

            path = Path(directory) / "pipeline.pkl"
            pipeline.save(path)
            loaded = ReliablePipeline.load(path)
            self.assertEqual(loaded.predict(incoming), predictions)
            self.assertEqual(
                loaded.assess_reliability(incoming).to_dict(),
                loaded.last_report.to_dict(),
            )

    def test_reliability_can_be_explicitly_disabled_for_lower_memory_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "not-created"
            pipeline = ReliablePipeline(
                MeanThresholdClassifier(),
                plot_dir=output_directory,
                reliability_required=False,
            ).fit(self.reference, [0, 0, 0, 1, 1, 1])
            incoming = [{"signal": 0.2, "group": "a"}]

            predictions = pipeline.predict(incoming)

            self.assertEqual(predictions, [0])
            self.assertIsNone(pipeline.last_report)
            self.assertFalse(output_directory.exists())
            self.assertFalse(pipeline.reliability_required)
            with self.assertRaisesRegex(RuntimeError, "requires reliability_required=True"):
                pipeline.predict_with_report(incoming)
            with self.assertRaisesRegex(RuntimeError, "Reliability assessment is disabled"):
                pipeline.assess_reliability(incoming)

    def test_evaluate_reports_classification_stats_tests_and_bias_variance(self):
        training = [
            {"signal": float(index * 0.05), "second": float(index % 4), "group": "low"}
            for index in range(10)
        ] + [
            {"signal": float(index + 20), "second": float(index % 4 + 10), "group": "high"}
            for index in range(10)
        ]
        training_y = [0] * 10 + [1] * 10
        test = [
            {"signal": float(index * 0.05), "second": float(index % 3), "group": "low"}
            for index in range(6)
        ] + [
            {"signal": float(index + 20), "second": float(index % 3 + 10), "group": "high"}
            for index in range(6)
        ]
        test_y = [0] * 6 + [1] * 6
        with tempfile.TemporaryDirectory() as directory:
            pipeline = ReliablePipeline(
                MeanThresholdClassifier(),
                plot_dir=Path(directory) / "plots",
                bootstrap_samples=20,
                bias_variance_folds=3,
                bias_variance_repeats=2,
            ).fit(training, training_y)

            evaluation = pipeline.evaluate(test, test_y)
            output = evaluation.to_dict()
            classification = output["classification"]
            tests = output["statistical_tests"]

            self.assertEqual(classification["accuracy"], 1.0)
            self.assertEqual(classification["matthews_correlation_coefficient"], 1.0)
            self.assertEqual(classification["confusion_matrix"], [[6, 0], [0, 6]])
            self.assertGreater(
                tests["feature_target_associations"]["anova_numeric_features"][0][
                    "statistic"
                ],
                0,
            )
            self.assertEqual(
                tests["feature_target_associations"]["manova_numeric_features"][
                    "status"
                ],
                "ok",
            )
            self.assertIn(
                "chi_square_categorical_features",
                tests["feature_target_associations"],
            )
            self.assertEqual(
                tests["training_vs_test_cohorts"]["manova_numeric_features"][
                    "status"
                ],
                "ok",
            )
            self.assertEqual(output["bias_variance"]["folds"], 3)
            self.assertEqual(output["bias_variance"]["repeats"], 2)
            self.assertAlmostEqual(
                output["bias_variance"]["bias_squared"]
                + output["bias_variance"]["prediction_variance"],
                output["bias_variance"]["mean_squared_probability_error"],
            )
            self.assertIsNotNone(pipeline.last_report.evaluation)
            with pipeline.last_report_path.open(encoding="utf-8") as file:
                saved_report = json.load(file)
            self.assertEqual(saved_report["predictions"], [0] * 6 + [1] * 6)
            self.assertIn("classification", saved_report["evaluation"])

    def test_evaluate_rejects_when_reliability_is_disabled(self):
        pipeline = ReliablePipeline(
            MeanThresholdClassifier(),
            reliability_required=False,
        ).fit(self.reference, [0, 0, 0, 1, 1, 1])

        with self.assertRaisesRegex(RuntimeError, "Reliability assessment is disabled"):
            pipeline.evaluate(self.reference, [0, 0, 0, 1, 1, 1])

    def test_multiclass_mcc_and_joint_manova_cover_all_groups(self):
        import numpy as np

        metrics = _classification_metrics(
            [0, 0, 1, 1, 2, 2, 0, 1, 2],
            [np.int64(value) for value in [0, 1, 1, 2, 2, 0, 0, 1, 2]],
        )
        manova = _manova(
            {
                "first": [0.0, 0.2, 1.0, 1.2, 2.0, 2.2, 0.1, 1.1, 2.1],
                "second": [0.1, 0.5, 0.2, 0.8, 0.4, 1.0, 0.3, 0.6, 0.9],
            },
            ["a", "a", "b", "b", "c", "c", "a", "b", "c"],
            "target_class",
        )

        self.assertAlmostEqual(
            metrics["matthews_correlation_coefficient"], 0.5
        )
        self.assertEqual(manova["status"], "ok")
        self.assertEqual(manova["group_count"], 3)
        self.assertEqual(manova["numerator_df"], 4.0)

    def test_manova_reports_singular_residual_covariance_as_unavailable(self):
        result = _manova(
            {
                "first": [0.0, 0.2, 1.0, 1.2, 2.0, 2.2, 0.1, 1.1, 2.1],
                "dependent": [0.1, 0.3, 1.1, 1.3, 2.1, 2.3, 0.2, 1.2, 2.2],
            },
            ["a", "a", "b", "b", "c", "c", "a", "b", "c"],
            "target_class",
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("residual covariance is singular", result["reason"])

    def test_fairness_reports_three_parity_metrics_and_bootstrap_intervals(self):
        report = assess_fairness(
            y_true=[1, 0, 1, 0],
            y_pred=[1, 0, 0, 1],
            sensitive_features={"group": ["A", "A", "B", "B"]},
            positive_label=1,
            labels=[0, 1],
            bootstrap_samples=100,
        )
        metrics = report["group"]["classes"]["1"]

        self.assertEqual(metrics["statistical_parity"]["gap"], 0.0)
        self.assertEqual(
            metrics["equalized_odds"]["true_positive_rate_gap"]["gap"], 1.0
        )
        self.assertEqual(
            metrics["equalized_odds"]["false_positive_rate_gap"]["gap"], 1.0
        )
        self.assertEqual(metrics["equalized_odds"]["maximum_gap"], 1.0)
        self.assertEqual(metrics["predictive_parity"]["gap"], 1.0)
        self.assertIsNotNone(
            metrics["equalized_odds"]["maximum_gap_confidence_interval"]
        )

    def test_fairness_requires_explicit_binary_positive_label(self):
        with self.assertRaisesRegex(ValueError, "positive_label is required"):
            assess_fairness(
                [0, 1],
                [0, 1],
                {"group": ["A", "B"]},
                positive_label=None,
                labels=[0, 1],
                bootstrap_samples=100,
            )

    def test_multiclass_fairness_is_one_vs_rest_and_normalizes_numpy_scalars(self):
        import numpy as np

        report = assess_fairness(
            y_true=[0, 1, 2, 0, 1, 2],
            y_pred=[0, 1, 0, 0, 2, 2],
            sensitive_features={"group": ["A", "A", "A", "B", "B", "B"]},
            positive_label=None,
            labels=[np.int64(0), np.int64(1), np.int64(2)],
            bootstrap_samples=100,
        )

        self.assertEqual(report["group"]["method"], "one_vs_rest_per_class")
        self.assertEqual(len(report["group"]["classes"]), 3)
        self.assertEqual(
            report["group"]["classes"]["1"]["statistical_parity"]["gap"],
            1 / 3,
        )

    def test_deterioration_detects_a_decreasing_accuracy_trend(self):
        history = []
        for timestamp, correct_count in enumerate((100, 80, 60, 40)):
            history.append(
                {
                    "timestamp": timestamp,
                    "y_true": [1] * 100,
                    "y_pred": [1] * correct_count + [0] * (100 - correct_count),
                }
            )

        report = assess_deterioration(history, bootstrap_samples=200)

        self.assertLess(report.accuracy_slope, 0)
        self.assertLess(report.slope_confidence_interval[1], 0)
        self.assertTrue(report.deterioration_detected)
        self.assertEqual(len(report.to_dict()["observations"]), 4)

    def test_explanations_compare_global_shapley_with_local_lime(self):
        training = [
            {"signal": float(index) / 10, "second": float(index % 4) / 3}
            for index in range(20)
        ]
        labels = [int(row["signal"] > 0.5) for row in training]
        with tempfile.TemporaryDirectory() as directory:
            pipeline = ReliablePipeline(
                ProbabilityClassifier(),
                plot_dir=Path(directory) / "plots",
                bootstrap_samples=20,
            ).fit(training, labels)

            report = pipeline.explain(training[::4], num_samples=100, max_instances=3)

        self.assertEqual(len(report.global_shap_importance), 2)
        self.assertEqual(len(report.local_explanations), 3)
        self.assertIn("mean_absolute_importance_spearman", report.reason_alignment)
        self.assertIn("alignment_features", report.reason_alignment)
        self.assertIn("lime_local_fidelity_r_squared", report.local_explanations[0])
        self.assertIn("not a probability of correctness", " ".join(report.to_dict()["limitations"]))

    def test_explanations_work_without_monitor_when_background_is_supplied(self):
        training = [
            {"signal": float(index) / 10, "second": float(index % 4) / 3}
            for index in range(20)
        ]
        labels = [int(row["signal"] > 0.5) for row in training]
        pipeline = ReliablePipeline(
            ProbabilityClassifier(),
            reliability_required=False,
        ).fit(training, labels)

        report = pipeline.explain(
            training[:2],
            background_data=training[2:12],
            num_samples=100,
        )

        self.assertEqual(report.reason_alignment["explained_instance_count"], 2)

    def test_external_estimator_can_be_explained_without_labels(self):
        training = [
            {"signal": float(index) / 10, "second": float(index % 4) / 3}
            for index in range(20)
        ]
        model = ProbabilityClassifier().fit(
            training, [int(row["signal"] > 0.5) for row in training]
        )
        report = explain_estimator(
            model,
            training[:4],
            background_data=training[4:],
            feature_importance={"signal": 0.8, "second": 0.2},
            num_samples=100,
        )

        result = report.to_dict()
        self.assertEqual(len(result["local_explanations"]), 4)
        self.assertIn("prediction", result["local_explanations"][0])
        self.assertEqual(result["label_evaluation"], None)
        self.assertEqual(
            result["supplied_importance_alignment"]["source"],
            "user-supplied; source/method not independently validated",
        )

    def test_explanation_export_includes_rows_labels_predictions_and_top_features(self):
        training = [
            {"signal": float(index) / 10, "second": float(index % 4) / 3}
            for index in range(12)
        ]
        labels = [int(row["signal"] > 0.5) for row in training]
        model = ProbabilityClassifier().fit(training, labels)

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "explanations.csv"
            report = explain_estimator(
                model,
                training,
                y_true=labels,
                background_data=training,
                num_samples=100,
                export_path=csv_path,
            )
            with csv_path.open(newline="", encoding="utf-8") as file:
                csv_rows = list(csv.DictReader(file))

            pickle_path = Path(directory) / "explanations.pkl"
            explain_estimator(
                model,
                training,
                y_true=labels,
                background_data=training,
                num_samples=100,
                export_path=pickle_path,
            )
            with pickle_path.open("rb") as file:
                pickle_rows = pickle.load(file)

        self.assertEqual(len(csv_rows), len(training))
        self.assertEqual(csv_rows[0]["signal"], str(training[0]["signal"]))
        self.assertEqual(csv_rows[0]["y_true"], str(labels[0]))
        self.assertEqual(csv_rows[0]["prediction"], str(model.predict(training)[0]))
        self.assertTrue(json.loads(csv_rows[0]["top_features"]))
        self.assertEqual(len(pickle_rows), len(training))
        self.assertEqual(pickle_rows[0]["y_true"], labels[0])
        self.assertEqual(report.export_path, str(csv_path))
        self.assertEqual(report.label_evaluation["sample_count"], len(training))

    def test_explanation_export_supports_joblib_when_installed(self):
        if importlib.util.find_spec("joblib") is None:
            self.skipTest("joblib is an optional export dependency.")
        import joblib

        training = [
            {"signal": float(index) / 10, "second": float(index % 4) / 3}
            for index in range(10)
        ]
        model = ProbabilityClassifier().fit(
            training, [int(row["signal"] > 0.5) for row in training]
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "explanations.joblib"
            explain_estimator(
                model,
                training,
                background_data=training,
                num_samples=100,
                export_path=destination,
            )
            rows = joblib.load(destination)

        self.assertEqual(len(rows), len(training))
        self.assertIn("top_features", rows[0])

    def test_pipeline_explain_exports_its_predictions(self):
        training = [
            {"signal": float(index) / 10, "second": float(index % 4) / 3}
            for index in range(12)
        ]
        labels = [int(row["signal"] > 0.5) for row in training]
        pipeline = ReliablePipeline(
            ProbabilityClassifier(),
            reliability_required=False,
        ).fit(training, labels)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pipeline-explanations.csv"
            report = pipeline.explain(
                training,
                background_data=training,
                y_true=labels,
                num_samples=100,
                export_path=destination,
            )
            with destination.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), len(training))
        self.assertEqual(report.export_path, str(destination))
        self.assertEqual(
            rows[0]["prediction"], str(pipeline.estimator.predict(training)[0])
        )

    def test_statistical_reason_audit_reports_validity_reliance_and_alignment(self):
        training = [
            {
                "signal": index / 30,
                "second": ((index * 7) % 29) / 29,
            }
            for index in range(30)
        ]
        labels = [
            int((row["signal"] > 0.5) != (index % 7 == 0))
            for index, row in enumerate(training)
        ]
        model = ProbabilityClassifier().fit(training, labels)
        audit_rows = [
            {
                "signal": (index + 0.5) / 30,
                "second": ((index * 11) % 31) / 31,
            }
            for index in range(30)
        ]
        audit_labels = [
            int((row["signal"] > 0.5) != (index % 6 == 0))
            for index, row in enumerate(audit_rows)
        ]
        report = audit_model_reasons(
            model,
            audit_rows,
            audit_labels,
            background_data=training,
            feature_importance={"signal": 0.8, "second": 0.2},
            conditional_on=["second"],
            groups={"cohort": ["A" if index % 2 else "B" for index in range(30)]},
            time_periods=[index // 10 for index in range(30)],
            counterfactuals={"signal": [0.0] * len(audit_rows)},
            negative_control_features=["second"],
            unavailable_features=["second"],
            bootstrap_samples=100,
            explanation_samples=100,
            max_explained_instances=5,
        )
        result = report.to_dict()

        self.assertEqual(result["sample_count"], 30)
        self.assertIn("accuracy_confidence_interval", result["predictive_validity"])
        self.assertIn("brier_score_confidence_interval", result["predictive_validity"])
        reliance = {item["feature"]: item for item in result["feature_reliance"]}
        self.assertEqual(
            reliance["signal"]["perturbation_method"],
            "nearest-neighbor conditional donor resampling",
        )
        self.assertEqual(reliance["signal"]["conditioned_on_features"], ["second"])
        self.assertIn("loss_increase_confidence_interval", reliance["signal"])
        self.assertIn(
            "negative_control_warning",
            reliance["second"],
        )
        self.assertIn("cohort", result["stability"]["groups"])
        self.assertIn("time_period", result["stability"]["time_periods"])
        self.assertIn(
            "prediction_flip_rate_confidence_interval",
            result["counterfactual_sensitivity"]["features"]["signal"],
        )
        self.assertEqual(result["logistic_association"]["status"], "ok")
        self.assertIn("fdr_bh_adjusted_p_value", result["logistic_association"]["odds_ratios"]["signal"])
        self.assertEqual(result["explanation_alignment"]["status"], "ok")
        self.assertEqual(result["supplied_importance_alignment"]["status"], "ok")
        self.assertIn("cannot establish that a model used causally correct reasons", result["interpretation"])
        self.assertEqual(
            result["explanations"]["local_explanations"][0]["top_features"][0][
                "feature"
            ],
            "signal",
        )

    def test_pipeline_reason_audit_uses_training_reference(self):
        training = [
            {"signal": index / 20, "second": (index % 5) / 5}
            for index in range(40)
        ]
        labels = [int(index >= 20) for index in range(40)]
        holdout = training[20:]
        holdout_labels = labels[20:]
        with tempfile.TemporaryDirectory() as directory:
            pipeline = ReliablePipeline(
                ProbabilityClassifier(),
                plot_dir=Path(directory) / "plots",
                bootstrap_samples=20,
            ).fit(training, labels)
            report = pipeline.audit_reasons(
                holdout,
                holdout_labels,
                bootstrap_samples=100,
                explanation_samples=100,
                max_explained_instances=2,
            )

        self.assertEqual(report.sample_count, 20)
        self.assertNotIn(
            "No training/reference background was supplied; X itself was used as the explanation background.",
            report.warnings,
        )

    def test_pipeline_reason_audit_requires_background_when_monitoring_disabled(self):
        training = [
            {"signal": index / 20, "second": (index % 5) / 5}
            for index in range(40)
        ]
        labels = [int(index >= 20) for index in range(40)]
        pipeline = ReliablePipeline(
            ProbabilityClassifier(),
            reliability_required=False,
        ).fit(training, labels)

        with self.assertRaisesRegex(ValueError, "background_data is required"):
            pipeline.audit_reasons(
                training[20:],
                labels[20:],
                bootstrap_samples=100,
                explanation_samples=100,
            )

        report = pipeline.audit_reasons(
            training[20:],
            labels[20:],
            background_data=training[:20],
            bootstrap_samples=100,
            explanation_samples=100,
        )
        self.assertNotIn(
            "No training/reference background was supplied; X itself was used as the explanation background.",
            report.warnings,
        )

    def test_pipeline_evaluation_attaches_fairness_report(self):
        data = [
            {"signal": 0.0, "group": "A"},
            {"signal": 1.0, "group": "A"},
            {"signal": 0.0, "group": "B"},
            {"signal": 1.0, "group": "B"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            pipeline = ReliablePipeline(
                MeanThresholdClassifier(),
                plot_dir=Path(directory) / "plots",
                bootstrap_samples=100,
                bias_variance_folds=2,
                bias_variance_repeats=1,
            ).fit(data, [0, 1, 0, 1])
            evaluation = pipeline.evaluate(
                data,
                [0, 1, 0, 1],
                sensitive_features={"demographic_group": ["A", "A", "B", "B"]},
                positive_label=1,
            )
            with pipeline.last_report_path.open(encoding="utf-8") as file:
                saved_report = json.load(file)

        self.assertEqual(
            evaluation.to_dict()["fairness"]["demographic_group"]["status"],
            "ok",
        )
        self.assertIn(
            "fairness",
            pipeline.last_report.evaluation,
        )
        self.assertIn("demographic_group", saved_report["evaluation"]["fairness"])

    def test_unfitted_pipeline_cannot_be_used_or_saved(self):
        pipeline = ReliablePipeline(MeanThresholdClassifier())

        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            pipeline.predict([])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "must be fitted"):
                pipeline.save(Path(directory) / "pipeline.pkl")


if __name__ == "__main__":
    unittest.main()
