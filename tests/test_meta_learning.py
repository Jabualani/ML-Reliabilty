import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ml_reliability import (
    CalibratedStackedClassifier,
    MetaFeatureTransformer,
    MetaPipelineClassifier,
    PerfMetaSelect,
    ReliablePipeline,
    StatisticalMetaClassifier,
    StatisticalMetaDTreeClassifier,
    StatisticalMetaLDAClassifier,
    StatisticalMetaNNClassifier,
    halving_grid_search,
    halving_random_search,
)


def _dataset():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(72, 4))
    y = ((X[:, 0] + 0.7 * X[:, 1] - 0.3 * X[:, 2]) > 0).astype(int)
    return X, y


class MetaLearningTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _dataset()
        self.base_estimators = [
            ("logreg", LogisticRegression(max_iter=300)),
            ("svc", SVC(probability=False)),
        ]

    def test_performance_selector_and_oof_meta_pipeline(self):
        selected = PerfMetaSelect(
            estimators=self.base_estimators,
            cv=3,
            scoring="accuracy",
        ).fit(self.X, self.y)
        self.assertIn(selected.best_name_, {"logreg", "svc"})
        self.assertEqual(selected.predict_proba(self.X[:5]).shape, (5, 2))

        stack = MetaPipelineClassifier(
            self.base_estimators,
            meta_classifier=LogisticRegression(max_iter=300),
            n_folds=3,
        ).fit(self.X, self.y)
        self.assertEqual(stack.meta_features_.shape, (len(self.y), 4))
        self.assertEqual(stack.predict_proba(self.X[:5]).shape, (5, 2))
        self.assertEqual(len(stack.predict(self.X[:5])), 5)

    def test_oof_meta_stack_supports_multiclass(self):
        rng = np.random.default_rng(91)
        X = rng.normal(size=(90, 3))
        y = np.arange(90) % 3
        X[:, 0] += y * 1.5
        model = MetaPipelineClassifier(
            [("logreg", LogisticRegression(max_iter=300))],
            n_folds=3,
        ).fit(X, y)

        probabilities = model.predict_proba(X[:8])

        self.assertEqual(probabilities.shape, (8, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_calibrated_stack_and_transformer_pipeline(self):
        stack = CalibratedStackedClassifier(
            base_estimators=self.base_estimators,
            n_folds=3,
        ).fit(self.X, self.y)
        probabilities = stack.predict_proba(self.X[:6])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

        pipeline = Pipeline(
            [
                (
                    "meta",
                    MetaFeatureTransformer(
                        estimators=self.base_estimators,
                        cv=3,
                    ),
                ),
                ("classifier", LogisticRegression(max_iter=300)),
            ]
        ).fit(self.X, self.y)
        self.assertEqual(pipeline.predict(self.X[:5]).shape, (5,))
        self.assertEqual(
            pipeline.named_steps["meta"].transform(self.X[:5]).shape,
            (5, 4),
        )

    def test_statistical_linear_and_nonlinear_classifiers(self):
        models = [
            StatisticalMetaClassifier(
                n_clusters=3,
                cv=3,
                use_monte_carlo=False,
            ),
            StatisticalMetaLDAClassifier(
                n_clusters=3,
                cv=3,
                estimation_method="irls",
                use_monte_carlo=False,
            ),
            StatisticalMetaDTreeClassifier(
                n_clusters=3,
                cv=3,
                use_monte_carlo=False,
                use_polynomial_interactions=True,
            ),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                fitted = clone(model).fit(self.X, self.y)
                probabilities = fitted.predict_proba(self.X[:7])
                self.assertEqual(probabilities.shape, (7, 2))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertEqual(fitted.predict(self.X[:7]).shape, (7,))

    def test_statistical_classifier_bootstrap_parameter_averaging(self):
        model = StatisticalMetaClassifier(
            n_clusters=3,
            cv=3,
            use_monte_carlo=True,
            n_mc_sims=3,
        ).fit(self.X, self.y)

        self.assertTrue(np.isfinite(model.beta_).all())
        self.assertTrue(np.isfinite(model.predict_proba(self.X[:5])).all())

    def test_global_and_local_raw_feature_importance(self):
        model = MetaPipelineClassifier(
            self.base_estimators,
            n_folds=3,
        ).fit(self.X, self.y)
        global_report = model.feature_importances(self.X[:12], top_n=2)
        local_report = model.local_feature_importances(self.X[:3], top_n=2)

        self.assertEqual(len(global_report["features"]), 2)
        self.assertEqual(len(local_report), 3)
        self.assertEqual(len(local_report[0]["top_features"]), 2)
        self.assertGreater(
            sum(item["top_feature_frequency"] for item in global_report["features"]),
            0,
        )
        self.assertLessEqual(
            sum(item["top_feature_frequency"] for item in global_report["features"]),
            12 * 2,
        )

    def test_meta_model_round_trips_inside_reliable_pipeline(self):
        estimator = MetaPipelineClassifier(
            self.base_estimators,
            n_folds=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            model = ReliablePipeline(
                estimator,
                plot_dir=Path(directory) / "plots",
                bootstrap_samples=20,
            ).fit(self.X, self.y)
            expected = model.estimator.predict(self.X[:8])
            artifact = Path(directory) / "model.pkl"
            model.save(artifact)
            loaded = ReliablePipeline.load(artifact)

        np.testing.assert_array_equal(loaded.estimator.predict(self.X[:8]), expected)

    def test_halving_search_factories_fit_meta_estimators(self):
        estimator = MetaPipelineClassifier(
            self.base_estimators[:1],
            meta_classifier=LogisticRegression(max_iter=300),
            n_folds=2,
        )
        grid = halving_grid_search(
            estimator,
            {"meta_classifier__C": [0.1, 1.0]},
            cv=2,
            factor=2,
            min_resources=24,
            max_resources=72,
            scoring="accuracy",
        ).fit(self.X, self.y)
        self.assertTrue(hasattr(grid, "best_estimator_"))

        randomized = halving_random_search(
            estimator,
            {"meta_classifier__C": [0.1, 1.0, 10.0]},
            n_candidates=2,
            cv=2,
            factor=2,
            min_resources=24,
            max_resources=72,
            scoring="accuracy",
            random_state=42,
        ).fit(self.X, self.y)
        self.assertTrue(hasattr(randomized, "best_estimator_"))

    def test_optional_neural_meta_classifier(self):
        model = StatisticalMetaNNClassifier(
            n_clusters=3,
            cv=3,
            epochs=2,
            batch_size=16,
            hidden_dims=(8,),
            n_mc_sims=2,
        ).fit(self.X, self.y)
        self.assertEqual(model.predict_proba(self.X[:4]).shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
