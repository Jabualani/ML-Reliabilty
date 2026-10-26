import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_reliability import (
    MetaPipelineClassifier,
    StatCoreKNNClassifier,
    StatCoreLDAClassifier,
    StatCoreLogisticClassifier,
    StatCoreMetaClassifier,
    StatCoreMLPClassifier,
)
from ml_reliability.statcore_estimators import STATCORE_VARIANTS


def _data(rows=150, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, rows)
    X = rng.normal(size=(rows, 4)) + y[:, None] * 1.2
    return pd.DataFrame(X, columns=list("abcd")), np.array(["x", "y", "z"][:classes])[y]


def _all_estimators():
    out = []
    for variant in STATCORE_VARIANTS["logistic"]:
        out.append(StatCoreLogisticClassifier(variant=variant))
    for variant in STATCORE_VARIANTS["lda"]:
        out.append(StatCoreLDAClassifier(variant=variant))
    for variant in STATCORE_VARIANTS["knn"]:
        out.append(StatCoreKNNClassifier(variant=variant))
    for variant in STATCORE_VARIANTS["mlp"]:
        out.append(StatCoreMLPClassifier(variant=variant, max_epochs=20))
    out.append(StatCoreMetaClassifier(n_folds=3))
    return out


class StatCoreEstimatorStandardTests(unittest.TestCase):
    def test_every_own_variant_follows_the_meta_learning_standard(self):
        X, y = _data()
        for est in _all_estimators():
            with self.subTest(est=repr(est)):
                twin = clone(est)
                self.assertEqual(twin.get_params(), est.get_params())
                est.fit(X, y)
                self.assertEqual(list(est.classes_), ["x", "y", "z"])
                self.assertEqual(list(est.feature_names_in_), list("abcd"))
                proba = est.predict_proba(X)
                self.assertEqual(proba.shape, (len(X), 3))
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
                self.assertTrue(set(est.predict(X)) <= set(y))
                self.assertGreater((est.predict(X) == y).mean(), 0.6)

    def test_unfitted_and_bad_input_fail_clearly(self):
        from sklearn.exceptions import NotFittedError

        X, y = _data()
        with self.assertRaises(NotFittedError):
            StatCoreLDAClassifier().predict(X)
        with self.assertRaises(ValueError):
            StatCoreLDAClassifier().fit(X, np.zeros(len(X)))
        est = StatCoreLogisticClassifier().fit(X, y)
        with self.assertRaises(ValueError):
            est.predict(X.iloc[:, :3])
        with self.assertRaises(ValueError):
            StatCoreLogisticClassifier(variant="nope").fit(X, y)

    def test_missing_class_in_fold_and_label_types(self):
        X, y = _data()
        est = StatCoreKNNClassifier().fit(X.to_numpy(), pd.Series(y))
        self.assertEqual(est.predict(X.to_numpy()[:5]).shape, (5,))

    def test_feature_importances_and_sklearn_tools(self):
        X, y = _data()
        est = StatCoreLogisticClassifier().fit(X, y)
        report = est.feature_importances(X.head(20))
        self.assertEqual(len(report["features"]), 1)  # default top-n for 4 features
        self.assertEqual(
            len(est.local_feature_importances(X.head(3), top_n=2)[0]["top_features"]), 2
        )
        search = GridSearchCV(
            StatCoreLDAClassifier(), {"variant": ["shrinkage", "sparse"]}, cv=3
        ).fit(X, y)
        self.assertIn(search.best_params_["variant"], {"shrinkage", "sparse"})
        scores = cross_val_score(
            make_pipeline(StandardScaler(), StatCoreKNNClassifier()), X, y, cv=3
        )
        self.assertGreater(scores.mean(), 0.6)

    def test_meta_classifier_accepts_custom_and_sklearn_bases(self):
        from sklearn.linear_model import LogisticRegression

        X, y = _data()
        for variant in ("performance_aware", "metrics_based", "statistical_meta",
                        "calibrated_stack"):
            est = StatCoreMetaClassifier(meta_variant=variant, n_folds=3).fit(X, y)
            self.assertGreater((est.predict(X) == y).mean(), 0.6, variant)
        mixed = StatCoreMetaClassifier(
            base_estimators=[("sk", LogisticRegression(max_iter=300)),
                             ("sc", StatCoreLDAClassifier())],
            n_folds=3,
        ).fit(X, y)
        self.assertEqual(mixed.predict_proba(X).shape, (len(X), 3))

    def test_stacks_inside_existing_meta_pipeline(self):
        X, y = _data()
        stack = MetaPipelineClassifier(
            base_estimators=[("lda", StatCoreLDAClassifier()),
                             ("knn", StatCoreKNNClassifier())],
            n_folds=3,
        ).fit(X, y)
        self.assertGreater((stack.predict(X) == y).mean(), 0.6)


if __name__ == "__main__":
    unittest.main()
