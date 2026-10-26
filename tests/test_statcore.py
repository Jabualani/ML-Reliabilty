"""Tests for ml_reliability.statcore (pure-NumPy models, every variant)."""

import unittest

import numpy as np

from ml_reliability.statcore import (LDA, PCA, KMeans, KNNClassifier, KNNRegressor,
                                     LogisticRegression, MetaLearner, MLPClassifier,
                                     MLPRegressor, NotFittedError, RidgeRegression)
from ml_reliability.statcore.evaluation.metrics import classification_metrics

CLS_KEYS = {"accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc"}
REG_KEYS = {"mse", "mae", "r2"}
REC_KEYS = {"reconstruction_mse", "reconstruction_mae", "relative_frobenius"}
CLU_KEYS = {"inertia", "n_clusters_found"}


def blobs(n=60, centers=((0, 0, 0), (5, 5, 0), (0, 5, 5)), seed=0, noise=1.0, extra=0):
    rng = np.random.default_rng(seed)
    C = np.array(centers, dtype=float)
    X = np.vstack([rng.normal(c, noise, size=(n, C.shape[1])) for c in C])
    y = np.repeat(np.arange(len(C)), n)
    if extra:
        X = np.hstack([X, rng.normal(size=(len(X), extra))])
    p = rng.permutation(len(X))
    return X[p], y[p]


def regression_data(n=150, seed=0, outliers=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X @ np.array([2.0, -1.0, 0.5, 0.0]) + 1.0 + 0.1 * rng.normal(size=n)
    if outliers:
        y[:5] += 50
    return X, y


def check_cls(tc, d):
    tc.assertTrue(CLS_KEYS <= set(d))
    tc.assertTrue(all(np.isfinite(v) for v in d.values()))


class TestPCA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(100, 6)) @ rng.normal(size=(6, 6))
        self.y = (self.X[:, 0] > 0).astype(int)

    def test_variants(self):
        for v in ("base", "supervised_row_aware", "robust", "sparse", "probabilistic"):
            with self.subTest(v=v):
                m = PCA(3, variant=v)
                Z = m.fit_transform(self.X, self.y if v == "supervised_row_aware" else None)
                self.assertEqual(Z.shape, (100, 3))
                self.assertEqual(m.inverse_transform(Z).shape, self.X.shape)
                np.testing.assert_allclose(Z, PCA(3, variant=v).fit_transform(
                    self.X, self.y if v == "supervised_row_aware" else None))
                r = m.evaluate(self.X)
                self.assertEqual(set(r), REC_KEYS)
                self.assertTrue(all(np.isfinite(x) for x in r.values()))

    def test_error_decreases_with_components(self):
        for v in ("base", "robust", "probabilistic", "supervised_row_aware"):
            errs = [PCA(k, variant=v).fit(self.X).evaluate(self.X)["reconstruction_mse"] for k in (1, 3, 5)]
            self.assertTrue(errs[0] > errs[1] > errs[2], (v, errs))

    def test_base_matches_exact_full_rank(self):
        m = PCA(6).fit(self.X)
        self.assertLess(m.evaluate(self.X)["reconstruction_mse"], 1e-20)

    def test_sparse_is_sparser(self):
        dense = PCA(2).fit(self.X).components_
        sp = PCA(2, variant="sparse", alpha=0.5).fit(self.X).components_
        self.assertGreater(np.sum(sp == 0), np.sum(dense == 0))

    def test_robust_resists_outliers(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 4)) * np.array([5, 1, 1, 1.0])
        Xo = X.copy()
        Xo[:10] = rng.normal(size=(10, 4)) * 1 + np.array([0, 80, 0, 0])
        true = np.abs(PCA(1).fit(X).components_[0, 0])
        base = np.abs(PCA(1).fit(Xo).components_[0, 0])
        rob = np.abs(PCA(1, variant="robust").fit(Xo).components_[0, 0])
        self.assertGreater(rob, base)
        self.assertGreater(rob, 0.9 * true)

    def test_ppca_noise(self):
        m = PCA(2, variant="probabilistic").fit(self.X)
        self.assertGreater(m.noise_variance_, 0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            PCA(variant="nope")
        with self.assertRaises(NotFittedError):
            PCA().transform(self.X)
        with self.assertRaises(ValueError):
            PCA(2).fit(np.array([[1.0, np.nan], [2, 3]]))
        with self.assertRaises(ValueError):
            PCA(10).fit(self.X)
        with self.assertRaises(ValueError):
            PCA(2).fit(self.X).transform(self.X[:, :3])


class TestLDA(unittest.TestCase):
    def setUp(self):
        self.X, self.y = blobs(seed=1)
        self.Xt, self.yt = blobs(seed=2)

    def fit(self, v):
        kw = {"X_target": self.Xt + 0.3} if v == "shift_corrected" else {}
        return LDA(variant=v).fit(self.X, self.y, **kw)

    def test_variants(self):
        for v in ("base", "shift_corrected", "shrinkage", "sparse", "heteroscedastic"):
            with self.subTest(v=v):
                m = self.fit(v)
                self.assertEqual(m.predict(self.Xt).shape, (180,))
                P = m.predict_proba(self.Xt)
                np.testing.assert_allclose(P.sum(1), 1.0)
                self.assertEqual(m.transform(self.Xt).shape, (180, 2))
                np.testing.assert_array_equal(m.predict(self.Xt), self.fit(v).predict(self.Xt))
                r = m.evaluate(self.Xt, self.yt)
                check_cls(self, r)
                self.assertGreater(r["accuracy"], 0.9)

    def test_variant_specifics(self):
        self.assertGreater(self.fit("shrinkage").shrinkage_, 0)
        self.assertTrue(0 <= LDA(variant="shrinkage", shrinkage=0.2).fit(self.X, self.y).shrinkage_ <= 1)
        self.assertEqual(len(self.fit("heteroscedastic").class_covariances_), 3)
        Xn = np.hstack([self.X, np.random.default_rng(0).normal(size=(180, 5))])
        sp = LDA(variant="sparse", threshold=1.0).fit(Xn, self.y)
        self.assertLess(len(sp.selected_features_), 8)
        self.assertGreater(len(self.fit("shift_corrected").sample_weights_), 0)

    def test_shift_weights_favour_target_region(self):
        rng = np.random.default_rng(0)
        Xs = rng.normal(size=(300, 2)); ys = (Xs[:, 0] > 0).astype(int)
        Xt = rng.normal(loc=[1.5, 0], size=(300, 2))
        m = LDA(variant="shift_corrected").fit(Xs, ys, X_target=Xt)
        self.assertGreater(np.corrcoef(m.sample_weights_, Xs[:, 0])[0, 1], 0.5)

    def test_errors(self):
        with self.assertRaises(ValueError):
            LDA(variant="x")
        with self.assertRaises(ValueError):
            LDA(variant="shift_corrected").fit(self.X, self.y)
        with self.assertRaises(NotFittedError):
            LDA().predict(self.X)
        with self.assertRaises(ValueError):
            LDA().fit(self.X, self.y[:-1])
        with self.assertRaises(ValueError):
            LDA().fit(self.X, np.zeros(len(self.y)))


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.X, self.y = blobs(n=50, centers=((0, 0), (8, 8), (0, 8)), seed=3, noise=0.8)

    @staticmethod
    def purity(labels, y):
        return sum(np.bincount(y[labels == k]).max() for k in np.unique(labels)) / len(y)

    def test_variants(self):
        for v in ("base", "topology_aware", "stat_model_selection", "robust", "better_init"):
            with self.subTest(v=v):
                m = KMeans(3, variant=v).fit(self.X)
                self.assertEqual(m.predict(self.X).shape, (150,))
                np.testing.assert_array_equal(m.labels_, KMeans(3, variant=v).fit(self.X).labels_)
                r = m.evaluate(self.X)
                self.assertEqual(set(r), CLU_KEYS)
                self.assertTrue(all(np.isfinite(x) for x in r.values()))
                self.assertEqual(r["n_clusters_found"], 3)
                self.assertGreater(self.purity(m.predict(self.X), self.y), 0.97)
                self.assertEqual(m.transform(self.X).shape, (150, 3))

    def test_model_selection_criteria(self):
        for crit in ("bic", "silhouette"):
            m = KMeans(variant="stat_model_selection", criterion=crit).fit(self.X)
            self.assertEqual(m.n_clusters_, 3)
            self.assertIn(3, m.scores_)

    def test_robust_ignores_outliers(self):
        X = np.vstack([self.X, [[200.0, 200.0]] * 3])
        c = KMeans(3, variant="robust", trim=0.05).fit(X).cluster_centers_
        self.assertLess(np.abs(c).max(), 20)

    def test_errors(self):
        with self.assertRaises(ValueError):
            KMeans(variant="x")
        with self.assertRaises(NotFittedError):
            KMeans().predict(self.X)
        with self.assertRaises(ValueError):
            KMeans(500).fit(self.X)


class TestLinear(unittest.TestCase):
    def setUp(self):
        self.X, self.y = blobs(seed=4)

    def test_logistic_variants(self):
        for v in ("base", "risk_calibrated", "stat_regularised", "robust_loss"):
            with self.subTest(v=v):
                m = LogisticRegression(variant=v).fit(self.X, self.y)
                P = m.predict_proba(self.X)
                self.assertEqual(P.shape, (180, 3))
                np.testing.assert_allclose(P.sum(1), 1.0)
                np.testing.assert_allclose(
                    P, LogisticRegression(variant=v).fit(self.X, self.y).predict_proba(self.X))
                r = m.evaluate(self.X, self.y)
                check_cls(self, r)
                self.assertGreater(r["accuracy"], 0.9)

    def test_logistic_string_labels_and_specifics(self):
        lab = np.array(["a", "b", "c"])[self.y]
        m = LogisticRegression().fit(self.X, lab)
        self.assertEqual(set(m.predict(self.X)), {"a", "b", "c"})
        self.assertGreater(LogisticRegression(variant="risk_calibrated").fit(self.X, self.y).temperature_, 0)
        self.assertIn(1.0, LogisticRegression(variant="stat_regularised", alpha_grid=[1.0, 0.01])
                      .fit(self.X, self.y).cv_scores_ or {1.0: 0})
        rng = np.random.default_rng(0)
        y = self.y.copy()
        flip = rng.choice(len(y), 20, replace=False)
        y[flip] = (y[flip] + 1) % 3
        rob = LogisticRegression(variant="robust_loss").fit(self.X, y)
        self.assertLess(rob.sample_weights_[flip].mean(), rob.sample_weights_.mean())

    def test_ridge_variants(self):
        X, y = regression_data()
        for v in ("base", "risk_calibrated", "stat_regularised", "robust_loss"):
            with self.subTest(v=v):
                m = RidgeRegression(variant=v).fit(X, y)
                self.assertEqual(m.predict(X).shape, (150,))
                np.testing.assert_allclose(m.predict(X), RidgeRegression(variant=v).fit(X, y).predict(X))
                r = m.evaluate(X, y)
                self.assertEqual(set(r), REG_KEYS)
                self.assertTrue(all(np.isfinite(x) for x in r.values()))
                self.assertGreater(r["r2"], 0.95)

    def test_ridge_specifics(self):
        X, y = regression_data(outliers=True)
        Xc, yc = regression_data(seed=9)
        base = RidgeRegression().fit(X, y)
        rob = RidgeRegression(variant="robust_loss").fit(X, y)
        true = np.array([2.0, -1.0, 0.5, 0.0])
        self.assertLess(np.abs(rob.coef_ / rob.scaler_.scale_ - true).sum(),
                        np.abs(base.coef_ / base.scaler_.scale_ - true).sum())
        self.assertGreater(RidgeRegression(variant="stat_regularised").fit(Xc, yc).alpha_, 0)
        cal = RidgeRegression(variant="risk_calibrated").fit(Xc, yc)
        lo, hi = cal.predict_interval(Xc)
        self.assertTrue(np.all(hi >= lo))
        with self.assertRaises(ValueError):
            RidgeRegression().fit(Xc, yc).predict_interval(Xc)

    def test_errors(self):
        for cls in (LogisticRegression, RidgeRegression):
            with self.assertRaises(ValueError):
                cls(variant="x")
            with self.assertRaises(NotFittedError):
                cls().predict(self.X)
        with self.assertRaises(ValueError):
            LogisticRegression().fit(self.X, self.y[:-2])
        with self.assertRaises(ValueError):
            RidgeRegression().fit(self.X, np.full(len(self.y), np.inf))


class TestKNN(unittest.TestCase):
    def setUp(self):
        self.X, self.y = blobs(seed=5)

    def test_classifier_variants(self):
        Xt, yt = blobs(seed=6)
        for v in ("base", "prototype_compressed", "learnable_metric", "stat_k_selection"):
            with self.subTest(v=v):
                m = KNNClassifier(variant=v).fit(self.X, self.y)
                np.testing.assert_allclose(m.predict_proba(Xt).sum(1), 1.0)
                np.testing.assert_array_equal(
                    m.predict(Xt), KNNClassifier(variant=v).fit(self.X, self.y).predict(Xt))
                r = m.evaluate(Xt, yt)
                check_cls(self, r)
                self.assertGreater(r["accuracy"], 0.9)

    def test_regressor_variants(self):
        X, y = regression_data()
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
        for v in ("base", "prototype_compressed", "learnable_metric", "stat_k_selection"):
            with self.subTest(v=v):
                m = KNNRegressor(variant=v, compression=0.4).fit(X, y)
                self.assertEqual(m.predict(X).shape, (150,))
                np.testing.assert_allclose(m.predict(X), KNNRegressor(variant=v, compression=0.4).fit(X, y).predict(X))
                r = m.evaluate(X, y)
                self.assertEqual(set(r), REG_KEYS)
                self.assertTrue(all(np.isfinite(x) for x in r.values()))
                self.assertGreater(r["r2"], 0.5)

    def test_specifics(self):
        m = KNNClassifier(variant="prototype_compressed", compression=0.1).fit(self.X, self.y)
        self.assertLess(m.n_prototypes_, 40)
        Xn = np.hstack([self.X, np.random.default_rng(0).normal(size=(180, 6)) * 3])
        lm = KNNClassifier(variant="learnable_metric").fit(Xn, self.y)
        self.assertGreater(lm.metric_diag_[:3].mean(), lm.metric_diag_[3:].mean())
        ks = KNNClassifier(variant="stat_k_selection").fit(self.X, self.y)
        self.assertIn(ks.k_, ks.cv_scores_)

    def test_errors(self):
        for cls in (KNNClassifier, KNNRegressor):
            with self.assertRaises(ValueError):
                cls(variant="x")
            with self.assertRaises(NotFittedError):
                cls().predict(self.X)
        with self.assertRaises(ValueError):
            KNNClassifier(n_neighbors=0).fit(self.X, self.y)


class TestMLP(unittest.TestCase):
    def setUp(self):
        self.X, self.y = blobs(n=50, seed=7)
        self.kw = dict(hidden_layer_sizes=(16,), max_epochs=60, random_state=0)

    def test_classifier_variants(self):
        for v in ("base", "stat_regularised", "reliability_head", "uncertainty_head"):
            with self.subTest(v=v):
                kw = dict(self.kw, wd_grid=(1e-3, 1e-1)) if v == "stat_regularised" else self.kw
                m = MLPClassifier(variant=v, **kw).fit(self.X, self.y)
                P = m.predict_proba(self.X)
                self.assertEqual(P.shape, (150, 3))
                np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-9)
                np.testing.assert_allclose(P, MLPClassifier(variant=v, **kw).fit(self.X, self.y).predict_proba(self.X))
                r = m.evaluate(self.X, self.y)
                check_cls(self, r)
                self.assertGreater(r["accuracy"], 0.9)
                s = m.reliability_score(self.X)
                self.assertEqual(s.shape, (150,))
                self.assertTrue(np.all(np.isfinite(s)))
                if v == "reliability_head":
                    self.assertTrue(np.all((s >= 0) & (s <= 1)))
                if v == "uncertainty_head":
                    u = m.predict_uncertainty(self.X)
                    self.assertTrue(np.all((u >= 0) & (u <= 1 + 1e-9)))
                if v == "stat_regularised":
                    self.assertIn(m.weight_decay_, (1e-3, 1e-1))

    def test_regressor_variants(self):
        X, y = regression_data(n=120)
        y = y + 0.0
        for v in ("base", "stat_regularised", "reliability_head", "uncertainty_head"):
            with self.subTest(v=v):
                kw = dict(self.kw, wd_grid=(1e-3, 1e-1)) if v == "stat_regularised" else self.kw
                m = MLPRegressor(variant=v, **kw).fit(X, y)
                self.assertEqual(m.predict(X).shape, (120,))
                np.testing.assert_allclose(m.predict(X), MLPRegressor(variant=v, **kw).fit(X, y).predict(X))
                r = m.evaluate(X, y)
                self.assertEqual(set(r), REG_KEYS)
                self.assertTrue(all(np.isfinite(x) for x in r.values()))
                self.assertGreater(r["r2"], 0.8)
                if v == "uncertainty_head":
                    u = m.predict_uncertainty(X)
                    self.assertTrue(np.all(u > 0))
                    self.assertTrue(np.all(np.isfinite(m.reliability_score(X))))
                if v == "reliability_head":
                    self.assertEqual(m.predict_reliability(X).shape, (120,))

    def test_errors(self):
        for cls in (MLPClassifier, MLPRegressor):
            with self.assertRaises(ValueError):
                cls(variant="x")
            with self.assertRaises(NotFittedError):
                cls().predict(self.X)
        with self.assertRaises(ValueError):
            MLPClassifier(**self.kw).predict_uncertainty(self.X)
        m = MLPClassifier(**self.kw).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            m.predict_uncertainty(self.X)
        with self.assertRaises(ValueError):
            m.predict(self.X[:, :2])


class SkLike:
    """Duck-typed estimator with only fit/predict (no predict_proba, no classes_)."""

    def fit(self, X, y):
        self.c = {k: X[y == k].mean(0) for k in np.unique(y)}
        return self

    def predict(self, X):
        ks = list(self.c)
        d = np.array([((X - self.c[k]) ** 2).sum(1) for k in ks])
        return np.array(ks)[d.argmin(0)]


class TestMeta(unittest.TestCase):
    def setUp(self):
        self.X, self.y = blobs(n=50, seed=8, noise=1.5)
        self.Xt, self.yt = blobs(n=50, seed=9, noise=1.5)

    def bases(self):
        return [("lda", LDA()), ("lr", LogisticRegression()), ("knn", KNNClassifier()),
                ("mlp", MLPClassifier(hidden_layer_sizes=(8,), max_epochs=20)), ("nc", SkLike())]

    def test_all_variants(self):
        for v in ("performance_aware", "metrics_based", "oof_stacking", "statistical_meta", "calibrated_stack"):
            with self.subTest(v=v):
                m = MetaLearner(self.bases(), meta_variant=v, n_folds=3, random_state=0).fit(self.X, self.y)
                P = m.predict_proba(self.Xt)
                self.assertEqual(P.shape, (150, 3))
                np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-9)
                self.assertEqual(m.predict(self.Xt).shape, (150,))
                m2 = MetaLearner(self.bases(), meta_variant=v, n_folds=3, random_state=0).fit(self.X, self.y)
                np.testing.assert_allclose(P, m2.predict_proba(self.Xt))
                r = m.evaluate(self.Xt, self.yt)
                check_cls(self, r)
                self.assertGreater(r["accuracy"], 0.85)

    def test_leakage_free_oof(self):
        class Memoriser:
            """Perfectly memorises training rows; OOF predictions must still look honest."""

            def fit(self, X, y):
                self.X_, self.y_ = X.copy(), y.copy()
                return self

            def predict(self, X):
                out = []
                for row in X:
                    hit = np.flatnonzero((self.X_ == row).all(1))
                    out.append(self.y_[hit[0]] if len(hit) else 0)
                return np.array(out)

        rng = np.random.default_rng(0)
        X = rng.normal(size=(90, 3)); y = rng.integers(0, 3, 90)
        m = MetaLearner([("mem", Memoriser())], meta_variant="oof_stacking", n_folds=3).fit(X, y)
        self.assertLess(m.performance_scores_["mem"], 0.6)
        self.assertEqual(m.oof_proba_.shape, (1, 90, 3))

    def test_performance_weights_favour_good_model(self):
        class Bad:
            def fit(self, X, y):
                self.k = np.unique(y)
                return self

            def predict(self, X):
                return np.full(len(X), self.k[0])

        m = MetaLearner([("good", LDA()), ("bad", Bad())], meta_variant="performance_aware", n_folds=3)
        m.fit(self.X, self.y)
        self.assertGreater(m.weights_[0], m.weights_[1])
        self.assertEqual(m.best_estimator_, "good")

    def test_calibrated_temperature(self):
        m = MetaLearner(self.bases()[:2], meta_variant="calibrated_stack", n_folds=3).fit(self.X, self.y)
        self.assertGreater(m.temperature_, 0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            MetaLearner(self.bases(), meta_variant="x")
        with self.assertRaises(ValueError):
            MetaLearner([])
        with self.assertRaises(ValueError):
            MetaLearner([("a", LDA()), ("a", LDA())])
        with self.assertRaises(ValueError):
            MetaLearner(self.bases(), n_folds=1)
        with self.assertRaises(NotFittedError):
            MetaLearner(self.bases()).predict(self.X)


class TestMetrics(unittest.TestCase):
    def test_classification_metrics_known(self):
        r = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(r["accuracy"], 0.75)
        self.assertAlmostEqual(r["mcc"], 0.5773502691896258, places=6)
        self.assertEqual(classification_metrics([0, 1], [0, 1])["mcc"], 1.0)


if __name__ == "__main__":
    unittest.main()
