import os
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from ml_reliability import (
    ActiveLearningStudy, DataIntelligence, FeatureConstructor, ab_test, compare_groups, describe,
    evaluate_row_subset, plot_result, power_analysis, root_cause, run_sections, scan_groups,
    time_series_analysis,
)


def _reg(n=400, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({"a": r.normal(size=n), "b": r.normal(size=n), "noise": r.normal(size=n),
                      "grp": r.choice(["u", "v", "w"], n)})
    y = 2 * X.a - X.b + 0.5 * r.normal(size=n)
    return X, y.to_numpy()


def _clf(n=500, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, 4)), columns=list("abcd"))
    y = ((X.a + 0.8 * X.b + 0.4 * r.normal(size=n)) > 0).astype(int).to_numpy()
    return X, y


class PowerTests(unittest.TestCase):
    def test_signal_vs_noise(self):
        X, y = _reg()
        res = power_analysis(X, y)
        self.assertGreater(res["predictive"]["ridge_cv_skill"], 0.8)
        self.assertEqual(res["explanatory"]["features"][0]["feature"], "a")
        noise = power_analysis(X, np.random.default_rng(1).normal(size=len(y)))
        self.assertLess(noise["predictive"]["ridge_cv_skill"], 0.05)
        self.assertIn("weak", noise["verdict"])

    def test_statistical_power_monotone(self):
        X, y = _reg()
        s = power_analysis(X, y)["statistical_power"]
        self.assertLess(s["required_n"]["large_f2_0.35"], s["required_n"]["small_f2_0.02"])
        self.assertGreater(s["power_at_observed_effect"], 0.99)

    def test_classification(self):
        X, y = _clf()
        res = power_analysis(X, y)
        self.assertEqual(res["task"], "classification")
        self.assertGreater(res["predictive"]["ridge_cv_skill"], 0.4)


class StatisticsToolkitTests(unittest.TestCase):
    def test_compare_two_and_many_groups(self):
        r = np.random.default_rng(0)
        df = pd.DataFrame({"g": np.repeat(["x", "y"], 150), "v": np.r_[r.normal(0, 1, 150), r.normal(1, 1, 150)]})
        res = compare_groups(df, "v", "g")
        self.assertTrue(res["significant"])
        self.assertGreater(abs(res["omnibus"]["cohens_d"]), 0.6)
        df["h"] = np.tile(["p", "q", "r"], 100)
        res3 = compare_groups(df, "v", "h")
        self.assertFalse(res3["significant"])
        self.assertEqual(len(res3["pairwise"]), 3)

    def test_scan_groups_finds_planted_column(self):
        X, y = _reg()
        X["target"] = y + np.where(X.grp == "u", 3.0, 0.0)
        res = scan_groups(X, "target")
        self.assertEqual(res["columns"][0]["column"], "grp")
        self.assertTrue(res["columns"][0]["significant"])

    def test_ab_test_binary_and_numeric(self):
        r = np.random.default_rng(0)
        df = pd.DataFrame({"arm": np.repeat(["A", "B"], 2000)})
        df["lead"] = np.r_[r.random(2000) < 0.10, r.random(2000) < 0.14].astype(int)
        res = ab_test(df, "arm", "lead")
        self.assertTrue(res["significant"])
        self.assertGreater(res["absolute_lift"], 0)
        self.assertIn("better", res["decision"])
        same = ab_test(df.assign(lead=r.integers(0, 2, len(df))), "arm", "lead")
        self.assertFalse(same["significant"])
        df["value"] = r.normal(10, 2, len(df))
        self.assertIn("p_mann_whitney", ab_test(df, "arm", "value"))

    def test_ab_sample_ratio_warning(self):
        r = np.random.default_rng(0)
        df = pd.DataFrame({"arm": ["A"] * 1500 + ["B"] * 500, "y": r.integers(0, 2, 2000)})
        self.assertTrue(any("sample-ratio" in w for w in ab_test(df, "arm", "y")["warnings"]))

    def test_root_cause_finds_segment(self):
        r = np.random.default_rng(0)
        n = 3000
        df = pd.DataFrame({"region": r.choice(["n", "s", "e", "w"], n), "age": r.integers(18, 80, n),
                           "plan": r.choice(["a", "b"], n)})
        p = np.where((df.region == "s") & (df.age > 60), 0.6, 0.05)
        df["churn"] = (r.random(n) < p).astype(int)
        res = root_cause(df, "churn")
        top = res["segments"][0]["segment"]
        self.assertIn("region == s", top)
        self.assertGreater(res["segments"][0]["lift"], 2)

    def test_time_series(self):
        r = np.random.default_rng(0)
        n = 240
        t = pd.date_range("2020-01-01", periods=n, freq="D")
        v = 0.05 * np.arange(n) + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + r.normal(0, 0.5, n)
        v[150:] += 4
        res = time_series_analysis(pd.DataFrame({"t": t, "v": v}), "t", "v")
        self.assertTrue(res["trend"]["significant"])
        self.assertEqual(res["seasonality"]["dominant_period"], 7)
        self.assertGreater(res["decomposition"]["seasonal_strength"], 0.5)
        self.assertTrue(res["change_point"]["significant"])
        self.assertLess(abs(res["change_point"]["index"] - 150), 30)
        self.assertAlmostEqual(res["trend"]["sen_slope_per_step"], 0.05, delta=0.03)


class ConstructionTests(unittest.TestCase):
    def _planted(self, n=600, seed=0):
        r = np.random.default_rng(seed)
        X = pd.DataFrame({"price": r.lognormal(3, 0.6, n), "area": r.lognormal(4, 0.5, n), "age": r.normal(size=n)})
        y = np.log(X.price / X.area) * 3 + 0.2 * r.normal(size=n)
        return X, y.to_numpy()

    def test_discovers_planted_ratio_and_holds_out(self):
        X, y = self._planted()
        fc = FeatureConstructor(random_state=0).fit(X, y)
        rep = fc.report
        self.assertTrue(rep["recipes"], rep)
        self.assertGreater(rep["holdout_gain"], 0.02)
        out = fc.transform(X.iloc[:20])
        self.assertEqual(len(out), 20)
        self.assertGreater(out.shape[1], X.shape[1])

    def test_pure_noise_adds_nothing(self):
        r = np.random.default_rng(3)
        X = pd.DataFrame(r.normal(size=(500, 4)), columns=list("abcd"))
        rep = FeatureConstructor(random_state=0).fit(X, r.normal(size=500)).report
        self.assertEqual(rep["recipes"], [])

    def test_leaky_column_is_never_a_base(self):
        X, y = self._planted()
        X["copy"] = y
        rep = FeatureConstructor().fit(X, y).report
        self.assertTrue(all("copy" not in r["inputs"] for r in rep["recipes"]))


class ActiveLearningTests(unittest.TestCase):
    def test_curves_and_ranking(self):
        X, y = _clf()
        rep = ActiveLearningStudy(strategies=["random", "margin", "kcenter", "coverage_margin", "diversity_static"],
                                  n_steps=4, n_repeats=2).run(X, y)
        self.assertEqual(set(rep["strategies"]), {"random", "margin", "kcenter", "coverage_margin", "diversity_static"})
        for st in rep["strategies"].values():
            self.assertEqual(len(st["mean_skill"]), len(rep["label_sizes"]))
        self.assertGreater(rep["ceiling_skill"], 0.5)
        self.assertIn("recommendation", rep)

    def test_all_strategies_and_learners_run(self):
        X, y = _clf(300)
        for learner in ("logistic", "knn", "lda"):
            rep = ActiveLearningStudy(learner=learner, n_steps=2, n_repeats=1).run(X, y)
            self.assertEqual(len(rep["strategies"]), len(ActiveLearningStudy.STRATEGIES))

    def test_row_selector_hook_and_subset_evaluation(self):
        X, y = _clf(300)
        calls = []

        def selector(M, n):
            calls.append(n)
            return np.arange(0, len(M), max(1, len(M) // n))[:n]

        rep = ActiveLearningStudy(strategies=["random"], row_selector=selector, n_steps=2, n_repeats=1).run(X, y)
        self.assertIn("row_selector", rep["strategies"])
        self.assertTrue(calls)
        ev = evaluate_row_subset(X, y, np.arange(0, 300, 3))
        self.assertGreater(ev["skill_retained"], 0.5)
        self.assertLess(ev["ks_mean"], 0.5)

    def test_bad_selector_and_strategy_rejected(self):
        X, y = _clf(200)
        with self.assertRaises(ValueError):
            ActiveLearningStudy(strategies=["nope"])
        with self.assertRaises(ValueError):
            ActiveLearningStudy(strategies=["random"], row_selector=lambda M, n: [10 ** 9], n_steps=1,
                                n_repeats=1).run(X, y)


class SectionsAndFacadeTests(unittest.TestCase):
    def test_run_sections_and_describe(self):
        X, y = _reg()
        df = X.assign(target=y)
        di = DataIntelligence()
        res = di.run_sections(df, "target", sections=("power", "group_scan", "construction"))
        self.assertEqual(set(res), {"power", "group_scan", "construction"})
        self.assertTrue(all("error" not in v for v in res.values()), res)
        self.assertIn("Verdict", describe(res))
        self.assertIsNotNone(plot_result(res["power"]))

    def test_errors_are_captured_not_raised(self):
        X, y = _reg()
        res = run_sections(X.assign(target=y), "target", ("time_series",))
        self.assertIn("error", res["time_series"])
        with self.assertRaises(ValueError):
            run_sections(X.assign(target=y), "target", ("bogus",))

    def test_nothing_written_by_default_and_export_opt_in(self):
        X, y = _reg(200)
        df = X.assign(target=y)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                DataIntelligence().run_sections(df, "target", ("power",))
                self.assertEqual(os.listdir(d), [])
                out = os.path.join(d, "exp")
                DataIntelligence(export_dir=out, export_formats=("json",)).run_sections(df, "target", ("power",))
                self.assertTrue(os.path.exists(os.path.join(out, "toolkit.json")))
            finally:
                os.chdir(cwd)

    def test_every_figure_renders(self):
        r = np.random.default_rng(0)
        X, y = _clf(300)
        t = pd.date_range("2021-01-01", periods=120)
        results = [
            power_analysis(X, y),
            compare_groups(pd.DataFrame({"g": np.repeat(["a", "b"], 60), "v": r.normal(size=120)}), "v", "g"),
            ab_test(pd.DataFrame({"g": np.repeat(["a", "b"], 100), "y": r.integers(0, 2, 200)}), "g", "y"),
            time_series_analysis(pd.DataFrame({"t": t, "v": r.normal(size=120).cumsum()}), "t", "v"),
            FeatureConstructor().fit(X, y).report,
            ActiveLearningStudy(strategies=["random", "margin"], n_steps=2, n_repeats=1).run(X, y),
        ]
        for res in results:
            self.assertIsNotNone(plot_result(res), res["kind"])


if __name__ == "__main__":
    unittest.main()
