import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml_reliability import (
    DataIntelligence,
    FeatureIntelligence,
    TargetIntelligence,
    multicollinearity_report,
)


def make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=(3, n))
    y = (a + 0.8 * b + 0.3 * rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({
        "a": a,
        "b": b,
        "c_noise": c,
        "a_copy": a * 2 + 0.01 * rng.normal(size=n),
        "leak": y + 0.01 * rng.normal(size=n),
        "const": 1.0,
        "group": rng.choice(["x", "y", "z"], n),
    }), y


class MulticollinearityTests(unittest.TestCase):
    def test_flags_redundant_pair_and_vif_drop(self):
        X, _ = make_data()
        rep = multicollinearity_report(X)
        pairs = {frozenset((p["a"], p["b"])) for p in rep["redundant_pairs"]}
        self.assertIn(frozenset(("a", "a_copy")), pairs)
        self.assertTrue(set(rep["recommended_drop"]) & {"a", "a_copy"})
        self.assertIn("const", rep["constant_columns"])

    def test_never_reads_target(self):
        X, _ = make_data()
        rep = TargetIntelligence(scope="multicollinearity").fit(X).report
        self.assertEqual(rep["scope"], "multicollinearity")
        self.assertNotIn("leakage", rep.to_dict())


class TargetIntelligenceTests(unittest.TestCase):
    def test_detects_leak_and_explainable_target(self):
        X, y = make_data()
        ti = TargetIntelligence(n_permutations=10, random_state=0).fit(X, y, target_name="outcome")
        flagged = {f["feature"] for f in ti.report["leakage"]["flags"]}
        self.assertIn("leak", flagged)
        self.assertFalse(ti.report["explainability"]["verdict"].startswith("not distinguishable"))
        self.assertLess(ti.report["summary_score"], 100)
        self.assertIn("TARGET INTELLIGENCE", ti.report.to_text())
        self.assertIsInstance(ti.report.to_json(), str)

    def test_noise_target_is_not_explainable(self):
        X, _ = make_data()
        rng = np.random.default_rng(5)
        X = X.drop(columns=["leak"])
        ti = TargetIntelligence(n_permutations=10).fit(X, rng.integers(0, 2, len(X)))
        self.assertTrue(ti.report["explainability"]["verdict"].startswith("not distinguishable"))

    def test_regression_and_transform(self):
        X, _ = make_data()
        y = 3 * X["a"] + X["b"] + 0.1 * np.random.default_rng(1).normal(size=len(X))
        ti = TargetIntelligence(n_permutations=5).fit(X.drop(columns="leak"), y.to_numpy())
        self.assertEqual(ti.report["target_profile"]["task"], "regression")
        self.assertIn("a", ti.best_features_ + ["a_copy"])
        self.assertTrue(set(ti.transform(X).columns) <= set(X.columns))

    def test_requires_y_for_full_scope(self):
        X, _ = make_data()
        with self.assertRaises(ValueError):
            TargetIntelligence().fit(X)

    def test_automl_hook_has_no_target_sections(self):
        X, _ = make_data()
        hook = TargetIntelligence().automl_hook(X)
        self.assertEqual(set(hook), {"multicollinearity_report", "summary_score"})


class FeatureIntelligenceTests(unittest.TestCase):
    def test_unsupervised_pruning(self):
        X, _ = make_data()
        fi = FeatureIntelligence().fit(X.drop(columns="leak"))
        rep = fi.report
        self.assertIn("const", rep["hazards"])
        self.assertTrue({"a", "a_copy"} & set(rep["correlation_dropped"]))
        self.assertFalse(rep["supervised"])

    def test_supervised_ranking_prefers_signal(self):
        X, y = make_data()
        fi = FeatureIntelligence(n_features=2).fit(X.drop(columns=["leak", "a_copy"]), y)
        self.assertEqual(len(fi.selected_features_), 2)
        self.assertNotIn("c_noise", fi.selected_features_)
        self.assertEqual(fi.transform(X).shape[1], 2)

    def test_bad_method(self):
        with self.assertRaises(ValueError):
            FeatureIntelligence(methods=("nope",))


class DataIntelligenceTests(unittest.TestCase):
    def test_all_modes_run(self):
        X, y = make_data()
        df = X.assign(outcome=y)
        df["when"] = pd.date_range("2024-01-01", periods=len(df), freq="D")
        df["trend"] = np.arange(len(df)) * 0.5 + np.random.default_rng(2).normal(size=len(df))
        for mode in ("comprehensive_eda", "target_intelligence", "feature_intelligence", "field_investigation",
                     "auto_question_generator", "llm_augmented", "autonomous"):
            with self.subTest(mode=mode):
                rep = DataIntelligence(mode=mode).analyze(df, target_column="outcome", time_column="when")
                self.assertTrue(0 <= rep.summary_score <= 100)
                self.assertIn("recommendations", rep.to_dict())
                rep.to_text()

    def test_eda_without_target_and_with_plots(self):
        X, _ = make_data()
        X.loc[::10, "b"] = np.nan
        with tempfile.TemporaryDirectory() as d:
            rep = DataIntelligence(plot_dir=d).analyze(X)
            self.assertTrue(rep["visualizations"])
            self.assertTrue(all(os.path.exists(p) for k, p in rep["visualizations"].items() if k != "error"))
        self.assertIn("const", rep["what_happened"]["hazards"]["constant"])
        self.assertTrue(rep["questions"])

    def test_trend_detected(self):
        df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=120), "v": np.arange(120) * 1.0,
                           "w": np.random.default_rng(0).normal(size=120)})
        rep = DataIntelligence().analyze(df, time_column="t")
        self.assertTrue(rep["what_then"]["series"]["v"]["significant_trend"])

    def test_llm_backend_callable_and_failure(self):
        X, _ = make_data()
        rep = DataIntelligence(mode="llm_augmented", llm_backend=lambda p: "summary").analyze(X)
        self.assertEqual(rep["llm_synthesis"], "summary")
        bad = DataIntelligence(mode="llm_augmented", llm_backend="custom_url").analyze(X)
        self.assertIn("failed", bad["llm_status"])

    def test_export_formats_and_html_embeds_figures(self):
        X, y = make_data()
        df = X.assign(outcome=y)
        with tempfile.TemporaryDirectory() as d:
            di = DataIntelligence(mode="autonomous", export_dir=os.path.join(d, "out"))
            di.analyze(df, target_column="outcome")
            for name in ("data_intelligence.json", "data_intelligence.txt", "data_intelligence.html"):
                self.assertTrue(os.path.exists(os.path.join(d, "out", name)), name)
            with open(os.path.join(d, "out", "data_intelligence.html"), encoding="utf-8") as handle:
                self.assertIn("data:image/png;base64", handle.read())
            written = di.export(os.path.join(d, "json_only"), formats="json", include_figures=False)
            self.assertEqual(set(written) - {"figures"}, {"json"})
            with self.assertRaises(ValueError):
                di.report_.export(d, formats=("pdf",))
            ti = TargetIntelligence(scope="multicollinearity").fit(X)
            self.assertTrue(os.path.exists(ti.report.export(d)["json"]))

    def test_input_validation(self):
        X, _ = make_data()
        with self.assertRaises(ValueError):
            DataIntelligence(mode="nope")
        with self.assertRaises(ValueError):
            DataIntelligence(mode="target_intelligence").analyze(X)
        with self.assertRaises(KeyError):
            DataIntelligence().analyze(X, target_column="missing")


class OutlierDetectorTests(unittest.TestCase):
    @staticmethod
    def _blob(n=400, seed=1):
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=(n, 4))
        Z[:5] += 9.0
        return Z

    def test_detectors_rank_planted_outliers_first(self):
        from ml_reliability import SelfOrganisingMap, isolation_forest_scores, local_outlier_factor, robust_mahalanobis

        Z = self._blob()
        iso = isolation_forest_scores(Z)
        lof = local_outlier_factor(Z, k=15)
        d2, cut, _ = robust_mahalanobis(Z)
        for name, score in (("iso", iso), ("lof", lof), ("maha", d2)):
            self.assertEqual(set(np.argsort(-score)[:5]), set(range(5)), name)
        self.assertTrue(0 < iso.min() and iso.max() < 1)
        self.assertGreater(lof[:5].min(), 1.5)
        som = SelfOrganisingMap(random_state=0).fit(Z)
        self.assertEqual(som.hits_.sum(), len(Z))
        self.assertEqual(som.u_matrix().shape, som.shape_)
        self.assertTrue(set(np.argsort(-som.quantization_error_row_)[:5]) >= {0, 1, 2, 3} or
                        som.quantization_error_row_[:5].mean() > som.quantization_error_row_[5:].mean())
        self.assertTrue(0.0 <= som.topographic_error_ <= 1.0)

    def test_ensemble_consensus_figures_and_no_default_writes(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
        df.iloc[:6] += 8.0
        with tempfile.TemporaryDirectory() as d:
            before = os.getcwd()
            os.chdir(d)
            try:
                di = DataIntelligence(mode="comprehensive_eda")
                rep = di.analyze(df)
                figs = di.plot()["figures"]
                self.assertEqual(os.listdir(d), [])
            finally:
                os.chdir(before)
        ens = rep["why"]["outliers"]["ensemble"]
        self.assertTrue(set(range(6)) <= set(ens["consensus"]["row_positions"]))
        self.assertIn("outlier_consensus", figs)
        self.assertIn("som", figs)
        self.assertIsNone(rep.details.get("visualizations"))


class QuestionDiscoveryTests(unittest.TestCase):
    """Someone with data but no question must get questions, candidate targets and next steps."""

    def _frame(self):
        rng = np.random.default_rng(5)
        n = 300
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        df = pd.DataFrame({"a": a, "b": b, "c": a + 0.5 * b + 0.2 * rng.normal(size=n),
                           "a_x10": 10 * a, "noise": rng.normal(size=n),
                           "group": rng.choice(["x", "y"], n)})
        df.loc[df["group"] == "x", "b"] += 3
        df.loc[rng.random(n) < 0.1, "noise"] = np.nan
        return df

    def test_no_target_gives_prioritised_plan_and_candidate_targets(self):
        report = DataIntelligence(mode="comprehensive_eda", random_state=0).analyze(self._frame())
        plan = report["question_plan"]
        self.assertTrue(plan)
        self.assertTrue(all({"category", "priority", "question", "evidence", "next_step"} <= set(p) for p in plan))
        self.assertEqual(plan[0]["category"], "target")
        cands = {c["column"]: c for c in report["candidate_targets"]}
        self.assertGreater(cands["c"]["skill"], 0.5)
        self.assertLess(cands["noise"]["skill"], 0.1)
        self.assertEqual(cands["a"]["derived_from"], "a_x10")
        self.assertIn("Questions the data raises", report.to_text())
        self.assertIn("categories", {p["category"] for p in plan})
        self.assertIn("missingness", {p["category"] for p in plan})

    def test_user_question_replaces_questions_but_plan_is_kept(self):
        report = DataIntelligence(mode="comprehensive_eda").analyze(self._frame(), user_question="why is c high?")
        self.assertEqual(report["questions"], ["why is c high?"])
        self.assertTrue(report["question_plan"])

    def test_with_target_the_plan_asks_about_the_target_and_has_no_candidates(self):
        report = DataIntelligence(mode="comprehensive_eda").analyze(self._frame(), target_column="c")
        self.assertNotIn("candidate_targets", report.details)
        self.assertEqual(report["question_plan"][0]["category"], "target")
        tiny = DataIntelligence().auto_generate_data_questions(pd.DataFrame({"a": [1, 2, 3], "b": [3, 1, 2]}))
        self.assertTrue(tiny)


if __name__ == "__main__":
    unittest.main()
