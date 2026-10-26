import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from ml_reliability import compare_methods, plot_critical_difference, summarize_repeats


def _scores():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.6, 0.9, size=12)
    return pd.DataFrame(
        {
            "A": base + 0.05 + rng.normal(0, 0.005, 12),
            "B": base + rng.normal(0, 0.005, 12),
            "C": base - 0.05 + rng.normal(0, 0.005, 12),
        },
        index=[f"d{i}" for i in range(12)],
    )


class BenchmarkTests(unittest.TestCase):
    def test_ranking_and_significance(self):
        report = compare_methods(_scores())
        self.assertEqual(report.average_ranks["A"], 1.0)
        self.assertEqual(report.average_ranks["C"], 3.0)
        self.assertLess(report.friedman["p_value"], 0.001)
        self.assertGreater(report.nemenyi["critical_difference"], 0)
        top = [r for r in report.pairwise if (r["method_a"], r["method_b"]) == ("A", "C")][0]
        self.assertEqual(top["wins_a"], 12)
        self.assertEqual(top["cliffs_delta"], 1.0)
        self.assertTrue(top["significant_after_holm"])
        self.assertIn("interpretation", report.to_dict())

    def test_lower_is_better_flips_ranks(self):
        report = compare_methods(_scores(), higher_is_better=False)
        self.assertEqual(report.average_ranks["C"], 1.0)

    def test_two_methods_skip_friedman(self):
        report = compare_methods(_scores()[["A", "B"]])
        self.assertEqual(report.friedman["status"], "unavailable")
        self.assertEqual(len(report.pairwise), 1)

    def test_validation(self):
        data = _scores()
        data.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            compare_methods(data)
        with self.assertRaises(ValueError):
            compare_methods(_scores()[["A"]])
        with self.assertRaises(ValueError):
            compare_methods(_scores(), alpha=2)

    def test_small_sample_warning(self):
        self.assertTrue(compare_methods(_scores().head(5)).warnings)

    def test_summarize_repeats(self):
        long = pd.DataFrame(
            {
                "dataset": ["x", "x", "y", "y"],
                "method": ["A", "A", "A", "A"],
                "score": [0.8, 0.6, 1.0, 0.0],
            }
        )
        table = summarize_repeats(long)
        self.assertAlmostEqual(table.loc["x", "A"], 0.7)
        self.assertAlmostEqual(table.loc["y", "A"], 0.5)

    def test_plot(self):
        figure, _ = plot_critical_difference(compare_methods(_scores()))
        self.assertIsNotNone(figure)


if __name__ == "__main__":
    unittest.main()
