import unittest

import pandas as pd

from ml_reliability import StatisticalDataCleaner, clean_statistical_data


class StatisticalDataCleanerTests(unittest.TestCase):
    def test_light_cleaner_imputes_without_clipping_outliers(self):
        data = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, None, 3.0, 100.0],
                "category": ["a", None, "a", "b", "a"],
            }
        )
        cleaner = StatisticalDataCleaner().fit(data)
        cleaned = cleaner.transform(data)

        self.assertFalse(cleaned.isna().any().any())
        self.assertEqual(cleaned.loc[4, "numeric"], 100.0)
        self.assertTrue(
            cleaner.get_cleaning_report()["numeric_outliers_detected"]["numeric"]
        )
        self.assertFalse(cleaner.get_cleaning_report()["outlier_capping_applied"])
        self.assertEqual(cleaned.columns.tolist(), data.columns.tolist())

    def test_moderate_cleaning_caps_outliers_and_function_defaults_to_light(self):
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
        light, cleaner = clean_statistical_data(data, return_cleaner=True)
        moderate = StatisticalDataCleaner("moderate").fit_transform(data)

        self.assertEqual(light.loc[4, "x"], 100.0)
        self.assertLess(moderate.loc[4, "x"], 100.0)
        self.assertEqual(cleaner.cleaning_intensity, "light")
