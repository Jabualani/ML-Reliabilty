"""Use cases 6-12: the Data Intelligence analysis toolkit (one question each).

  6.  Can the data predict it, or only explain it?   power_analysis
  7.  Do groups differ?                              compare_groups / scan_groups
  8.  Which page/service is better?                  ab_test
  9.  Why is the bad outcome happening?              root_cause
  10. What does the series do over time?             time_series_analysis
  11. Can I build better features?                   FeatureConstructor
  12. Which rows should I label / keep?              ActiveLearningStudy + your own row selector
  13. Several sections in one call (and in AutoML)   run_sections

All data below is simulated, so the answers are known. Nothing is written to disk.
Run:  python examples/02_data_intelligence_toolkit.py
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer  # dataset loader only

from ml_reliability import (
    ActiveLearningStudy,
    DataIntelligence,
    FeatureConstructor,
    ab_test,
    compare_groups,
    describe,
    evaluate_row_subset,
    power_analysis,
    plot_result,
    root_cause,
    run_sections,
    scan_groups,
    time_series_analysis,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- 6. predictive vs explanatory power
banner("6. POWER ANALYSIS: predictive vs explanatory vs statistical power")
X = pd.DataFrame(rng.normal(size=(500, 6)), columns=list("abcdef"))
y_signal = 2 * X["a"] - X["b"] + rng.normal(0, 0.5, 500)
y_noise = pd.Series(rng.normal(size=500))
print("--- target with signal");  print(describe(power_analysis(X, y_signal)))
print("--- pure-noise target");    print(describe(power_analysis(X, y_noise)))
small = pd.DataFrame(rng.normal(size=(60, 30)))
print("--- 60 rows, 30 columns, noise target (explains but does not predict)")
print(describe(power_analysis(small, rng.normal(size=60))))

# ---------------------------------------------------------------- 7. group comparison
banner("7. GROUP COMPARISON: group-by means + hypothesis tests")
shop = pd.DataFrame({"region": rng.choice(["north", "south", "east"], 600),
                     "channel": rng.choice(["web", "store"], 600)})
shop["basket"] = 50 + 8 * (shop["region"] == "south") + rng.normal(0, 12, 600)
print(describe(compare_groups(shop, value="basket", group="region")))
print("\nscan every categorical column against the target at once (Holm-corrected):")
print(describe(scan_groups(shop, target="basket")))

# ---------------------------------------------------------------- 8. A/B test
banner("8. A/B TEST: which variant generates more leads?")
ab = pd.DataFrame({"page": np.repeat(["A", "B"], 1000)})
ab["lead"] = np.r_[rng.random(1000) < 0.10, rng.random(1000) < 0.14].astype(int)
print(describe(ab_test(ab, group="page", outcome="lead")))

# ---------------------------------------------------------------- 9. root cause
banner("9. ROOT CAUSE: which segment concentrates the bad outcome?")
n = 2000
rc = pd.DataFrame({"plan": rng.choice(["basic", "plus", "pro"], n), "age": rng.integers(18, 80, n),
                   "tickets": rng.poisson(2, n), "region": rng.choice(["n", "s", "e", "w"], n)})
p = 0.08 + 0.45 * ((rc["plan"] == "basic") & (rc["tickets"] >= 4))
rc["churn"] = (rng.random(n) < p).astype(int)
print(describe(root_cause(rc, outcome="churn")))

# ---------------------------------------------------------------- 10. time series
banner("10. TIME SERIES: trend, seasonality, stationarity, change point")
t = pd.date_range("2025-01-01", periods=365, freq="D")
sales = 100 + 0.03 * np.arange(365) + 12 * np.sin(2 * np.pi * np.arange(365) / 7) + rng.normal(0, 3, 365)
sales[220:] += 25                                              # level shift at index 220
ts = pd.DataFrame({"date": t, "sales": sales})
res = time_series_analysis(ts, "date", "sales")
print(describe(res))
fig = plot_result(res)                                         # matplotlib figure(s), in memory
print("time-series figure(s) built:", type(fig).__name__)

# ---------------------------------------------------------------- 11. feature construction
banner("11. FEATURE CONSTRUCTION: iterative, distribution-aware, checked on untouched rows")
area = rng.lognormal(4, 0.5, 800)
price = rng.lognormal(5, 0.6, 800)
fc_df = pd.DataFrame({"price": price, "area": area, "noise": rng.normal(size=800)})
fc_y = np.log(price / area) + rng.normal(0, 0.1, 800)           # target depends on a ratio of two columns
fc = FeatureConstructor(random_state=0).fit(fc_df, fc_y)
print(describe(fc.report))
print("transform() applies the accepted recipe to new data:", fc.transform(fc_df.head(3)).shape)

# ---------------------------------------------------------------- 12. active learning + row selector
banner("12. ACTIVE LEARNING: which rows are worth labelling (and which strategy wins here?)")
bc = load_breast_cancer(as_frame=True)
Xb = bc.data.iloc[:, :10]
yb = bc.target.to_numpy()
study = ActiveLearningStudy(learner="knn", n_steps=5, n_repeats=2).run(Xb, yb)   # learner: logistic/knn/lda/mlp/own
print(describe(study))


def my_selector(X, n):
    """Plug in ANY row-selection method of your own: return n row indices."""
    return np.arange(0, len(X), max(1, len(X) // n))[:n]


print("\nhow much skill does a 100-row subset keep? (the helper holds out part of the data to score it)")
print(evaluate_row_subset(Xb, yb, my_selector(Xb.to_numpy(), 100)))

# ---------------------------------------------------------------- 13. several sections at once
banner("13. run_sections: pick sections, get one dict (same names work inside ReliableAutoML)")
frame = Xb.assign(target=yb)
out = run_sections(frame, "target", sections=("power", "group_scan", "construction"))
print("sections returned:", list(out))
print("equivalent on the engine: DataIntelligence().run_sections(frame, 'target', sections=('power',))")
