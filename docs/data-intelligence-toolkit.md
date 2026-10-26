# Data Intelligence toolkit: usage guide and applications

This guide covers the analysis toolkit in `ml_reliability.data_intelligence`: **power analysis**,
**group comparison and hypothesis tests**, **A/B testing**, **root-cause search**, **time-series analysis**,
**feature construction** and **active learning**. Everything is NumPy/StatCore plus classical statistics
(scipy, statsmodels); no scikit-learn models. Nothing is written to disk unless you pass `export_dir=` or
call `fig.savefig(...)` yourself.

```python
from ml_reliability import (DataIntelligence, power_analysis, compare_groups, scan_groups, ab_test,
                            root_cause, time_series_analysis, FeatureConstructor, ActiveLearningStudy,
                            evaluate_row_subset, run_sections, describe, plot_result)
```

Every function returns a plain `dict` with a `"kind"` and a list of plain-language `"findings"`.
`describe(result)` prints the findings, `plot_result(result)` returns a matplotlib figure (or `None`).

## Which tool answers which question?

| Your situation | Tool | Question it answers |
|---|---|---|
| Have data, no question | `DataIntelligence().analyze(df)` then `question_plan` | What is worth asking? |
| Have a target, unsure it is learnable | `power_analysis` | Does X explain it, does X predict it, is n big enough? |
| "Why is the number different?" | `compare_groups`, `scan_groups` | Which groups differ, by how much, is it real? |
| Two variants (pages, prices, services) | `ab_test` | Which is better, how sure, how much data is needed? |
| "What is driving the bad outcome?" | `root_cause` | Which features and segments concentrate it? |
| Dated data | `time_series_analysis` | Trend, seasonality, level shifts, stationarity |
| Raw features are weak | `FeatureConstructor` | Which new columns help, and does it hold on unseen rows? |
| Labels are expensive | `ActiveLearningStudy`, `evaluate_row_subset` | Which rows to label or keep first? |
| One call, several of the above | `DataIntelligence().run_sections(...)` | Selected sections in one dict |

## 1. Power analysis (predictive vs explanatory vs statistical)

"Power" has three meanings; the report separates them.

* **Predictive power**: out-of-fold skill (ridge and KNN). Skill is R² for regression and chance-adjusted
  balanced accuracy for classification (0 = chance, 1 = perfect).
* **Explanatory power**: in-sample variance accounted for, with an F test, per-feature p-values
  (Benjamini-Hochberg adjusted) and, for regression, each feature's partial R².
* **Statistical power**: can the F test detect an effect of Cohen's f² at your n? Reports the minimum
  detectable f² at 80 % power and the n needed for small/medium/large effects.

```python
res = power_analysis(df.drop(columns="price"), df["price"].to_numpy())
print(describe(res))
print(res["verdict"], res["gap_explanatory_minus_predictive"])
plot_result(res).savefig("power.png")      # only if you want it on disk
```

How to read the verdict:

| Verdict | Meaning | What to do |
|---|---|---|
| explains and predicts consistently | simple, stable signal | a simple model may be enough |
| predicts better non-linearly | KNN beats ridge by > 0.05 | try tree/NN/StatCore MLP models |
| large explanatory-minus-predictive gap | in-sample story is partly noise | trust only cross-validated numbers |
| explains but does not predict | classic overfitting (small n, many columns) | collect data or cut features |
| weak on both | no usable signal | revisit the target or the data |

Measured (see `docs/latex/experiments/toolkit-results.json`): with a real signal ridge CV skill 0.956 and
in-sample 0.957; pure noise gives CV -0.02 and verdict "weak on both"; n = 60 with 30 columns and a noise
target has in-sample 0.42 but CV -1.32 (verdict "explains but does not predict").

## 2. Comparative analysis, hypothesis testing

`compare_groups(df, value, group)` picks Welch t or Mann-Whitney (two groups) or ANOVA or Kruskal (three or more)
from a normality check, adds Cohen's d / eta², CIs per group and Holm-corrected pairwise tests.
A categorical `value` uses a chi-square test with Cramér's V.
`scan_groups(df, target)` runs this for every categorical column and ranks them (BH corrected):
"which attribute separates the outcome?".

```python
r = compare_groups(df, value="basket_value", group="channel")
print(describe(r)); print(r["pairwise"][0])
print(scan_groups(df, "basket_value")["columns"][:3])
```

## 3. A/B testing

```python
r = ab_test(df, group="page", outcome="lead")            # lead is 0/1 or numeric
print(r["decision"], r["relative_lift"], r["ci95_difference"])
print(r["min_detectable_absolute_lift"], r["n_per_arm_for_observed_lift"], r["warnings"])
```

Binary outcomes: two-proportion z, chi-square and Fisher (used automatically when expected counts < 5), CI
for the lift. Numeric outcomes: Welch t, Mann-Whitney, Cohen's d. Also reports a **sample-ratio
mismatch** warning (arms unevenly sized), the smallest lift the experiment could detect and the sample
size the observed lift would need. Decide the sample size first; peeking and stopping on the first
small p-value inflates false positives. Measured over 200 simulated null experiments (1 000 per arm):
false-positive rate 0.055; power to detect 10 % to 14 % conversion: 0.795.

## 4. Root-cause search

```python
r = root_cause(df, outcome="churn")                        # or bad="late", or direction="low" for numerics
for s in r["segments"][:3]:
    print(s["segment"], s["bad_rate"], s["lift"], s["support"], s["p_adjusted_bh"])
```

Finds single features and depth-2 segments (e.g. `region == s AND age in tercile 3/3`) whose bad-outcome
rate exceeds the base rate (one-sided binomial test, BH corrected, minimum support). It is
**associational**: confirm a cause with an experiment or domain knowledge. In 20 simulated datasets with a
planted segment it ranked that segment first 20 times.

## 4b. Time series

```python
r = time_series_analysis(df, time_column="date", value_column="sales")
print(describe(r)); plot_result(r).savefig("ts.png")
```

Mann-Kendall trend with Sen slope, periodogram seasonality with Fisher's g test, classical decomposition
(seasonal and trend strength), ADF + KPSS stationarity, Ljung-Box on the residuals and a CUSUM change point
with a permutation p-value. Pass `period=7` to force the seasonal period. Caveats: the Sen slope is biased
when there is a level shift (measured 0.041 vs true 0.030 with a +3 shift), and the change-point p-value
ignores autocorrelation, so treat both as leads.

## 5. Feature construction (iterative, distribution-aware)

```python
fc = FeatureConstructor(random_state=0).fit(X_train, y_train)
print(describe(fc.report)); X_new = fc.transform(X_test)
```

1. Candidates come from each column's *empirical distribution*: log / signed-log for skew, rank-Gauss for heavy
   tails, density-valley thresholds, plus products, differences, sums and (sign-safe) ratios of the strongest columns.
2. Every candidate is *screened* on one resampling, then the best few are *confirmed* on several fresh fold
   assignments. A candidate is kept only if `mean gain - std of gain > min_gain`.
3. The kept feature joins the pool, so later rounds can build on it (`max_depth` caps this).
4. 25 % of rows are never touched by the search; `holdout_gain` reports whether the gain survived.
   Columns that almost equal the target (|corr| >= 0.95) are never used as inputs.

All constructions are target-free (thresholds and ranks come from X); only the *selection* uses y.
Measured over 8 seeds: planted log-ratio target, holdout gain +0.141 (found `log1p(price)` and `log1p(area)`,
whose difference is the log-ratio); planted step on a heavy-tailed column, +0.399; pure-noise target:
2 of 8 runs accepted a feature but the holdout gain averaged -0.0009. So **read `holdout_gain`**: it is
what catches selection optimism.

## 6. Active learning and row selection

Simulates "label a few rows, train, ask for more" and compares strategies by their learning curve.

```python
rep = ActiveLearningStudy(learner="knn", n_steps=8, n_repeats=5).run(X, y)
print(rep["ranking"], rep["recommendation"])
plot_result(rep).savefig("al.png")
```

Strategies: `random` (baseline), `least_confidence`, `margin`, `entropy`, `committee` (bootstrap vote entropy),
`kcenter` (diversity), `density_margin` (uncertainty x representativeness), `coverage_margin` (our variant:
margin uncertainty down-weighted near already-labelled rows, batch-wise) and `diversity_static`
(label-free farthest-first prefix). Learners: `"logistic"`, `"knn"`, `"lda"`, `"mlp"` (StatCore), a factory
returning a fresh estimator, or any estimator with `fit`/`predict_proba` (e.g. an SVM or decision tree from
another library). Classification only.

Measured (area under the learning curve, 1 200 rows, 5 repeats; random = baseline):

| Data / learner | random | margin | coverage_margin | diversity_static | best |
|---|---|---|---|---|---|
| rings / logistic | 0.708 | 0.718 | 0.704 | 0.716 | least_confidence 0.718 |
| rings / KNN | 0.689 | 0.712 | 0.712 | 0.717 | diversity_static 0.717 |
| rings / LDA | 0.706 | 0.709 | 0.715 | 0.708 | committee 0.716 |
| imbalanced / logistic | 0.685 | 0.698 | 0.710 | 0.720 | diversity_static 0.720 |
| imbalanced / KNN | 0.261 | 0.384 | 0.480 | 0.478 | coverage_margin 0.480 |
| imbalanced / LDA | 0.680 | 0.723 | 0.718 | 0.771 | diversity_static 0.771 |

The honest reading: no strategy wins everywhere; gains over random are small for well-behaved learners and
large for KNN on imbalanced data; on imbalanced data a curve can beat the all-rows "ceiling" because selecting
rows also rebalances classes. The report states when random is competitive.

**Plugging in your own row selector** (any row-selection or sampling method you already have):

```python
def my_selector(X_array, n):          # must return n row positions of X_array
    return my_sampler.select(X_array, n)

ActiveLearningStudy(row_selector=my_selector).run(X, y)       # adds a static "row_selector" curve
evaluate_row_subset(X, y, idx=my_selector(X.to_numpy(), 150)) # skill retained vs all rows + KS / class balance
```

Only the returned indices are used. `evaluate_row_subset` reports `skill_retained`, `ks_mean`
(distribution fidelity) and `class_balance_tv`, so a selector can be compared with the built-in baselines.

## Running sections together, and in AutoML

```python
di = DataIntelligence(export_dir=None)                    # set a folder to save json/txt/html
res = di.run_sections(df, "target", sections=("power", "group_scan", "construction"),
                      time_column=None, active_learning={"learner": "knn"})
print(describe(res))
```

Sections: `power`, `group_scan`, `root_cause`, `construction`, `active_learning`, `time_series`. Errors become
`{"error": ...}` instead of exceptions.

```python
from reliableml_automl import ReliableAutoML

auto = ReliableAutoML("label", data_intelligence=("multicollinearity", "power", "construction", "active_learning"))
auto.fit(df)
auto.data_intelligence_report["sections"]["power"]["verdict"]
```

`data_intelligence` accepts `None`, a name, or a sequence of `multicollinearity`, `profile`, `power`,
`construction`, `active_learning`, `group_scan`, `root_cause`. All are **report-only** and computed on the
*training partition only*; they never change the pipeline, the partition or the leaderboard (a test checks
the leaderboard is identical). Default stays `"multicollinearity"`.

## Applications

| Domain | Workflow |
|---|---|
| **Marketing / web** | `ab_test` for the page or offer; `compare_groups` by channel; `root_cause` on drop-off |
| **Churn / customer analytics** | `scan_groups` for separating attributes; `root_cause` for the segment; `power_analysis` before modelling |
| **Real-estate / pricing** | `FeatureConstructor` (price/area ratios, log transforms); `power_analysis` for the explanatory gap |
| **Sensors / operations** | `time_series_analysis` for drift, seasonality, level shifts; feed shifts into `DriftDetector` |
| **Healthcare / labelling budgets** | `ActiveLearningStudy` to estimate labels needed; `evaluate_row_subset` for a curated sample |
| **Model governance** | `power_analysis` (is n enough, is the story real) before sign-off, next to `evaluate`, fairness and the reason audit |
| **Small-data science** | the "explains but does not predict" verdict and required-n table before collecting more data |
| **Data-poor teams with no question** | `DataIntelligence().analyze(df)` then `question_plan`, then the matching tool above |

## Limits to keep in mind

* Power analysis uses a linear (F-test) approximation; for classification the F test is approximate.
* Root cause and group scans are associational. Multiple testing is controlled (BH/Holm) but selection effects remain.
* Feature construction improves *linear-ridge-measured* skill; check the gain with your final model.
* Active learning is a simulation on labelled data; real annotators add cost and noise.
* Active learning for regression is not implemented.
