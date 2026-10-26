# ML Reliability: technical reference

> This page is the detailed reference (drift metric definitions, evaluation, explanation and audit
> semantics, save/load). The [README](../README.md) is the user guide, organised by use case, and
> the runnable scripts are in [`examples/`](../examples).

`ReliableML` is a Python library for building and using machine-learning
models with reliability checks alongside their predictions. Its own
meta-learning classifiers can be fitted directly or used inside
`ReliablePipeline`. For data partitioning, use the companion
[ATTP](https://github.com/Jabualani/ATTP) methods; standard scikit-learn
estimators and partitioners remain available when they better suit a project.

The initial release records training feature distributions during `fit`, then
compares unseen features against that reference at prediction time. Reports
include per-feature Kolmogorov-Smirnov (KS) statistic, Kullback-Leibler (KL)
divergence, Population Stability Index (PSI), and 1-D Wasserstein distance,
along with a moment-based Drift Score Index (DSI), categorical chi-square
homogeneity test, drift flags, and an overall PSI-derived distribution
similarity indicator. Separate methods assess input covariate shift and
label-dependent feature-target concept shift.

## Quick start

This Core example uses ReliableML's `MetaPipelineClassifier` together with
ATTP's S-ATTP classifier partitioner. Install ATTP separately when using it
directly with Core:

```powershell
python -m pip install -e .
python -m pip install i-attp==0.2.3
```

The input features passed to `s_attp_clf` and the meta-learner must be finite
numeric values. Prepare/encode them first, and keep a separate untouched,
representative final evaluation set: S-ATTP chooses a partition using its own
split procedure, so its test portion is not an unbiased final performance
estimate.

```python
import attp
from ml_reliability import ReliablePipeline
from ml_reliability import MetaPipelineClassifier

partition = attp.s_attp_clf(X_development, y_development)
X_train = X_development.loc[partition.train_indices]
y_train = y_development.loc[partition.train_indices]
X_partition_test = X_development.loc[partition.test_indices]

meta_model = MetaPipelineClassifier(
    n_folds=3,
    random_state=42,
)
model = ReliablePipeline(
    estimator=meta_model,
    drift_threshold=0.2,
    plot_dir="reliability_plots",
    reliability_required=True,
).fit(X_train, y_train)

predictions, report = model.predict_with_report(X_partition_test)
print(predictions)
print(report.to_dict())
print(model.last_plot_path)
print(model.last_report_path)

model.save("reliable-model.pkl")
```

Automatic classifier selection and ATTP partitioning are available as a
separate companion distribution, [ReliableML-AutoML](../automl/README.md), so
the core ReliableML installation does not require its extra dependencies.
AutoML uses `s_attp_clf` by default, performs light nested hyperparameter
tuning, and compares ReliableML's `MetaPipelineClassifier` and
`PerfMetaSelect`. Standard classifier baselines, other ATTP partitioners, and
your own candidate estimators are available as explicit options.

The wrapped estimator receives the original `X` in both `fit` and `predict`.
Use the same prepared feature representation for training and inference.
Numeric features are assessed using reference-quantile bins; categorical
features use the categories observed during fitting. Missing values and
unseen categories are tracked separately; numeric values outside the observed
training range are tracked as underflow or overflow. Input features and column
names must match between fit and inference.

KL is reported in the reference-to-incoming direction, `KL(reference || incoming)`,
using smoothed proportions over the fitted bins. PSI uses the standard sum
`(incoming_share - reference_share) * ln(incoming_share / reference_share)`
over those same bins. For numeric features, KS is the maximum absolute
difference between the two empirical CDFs, and Wasserstein is the 1-D
earth-mover distance between their finite numeric observations. Missing and
non-numeric values are excluded from KS and Wasserstein; these metrics are
`None` for categorical features or if either side has no finite numeric
observations. KL and PSI are computed for both numeric and categorical
features, with a small additive smoothing term for empty bins. A feature's
`drifted` flag continues to use the configurable PSI threshold.

DSI is a library-defined moment distance because the thesis describes the
index but does not give a unique formula: it is the Euclidean distance between
four moment changes, with mean and standard-deviation changes normalized by
the reference standard deviation, plus differences in skewness and excess
kurtosis. It is numeric-only; `dsi_threshold` controls its separate flag.
Categorical features use a two-sample Pearson chi-square homogeneity test over
the reference categories plus an unseen-category bucket. The report exposes
the statistic and p-value; `chi_square_alpha` controls the test flag. Results
with very small expected category counts should be interpreted cautiously.
The legacy `drifted`/`drifted_features` fields remain PSI-threshold indicators;
DSI and chi-square have their own `dsi_drifted` and `chi_square_drifted` flags,
and `covariate_shift_detected` combines the per-feature flags.

`assess_covariate_shift(X)` compares only input feature distributions and
returns the same metrics as `assess_reliability(X)`, including the aggregate
`covariate_shift_detected` flag. It does not use labels or estimate a joint
multivariate density; per-feature tests can miss shifts visible only through
feature interactions.

`fit(X, y)` stores training labels for optional `assess_concept_shift(X, y)`.
Concept shift is estimated per feature by binning numeric values using
training quantiles (or using fitted categories) and comparing conditional
target-label distributions with weighted Jensen-Shannon, directional
KL(reference || incoming), and PSI divergences. The default shifted flag uses
the Jensen-Shannon threshold of `0.1` nats and can be set with
`concept_shift_threshold`. This is a univariate approximation, requires
incoming labels, and may miss interactions; it is a retrospective evaluation,
not an unlabeled inference-time warning.

## Labeled classification evaluation

Prediction-time distribution reliability is unlabeled. Classification
performance is only measurable once ground-truth test labels arrive, so call
`evaluate(X_test, y_test)` after `predict()` (or instead of a separate
`predict_with_report()` call). It returns an `EvaluationReport`, attaches it
to `model.last_report.evaluation`, and updates the saved prediction JSON.

```python
evaluation = model.evaluate(X_test, y_test)
print(evaluation.classification["matthews_correlation_coefficient"])
print(evaluation.classification["confusion_matrix"])
print(evaluation.statistical_tests["feature_target_associations"])
print(evaluation.statistical_tests["training_vs_test_cohorts"])
print(evaluation.bias_variance)
```

The classification section includes confusion counts, accuracy, balanced
accuracy, per-class precision/recall/specificity/F1, macro and support-weighted
summaries, and multiclass Matthews correlation coefficient (MCC). The
statistical report is deliberately split into:

1. **Feature-target associations:** numeric-feature one-way ANOVA, a joint
   numeric-feature MANOVA, and categorical-feature chi-square tests across the
   observed test target classes.
2. **Training-versus-test cohorts:** per-numeric-feature ANOVA, joint numeric
   MANOVA, and per-categorical-feature chi-square tests comparing fitted
   reference rows with the evaluated rows.

Each test result includes its statistic, p-value, degrees of freedom where
applicable, and expected-count cautions or an explicit unavailable reason.
Tests require suitable group sizes and distributional assumptions; p-values
are not causal conclusions. MANOVA requires at least two numeric features and
sufficient complete rows.

`evaluate()` also estimates an empirical one-hot/Brier bias-variance
decomposition using repeated stratified CV. The default uses five folds and
five repeats (25 refits), reduced to fewer folds when the smallest class is
smaller. Configure `bias_variance_folds` and `bias_variance_repeats` on the
pipeline; set the repeats to one to reduce work. This is compute-intensive and
is a finite-sample out-of-fold decomposition, not a population irreducible
noise estimate. If the estimator implements `predict_proba`, those are used;
otherwise, hard class predictions are one-hot encoded.

When the fitted estimator implements `predict_proba`, the classification
report also includes `probability_calibration`: mean classwise Brier score,
log loss, and top-label expected calibration error (ECE) with equal-width
confidence-bin details. Lower values indicate better probability calibration
on that labeled evaluation batch; ECE depends on the chosen binning and sample
size. Estimators without `predict_proba` report calibration as unavailable.
These are evaluation-set diagnostics, not guarantees about future
calibration.

## ReliableML meta-learning classifiers

ReliableML's meta-learners are a recommended Core starting point. These
scikit-learn-compatible classifiers learn from out-of-fold predictions or
statistical meta-features. `MetaPipelineClassifier` is the general-purpose
OOF stack shown below; its base-model pool is configurable. Use your own
estimators or the included scikit-learn-compatible choices. Candidate
estimators should implement `fit` and `predict`; probability output is
preferred, while `decision_function` and hard predictions are supported as
fallbacks. Input features must be finite numeric values.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from ml_reliability import (
    MetaPipelineClassifier,
    ReliablePipeline,
    StatisticalMetaDTreeClassifier,
    StatisticalMetaLDAClassifier,
)

base_candidates = [
    ("nb", GaussianNB()),
    ("logreg", LogisticRegression(max_iter=500)),
    ("lda", LinearDiscriminantAnalysis()),
    ("tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
]

meta_model = MetaPipelineClassifier(
    base_estimators=base_candidates,
    meta_classifier=LogisticRegression(max_iter=500),
    n_folds=5,
    random_state=42,
)
model = ReliablePipeline(meta_model, plot_dir="reliability_plots").fit(X_train, y_train)
predictions = model.predict(X_test)

linear_model = StatisticalMetaLDAClassifier(
    estimation_method="mle", use_monte_carlo=True, n_mc_sims=20
)
nonlinear_model = StatisticalMetaDTreeClassifier(
    n_clusters=8, estimation_method="elasticnet", n_mc_sims=20
)
```

Other ReliableML estimators include `PerfMetaSelect` (cross-validated candidate selection),
`CalibratedStackedClassifier` (OOF stacking plus a calibrated final estimator),
`MetaFeatureTransformer` (OOF prediction features for composition with, for
example, a logistic-regression step), `MetaPipelineClassifier` (a complete
OOF-trained stack), `StatisticalMetaClassifier`, `StatisticalMetaLDAClassifier`,
`StatisticalMetaDTreeClassifier`/`StatisticalMetaTreeClassifier`, and
`StatisticalMetaNNClassifier`. The general statistical classifier supports
`is_nonlinear=True` for its nonlinear variant. The statistical LDA, tree, and
neural classifiers currently support binary classification. The general OOF
stacking and candidate selector support multiclass targets when their
constituent estimators do.

Core lets you choose and fit a ReliableML estimator yourself; it does not
automatically compare all meta-learners. For automatic candidate comparison,
use ReliableML-AutoML. In AutoML, you can also pass a ReliableML meta-learner
or another compatible estimator through `candidates` and define its tuning
space.

`StatisticalMetaNNClassifier` is implemented with NumPy and does not require
PyTorch or another optional neural-network package. The meta-learning
estimators require scikit-learn, installed as a core dependency.

Meta classifiers expose `predict`, `predict_proba`, and standard
scikit-learn parameters, so they can be cloned, put into cross-validation,
saved in `ReliablePipeline`, and restored with `ReliablePipeline.load()`. A
saved meta pipeline retains its fitted base models and transforms; prediction
uses the same raw feature input representation and feature order as training.
Use out-of-fold features during stack training to avoid training the final
learner on in-sample base predictions.

Global and local raw-feature importance are available on the classifiers:

```python
global_importance = meta_model.feature_importances(X_test)
local_importance = meta_model.local_feature_importances(X_test[:5])
print(global_importance["features"])
print(local_importance[0]["top_features"])
```

These model-agnostic scores replace one raw feature at a time with its median
over the supplied data and measure changes in the probability of the model's
predicted class. The global report includes mean absolute probability change
and how often a feature appears in the per-row top-feature set. It is a
sensitivity ranking, not a causal attribution or proof of correctness; use a
representative reference set and review plausible feature dependencies.

Successive-halving hyperparameter search is available through thin factories
that return scikit-learn search objects:

```python
from ml_reliability import halving_grid_search, halving_random_search

grid = halving_grid_search(
    meta_model,
    {"meta_classifier__C": [0.01, 0.1, 1.0, 10.0]},
    cv=3,
    scoring="roc_auc",
    random_state=42,
)
grid.fit(X_train, y_train)
best_meta_model = grid.best_estimator_
```

Use `halving_random_search(estimator, param_distributions, ...)` similarly,
with `n_candidates` and a list/distribution per parameter. These helpers
enable scikit-learn's experimental successive-halving API. Choose a parameter
path present in the configured estimator (for example,
`meta_classifier__C` requires a non-`None` meta-classifier). Always keep
preprocessing inside a scikit-learn pipeline when it learns from data, so
cross-validation and halving fit that preprocessing separately within folds.

## Statistical data cleaning

The standalone cleaning function uses light mode by default:

```python
from ml_reliability import clean_statistical_data

cleaned, cleaner = clean_statistical_data(
    dataframe.drop(columns=["y"]),
    cleaning_intensity="light",
    return_cleaner=True,
)
print(cleaner.get_cleaning_report())
```

Light mode imputes missing numeric values using binary mode, near-normal mean,
or otherwise median; categorical values use the training mode (or a missing
sentinel for all-missing columns). It reports IQR outliers but does not clip
them. Moderate and heavy modes apply fit-time IQR caps. Keep the cleaner
inside a scikit-learn pipeline when using cross-validation, so transformations
are fitted only on each training fold.

Convert datetime columns before using this cleaner. Follow the requirements
of the estimator and preprocessing steps you choose for other input types.

### Fairness assessment

`evaluate()` accepts protected-group data separately from model features so
that protected attributes are not accidentally passed to the estimator:

```python
evaluation = model.evaluate(
    X_test,
    y_test,
    sensitive_features={"gender": gender_test, "region": region_test},
    positive_label=1,
)
print(evaluation.fairness)
```

Binary classification requires an explicit `positive_label`. For multiclass
classification, each observed/model class is evaluated one-vs-rest. Reports
include statistical-parity selection-rate gaps, equalized-odds TPR/FPR gaps,
and predictive-parity positive-predictive-value gaps, along with group
denominators and bootstrap confidence intervals. Gaps are max-minus-min over
the supplied groups; zero means equal observed rates, not proof of fairness.
These descriptive comparisons do not set a legal or application-specific
fairness threshold.

### Model deterioration over time

Provide at least three dated, labeled prediction batches to estimate an
accuracy trend and bootstrap interval for its slope:

```python
history = [
    {"timestamp": "2026-01-01", "y_true": y_jan, "y_pred": pred_jan},
    {"timestamp": "2026-02-01", "y_true": y_feb, "y_pred": pred_feb},
    {"timestamp": "2026-03-01", "y_true": y_mar, "y_pred": pred_mar},
]
deterioration = model.assess_deterioration(history)
print(deterioration.accuracy_slope)
print(deterioration.slope_confidence_interval)
print(deterioration.deterioration_detected)
```

Numeric timestamps or ISO date/time strings are accepted. Each batch contributes
one equally weighted accuracy observation. The slope unit is accuracy per
numeric time unit or per day for ISO timestamps. A deterioration flag is raised
when the bootstrap interval for the linear accuracy slope lies entirely below
zero. The method does not adjust for changes in case mix, temporal dependence,
or evaluation protocol.

### Global and local explanation consistency

Model-agnostic Monte Carlo Shapley values and LIME-style local explanations
can be compared for probability-capable classifiers:

```python
explanation = model.explain(X_test, class_label=1)
print(explanation.global_shap_importance)
print(explanation.local_explanations[0])
print(explanation.reason_alignment)
```

The global ranking is mean absolute Shapley importance over the explained rows;
each local explanation includes SHAP values and LIME weights. The report
summarizes absolute-importance Spearman rank agreement, top-feature overlap,
sign agreement, and a bootstrap interval for mean rank agreement. This
quantifies whether two explanation methods tell a consistent story for the
selected rows. It is **not** a prediction confidence, proof that the model
used a correct reason, or causal evidence. Explanations currently require
finite numeric features, an estimator with `predict`, `predict_proba`, and `classes_`,
and at least two input features. Causal AI is not included: credible causal
analysis requires a justified causal graph and intervention assumptions,
which cannot be inferred from a fitted predictor alone. Explanations are
computed with Monte Carlo permutations and local surrogate perturbations;
`num_samples` controls their compute/accuracy trade-off. If pipeline
reliability monitoring was disabled to save memory, pass `background_data`
explicitly to `explain()`.

The standalone `explain_estimator()` function supports an already-fitted
external classifier without requiring labels. Supply `y_true` only when you
also want descriptive accuracy on those rows; labels are not used to produce
the explanations. Optional `feature_importance` values are reported as
user-supplied scores and compared with global and per-row attribution rankings;
they are not relabeled as SHAP values. A representative training/reference
`background_data` is recommended. If omitted, `X` is used and the report
includes a warning.

Both `explain_estimator()` and `ReliablePipeline.explain()` can export
original feature columns, optional `y_true`, generated model predictions, and
each row's top Shapley features:

```python
from ml_reliability import explain_estimator

report = explain_estimator(
    fitted_classifier,
    X_test,
    y_true=y_test,  # optional
    background_data=X_train,
    feature_importance={"age": 0.42, "income": 0.31},  # optional
    export_path="explanation-results.csv",
)
print(report.label_evaluation)
print(report.export_path)
```

Providing `export_path` is the explicit opt-in to writing potentially
sensitive input data. CSV is the default export format; `.pkl`/`.pickle` and
`.joblib` suffixes select pickle and joblib formats, or use `export_format`
explicitly. Joblib export requires the optional `joblib` package. Exports
include every input row (so explanations can be more expensive); without an
export path, explanations default to at most 50 rows unless `max_instances` is
set. When exporting, set `max_instances` to cover all rows or leave it unset.
CSV stores the top-features list as JSON in one column. The optional accuracy
is descriptive for the supplied rows and does not include a confidence
interval; use `audit_reasons()` or `evaluate()` for the fuller labeled-holdout
assessment.

`ReliablePipeline.predict(X)` retains the usual estimator behavior and returns
predictions only. Use `assess_reliability(X)` for a report without predictions,
or `predict_with_report(X)` to get both.

## StatCore models (`ml_reliability.statcore`)

Pure-NumPy models, each with `variant=` options and an `.evaluate(...)` method: `PCA`, `LDA`, `KMeans`, `LogisticRegression`, `RidgeRegression`, `KNNClassifier`, `KNNRegressor`, `MLPClassifier`, `MLPRegressor`, and a `MetaLearner` with five meta-variants. See `src/ml_reliability/statcore/SPEC.md` for the variants.

```python
from ml_reliability.statcore import LDA, MLPClassifier, MetaLearner

meta = MetaLearner(
    base_estimators=[("lda", LDA()), ("mlp", MLPClassifier())],
    meta_variant="metrics_based",
    n_folds=5,
)
meta.fit(X, y)
print(meta.evaluate(X_test, y_test))
```

### scikit-learn compatible StatCore estimators

`StatCoreLogisticClassifier`, `StatCoreLDAClassifier`, `StatCoreKNNClassifier`, `StatCoreMLPClassifier` and `StatCoreMetaClassifier` wrap the own (non-base) StatCore variants to the same standard as the `meta_learning` estimators: `get_params`/`clone`, validated input, `feature_names_in_`, `classes_`, aligned `predict_proba`, and `feature_importances`. They work in pipelines, grid searches, `MetaPipelineClassifier` stacks and as AutoML candidates. The numerical work is still done by StatCore (NumPy/SciPy).

```python
from ml_reliability import StatCoreKNNClassifier, StatCoreMetaClassifier

knn = StatCoreKNNClassifier(variant="stat_k_selection").fit(X, y)
stack = StatCoreMetaClassifier(meta_variant="calibrated_stack").fit(X, y)
```

The AutoML companion (`ReliableML-AutoML`) adds these as the candidates `sc_logistic`, `sc_lda`, `sc_knn`, `sc_mlp` and `sc_meta` next to `meta_pipeline` and `perf_meta_select` and selects the best by nested cross-validation (`include_statcore=False` turns them off).

## Multi-dataset method comparison (manual benchmark)

A standalone analysis, separate from `evaluate()` and from AutoML. Datasets are the independent units: aggregate folds, repeats and seeds within each dataset first.

```python
from ml_reliability import compare_methods, summarize_repeats, plot_critical_difference

table = summarize_repeats(long_results, dataset="dataset", method="method", score="score")
report = compare_methods(table, higher_is_better=True)
print(report.average_ranks, report.friedman, report.nemenyi["critical_difference"])
figure, axes = plot_critical_difference(report)
```

It reports average ranks, the Friedman test, Nemenyi critical difference, and pairwise Wilcoxon signed-rank tests with Holm correction, win/tie/loss counts, bootstrap CIs of mean differences, and a paired dominance effect size. Fewer than 10 datasets triggers a low-power warning.

## Publishing to PyPI

GitHub Actions tests the package on Python 3.10–3.14 for pushes and pull
requests to `main`, and builds and validates wheel/source distributions. To
publish, configure a PyPI trusted publisher for this GitHub repository,
workflow file `.github/workflows/publish.yml`, and GitHub environment `pypi`.
Set `PACKAGE_VERSION` in that workflow to the version to publish, then run
**Test and publish to PyPI** manually from the Actions tab on the `main`
branch. The workflow uses PyPI's OIDC trusted-publishing flow; no PyPI token
is stored in GitHub secrets. A GitHub Release is not required. Contributor
and citation metadata is available in `pyproject.toml` and `CITATION.cff`.

## Data Intelligence (`ml_reliability.data_intelligence`)

One script that profiles a dataset before modelling: target intelligence
(leakage screen, explainability with a permutation p-value, partial
correlation, alternative targets), feature intelligence (VIF/multicollinearity,
multi-method ranking, feature-subset search) and an outlier layer written in
NumPy: isolation forest, local outlier factor, robust Mahalanobis distance and
a Self-Organising Map, combined into a consensus (a row flagged by at least two
detectors). No scikit-learn models are used.

```python
from ml_reliability import DataIntelligence

di = DataIntelligence(mode="autonomous")  # nothing is written to disk
report = di.analyze(df, target_column="y")
print(report.to_text())
di.plot()["figures"]["som"]              # matplotlib figures, in memory

# Opt-in saving
di = DataIntelligence(export_dir="di_report")   # json/txt/html (+ figures)
di.export("di_report", formats=("html",))
```

**No target and no question?** `analyze(df)` returns `report["question_plan"]`
(prioritised questions, each with its evidence and the call that answers it) and
`report["candidate_targets"]` (which columns the rest of the data can predict).
`DataIntelligence().question_plan(df)` gives the same plan without the full report.

```python
report = DataIntelligence().analyze(df)          # no target, no question
for q in report["question_plan"][:5]:
    print(q["priority"], q["category"], q["question"], "->", q["next_step"])
```

**Nothing is saved unless you ask.** `plot_dir` and `export_dir` default to
`None`. `ReliableAutoML(export_dir=...)` writes the report, leaderboard and
data-intelligence JSON, and `export_model=True` adds the fitted pipeline.
No single outlier detector wins every scenario; treat the consensus as triage,
not as a verdict (see `docs/latex/data-intelligence.tex`).

A full walk-through (raw data → cleaning → feature selection → meta-learning and
StatCore → evaluation, fairness, explanation, audit → drift and deterioration →
AutoML) is in `docs/examples/data_intelligence_workflow.py`.

### Analysis toolkit: power, A/B, root cause, time series, construction, active learning

```python
from ml_reliability import (power_analysis, compare_groups, scan_groups, ab_test, root_cause,
                            time_series_analysis, FeatureConstructor, ActiveLearningStudy,
                            evaluate_row_subset, DataIntelligence, describe, plot_result)

print(describe(power_analysis(X, y)))                     # predictive vs explanatory vs statistical power
print(describe(ab_test(df, group="page", outcome="lead")))  # which variant wins, with CI and sample-size check
print(describe(root_cause(df, outcome="churn")))          # segments that concentrate the bad outcome
print(describe(time_series_analysis(df, "date", "sales")))  # trend, seasonality, stationarity, change point
fc = FeatureConstructor().fit(X_train, y_train)           # iterative, distribution-aware, holdout-checked
rep = ActiveLearningStudy(learner="knn").run(X, y)        # uncertainty / committee / diversity / own variants

DataIntelligence().run_sections(df, "target", sections=("power", "group_scan", "construction"))
```

* **Power analysis** separates predictive power (cross-validated skill), explanatory power (in-sample
  R²/F-test/per-feature tests) and statistical power (minimum detectable effect and required n).
* **Feature construction** generates candidates from each column's distribution, confirms them across
  resamples, and reports a gain on rows the search never saw (`holdout_gain`).
* **Active learning** compares query strategies by learning curve and accepts your own
  `row_selector(X, n) -> indices` (any sampling method of your own); only the returned indices are used.
* `ReliableAutoML(data_intelligence=("multicollinearity", "power", "construction", "active_learning"))`
  adds these as report-only sections on the training partition.

Full usage guide, measured results and applications: [`docs/data-intelligence-toolkit.md`](data-intelligence-toolkit.md).

## Method documentation

LaTeX source for the overall methodology and each implemented method family is
in `docs/latex/`, including a statistical data-cleaning paper and a separate
S-ATTP partitioning paper. Build
standalone PDFs into `docs/pdfs/` with:

```powershell
python .\docs\latex\build_pdfs.py
```

The script uses Tectonic or `pdflatex` and fails explicitly if neither is
installed. These IEEE-style documents are research-manuscript drafts, not
empirical evidence of model superiority or completed conference submissions.
The ATTP draft describes the inspected numeric classifier-splitting
implementation; it does not claim that every method listed in the broader
ATTP reference architecture is implemented. Reproducible experiments,
verified results, final affiliations, and venue-specific formatting are still
needed before submission.

## Distribution visualizations and confidence intervals

Seaborn is installed with the library. By default, `fit()` saves a statistical
training reference plot to `reliability_plots/training-distributions.png`.
`predict()` compares incoming features with the fitted training reference,
updates `model.last_report`, and automatically writes a comparison plot and
JSON report to the configured `plot_dir`, using unique prediction filenames.
Prediction artifacts accumulate in this directory; apply your own retention
policy for long-running deployments.
The trained distribution profile is stored inside the fitted model artifact,
so a loaded model can compare later inputs without the original training file.

`predict()` continues to return predictions; use `model.last_report` or
`predict_with_report()` to read its accompanying metrics. Report fields include
deterministic-seed basic-bootstrap confidence intervals for the overall
similarity score, mean PSI, and available per-feature PSI, KL, KS, Wasserstein,
DSI, and chi-square statistic estimates. The default is a 95% interval from 200
bootstrap resamples; configure it with `confidence_level`,
`bootstrap_samples`, and `random_seed`.

Reliability is required by default. When enabled, prediction computes the
distribution report and saves the plot and JSON output (including the
predictions). This intentionally makes ordinary `predict()` apply reliability
checks rather than silently returning unaccompanied predictions. For very
large datasets or memory-constrained inference, explicitly set
`reliability_required=False`: fitting will then skip reference-statistic
storage and baseline plots, and prediction returns model predictions without
reliability calculations or saved reports/plots. Reliability assessment
methods and `predict_with_report()` raise an explicit error in that mode.
This opt-out saves monitoring memory, but of course does not provide
reliability protection. The default-on monitor retains the training feature
observations for accurate empirical KS/Wasserstein, bootstrap intervals, and
plots; large reference datasets can therefore require substantial memory.

Comparison visuals use Seaborn density histograms and empirical CDFs for
numeric features, and categorical proportion bars with bootstrap intervals.
Metric intervals are shown in each feature's title and the overall similarity
interval appears above the figure. Training-only plots show bootstrap intervals
for numeric means and categorical proportions. `plot_distributions(X_unseen)`
returns a Matplotlib figure and axes for interactive use or custom saving;
`plot_reference_distributions()` plots the stored training baseline.
Optionally select features with `feature_names=["age", "region"]`.

Intervals quantify sampling variability in the reported distribution
statistics, not prediction confidence or probability of correctness. Interior
intervals use the basic bootstrap construction; metrics at natural boundaries
(for example PSI=0 or similarity=1) use one-sided bootstrap bounds so the
interval respects the metric range. The
bootstrap assumes independent, identically distributed rows; for temporally
or otherwise dependent data, these intervals may be too narrow because block
bootstrap is not yet implemented. Chi-square p-values are reported as test
results; the bootstrap interval applies to the chi-square statistic, not its
p-value. Concept-shift divergences do not yet include confidence intervals.

## Interpreting the report

These input-distribution statistics measure feature-distribution change, not
model correctness. The separate labeled concept-shift method compares a
univariate approximation of the feature-target relationship. The
`similarity_score` is `exp(-mean_psi)`, a convenient value between 0 and 1;
it is **not** a calibrated probability, confidence score, or guarantee that
predictions are reliable. Choose the drift threshold for the application and
validate it against representative data. Performance trends, fairness gaps,
and explanation consistency are separate assessments with their own
assumptions; none alone establishes prediction correctness.

### Statistical reason audit

`audit_reasons(X_holdout, y_holdout)` is the higher-level diagnostic for the
question “does the fitted model appear to rely on statistically supported,
stable reasons?” It requires labeled holdout rows and a classifier with
`predict_proba` and `classes_`. `ReliablePipeline.audit_reasons()` uses stored
training observations as the explanation background. For an independently
fitted model, call `audit_model_reasons()` and pass its training/background
data explicitly. If `ReliablePipeline` was created with
`reliability_required=False`, `audit_reasons()` requires an explicit
`background_data` rather than treating the holdout as its own reference:

```python
audit = model.audit_reasons(
    X_holdout,
    y_holdout,
    positive_label=1,
    conditional_on=["age", "region"],
    groups={"sex": sex_holdout},
    time_periods=month_holdout,
    counterfactuals={"income": plausible_income_changes},
    feature_importance={"age": 0.42, "income": 0.31},
    negative_control_features=["random_control"],
    unavailable_features=["post_outcome_status"],
)
print(audit.predictive_validity)
print(audit.feature_reliance)
print(audit.logistic_association)
print(audit.explanations["global_shap_importance"])
print(audit.explanations["local_explanations"])
print(audit.explanation_alignment)
```

The audit reports bootstrap intervals for holdout accuracy, multiclass Brier
score, top-label calibration error, and per-feature held-out Brier-loss
increases under donor perturbations. `conditional_on` requests nearest-neighbor
donor sampling conditional on the named numeric variables (excluding the
feature being tested from its own conditioning set). This reduces some
unrealistic combinations but is only an approximation to a conditional
distribution; sparse regions and unmodeled dependencies can still mislead.
Without `conditional_on`, the audit uses marginal donor perturbations and
explicitly warns about feature correlation.

Optional `groups` and `time_periods` report feature-reliance estimates and
within-slice bootstrap intervals. Pass domain-valid, row-aligned replacement
values through `counterfactuals` to estimate prediction-flip rates and
probability changes with intervals. The library cannot determine whether a
counterfactual is plausible: that requires application/domain constraints.
Mark known negative controls and features unavailable at prediction time to
surface warning flags; these flags do not prove leakage or a shortcut.

For binary or selected one-vs-rest labels, an unregularized logistic model
reports standardized-feature odds ratios, confidence intervals, p-values, and
Benjamini–Hochberg adjusted p-values. These are conditional associations with
the observed outcome. The report compares row-level signed Shapley
contributions with logistic log-odds contributions and also compares any
user-supplied importance mapping against global Shapley ranking. Agreement
means the fitted model, a simple statistical association model, and/or the
supplied scores show consistent patterns; it does not prove those patterns
are correct, causal, or non-spurious. Examine the holdout design, confounding,
measurement process, and external/domain evidence before drawing conclusions.

This audit is for finite numeric features and can be computationally
expensive: it refits predictions repeatedly for bootstrap perturbation and
explanations. Use representative holdout data that was not used to fit or
select the model. Calibration and performance intervals assume independent
rows; grouped or time-dependent data may require an appropriate block or
cluster bootstrap beyond the current implementation.

## Save and load

`save()` uses Python pickle so that the estimator and its fitted reference
profile can travel together. The stored profile includes the training feature
observations used by bootstrap and plotting, so protect the artifact as you
would the original feature data. Only load artifacts from trusted sources:
unpickling an untrusted file can execute arbitrary code.

## Development

```console
python -m pip install -e .
python -m unittest discover -s tests
```

## Run the example

After installing the package and its Seaborn dependency, run the demo:

```console
$env:PYTHONPATH = "src"
python examples/reliable_demo.py
```

The example prints predictions and metric confidence intervals for similar
and deliberately shifted batches, automatically writes baseline/comparison
plots and JSON reports to `reliability_plots/`, and demonstrates saving and
reloading the model.

## Research and contact

ReliableML is developed as part of research in statistical machine-learning
reliability. For research enquiries, contact Jabulani Monchwe at
[MonchweJ.25@gmail.com](mailto:MonchweJ.25@gmail.com) or see
[Google Scholar](https://scholar.google.com/citations?user=DqA_HV4AAAAJ&hl=en)
and [LinkedIn](https://www.linkedin.com/in/jabulani-monchwe-7a145117a/).

The **Stats, Math & ML (Applied ML & Methodology Development ML)** group can be
joined through its [WhatsApp invitation](https://chat.whatsapp.com/KLB5zk5w7xVIJHaM65KxPa).
The wider IDEA-Research Club community has a separate
[WhatsApp group invitation](https://chat.whatsapp.com/FOIpSTDLH0X6E6CIL6IZfn).

## Research roadmap

The workflow in `ML-Reliability.pdf` also motivates future work on model
selection and meta-learning, evaluation alignment, causal analysis, categorical
explanations, and broader temporal monitoring. Current deterioration, fairness,
and explanation reports remain separate assessments with documented limits
and independently testable outputs.
