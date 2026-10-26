# ML Reliability

**ReliableML** is a Python library for building machine-learning models whose reliability you can
*measure*: it investigates the data before modelling, offers its own pure-NumPy models, selects among
them automatically, and then checks drift, performance, fairness, deterioration and the *reasons* behind
predictions.

**Your own models come first.** ReliableML's meta-learners and the NumPy `statcore` models are the
defaults everywhere. scikit-learn estimators are accepted as optional extras, never required for the
new tools (Data Intelligence uses no scikit-learn models at all; scikit-learn is used for dataset loaders
and fold splitting in the examples).

| | |
|---|---|
| **Install** | `pip install ReliableML` (core) Â· `pip install ReliableML-AutoML` (companion, separate environment) |
| **Python** | 3.10+ |
| **Nothing is saved by default** | plots, reports, models and exports are written only when you pass a folder/path |
| **Full reference** | [docs/reference.md](docs/reference.md) (metric definitions, report fields, semantics) |
| **Papers (LaTeX + PDF)** | [docs/latex](docs/latex) Â· [docs/pdfs](docs/pdfs) |

## The four sections

The library is organised as four sections. Each has use cases underneath and a runnable script you can copy.

| # | Section | Use it when | Script |
|---|---|---|---|
| 1 | [Data Intelligence](#1-data-intelligence) | You have data but no question, or want to check the data/target *before* modelling | [01](examples/01_data_intelligence_discovery.py), [02](examples/02_data_intelligence_toolkit.py) |
| 2 | [Meta-learning and StatCore](#2-meta-learning-and-statcore-your-own-models) | You want to fit models (your own) and compare them | [03](examples/03_meta_learning_vs_statcore.py) |
| 3 | [AutoML](#3-automl-reliableml-automl) | You want models selected, tuned and cleaned for you | [04](examples/04_automl_use_cases.py) |
| 4 | [Reliability analysis](#4-reliability-analysis-in-isolation) | You have a fitted model and want drift, evaluation, fairness, deterioration, explanation and audit | [05](examples/05_reliability_in_isolation.py) |

**Suggested order in a project:** Data Intelligence â†’ cleaning â†’ Data Intelligence again (feature subset, what
skill to beat) â†’ models (section 2 or 3) â†’ evaluation, fairness, explanation, audit â†’ deploy â†’ drift on every
batch â†’ deterioration once labels arrive. Deterioration, drift, fairness and the audit are not one-off steps:
fit once, then re-run the checks as data arrives. `ReliablePipeline` is the wrapper that stores the training
reference so those checks can be run later. A complete walk-through is in
[docs/examples/data_intelligence_workflow.py](docs/examples/data_intelligence_workflow.py).

### Run the examples

```powershell
python -m pip install -e .                          # core
$env:PYTHONPATH = "src"                             # when running from a clone
python examples/01_data_intelligence_discovery.py   # 1 min
python examples/02_data_intelligence_toolkit.py     # 1-2 min
python examples/03_meta_learning_vs_statcore.py     # 2-4 min
python examples/05_reliability_in_isolation.py      # 1-2 min
# the AutoML script needs the companion (see section 3)
python examples/04_automl_use_cases.py              # several minutes
```

Each script is self-contained, numbers its use cases (1-39 across the five scripts), prints what it
finds, and writes to a temporary folder that it deletes. Copy a block, replace the demo `DataFrame` with your own.

---

## 1. Data Intelligence

`ml_reliability.data_intelligence` is **one script** (pure NumPy plus SciPy/matplotlib) for the situation
"I have the data; what should I ask of it?". Entry point: `DataIntelligence`.

```python
from ml_reliability import DataIntelligence
report = DataIntelligence(mode="autonomous").analyze(df, target_column="y")   # nothing is written
print(report.to_text())
figures = DataIntelligence().plot()   # after analyze(): matplotlib figures in memory
```

**Modes** (`DataIntelligence(mode=...)`): `comprehensive_eda` (default), `target_intelligence`,
`feature_intelligence`, `field_investigation`, `auto_question_generator`, `llm_augmented` (optional LLM
backend), `autonomous` (everything that applies).

**Common options:** `vif_threshold`, `corr_threshold`, `leakage_threshold`, `max_rows` (seeded sample for the
expensive parts), `outlier_max_rows`, `som_grid`, `plot_dir` / `export_dir` (opt-in saving), `random_state`.

### Use cases (script [01](examples/01_data_intelligence_discovery.py))

| # | Use case | Call |
|---|---|---|
| 1 | **No target and no question.** Get a prioritised question plan and which columns the rest of the data can predict | `DataIntelligence().analyze(df)` â†’ `report["question_plan"]`, `report["candidate_targets"]` |
| 2 | **A target, before modelling.** Leakage screen, hazards (constant / high-cardinality), VIF multicollinearity, explainability verdict with permutation p-value, feature-subset search, alternative targets | `analyze(df, target_column="y")` â†’ `report["target_intelligence"]` |
| 3 | **Outlier triage.** Isolation forest, LOF, robust Mahalanobis and a Self-Organising Map, with a consensus (a row flagged by at least two detectors) | `isolation_forest_scores`, `local_outlier_factor`, `robust_mahalanobis`, `SelfOrganisingMap`; or the `outlier_consensus` / `som` figures |
| 4 | **Screen an incoming batch** with the same engine, no target needed | `DataIntelligence(mode="comprehensive_eda").analyze(batch)` |
| 4b | **Deep dive into one field.** Investigate a column or target: its profile and hazards, ranked drivers (Pearson, Spearman, mutual information), strongly related pairs, sub-groups and the follow-up questions they raise; then drill into the flagged rows | `DataIntelligence(mode="field_investigation").analyze(df, target_column="col")` â†’ `report["what_happened"]`, `report["why"]["drivers"]`, `report["question_plan"]` |
| 5 | **Save only when you ask** | `DataIntelligence(export_dir="out")` or `di.export("out", formats=("html",))` |

Lessons the script demonstrates: a leaked column makes feature search "keep" the leak, so act on the leakage
screen and *re-run* the analysis; no single outlier detector wins every scenario, so treat the consensus as
triage, not a verdict.

### The analysis toolkit (script [02](examples/02_data_intelligence_toolkit.py))

| # | Question | Function | Returns |
|---|---|---|---|
| 6 | Can the data **predict** the target, or only **explain** it? | `power_analysis(X, y)` | predictive skill (cross-validated), explanatory skill (in-sample RÂ², F-test), statistical power (minimum detectable effect, required n) and a verdict |
| 7 | Do groups differ? (descriptive, group-by means, t-test / ANOVA / non-parametric) | `compare_groups(df, value, group)`, `scan_groups(df, target)` | test choice, p-values, effect sizes, Holm/BH correction across columns |
| 8 | Which page or service is better? (A/B testing) | `ab_test(df, group, outcome)` | lift, confidence interval, p-value, smallest detectable lift at your sample size |
| 9 | Why is the bad outcome happening? (root cause) | `root_cause(df, outcome)` | segments that concentrate the outcome, with lift and rate; associations, not causes |
| 10 | What does the series do over time? | `time_series_analysis(df, time_col, value_col)` | trend (Mann-Kendall, Sen slope), seasonality period and strength, stationarity, change point |
| 11 | Can I construct better features? | `FeatureConstructor().fit(X, y)` | accepted features, search gain and **gain on untouched rows**; `.transform(X)` applies the recipe |
| 12 | Which rows are worth labelling / keeping? | `ActiveLearningStudy(learner="knn").run(X, y)`, `evaluate_row_subset(X, y, idx)` | learning curves for uncertainty / committee / diversity strategies; skill retained by a row subset |
| 13 | Several of the above at once | `run_sections(df, "y", sections=("power", "group_scan", "construction"))` | one dict; also `DataIntelligence().run_sections(...)` and `describe(result)` for plain text |

Options worth knowing:

* `power_analysis(task="auto"|"regression"|"classification", n_folds, alpha, power)`.
* `compare_groups(alpha, min_group)`, `scan_groups(max_levels, top)`; `ab_test(control=, treatment=, alpha, power)`.
* `root_cause(bad=, direction="high"|"low", max_depth, features=)`: you can define what "bad" means.
* `time_series_analysis(period=)` fixes the seasonal period instead of detecting it.
* `ActiveLearningStudy(learner="logistic"|"knn"|"lda"|"mlp"| factory | any estimator with predict_proba)`;
  pass `row_selector(X, n) -> indices` to evaluate **your own** row-selection method: only the returned
  indices are used. Use `evaluate_row_subset(X, y, idx)` to score any subset you already have.
* `plot_result(result)` returns the figure for a toolkit result; `describe(result)` the plain-text summary.

**Limits (read before trusting):** power analysis uses a linear F approximation, also for classification;
feature construction can accept a feature on pure noise in a minority of runs (the untouched-holdout check is
there to expose that); no active-learning strategy wins everywhere; the Sen slope is biased by a level shift;
Bayesian A/B testing and regression active learning are not implemented. Measured numbers and more applications
(marketing, healthcare, operations, finance, manufacturing, education, web) are in
[docs/data-intelligence-toolkit.md](docs/data-intelligence-toolkit.md).

---

## 2. Meta-learning and StatCore (your own models)

Two families, both with scikit-learn-compatible wrappers (`fit`, `predict`, `predict_proba`, `clone`,
`feature_importances`):

* **StatCore**: pure-NumPy models with named variants (`ml_reliability.statcore`).
* **ReliableML meta-learners**: stack or select among base models using out-of-fold predictions or
  statistical meta-features. Give them your own (StatCore) base models, or add scikit-learn ones.

| Model | Variants / options | Targets |
|---|---|---|
| `StatCoreLogisticClassifier` | `base`, `risk_calibrated`, `stat_regularised`, `robust_loss` | multiclass |
| `StatCoreLDAClassifier` | `base`, `shrinkage`, `sparse`, `heteroscedastic` | multiclass |
| `StatCoreKNNClassifier` | `base`, `prototype_compressed`, `learnable_metric`, `stat_k_selection` | multiclass |
| `StatCoreMLPClassifier` | `base`, `stat_regularised`, `reliability_head`, `uncertainty_head` | multiclass |
| `StatCoreMetaClassifier` | `meta_variant`: `performance_aware`, `metrics_based`, `oof_stacking`, `statistical_meta`, `calibrated_stack` | multiclass |
| `MetaPipelineClassifier` | `base_estimators`, `meta_classifier`, `n_folds` | multiclass |
| `PerfMetaSelect` | `estimators`, `cv`, `scoring` (cross-validated candidate selection) | multiclass |
| `CalibratedStackedClassifier` | `base_estimators`, `meta_estimator`, `calibration_method` | multiclass |
| `MetaFeatureTransformer` | OOF probabilities as features, for use in your own pipeline | multiclass |
| `StatisticalMetaLDAClassifier` / `StatisticalMetaDTreeClassifier` / `StatisticalMetaNNClassifier` (NumPy) | `estimation_method`, `n_mc_sims`, `n_clusters`, ... | binary |
| `statcore.RidgeRegression`, `KNNRegressor`, `MLPRegressor` | variants, `predict_interval` / `predict_uncertainty` | regression |
| `statcore.PCA`, `KMeans` | `robust`, `sparse`, `topology_aware`, ... | unsupervised |

### Use cases (script [03](examples/03_meta_learning_vs_statcore.py))

| # | Use case | Example |
|---|---|---|
| 14 | Compare every StatCore classifier variant | `StatCoreKNNClassifier(variant="stat_k_selection").fit(X, y)` |
| 15 | Stack StatCore base models with each meta-variant | `StatCoreMetaClassifier(meta_variant="calibrated_stack", n_folds=3)` |
| 16 | ReliableML meta-learners on **your own** base models | `MetaPipelineClassifier(base_estimators=[("lr", StatCoreLogisticClassifier()), ("knn", StatCoreKNNClassifier())])` |
| 17 | Statistical meta-learners (binary, NumPy only) | `StatisticalMetaNNClassifier()` |
| 18 | Raw StatCore beyond classification: regression, uncertainty heads, PCA, k-means | `MLPClassifier(variant="uncertainty_head").predict_uncertainty(X)` |
| 19 | Head-to-head across datasets with Friedman, Nemenyi and Wilcoxon-Holm | `compare_methods(summarize_repeats(results, ...))` |
| 20 | Tune an own model with successive halving | `halving_grid_search(StatCoreKNNClassifier(), {"n_neighbors": [3, 5, 9]}, cv=3)` |

Adding a scikit-learn model is just another `(name, estimator)` pair:
`MetaPipelineClassifier(base_estimators=own_bases + [("rf", RandomForestClassifier())])`.

Measured by the script (balanced accuracy, breast cancer, 30% holdout): StatCore logistic 0.963, StatCore MLP
with reliability head 0.966, StatCore meta-variants 0.944-0.955, `PerfMetaSelect` on own bases 0.963,
statistical meta-learners 0.938-0.949. Across three datasets the Friedman test is not significant (p=0.44,
three datasets is too few): do not read ranks as a verdict. Use 10 or more datasets for a real comparison.

**Candidate-estimator requirements:** finite numeric inputs; `fit` and `predict`; `predict_proba` preferred.
Keep any data-fitted preprocessing inside a pipeline so cross-validation refits it per fold.

---

## 3. AutoML (`ReliableML-AutoML`)

A separate distribution, so Core users need no extra dependencies.

```powershell
py -3.15 -m venv .venv-automl
.\.venv-automl\Scripts\Activate.ps1
python -m pip install -e . ; python -m pip install -e .\automl
```

```python
from reliableml_automl import ReliableAutoML
auto = ReliableAutoML("y").fit(df)                    # df contains the target column
print(auto.selected_model_name, auto.leaderboard)
predictions, report = auto.predict_with_report(new_features)
```

By default it partitions with ATTP `s_attp_clf`, compares `meta_pipeline`, `perf_meta_select` and the StatCore
models `sc_logistic`, `sc_lda`, `sc_knn`, `sc_mlp`, `sc_meta` by nested cross-validation (light tuning), and
reports Data Intelligence on the training partition only. Scikit-learn models are an opt-in `candidates` entry.

### Options

| Option | What it does | Values |
|---|---|---|
| `candidates` | Your own `(name, estimator)` list; replaces the defaults | any estimator with `fit`/`predict`(`_proba`) |
| `include_statcore` | Keep or drop the five StatCore defaults | `True` / `False` |
| `light_param_grids`, `extended_param_distributions` | Tuning space per candidate name | `{"name": {"param": [...]}}` |
| `tuning`, `tuning_iterations`, `tuning_cv` | Tuning depth | `"light"` (default grid) / `"extended"` (randomized) |
| `cv`, `scoring`, `random_state` | Outer cross-validation and metric | `scoring` e.g. `"balanced_accuracy"` |
| `partitioner`, `partitioner_options`, `partitioner_result_adapter` | How train/test rows are chosen | `s_attp_clf` (default), `s_attp_cluster`, `s_attp_reg`, `s_attp_ts`, `train_test_split`, or your callable |
| `cleaning_intensity` | Statistical cleaning strength | `"light"` / `"moderate"` / `"heavy"` |
| `scaling_strategy` | Numeric scaling | `"auto"`, `"standard"`, `"robust"`, `"minmax"`, `"none"` |
| `ordinal_features` | Ordered categories | `{"grade": ["low", "mid", "high"]}` |
| `data_intelligence` | Report-only sections on the training partition | `None`, a name, or a sequence of `multicollinearity` (default), `profile`, `power`, `construction`, `active_learning`, `group_scan`, `root_cause` |
| `vif_threshold` | Collinearity cut-off | default 10 |
| `export_dir`, `export_formats`, `export_model` | Opt-in saving | folder / `("json","txt","html")` / `True` |
| `plot_dir`, `reliability_options`, `progress_bar` | Reliability plots, `ReliablePipeline` settings, progress | `reliability_options={"reliability_required": False}` skips the drift reference |

### Use cases (script [04](examples/04_automl_use_cases.py))

| # | Use case | Key options |
|---|---|---|
| 21 | Defaults: ATTP split, 7 candidates, nested tuning | `ReliableAutoML("y")` |
| 22 | **Own models only**, with your own grids | `candidates=[...]`, `light_param_grids=...` |
| 23 | Own models vs scikit-learn on one leaderboard (optional) | add `("sk_forest", RandomForestClassifier())` |
| 24 | Messy data: numbers stored as text, missing values, categories, ordinal columns | `ordinal_features=` |
| 25 | Cleaning and scaling choices | `cleaning_intensity=`, `scaling_strategy=` |
| 26 | Data Intelligence inside AutoML (never changes the model) | `data_intelligence=("profile", "power", "group_scan", ...)` |
| 27 | Choose the partitioner: ATTP variants, plain split, or your function | `partitioner=` |
| 28 | Light vs extended tuning | `tuning="extended"` |
| 29 | Saving, only when asked; reload as a `ReliablePipeline` | `export_dir=`, `export_report()`, `save()` |
| 30 | Reliability on the result (drift report, evaluation) | `predict_with_report`, `evaluate` |

The selected model is "the best candidate under this search and CV design", not globally optimal. In the
script's own run a scikit-learn forest edged out the StatCore models on one dataset and not on another:
that is why AutoML compares them instead of assuming. Details: [automl/README.md](automl/README.md).

---

## 4. Reliability analysis in isolation

These checks work on **any** fitted classifier (yours, StatCore, or scikit-learn), wrapped in
`ReliablePipeline` or called as standalone functions. Each is independent.

```python
from ml_reliability import ReliablePipeline, StatCoreLogisticClassifier
model = ReliablePipeline(StatCoreLogisticClassifier(), plot_dir="reliability_plots").fit(X_train, y_train)
```

`fit` stores the training reference. Options: `drift_threshold` (PSI), `dsi_threshold`, `chi_square_alpha`,
`concept_shift_threshold`, `n_bins`, `confidence_level`, `bootstrap_samples`, `bias_variance_folds` /
`bias_variance_repeats`, `reliability_required=False` (opt out of monitoring), `plot_dir`.

### Use cases (script [05](examples/05_reliability_in_isolation.py))

| # | Check | Call | Needs labels | What you learn |
|---|---|---|---|---|
| 31 | **Cleaning** (when data is not clean) | `clean_statistical_data(df, cleaning_intensity="moderate")` | no | imputation and outlier handling, with a report; keep it fitted on training rows only |
| 32 | **Drift**: covariate and concept shift | `model.assess_reliability(X)`, `assess_covariate_shift(X)`, `assess_concept_shift(X, y)`; stand-alone `DriftMonitor().fit(X, y)` | covariate: no Â· concept: yes | per-feature KS, KL, PSI, Wasserstein, DSI, chi-square; flags; similarity score |
| 33 | **Evaluation** | `model.evaluate(X, y, positive_label=1)` | yes | accuracy, MCC, per-class metrics, calibration (Brier, log loss, ECE), ANOVA/MANOVA/chi-square tests, bias-variance by repeated CV |
| 34 | **Fairness** | `model.evaluate(X, y, sensitive_features={"site": s}, positive_label=1)` | yes | selection-rate, equalized-odds and predictive-parity gaps with bootstrap intervals |
| 35 | **Deterioration** | `model.assess_deterioration(history)` or `assess_deterioration(history)` | yes (dated batches, 3 or more) | accuracy slope with an interval; a flag when the whole interval is below zero |
| 36 | **Explainability** | `model.explain(X, class_label=1)`; stand-alone `explain_estimator(model, X, export_path="e.csv")` | optional | Monte-Carlo Shapley vs LIME agreement; optional CSV export (opt-in) |
| 37 | **Reason audit** | `model.audit_reasons(X, y, groups=, negative_control_features=, unavailable_features=, counterfactuals=, conditional_on=)`; stand-alone `audit_model_reasons(model, X, y, background_data=)` | yes | whether the model relies on statistically supported, stable reasons; flags negative controls and post-outcome features |
| 38 | **Visualisation** | `model.plot_distributions(X_new)`, `plot_reference_distributions()` | no | reference vs incoming distributions with intervals; `figure.savefig(...)` is your call |
| 39 | **Pipeline options** | `reliability_required=False`, `drift_threshold=0.1`, `save()` / `ReliablePipeline.load()` | no | opt out, stricter thresholds, persistence |

What the script measured: on a shifted batch the similarity score fell from 0.88 to 0.02 and four features
were flagged; on a simulated degrading history the accuracy slope was -0.0049 per day with an interval
entirely below zero (detected), versus +0.0003 (not detected) on the stable one; Shapley and LIME alignment
was 0.84 on the 0-1 scale; fairness gaps are printed with intervals so you can judge whether a gap is noise.

Interpretation rules: these are descriptive diagnostics, not guarantees. The similarity score
(`exp(-mean_psi)`) is not a probability of correctness; Shapley/LIME agreement does not prove a model used
the right reason; fairness gaps are max-minus-min, and you choose the threshold; intervals assume independent
rows. Where to place each check in a project is in the "Suggested order" above.

`ReliablePipeline.save()` uses pickle and stores training feature observations: protect the file and only load
artifacts you trust.

---

## Papers and documentation

| Document | Purpose |
|---|---|
| [docs/reference.md](docs/reference.md) | Metric definitions, report fields and semantics for sections 2 and 4 |
| [docs/data-intelligence-toolkit.md](docs/data-intelligence-toolkit.md) | Toolkit guide, measured results, applications |
| [automl/README.md](automl/README.md) | AutoML options, partitioners, exports |
| [docs/latex](docs/latex) / [docs/pdfs](docs/pdfs) | Papers: Data Intelligence, ReliableML, cleaning, S-ATTP, AutoML comparison |
| [docs/colab](docs/colab) | Colab notebooks |

Build the PDFs with `python docs/latex/build_pdfs.py` (Tectonic or `pdflatex`). The papers are
research-manuscript drafts, not evidence of model superiority.

## Development

```console
python -m pip install -e .
python -m unittest discover -s tests            # core suite
python -m unittest discover -s automl/tests     # run in the AutoML environment
```

GitHub Actions tests Python 3.10-3.14 and builds wheel and source distributions. Publishing uses PyPI trusted
publishing (workflow `.github/workflows/publish.yml`, environment `pypi`): set `PACKAGE_VERSION`, then run
**Test and publish to PyPI** manually from `main`. Citation metadata: `CITATION.cff`.

## Research and contact

ReliableML is developed as part of research in statistical machine-learning reliability. Contact Jabulani
Monchwe at [MonchweJ.25@gmail.com](mailto:MonchweJ.25@gmail.com), or see
[Google Scholar](https://scholar.google.com/citations?user=DqA_HV4AAAAJ&hl=en) and
[LinkedIn](https://www.linkedin.com/in/jabulani-monchwe-7a145117a/).

The **Stats, Math & ML (Applied ML & Methodology Development ML)** group can be joined through its
[WhatsApp invitation](https://chat.whatsapp.com/KLB5zk5w7xVIJHaM65KxPa); the wider IDEA-Research Club has a
separate [WhatsApp group](https://chat.whatsapp.com/FOIpSTDLH0X6E6CIL6IZfn).

## Research roadmap

Future work: model selection and meta-learning, evaluation alignment, causal analysis, categorical
explanations and broader temporal monitoring. Current deterioration, fairness and explanation reports remain
descriptive; causal AI is not included, because credible causal analysis needs a justified causal graph and
intervention assumptions that a fitted predictor cannot supply.
