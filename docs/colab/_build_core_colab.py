"""Build the Google Colab notebooks for the core ReliableML method papers.

Run from anywhere:  python docs/colab/_build_core_colab.py
Writes <name>.ipynb next to this script (plain json, no extra dependencies).
The statcore-*.ipynb and automl.ipynb notebooks are produced elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/Jabualani/ML-Reliabilty"
COLAB_BASE = "https://colab.research.google.com/github/Jabualani/ML-Reliabilty/blob/main/docs/colab/"


def _lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def header(name: str, title: str, paper: str, blurb: str) -> dict:
    badge = (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({COLAB_BASE}{name}.ipynb)"
    )
    return md(
        f"# {title}\n\n{badge}\n\n"
        f"Companion notebook for the ReliableML paper `docs/latex/{paper}.tex`. "
        f"{blurb}\n\n"
        "All numbers shown below are computed by the cells themselves (fixed random seeds); "
        "nothing is hard-coded. Run the cells top to bottom."
    )


def setup_cells(extra_pip: str = "") -> list[dict]:
    extra = f"\n{extra_pip.strip()}\n" if extra_pip.strip() else ""
    return [
        md(
            "## 0. Setup (parameters)\n\n"
            "**Parameter cell - choose ONE installation source.** The default installs the released "
            "package from PyPI. To try the unreleased development version, comment out option A and "
            "uncomment option B. If neither works (for example a restricted network), run the "
            "fallback cell below it and upload a zip of the repository."
        ),
        code(
            "#@title PARAMETERS: install ReliableML\n"
            "# ---- Option A (default): released version from PyPI ----\n"
            "!pip install -q ReliableML\n"
            "# ---- Option B: unreleased version from GitHub (uncomment; comment option A) ----\n"
            f'# !pip install -q "git+{REPO_URL}.git"\n'
            "# Plots need matplotlib (preinstalled on Colab); locally: pip install matplotlib"
            f"{extra}"
        ),
        code(
            "#@title Fallback: upload a repository zip (only runs if ml_reliability is not importable)\n"
            "import importlib.util, pathlib, sys, zipfile\n"
            "\n"
            'if importlib.util.find_spec("ml_reliability") is None:\n'
            "    from google.colab import files  # Colab only\n"
            "\n"
            "    # GitHub: Code > Download ZIP, then pick the file in the dialog\n"
            "    uploaded = files.upload()\n"
            "    for archive in uploaded:\n"
            '        zipfile.ZipFile(archive).extractall("repo_zip")\n'
            '    package_init = next(pathlib.Path("repo_zip").rglob("ml_reliability/__init__.py"))\n'
            "    sys.path.insert(0, str(package_init.parent.parent))\n"
            "    # the repo's dependencies (numpy, pandas, scikit-learn, scipy, statsmodels) are\n"
            "    # preinstalled on Colab; otherwise: !pip install -q numpy pandas scikit-learn scipy statsmodels\n"
            "\n"
            "import ml_reliability\n"
            'print("ml_reliability loaded from:", pathlib.Path(ml_reliability.__file__).parent)'
        ),
    ]


def closing(interpretation: str, limitations: str) -> list[dict]:
    return [
        md(
            "## Interpretation and limitations\n\n"
            f"**How to read the output.** {interpretation}\n\n"
            f"**Limitations.** {limitations}\n\n"
            "These notebooks are methodological demonstrations on synthetic data and small bundled "
            "scikit-learn datasets. They do not show that any method improves performance on your "
            "data; that requires a prespecified evaluation on independent data."
        )
    ]


def write_notebook(name: str, cells: list[dict]) -> Path:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "colab": {"name": f"{name}.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    path = HERE / f"{name}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


BUILDERS: dict = {}


def register(name: str):
    def wrap(function):
        BUILDERS[name] = function
        return function

    return wrap


# ==== NOTEBOOKS-START ====
@register("overview")
def overview() -> list[dict]:
    cells = [
        header(
            "overview",
            "ReliableML overview: a statistical workflow for machine-learning reliability",
            "overview",
            "It walks through the three stages of the paper - data partitioning, "
            "meta-learning, and post-deployment evidence - with one small end-to-end run.",
        )
    ]
    cells += setup_cells(
        "# Optional: ATTP (S-ATTP partitioning, used in stage 1 when available)\n"
        "!pip install -q i-attp==0.2.3"
    )
    cells += [
        md(
            r"""## 1. The workflow and its key statistics

Let $\mathcal D=\{(x_i,y_i)\}_{i=1}^n$ be labeled development data and $P_{\mathrm{ref}}$ the feature distribution represented at fit time. The paper separates **what needs labels** from **what does not**:

| Evidence | Needs | API |
|---|---|---|
| Feature shift (PSI, KL, KS, ...) | unlabeled batch + fitted reference | `predict_with_report`, `assess_reliability` |
| Predictive evaluation, calibration | labeled representative batch | `evaluate` |
| Group rates (fairness) | labels + protected groups + positive class | `evaluate(sensitive_features=...)` |
| Deterioration | $\ge 3$ dated labeled batches | `assess_deterioration` |
| Explanation | background data + target class | `explain` |
| Reason audit | labeled holdout + background | `audit_reasons` |

**Cross-fitted meta-learning.** With $K$ stratified folds and $M$ base estimators, row $i$ receives out-of-fold class probabilities $\hat p_{im}^{(-k(i))}$ and
$$Z_i=\big[\hat p_{i1,1}^{(-k(i))},\ldots,\hat p_{i1,C}^{(-k(i))},\ldots,\hat p_{iM,1}^{(-k(i))},\ldots,\hat p_{iM,C}^{(-k(i))}\big],\qquad \hat p(y\mid x)=g\big(Z(x)\big),$$
where $g$ is fitted to $(Z_i,y_i)$ and, at inference, base models refitted on all training rows produce $Z(x)$.

**Reference-based drift.** With reference proportions $p_b$ and current proportions $q_b$ (smoothed by $\epsilon>0$),
$$\operatorname{PSI}(p,q)=\sum_b (q_b-p_b)\log\frac{q_b}{p_b},\qquad D_{\mathrm{KL}}(p\Vert q)=\sum_b p_b\log\frac{p_b}{q_b},\qquad S_{\mathrm{display}}=\exp(-\overline{\operatorname{PSI}}).$$
$S_{\mathrm{display}}$ is a descriptive transform, **not** a probability of correctness.

**Multiclass MCC** from a confusion matrix with $t_k$ true counts, $p_k$ predicted counts, $c=\sum_k C_{kk}$, $s=\sum_{ij}C_{ij}$:
$$\operatorname{MCC}=\frac{cs-\sum_k t_kp_k}{\sqrt{(s^2-\sum_k p_k^2)(s^2-\sum_k t_k^2)}}.$$

**Group gaps.** For a rate $r_a$ (selection rate, TPR, FPR, PPV) over groups $a\in\mathcal A$: $\Delta_r=\max_a r_a-\min_a r_a$.

**Deterioration.** For dated batches with accuracy $a_j$ at time $t_j$: $\hat\beta_1=\frac{\sum_j (t_j-\bar t)(a_j-\bar a)}{\sum_j (t_j-\bar t)^2}$; deterioration is flagged only if the bootstrap interval for $\hat\beta_1$ lies wholly below $0$.

**Shapley attribution.** $\phi_j(f,x)=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(d-|S|-1)!}{d!}\,[v_x(S\cup\{j\})-v_x(S)]$, estimated by Monte Carlo permutations."""
        ),
        md(
            "## 2. Data and stage 1: keep a final evaluation set, then partition\n\n"
            "We use scikit-learn's breast-cancer dataset (a bundled data source) with six numeric "
            "features. A final evaluation set is held out *first*; the (optional) ATTP partition is "
            "built from development data only, because S-ATTP chooses its split using its own test "
            "score and is therefore not an unbiased final estimate."
        ),
        code(
            """
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_reliability import (
    MetaPipelineClassifier,
    ReliablePipeline,
    assess_deterioration,
)

SEED = 42
FEATURES = ["mean radius", "mean texture", "mean smoothness",
            "mean compactness", "mean symmetry", "mean fractal dimension"]
data = load_breast_cancer(as_frame=True)
X_all, y_all = data.data[FEATURES], data.target
print("rows:", len(X_all), "| class counts:", y_all.value_counts().to_dict())

# Untouched final evaluation set, separated before any partitioning or model selection.
X_dev, X_final, y_dev, y_final = train_test_split(
    X_all, y_all, test_size=0.3, stratify=y_all, random_state=SEED)

try:
    import attp  # optional companion package (pip install i-attp==0.2.3)
    part = attp.s_attp_clf(X_dev, y_dev, random_state=SEED)
    X_train, y_train, X_part_test, y_part_test = part
    print(f"S-ATTP: split={part.best_split}, clusters={part.best_n_clusters}, "
          f"LDA comps={part.n_lda_components}, internal test score={part.test_score:.3f} "
          "(a tuning criterion, not a final estimate)")
except ImportError:
    X_train, X_part_test, y_train, y_part_test = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED)
    print("ATTP not installed -> plain stratified split of the development data")
print("train rows:", len(X_train), "| partition-test rows:", len(X_part_test), "| final rows:", len(X_final))
"""
        ),
        md(
            "## 3. Stage 2: fit a cross-fitted meta-learner inside `ReliablePipeline`\n\n"
            "`ReliablePipeline.fit` fits a `DriftMonitor` on the training features (the reference), "
            "then fits the estimator, and saves a reference-distribution plot."
        ),
        code(
            """
base = [
    ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ("tree", DecisionTreeClassifier(max_depth=3, random_state=SEED)),
]
model = ReliablePipeline(
    MetaPipelineClassifier(base_estimators=base, n_folds=3, random_state=SEED),
    drift_threshold=0.2,
    plot_dir="reliability_plots",
    bias_variance_folds=3,
    bias_variance_repeats=2,
).fit(X_train, y_train)

print("meta-classifier:", type(model.estimator.meta_classifier_).__name__)
fig, _ = model.plot_reference_distributions()
plt.show()
"""
        ),
        md(
            "## 4. Stage 3a: unlabeled monitoring at prediction time\n\n"
            "A copy of the final set is shifted (`mean radius` scaled by 1.3, plus noise) to mimic a "
            "population change. No labels are used here."
        ),
        code(
            """
rng = np.random.default_rng(SEED)
X_shifted = X_final.copy()
X_shifted["mean radius"] = X_shifted["mean radius"] * 1.3 + rng.normal(0, 0.5, len(X_shifted))

rows = []
for name, batch in [("final (same population)", X_final), ("shifted copy", X_shifted)]:
    _, report = model.predict_with_report(batch)
    rows.append({
        "batch": name,
        "similarity exp(-mean PSI)": round(report.similarity_score, 3),
        "mean PSI": round(report.mean_psi, 3),
        "covariate shift flagged": report.covariate_shift_detected,
        "drifted features": ", ".join(report.drifted_features) or "-",
    })
pd.DataFrame(rows)
"""
        ),
        code(
            """
fig, _ = model.plot_distributions(X_shifted, feature_names=["mean radius", "mean texture"])
plt.show()
"""
        ),
        md(
            "## 5. Stage 3b: labeled analyses once outcomes arrive\n\n"
            "`evaluate` adds classification metrics (including MCC), probability calibration, "
            "statistical tests and a repeated-CV Brier decomposition. The protected attribute below "
            "is **synthetic** (a texture split, passed separately from the model features) purely to "
            "show the report layout."
        ),
        code(
            """
group = np.where(X_final["mean texture"] > X_final["mean texture"].median(), "high_texture", "low_texture")
evaluation = model.evaluate(X_final, y_final, sensitive_features={"texture_group": group}, positive_label=1)

clf = evaluation.classification
print("accuracy:", round(clf["accuracy"], 3), "| balanced accuracy:", round(clf["balanced_accuracy"], 3),
      "| MCC:", round(clf["matthews_correlation_coefficient"], 3))
print("confusion matrix (rows=true, cols=pred):", clf["labels"], clf["confusion_matrix"])
print("Brier:", round(clf["probability_calibration"]["mean_classwise_brier_score"], 4),
      "| ECE:", round(clf["probability_calibration"]["top_label_expected_calibration_error"], 4))
bv = evaluation.bias_variance
print("empirical Brier decomposition: bias^2 =", round(bv["bias_squared"], 4),
      "| variance =", round(bv["prediction_variance"], 4))

fair = evaluation.fairness["texture_group"]["classes"]["1"]
pd.DataFrame(fair["per_group"]).round(3)
"""
        ),
        code(
            """
print("statistical-parity gap:", round(fair["statistical_parity"]["gap"], 3),
      "CI", [round(v, 3) for v in fair["statistical_parity"]["confidence_interval"]])
print("equalized-odds max gap:", round(fair["equalized_odds"]["maximum_gap"], 3))
print("predictive-parity gap :", round(fair["predictive_parity"]["gap"], 3))
"""
        ),
        md(
            "### Deterioration over dated batches (simulated drift)\n\n"
            "Six monthly batches are resampled from the final set with a growing shift in "
            "`mean radius`; accuracy of the fitted model on each batch is regressed on time."
        ),
        code(
            """
history = []
dates = pd.date_range("2026-01-01", periods=6, freq="MS")
for step, date in enumerate(dates):
    idx = rng.choice(len(X_final), size=120, replace=True)
    batch = X_final.iloc[idx].copy()
    batch["mean radius"] = batch["mean radius"] * (1 + 0.12 * step)
    history.append({"timestamp": date.date().isoformat(),
                    "y_true": y_final.iloc[idx].to_numpy(),
                    "y_pred": model.estimator.predict(batch)})
det = assess_deterioration(history, bootstrap_samples=500)
print("accuracy per batch:", [round(float(o["accuracy"]), 3) for o in det.observations])
print("slope (accuracy/day):", f"{det.accuracy_slope:.5f}",
      "| 95% CI:", [f"{v:.5f}" for v in det.slope_confidence_interval],
      "| deterioration flagged:", det.deterioration_detected)
"""
        ),
        md(
            "### Explanation and reason audit (brief)\n\n"
            "Both need a probability-capable classifier and finite numeric features. Sample counts are "
            "kept small here for speed; increase `num_samples`/`explanation_samples` for real use."
        ),
        code(
            """
explanation = model.explain(X_final, class_label=1, max_instances=15, num_samples=100)
print(pd.DataFrame(explanation.global_shap_importance).round(4).to_string(index=False))
al = explanation.reason_alignment
print("mean Spearman agreement SHAP vs LIME:", round(al["mean_absolute_importance_spearman"], 3),
      "| 95% CI", [round(v, 3) for v in al["mean_spearman_confidence_interval"]])

audit = model.audit_reasons(X_final, y_final, positive_label=1, bootstrap_samples=100,
                            explanation_samples=100, max_explained_instances=15)
pv = audit.predictive_validity
print("audit accuracy:", round(pv["accuracy"], 3), "CI", [round(v, 3) for v in pv["accuracy_confidence_interval"]],
      "| Brier:", round(pv["brier_score"], 3), "| ECE:", round(pv["top_label_expected_calibration_error"], 3))
pd.DataFrame(audit.feature_reliance)[["feature", "loss_increase", "loss_increase_confidence_interval", "positive_lower_bound"]]
"""
        ),
    ]
    cells += closing(
        "Drift output is about the *inputs* only: a high similarity or a missing flag says nothing about "
        "accuracy. Accuracy, MCC, calibration, group gaps, deterioration slope, explanations and audit "
        "reliance each answer a different question and need the data listed in the table in section 1. "
        "Intervals are bootstrap intervals that assume independent rows.",
        "Explanation agreement is not correctness or causality; fairness gaps are descriptive and the "
        "group here is synthetic; the deterioration run is a simulation with a linear trend; the "
        "S-ATTP internal score is a tuning criterion. Pickled artifacts can execute code when loaded - "
        "load only trusted files. No integrated benchmark or superiority claim is made.",
    )
    return cells


@register("statistical-data-cleaning")
def cleaning() -> list[dict]:
    cells = [
        header(
            "statistical-data-cleaning",
            "Leakage-aware statistical data cleaning",
            "statistical-data-cleaning",
            "It demonstrates `StatisticalDataCleaner` / `clean_statistical_data`: fit-time imputation "
            "rules, IQR outlier diagnostics, the three intensities, and fold-safe use in a pipeline.",
        )
    ]
    cells += setup_cells()
    cells += [
        md(
            r"""## 1. Rules implemented

For a numeric feature $X_j$ let $\mathcal O_j$ be its observed values.

* **Binary** (at most two distinct observed values): impute the observed **mode**.
* **Approximately normal** (heuristic, *not* a hypothesis test): at least 10 observations, $|\text{mean}-\text{median}|\le 0.1\,\sigma$ (population SD), $|\text{skew}|<0.5$ and $|\text{excess kurtosis}|<1.5$: impute the **mean** $\hat\mu_j=|\mathcal O_j|^{-1}\sum_{x\in\mathcal O_j}x$.
* **Otherwise**: impute the **median** $\hat m_j=\operatorname{median}(\mathcal O_j)$.
* **Categorical**: observed mode; an all-missing categorical column gets the sentinel `__reliableml_missing__`.
* **All-missing numeric**: a warning, fill with $0$, and the feature is listed as uninformative.

**IQR fences** (Tukey), learned on fit data only:
$$IQR_j=Q_{3j}-Q_{1j},\qquad L_j=Q_{1j}-k\,IQR_j,\qquad U_j=Q_{3j}+k\,IQR_j,$$
with $k=1.5$ (light, moderate) or $k=1.0$ (heavy). **Light** imputes only and *flags* values outside $[L_j,U_j]$; **moderate/heavy** also clip to $[L_j,U_j]$. The bounds are reused unchanged in `transform`.

**Leakage control.** Inside cross-validation the cleaner must be refit on each training fold:
$$\widehat f^{(-v)}=\operatorname{Fit}\!\big(\widehat T^{(-v)}(X^{(-v)}_{\text{train}}),y^{(-v)}_{\text{train}}\big),\qquad \widehat y^{(-v)}_{\text{valid}}=\widehat f^{(-v)}\!\big(\widehat T^{(-v)}(X^{(-v)}_{\text{valid}})\big).$$"""
        ),
        md("## 2. Synthetic data with every case the rules distinguish"),
        code(
            """
import warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from ml_reliability import StatisticalDataCleaner, clean_statistical_data

rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({
    "symmetric": rng.normal(50, 5, n),            # approx. normal -> mean
    "contaminated": rng.normal(50, 5, n),         # same law + injected outliers -> heuristic rejects normality
    "skewed": rng.lognormal(3, 0.8, n),            # right-skewed -> median
    "binary": rng.integers(0, 2, n).astype(float), # <=2 values -> mode
    "city": rng.choice(["north", "south", "west"], n, p=[0.5, 0.3, 0.2]),
    "all_missing": np.full(n, np.nan),             # unidentifiable numeric
})
df.loc[rng.choice(n, 8, replace=False), "contaminated"] = 120.0   # injected outliers
for col in ["symmetric", "contaminated", "skewed", "binary", "city"]:
    df.loc[rng.random(n) < 0.10, col] = np.nan                 # ~10% missing completely at random

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    cleaned, cleaner = clean_statistical_data(df, cleaning_intensity="light", return_cleaner=True)
print("warnings:", [str(w.message) for w in caught])
print("remaining NaNs after cleaning:", int(cleaned.isna().sum().sum()))
pd.DataFrame({"missing at fit": cleaner.missing_counts_, "fill value": cleaner.fill_values_}).astype(str)
"""
        ),
        code(
            """
report = cleaner.get_cleaning_report()
for key, value in report.items():
    print(f"{key}: {value}")

observed = df["symmetric"].dropna()
print()
print("symmetric: mean=%.3f median=%.3f -> fill=%.3f (mean rule)" % (observed.mean(), observed.median(), cleaner.fill_values_["symmetric"]))
observed = df["contaminated"].dropna()
print("contaminated: mean=%.3f median=%.3f -> fill=%.3f (heuristic rejects normality -> median)" % (observed.mean(), observed.median(), cleaner.fill_values_["contaminated"]))
observed = df["skewed"].dropna()
print("skewed   : mean=%.3f median=%.3f -> fill=%.3f (median rule)" % (observed.mean(), observed.median(), cleaner.fill_values_["skewed"]))
"""
        ),
        md(
            "## 3. Intensity: light only flags, moderate/heavy clip\n\n"
            "Observed minimum/maximum of the `contaminated` column (which contains injected extreme values) "
            "after each mode, with the fit-time fences $L_j, U_j$."
        ),
        code(
            """
rows = []
for intensity in ["light", "moderate", "heavy"]:
    out, fitted = clean_statistical_data(df, cleaning_intensity=intensity, return_cleaner=True)
    low, high = fitted.outlier_bounds_["contaminated"]
    rows.append({"intensity": intensity, "L": round(low, 2), "U": round(high, 2),
                 "min after": round(out["contaminated"].min(), 2), "max after": round(out["contaminated"].max(), 2),
                 "capping applied": fitted.get_cleaning_report()["outlier_capping_applied"]})
display_table = pd.DataFrame(rows)
print(display_table.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharex=True)
for ax, intensity in zip(axes, ["light", "moderate", "heavy"]):
    out, fitted = clean_statistical_data(df, cleaning_intensity=intensity, return_cleaner=True)
    ax.hist(out["contaminated"], bins=30, color="#4c78a8")
    low, high = fitted.outlier_bounds_["contaminated"]
    ax.axvline(low, color="crimson", ls="--"); ax.axvline(high, color="crimson", ls="--")
    ax.set_title(f"{intensity} mode"); ax.set_xlabel("contaminated")
plt.tight_layout(); plt.show()
"""
        ),
        md(
            "## 4. Fit-time statistics are reused on new data (no refitting)\n\n"
            "The cleaner is fitted on a training split; the test split is imputed with the *training* "
            "values, and a batch with a different column order is rejected."
        ),
        code(
            """
train, test = df.iloc[:300].copy(), df.iloc[300:].copy()
cleaner_tr = StatisticalDataCleaner().fit(train)
test_clean = cleaner_tr.transform(test)
print("fill value learned on train for 'skewed':", round(cleaner_tr.fill_values_["skewed"], 4))
print("test-set median of 'skewed' (NOT used) :", round(test["skewed"].median(), 4))
filled = test["skewed"].isna()
print("values written into missing test cells:", sorted(set(np.round(test_clean.loc[filled, "skewed"], 4))))
try:
    cleaner_tr.transform(test[test.columns[::-1]])
except ValueError as error:
    print("schema check ->", error)
"""
        ),
        md(
            "## 5. Bundled dataset: cleaning inside cross-validation\n\n"
            "Breast-cancer features get ~10% missing-completely-at-random values. Placing the cleaner "
            "*inside* the scikit-learn `Pipeline` refits it on every training fold; the per-fold "
            "imputation value for one feature shows that each fold learns its own statistics."
        ),
        code(
            """
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

bc = load_breast_cancer(as_frame=True)
Xb = bc.data.iloc[:, :10].copy()
yb = bc.target
mask_rng = np.random.default_rng(1)
Xb = Xb.mask(mask_rng.random(Xb.shape) < 0.10)
print("fraction missing:", round(float(Xb.isna().mean().mean()), 3))

pipe = Pipeline([
    ("clean", StatisticalDataCleaner("light")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, Xb, yb, cv=cv, scoring="accuracy")
print("5-fold accuracy:", np.round(scores, 3), "| mean:", round(scores.mean(), 3))

fold_fill = []
for train_idx, _ in cv.split(Xb, yb):
    fitted = StatisticalDataCleaner().fit(Xb.iloc[train_idx])
    fold_fill.append(fitted.fill_values_["mean radius"])
print("per-fold fill value for 'mean radius':", np.round(fold_fill, 4))
"""
        ),
    ]
    cells += closing(
        "The cleaning report lists fit-time missing counts and IQR outlier *flags*; a flag is a prompt to "
        "investigate, not a deletion instruction. Compare the three intensities to see what capping "
        "changes before choosing it. Per-fold fill values differ because each fold fits its own cleaner, "
        "which is what keeps validation rows from influencing imputation.",
        "Single-value imputation understates uncertainty and can distort correlations; the normality "
        "decision is a heuristic that is unstable for small samples; IQR fences are univariate and can "
        "overflag skewed data; capping compresses legitimate extremes (hence light is the default); "
        "zero-fill of an all-missing column is a software fallback, not an estimate. The transformer does "
        "no encoding, scaling, multiple imputation or missingness-mechanism modeling.",
    )
    return cells


@register("attp-method")
def attp_method() -> list[dict]:
    cells = [
        header(
            "attp-method",
            "S-ATTP: cluster-stratified train/test partitioning and its validation limits",
            "attp-method",
            "S-ATTP lives in the companion package ATTP (`i-attp`), which ReliableML recommends for "
            "partitioning. This notebook uses the real `attp.s_attp_clf` API and then reproduces its "
            "LDA -> K-means -> strata recipe to expose the selection and leakage issues the paper discusses.",
        )
    ]
    cells += setup_cells(
        "# ATTP is a separate package (the S-ATTP classifier partitioner used in this notebook)\n"
        "!pip install -q i-attp==0.2.3"
    )
    cells += [
        md(
            r"""## 1. Method

Let $X\in\mathbb R^{n\times p}$ be a finite numeric matrix and $y_i\in\{1,\dots,C\}$.

**Supervised projection (LDA).** With class means $\bar x_c$, global mean $\bar x$ and class sizes $n_c$,
$$S_W=\sum_{c}\sum_{i:y_i=c}(x_i-\bar x_c)(x_i-\bar x_c)^\top,\qquad S_B=\sum_c n_c(\bar x_c-\bar x)(\bar x_c-\bar x)^\top,\qquad J(w)=\frac{w^\top S_Bw}{w^\top S_Ww}.$$
At most $\min(p,C-1)$ discriminant components exist; components are added until the cumulative explained variance reaches a threshold (default $0.85$), giving $Z=XW$.

**Clusters and joint strata.** For each candidate $k$, K-means minimizes $J_k=\sum_{j=1}^k\sum_{i:z_i\in C_j}\lVert z_i-\mu_j\rVert_2^2$. The stratum of row $i$ is $S_i=(y_i,c_i)$. Rows in singleton strata cannot appear in both partitions and are always assigned to **training**.

**Candidate selection.** For each train fraction $\alpha$ and cluster count $k$ a classifier $h_{\alpha,k}$ is fitted on the training part and scored on the candidate test part:
$$(\alpha^\star,k^\star)=\arg\max_{(\alpha,k)\in\mathcal A}\widehat Q_{\mathrm{test}}(h_{\alpha,k}).$$

**Why the returned test score is not a final estimate.** LDA and K-means see *all* rows supplied to the call before any split exists, and the maximum over candidates is itself selected on the test part (selection optimism). A defensible protocol holds out an untouched outer set *first* and runs everything else, including `s_attp_clf`, on the remaining data."""
        ),
        md("## 2. Run S-ATTP on a 3-class dataset and inspect the result\n\nScikit-learn's wine dataset (13 numeric features, 3 classes) is a bundled data source here; it gives 2 LDA components, which can be plotted."),
        code(
            """
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import attp
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, load_breast_cancer, make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

wine = load_wine(as_frame=True)
Xw, yw = wine.data, wine.target
part = attp.s_attp_clf(Xw, yw, splits=(0.7, 0.8), min_clusters=2, max_clusters=8, random_state=0)

print("best train fraction :", part.best_split)
print("best #clusters      :", part.best_n_clusters)
print("LDA components      :", part.n_lda_components, "| explained variance ratio:", np.round(part.explained_variance_ratio, 3),
      "| total:", round(part.total_explained_variance, 3))
print("train/test rows     :", len(part.X_train), "/", len(part.X_test))
print("internal scores     : train=%.3f test=%.3f" % (part.train_score, part.test_score))
part.metrics.head(8)
"""
        ),
        md("The `metrics` table lists every feasible candidate $(\\alpha,k)$; the winner is simply the row with the largest test score."),
        code(
            """
metrics = part.metrics
print("candidates evaluated:", len(metrics))
print("max candidate test score : %.3f" % metrics["test_score"].max())
print("mean candidate test score: %.3f" % metrics["test_score"].mean())
print("spread (max - min)       : %.3f" % (metrics["test_score"].max() - metrics["test_score"].min()))
"""
        ),
        md(
            "## 3. Reproduce the recipe: LDA projection, clusters and strata\n\n"
            "The same scikit-learn calls the package uses (LDA with the selected number of components, "
            "then K-means with the selected $k$) are re-run to draw the projection and count singleton strata."
        ),
        code(
            """
lda = LinearDiscriminantAnalysis(n_components=part.n_lda_components).fit(Xw, yw)
Z = lda.transform(Xw)
clusters = KMeans(n_clusters=part.best_n_clusters, random_state=0, n_init="auto").fit_predict(Z)
strata = pd.crosstab(pd.Series(yw.to_numpy(), name="class"), pd.Series(clusters, name="cluster"))
print(strata)
sizes = pd.Series(list(zip(yw.to_numpy(), clusters))).value_counts()
singletons = int((sizes == 1).sum())
print("singleton strata (forced into training):", singletons)

in_train = Xw.index.isin(part.train_indices)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
zy = Z[:, 1] if Z.shape[1] > 1 else np.zeros(len(Z))
axes[0].scatter(Z[:, 0], zy, c=yw, cmap="viridis", s=18); axes[0].set_title("LDA projection coloured by class")
axes[1].scatter(Z[:, 0], zy, c=clusters, cmap="tab10", s=18)
axes[1].scatter(Z[~in_train, 0], zy[~in_train], facecolors="none", edgecolors="k", s=60, label="partition test")
axes[1].set_title(f"K-means clusters (k={part.best_n_clusters}); circles = test rows"); axes[1].legend()
for ax in axes: ax.set_xlabel("LD1"); ax.set_ylabel("LD2")
plt.tight_layout(); plt.show()

print("class proportions  all/train/test:",
      np.round(yw.value_counts(normalize=True).sort_index().to_numpy(), 3),
      np.round(part.y_train.value_counts(normalize=True).sort_index().to_numpy(), 3),
      np.round(part.y_test.value_counts(normalize=True).sort_index().to_numpy(), 3))
"""
        ),
        md(
            "## 4. Outer-evaluation protocol on a bundled binary dataset\n\n"
            "Breast cancer: hold out an **untouched outer test set first**, run `s_attp_clf` on the "
            "remaining development data only, fit the final model on the ATTP training rows, and "
            "evaluate once on the outer set. Repeating over several outer splits compares ATTP's "
            "internal (selected) test score with the honest outer score."
        ),
        code(
            """
bc = load_breast_cancer(as_frame=True)
Xb, yb = bc.data, bc.target
rows = []
for seed in range(8):
    X_dev, X_outer, y_dev, y_outer = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=seed)
    result = attp.s_attp_clf(X_dev, y_dev, splits=(0.7, 0.8), min_clusters=2, max_clusters=6, random_state=seed)
    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(result.X_train, result.y_train)
    rows.append({"outer seed": seed, "k*": result.best_n_clusters,
                 "ATTP internal test score": result.test_score,
                 "outer test accuracy": final.score(X_outer, y_outer)})
outer = pd.DataFrame(rows)
outer["internal - outer"] = outer["ATTP internal test score"] - outer["outer test accuracy"]
print(outer.round(3).to_string(index=False))
print("mean internal - outer difference: %.3f (sd %.3f); note the two scores use different classifiers" %
      (outer["internal - outer"].mean(), outer["internal - outer"].std()))
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(outer["outer seed"], outer["ATTP internal test score"], "o-", label="ATTP internal test score")
ax.plot(outer["outer seed"], outer["outer test accuracy"], "s-", label="untouched outer test accuracy")
ax.set_xlabel("outer split seed"); ax.set_ylabel("accuracy"); ax.legend(); plt.tight_layout(); plt.show()
"""
        ),
        md(
            "## 5. Synthetic data: a rare class creates singleton strata\n\n"
            "With a small minority class and many clusters, some class-cluster cells contain one row; "
            "those rows are forced into training, so the realized split can depart from the nominal fraction."
        ),
        code(
            """
Xs, ys = make_classification(n_samples=300, n_features=6, n_informative=4, n_redundant=0,
                             weights=[0.93, 0.07], class_sep=1.0, random_state=3)
Xs = pd.DataFrame(Xs, columns=[f"f{i}" for i in range(6)]); ys = pd.Series(ys)
res = attp.s_attp_clf(Xs, ys, splits=(0.8,), min_clusters=6, max_clusters=10, random_state=0)

lda_s = LinearDiscriminantAnalysis(n_components=res.n_lda_components).fit(Xs, ys)
cl = KMeans(n_clusters=res.best_n_clusters, random_state=0, n_init="auto").fit_predict(lda_s.transform(Xs))
cell_sizes = pd.Series(list(zip(ys.to_numpy(), cl))).value_counts()
print("minority rows:", int((ys == 1).sum()), "| k* =", res.best_n_clusters, "| singleton strata:", int((cell_sizes == 1).sum()))
print("nominal train fraction: %.2f | realized train fraction: %.3f" % (res.best_split, len(res.X_train) / len(Xs)))
print("minority rows in test:", int((res.y_test == 1).sum()), "of", int((ys == 1).sum()))
"""
        ),
    ]
    cells += closing(
        "`SplitResult` exposes the chosen train fraction, cluster count, LDA components, explained variance, "
        "internal scores and the full candidate table. The internal test score is a *tuning criterion*; the "
        "outer-protocol table shows how it compares with scores on data that took no part in the search. "
        "The differences also depend on the classifier used inside ATTP versus the final model.",
        "LDA is supervised and linear, K-means prefers spherical Euclidean clusters and is scale-sensitive; "
        "inputs must be finite numeric (no imputation, encoding or scaling is done); singleton strata always "
        "go to training; a score-based choice can favor splits that are easy for the chosen classifier rather "
        "than representative of a target population. S-ATTP neither detects nor guarantees an out-of-distribution "
        "validation set, and temporal or grouped deployments need temporal/group outer splits. Only the S-ATTP "
        "classifier partitioner is covered; the wider ATTP taxonomy is out of scope.",
    )
    return cells


@register("distribution-shift")
def distribution_shift() -> list[dict]:
    cells = [
        header(
            "distribution-shift",
            "Reference-based distribution-shift monitoring",
            "distribution-shift",
            "It uses `DriftMonitor` (also embedded in `ReliablePipeline`) to compute PSI, KL, KS, "
            "Wasserstein, the moment-based DSI and categorical chi-square tests against a fit-time reference.",
        )
    ]
    cells += setup_cells()
    cells += [
        md(
            r"""## 1. Statistics

`DriftMonitor.fit` records, per feature, reference quantile bins (numeric) or categories, with extra buckets for underflow/overflow (numeric), unseen categories and missing values. With reference counts $n^{(r)}_b$ and current counts $n^{(c)}_b$ over $B$ buckets, smoothed normalized proportions $p_b,q_b$ (additive $\epsilon=10^{-6}$ per bucket) give
$$\operatorname{PSI}(p,q)=\sum_{b=1}^{B}(q_b-p_b)\log\frac{q_b}{p_b},\qquad D_{\mathrm{KL}}(p\Vert q)=\sum_{b=1}^{B}p_b\log\frac{p_b}{q_b}\ \text{(reference to current)}.$$
Drift flags use PSI against a configurable threshold (`drift_threshold`, default $0.2$) - a convention to be calibrated on your own data, not a universal cutoff.

**Numeric features.** Empirical two-sample KS and 1-D Wasserstein-1:
$$D_{n,m}=\sup_x\lvert\widehat F_{\mathrm{ref}}(x)-\widehat F_{\mathrm{cur}}(x)\rvert,\qquad W_1=\int_{\mathbb R}\lvert\widehat F_{\mathrm{ref}}(x)-\widehat F_{\mathrm{cur}}(x)\rvert\,dx.$$
The library-defined **Drift Score Index** (not a standard test) with mean $\mu$, SD $\sigma$, skewness $\gamma$, excess kurtosis $\kappa$:
$$\operatorname{DSI}^2=\Big(\tfrac{\mu_c-\mu_r}{s}\Big)^2+\Big(\tfrac{\sigma_c-\sigma_r}{s}\Big)^2+(\gamma_c-\gamma_r)^2+(\kappa_c-\kappa_r)^2,\quad s=\max\{\sigma_r,10^{-12}\max(1,\lvert\mu_r\rvert)\}.$$

**Categorical features.** Pearson chi-square homogeneity over reference categories plus an unseen bucket (p-value unreliable for small expected counts).

**Aggregate display** $S_{\mathrm{display}}=\exp(-\overline{\operatorname{PSI}})$ - descriptive, not a probability of correctness. Marginal stability cannot rule out concept drift $P(Y\mid X)$; the separate `assess_concept_shift` needs labels and compares conditional target distributions per feature with a weighted Jensen-Shannon divergence."""
        ),
        md("## 2. Synthetic reference and a first comparison"),
        code(
            """
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from ml_reliability import DriftMonitor

rng = np.random.default_rng(7)

def make_batch(n, *, shift=0.0, scale=1.0, cat_p=(0.5, 0.3, 0.2), extra_cat=0.0):
    categories = rng.choice(["a", "b", "c"], size=n, p=cat_p)
    if extra_cat:
        categories = np.where(rng.random(n) < extra_cat, "d", categories)  # unseen category
    return pd.DataFrame({
        "x_normal": rng.normal(shift, scale, n),
        "x_skewed": rng.lognormal(0, 0.6, n),
        "category": categories,
    })

reference = make_batch(3000)
monitor = DriftMonitor(n_bins=10, drift_threshold=0.2, bootstrap_samples=50, random_seed=0).fit(reference)

scenarios = {
    "no shift (fresh batch)": make_batch(600),
    "mean +0.5 sd": make_batch(600, shift=0.5),
    "variance x2": make_batch(600, scale=2.0),
    "category mix changed": make_batch(600, cat_p=(0.2, 0.3, 0.5)),
    "unseen category 'd' (15%)": make_batch(600, extra_cat=0.15),
}
reports = {name: monitor.compare(batch) for name, batch in scenarios.items()}
pd.DataFrame([{
    "scenario": name,
    "similarity exp(-mean PSI)": round(r.similarity_score, 3),
    "mean PSI": round(r.mean_psi, 3),
    "covariate shift flagged": r.covariate_shift_detected,
    "PSI-drifted features": ", ".join(r.drifted_features) or "-",
} for name, r in reports.items()])
"""
        ),
        code(
            """
def feature_table(report):
    return pd.DataFrame([{
        "feature": f.name, "PSI": f.psi, "KL(ref||cur)": f.kl_divergence, "KS": f.ks_statistic,
        "Wasserstein": f.wasserstein_distance, "DSI": f.dsi, "chi2 p": f.chi_square_p_value,
        "PSI flag": f.drifted, "DSI flag": f.dsi_drifted, "chi2 flag": f.chi_square_drifted,
    } for f in report.features]).round(4)

print("Scenario: mean +0.5 sd")
print(feature_table(reports["mean +0.5 sd"]).to_string(index=False))
print()
print("Scenario: unseen category 'd' (15%)")
print(feature_table(reports["unseen category 'd' (15%)"]).to_string(index=False))
ci = reports["mean +0.5 sd"].features[0].confidence_intervals
print("\\nbootstrap CIs for x_normal under +0.5 sd:", {k: tuple(round(v, 3) for v in pair) for k, pair in ci.items()})
"""
        ),
        md("## 3. Independent check of the formulas\n\nKS, Wasserstein and the moment score recomputed with SciPy/NumPy, and PSI/KL/chi-square recomputed by hand on the categorical feature, are compared with the report values."),
        code(
            """
batch = scenarios["mean +0.5 sd"]
report = reports["mean +0.5 sd"]
f = {x.name: x for x in report.features}

ks = stats.ks_2samp(reference["x_normal"], batch["x_normal"]).statistic
w1 = stats.wasserstein_distance(reference["x_normal"], batch["x_normal"])
def moments(v):
    v = np.asarray(v); return v.mean(), v.std(), stats.skew(v), stats.kurtosis(v)
mr, sr, gr, kr = moments(reference["x_normal"]); mc, sc, gc, kc = moments(batch["x_normal"])
s = max(sr, 1e-12 * max(1, abs(mr)))
dsi = np.sqrt(((mc - mr) / s) ** 2 + ((sc - sr) / s) ** 2 + (gc - gr) ** 2 + (kc - kr) ** 2)
print("KS          report=%.6f  scipy=%.6f" % (f["x_normal"].ks_statistic, ks))
print("Wasserstein report=%.6f  scipy=%.6f" % (f["x_normal"].wasserstein_distance, w1))
print("DSI         report=%.6f  manual=%.6f" % (f["x_normal"].dsi, dsi))

# categorical feature: buckets = a, b, c, unseen, missing; eps = 1e-6 smoothing
cat_report = {x.name: x for x in reports["category mix changed"].features}["category"]
cat_batch = scenarios["category mix changed"]["category"]
ref_counts = np.array([(reference["category"] == c).sum() for c in "abc"] + [0, 0], dtype=float)
cur_counts = np.array([(cat_batch == c).sum() for c in "abc"] + [0, 0], dtype=float)
eps = 1e-6
p = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
q = (cur_counts + eps) / (cur_counts.sum() + eps * len(cur_counts))
print("PSI         report=%.6f  manual=%.6f" % (cat_report.psi, np.sum((q - p) * np.log(q / p))))
print("KL(ref||cur) report=%.6f  manual=%.6f" % (cat_report.kl_divergence, np.sum(p * np.log(p / q))))
chi2, pval, _, _ = stats.chi2_contingency(np.vstack([ref_counts[:3], cur_counts[:3]]), correction=False)
print("chi-square  report=%.4f (p=%.3g)  scipy=%.4f (p=%.3g)" % (cat_report.chi_square_statistic, cat_report.chi_square_p_value, chi2, pval))
"""
        ),
        md("## 4. Plots: reference vs incoming"),
        code(
            """
fig, axes = monitor.plot_reference_distributions()
plt.show()
fig, axes = monitor.plot_distributions(scenarios["mean +0.5 sd"], report=reports["mean +0.5 sd"])
plt.show()
"""
        ),
        md(
            "## 5. Sensitivity and false alarms\n\n"
            "The paper stresses checking thresholds on in-domain batches and on designed shifts. Below, "
            "(a) PSI/KS/Wasserstein/DSI of `x_normal` versus the size of a mean shift, and (b) how often "
            "fresh in-domain batches trip the PSI flag."
        ),
        code(
            """
rows = []
for delta in np.arange(0.0, 2.01, 0.25):
    r = monitor.compare(make_batch(500, shift=delta))
    x = {m.name: m for m in r.features}["x_normal"]
    rows.append({"shift (sd)": delta, "PSI": x.psi, "KS": x.ks_statistic, "Wasserstein": x.wasserstein_distance, "DSI": x.dsi,
                 "similarity": r.similarity_score})
curve = pd.DataFrame(rows)
print(curve.round(3).to_string(index=False))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(curve["shift (sd)"], curve["PSI"], "o-", label="PSI"); axes[0].axhline(0.2, color="crimson", ls="--", label="threshold 0.2")
axes[0].plot(curve["shift (sd)"], curve["KS"], "s-", label="KS"); axes[0].set_xlabel("mean shift (sd)"); axes[0].legend()
axes[1].plot(curve["shift (sd)"], curve["similarity"], "o-"); axes[1].set_xlabel("mean shift (sd)"); axes[1].set_ylabel("exp(-mean PSI)")
plt.tight_layout(); plt.show()

flags = []
for _ in range(40):
    r = monitor.compare(make_batch(300))
    flags.append(r.covariate_shift_detected)
print("in-domain batches (n=300) flagged as covariate shift: %d of 40" % sum(flags))
"""
        ),
        md(
            "## 6. Bundled dataset and covariate vs concept shift\n\n"
            "Breast cancer: reference = a training split. A random test batch is compared with a "
            "*selected* batch (largest tumours only: covariate shift by selection). Then labels with "
            "unchanged inputs are corrupted to show that **concept shift can occur with no covariate shift**."
        ),
        code(
            """
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

bc = load_breast_cancer(as_frame=True)
cols = ["mean radius", "mean texture", "mean smoothness", "mean compactness"]
Xtr, Xte, ytr, yte = train_test_split(bc.data[cols], bc.target, test_size=0.4, stratify=bc.target, random_state=0)
bc_monitor = DriftMonitor(bootstrap_samples=50, concept_shift_threshold=0.1, random_seed=0).fit(Xtr, ytr)

selected = Xte[Xte["mean radius"] > Xte["mean radius"].quantile(0.7)]
for name, batch in [("random test batch", Xte), ("largest 30% by radius", selected)]:
    r = bc_monitor.assess_covariate_shift(batch)
    print(f"{name:24s} n={r.sample_count:3d} similarity={r.similarity_score:.3f} mean PSI={r.mean_psi:.3f} "
          f"flagged={r.covariate_shift_detected} drifted={list(r.drifted_features)}")
fig, axes = bc_monitor.plot_distributions(selected, feature_names=["mean radius", "mean texture"])
plt.show()

flip = np.random.default_rng(0).random(len(yte)) < 0.4
y_corrupted = np.where(flip, 1 - yte.to_numpy(), yte.to_numpy())
for name, labels in [("same labels", yte), ("40% of labels flipped", y_corrupted)]:
    concept = bc_monitor.assess_concept_shift(Xte, labels)
    cov = bc_monitor.assess_covariate_shift(Xte)
    print(f"{name:22s} covariate flagged={cov.covariate_shift_detected} | mean JS={concept.mean_js_divergence:.4f} "
          f"| shifted features={list(concept.shifted_features)}")
"""
        ),
    ]
    cells += closing(
        "PSI, KL, KS, Wasserstein and DSI measure how far the incoming marginals are from the reference; "
        "PSI drives the default flag, DSI and chi-square have their own flags, and `covariate_shift_detected` "
        "combines them. Bootstrap intervals describe row-resampling variability. A low shift score does not "
        "imply good accuracy and a high one does not imply poor accuracy - check labeled performance when "
        "outcomes arrive. The false-alarm count above is an example of calibrating the threshold on in-domain batches.",
        "Per-feature (marginal) statistics miss shifts that only affect feature interactions; the concept-shift "
        "check is univariate and needs labels, so it is retrospective; chi-square p-values need adequate "
        "expected counts; the intervals assume independent rows (not clustered or temporal data); the DSI is "
        "library-defined, not an inferential test; thresholds must be set with domain data. Reference age, "
        "schema, units and batch size all affect the numbers.",
    )
    return cells


@register("evaluation")
def evaluation() -> list[dict]:
    cells = [
        header(
            "evaluation",
            "Labeled evaluation: metrics, calibration, statistical tests, bias/variance",
            "evaluation",
            "It calls `ReliablePipeline.evaluate`, which adds label-dependent analyses to an ordinary "
            "prediction: confusion-matrix metrics (including MCC), probability calibration, ANOVA/MANOVA/"
            "chi-square tests and a repeated-CV Brier bias/variance decomposition.",
        )
    ]
    cells += setup_cells()
    cells += [
        md(
            r"""## 1. Metrics

With confusion matrix $C\in\mathbb N^{K\times K}$ ($C_{ij}$: true class $i$, predicted $j$), $s=\sum_{ij}C_{ij}$, $c=\sum_k C_{kk}$, $t_k=\sum_j C_{kj}$, $p_k=\sum_i C_{ik}$:
$$\operatorname{acc}=\frac{c}{s},\qquad \operatorname{BA}=\frac1K\sum_k\frac{C_{kk}}{t_k},\qquad
\operatorname{MCC}=\frac{c\,s-\sum_k p_kt_k}{\sqrt{(s^2-\sum_kp_k^2)(s^2-\sum_kt_k^2)}}.$$
Per class (one-vs-rest): precision, recall (TPR), specificity (TNR) and $F_1$, plus macro and support-weighted averages. MCC is $0$ for chance-level and can be low when accuracy is high under heavy imbalance.

**Calibration** (needs `predict_proba`): the multiclass Brier score $\frac1n\sum_i\sum_k(\hat p_{ik}-y_{ik})^2$ (class-wise mean reported), log loss, and the top-label expected calibration error over confidence bins $B_m$:
$$\operatorname{ECE}=\sum_m\frac{\lvert B_m\rvert}{n}\,\bigl\lvert\operatorname{acc}(B_m)-\operatorname{conf}(B_m)\bigr\rvert.$$

**Association tests** (descriptive, assumption-dependent): per numeric feature one-way ANOVA across labels, $F=\frac{\mathrm{SS}_B/(K-1)}{\mathrm{SS}_W/(n-K)}$; a joint MANOVA with Wilks' $\Lambda=\det(E)/\det(E+H)$; chi-square for categorical features. The same tests compare training and test cohorts.

**Bias/variance.** Repeated stratified CV gives out-of-fold probability vectors $\hat p^{(r)}_i$ over repeats $r$. With the one-hot label $y_i$ and mean prediction $\bar p_i$:
$$\underbrace{\overline{\lVert\bar p_i-y_i\rVert^2}}_{\text{bias}^2}+\underbrace{\overline{\tfrac1R\textstyle\sum_r\lVert\hat p^{(r)}_i-\bar p_i\rVert^2}}_{\text{variance}}=\text{mean squared probability error}.$$
This is an *empirical* decomposition of Brier error, not a population irreducible-noise estimate."""
        ),
        md("## 2. Binary example: breast cancer"),
        code(
            """
import json, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from ml_reliability import ReliablePipeline

warnings.filterwarnings("ignore", category=UserWarning)
data = load_breast_cancer(as_frame=True)
X = data.data[["mean radius", "mean texture", "mean smoothness", "mean compactness", "mean symmetry"]]
y = data.target  # 0 = malignant, 1 = benign
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)

pipe = ReliablePipeline(
    RandomForestClassifier(n_estimators=100, random_state=0),
    plot_dir="reliability_plots", bias_variance_folds=3, bias_variance_repeats=3, random_seed=0,
).fit(Xtr, ytr)
report = pipe.evaluate(Xte, yte, positive_label=1)
result = report.to_dict()
cls = result["classification"]
print("n =", result["sample_count"], "| labels:", cls["labels"])
print({k: round(cls[k], 4) for k in ["accuracy", "balanced_accuracy", "matthews_correlation_coefficient"]})
print(pd.DataFrame(cls["per_class"]).round(3).to_string(index=False))
"""
        ),
        code(
            """
cm = np.array(cls["confusion_matrix"])
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks(range(len(cm)), cls["labels"]); axes[0].set_yticks(range(len(cm)), cls["labels"])
axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true"); axes[0].set_title("Confusion matrix")
for i in range(len(cm)):
    for j in range(len(cm)):
        axes[0].text(j, i, cm[i, j], ha="center", va="center")

cal = result["classification"]["probability_calibration"]
bins = pd.DataFrame(cal["confidence_bins"])
bins = bins[bins["sample_count"] > 0]
axes[1].plot([0, 1], [0, 1], "k--", label="perfect")
axes[1].scatter(bins["mean_confidence"], bins["accuracy"], s=20 + 4 * bins["sample_count"], label="bins (size ~ count)")
axes[1].set_xlabel("mean confidence"); axes[1].set_ylabel("accuracy"); axes[1].set_title("Reliability diagram")
axes[1].legend()
plt.tight_layout(); plt.show()
print({k: round(v, 4) for k, v in cal.items() if isinstance(v, float)})
"""
        ),
        md("### Verify the headline metrics against sklearn and the formulas"),
        code(
            """
pred = pipe.predict(Xte)
print("MCC report :", round(cls["matthews_correlation_coefficient"], 6))
print("MCC sklearn:", round(matthews_corrcoef(yte, pred), 6))
s = cm.sum(); c = np.trace(cm); t = cm.sum(axis=1); p = cm.sum(axis=0)
print("MCC formula:", round((c * s - (p * t).sum()) / np.sqrt((s**2 - (p**2).sum()) * (s**2 - (t**2).sum())), 6))
print("accuracy    :", round(cls["accuracy"], 6), "vs", round(c / s, 6))
"""
        ),
        md("## 3. Association tests and bias/variance"),
        code(
            """
tests = result["statistical_tests"]
anova = pd.DataFrame(tests["feature_target_associations"]["anova_numeric_features"])
print(anova[["feature", "statistic", "p_value", "group_sizes"]].round(4).to_string(index=False))
print("MANOVA (features vs label):")
print(json.dumps(tests["feature_target_associations"]["manova_numeric_features"], indent=1, default=str)[:900])
print("\\nTrain-vs-test cohort ANOVA (a split-balance check):")
cohort = pd.DataFrame(tests["training_vs_test_cohorts"]["anova_numeric_features"])
print(cohort[["feature", "statistic", "p_value"]].round(4).to_string(index=False))
bv = result["bias_variance"]
print("\\nBias/variance:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bv.items() if k in
      ["bias_squared", "prediction_variance", "mean_squared_probability_error", "folds", "repeats"]})
"""
        ),
        md(
            "## 4. Multiclass example: wine\n\n"
            "Multiclass tasks do not take `positive_label`; per-class metrics are one-vs-rest."
        ),
        code(
            """
wine = load_wine(as_frame=True)
Xw = wine.data[["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "flavanoids"]]
Xwtr, Xwte, ywtr, ywte = train_test_split(Xw, wine.target, test_size=0.4, random_state=0, stratify=wine.target)
wine_pipe = ReliablePipeline(LogisticRegression(max_iter=2000), plot_dir="reliability_plots",
                             bias_variance_folds=3, bias_variance_repeats=3, random_seed=0).fit(Xwtr, ywtr)
wine_res = wine_pipe.evaluate(Xwte, ywte).to_dict()["classification"]
print("labels:", wine_res["labels"])
print(np.array(wine_res["confusion_matrix"]))
print({k: round(wine_res[k], 4) for k in ["accuracy", "balanced_accuracy", "matthews_correlation_coefficient"]})
print("MCC sklearn:", round(matthews_corrcoef(ywte, wine_pipe.predict(Xwte)), 4))
print(pd.DataFrame(wine_res["per_class"]).round(3).to_string(index=False))
"""
        ),
        md("## 5. Why accuracy alone misleads: synthetic imbalanced data"),
        code(
            """
rng = np.random.default_rng(1)
n = 3000
Xs = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"x{i}" for i in range(4)])
logit = -4.0 + 1.2 * Xs["x0"] + 0.8 * Xs["x1"]  # about 3% positives, weak signal
ys = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
print("positive rate:", round(ys.mean(), 4))
Xstr, Xste, ystr, yste = train_test_split(Xs, ys, test_size=0.4, random_state=0, stratify=ys)
imb = ReliablePipeline(LogisticRegression(max_iter=1000), plot_dir="reliability_plots",
                       bias_variance_folds=3, bias_variance_repeats=3, random_seed=0).fit(Xstr, ystr)
imb_cls = imb.evaluate(Xste, yste, positive_label=1).to_dict()["classification"]
print({k: round(imb_cls[k], 4) for k in ["accuracy", "balanced_accuracy", "matthews_correlation_coefficient"]})
print("confusion matrix:\\n", np.array(imb_cls["confusion_matrix"]))
print("majority-class baseline accuracy:", round(1 - yste.mean(), 4))
"""
        ),
    ]
    cells += closing(
        "Accuracy, balanced accuracy and MCC answer different questions; the imbalanced example shows accuracy "
        "close to the majority-class baseline while recall for the rare class can be poor. The reliability "
        "diagram compares confidence with observed accuracy bin by bin (sparse bins are noisy). Treat ANOVA/"
        "MANOVA p-values as screening summaries. The bias/variance numbers are an empirical Brier "
        "decomposition from out-of-fold predictions on the evaluated batch, so with a single repeat the "
        "variance term is zero by construction.",
        "Metrics on a small test set carry substantial sampling error (no intervals are shown for the point "
        "metrics here); ANOVA/MANOVA assume independence, normality and equal variances, and are not tested; "
        "multiple tests are not corrected; ECE depends on binning; cross-validation refits clones of the "
        "estimator on the evaluated batch, so the decomposition describes the learning procedure, not the "
        "fitted model; none of it is causal.",
    )
    return cells


@register("fairness")
def fairness() -> list[dict]:
    cells = [
        header(
            "fairness",
            "Group fairness diagnostics with uncertainty",
            "fairness",
            "It uses the `sensitive_features` argument of `ReliablePipeline.evaluate` to report per-group "
            "selection, error and precision rates, max-minus-min gaps and row-bootstrap intervals.",
        )
    ]
    cells += setup_cells()
    cells += [
        md(
            r"""## 1. Quantities

For a positive class and groups $g\in\mathcal G$ with true label $Y$, prediction $\hat Y$ and sensitive attribute $A$ (never used as a model input here; it is only used to *audit* the predictions):
$$\mathrm{SR}_g=P(\hat Y=1\mid A=g),\quad \mathrm{TPR}_g=P(\hat Y=1\mid Y=1,A=g),\quad \mathrm{FPR}_g=P(\hat Y=1\mid Y=0,A=g),\quad \mathrm{PPV}_g=P(Y=1\mid \hat Y=1,A=g).$$
Gaps are max-minus-min across groups, $\Delta(m)=\max_g m_g-\min_g m_g$:
$$\text{statistical parity: }\Delta(\mathrm{SR}),\qquad \text{equalized odds: }\max\{\Delta(\mathrm{TPR}),\Delta(\mathrm{FPR})\},\qquad \text{predictive parity: }\Delta(\mathrm{PPV}).$$
Zero means equal *observed* rates. Each gap gets a within-group row-bootstrap percentile interval; groups with an empty denominator (for example no actual negatives) have undefined rates and are reported as such. Multiclass problems are evaluated one-vs-rest per class.

**Criteria conflict.** If base rates $P(Y=1\mid A=g)$ differ, an imperfect classifier cannot in general satisfy statistical parity, equalized odds and predictive parity at once; which gap matters is a policy choice, not a statistical one. Small gaps in a small group can be pure noise - the intervals and the denominators are part of the result."""
        ),
        md(
            "## 2. Synthetic data with different base rates\n\n"
            "The group `A` is a purely illustrative synthetic attribute. The model never sees it, but the "
            "feature distribution and the base rate differ by group, so error rates can differ anyway."
        ),
        code(
            """
import warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ml_reliability import ReliablePipeline

warnings.filterwarnings("ignore", category=UserWarning)
rng = np.random.default_rng(3)
n = 4000
A = rng.choice(["g1", "g2"], size=n, p=[0.6, 0.4])
x0 = rng.normal(size=n)
x1 = rng.normal(size=n) + np.where(A == "g2", 0.8, 0.0)  # proxy feature shifted by group
logit = -0.2 + 1.0 * x0 + 0.5 * x1 - np.where(A == "g2", 0.8, 0.0)  # different base rate
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X = pd.DataFrame({"x0": x0, "x1": x1, "x2": rng.normal(size=n)})
print(pd.Series(y).groupby(A).mean().round(3).rename("base rate P(Y=1|A)"))

Xtr, Xte, ytr, yte, Atr, Ate = train_test_split(X, y, A, test_size=0.4, random_state=0, stratify=y)
pipe = ReliablePipeline(LogisticRegression(max_iter=1000), plot_dir="reliability_plots",
                        bias_variance_folds=3, bias_variance_repeats=1, bootstrap_samples=200, random_seed=0).fit(Xtr, ytr)
ev = pipe.evaluate(Xte, yte, sensitive_features={"A": Ate}, positive_label=1).to_dict()
fair = ev["fairness"]["A"]["classes"]["1"]
groups = pd.DataFrame(fair["per_group"])
print(groups.round(3).to_string(index=False))
"""
        ),
        code(
            """
def gap_row(name, block):
    low, high = block["confidence_interval"]
    return {"criterion": name, "gap": round(block["gap"], 3), "95% CI": f"[{low:.3f}, {high:.3f}]"}

eo = fair["equalized_odds"]
table = pd.DataFrame([
    gap_row("statistical parity (SR)", fair["statistical_parity"]),
    gap_row("equalized odds: TPR", eo["true_positive_rate_gap"]),
    gap_row("equalized odds: FPR", eo["false_positive_rate_gap"]),
    gap_row("predictive parity (PPV)", fair["predictive_parity"]),
])
print(table.to_string(index=False))
print("equalized-odds maximum gap:", round(eo["maximum_gap"], 3))

fig, ax = plt.subplots(figsize=(7, 3.8))
metrics = ["selection_rate", "true_positive_rate", "false_positive_rate", "positive_predictive_value"]
width = 0.8 / len(groups)
for i, row in groups.iterrows():
    ax.bar(np.arange(len(metrics)) + i * width, [row[m] for m in metrics], width, label=f"{row['group']} (n={row['sample_count']})")
ax.set_xticks(np.arange(len(metrics)) + width / 2, ["SR", "TPR", "FPR", "PPV"]); ax.set_ylim(0, 1)
ax.set_title("Per-group rates"); ax.legend(); plt.show()
"""
        ),
        md("### Verify the rates by hand (pandas)"),
        code(
            """
pred = pipe.predict(Xte)
frame = pd.DataFrame({"A": Ate, "y": yte, "pred": pred})
manual = frame.groupby("A").apply(lambda d: pd.Series({
    "n": len(d), "SR": (d.pred == 1).mean(),
    "TPR": ((d.pred == 1) & (d.y == 1)).sum() / (d.y == 1).sum(),
    "FPR": ((d.pred == 1) & (d.y == 0)).sum() / (d.y == 0).sum(),
    "PPV": ((d.pred == 1) & (d.y == 1)).sum() / (d.pred == 1).sum()}), include_groups=False)
print(manual.round(3))
"""
        ),
        md(
            "## 3. Real data: breast cancer with an illustrative group\n\n"
            "The group below (tumour size above/below the median) is **not** a protected attribute; it only "
            "demonstrates the API on a bundled dataset and the effect of small denominators (few actual "
            "negatives in one group make FPR undefined or very noisy). Two attributes can be audited in one call."
        ),
        code(
            """
bc = load_breast_cancer(as_frame=True)
Xb = bc.data[["mean radius", "mean texture", "mean smoothness", "mean compactness", "mean symmetry"]]
yb = bc.target
Xbtr, Xbte, ybtr, ybte = train_test_split(Xb, yb, test_size=0.4, random_state=0, stratify=yb)
size = np.where(Xbte["mean radius"] > Xbte["mean radius"].median(), "large", "small")
texture = np.where(Xbte["mean texture"] > Xbte["mean texture"].median(), "rough", "smooth")
bpipe = ReliablePipeline(RandomForestClassifier(100, random_state=0), plot_dir="reliability_plots",
                         bias_variance_folds=3, bias_variance_repeats=1, bootstrap_samples=200, random_seed=0).fit(Xbtr, ybtr)
bres = bpipe.evaluate(Xbte, ybte, sensitive_features={"size": size, "texture": texture}, positive_label=1).to_dict()["fairness"]
for attribute, block in bres.items():
    cls = block["classes"]["1"]
    print(f"--- attribute: {attribute}")
    print(pd.DataFrame(cls["per_group"])[["group", "sample_count", "actual_positive_count", "actual_negative_count",
          "selection_rate", "true_positive_rate", "false_positive_rate"]].round(3).to_string(index=False))
    print("parity gap", round(cls["statistical_parity"]["gap"], 3), "CI", [round(v, 3) for v in cls["statistical_parity"]["confidence_interval"]],
          "| equalized-odds max gap", round(cls["equalized_odds"]["maximum_gap"], 3))
"""
        ),
        md("## 4. Multiclass (wine), one-vs-rest per class"),
        code(
            """
wine = load_wine(as_frame=True)
Xw = wine.data[["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "flavanoids"]]
Xwtr, Xwte, ywtr, ywte = train_test_split(Xw, wine.target, test_size=0.5, random_state=0, stratify=wine.target)
batch = np.where(Xwte["alcohol"] > Xwte["alcohol"].median(), "high_alcohol", "low_alcohol")
wpipe = ReliablePipeline(LogisticRegression(max_iter=2000), plot_dir="reliability_plots",
                         bias_variance_folds=3, bias_variance_repeats=1, bootstrap_samples=200, random_seed=0).fit(Xwtr, ywtr)
wres = wpipe.evaluate(Xwte, ywte, sensitive_features={"alcohol_group": batch}).to_dict()["fairness"]["alcohol_group"]
for label, cls in wres["classes"].items():
    print(f"class {label}: parity gap = {cls['statistical_parity']['gap']:.3f}, "
          f"equalized-odds max gap = {cls['equalized_odds']['maximum_gap']}")
"""
        ),
    ]
    cells += closing(
        "Read the per-group table first: the denominators (sample, actual-positive, actual-negative counts) "
        "tell you how much each rate can be trusted. The intervals around each gap show whether an observed "
        "disparity is distinguishable from sampling noise; the synthetic example shows that excluding the "
        "sensitive attribute from the model does not remove group differences when a proxy feature and "
        "different base rates exist, and that the criteria give gaps of different size (they measure different things).",
        "Gaps are observational and descriptive: they do not identify causes, justify a legal or ethical "
        "conclusion, or tell you which fairness criterion should apply. Intervals are within-group row "
        "bootstraps (independent rows assumed), no multiplicity correction is applied across criteria, "
        "attributes or classes, and small or empty cells give undefined or unstable rates. Intersectional "
        "groups are not formed automatically. Labels may themselves be biased.",
    )
    return cells


@register("deterioration")
def deterioration() -> list[dict]:
    cells = [
        header(
            "deterioration",
            "Accuracy deterioration across dated labeled batches",
            "deterioration",
            "It uses `assess_deterioration` (also `ReliablePipeline.assess_deterioration`): per-batch accuracy, "
            "an OLS slope on time, a within-batch bootstrap interval and a conservative flag.",
        )
    ]
    cells += setup_cells()
    cells += [
        md(
            r"""## 1. Method

For labeled batches $j=1,\dots,J$ ordered by timestamp $t_j$ with $n_j$ rows, accuracy is $a_j=\frac1{n_j}\sum_i\mathbf 1\{\hat y_{ji}=y_{ji}\}$. Ordinary least squares fits (each batch weighted equally)
$$a_j=\beta_0+\beta_1t_j+\varepsilon_j,\qquad \hat\beta_1=\frac{\sum_j(t_j-\bar t)(a_j-\bar a)}{\sum_j(t_j-\bar t)^2}.$$
ISO date strings are converted to days, so $\beta_1$ is "accuracy change per day"; numeric timestamps keep their own unit. At least three distinct timestamps are required, and numeric and ISO timestamps may not be mixed.

**Within-batch bootstrap.** For replicate $b=1,\dots,B$ resample $n_j$ rows *with replacement inside every batch*, recompute $a^{(b)}_j$, refit and store $\hat\beta_1^{(b)}$. The percentile interval at confidence $1-\alpha$ is
$$\bigl[\,q_{\alpha/2}(\hat\beta_1^{(b)}),\;q_{1-\alpha/2}(\hat\beta_1^{(b)})\,\bigr].$$
The conservative flag is $\texttt{deterioration\_detected}=\mathbf 1\{q_{1-\alpha/2}<0\}$: the whole interval must be below zero. The regression $p$-value is the ordinary slope test and is reported separately; it ignores within-batch sampling error. This is a descriptive first-order trend, not a change-point or causal test."""
        ),
        md("## 2. Synthetic dated batches: drifting model vs stable control"),
        code(
            """
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from ml_reliability import assess_deterioration

rng = np.random.default_rng(11)
dates = pd.date_range("2025-01-01", periods=10, freq="7D").strftime("%Y-%m-%d").tolist()

def make_history(accuracy_by_batch, n=200):
    history = []
    for date, acc in zip(dates, accuracy_by_batch):
        y_true = rng.integers(0, 2, n)
        correct = rng.random(n) < acc
        history.append({"timestamp": date, "y_true": y_true, "y_pred": np.where(correct, y_true, 1 - y_true)})
    return history

decline = np.linspace(0.92, 0.80, len(dates))  # generating accuracy falls 0.12 over 63 days
stable = np.full(len(dates), 0.88)
reports = {
    "declining": assess_deterioration(make_history(decline), bootstrap_samples=1000, random_seed=0),
    "stable": assess_deterioration(make_history(stable), bootstrap_samples=1000, random_seed=0),
}
for name, rep in reports.items():
    low, high = rep.slope_confidence_interval
    print(f"{name:10s} slope/day={rep.accuracy_slope:+.5f}  95% CI=[{low:+.5f}, {high:+.5f}]  "
          f"p={rep.p_value:.4f}  deterioration_detected={rep.deterioration_detected}")
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, (name, rep) in zip(axes, reports.items()):
    days = np.array([(pd.Timestamp(o["timestamp"]) - pd.Timestamp(dates[0])).days for o in rep.observations])
    acc = np.array([float(o["accuracy"]) for o in rep.observations])
    fit = stats.linregress(days, acc)
    ax.plot(days, acc, "o", label="batch accuracy")
    ax.plot(days, fit.intercept + fit.slope * days, "-", label="OLS fit")
    ax.set_title(f"{name}: slope {rep.accuracy_slope:+.4f}/day, CI {tuple(round(v, 4) for v in rep.slope_confidence_interval)}", fontsize=9)
    ax.set_xlabel("days since first batch"); ax.legend()
axes[0].set_ylabel("accuracy"); plt.tight_layout(); plt.show()
"""
        ),
        md("### Verify the slope and p-value against scipy"),
        code(
            """
rep = reports["declining"]
days = np.array([(pd.Timestamp(o["timestamp"]) - pd.Timestamp(dates[0])).days for o in rep.observations])
acc = np.array([float(o["accuracy"]) for o in rep.observations])
fit = stats.linregress(days, acc)
print("slope  report:", round(rep.accuracy_slope, 8), " scipy:", round(fit.slope, 8))
print("p      report:", round(rep.p_value, 8), " scipy:", round(fit.pvalue, 8))
print(pd.DataFrame([{"timestamp": o["timestamp"], "n": o["sample_count"], "accuracy": float(o["accuracy"])} for o in rep.observations]).round(3).to_string(index=False))
"""
        ),
        md(
            "### Input requirements\n\n"
            "Numeric timestamps work too (slope per unit). Fewer than three distinct dates, or mixed "
            "timestamp types, raise a clear error rather than returning a meaningless slope."
        ),
        code(
            """
numeric = [{"timestamp": i, "y_true": h["y_true"], "y_pred": h["y_pred"]} for i, h in enumerate(make_history(decline))]
num_rep = assess_deterioration(numeric, bootstrap_samples=500, random_seed=0)
print("numeric time: slope per unit =", round(num_rep.accuracy_slope, 5))

for label, bad in {"two dates only": make_history(decline)[:2],
                   "mixed timestamp types": [dict(make_history(decline)[0], timestamp=0)] + make_history(decline)[1:]}.items():
    try:
        assess_deterioration(bad, bootstrap_samples=100)
    except Exception as error:
        print(f"{label}: {type(error).__name__}: {error}")
"""
        ),
        md(
            "## 3. Real data: breast cancer ordered into pseudo-batches\n\n"
            "Breast cancer has no timestamps. To exercise the API on a bundled dataset we sort the test rows "
            "by a feature and cut them into ordered pseudo-batches. **This is not real time**: it shows how "
            "performance changes along that feature, and the output must not be read as temporal deterioration."
        ),
        code(
            """
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

bc = load_breast_cancer(as_frame=True)
Xb = bc.data[["mean radius", "mean texture", "mean smoothness", "mean compactness", "mean symmetry"]]
Xtr, Xte, ytr, yte = train_test_split(Xb, bc.target, test_size=0.5, random_state=0, stratify=bc.target)
model = RandomForestClassifier(100, random_state=0).fit(Xtr, ytr)
order = np.argsort(Xte["mean texture"].to_numpy())
pseudo = []
for k, idx in enumerate(np.array_split(order, 6)):
    rows = Xte.iloc[idx]
    pseudo.append({"timestamp": k, "y_true": yte.iloc[idx].to_numpy(), "y_pred": model.predict(rows)})
bc_rep = assess_deterioration(pseudo, bootstrap_samples=1000, random_seed=0)
print("pseudo-batch accuracies:", [round(float(o["accuracy"]), 3) for o in bc_rep.observations], "| n per batch:", [o["sample_count"] for o in bc_rep.observations])
print("slope", round(bc_rep.accuracy_slope, 4), "CI", tuple(round(v, 4) for v in bc_rep.slope_confidence_interval),
      "p", round(bc_rep.p_value, 3), "flag", bc_rep.deterioration_detected)
"""
        ),
    ]
    cells += closing(
        "The slope is the estimated accuracy change per unit time; the interval is the plausible range given "
        "within-batch sampling noise, and the flag fires only if the entire interval is negative. The stable "
        "control shows what 'no detected deterioration' looks like with the same batch sizes, and the pseudo-"
        "batch example shows that a slope can be produced by any ordering, which is why the ordering must be "
        "real time to be interpreted as deterioration.",
        "Accuracy needs delayed labels; the linear trend can miss abrupt breaks, seasonality or recovery; "
        "batches are weighted equally regardless of size; the bootstrap assumes independent rows within a "
        "batch (autocorrelated or repeated patients violate this); the regression p-value ignores within-batch "
        "error and is shown only for reference; a declining trend may reflect case-mix, policy or label "
        "changes rather than model decay; and with few batches the power to detect a decline is low.",
    )
    return cells


@register("meta-learning")
def meta_learning() -> list[dict]:
    cells = [
        header(
            "meta-learning",
            "Leakage-aware stacking and statistical meta-learners",
            "meta-learning",
            "It covers `MetaFeatureTransformer`, `MetaPipelineClassifier`, `PerfMetaSelect`, "
            "`CalibratedStackedClassifier`, the `StatisticalMeta*` classifiers and `halving_grid_search`.",
        )
    ]
    cells += setup_cells()
    cells += [
        md(
            r"""## 1. Methods

**Out-of-fold (OOF) meta-features.** With $K$-fold stratified CV and base learners $f_1,\dots,f_M$, row $i$ in validation fold $k$ gets class-probability meta-features from models that never saw it:
$$z_{im}=\bigl(\hat p_{f_m^{(-k)}}(Y=c\mid x_i)\bigr)_{c\in\mathcal C},\qquad z_i=[z_{i1},\dots,z_{iM}].$$
The meta-estimator $g$ is fit on $\{(z_i,y_i)\}$. At prediction time each $f_m$ is refit on all training rows and $\hat p(x)=g([\hat p_{f_1}(x),\dots,\hat p_{f_M}(x)])$. Class columns are aligned across folds. Feeding $g$ *in-sample* base probabilities instead would let it trust overfit learners.

**Performance selection** (`PerfMetaSelect`): $\bar s_m=\frac1K\sum_k s(f_m^{(-k)},D_k)$ and $m^\star=\arg\max_m\bar s_m$ with the best learner refit on all data. The CV score $\bar s_{m^\star}$ is optimistically biased as an estimate of future performance because it is the maximum of several noisy scores - use an outer test set.

**Calibrated stacking** (`CalibratedStackedClassifier`): the meta-estimator is wrapped in `CalibratedClassifierCV` (sigmoid or isotonic).

**Statistical meta-learners** (binary only): LDA projections and KMeans cluster distances $d_k(x)$ are turned into features, e.g. $\psi_k(x)=\exp(-\gamma_kd_k(x)^2)$, optionally with trees and degree-2 interactions, then
$$P(Y=1\mid x)=\sigma(\beta_0+\beta^\top u(x)),\qquad \ell(\beta)=-\sum_i\{y_i\log p_i+(1-y_i)\log(1-p_i)\}+\lambda\lVert\beta_{-0}\rVert_2^2.$$
The NN variant averages $B$ Monte-Carlo-dropout passes, $\hat p=\frac1B\sum_b\sigma(f_{\theta,m_b}(x))$ - a stochastic summary, not a calibrated uncertainty interval.

`feature_importances` / `local_feature_importances` report raw-feature importance as the change in predicted probability when a feature is replaced by a baseline (a model-behaviour probe, not causality)."""
        ),
        md(
            "## 2. Data and an honest comparison protocol\n\n"
            "All models are compared on an untouched outer test split. Breast cancer (binary) is the main "
            "dataset; a synthetic non-linear task shows where stacking can help."
        ),
        code(
            """
import time, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml_reliability import (
    CalibratedStackedClassifier, MetaFeatureTransformer, MetaPipelineClassifier, PerfMetaSelect,
    StatisticalMetaDTreeClassifier, StatisticalMetaLDAClassifier, StatisticalMetaNNClassifier,
    halving_grid_search,
)

warnings.filterwarnings("ignore")

def base_learners():
    return [
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("forest", RandomForestClassifier(n_estimators=100, random_state=0)),
        ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))),
    ]

def compare(models, Xtr, ytr, Xte, yte):
    rows = []
    for name, model in models.items():
        start = time.time()
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        rows.append({"model": name, "AUC": roc_auc_score(yte, proba), "Brier": brier_score_loss(yte, proba),
                     "accuracy": accuracy_score(yte, model.predict(Xte)), "fit seconds": time.time() - start})
    return pd.DataFrame(rows).set_index("model")

bc = load_breast_cancer(as_frame=True)
X, y = bc.data, bc.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print("train", Xtr.shape, "test", Xte.shape)
"""
        ),
        code(
            """
models = {f"base: {name}": est for name, est in base_learners()}
models["MetaPipeline (OOF, LR meta)"] = MetaPipelineClassifier(base_estimators=base_learners(), n_folds=5, random_state=0)
models["PerfMetaSelect"] = PerfMetaSelect(estimators=base_learners(), cv=5, scoring="roc_auc", random_state=0)
models["CalibratedStacked"] = CalibratedStackedClassifier(base_estimators=base_learners(), n_folds=5, random_state=0)
models["StatisticalMeta-LDA"] = StatisticalMetaLDAClassifier(n_clusters=4, n_mc_sims=5, cv=3, random_state=0)
models["StatisticalMeta-DTree"] = StatisticalMetaDTreeClassifier(n_clusters=6, n_mc_sims=5, cv=3, random_state=0)
models["StatisticalMeta-NN"] = StatisticalMetaNNClassifier(n_clusters=6, epochs=20, n_mc_sims=5, cv=3, random_state=0)
results = compare(models, Xtr, ytr, Xte, yte)
print(results.round(4).to_string())
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
results["AUC"].plot.barh(ax=axes[0], title="Test AUC"); results["Brier"].plot.barh(ax=axes[1], title="Test Brier (lower is better)")
axes[0].set_xlim(results["AUC"].min() - 0.01, 1.0); plt.tight_layout(); plt.show()
print("PerfMetaSelect selected:", models["PerfMetaSelect"].best_name_, {k: round(v, 4) for k, v in models["PerfMetaSelect"].cv_scores_.items()})
"""
        ),
        md(
            "With ~170 test rows, differences of a few thousandths in AUC are within sampling noise; this table "
            "is a smoke test of the API and protocol, not a ranking of methods."
        ),
        md("## 3. Why out-of-fold matters: a leakage demonstration"),
        code(
            """
# Noisy labels (20% flipped): a deep forest memorizes the training set, a linear model cannot.
Xn, yn = make_classification(n_samples=1500, n_features=12, n_informative=5, flip_y=0.2, class_sep=1.0, random_state=2)
Xntr, Xnte, yntr, ynte = train_test_split(Xn, yn, test_size=0.4, random_state=0, stratify=yn)
learners = [("forest", RandomForestClassifier(n_estimators=100, random_state=0)),
            ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))]

Z_oof = MetaFeatureTransformer(estimators=learners, cv=5, random_state=0).fit_transform(Xntr, yntr)
fitted = [clone(estimator).fit(Xntr, yntr) for _, estimator in learners]
Z_in = np.hstack([m.predict_proba(Xntr) for m in fitted])
Z_test = np.hstack([m.predict_proba(Xnte) for m in fitted])
print("forest in-sample accuracy:", round(accuracy_score(yntr, fitted[0].predict(Xntr)), 3),
      "| forest test accuracy:", round(accuracy_score(ynte, fitted[0].predict(Xnte)), 3))
print("meta-feature columns: forest P(0), forest P(1), logreg P(0), logreg P(1)")
for name, Z in [("meta-learner on IN-SAMPLE features", Z_in), ("meta-learner on OOF features", Z_oof)]:
    meta = LogisticRegression(max_iter=1000).fit(Z, yntr)
    proba = meta.predict_proba(Z_test)[:, 1]
    print(f"{name:36s} coef={np.round(meta.coef_[0], 2)} test AUC={roc_auc_score(ynte, proba):.4f} Brier={brier_score_loss(ynte, proba):.4f}")
"""
        ),
        md("## 4. Inspect meta-features, importances and tuning"),
        code(
            """
mp = models["MetaPipeline (OOF, LR meta)"]
Z = mp.transform_meta_features(Xte)
print("meta-feature matrix for test rows:", Z.shape, "(M learners x 2 classes)")
imp = mp.feature_importances(Xte.iloc[:40], top_n=5)
print(pd.DataFrame(imp["features"]).round(4).to_string(index=False))
"""
        ),
        code(
            """
search = halving_grid_search(
    MetaPipelineClassifier(base_estimators=base_learners(), n_folds=3, random_state=0),
    {"meta_classifier": [LogisticRegression(C=0.1, max_iter=2000), LogisticRegression(C=1.0, max_iter=2000),
                         LogisticRegression(C=10.0, max_iter=2000)]},
    cv=3, scoring="roc_auc", random_state=0, factor=2, min_resources="exhaust",
).fit(Xtr, ytr)
print("best meta-classifier:", search.best_params_["meta_classifier"], "| CV AUC", round(search.best_score_, 4))
print("outer test AUC of the tuned model:", round(roc_auc_score(yte, search.predict_proba(Xte)[:, 1]), 4))
"""
        ),
        md("## 5. Synthetic non-linear task and a multiclass example"),
        code(
            """
Xs, ys = make_classification(n_samples=1500, n_features=12, n_informative=6, n_redundant=2, n_clusters_per_class=3,
                             flip_y=0.05, class_sep=0.8, random_state=1)
Xs = pd.DataFrame(Xs, columns=[f"f{i}" for i in range(12)])
Xstr, Xste, ystr, yste = train_test_split(Xs, ys, test_size=0.3, random_state=0, stratify=ys)
synth = {"logreg": base_learners()[0][1], "forest": base_learners()[1][1],
         "MetaPipeline": MetaPipelineClassifier(base_estimators=base_learners(), n_folds=5, random_state=0),
         "StatisticalMeta-LDA": StatisticalMetaLDAClassifier(n_clusters=6, n_mc_sims=5, cv=3, random_state=0)}
print(compare(synth, Xstr, ystr, Xste, yste).round(4).to_string())

wine = load_wine(as_frame=True)
Xwtr, Xwte, ywtr, ywte = train_test_split(wine.data, wine.target, test_size=0.3, random_state=0, stratify=wine.target)
selector = PerfMetaSelect(estimators=base_learners(), cv=5, scoring="accuracy", random_state=0).fit(Xwtr, ywtr)
stack = MetaPipelineClassifier(base_estimators=base_learners(), n_folds=5, random_state=0).fit(Xwtr, ywtr)
print("wine (3 classes) | PerfMetaSelect picked", selector.best_name_, "acc", round(accuracy_score(ywte, selector.predict(Xwte)), 4),
      "| MetaPipeline acc", round(accuracy_score(ywte, stack.predict(Xwte)), 4))
"""
        ),
    ]
    cells += closing(
        "Compare methods only on data none of them touched. OOF meta-features keep the meta-learner honest about "
        "how reliable each base learner is on unseen rows; the leakage cell shows how in-sample probabilities of "
        "a flexible learner look nearly perfect and mislead the meta-learner. Selection scores and tuning "
        "scores are optimistic; calibrated stacking changes probability quality (Brier), not necessarily ranking.",
        "Stacking is not guaranteed to beat the best base learner, particularly on small data (a few test rows "
        "decide the ranking); extra computation scales with folds x learners; the statistical meta-learners are "
        "binary only and have many hyper-parameters; Monte-Carlo averaging is not an uncertainty interval; "
        "OOF folds assume exchangeable rows (use grouped or temporal splits for clustered or time-ordered data); "
        "feature importances probe model behaviour under baseline replacement and are not causal.",
    )
    return cells


@register("explainability")
def explainability() -> list[dict]:
    cells = [
        header(
            "explainability",
            "Explanation consistency: SHAP-like attributions, LIME-like surrogates and alignment",
            "explainability",
            "It uses `explain_estimator` (or `ReliablePipeline.explain`): permutation-sampled Shapley "
            "attributions, local ridge surrogates, and rank/sign/top-feature agreement between them.",
        )
    ]
    cells += setup_cells()
    cells += [
        md(
            r"""## 1. Quantities

For a model output $f$ (here the predicted probability of the explained class), $d$ features $F$ and a value function $v_x(S)$ that sets features in $S$ to the instance's values and the rest to background draws, the Shapley value is
$$\phi_j(f,x)=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(d-|S|-1)!}{d!}\,\bigl[v_x(S\cup\{j\})-v_x(S)\bigr].$$
ReliableML *estimates* it by Monte-Carlo permutation sampling (`num_samples` permutations, so results carry sampling error and depend on the background data). With a single fixed baseline the efficiency property is $\sum_j\phi_j=f(x)-f(\text{baseline})$.

**LIME-like surrogate.** Perturbations $z$ are drawn from marginal Gaussians built from the background, weighted by a distance kernel $\pi_x(z)$, and a ridge-stabilized weighted linear model is fitted:
$$g_x^\star=\arg\min_{g\in\mathcal G}\sum_{z\in\mathcal Z}\pi_x(z)\bigl(f(z)-g(z)\bigr)^2+\Omega(g),\qquad \pi_x(z)=\exp\!\bigl(-(\lVert z-x\rVert/\sigma)^2\bigr).$$
The weighted local $R^2$ ("fidelity") says how well the surrogate fits its *synthetic* neighbourhood.

**Global importance and alignment.** $I_j=\frac1m\sum_{i=1}^m\lvert\hat\phi_j(f,x_i)\rvert$. Between the SHAP-like and LIME-like vectors the report gives per-row Spearman rank correlation of absolute importances, top-feature overlap and sign agreement, and a bootstrap interval for the mean Spearman. These measure *agreement between two explanation procedures*, not correctness of the explanation or evidence of causation."""
        ),
        md("## 2. Breast cancer: global and local explanations"),
        code(
            """
import os, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml_reliability import explain_estimator

warnings.filterwarnings("ignore")
bc = load_breast_cancer(as_frame=True)
features = ["mean radius", "mean texture", "mean smoothness", "mean compactness", "mean concavity", "mean symmetry"]
X, y = bc.data[features], bc.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model = RandomForestClassifier(n_estimators=150, random_state=0).fit(Xtr, ytr)

report = explain_estimator(model, Xte, y_true=yte, background_data=Xtr.sample(100, random_state=0), class_label=1,
                           max_instances=30, num_samples=300, random_seed=0, bootstrap_samples=200)
global_importance = pd.DataFrame(report.global_shap_importance)
print(global_importance.round(4).to_string(index=False))
print("alignment:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report.reason_alignment.items()})
print("warnings:", report.warnings)
"""
        ),
        code(
            """
local = report.local_explanations[0]
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in local.items()
       if k in ["row_index", "prediction", "predicted_probability_for_class", "lime_local_fidelity_r_squared",
                "absolute_importance_spearman", "top_feature_overlap", "sign_agreement"]})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
global_importance.sort_values("mean_absolute_shapley_value").plot.barh(x="feature", y="mean_absolute_shapley_value", ax=axes[0], legend=False)
axes[0].set_title("Global: mean |SHAP|")
compare = pd.DataFrame({"SHAP": local["shap_values"], "LIME": local["lime_weights"]})
compare.plot.barh(ax=axes[1]); axes[1].set_title(f"Local explanation, row {local['row_index']}"); axes[1].axvline(0, color="k", lw=0.5)
plt.tight_layout(); plt.show()
"""
        ),
        md(
            "### Efficiency check\n\n"
            "The Shapley values of one instance should add up to $f(x)$ minus the average prediction over the "
            "background (approximately, because background rows and permutations are sampled)."
        ),
        code(
            """
background = Xtr.sample(100, random_state=0)
mean_background = model.predict_proba(background)[:, list(model.classes_).index(1)].mean()
rows = []
for local in report.local_explanations[:8]:
    total = sum(local["shap_values"].values())
    rows.append({"row": local["row_index"], "f(x)": local["predicted_probability_for_class"],
                 "f(x) - mean f(background)": local["predicted_probability_for_class"] - mean_background, "sum of SHAP": total})
print(pd.DataFrame(rows).round(3).to_string(index=False))
"""
        ),
        md(
            "## 3. Synthetic model with known ground truth\n\n"
            "A logistic data-generating process with coefficients $(2,-1,0.5,0,0)$ on standardized features "
            "lets us compare the recovered global ranking with the truth."
        ),
        code(
            """
rng = np.random.default_rng(5)
n = 3000
coef = np.array([2.0, -1.0, 0.5, 0.0, 0.0])
Xs = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
ys = (rng.random(n) < 1 / (1 + np.exp(-(Xs.to_numpy() @ coef)))).astype(int)
Xstr, Xste, ystr, yste = train_test_split(Xs, ys, test_size=0.3, random_state=0)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xstr, ystr)

true_importance = dict(zip(Xs.columns, np.abs(coef)))
rep_s = explain_estimator(lr, Xste, background_data=Xstr.sample(200, random_state=0), class_label=1, max_instances=40,
                          num_samples=300, random_seed=0, bootstrap_samples=200, feature_importance=true_importance)
imp = pd.DataFrame(rep_s.global_shap_importance).set_index("feature")["mean_absolute_shapley_value"]
table = pd.DataFrame({"mean |SHAP|": imp, "true |coef|": pd.Series(true_importance)})
print(table.round(3).to_string())
print("Spearman(mean |SHAP|, true |coef|) =", round(stats.spearmanr(table["mean |SHAP|"], table["true |coef|"])[0], 3))
sia = rep_s.supplied_importance_alignment
print("supplied-importance alignment: global Spearman =", round(sia["global_absolute_importance_spearman"], 3),
      "| mean per-row top-feature overlap =", round(np.mean([r["top_feature_overlap"] for r in sia["per_row"]]), 3))
"""
        ),
        md("### Sensitivity to `num_samples` and the random seed"),
        code(
            """
rows = []
for num_samples in (100, 300, 1000):
    for seed in (0, 1):
        r = explain_estimator(lr, Xste, background_data=Xstr.sample(200, random_state=0), class_label=1, max_instances=20,
                              num_samples=num_samples, random_seed=seed, bootstrap_samples=100)
        top = pd.DataFrame(r.global_shap_importance).set_index("feature")["mean_absolute_shapley_value"]
        rows.append({"num_samples": num_samples, "seed": seed, "mean |SHAP| x0": top["x0"], "x1": top["x1"], "x3 (null)": top["x3"],
                     "alignment": r.reason_alignment["alignment_score_0_to_1"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
"""
        ),
        md(
            "## 4. Optional export (privacy caution)\n\n"
            "`export_path` writes the **input rows, predictions, optional labels and top features** to disk - "
            "that can contain personal or sensitive data, so it is opt-in and requires covering every row."
        ),
        code(
            """
small = Xte.iloc[:10]
explain_estimator(model, small, y_true=yte.iloc[:10], background_data=Xtr.sample(50, random_state=0), class_label=1,
                  num_samples=100, random_seed=0, bootstrap_samples=100, export_path="explanations_demo.csv")
print(pd.read_csv("explanations_demo.csv").iloc[:3, :8].round(3).to_string(index=False))
os.remove("explanations_demo.csv")  # do not leave row-level data lying around
"""
        ),
    ]
    cells += closing(
        "Global importance ranks features by average absolute attribution; local views show how one prediction is "
        "assembled. The alignment numbers say whether two different explanation procedures tell a similar story, "
        "and the sensitivity table shows how much of the output is just sampling noise from `num_samples` and the "
        "seed. In the synthetic case, the null features should receive small attributions.",
        "Attributions describe the model's behaviour under a chosen background distribution and perturbation "
        "scheme; they are not causal effects. Correlated or constrained features can produce unrealistic "
        "perturbed rows; Monte-Carlo estimates are noisy at small `num_samples`; the LIME-like fidelity only "
        "measures fit to a synthetic neighbourhood; agreement between the two methods is not correctness; "
        "only finite numeric features and models with `predict_proba` are supported; exported files can leak "
        "sensitive data.",
    )
    return cells


@register("reason-audit")
def reason_audit() -> list[dict]:
    cells = [
        header(
            "reason-audit",
            "Reason audit: what does the model rely on, and is it stable?",
            "reason-audit",
            "It uses `audit_model_reasons` (or `ReliablePipeline.audit_reasons`): predictive validity with "
            "bootstrap intervals, perturbation-based feature reliance, slice stability, counterfactual "
            "sensitivity, logistic association and explanation alignment.",
        )
    ]
    cells += setup_cells()
    cells += [
        md(
            r"""## 1. Quantities

**Predictive validity** on a labeled holdout $D$ ($n$ rows, probabilities $p_i$, one-hot $e_{y_i}$):
$$B=\frac1n\sum_{i=1}^n\lVert p_i-e_{y_i}\rVert_2^2,\qquad \operatorname{ECE}=\sum_{b=1}^{10}\frac{\lvert I_b\rvert}{n}\,\bigl\lvert\operatorname{acc}(I_b)-\operatorname{conf}(I_b)\bigr\rvert,$$
with row-bootstrap intervals for accuracy, Brier score and ECE.

**Perturbation-based feature reliance.** With multiclass Brier loss $L(f;D)$ and $D^{(j)}$ the data in which feature $j$ is replaced by donor values from other audit rows (optionally nearest donors on `conditional_on` features),
$$\Delta_j=\mathbb E\bigl[L(f;D^{(j)})-L(f;D)\bigr].$$
A positive *lower* interval bound is evidence of reliance **under this perturbation procedure** - not a causal effect, nor necessity in the data-generating process. **Negative controls** (`negative_control_features`) should show no reliance; **unavailable features** (`unavailable_features`, not known at prediction time) with positive reliance are leakage/shortcut indicators.

**Stability** repeats the reliance calculation within `groups` and `time_periods` slices. **Counterfactual sensitivity** replaces a column by user-supplied values and reports the prediction-flip rate and $\overline{\lvert\Delta\hat p_+\rvert}$ with row-bootstrap intervals. **Logistic association** fits an unregularized logistic model on standardized features and reports odds ratios per standard deviation with Benjamini-Hochberg adjusted $p$-values - a descriptive summary of associations in the audit data, not a model explanation. Explanation alignment compares SHAP-like and LIME-like attributions (see the explainability notebook)."""
        ),
        md(
            "## 2. Synthetic data with a planted shortcut\n\n"
            "Features: `signal` and `weak` truly drive the label; `noise` and `negctl` do not; `leak` is a noisy "
            "copy of the label that would not be available at prediction time. The model sees all of them."
        ),
        code(
            """
import warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ml_reliability import ReliablePipeline, audit_model_reasons

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
n = 900
X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["signal", "weak", "noise", "negctl"])
y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * X["signal"] + 0.5 * X["weak"])))).astype(int)
X["leak"] = y + rng.normal(scale=0.3, size=n)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=0)
forest = RandomForestClassifier(n_estimators=25, random_state=0).fit(Xtr, ytr)

group = np.where(Xte["signal"] > 0, "signal>0", "signal<=0")
period = np.array(["period_1", "period_2", "period_3"])[np.arange(len(Xte)) % 3]  # illustrative split
audit = audit_model_reasons(
    forest, Xte, yte, positive_label=1, background_data=Xtr,
    groups={"sign_of_signal": group}, time_periods=period,
    counterfactuals={"signal": np.zeros(len(Xte))},
    negative_control_features=("negctl",), unavailable_features=("leak",),
    bootstrap_samples=100, explanation_samples=100, max_explained_instances=15, random_seed=0,
)
pv = audit.predictive_validity
print(f"n={audit.sample_count}  accuracy={pv['accuracy']:.3f} {tuple(round(v, 3) for v in pv['accuracy_confidence_interval'])}"
      f"  Brier={pv['brier_score']:.3f} {tuple(round(v, 3) for v in pv['brier_score_confidence_interval'])}"
      f"  ECE={pv['top_label_expected_calibration_error']:.3f} {tuple(round(v, 3) for v in pv['expected_calibration_error_confidence_interval'])}")
"""
        ),
        code(
            """
reliance = pd.DataFrame([{
    "feature": r["feature"], "loss_increase": r["loss_increase"],
    "CI low": r["loss_increase_confidence_interval"][0], "CI high": r["loss_increase_confidence_interval"][1],
    "lower bound > 0": r["positive_lower_bound"],
    "negative control relied on": r.get("negative_control_warning", ""),
    "unavailable feature relied on": r.get("unavailable_at_prediction_time_warning", ""),
} for r in audit.feature_reliance])
print(reliance.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3.6))
order = reliance.sort_values("loss_increase")
ax.barh(order["feature"], order["loss_increase"], xerr=[order["loss_increase"] - order["CI low"], order["CI high"] - order["loss_increase"]], capsize=3)
ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("Brier-loss increase when the feature is perturbed"); ax.set_title("Feature reliance with 95% bootstrap CI")
plt.tight_layout(); plt.show()
"""
        ),
        code(
            """
print("Stability across slices (loss increase, lower CI bound > 0 marked with *):")
rows = []
for kind, slices in [("group", audit.stability["groups"]["sign_of_signal"]), ("period", audit.stability["time_periods"]["time_period"])]:
    for item in slices:
        row = {"slice": f"{kind}:{item['value']}", "n": item["sample_count"]}
        for feature, stats_ in item["feature_reliance"].items():
            low = stats_["loss_increase_confidence_interval"][0]
            row[feature] = f"{stats_['loss_increase']:.3f}{'*' if low > 0 else ''}"
        rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
print("between-slice SD of reliance:", {k: round(v["between_slice_standard_deviation"], 4) for k, v in audit.stability["features"].items()})
cf = audit.counterfactual_sensitivity["features"]["signal"]
print("Counterfactual signal:=0 -> flip rate", round(cf["prediction_flip_rate"], 3), tuple(round(v, 3) for v in cf["prediction_flip_rate_confidence_interval"]),
      "| mean |change in P(Y=1)|", round(cf["mean_absolute_positive_probability_change"], 3))
print()
odds = pd.DataFrame(audit.logistic_association["odds_ratios"]).T
print(odds[["odds_ratio_per_one_standard_deviation", "p_value", "fdr_bh_adjusted_p_value"]].round(4).to_string())
print("\\nexplanation alignment:", {k: v for k, v in audit.explanation_alignment.items() if k in ["status", "alignment_score_0_to_1", "mean_absolute_importance_spearman"]})
print("\\nwarnings:")
for w in audit.warnings:
    print(" -", w)
"""
        ),
        md(
            "## 3. Real data through the pipeline: breast cancer\n\n"
            "`ReliablePipeline.audit_reasons` reuses the fitted model and the training data as background. The "
            "groups and time periods below are **illustrative**: tumour-size groups and row-order periods, not real "
            "protected groups or real time. Features are chosen to avoid near-duplicates (radius, perimeter and "
            "area are almost collinear, which makes odds ratios unstable)."
        ),
        code(
            """
bc = load_breast_cancer(as_frame=True)
cols = ["mean radius", "mean texture", "mean smoothness", "mean compactness", "mean symmetry", "mean fractal dimension"]
Xb, yb = bc.data[cols], bc.target
Xbtr, Xbte, ybtr, ybte = train_test_split(Xb, yb, test_size=0.35, random_state=0, stratify=yb)
pipe = ReliablePipeline(LogisticRegression(max_iter=5000), plot_dir="reliability_plots", random_seed=0).fit(Xbtr, ybtr)
size = np.where(Xbte["mean radius"] > Xbte["mean radius"].median(), "large", "small")
period_b = np.array(["early", "late"])[(np.arange(len(Xbte)) >= len(Xbte) // 2).astype(int)]
bc_audit = pipe.audit_reasons(
    Xbte, ybte, positive_label=1, groups={"size": size}, time_periods=period_b,
    counterfactuals={"mean texture": np.full(len(Xbte), Xbtr["mean texture"].median())},
    conditional_on=["mean radius"], bootstrap_samples=100, explanation_samples=100, max_explained_instances=15,
)
pv = bc_audit.predictive_validity
print(f"accuracy={pv['accuracy']:.3f}  Brier={pv['brier_score']:.3f}  ECE={pv['top_label_expected_calibration_error']:.3f}")
print(pd.DataFrame([{"feature": r["feature"], "loss_increase": r["loss_increase"], "CI": tuple(round(v, 4) for v in r["loss_increase_confidence_interval"]),
                     "lower>0": r["positive_lower_bound"]} for r in bc_audit.feature_reliance]).round(4).to_string(index=False))
print(pd.DataFrame(bc_audit.logistic_association["odds_ratios"]).T[["odds_ratio_per_one_standard_deviation", "fdr_bh_adjusted_p_value"]].round(4).to_string())
print("alignment:", {k: v for k, v in bc_audit.explanation_alignment.items() if k in ["status", "alignment_score_0_to_1"]})
"""
        ),
    ]
    cells += closing(
        "A positive lower bound in the reliance table says the model's loss rises when that feature is scrambled, "
        "i.e. the model uses it. The planted `leak` feature dominates and is flagged because it was declared "
        "unavailable at prediction time; the negative control and `noise` should have intervals that include "
        "zero. Slice tables show whether reliance is stable across groups and periods (small slices are noisy). "
        "Counterfactual rows test the model, not the world.",
        "Reliance is relative to the perturbation scheme (marginal donors create unrealistic rows when features "
        "are correlated; conditioning only approximates the conditional distribution); it is not causal and "
        "does not identify necessity. Odds ratios are descriptive associations that become unstable under "
        "collinearity or separation; intervals assume independent rows and no multiplicity control beyond the "
        "reported BH adjustment; the auditor must declare which features are unavailable or negative controls; "
        "the audit data must be representative holdout data not used for training.",
    )
    return cells


@register("pipeline")
def pipeline() -> list[dict]:
    cells = [
        header(
            "pipeline",
            "ReliablePipeline: the reliability-aware prediction lifecycle",
            "pipeline",
            "It walks through fit, predict with a reliability report and plots, persistence, the "
            "`reliability_required` switch, and how the labeled analyses (evaluation, deterioration, "
            "explanation, reason audit) plug into the same object.",
        )
    ]
    cells += setup_cells()
    cells += [
        md(
            r"""## 1. Design

`ReliablePipeline(estimator, ...)` wraps any scikit-learn-style estimator and a `DriftMonitor`:

1. **`fit(X, y)`** fits the monitor's reference profiles (quantile bins, categories, moments, and the label-conditional distributions) *and* the estimator, and writes `training-distributions.png` to `plot_dir`.
2. **`predict(X)`** returns ordinary predictions, and as a side effect compares the batch with the reference and writes `prediction-<timestamp>.png/.json` (the report, predictions and output type). The latest report is in `last_report`, the file paths in `last_plot_path` and `last_report_path`.
3. **`predict_with_report(X)`** returns `(predictions, report)`.

The report's display score is $S=\exp(-\overline{\mathrm{PSI}})$ over features, with per-feature PSI, KL, KS, Wasserstein, DSI and chi-square statistics (see the distribution-shift notebook). It is descriptive: a **similarity-to-training** summary, not a probability that predictions are correct.

| Analysis | Method | Data requirement |
|---|---|---|
| Feature shift | `assess_covariate_shift(X)` | unlabeled batch and fitted reference |
| Concept shift | `assess_concept_shift(X, y)` | labeled batch |
| Classification evaluation | `evaluate(X, y_true, ...)` | labeled representative batch |
| Fairness | `evaluate(..., sensitive_features=...)` | labels, groups, positive class |
| Deterioration | `assess_deterioration(history)` | $\ge3$ dated labeled batches |
| Explanation | `explain(X, class_label=...)` | background data and target class |
| Reason audit | `audit_reasons(X, y_true, ...)` | labeled holdout, reference background |"""
        ),
        md("## 2. Fit and predict"),
        code(
            """
import json, os, tempfile, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml_reliability import ReliablePipeline, assess_deterioration

warnings.filterwarnings("ignore")
bc = load_breast_cancer(as_frame=True)
cols = ["mean radius", "mean texture", "mean smoothness", "mean compactness", "mean symmetry", "mean fractal dimension"]
X, y = bc.data[cols], bc.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)

pipe = ReliablePipeline(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                        plot_dir="reliability_plots", random_seed=0).fit(Xtr, ytr)
print("files after fit:", sorted(os.listdir("reliability_plots")))

predictions, report = pipe.predict_with_report(Xte)
print("files after predict:", sorted(os.listdir("reliability_plots")))
print("in-domain batch -> similarity", round(report.similarity_score, 3), "| mean PSI", round(report.mean_psi, 4),
      "| covariate shift flagged:", report.covariate_shift_detected, "| drifted:", list(report.drifted_features))
"""
        ),
        code(
            """
# The artifacts the pipeline wrote on the last predict()
payload = json.loads(pipe.last_report_path.read_text())
print("report JSON keys:", list(payload))
print("output_type:", payload["output_type"], "| first predictions:", payload["predictions"][:5])
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(mpimg.imread(pipe.last_plot_path)); ax.axis("off"); ax.set_title(pipe.last_plot_path.name)
plt.show()
"""
        ),
        md("### A shifted batch vs. an in-domain batch"),
        code(
            """
shifted = Xte.copy()
shifted["mean radius"] = shifted["mean radius"] * 1.5 + 2.0   # e.g. a unit/instrument change
shifted["mean texture"] = shifted["mean texture"] + 6.0
for name, batch in [("in-domain", Xte), ("shifted", shifted)]:
    rep = pipe.assess_covariate_shift(batch)
    print(f"{name:10s} similarity={rep.similarity_score:.3f} mean PSI={rep.mean_psi:.3f} flagged={rep.covariate_shift_detected} drifted={list(rep.drifted_features)}")
print("accuracy on in-domain:", round((pipe.predict(Xte) == yte).mean(), 3), "| on shifted:", round((pipe.predict(shifted) == yte).mean(), 3),
      "(a shift report is a warning to check outcomes, not a measure of accuracy)")
concept = pipe.assess_concept_shift(Xte, yte)
print("concept shift (labels as in training):", "mean JS =", round(concept.mean_js_divergence, 4), "| shifted:", list(concept.shifted_features))
"""
        ),
        md(
            "## 3. Persistence, schema checks and the `reliability_required` switch\n\n"
            "`save`/`load` use **pickle**: only load files you created or fully trust, with the same library and "
            "scikit-learn versions. Batches must have the same columns as training."
        ),
        code(
            """
with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "pipeline.pkl")
    pipe.save(path)
    restored = ReliablePipeline.load(path)
    print("round trip predictions identical:", bool((restored.predict(Xte) == predictions).all()))

try:
    pipe.predict(Xte.drop(columns=["mean radius"]))
except Exception as error:
    print("missing column ->", type(error).__name__ + ":", str(error)[:120])

plain = ReliablePipeline(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                         reliability_required=False, plot_dir="reliability_plots_off").fit(Xtr, ytr)
print("reliability_required=False: predictions work:", plain.predict(Xte)[:3], "| last_report:", plain.last_report)
for call in (lambda: plain.assess_covariate_shift(Xte), lambda: plain.predict_with_report(Xte)):
    try:
        call()
    except RuntimeError as error:
        print("  ->", str(error)[:90])
"""
        ),
        md(
            "## 4. Labeled analyses on the same object\n\n"
            "Each call below is a one-liner on the fitted pipeline; their settings are kept small here for "
            "speed. The dedicated notebooks (evaluation, fairness, deterioration, explainability, reason-audit) "
            "explain the outputs in detail."
        ),
        code(
            """
size = np.where(Xte["mean radius"] > Xte["mean radius"].median(), "large", "small")  # illustrative group, not protected
fast = ReliablePipeline(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), plot_dir="reliability_plots",
                        bias_variance_folds=3, bias_variance_repeats=3, random_seed=0).fit(Xtr, ytr)
ev = fast.evaluate(Xte, yte, sensitive_features={"size": size}, positive_label=1).to_dict()
cls = ev["classification"]
print("evaluate  -> accuracy", round(cls["accuracy"], 3), "| MCC", round(cls["matthews_correlation_coefficient"], 3),
      "| ECE", round(cls["probability_calibration"]["top_label_expected_calibration_error"], 3),
      "| parity gap", round(ev["fairness"]["size"]["classes"]["1"]["statistical_parity"]["gap"], 3))

batches = np.array_split(np.argsort(Xte["mean texture"].to_numpy()), 4)  # pseudo-batches, NOT real time
history = [{"timestamp": k, "y_true": yte.iloc[i].to_numpy(), "y_pred": fast.estimator.predict(Xte.iloc[i])} for k, i in enumerate(batches)]
det = fast.assess_deterioration(history, bootstrap_samples=200)
print("deterioration -> slope", round(det.accuracy_slope, 4), "| flagged:", det.deterioration_detected, "(pseudo-batches)")

exp = fast.explain(Xte.iloc[:20], class_label=1, background_data=Xtr.sample(50, random_state=0), max_instances=10, num_samples=100)
print("explain   -> top feature:", exp.global_shap_importance[0]["feature"], "| alignment:", round(exp.reason_alignment["alignment_score_0_to_1"], 3))

aud = fast.audit_reasons(Xte.iloc[:120], yte.iloc[:120], positive_label=1, bootstrap_samples=100, explanation_samples=100, max_explained_instances=10)
top = max(aud.feature_reliance, key=lambda r: r["loss_increase"])
print("audit     -> most relied-on feature:", top["feature"], "| loss increase", round(top["loss_increase"], 4))
"""
        ),
    ]
    cells += closing(
        "A reliability report summarizes how a new batch compares with the training reference; it appears (with "
        "plots and JSON) on every `predict`, so monitoring is automatic and leaves an audit trail. The shifted "
        "example shows a flagged batch, and the accuracy comparison reminds you that the shift report does not "
        "measure correctness - labeled evaluation, deterioration and the reason audit answer that when "
        "outcomes become available.",
        "The reference is the training data only; reports are per batch with sampling noise at small batch sizes; "
        "thresholds (PSI 0.2 etc.) are conventions to be calibrated; the pipeline writes files on every predict "
        "(set `plot_dir` deliberately and treat reports as potentially sensitive); pickled pipelines are not a "
        "safe interchange format; supported estimators must be scikit-learn compatible, and the explanation and "
        "audit tools additionally need `predict_proba`, `classes_` and finite numeric features.",
    )
    return cells


# ==== NOTEBOOKS-END ====


def main() -> None:
    for name, builder in BUILDERS.items():
        path = write_notebook(name, builder())
        print("wrote", path.relative_to(HERE.parent.parent))


if __name__ == "__main__":
    main()
