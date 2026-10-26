"""End-to-end ReliableML workflow, with Data Intelligence deciding what happens at each stage.

Order of work (and why):
  1. Data Intelligence on the RAW data      -> hazards, leakage, outliers, collinearity, "is the target learnable?"
  2. Cleaning                               -> act on step 1 (drop leaks/hazards, then statistical cleaning)
  3. Data Intelligence on the CLEAN data    -> feature subset, baseline skill the models must beat
  4. Models: meta-learning + StatCore       -> fitted inside ReliablePipeline (drift reference is stored at fit)
  5. Evaluation, fairness, explainability, reason audit
  6. Monitoring: drift, deterioration, and Data Intelligence on incoming batches
  7. AutoML (separate package)              -> same intelligence, report-only, on its training partition

Nothing is written to your project by this script (ReliablePipeline needs a plot folder, so it gets a temp dir).
Run:  python docs/examples/data_intelligence_workflow.py        (step 7 runs when reliableml_automl is installed)
"""
from __future__ import annotations

import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer  # dataset loader only

from ml_reliability import (
    DataIntelligence,
    DriftMonitor,
    MetaPipelineClassifier,
    ReliablePipeline,
    StatCoreKNNClassifier,
    StatCoreLogisticClassifier,
    clean_statistical_data,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


# ------------------------------------------------------------------ a deliberately messy dataset
raw = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "y"})
raw = raw[[c for c in raw.columns if c.startswith("mean") or c == "y"]].copy()
raw["leaky_score"] = raw["y"] + rng.normal(0, 0.02, len(raw))           # planted: near-copy of the target
raw["mean_radius_cm"] = raw["mean radius"] * 0.1 + rng.normal(0, 1e-3, len(raw))  # planted: duplicate unit
raw["constant_flag"] = 1.0                                                  # planted: hazard
raw.loc[raw.sample(frac=0.05, random_state=1).index, "mean texture"] = np.nan  # planted: missing values
raw.loc[:6, ["mean radius", "mean area"]] *= 5                              # planted: gross outliers
raw["site"] = rng.choice(["A", "B"], len(raw))                              # sensitive attribute for fairness
raw = raw.sample(frac=1.0, random_state=3).reset_index(drop=True)
dev, prod = raw.iloc[:400].copy(), raw.iloc[400:].copy()                    # development vs later "production" rows

# ------------------------------------------------------------------ 1. intelligence on raw data
banner("1. DATA INTELLIGENCE ON RAW DATA (autonomous mode: target + features + outliers + EDA)")
di = DataIntelligence(mode="autonomous", random_state=0)                    # saves nothing
report = di.analyze(dev.drop(columns="site"), target_column="y")
print("\n".join(report.to_text().splitlines()[:14]))

ti = report["target_intelligence"]
leaky = [f["feature"] for f in ti["leakage"]["flags"] if f["severity"] == "high"]
hazards = report["feature_intelligence"]["hazards"]
dropped_vif = ti["multicollinearity"]["recommended_drop"]
print("\nleakage (high):", leaky)
print("hazard columns:", hazards)
print("VIF suggests dropping:", dropped_vif)
figures = di.plot()["figures"]                                              # in memory; di.export(dir) to save
print("figures available (not saved):", list(figures))

# ------------------------------------------------------------------ 2. cleaning, driven by step 1
banner("2. CLEANING (acting on step 1)")
features = [c for c in dev.columns if c not in ("y", "site") and c not in leaky and c not in hazards]
clean_X, cleaner = clean_statistical_data(dev[features], cleaning_intensity="moderate", return_cleaner=True)
print(f"kept {len(features)} columns; missing cells after cleaning: {int(clean_X.isna().sum().sum())}")

# ------------------------------------------------------------------ 3. intelligence on clean data
banner("3. DATA INTELLIGENCE ON CLEAN DATA (what is achievable, which features to keep)")
clean = clean_X.assign(y=dev["y"].to_numpy())
di2 = DataIntelligence(mode="target_intelligence", random_state=0)
rep2 = di2.analyze(clean, target_column="y")
ti2 = rep2["target_intelligence"]
print("explainability verdict:", ti2["explainability"]["verdict"],
      f"(linear skill {ti2['explainability']['linear_skill']:.3f}, knn {ti2['explainability']['knn_skill']:.3f})")
best = ti2["feature_selection"]["best_features"]
print(f"feature search ({ti2['feature_selection']['best_method']}) keeps {len(best)} of {len(features)}:", best)
print("remaining leakage flags:", [(f["feature"], f["severity"]) for f in ti2["leakage"]["flags"]])
X_dev = clean[best]
y_dev = dev["y"].reset_index(drop=True)

# ------------------------------------------------------------------ 4. models
banner("4. MODELS: meta-learning + StatCore inside ReliablePipeline")
split = int(0.75 * len(X_dev))
X_tr, X_te = X_dev.iloc[:split], X_dev.iloc[split:]
y_tr, y_te = y_dev.iloc[:split], y_dev.iloc[split:]
candidates = {
    "meta_pipeline": MetaPipelineClassifier(n_folds=3, random_state=0),
    "statcore_knn": StatCoreKNNClassifier(),
    "statcore_logistic": StatCoreLogisticClassifier(),
}
scores = {name: float((m.fit(X_tr, y_tr).predict(X_te) == y_te.to_numpy()).mean()) for name, m in candidates.items()}
for name, acc in scores.items():
    print(f"  {name:<18} holdout accuracy {acc:.3f}")
chosen = max(scores, key=scores.get)
print("chosen:", chosen, "| step-3 linear skill (balanced, chance-adjusted):", round(ti2["explainability"]["linear_skill"], 3))

plot_dir = tempfile.mkdtemp(prefix="reliableml_")                           # throw-away; not your project folder
model = ReliablePipeline(candidates[chosen], plot_dir=plot_dir, bias_variance_folds=3,
                         bias_variance_repeats=2, bootstrap_samples=200).fit(X_tr, y_tr)

# ------------------------------------------------------------------ 5. evaluation, fairness, explanations, audit
banner("5. EVALUATION, FAIRNESS, EXPLAINABILITY, REASON AUDIT")
site_te = dev["site"].iloc[:split + len(X_te)].iloc[split:].to_numpy()
ev = model.evaluate(X_te, y_te, sensitive_features={"site": site_te}, positive_label=1).to_dict()
print("evaluation keys:", list(ev)[:8])
exp = model.explain(X_te, max_instances=30, num_samples=100)
print("explanation:", type(exp).__name__)
audit = model.audit_reasons(X_te, y_te, max_explained_instances=20, explanation_samples=50, bootstrap_samples=100)
print("reason audit:", type(audit).__name__)

# ------------------------------------------------------------------ 6. monitoring
banner("6. MONITORING: drift, deterioration, and intelligence on incoming data")
prod_X = cleaner.transform(prod[features])[best].reset_index(drop=True)
prod_y = prod["y"].reset_index(drop=True)
prod_X["mean texture"] = prod_X["mean texture"] * 1.6 if "mean texture" in prod_X else 0   # simulated shift
monitor = DriftMonitor().fit(X_tr, y_tr)
cov = monitor.assess_covariate_shift(prod_X)
print("covariate shift detected:", cov.covariate_shift_detected, "| drifted:", cov.drifted_features)

batches = np.array_split(np.arange(len(prod_X)), 4)
history = [{"timestamp": f"2026-0{i + 1}-01", "y_true": prod_y.iloc[b].tolist(),
            "y_pred": model.predict(prod_X.iloc[b]).tolist()} for i, b in enumerate(batches)]
det = model.assess_deterioration(history, bootstrap_samples=200)
print("deterioration detected:", det.deterioration_detected, f"| accuracy slope {det.accuracy_slope:+.4f}/period, CI {det.slope_confidence_interval}")

di_prod = DataIntelligence(mode="comprehensive_eda", random_state=0).analyze(prod_X)
print("incoming-batch intelligence score:", round(di_prod.summary_score, 1),
      "| first recommendation:", (di_prod["recommendations"] or ["-"])[0])

# ------------------------------------------------------------------ 7. AutoML
banner("7. AUTOML: same intelligence, report-only, on its training partition")
try:
    from reliableml_automl import ReliableAutoML

    automl = ReliableAutoML("y", data_intelligence="profile", progress_bar=False,   # "multicollinearity" = lighter
                            reliability_options={"reliability_required": False},
                            plot_dir=tempfile.mkdtemp(prefix="reliableml_automl_"))
    automl.fit(dev.drop(columns=["site"]))
    print("selected:", automl.selected_model_name)
    di_rep = automl.data_intelligence_report
    print("highly collinear columns suggested for removal:", di_rep["recommended_drop"])
    print("hazards:", di_rep["profile"]["hazards"])
    print("profile score:", round(di_rep["profile"]["summary_score"], 1))
    # ReliableAutoML(..., export_dir="automl_out") or automl.export_report("automl_out") saves everything: opt-in only.
except ImportError:
    print("reliableml_automl is not installed in this environment; skipping.")

# ------------------------------------------------------------------ 8. Analysis toolkit
banner("8. TOOLKIT: power analysis, feature construction, active learning, row-selector hook")
from ml_reliability import (ActiveLearningStudy, FeatureConstructor, describe, evaluate_row_subset,
                            power_analysis)

feats, labels = dev[features], dev["y"].to_numpy()        # leaks and hazards from step 1 already removed
print(describe(power_analysis(feats, labels)))                      # predictive vs explanatory vs statistical power

fc = FeatureConstructor(random_state=0).fit(feats, labels)          # iterative, distribution-aware, holdout-checked
print(describe(fc.report))

study = ActiveLearningStudy(learner="knn", n_steps=5, n_repeats=2).run(feats, labels)
print(describe(study))


def my_selector(X, n):          # stand-in for your own row selector: indices only
    return np.arange(0, len(X), max(1, len(X) // n))[:n]


print(evaluate_row_subset(feats, labels, my_selector(feats.to_numpy(), 100)))
