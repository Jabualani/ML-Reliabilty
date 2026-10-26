"""Use cases 31-39: the reliability checks in isolation (own models first, scikit-learn models also work).

Every check below is independent: you can run just the one you need, on any fitted classifier.

  31. Cleaning first (when data is not clean)      clean_statistical_data
  32. Drift: covariate + concept shift             ReliablePipeline / DriftMonitor (works for ANY model or no model)
  33. Evaluation: metrics, calibration, bias-variance   pipeline.evaluate
  34. Fairness: group gaps with confidence intervals    evaluate(sensitive_features=...)
  35. Deterioration: is accuracy trending down?    assess_deterioration (stable vs degrading history)
  36. Explainability: SHAP vs LIME consistency      pipeline.explain / explain_estimator (+ CSV export)
  37. Reason audit: does the model rely on defensible reasons?   audit_reasons / audit_model_reasons
  38. Visualisation of drift                        plot_distributions / plot_reference_distributions
  39. Pipeline options: opt out, thresholds, save/load

Suggested order in a project: clean -> fit (reference stored) -> evaluate (+fairness) -> explain -> audit
-> deploy -> monitor drift on each batch -> deterioration once labels arrive.
Nothing is written to your folders: plots go to a temp directory that is deleted at the end.
Run:  python examples/05_reliability_in_isolation.py
"""
from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from ml_reliability import (
    DriftMonitor,
    MetaPipelineClassifier,
    ReliablePipeline,
    StatCoreKNNClassifier,
    StatCoreLDAClassifier,
    StatCoreLogisticClassifier,
    assess_deterioration,
    audit_model_reasons,
    clean_statistical_data,
    explain_estimator,
)

warnings.filterwarnings("ignore")
TMP = Path(tempfile.mkdtemp(prefix="reliableml_checks_"))
rng = np.random.default_rng(0)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def short(x, n=6):
    """Compact preview of a dict/list for printing."""
    if isinstance(x, dict):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in list(x.items())[:n]}
    return x[:n] if isinstance(x, (list, tuple)) else x


# ---------------------------------------------------------------- data
raw = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "y"})
raw = raw[[c for c in raw.columns if c.startswith("mean")] + ["y"]].sample(frac=1, random_state=0).reset_index(drop=True)
cols = [c for c in raw.columns if c != "y"][:6]
raw["random_control"] = rng.normal(size=len(raw))                      # known negative control for the audit
raw["site"] = rng.choice(["A", "B"], len(raw))                           # sensitive attribute: NEVER a model feature
raw.loc[raw.sample(frac=0.05, random_state=1).index, "mean texture"] = np.nan    # dirty on purpose

# ---------------------------------------------------------------- 31. cleaning
banner("31. CLEANING FIRST (assuming the data was not clean)")
features = cols + ["random_control"]
X_all, cleaner = clean_statistical_data(raw[features], cleaning_intensity="moderate", return_cleaner=True)
print("missing cells before/after:", int(raw[features].isna().sum().sum()), "->", int(X_all.isna().sum().sum()))
print("cleaning report keys:", list(cleaner.get_cleaning_report())[:5])
y_all, site_all = raw["y"], raw["site"].to_numpy()
tr, ho, prod = slice(0, 300), slice(300, 420), slice(420, None)         # train / labelled holdout / later production
X_tr, y_tr = X_all.iloc[tr], y_all.iloc[tr]
X_ho, y_ho, site_ho = X_all.iloc[ho], y_all.iloc[ho], site_all[ho]

# ---------------------------------------------------------------- 32. drift
banner("32. DRIFT: covariate shift (no labels) and concept shift (labels)")
own_model = MetaPipelineClassifier(base_estimators=[("lr", StatCoreLogisticClassifier()), ("lda", StatCoreLDAClassifier()),
                                                    ("knn", StatCoreKNNClassifier())], n_folds=3)
model = ReliablePipeline(own_model, plot_dir=TMP / "plots", bias_variance_folds=3, bias_variance_repeats=2,
                         bootstrap_samples=200).fit(X_tr, y_tr)            # the training reference is stored here
shifted = X_all.iloc[prod].copy()
shifted["mean texture"] *= 1.7
shifted["mean radius"] += 3
r_same = model.assess_reliability(X_ho)
r_shift = model.assess_reliability(shifted)
print(f"similarity (1 = same distribution)  holdout {r_same.similarity_score:.3f} | shifted {r_shift.similarity_score:.3f}")
print("covariate shift detected          :", r_same.covariate_shift_detected, "|", r_shift.covariate_shift_detected)
print("drifted features on shifted batch :", r_shift.drifted_features)
concept = model.assess_concept_shift(X_ho, y_ho)
print("concept shift (labels needed)     :", type(concept).__name__, "| keys:", list(concept.to_dict())[:5])
print("\nNo model at all? DriftMonitor is stand-alone:")
mon = DriftMonitor(drift_threshold=0.2).fit(X_tr, y_tr)
print("  drifted:", mon.assess_covariate_shift(shifted).drifted_features[:4], "...")
print("\nWorks next to a scikit-learn model too:  ReliablePipeline(RandomForestClassifier()).fit(X, y)")

# ---------------------------------------------------------------- 33. evaluation
banner("33. EVALUATION: metrics, calibration, statistical tests, bias-variance")
ev = model.evaluate(X_ho, y_ho, positive_label=1)
c = ev.classification
print("accuracy", round(c["accuracy"], 3), "| balanced", round(c["balanced_accuracy"], 3),
      "| MCC", round(c["matthews_correlation_coefficient"], 3))
print("confusion matrix:", c["confusion_matrix"])
print("calibration:", short(c.get("probability_calibration", {}), 4))
print("training-vs-test cohort tests:", list(ev.statistical_tests["training_vs_test_cohorts"])[:3])
print("bias/variance:", short(ev.bias_variance, 5))

# ---------------------------------------------------------------- 34. fairness
banner("34. FAIRNESS: selection-rate, TPR/FPR and PPV gaps between groups")
ev_f = model.evaluate(X_ho, y_ho, sensitive_features={"site": site_ho}, positive_label=1)
fair = ev_f.fairness["site"]["classes"]["1"]
print("selection rate per group:", {g["group"]: round(g["selection_rate"], 2) for g in fair["per_group"]})
print("statistical-parity gap  :", round(fair["statistical_parity"]["gap"], 3), "CI", [round(v, 3) for v in fair["statistical_parity"]["confidence_interval"]])
print("equalized-odds max gap  :", round(fair["equalized_odds"]["maximum_gap"], 3), "| predictive-parity gap:", round(fair["predictive_parity"]["gap"], 3))
print("Gaps are descriptive (max - min across groups); you choose the acceptable threshold.")

# ---------------------------------------------------------------- 35. deterioration
banner("35. DETERIORATION: is accuracy trending down? (needs >= 3 dated, labelled batches)")
Xp, yp = X_all.iloc[prod].reset_index(drop=True), y_all.iloc[prod].reset_index(drop=True)
parts = np.array_split(np.arange(len(Xp)), 4)
stable = [{"timestamp": f"2026-0{i + 1}-01", "y_true": yp.iloc[b].tolist(), "y_pred": model.predict(Xp.iloc[b]).tolist()}
          for i, b in enumerate(parts)]
degrading = []
for i, b in enumerate(parts):                       # simulate predictions getting worse every month
    pred = model.predict(Xp.iloc[b]).copy()
    flip = rng.random(len(b)) < 0.08 * i * 2.2
    degrading.append({"timestamp": f"2026-0{i + 1}-01", "y_true": yp.iloc[b].tolist(),
                      "y_pred": np.where(flip, 1 - pred, pred).tolist()})
for label, hist in [("stable    ", stable), ("degrading ", degrading)]:
    d = assess_deterioration(hist, bootstrap_samples=300)
    print(f"{label} slope {d.accuracy_slope:+.4f}/day  CI {tuple(round(v, 4) for v in d.slope_confidence_interval)}"
          f"  deterioration detected: {d.deterioration_detected}")

# ---------------------------------------------------------------- 36. explainability
banner("36. EXPLAINABILITY: do Shapley and LIME tell the same story?")
exp = model.explain(X_ho, class_label=1, max_instances=25, num_samples=100)
print("global importance (mean |SHAP|):", {d["feature"]: round(d["mean_absolute_shapley_value"], 3)
                                          for d in exp.global_shap_importance[:4]})
print("reason alignment:", short(exp.reason_alignment, 4))
out_csv = TMP / "explanations.csv"
explain_estimator(model.estimator, X_ho.iloc[:20], y_true=y_ho.iloc[:20], background_data=X_tr, num_samples=100,
                  bootstrap_samples=100, export_path=out_csv)             # opt-in: writes rows + top features
print("standalone explain_estimator exported:", out_csv.name, f"({len(pd.read_csv(out_csv))} rows)")

# ---------------------------------------------------------------- 37. reason audit
banner("37. REASON AUDIT: statistically supported, stable reasons? (negative control flagged)")
audit = model.audit_reasons(X_ho, y_ho, positive_label=1, groups={"site": site_ho},
                            negative_control_features=["random_control"], conditional_on=["mean radius"],
                            max_explained_instances=20, explanation_samples=100, bootstrap_samples=100)
print("predictive validity: accuracy", round(audit.predictive_validity["accuracy"], 3),
      "CI", [round(v, 3) for v in audit.predictive_validity["accuracy_confidence_interval"]])
print("feature reliance  :", {r["feature"]: (round(r["loss_increase"], 3), "supported" if r["positive_lower_bound"] else "not supported")
                              for r in audit.feature_reliance[:4]})
print("first caution     :", audit.warnings[0] if getattr(audit, "warnings", None) else "-")
audit2 = audit_model_reasons(StatCoreKNNClassifier().fit(X_tr, y_tr), X_ho, y_ho, background_data=X_tr,
                             positive_label=1, max_explained_instances=15, explanation_samples=100, bootstrap_samples=100)
print("standalone audit of any fitted classifier ->", type(audit2).__name__)

# ---------------------------------------------------------------- 38. visualisation
banner("38. VISUALISATION: reference vs incoming distributions")
fig, _ = model.plot_distributions(shifted, feature_names=["mean texture", "mean radius"])
fig.savefig(TMP / "shift.png", dpi=80)
ref_fig, _ = model.plot_reference_distributions(feature_names=["mean texture"])
print("figures built in memory and saved only to the temp folder:", (TMP / "shift.png").exists())

# ---------------------------------------------------------------- 39. options
banner("39. PIPELINE OPTIONS: opt out of monitoring, thresholds, save/load")
light = ReliablePipeline(StatCoreLogisticClassifier(), reliability_required=False).fit(X_tr, y_tr)
print("reliability_required=False -> plain predictions only:", light.predict(X_ho)[:5].tolist())
strict = ReliablePipeline(StatCoreLogisticClassifier(), drift_threshold=0.1, plot_dir=TMP / "p2").fit(X_tr, y_tr)
print("stricter PSI threshold flags more:", strict.assess_reliability(shifted).drifted_features[:3], "...")
model.save(TMP / "model.pkl")
back = ReliablePipeline.load(TMP / "model.pkl")
print("reloaded predictions identical:", bool((back.predict(X_ho) == model.predict(X_ho)).all()))

shutil.rmtree(TMP, ignore_errors=True)
print("\ntemporary files removed.")
