"""Use cases 1-5: Data Intelligence when you have data but not (yet) a question.

  1. No target, no question     -> question plan + which columns the rest of the data can predict
  2. A target, before modelling -> leakage, hazards, collinearity (VIF), explainability, feature search
  3. Outlier triage             -> isolation forest, LOF, robust Mahalanobis and a SOM, as a consensus
  4. Incoming-batch screening   -> same engine on new data (EDA mode)
  4b. Deep dive              -> investigate one field: drivers, pairs, follow-up questions, suspect rows
  5. Opt-in saving              -> report / figures only written when you ask

Nothing is written to disk unless SAVE_DIR is set.   Run:  python examples/01_data_intelligence_discovery.py
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer  # dataset loader only, no sklearn models

from ml_reliability import (
    DataIntelligence,
    SelfOrganisingMap,
    isolation_forest_scores,
    local_outlier_factor,
    robust_mahalanobis,
)

warnings.filterwarnings("ignore")
SAVE_DIR = None          # e.g. "di_output" to save json/txt/html + figures
rng = np.random.default_rng(0)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# A deliberately messy table: a leak, a duplicate unit, a constant column, gross outliers.
df = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "y"})
df = df[[c for c in df.columns if c.startswith("mean") or c == "y"]].copy()
df["leaky_score"] = df["y"] + rng.normal(0, 0.02, len(df))
df["mean_radius_cm"] = df["mean radius"] * 0.1 + rng.normal(0, 1e-3, len(df))
df["constant_flag"] = 1.0
df.loc[:6, ["mean radius", "mean area"]] *= 5
df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
dev, incoming = df.iloc[:400].copy(), df.iloc[400:].copy()

# ---------------------------------------------------------------- 1. no target, no question
banner("1. NO TARGET, NO QUESTION: what should I ask of this data?")
report = DataIntelligence().analyze(dev.drop(columns="y"))
for q in report["question_plan"][:5]:
    print(f"  [{q['priority']}] {q['category']}: {q['question']}\n      -> {q['next_step']}")
print("columns the rest of the data can predict (candidate targets):")
for c in report["candidate_targets"][:4]:
    print("  ", c)

# ---------------------------------------------------------------- 2. a target, before modelling
banner("2. A TARGET, BEFORE MODELLING: leakage, hazards, collinearity, feature search")
di = DataIntelligence(mode="autonomous", random_state=0, export_dir=SAVE_DIR)
report = di.analyze(dev, target_column="y")
ti = report["target_intelligence"]
print("leakage (high)     :", [f["feature"] for f in ti["leakage"]["flags"] if f["severity"] == "high"])
print("hazard columns     :", report["feature_intelligence"]["hazards"])
print("VIF suggests drop  :", ti["multicollinearity"]["recommended_drop"])
print("explainability     :", ti["explainability"]["verdict"])
print("features worth keeping:", ti["feature_selection"]["best_features"][:6], "...")
print("summary score      :", round(report.summary_score, 1))
print("recommendations    :", report["recommendations"][:3])

# Act on what it found, then ask again: the feature search is only meaningful once the leak is gone.
leaks = [f["feature"] for f in ti["leakage"]["flags"] if f["severity"] == "high"]
hazards = report["feature_intelligence"]["hazards"]
cleaned = dev.drop(columns=leaks + hazards)
ti2 = DataIntelligence(mode="target_intelligence", random_state=0).analyze(cleaned, "y")["target_intelligence"]
print("after dropping leaks/hazards -> explainability:", ti2["explainability"]["verdict"],
      "| keep:", ti2["feature_selection"]["best_features"][:6])

# ---------------------------------------------------------------- 3. outlier triage
banner("3. OUTLIER TRIAGE: four detectors, one consensus (a row flagged by >= 2)")
num = dev.drop(columns=["y", "constant_flag"])
Z = ((num - num.median()) / (num.quantile(0.75) - num.quantile(0.25) + 1e-9)).to_numpy()
iso = isolation_forest_scores(Z, random_state=0)
lof = local_outlier_factor(Z, k=20)
d2, cutoff, _ = robust_mahalanobis(Z)
som = SelfOrganisingMap(epochs=30, random_state=0).fit(Z)
som_err = som.unit_quantization_error()
flags = np.c_[iso > np.quantile(iso, 0.97), lof > np.quantile(lof, 0.97), d2 > cutoff]
consensus = np.where(flags.sum(1) >= 2)[0]
print("flagged by isolation forest:", int(flags[:, 0].sum()), "| LOF:", int(flags[:, 1].sum()),
      "| Mahalanobis:", int(flags[:, 2].sum()))
print("consensus rows (>=2 detectors):", consensus[:15].tolist())
print("SOM quantisation error:", round(float(som.quantization_error_), 3),
      "| u_matrix and component planes: som.u_matrix(), som.component_planes()")
# ---------------------------------------------------------------- 4. incoming batch
banner("4. INCOMING-BATCH SCREENING: the same engine on new data (no target needed)")
batch = incoming.drop(columns=["y", "leaky_score", "constant_flag"]).copy()
batch["mean texture"] *= 1.6                                   # simulated shift
rep_batch = DataIntelligence(mode="comprehensive_eda", random_state=0).analyze(batch)
print("batch score:", round(rep_batch.summary_score, 1), "| first recommendation:",
      (rep_batch["recommendations"] or ["-"])[0])

# ---------------------------------------------------------------- 4b. deep dive
banner("4b. DEEP DIVE: investigate one field, then the rows and groups behind it")
field = "mean area"
inv = DataIntelligence(mode="field_investigation", random_state=0).analyze(
    dev.drop(columns=["leaky_score", "constant_flag"]), target_column=field)
print("what happened:", {k: v for k, v in inv["what_happened"]["hazards"].items() if v} or "no hazards")
print("why - strongest drivers of", inv["why"]["investigated_field"], ":")
for d in inv["why"]["drivers"][:4]:
    print("  ", d)
print("strongly related pairs:", [p for p in inv["why"]["strong_pairs"]][:3])
print("follow-up questions the dive raised:")
for q in inv["question_plan"][:4]:
    print(f"  [{q['priority']}] {q['question']}\n      -> {q['next_step']}")
# Drill into the flagged rows themselves, then ask the next question of that subset.
suspects = dev.iloc[consensus]
print("suspect rows vs the rest (median):")
print(pd.DataFrame({"suspects": suspects[["mean radius", "mean area"]].median(),
                    "rest": dev.drop(dev.index[consensus])[["mean radius", "mean area"]].median()}).round(2))

# ---------------------------------------------------------------- 5. opt-in saving
banner("5. OPT-IN SAVING")
print("figures available in memory:", list(di.plot()["figures"]))
if SAVE_DIR:
    print(di.export(SAVE_DIR, formats=("json", "html")))
else:
    print("SAVE_DIR is None, so nothing was written. Set it, or call di.export('folder', formats=('html',)).")
