"""Use cases 14-20: ReliableML meta-learners vs StatCore (own models first, scikit-learn bases optional).

  14. StatCore classifiers, every variant (pure NumPy)       StatCoreLogistic / LDA / KNN / MLP
  15. StatCore meta-learners, all five meta-variants          StatCoreMetaClassifier
  16. ReliableML meta-learners stacked on OWN base models      MetaPipeline / PerfMetaSelect / CalibratedStack
  17. ReliableML statistical meta-learners (binary)            StatisticalMeta LDA / DTree / NN
  18. Raw StatCore: regression, uncertainty heads, PCA, k-means
  19. Head-to-head on several datasets + Friedman / Nemenyi    compare_methods
  20. Tuning an own model with successive halving              halving_grid_search

scikit-learn is used here only for dataset loaders and fold splitting; every model scored is ReliableML's or StatCore's.
To add a scikit-learn base model, see the commented block in use case 16.
Run:  python examples/03_meta_learning_vs_statcore.py          (about 2-4 minutes)
"""
from __future__ import annotations

import time
import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml_reliability import (
    CalibratedStackedClassifier,
    MetaPipelineClassifier,
    PerfMetaSelect,
    StatCoreKNNClassifier,
    StatCoreLDAClassifier,
    StatCoreLogisticClassifier,
    StatCoreMetaClassifier,
    StatCoreMLPClassifier,
    StatisticalMetaDTreeClassifier,
    StatisticalMetaLDAClassifier,
    StatisticalMetaNNClassifier,
    compare_methods,
    halving_grid_search,
    summarize_repeats,
)
from ml_reliability.statcore import (
    KMeans,
    KNNRegressor,
    MLPClassifier,
    MLPRegressor,
    PCA,
    RidgeRegression,
)

warnings.filterwarnings("ignore")


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def bal_acc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def load(fn):
    b = fn(as_frame=True)
    return b.data.reset_index(drop=True), b.target.reset_index(drop=True)


X, y = load(load_breast_cancer)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


def score(model, Xa=X_tr, ya=y_tr, Xb=X_te, yb=y_te):
    t = time.time()
    p = clone(model).fit(Xa, ya).predict(Xb)
    return bal_acc(yb, p), time.time() - t


def table(title, rows):
    print(f"\n{title}\n  {'model':<44}{'bal. acc.':>10}{'seconds':>9}")
    for name, (a, s) in rows.items():
        print(f"  {name:<44}{a:>10.3f}{s:>9.1f}")


# ---------------------------------------------------------------- 14. StatCore classifiers, every variant
banner("14. STATCORE CLASSIFIERS: every variant (pure NumPy, sklearn-compatible wrappers)")
rows = {}
for v in ("base", "risk_calibrated", "stat_regularised", "robust_loss"):
    rows[f"StatCoreLogistic[{v}]"] = score(StatCoreLogisticClassifier(variant=v))
for v in ("base", "shrinkage", "sparse", "heteroscedastic"):
    rows[f"StatCoreLDA[{v}]"] = score(StatCoreLDAClassifier(variant=v))
for v in ("base", "prototype_compressed", "learnable_metric", "stat_k_selection"):
    rows[f"StatCoreKNN[{v}]"] = score(StatCoreKNNClassifier(variant=v))
for v in ("base", "stat_regularised", "reliability_head", "uncertainty_head"):
    rows[f"StatCoreMLP[{v}]"] = score(StatCoreMLPClassifier(variant=v))
table("breast cancer, 30% holdout", rows)

# ---------------------------------------------------------------- 15. StatCore meta-learners
banner("15. STATCORE META-LEARNERS: five meta-variants over StatCore base models")
rows = {f"StatCoreMeta[{v}]": score(StatCoreMetaClassifier(meta_variant=v, n_folds=3))
        for v in ("performance_aware", "metrics_based", "oof_stacking", "statistical_meta", "calibrated_stack")}
table("breast cancer, 30% holdout", rows)

# ---------------------------------------------------------------- 16. ReliableML meta-learners on OWN bases
banner("16. RELIABLEML META-LEARNERS stacked on your own (StatCore) base models")
own_bases = [("sc_logistic", StatCoreLogisticClassifier()), ("sc_lda", StatCoreLDAClassifier()),
             ("sc_knn", StatCoreKNNClassifier())]
rows = {
    "MetaPipelineClassifier(own bases)": score(MetaPipelineClassifier(base_estimators=own_bases, n_folds=3)),
    "PerfMetaSelect(own bases)": score(PerfMetaSelect(estimators=own_bases, cv=3, scoring="balanced_accuracy")),
    "CalibratedStackedClassifier(own bases)": score(CalibratedStackedClassifier(base_estimators=own_bases, n_folds=3)),
}
table("breast cancer, 30% holdout", rows)
print("\nTo mix in scikit-learn models (optional):")
print("  from sklearn.ensemble import RandomForestClassifier")
print("  bases = own_bases + [('rf', RandomForestClassifier(n_estimators=200, random_state=0))]")
print("  MetaPipelineClassifier(base_estimators=bases, n_folds=3)")

# ---------------------------------------------------------------- 17. statistical meta-learners
banner("17. RELIABLEML STATISTICAL META-LEARNERS (binary targets, NumPy only)")
rows = {
    "StatisticalMetaLDA": score(StatisticalMetaLDAClassifier(n_mc_sims=10)),
    "StatisticalMetaDTree": score(StatisticalMetaDTreeClassifier(n_clusters=8, n_mc_sims=10)),
    "StatisticalMetaNN": score(StatisticalMetaNNClassifier()),
}
table("breast cancer, 30% holdout", rows)

# ---------------------------------------------------------------- 18. raw StatCore
banner("18. RAW STATCORE: regression, uncertainty heads, PCA, k-means")
Xd, yd = load_diabetes(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3, random_state=0)
for name, m in [("RidgeRegression[stat_regularised]", RidgeRegression(variant="stat_regularised")),
                ("KNNRegressor[stat_k_selection]", KNNRegressor(variant="stat_k_selection")),
                ("MLPRegressor[uncertainty_head]", MLPRegressor(variant="uncertainty_head"))]:
    m.fit(Xd_tr, yd_tr)
    print(f"  {name:<36} evaluate -> {m.evaluate(Xd_te, yd_te)}")

mlp = MLPClassifier(variant="uncertainty_head").fit(X_tr.to_numpy(), y_tr.to_numpy())
unc = mlp.predict_uncertainty(X_te.to_numpy())
print("\n  MLP uncertainty head: mean uncertainty on correct vs wrong predictions")
wrong = mlp.predict(X_te.to_numpy()) != y_te.to_numpy()
print(f"    correct {np.mean(np.asarray(unc)[~wrong]):.3f} | wrong {np.mean(np.asarray(unc)[wrong]) if wrong.any() else float('nan'):.3f}")

pca = PCA(n_components=2, variant="robust").fit(X_tr.to_numpy())
print("\n  PCA[robust] 2-D projection shape:", pca.transform(X_te.to_numpy()).shape)
km = KMeans(n_clusters=2, variant="robust").fit(X_tr.to_numpy())
print("  KMeans[robust] cluster sizes:", np.bincount(km.predict(X_te.to_numpy())).tolist())

# ---------------------------------------------------------------- 19. head-to-head
banner("19. HEAD-TO-HEAD: 3 datasets x repeated stratified CV, then Friedman / Nemenyi")
methods = {
    "sc_logistic": StatCoreLogisticClassifier(),
    "sc_lda": StatCoreLDAClassifier(),
    "sc_knn": StatCoreKNNClassifier(),
    "sc_meta": StatCoreMetaClassifier(n_folds=3),
    "meta_pipeline_own": MetaPipelineClassifier(base_estimators=own_bases, n_folds=3),
}
long = []
for ds_name, fn in {"breast_cancer": load_breast_cancer, "wine": load_wine, "iris": load_iris}.items():
    Xs, ys = load(fn)
    for rep in range(2):
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=rep).split(Xs, ys):
            for name, m in methods.items():
                p = clone(m).fit(Xs.iloc[tr], ys.iloc[tr]).predict(Xs.iloc[te])
                long.append({"dataset": ds_name, "method": name, "score": bal_acc(ys.iloc[te], p)})
long = pd.DataFrame(long)
summary = summarize_repeats(long, dataset="dataset", method="method", score="score")
print(summary.round(3).to_string())
cmp = compare_methods(summary, higher_is_better=True)
print("\naverage ranks (1 = best):", {k: round(v, 2) for k, v in cmp.average_ranks.items()})
print("Friedman:", cmp.friedman)
print("(with only 3 datasets the test is under-powered: use 10+ for a real conclusion)")

# ---------------------------------------------------------------- 20. tuning
banner("20. TUNING an own model with successive halving")
search = halving_grid_search(StatCoreKNNClassifier(variant="base"),
                             {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]},
                             cv=3, scoring="balanced_accuracy", random_state=0)
search.fit(X_tr, y_tr)
print("best params:", search.best_params_, "| holdout balanced accuracy:",
      round(bal_acc(y_te, search.best_estimator_.predict(X_te)), 3))
