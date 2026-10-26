import json, numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from ml_reliability import *

d = load_breast_cancer(as_frame=True)
X = d.data.iloc[:, :6].copy(); y = d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
rng = np.random.default_rng(0)
grp = pd.Series(rng.choice(["A", "B"], size=len(Xte)), index=Xte.index)
m = ReliablePipeline(MetaPipelineClassifier(n_folds=3, random_state=0), plot_dir="plots", bias_variance_repeats=1, bias_variance_folds=3).fit(Xtr, ytr)
ev = m.evaluate(Xte, yte, sensitive_features={"grp": grp.values}, positive_label=1)
def show(o, depth=0, maxd=3):
    if isinstance(o, dict) and depth < maxd:
        for k, v in o.items():
            print("  " * depth + str(k), type(v).__name__ if isinstance(v, (dict, list)) else repr(v)[:80])
            show(v, depth + 1, maxd)
show(ev.to_dict(), 0, 4)
ex = m.explain(Xte, class_label=1, max_instances=10, num_samples=100)
print(ex.global_shap_importance[:2]); print(ex.local_explanations[0].keys()); print(ex.reason_alignment)
au = m.audit_reasons(Xte, yte, positive_label=1, bootstrap_samples=100, explanation_samples=50, max_explained_instances=10,
    groups={"grp": grp.values}, counterfactuals={X.columns[0]: np.full(len(Xte), X.iloc[:,0].median())})
show(au.to_dict(), 0, 3)
print(au.feature_reliance[:2])
