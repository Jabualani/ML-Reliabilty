import json, numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml_reliability import *

rng = np.random.default_rng(0)
n = 1200
X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
y = (X["a"] + 0.5 * X["b"] + rng.normal(scale=1.0, size=n) > 0).astype(int)
X["leak"] = y + rng.normal(scale=0.3, size=n)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
rf = RandomForestClassifier(100, random_state=0, min_samples_leaf=5).fit(Xtr, ytr)
au = audit_model_reasons(rf, Xte, yte, background_data=Xtr, positive_label=1, unavailable_features=("leak",), negative_control_features=("c",),
   bootstrap_samples=100, explanation_samples=100, max_explained_instances=20)
a = au.to_dict()
print(a["feature_reliance"])
print(a["warnings"])
print(json.dumps(a["explanations"], default=str)[:1500])
print(json.dumps(a["explanation_alignment"], default=str)[-800:])
print(a["stability"].keys())

ex = explain_estimator(rf, Xte, background_data=Xtr, class_label=1, max_instances=5, num_samples=200)
l = ex.local_explanations[0]
print(json.dumps(l, default=str)[:1500])
print(ex.warnings)
