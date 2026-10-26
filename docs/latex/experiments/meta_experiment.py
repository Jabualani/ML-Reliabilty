"""Reproducible held-out comparison of StatCore MetaLearner variants.

Run from the repository root with .venv\\Scripts\\python.exe
 docs\\latex\\experiments\\meta_experiment.py --seeds 42.
The only sklearn estimator in this experiment is the explicit StackingClassifier
reference; StatCore component learners are from the NumPy StatCore package.
"""
from __future__ import annotations

import argparse
import json
import platform

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ml_reliability.statcore import KNNClassifier, LDA, LogisticRegression, MetaLearner
from ml_reliability.statcore.evaluation.metrics import classification_metrics


def datasets():
    for name, loader in (("wine", load_wine), ("breast_cancer", load_breast_cancer)):
        bunch = loader()
        yield name, np.asarray(bunch.data, dtype=float), np.asarray(bunch.target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    variants = ("performance_aware", "metrics_based", "oof_stacking",
                "statistical_meta", "calibrated_stack")
    base = [
        ("logistic", LogisticRegression(alpha=1e-2, random_state=0)),
        ("lda", LDA(variant="shrinkage", random_state=0)),
        ("knn", KNNClassifier(n_neighbors=5, variant="base", random_state=0)),
    ]
    baseline = StackingClassifier(
        estimators=[
            ("logistic", make_pipeline(StandardScaler(), SkLogisticRegression(max_iter=1500))),
            ("lda", LinearDiscriminantAnalysis()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ], final_estimator=SkLogisticRegression(max_iter=1500),
        stack_method="predict_proba", cv=3, n_jobs=1,
    )
    print(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                      "scikit_learn": sklearn.__version__, "seeds": args.seeds,
                      "protocol": "stratified 25% held-out; 3-fold OOF; seed-specific"}))
    for dataset, X, y in datasets():
        for seed in args.seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=y)
            for variant in variants:
                estimator = MetaLearner(base, meta_variant=variant, n_folds=3,
                                         random_state=seed).fit(X_train, y_train)
                result = classification_metrics(y_test, estimator.predict(X_test),
                    estimator.predict_proba(X_test), classes=estimator.classes_)
                print(json.dumps({"dataset": dataset, "seed": seed, "model": variant,
                                  "n_train": len(y_train), "n_test": len(y_test), **result}))
            model = baseline.fit(X_train, y_train)
            result = classification_metrics(y_test, model.predict(X_test),
                model.predict_proba(X_test), classes=model.classes_)
            print(json.dumps({"dataset": dataset, "seed": seed,
                              "model": "sklearn_stacking_baseline",
                              "n_train": len(y_train), "n_test": len(y_test), **result}))


if __name__ == "__main__":
    main()
