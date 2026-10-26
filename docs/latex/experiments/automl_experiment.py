"""Held-out ReliableAutoML experiment on mixed-type, Wine, and Cancer data.

Run with .venv-automl\\Scripts\\python.exe
 docs\\latex\\experiments\\automl_experiment.py --seeds 42.
Each run reserves an outer stratified test split before ReliableAutoML sees
its development rows. The explicit sklearn LogisticRegression pipeline is a
plain baseline; the AutoML leaderboard uses all built-in candidates.
"""
from __future__ import annotations

import argparse
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reliableml_automl import ReliableAutoML


def make_mixed(seed: int, n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(42, 13, n)
    income = rng.lognormal(10.5, 0.55, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    plan = rng.choice(["basic", "plus", "premium"], n, p=[0.5, 0.32, 0.18])
    score = (0.035 * (age - 40) + 0.7 * (plan == "premium")
             - 0.25 * (region == "south") + 0.35 * rng.normal(size=n))
    target = (score > np.quantile(score, 0.53)).astype(int)
    age[rng.random(n) < 0.08] = np.nan
    income[rng.random(n) < 0.07] = np.nan
    region = region.astype(object)
    plan = plan.astype(object)
    region[rng.random(n) < 0.06] = None
    plan[rng.random(n) < 0.05] = None
    return pd.DataFrame({"age": age, "income": income, "region": region,
                         "plan": plan, "outcome": target})


def datasets(seed: int):
    yield "synthetic_mixed", make_mixed(seed)
    for name, loader in (("wine", load_wine), ("breast_cancer", load_breast_cancer)):
        bunch = loader()
        frame = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        frame["outcome"] = bunch.target
        yield name, frame


def metric_record(y_true, prediction, probability):
    return {"accuracy": float(accuracy_score(y_true, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
            "log_loss": float(log_loss(y_true, probability))}


def baseline_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return Pipeline([("prep", ColumnTransformer(transformers)),
                     ("classifier", LogisticRegression(max_iter=2000,
                                                        class_weight="balanced"))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output", default=None,
                        help="optional JSONL output path; stdout is always emitted")
    args = parser.parse_args()
    records = []
    metadata = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                "seeds": args.seeds, "outer_test_fraction": 0.25,
                "outer_cv": 3, "inner_cv": 2, "tuning": "light",
                "scoring": "balanced_accuracy", "include_statcore": True,
                "partitioner": "s_attp_clf"}
    print(json.dumps({"metadata": metadata}))
    for seed in args.seeds:
        for dataset, frame in datasets(seed):
            development, heldout = train_test_split(
                frame, test_size=0.25, random_state=seed,
                stratify=frame["outcome"])
            automl = ReliableAutoML(
                target_column="outcome", random_state=seed,
                partitioner="s_attp_clf", include_statcore=True,
                tuning="light", cv=3, tuning_cv=2,
                progress_bar=False,
            ).fit(development)
            X_test = heldout.drop(columns=["outcome"])
            y_test = heldout["outcome"].to_numpy()
            prediction = np.asarray(automl.predict(X_test))
            probability = np.asarray(automl.best_pipeline.predict_proba(X_test))
            record = {"dataset": dataset, "seed": seed, "model": "automl_selected",
                      "selected_model": automl.selected_model_name,
                      "leaderboard": automl.leaderboard,
                      "n_development": len(development), "n_heldout": len(heldout),
                      **metric_record(y_test, prediction, probability)}
            records.append(record)
            print(json.dumps(record, default=str))

            X_development = development.drop(columns=["outcome"])
            baseline = baseline_pipeline(X_development).fit(
                X_development, development["outcome"])
            baseline_probability = baseline.predict_proba(X_test)
            baseline_record = {
                "dataset": dataset, "seed": seed,
                "model": "sklearn_logistic_baseline",
                "selected_model": None,
                "n_development": len(development), "n_heldout": len(heldout),
                **metric_record(y_test, baseline.predict(X_test), baseline_probability)}
            records.append(baseline_record)
            print(json.dumps(baseline_record))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            for item in records:
                stream.write(json.dumps(item, default=str) + "\n")


if __name__ == "__main__":
    main()

