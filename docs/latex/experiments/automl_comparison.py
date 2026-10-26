"""Family-based AutoML comparison on identical held-out splits.

Local use (only ReliableAutoML and a plain logistic baseline are run on the
development machine):
    .venv-automl\\Scripts\\python.exe docs\\latex\\experiments\\automl_comparison.py
        --frameworks reliableml sklearn_logistic --seeds 42 43 44
        --output docs\\latex\\experiments\\automl-comparison-results.json
Competitor adapters (flaml, tpot, autogluon, h2o, autosklearn) are only meant
for the Colab notebook docs/colab/automl.ipynb; each is wrapped so that an
installation or runtime failure is recorded instead of aborting the run.

Protocol: per (dataset, seed) one stratified 25% test split is made with
train_test_split(random_state=seed); every framework sees the same development
rows and is scored on the same untouched test rows. Competitors get a wall-clock
budget of --budget seconds; ReliableAutoML has no time budget and its measured
wall time is recorded.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split

TARGET = "outcome"
TEST_FRACTION = 0.25


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
                         "plan": plan, TARGET: target})


def load_frame(name: str, seed: int) -> pd.DataFrame:
    if name == "synthetic_mixed":
        return make_mixed(seed)
    loader = {"wine": load_wine, "breast_cancer": load_breast_cancer,
              "digits": load_digits}[name]
    bunch = loader()
    frame = pd.DataFrame(bunch.data, columns=[str(c) for c in bunch.feature_names])
    frame[TARGET] = bunch.target
    return frame


def split(frame: pd.DataFrame, seed: int):
    return train_test_split(frame, test_size=TEST_FRACTION, random_state=seed,
                            stratify=frame[TARGET])


def metric_record(y_true, prediction, probability, classes) -> dict:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-15, 1.0)
    probability = probability / probability.sum(axis=1, keepdims=True)
    return {"accuracy": float(accuracy_score(y_true, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
            "log_loss": float(log_loss(y_true, probability, labels=list(classes)))}


def numeric_design(train_X: pd.DataFrame):
    """Median-impute + one-hot transformer for frameworks needing numeric input."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    numeric = train_X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in train_X.columns if c not in numeric]
    parts = []
    if numeric:
        parts.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            categorical))
    return ColumnTransformer(parts)


def describe_object(obj) -> dict:
    """Describe the structure of a report object without assuming its class."""
    info = {"type": type(obj).__name__}
    if isinstance(obj, dict):
        info["keys"] = sorted(str(k) for k in obj)
    elif hasattr(obj, "__dict__"):
        info["fields"] = sorted(k for k in vars(obj) if not k.startswith("_"))
    return info


# --------------------------------------------------------------------------
# Adapters: each returns (prediction, probability, classes, info)
# --------------------------------------------------------------------------
def run_reliableml(train, test_X, seed, budget):
    from reliableml_automl import ReliableAutoML
    automl = ReliableAutoML(target_column=TARGET, random_state=seed,
                            partitioner="s_attp_clf", include_statcore=True,
                            tuning="light", cv=3, tuning_cv=2,
                            progress_bar=False).fit(train)
    prediction = np.asarray(automl.predict(test_X))
    probability = np.asarray(automl.best_pipeline.predict_proba(test_X))
    classes = list(automl.best_pipeline.classes_)
    fit_report = automl.fit_report
    info = {"selected_model": automl.selected_model_name,
            "leaderboard_size": len(automl.leaderboard),
            "leaderboard": automl.leaderboard,
            "partition_sizes": fit_report.get("partition_sizes"),
            "n_warnings": len(fit_report.get("warnings", [])),
            "cleaning_report_keys": sorted(automl.cleaning_report),
            "fit_report_keys": sorted(fit_report)}
    try:
        _, report = automl.predict_with_report(test_X)
        info["predict_with_report"] = describe_object(report)
    except Exception as error:  # recorded, not fatal
        info["predict_with_report"] = {"error": repr(error)}
    directory = Path(tempfile.mkdtemp(prefix="raml_save_", dir=Path.cwd()))
    try:
        automl.save(directory / "artifact")
        files = [p for p in directory.rglob("*") if p.is_file()]
        info["saved_artifact_files"] = len(files)
        info["saved_artifact_bytes"] = int(sum(p.stat().st_size for p in files))
    except Exception as error:
        info["saved_artifact"] = {"error": repr(error)}
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    return prediction, probability, classes, info


def run_sklearn_logistic(train, test_X, seed, budget):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = train.drop(columns=[TARGET]), train[TARGET]
    pipe = Pipeline([("prep", numeric_design(X)), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000,
                                                class_weight="balanced"))]).fit(X, y)
    return (pipe.predict(test_X), pipe.predict_proba(test_X),
            list(pipe.classes_), {"selected_model": "LogisticRegression"})


def run_flaml(train, test_X, seed, budget):
    from flaml import AutoML
    X, y = train.drop(columns=[TARGET]), train[TARGET]
    automl = AutoML()
    automl.fit(X, y, task="classification", time_budget=budget, seed=seed,
               verbose=0, n_jobs=1)
    return (automl.predict(test_X), automl.predict_proba(test_X),
            list(automl.classes_),
            {"selected_model": str(automl.best_estimator),
             "best_config": str(automl.best_config),
             "optimised_metric": "flaml default (log_loss for classification)"})


def run_tpot(train, test_X, seed, budget):
    from tpot import TPOTClassifier
    X, y = train.drop(columns=[TARGET]), train[TARGET]
    design = numeric_design(X).fit(X)
    tpot = TPOTClassifier(search_space="linear-light",
                          scorers=["balanced_accuracy"], scorers_weights=[1],
                          cv=3, max_time_mins=budget / 60.0, random_state=seed,
                          n_jobs=1, verbose=0)
    tpot.fit(design.transform(X), y)
    model = tpot.fitted_pipeline_
    Xt = design.transform(test_X)
    return (model.predict(Xt), model.predict_proba(Xt), list(model.classes_),
            {"selected_model": str(model)[:300]})


def run_autogluon(train, test_X, seed, budget):
    from autogluon.tabular import TabularPredictor
    directory = Path(tempfile.mkdtemp(prefix="ag_", dir=Path.cwd()))
    try:
        predictor = TabularPredictor(label=TARGET, eval_metric="balanced_accuracy",
                                     path=str(directory), verbosity=0)
        predictor.fit(train, time_limit=budget, presets="best_quality")
        board = predictor.leaderboard(silent=True)
        proba = predictor.predict_proba(test_X)
        info = {"selected_model": str(predictor.model_best),
                "leaderboard_size": int(len(board)),
                "preset": "best_quality"}
        return (predictor.predict(test_X).to_numpy(), proba.to_numpy(),
                list(proba.columns), info)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def run_h2o(train, test_X, seed, budget):
    import h2o
    from h2o.automl import H2OAutoML
    h2o.init(verbose=False, nthreads=1)
    h2o.no_progress()
    try:
        frame = train.copy()
        frame[TARGET] = frame[TARGET].astype(str)
        hf = h2o.H2OFrame(frame)
        hf[TARGET] = hf[TARGET].asfactor()
        aml = H2OAutoML(max_runtime_secs=budget, seed=seed, nfolds=5,
                        sort_metric="mean_per_class_error", verbosity=None)
        aml.train(y=TARGET, training_frame=hf)
        out = aml.leader.predict(h2o.H2OFrame(test_X)).as_data_frame()
        classes = [c for c in out.columns if c != "predict"]
        order = sorted(classes)
        return (out["predict"].to_numpy(), out[order].to_numpy(), order,
                {"selected_model": str(aml.leader.model_id),
                 "leaderboard_size": int(aml.leaderboard.nrows)})
    finally:
        h2o.cluster().shutdown(prompt=False)


def run_autosklearn(train, test_X, seed, budget):
    import autosklearn.classification as ask
    X, y = train.drop(columns=[TARGET]), train[TARGET]
    for column in X.columns:
        if X[column].dtype == object:
            X[column] = X[column].astype("category")
            test_X[column] = pd.Categorical(test_X[column],
                                            categories=X[column].cat.categories)
    model = ask.AutoSklearnClassifier(time_left_for_this_task=int(budget),
                                      seed=seed, n_jobs=1,
                                      metric=__import__("autosklearn.metrics",
                                                        fromlist=["x"]).balanced_accuracy)
    model.fit(X, y)
    return (model.predict(test_X), model.predict_proba(test_X),
            list(model.classes_),
            {"selected_model": f"ensemble of {len(model.get_models_with_weights())} models"})


ADAPTERS = {"reliableml": run_reliableml, "sklearn_logistic": run_sklearn_logistic,
            "flaml": run_flaml, "tpot": run_tpot, "autogluon": run_autogluon,
            "h2o": run_h2o, "autosklearn": run_autosklearn}


def versions() -> dict:
    out = {}
    for name in ("numpy", "pandas", "sklearn", "flaml", "tpot", "autogluon.tabular",
                 "h2o", "autosklearn", "reliableml_automl", "ml_reliability"):
        try:
            module = __import__(name, fromlist=["x"])
            out[name] = getattr(module, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frameworks", nargs="+", default=["reliableml", "sklearn_logistic"])
    parser.add_argument("--datasets", nargs="+",
                        default=["synthetic_mixed", "wine", "breast_cancer", "digits"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--budget", type=float, default=60.0,
                        help="seconds per competitor fit (ReliableAutoML ignores it)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    meta = {"python": platform.python_version(), "platform": platform.platform(),
            "versions": versions(), "seeds": args.seeds, "datasets": args.datasets,
            "test_fraction": TEST_FRACTION, "budget_seconds": args.budget,
            "frameworks": args.frameworks}
    records = []
    for seed in args.seeds:
        for dataset in args.datasets:
            frame = load_frame(dataset, seed)
            train, test = split(frame, seed)
            test_X, y_test = test.drop(columns=[TARGET]), test[TARGET].to_numpy()
            for name in args.frameworks:
                record = {"framework": name, "dataset": dataset, "seed": seed,
                          "n_development": len(train), "n_test": len(test),
                          "n_features": frame.shape[1] - 1,
                          "n_classes": int(frame[TARGET].nunique())}
                start = time.perf_counter()
                try:
                    pred, proba, classes, info = ADAPTERS[name](
                        train.copy(), test_X.copy(), seed, args.budget)
                    record["wall_seconds"] = time.perf_counter() - start
                    record.update(metric_record(y_test, pred, proba, classes))
                    record["info"] = info
                except Exception as error:
                    record["wall_seconds"] = time.perf_counter() - start
                    record["error"] = repr(error)
                    record["traceback_tail"] = traceback.format_exc()[-600:]
                records.append(record)
                shown = {k: record.get(k) for k in
                         ("framework", "dataset", "seed", "balanced_accuracy",
                          "log_loss", "wall_seconds", "error")}
                print(json.dumps(shown, default=str), flush=True)
    if args.output:
        Path(args.output).write_text(
            json.dumps({"metadata": meta, "records": records}, indent=1, default=str),
            encoding="utf-8")


if __name__ == "__main__":
    main()
