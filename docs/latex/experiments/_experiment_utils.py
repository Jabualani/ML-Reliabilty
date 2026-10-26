from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def _synthetic(kind: str, seed: int, n: int = 640):
    rng = np.random.default_rng(seed + (10000 if kind == "regression" else 0))
    x = rng.uniform(-2.0, 2.0, size=(n, 2))
    noise = rng.normal(size=n)
    if kind == "classification":
        y = ((x[:, 0] ** 2 + 0.65 * x[:, 1] ** 2 + 0.25 * noise) > 1.65).astype(int)
        flip = rng.random(n) < 0.10
        y[flip] = 1 - y[flip]
    else:
        y = np.sin(1.4 * x[:, 0]) + 0.35 * x[:, 1] ** 2 + 0.30 * noise
    irrelevant = rng.normal(size=(n, 18))
    return np.column_stack((x, irrelevant)), y


def datasets(kind: str, seed: int):
    X, y = _synthetic(kind, seed)
    yield "synthetic_nonlinear_irrelevant", X, y
    bunch = load_breast_cancer() if kind == "classification" else load_diabetes()
    yield "sklearn_breast_cancer" if kind == "classification" else "sklearn_diabetes", bunch.data, bunch.target


def split_data(X, y, seed: int, stratify: bool):
    rng = np.random.default_rng(seed)
    n_test = int(np.ceil(0.25 * len(y)))
    if not stratify:
        order = rng.permutation(len(y))
        return X[order[n_test:]], X[order[:n_test]], y[order[n_test:]], y[order[:n_test]]
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    raw = counts * (n_test / len(y))
    allocation = np.floor(raw).astype(int)
    remaining = n_test - int(allocation.sum())
    tie_break = rng.random(len(counts))
    order = np.lexsort((tie_break, -(raw - allocation)))
    allocation[order[:remaining]] += 1
    train_ids, test_ids = [], []
    for c, count in enumerate(allocation):
        ids = np.flatnonzero(inverse == c)
        ids = rng.permutation(ids)
        test_ids.extend(ids[:count])
        train_ids.extend(ids[count:])
    train_ids = rng.permutation(np.asarray(train_ids, dtype=int))
    test_ids = rng.permutation(np.asarray(test_ids, dtype=int))
    return X[train_ids], X[test_ids], y[train_ids], y[test_ids]


def ece_equal_width(y, prob, bins: int = 15) -> float:
    pred = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ans = 0.0
    for j in range(bins):
        mask = (conf >= edges[j]) & ((conf < edges[j + 1]) if j < bins - 1 else (conf <= edges[j + 1]))
        if mask.any():
            ans += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ans)


def _classification_metrics(y, prob):
    prob = np.clip(np.asarray(prob), 1e-12, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    truth = np.searchsorted(np.unique(y), y)
    return {
        "accuracy": float(np.mean(truth == prob.argmax(axis=1))),
        "log_loss": float(-np.log(prob[np.arange(len(y)), truth]).mean()),
        "ece": ece_equal_width(truth, prob),
    }


def _regression_metrics(y, pred):
    residual = np.asarray(y) - np.asarray(pred)
    total = np.sum((y - np.mean(y)) ** 2)
    return {"mse": float(np.mean(residual ** 2)), "mae": float(np.mean(np.abs(residual))),
            "r2": float(1.0 - np.sum(residual ** 2) / total)}


def _make_models(family: str, kind: str, seed: int):
    from ml_reliability.statcore.knn.knn import KNNClassifier, KNNRegressor
    from ml_reliability.statcore.mlp.mlp import MLPClassifier, MLPRegressor

    is_class = kind == "classification"
    if family == "knn":
        baseline = KNeighborsClassifier(n_neighbors=5) if is_class else KNeighborsRegressor(n_neighbors=5)
        baseline = make_pipeline(StandardScaler(), baseline)
        variants = {
            name: (KNNClassifier if is_class else KNNRegressor)(
                variant=name, n_neighbors=5, weights="uniform", compression=0.35,
                metric_iters=40, metric_lr=0.1, random_state=seed,
            )
            for name in ("base", "prototype_compressed", "learnable_metric", "stat_k_selection")
        }
        return {"sklearn_standard": baseline, **variants}

    baseline_cls = SklearnMLPClassifier if is_class else SklearnMLPRegressor
    baseline = make_pipeline(
        StandardScaler(),
        baseline_cls(hidden_layer_sizes=(32, 32), alpha=1e-2, batch_size=64,
                     learning_rate_init=1e-2, max_iter=100, early_stopping=True,
                     n_iter_no_change=12, random_state=seed),
    )
    variants = {
        name: (MLPClassifier if is_class else MLPRegressor)(
            hidden_layer_sizes=(32, 32), variant=name, learning_rate=1e-2,
            weight_decay=1e-2, max_epochs=100, batch_size=64, patience=12,
            validation_fraction=0.15, dropout=0.1, mc_samples=20, head_fraction=0.2,
            random_state=seed,
        )
        for name in ("base", "stat_regularised", "reliability_head", "uncertainty_head")
    }
    return {"sklearn_standard": baseline, **variants}


def run(family: str):
    rows = []
    seeds = (11, 23, 47)
    # Report both classification and regression for each model family.
    for task in ("classification", "regression"):
        for seed in seeds:
            for dataset, X, y in datasets(task, seed):
                Xtr, Xte, ytr, yte = split_data(
                    X, y, seed=seed, stratify=(task == "classification")
                )
                for name, model in _make_models(family, task, seed).items():
                    # Warm up the estimator-specific numerical path before timing.
                    model.fit(Xtr, ytr)
                    if task == "classification":
                        model.predict_proba(Xte)
                    else:
                        model.predict(Xte)
                    start = time.perf_counter()
                    model.fit(Xtr, ytr)
                    if task == "classification":
                        if name == "sklearn_standard":
                            prob = model.predict_proba(Xte)
                            # sklearn pipeline class order matches the sorted labels in these datasets.
                            classes = model.classes_
                            if not np.array_equal(classes, np.arange(len(classes))):
                                order = [int(np.flatnonzero(classes == c)[0]) for c in np.arange(len(classes))]
                                prob = prob[:, order]
                        else:
                            prob = model.predict_proba(Xte)
                        metrics = _classification_metrics(yte, prob)
                    else:
                        pred = model.predict(Xte)
                        metrics = _regression_metrics(yte, pred)
                    elapsed = time.perf_counter() - start
                    rows.append({"family": family, "task": task, "dataset": dataset,
                                 "seed": seed, "model": name, "runtime_s": elapsed, **metrics})
    _summarise(rows, family)


def _summarise(rows, family):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["dataset"], row["model"])].append(row)
    print(f"\n{family.upper()} experiment; mean +/- sample SD over 3 predefined seeds")
    print("Each record uses a 75/25 split; runtime is fit plus prediction after one untimed warm-up.")
    for task in ("classification", "regression"):
        metrics = ("accuracy", "log_loss", "ece", "runtime_s") if task == "classification" else ("mse", "mae", "r2", "runtime_s")
        print(f"\n[{task}]")
        print("dataset,model," + ",".join(f"{m}_mean,{m}_sd" for m in metrics))
        for dataset in ("synthetic_nonlinear_irrelevant", "sklearn_breast_cancer" if task == "classification" else "sklearn_diabetes"):
            for model in (["sklearn_standard"] + (["base", "prototype_compressed", "learnable_metric", "stat_k_selection"] if family == "knn" else ["base", "stat_regularised", "reliability_head", "uncertainty_head"])):
                vals = grouped[(task, dataset, model)]
                cells = []
                for metric in metrics:
                    a = np.array([v[metric] for v in vals], dtype=float)
                    sd = a.std(ddof=1) if len(a) > 1 else 0.0
                    cells.extend((f"{a.mean():.4f}", f"{sd:.4f}"))
                print(dataset + "," + model + "," + ",".join(cells))


if __name__ == "__main__":
    if "__FAMILY__" == "__FAMILY__":
        run("__FAMILY__")

