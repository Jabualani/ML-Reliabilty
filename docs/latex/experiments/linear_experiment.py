"""Reproducible LogisticRegression and RidgeRegression comparisons."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from ml_reliability.statcore.linear.linear import (  # noqa: E402
    LogisticRegression,
    RidgeRegression,
    VARIANTS,
)

SEEDS = (11, 23, 37, 51, 79)
CLASS_CASES = ("clean", "shifted", "label_noise", "outliers", "high_dim", "breast_cancer")
REG_CASES = ("clean", "shifted", "noisy", "outliers", "high_dim", "diabetes")


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(round(0.25 * n))
    return np.sort(idx[k:]), np.sort(idx[:k])


def class_data(case: str, seed: int):
    rng = np.random.default_rng(seed)
    if case == "breast_cancer":
        ds = load_breast_cancer()
        tr, te = split_indices(len(ds.target), seed)
        return ds.data[tr], ds.target[tr], ds.data[te], ds.target[te]
    ntr, nte = (140, 300) if case == "high_dim" else (360, 600)
    d = 80 if case == "high_dim" else 12
    info = 10 if case == "high_dim" else min(d, 8)
    beta = np.zeros(d)
    beta[:info] = np.array([1.2, -1.0, 0.8, -0.7, 0.5, -0.4, 0.3, -0.2, 0.15, -0.1])[:info]
    shift = np.zeros(d)
    if case == "shifted":
        shift[:info] = np.linspace(0.35, 0.9, info)

    def sample(n: int, offset: np.ndarray):
        x = rng.normal(size=(n, d)) + offset
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
        return x, rng.binomial(1, p)

    xtr, ytr = sample(ntr, np.zeros(d))
    xte, yte = sample(nte, shift)
    if case == "label_noise":
        flip = rng.random(ntr) < 0.15
        ytr[flip] = 1 - ytr[flip]
    elif case == "outliers":
        out = rng.random(ntr) < 0.08
        xtr[out] = rng.normal(0, 6, size=(out.sum(), d))
    return xtr, ytr, xte, yte


def class_scores(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    pred = p.argmax(axis=1)
    acc = float(np.mean(pred == y))
    loss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())
    conf, correct = p.max(axis=1), pred == y
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for j in range(10):
        mask = (conf >= edges[j]) & (conf < edges[j + 1] if j < 9 else conf <= edges[j + 1])
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return acc, loss, float(ece)


def regression_data(case: str, seed: int):
    rng = np.random.default_rng(seed)
    if case == "diabetes":
        ds = load_diabetes()
        tr, te = split_indices(len(ds.target), seed)
        return ds.data[tr], ds.target[tr], ds.data[te], ds.target[te]
    ntr, nte = (100, 300) if case == "high_dim" else (360, 600)
    d = 80 if case == "high_dim" else 12
    info = 10 if case == "high_dim" else min(d, 8)
    beta = np.zeros(d)
    beta[:info] = np.linspace(1.5, -0.7, info)
    offset = np.zeros(d)
    if case == "shifted":
        offset[:info] = np.linspace(0.4, 0.9, info)
    xtr = rng.normal(size=(ntr, d))
    xte = rng.normal(size=(nte, d)) + offset
    ytr = xtr @ beta + rng.normal(0, 1.0, ntr)
    yte = xte @ beta + rng.normal(0, 1.0, nte)
    if case == "noisy":
        ytr += rng.normal(0, 3.0, ntr)
    elif case == "outliers":
        out = rng.random(ntr) < 0.08
        ytr[out] += rng.normal(0, 20.0, out.sum())
    return xtr, ytr, xte, yte


def reg_scores(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    err = pred - y
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")
    return mse, mae, r2


def summarize(title: str, cases: tuple[str, ...], methods: list[str], scores: dict, metrics: tuple[str, ...]) -> None:
    print(title)
    print("case,method," + ",".join(f"{m}_{s}" for m in metrics for s in ("mean", "sd")))
    for case in cases:
        for method in methods:
            a = np.asarray(scores[(case, method)])
            means, sds = a.mean(axis=0), a.std(axis=0, ddof=1)
            print(f"{case},{method}," + ",".join(f"{v:.4f}" for pair in zip(means, sds) for v in pair))


def main() -> None:
    methods = ["sklearn_standard", *VARIANTS]
    c_scores: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for case in CLASS_CASES:
        for seed in SEEDS:
            xtr, ytr, xte, yte = class_data(case, seed)
            baseline = make_pipeline(
                StandardScaler(),
                SklearnLogisticRegression(C=1.0 / (len(ytr) * 1e-2), max_iter=1000, random_state=seed),
            )
            baseline.fit(xtr, ytr)
            c_scores.setdefault((case, "sklearn_standard"), []).append(class_scores(yte, baseline.predict_proba(xte)))
            for variant in VARIANTS:
                model = LogisticRegression(variant=variant, random_state=seed)
                with np.errstate(divide="ignore", invalid="ignore"):
                    model.fit(xtr, ytr)
                c_scores.setdefault((case, variant), []).append(class_scores(yte, model.predict_proba(xte)))
    summarize("CLASSIFICATION (accuracy, log-loss, 10-bin confidence ECE)", CLASS_CASES, methods, c_scores,
              ("accuracy", "logloss", "ece"))

    r_scores: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for case in REG_CASES:
        for seed in SEEDS:
            xtr, ytr, xte, yte = regression_data(case, seed)
            baseline = make_pipeline(StandardScaler(), SklearnRidge(alpha=1.0))
            baseline.fit(xtr, ytr)
            r_scores.setdefault((case, "sklearn_standard"), []).append(reg_scores(yte, baseline.predict(xte)))
            for variant in VARIANTS:
                model = RidgeRegression(variant=variant, random_state=seed)
                model.fit(xtr, ytr)
                r_scores.setdefault((case, variant), []).append(reg_scores(yte, model.predict(xte)))
    summarize("REGRESSION (MSE, MAE, R-squared)", REG_CASES, methods, r_scores, ("mse", "mae", "r2"))


if __name__ == "__main__":
    main()

