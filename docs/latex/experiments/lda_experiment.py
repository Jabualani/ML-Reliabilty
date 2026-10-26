"""Reproducible LDA comparison; run with the repository .venv Python."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from ml_reliability.statcore.lda.lda import LDA, VARIANTS  # noqa: E402

SEEDS = (11, 23, 37, 51, 79)
CASES = ("clean", "shifted", "label_noise", "outliers", "high_dim", "heteroscedastic", "wine")


def softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)
    p = np.exp(a)
    return p / p.sum(axis=1, keepdims=True)


def stratified_split(y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * fraction)))
        n_test = min(n_test, len(idx) - 1)
        te.extend(idx[:n_test])
        tr.extend(idx[n_test:])
    return np.array(sorted(tr)), np.array(sorted(te))


def synthetic(case: str, seed: int):
    rng = np.random.default_rng(seed)
    ntr, nte = (120, 300) if case == "high_dim" else (360, 600)
    d = 80 if case == "high_dim" else 12
    informative = 10 if case == "high_dim" else min(8, d)
    classes = 3
    centers = np.zeros((classes, d))
    centers[:, :informative] = np.array([
        np.r_[-2.0, 0.0, np.zeros(informative - 2)],
        np.r_[2.0, 0.0, np.zeros(informative - 2)],
        np.r_[0.0, 2.2, np.zeros(informative - 2)],
    ])
    scale = np.ones((classes, d))
    if case == "heteroscedastic":
        scale[0, :informative] = 0.55
        scale[1, :informative] = 1.8
    shift = np.zeros(d)
    if case == "shifted":
        shift[:informative] = np.linspace(0.5, 1.1, informative)

    def sample(n: int, offset: np.ndarray):
        y = rng.integers(classes, size=n)
        x = centers[y] + rng.normal(size=(n, d)) * scale[y] + offset
        return x, y

    xtr, ytr = sample(ntr, np.zeros(d))
    xte, yte = sample(nte, shift)
    if case == "label_noise":
        noisy = rng.random(ntr) < 0.15
        ytr[noisy] = (ytr[noisy] + rng.integers(1, classes, noisy.sum())) % classes
    elif case == "outliers":
        out = rng.random(ntr) < 0.08
        xtr[out] = rng.normal(0, 6, size=(out.sum(), d))
    xadapt, _ = sample(500, shift)
    return xtr, ytr, xte, yte, (xadapt if case == "shifted" else None)


def dataset(case: str, seed: int):
    if case != "wine":
        return synthetic(case, seed)
    ds = load_wine()
    tr, te = stratified_split(ds.target, 0.25, seed)
    return ds.data[tr], ds.target[tr], ds.data[te], ds.target[te], None


def metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    classes = np.arange(p.shape[1])
    pred = classes[p.argmax(axis=1)]
    accuracy = float(np.mean(pred == y))
    logloss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)).mean())
    confidence = p.max(axis=1)
    correct = pred == y
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for j in range(10):
        mask = (confidence >= edges[j]) & (confidence < edges[j + 1] if j < 9 else confidence <= edges[j + 1])
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return accuracy, logloss, float(ece)


def main() -> None:
    methods = ["sklearn_LDA", *VARIANTS]
    scores: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for case in CASES:
        for seed in SEEDS:
            xtr, ytr, xte, yte, xadapt = dataset(case, seed)
            reference = LinearDiscriminantAnalysis(solver="lsqr")
            reference.fit(xtr, ytr)
            scores.setdefault((case, "sklearn_LDA"), []).append(metrics(yte, reference.predict_proba(xte)))
            for variant in VARIANTS:
                model = LDA(variant=variant, random_state=seed)
                if variant == "shift_corrected":
                    if xadapt is None:
                        # Supply a disjoint, unlabeled sample from the source domain.
                        xadapt = xtr.copy()
                    model.fit(xtr, ytr, X_target=xadapt)
                else:
                    model.fit(xtr, ytr)
                scores.setdefault((case, variant), []).append(metrics(yte, model.predict_proba(xte)))
    print("case,method,accuracy_mean,accuracy_sd,logloss_mean,logloss_sd,ece_mean,ece_sd")
    for case in CASES:
        for method in methods:
            a = np.asarray(scores[(case, method)])
            means, sds = a.mean(axis=0), a.std(axis=0, ddof=1)
            print(f"{case},{method}," + ",".join(f"{v:.4f}" for pair in zip(means, sds) for v in pair))


if __name__ == "__main__":
    main()
