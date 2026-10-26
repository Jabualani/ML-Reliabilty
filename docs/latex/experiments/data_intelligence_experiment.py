"""Data-intelligence experiments: planted-outlier benchmark and real-data runs.

Writes ``data_intelligence-results.json`` here and the paper figures to ``../figures``.
Run from the repository root:  PYTHONPATH=src python docs/latex/experiments/data_intelligence_experiment.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ml_reliability import (
    DataIntelligence,
    SelfOrganisingMap,
    isolation_forest_scores,
    local_outlier_factor,
    robust_mahalanobis,
)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
SEEDS = range(10)
N_OUT = 25


def auroc(score, truth):
    ranks = stats.rankdata(score)
    n1, n0 = truth.sum(), (~truth).sum()
    return float((ranks[truth].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def scenario(name, seed):
    rng = np.random.default_rng(seed)
    p = 6
    sigma = 0.8 * np.ones((p, p)) + 0.2 * np.eye(p)
    if name == "global":
        X = rng.multivariate_normal(np.zeros(p), 0.6 * np.ones((p, p)) + 0.4 * np.eye(p), 475)
        O = rng.normal(size=(N_OUT, p)) + 5.0 * np.sign(rng.normal(size=(N_OUT, p)))
    elif name == "correlation_breaking":
        X = rng.multivariate_normal(np.zeros(p), sigma, 475)
        O = rng.normal(size=(N_OUT, p))
    elif name == "local_density":
        X = np.vstack([rng.normal(scale=0.3, size=(250, p)), rng.normal(loc=8.0, scale=3.0, size=(225, p))])
        direction = rng.normal(size=(N_OUT, p))
        O = 1.6 * direction / np.linalg.norm(direction, axis=1, keepdims=True) * np.sqrt(p) / 2.5
    elif name == "between_clusters":
        centres = np.array([[-6.0] * p, [0.0] * p, [6.0] * p])
        X = np.vstack([rng.normal(loc=c, scale=0.7, size=(159, p)) for c in centres])[:475]
        O = rng.normal(loc=-3.0, scale=0.4, size=(N_OUT, p))
    else:
        raise ValueError(name)
    data = np.vstack([X, O])
    truth = np.r_[np.zeros(len(X), bool), np.ones(len(O), bool)]
    return data, truth


def detector_scores(Z, seed):
    sd = Z.std(0)
    std = (Z - Z.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
    iso = isolation_forest_scores(std, random_state=seed)
    lof = local_outlier_factor(std, k=20)
    d2, _, _ = robust_mahalanobis(Z)
    som = SelfOrganisingMap(random_state=seed).fit(std).quantization_error_row_
    ens = np.mean([stats.rankdata(v) / len(v) for v in (iso, lof, d2, som)], axis=0)
    return {"isolation_forest": iso, "lof": lof, "mahalanobis": d2, "som_quantization": som, "ensemble_rank": ens}


def benchmark():
    out = {}
    for name in ("global", "correlation_breaking", "local_density", "between_clusters"):
        per = {}
        for seed in SEEDS:
            Z, truth = scenario(name, seed)
            for method, score in detector_scores(Z, seed).items():
                top = np.argsort(-score)[:N_OUT]
                per.setdefault(method, {"auroc": [], "precision_at_k": []})
                per[method]["auroc"].append(auroc(score, truth))
                per[method]["precision_at_k"].append(float(truth[top].mean()))
        out[name] = {m: {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()} for m, d in per.items()}
    return out


def real_runs():
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

    rows = {}
    for name, loader in (("breast_cancer", load_breast_cancer), ("wine", load_wine), ("diabetes", load_diabetes)):
        frame = loader(as_frame=True).frame
        t0 = time.time()
        rep = DataIntelligence(mode="autonomous").analyze(frame, target_column="target")
        ens = rep["why"]["outliers"].get("ensemble", {})
        rows[name] = {
            "rows": len(frame), "columns": frame.shape[1], "summary_score": rep.summary_score,
            "target_verdict": rep["target_intelligence"]["explainability"]["verdict"],
            "consensus_outlier_rows": ens.get("consensus", {}).get("n_rows"),
            "som_quantization_error": ens.get("som", {}).get("quantization_error"),
            "som_topographic_error": ens.get("som", {}).get("topographic_error"),
            "seconds": round(time.time() - t0, 1),
        }
    return rows


def paper_figures():
    from sklearn.datasets import load_breast_cancer

    FIGURES.mkdir(exist_ok=True)
    d = load_breast_cancer(as_frame=True).frame
    rng = np.random.default_rng(0)
    d["when"] = pd.date_range("2023-01-01", periods=len(d), freq="D")
    d["leaky_score"] = d["target"] + rng.normal(0, 0.01, len(d))
    d.loc[::9, "mean radius"] = np.nan
    cols = [c for c in d.columns if c not in ("target", "when", "leaky_score")][:8]
    d.loc[:7, cols] = d.loc[:7, cols] * rng.uniform(2.5, 4.0, size=(8, len(cols)))
    di = DataIntelligence(mode="autonomous", plot_dir=str(FIGURES))
    rep = di.analyze(d, target_column="target", time_column="when")
    return {"summary_score": rep.summary_score, "figures": sorted(rep["visualizations"]),
            "consensus_rows": rep["why"]["outliers"]["ensemble"]["consensus"]["n_rows"]}


if __name__ == "__main__":
    results = {"benchmark": benchmark(), "real": real_runs(), "figures": paper_figures(),
               "settings": {"seeds": len(list(SEEDS)), "planted_outliers": N_OUT, "rows": 500}}
    (HERE / "data_intelligence-results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
