"""Shared evaluation metrics for all StatCore models (pure NumPy)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def classification_metrics(y_true, y_pred, y_proba: Optional[np.ndarray] = None, classes=None) -> Dict[str, float]:
    """accuracy, macro precision/recall/F1 and multiclass MCC (Gorodkin's R_K).

    If ``y_proba`` (columns ordered as sorted classes) is given, ``log_loss`` and
    ``ece`` (expected calibration error on max-probability) are added.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    union = np.unique(np.concatenate([y_true, y_pred]))
    prec, rec, f1 = [], [], []
    for c in union:
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        prec.append(p)
        rec.append(r)
        f1.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    n = len(y_true)
    idx_t = np.searchsorted(union, y_true)
    idx_p = np.searchsorted(union, y_pred)
    C = np.zeros((len(union), len(union)))
    np.add.at(C, (idx_t, idx_p), 1.0)
    t_k, p_k, correct = C.sum(1), C.sum(0), np.trace(C)
    cov_xy = correct * n - float(t_k @ p_k)
    cov_xx = n ** 2 - float(p_k @ p_k)
    cov_yy = n ** 2 - float(t_k @ t_k)
    denom = np.sqrt(cov_xx * cov_yy)
    out = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "f1_macro": float(np.mean(f1)),
        "mcc": float(cov_xy / denom) if denom > 0 else 0.0,
    }
    if y_proba is not None:
        out.update(calibration_metrics(y_true, y_proba, classes=classes))
    return out


def calibration_metrics(y_true, y_proba, n_bins: int = 10, classes=None) -> Dict[str, float]:
    """log_loss and ece; columns of y_proba follow ``classes`` (default sorted unique y_true)."""
    y_true = np.asarray(y_true).ravel()
    P = np.asarray(y_proba, dtype=float)
    classes = np.unique(y_true) if classes is None else np.asarray(classes)
    if P.ndim != 2 or P.shape[0] != len(y_true):
        raise ValueError("y_proba must be (n_samples, n_classes)")
    if P.shape[1] < len(classes):
        raise ValueError("y_proba has fewer columns than classes in y_true")
    pred_idx = P.argmax(1)
    conf = P.max(1)
    t_idx = np.clip(np.searchsorted(classes, y_true), 0, len(classes) - 1)
    known = classes[t_idx] == y_true
    p_true = np.where(known, P[np.arange(len(y_true)), t_idx], 0.0)
    ll = -np.mean(np.log(np.clip(p_true, 1e-12, 1.0)))
    acc = ((pred_idx == t_idx) & known).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return {"log_loss": float(ll), "ece": float(ece)}


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    resid = y_true - y_pred
    mse = float(np.mean(resid ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else 0.0
    return {"mse": mse, "mae": float(np.mean(np.abs(resid))), "r2": r2}


def reconstruction_metrics(X_true, X_pred) -> Dict[str, float]:
    X_true = np.asarray(X_true, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    if X_true.shape != X_pred.shape:
        raise ValueError("X_true and X_pred must have the same shape")
    diff = X_true - X_pred
    return {
        "reconstruction_mse": float(np.mean(diff ** 2)),
        "reconstruction_mae": float(np.mean(np.abs(diff))),
        "relative_frobenius": float(np.linalg.norm(diff) / (np.linalg.norm(X_true) + 1e-12)),
    }


def clustering_metrics(X, labels, centers=None) -> Dict[str, float]:
    """inertia (sum of squared distances to ``centers`` or to cluster means) and n_clusters_found."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    inertia = 0.0
    for i, c in enumerate(unique):
        pts = X[labels == c]
        center = pts.mean(axis=0) if centers is None else np.asarray(centers)[int(c)]
        inertia += float(np.sum((pts - center) ** 2))
    return {"inertia": float(inertia), "n_clusters_found": int(len(unique))}
