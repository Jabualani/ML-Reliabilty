"""Shared helpers for StatCore: validation, folds, softmax, small optimisers."""

from __future__ import annotations

import numpy as np


class NotFittedError(RuntimeError):
    """Raised when a model is used before ``fit``."""


def check_X(X, name="X", min_samples=1, n_features=None):
    X = np.asarray(X)
    if X.dtype == object or not (np.issubdtype(X.dtype, np.number) or X.dtype == bool):
        try:
            X = X.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
    X = X.astype(float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {X.shape}")
    if X.shape[0] < min_samples:
        raise ValueError(f"{name} needs at least {min_samples} samples, got {X.shape[0]}")
    if X.shape[1] < 1:
        raise ValueError(f"{name} must have at least one feature")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"{name} has {X.shape[1]} features, expected {n_features}")
    return X


def check_y(y, n_samples, numeric=False, name="y"):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {y.shape}")
    if y.shape[0] != n_samples:
        raise ValueError(f"{name} has {y.shape[0]} samples but X has {n_samples}")
    if numeric:
        try:
            y = y.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not np.all(np.isfinite(y)):
            raise ValueError(f"{name} contains NaN or infinite values")
    elif y.dtype.kind == "f" and not np.all(np.isfinite(y)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return y


def check_Xy(X, y, numeric_y=False, min_samples=2):
    X = check_X(X, min_samples=min_samples)
    y = check_y(y, X.shape[0], numeric=numeric_y)
    return X, y


def check_fitted(est, attr="is_fitted_"):
    if not getattr(est, attr, False):
        raise NotFittedError(f"{type(est).__name__} is not fitted; call fit() first")


def check_variant(variant, allowed, owner="model"):
    if variant not in allowed:
        raise ValueError(f"Unknown {owner} variant {variant!r}; choose from {tuple(allowed)}")
    return variant


def softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def stratified_folds(y, n_folds, seed):
    """Return an array of fold ids, stratified by label, deterministic in seed."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    offset = 0
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        folds[idx] = (np.arange(len(idx)) + offset) % n_folds
        offset += len(idx)
    return folds


def kfold_ids(n, n_folds, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.empty(n, dtype=int)
    folds[perm] = np.arange(n) % n_folds
    return folds


def holdout_split(n, fraction, seed, stratify=None):
    """Return (train_idx, hold_idx); stratified when labels are given."""
    rng = np.random.default_rng(seed)
    if stratify is None:
        perm = rng.permutation(n)
        k = max(1, int(round(n * fraction)))
        return np.sort(perm[k:]), np.sort(perm[:k])
    hold = []
    for c in np.unique(stratify):
        idx = np.flatnonzero(stratify == c)
        rng.shuffle(idx)
        k = int(round(len(idx) * fraction))
        if len(idx) > 1:
            k = min(max(k, 1), len(idx) - 1)
        hold.append(idx[:k])
    hold = np.sort(np.concatenate(hold))
    mask = np.ones(n, dtype=bool)
    mask[hold] = False
    return np.flatnonzero(mask), hold


def golden_min(f, lo, hi, iters=60):
    """Minimise a unimodal scalar function on [lo, hi]."""
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def pairwise_sq_dists(A, B):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


class Standardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        s = X.std(axis=0)
        self.scale_ = np.where(s > 1e-12, s, 1.0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_
