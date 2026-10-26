"""PCA family (pure NumPy) with reliability-oriented variants."""

from __future__ import annotations

import numpy as np

from .._common import check_X, check_fitted, check_variant, check_y
from ..evaluation.metrics import reconstruction_metrics

VARIANTS = ("base", "supervised_row_aware", "robust", "sparse", "probabilistic")


class PCA:
    """Principal component analysis with selectable variants.

    Variants
    --------
    base : SVD of the centred data matrix.
    supervised_row_aware : weighted PCA; rows are down-weighted by leverage/outlyingness and, when
        ``y`` is given, re-balanced by inverse class frequency so minority classes shape the axes.
    robust : median-centred PCA fitted by iteratively re-weighted (Huber) least squares on the
        orthogonal reconstruction residuals, so gross outliers do not rotate the components.
    sparse : sparse PCA via L1-penalised power iterations (soft-thresholded loadings, ``alpha``)
        with deflation; components are interpretable and use few features.
    probabilistic : PPCA (Tipping & Bishop) closed-form ML solution with isotropic noise;
        ``transform`` returns the posterior mean of the latent variables.
    """

    def __init__(self, n_components=2, variant="base", alpha=0.1, n_iter=100, random_state=0):
        check_variant(variant, VARIANTS, "PCA")
        self.n_components = n_components
        self.variant = variant
        self.alpha = alpha
        self.n_iter = n_iter
        self.random_state = random_state
        self.is_fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        X = check_X(X, min_samples=2)
        n, d = X.shape
        k = self.n_components
        if not isinstance(k, (int, np.integer)) or k < 1 or k > min(n, d):
            raise ValueError(f"n_components must be an integer in [1, {min(n, d)}]")
        if y is not None:
            y = check_y(y, n)
        self.n_features_in_ = d
        self.noise_variance_ = 0.0
        w = np.ones(n)

        if self.variant == "supervised_row_aware":
            w = self._row_weights(X, y)
            self.row_weights_ = w
            mean = (w[:, None] * X).sum(0) / w.sum()
            comps, sv = self._weighted_svd(X - mean, w, k)
        elif self.variant == "robust":
            mean = np.median(X, axis=0)
            comps, sv, w = self._robust_fit(X - mean, k)
            self.row_weights_ = w
        elif self.variant == "sparse":
            mean = X.mean(0)
            comps, sv = self._sparse_fit(X - mean, k)
        else:
            mean = X.mean(0)
            comps, sv = self._weighted_svd(X - mean, w, k)

        Xc = X - mean
        if self.variant == "probabilistic":
            _, s_all, Vt = np.linalg.svd(Xc, full_matrices=False)
            eig = s_all ** 2 / n
            rest = eig[k:]
            s2 = float(rest.sum() / max(d - k, 1)) if d > k else 0.0
            s2 = max(s2, 1e-10)
            W = Vt[:k].T * np.sqrt(np.maximum(eig[:k] - s2, 0.0))
            self.noise_variance_ = s2
            self.W_ = W
            comps = Vt[:k]
            self.explained_variance_ = eig[:k]
        else:
            self.explained_variance_ = np.var(Xc @ np.linalg.pinv(comps), axis=0, ddof=0) \
                if self.variant == "sparse" else sv ** 2 / max(w.sum(), 1.0)

        self.mean_ = mean
        self.components_ = comps
        total = float(np.sum(Xc ** 2) / n)
        self.explained_variance_ratio_ = self.explained_variance_ / total if total > 0 else \
            np.zeros(k)
        self._pinv = np.linalg.pinv(comps)
        self.is_fitted_ = True
        return self

    def _row_weights(self, X, y):
        n = X.shape[0]
        mu = X.mean(0)
        _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        scores = (X - mu) @ Vt[:k].T
        var = np.maximum(s[:k] ** 2 / n, 1e-12)
        d2 = np.sum(scores ** 2 / var, axis=1)
        cutoff = np.median(d2) * 2.5 + 1e-12
        w = np.where(d2 > cutoff, cutoff / np.maximum(d2, 1e-12), 1.0)
        if y is not None:
            classes, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
            w = w * (n / (len(classes) * counts))[inv]
        return w * n / w.sum()

    @staticmethod
    def _weighted_svd(Xc, w, k):
        _, s, Vt = np.linalg.svd(np.sqrt(w)[:, None] * Xc, full_matrices=False)
        return Vt[:k], s[:k]

    def _robust_fit(self, Xc, k):
        n = Xc.shape[0]
        med_r = np.linalg.norm(Xc / (1.4826 * np.median(np.abs(Xc), axis=0) + 1e-12), axis=1)
        c0 = np.median(med_r) + 1.345 * 1.4826 * np.median(np.abs(med_r - np.median(med_r)))
        w = np.where(med_r <= c0, 1.0, c0 / np.maximum(med_r, 1e-12)) ** 2
        comps = sv = None
        for _ in range(10):
            mu_w = (w[:, None] * Xc).sum(0) / w.sum()
            comps, sv = self._weighted_svd(Xc - mu_w, w, k)
            resid = (Xc - mu_w) - ((Xc - mu_w) @ comps.T) @ comps
            r = np.linalg.norm(resid, axis=1)
            scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
            c = np.median(r) + 1.345 * scale
            w_new = np.where(r <= c, 1.0, c / np.maximum(r, 1e-12))
            if np.max(np.abs(w_new - w)) < 1e-6:
                w = w_new
                break
            w = w_new
        return comps, sv, w

    def _sparse_fit(self, Xc, k):
        rng = np.random.default_rng(self.random_state)
        R = Xc.copy()
        comps = []
        for _ in range(k):
            _, _, Vt = np.linalg.svd(R, full_matrices=False)
            v = Vt[0] + 1e-3 * rng.standard_normal(R.shape[1])
            v /= np.linalg.norm(v)
            for _ in range(self.n_iter):
                u = R @ v
                nu = np.linalg.norm(u)
                if nu < 1e-12:
                    break
                u /= nu
                z = R.T @ u
                thr = self.alpha * np.max(np.abs(z))
                v_new = np.sign(z) * np.maximum(np.abs(z) - thr, 0.0)
                nv = np.linalg.norm(v_new)
                if nv < 1e-12:
                    v_new, nv = z, np.linalg.norm(z)
                v_new /= max(nv, 1e-12)
                if np.linalg.norm(v_new - v) < 1e-8:
                    v = v_new
                    break
                v = v_new
            comps.append(v)
            R = R - np.outer(R @ v, v)
        comps = np.array(comps)
        sv = np.linalg.norm(Xc @ comps.T, axis=0)
        return comps, sv

    # ------------------------------------------------------------ transform
    def _check_input(self, X):
        check_fitted(self)
        return check_X(X, n_features=self.n_features_in_)

    def transform(self, X):
        X = self._check_input(X)
        Xc = X - self.mean_
        if self.variant == "probabilistic":
            W = self.W_
            M = W.T @ W + self.noise_variance_ * np.eye(W.shape[1])
            return Xc @ W @ np.linalg.inv(M).T
        return Xc @ self._pinv

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, Z):
        check_fitted(self)
        Z = check_X(Z, name="Z", n_features=self.n_components)
        if self.variant == "probabilistic":
            W = self.W_
            M = W.T @ W + self.noise_variance_ * np.eye(W.shape[1])
            # orthogonal-projection reconstruction from the posterior mean
            return Z @ (M @ np.linalg.pinv(W)) + self.mean_
        return Z @ self.components_ + self.mean_

    def evaluate(self, X, y=None):
        X = self._check_input(X)
        return reconstruction_metrics(X, self.inverse_transform(self.transform(X)))

    def score_samples(self, X):
        """Per-sample reconstruction error (useful as an outlier / reliability score)."""
        X = self._check_input(X)
        return np.mean((X - self.inverse_transform(self.transform(X))) ** 2, axis=1)


__all__ = ["PCA"]
