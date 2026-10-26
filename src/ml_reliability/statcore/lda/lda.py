"""Linear / quadratic discriminant analysis (pure NumPy) with reliability variants."""

from __future__ import annotations

import numpy as np

from .._common import check_fitted, check_variant, check_X, check_Xy, softmax
from ..evaluation.metrics import classification_metrics

VARIANTS = ("base", "shift_corrected", "shrinkage", "sparse", "heteroscedastic")


def ledoit_wolf_shrinkage(Xc, S=None):
    """Analytic Ledoit-Wolf shrinkage intensity towards mu*I for centred data ``Xc``."""
    n, d = Xc.shape
    if S is None:
        S = Xc.T @ Xc / n
    mu = np.trace(S) / d
    delta2 = float(np.sum((S - mu * np.eye(d)) ** 2)) / d
    if delta2 <= 1e-18:
        return 1.0
    sq = np.sum(Xc ** 2, axis=1)
    quad = np.einsum("ij,jk,ik->i", Xc, S, Xc)
    b2 = float(np.sum(sq ** 2 - 2 * quad + np.sum(S ** 2))) / (n ** 2 * d)
    return float(np.clip(min(b2, delta2) / delta2, 0.0, 1.0))


def _domain_weights(Xs, Xt, clip=20.0, l2=1.0, iters=25):
    """Importance weights p_t(x)/p_s(x) from a Newton-fitted logistic domain classifier."""
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-12
    A = np.vstack([(Xs - mu) / sd, (Xt - mu) / sd])
    A = np.hstack([A, np.ones((len(A), 1))])
    t = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    beta = np.zeros(A.shape[1])
    reg = l2 * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ beta, -30, 30)))
        g = A.T @ (p - t) + reg @ beta
        H = (A * (p * (1 - p))[:, None]).T @ A + reg + 1e-8 * np.eye(len(beta))
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    ps = 1.0 / (1.0 + np.exp(-np.clip(A[: len(Xs)] @ beta, -30, 30)))
    w = ps / np.maximum(1 - ps, 1e-12) * (len(Xs) / len(Xt))
    w = np.minimum(w, clip)
    return w * len(Xs) / w.sum()


class LDA:
    """Discriminant analysis classifier with selectable variants.

    Variants
    --------
    base : classic LDA with a pooled covariance (tiny ridge ``reg``) and empirical priors.
    shift_corrected : covariate-shift-corrected LDA; ``fit(X, y, X_target=...)`` estimates
        importance weights p_target(x)/p_source(x) with a logistic domain classifier and fits
        weighted priors, means and covariance. ``X_target`` is required.
    shrinkage : pooled covariance shrunk towards a scaled identity with the analytic
        Ledoit-Wolf intensity (or a fixed ``shrinkage`` in [0, 1]); stable when n is close to d.
    sparse : nearest-shrunken-centroids discriminant: class means are soft-thresholded towards
        the global mean (``threshold``) with diagonal covariance, discarding uninformative features.
    heteroscedastic : QDA-like model with class-specific covariances, each regularised towards
        the pooled covariance by ``reg_param``.
    """

    def __init__(self, variant="base", n_components=None, shrinkage=None, reg=1e-6,
                 threshold=1.0, reg_param=0.3, random_state=0):
        check_variant(variant, VARIANTS, "LDA")
        self.variant = variant
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.reg = reg
        self.threshold = threshold
        self.reg_param = reg_param
        self.random_state = random_state
        self.is_fitted_ = False

    def fit(self, X, y, X_target=None):
        X, y = check_Xy(X, y)
        n, d = X.shape
        classes, yi = np.unique(y, return_inverse=True)
        c = len(classes)
        if c < 2:
            raise ValueError("LDA needs at least two classes")
        w = np.ones(n)
        if self.variant == "shift_corrected":
            if X_target is None:
                raise ValueError("variant 'shift_corrected' requires X_target")
            Xt = check_X(X_target, "X_target", n_features=d)
            w = _domain_weights(X, Xt)
            self.sample_weights_ = w
        elif X_target is not None:
            raise ValueError("X_target is only used by variant 'shift_corrected'")

        Wk = np.array([w[yi == k].sum() for k in range(c)])
        if np.any(np.array([np.sum(yi == k) for k in range(c)]) < 1):
            raise ValueError("empty class")
        priors = Wk / Wk.sum()
        means = np.array([(w[yi == k, None] * X[yi == k]).sum(0) / Wk[k] for k in range(c)])
        Xc = X - means[yi]
        cov = (w[:, None] * Xc).T @ Xc / w.sum()
        self.shrinkage_ = None

        if self.variant == "shrinkage":
            if self.shrinkage is None:
                s = ledoit_wolf_shrinkage(Xc, cov)
            else:
                s = float(self.shrinkage)
                if not 0.0 <= s <= 1.0:
                    raise ValueError("shrinkage must be in [0, 1]")
            cov = (1 - s) * cov + s * np.trace(cov) / d * np.eye(d)
            self.shrinkage_ = s
        elif self.variant == "sparse":
            var = np.diag(cov).copy()
            s = np.sqrt(var)
            s0 = np.median(s)
            counts = np.array([np.sum(yi == k) for k in range(c)], dtype=float)
            gmean = X.mean(0)
            mk = np.sqrt(np.maximum(1.0 / counts - 1.0 / n, 1e-12))
            dk = (means - gmean) / (mk[:, None] * (s + s0))
            dk = np.sign(dk) * np.maximum(np.abs(dk) - self.threshold, 0.0)
            means = gmean + mk[:, None] * (s + s0) * dk
            self.selected_features_ = np.flatnonzero(np.any(dk != 0, axis=0))
            cov = np.diag(var)

        cov = cov + self.reg * np.eye(d) * max(np.trace(cov) / d, 1e-12)
        self.classes_ = classes
        self.priors_ = priors
        self.means_ = means
        self.covariance_ = cov
        self.n_features_in_ = d
        self.xbar_ = (priors[:, None] * means).sum(0)

        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, 1e-12)
        self._whiten = evecs / np.sqrt(evals)
        self._prec = self._whiten @ self._whiten.T

        if self.variant == "heteroscedastic":
            rp = float(self.reg_param)
            if not 0.0 <= rp <= 1.0:
                raise ValueError("reg_param must be in [0, 1]")
            self._cls_chol, self._cls_logdet = [], []
            self.class_covariances_ = []
            for k in range(c):
                Xk = X[yi == k] - means[k]
                wk = w[yi == k]
                Sk = (wk[:, None] * Xk).T @ Xk / wk.sum()
                Sk = (1 - rp) * Sk + rp * cov + 1e-8 * np.eye(d)
                self.class_covariances_.append(Sk)
                L = np.linalg.cholesky(Sk)
                self._cls_chol.append(L)
                self._cls_logdet.append(2.0 * np.sum(np.log(np.diag(L))))

        # discriminant directions: SVD of whitened, prior-weighted class means
        M = (means - self.xbar_) @ self._whiten
        M = np.sqrt(priors)[:, None] * M
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        r = min(c - 1, d)
        if self.n_components is not None:
            if not 1 <= self.n_components <= r:
                raise ValueError(f"n_components must be in [1, {r}]")
            r = self.n_components
        self.scalings_ = self._whiten @ Vt[:r].T
        self.is_fitted_ = True
        return self

    def _decision(self, X):
        check_fitted(self)
        X = check_X(X, n_features=self.n_features_in_)
        logp = np.log(np.maximum(self.priors_, 1e-300))
        if self.variant == "heteroscedastic":
            out = np.empty((len(X), len(self.classes_)))
            for k in range(len(self.classes_)):
                z = np.linalg.solve(self._cls_chol[k], (X - self.means_[k]).T)
                out[:, k] = -0.5 * (np.sum(z ** 2, axis=0) + self._cls_logdet[k]) + logp[k]
            return out
        lin = X @ self._prec @ self.means_.T
        const = -0.5 * np.sum((self.means_ @ self._prec) * self.means_, axis=1)
        return lin + const + logp

    def decision_function(self, X):
        return self._decision(X)

    def predict_proba(self, X):
        return softmax(self._decision(X))

    def predict(self, X):
        check_fitted(self)
        return self.classes_[np.argmax(self._decision(X), axis=1)]

    def transform(self, X):
        check_fitted(self)
        X = check_X(X, n_features=self.n_features_in_)
        return (X - self.xbar_) @ self.scalings_

    def fit_transform(self, X, y, X_target=None):
        return self.fit(X, y, X_target=X_target).transform(X)

    def evaluate(self, X, y):
        X, y = check_Xy(X, y, min_samples=1)
        return classification_metrics(y, self.predict(X), self.predict_proba(X), classes=self.classes_)
