"""Multinomial logistic regression and ridge regression (NumPy; L-BFGS from SciPy for logistic)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .._common import (Standardizer, check_fitted, check_variant, check_X, check_Xy, golden_min,
                       holdout_split, softmax, stratified_folds)
from ..evaluation.metrics import classification_metrics, regression_metrics

VARIANTS = ("base", "risk_calibrated", "stat_regularised", "robust_loss")


def _fit_softmax(Z, Y, alpha, w=None, x0=None, max_iter=300):
    """Weighted multinomial logistic with L2 penalty 0.5*alpha*||W||^2 on the mean loss."""
    n, d = Z.shape
    c = Y.shape[1]
    w = np.ones(n) if w is None else w
    wn = w / w.sum()

    def fun(theta):
        W = theta[: c * d].reshape(c, d)
        b = theta[c * d:]
        logits = Z @ W.T + b
        P = softmax(logits)
        ll = -np.sum(wn * np.sum(Y * np.log(np.clip(P, 1e-300, None)), axis=1))
        G = wn[:, None] * (P - Y)
        gW = G.T @ Z + alpha * W
        return ll + 0.5 * alpha * np.sum(W ** 2), np.r_[gW.ravel(), G.sum(0)]

    x0 = np.zeros(c * d + c) if x0 is None else x0
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    return res.x[: c * d].reshape(c, d), res.x[c * d:]


class LogisticRegression:
    """L2-regularised multinomial logistic regression with selectable variants.

    Variants
    --------
    base : standardised features, softmax regression, fixed L2 strength ``alpha``.
    risk_calibrated : fits on a stratified split and learns a post-hoc temperature on the
        held-out part (NLL minimisation), giving calibrated probabilities (``temperature_``).
    stat_regularised : picks ``alpha`` from ``alpha_grid`` by k-fold cross-validated log-loss.
    robust_loss : Huber-type M-estimation on the log-loss; samples whose NLL exceeds a robust
        cut-off (median + 1.345*MAD) are iteratively down-weighted, resisting label noise.
    """

    def __init__(self, variant="base", alpha=1e-2, alpha_grid=None, cv=5, calib_fraction=0.25,
                 max_iter=300, random_state=0):
        check_variant(variant, VARIANTS, "LogisticRegression")
        self.variant = variant
        self.alpha = alpha
        self.alpha_grid = alpha_grid
        self.cv = cv
        self.calib_fraction = calib_fraction
        self.max_iter = max_iter
        self.random_state = random_state
        self.is_fitted_ = False

    def _onehot(self, yi, c):
        return np.eye(c)[yi]

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        classes, yi = np.unique(y, return_inverse=True)
        c = len(classes)
        if c < 2:
            raise ValueError("LogisticRegression needs at least two classes")
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.temperature_ = 1.0
        self.alpha_ = self.alpha
        self.scaler_ = Standardizer().fit(X)
        Z = self.scaler_.transform(X)
        Y = self._onehot(yi, c)

        if self.variant == "risk_calibrated":
            tr, ho = holdout_split(len(X), self.calib_fraction, self.random_state, stratify=yi)
            self.coef_, self.intercept_ = _fit_softmax(Z[tr], Y[tr], self.alpha, max_iter=self.max_iter)
            logits = Z[ho] @ self.coef_.T + self.intercept_

            def nll(logt):
                P = softmax(logits / np.exp(logt))
                return -np.mean(np.log(np.clip(P[np.arange(len(ho)), yi[ho]], 1e-12, None)))

            self.temperature_ = float(np.exp(golden_min(nll, -3.0, 3.0)))
        elif self.variant == "stat_regularised":
            grid = np.asarray(self.alpha_grid if self.alpha_grid is not None
                              else np.logspace(-4, 1, 6), dtype=float)
            if np.any(grid < 0):
                raise ValueError("alpha_grid must be non-negative")
            k = int(min(self.cv, np.bincount(yi).min() if np.bincount(yi).min() >= 2 else 2, len(X)))
            k = max(k, 2)
            folds = stratified_folds(yi, k, self.random_state)
            scores = []
            for a in grid:
                loss = 0.0
                for f in range(k):
                    tr, te = folds != f, folds == f
                    W, b = _fit_softmax(Z[tr], Y[tr], a, max_iter=100)
                    P = softmax(Z[te] @ W.T + b)
                    loss += -np.sum(Y[te] * np.log(np.clip(P, 1e-12, None)))
                scores.append(loss / len(X))
            self.cv_scores_ = dict(zip(grid.tolist(), scores))
            self.alpha_ = float(grid[int(np.argmin(scores))])
            self.coef_, self.intercept_ = _fit_softmax(Z, Y, self.alpha_, max_iter=self.max_iter)
        elif self.variant == "robust_loss":
            w = np.ones(len(X))
            for _ in range(6):
                self.coef_, self.intercept_ = _fit_softmax(Z, Y, self.alpha, w=w, max_iter=self.max_iter)
                P = softmax(Z @ self.coef_.T + self.intercept_)
                nll = -np.log(np.clip(P[np.arange(len(X)), yi], 1e-12, None))
                mad = 1.4826 * np.median(np.abs(nll - np.median(nll))) + 1e-6
                cut = np.median(nll) + 1.345 * mad
                w = np.where(nll <= cut, 1.0, cut / nll)
            self.sample_weights_ = w
        else:
            self.coef_, self.intercept_ = _fit_softmax(Z, Y, self.alpha, max_iter=self.max_iter)
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        check_fitted(self)
        X = check_X(X, n_features=self.n_features_in_)
        return (self.scaler_.transform(X) @ self.coef_.T + self.intercept_) / self.temperature_

    def predict_proba(self, X):
        return softmax(self.decision_function(X))

    def predict(self, X):
        check_fitted(self)
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def evaluate(self, X, y):
        X, y = check_Xy(X, y, min_samples=1)
        return classification_metrics(y, self.predict(X), self.predict_proba(X), classes=self.classes_)


class RidgeRegression:
    """Ridge regression (closed form, unpenalised intercept) with selectable variants.

    Variants
    --------
    base : standardised features, fixed L2 strength ``alpha``.
    risk_calibrated : fits on a split, then linearly recalibrates predictions on the held-out
        part and stores a split-conformal residual quantile (see ``predict_interval``).
    stat_regularised : chooses ``alpha`` from ``alpha_grid`` by generalised cross-validation.
    robust_loss : Huber loss via IRLS with MAD scale; outliers get bounded influence.
    """

    def __init__(self, variant="base", alpha=1.0, alpha_grid=None, calib_fraction=0.25,
                 huber_k=1.345, random_state=0):
        check_variant(variant, VARIANTS, "RidgeRegression")
        self.variant = variant
        self.alpha = alpha
        self.alpha_grid = alpha_grid
        self.calib_fraction = calib_fraction
        self.huber_k = huber_k
        self.random_state = random_state
        self.is_fitted_ = False

    @staticmethod
    def _solve(Z, y, lam, w=None):
        n, d = Z.shape
        w = np.ones(n) if w is None else w
        Za = np.hstack([np.ones((n, 1)), Z])
        A = (Za * w[:, None]).T @ Za
        pen = lam * np.eye(d + 1)
        pen[0, 0] = 0.0
        beta = np.linalg.solve(A + pen + 1e-10 * np.eye(d + 1), (Za * w[:, None]).T @ y)
        return beta[1:], beta[0]

    def fit(self, X, y):
        X, y = check_Xy(X, y, numeric_y=True)
        n = len(X)
        self.n_features_in_ = X.shape[1]
        self.scaler_ = Standardizer().fit(X)
        Z = self.scaler_.transform(X)
        self.alpha_ = float(self.alpha)
        if self.alpha_ < 0:
            raise ValueError("alpha must be non-negative")
        self.calib_ = (0.0, 1.0)
        self.conformal_q_ = None

        if self.variant == "risk_calibrated":
            tr, ho = holdout_split(n, self.calib_fraction, self.random_state)
            self.coef_, self.intercept_ = self._solve(Z[tr], y[tr], self.alpha_)
            p = Z[ho] @ self.coef_ + self.intercept_
            if len(ho) >= 3 and np.std(p) > 1e-12:
                b = np.cov(p, y[ho], bias=True)[0, 1] / np.var(p)
                a = y[ho].mean() - b * p.mean()
                self.calib_ = (float(a), float(b))
            res = np.abs(y[ho] - (self.calib_[0] + self.calib_[1] * p))
            self.conformal_res_ = np.sort(res)
        elif self.variant == "stat_regularised":
            grid = np.asarray(self.alpha_grid if self.alpha_grid is not None
                              else np.logspace(-6, 3, 46), dtype=float)
            zc = Z - Z.mean(0)
            yc = y - y.mean()
            U, s, _ = np.linalg.svd(zc, full_matrices=False)
            Uy = U.T @ yc
            best = (np.inf, None)
            for a in grid:
                lam = a
                shrink = s ** 2 / (s ** 2 + lam)
                resid = np.sum(yc ** 2) - np.sum(Uy ** 2) + np.sum(((1 - shrink) * Uy) ** 2)
                dof = 1.0 + shrink.sum()
                gcv = n * resid / max(n - dof, 1e-8) ** 2
                if gcv < best[0]:
                    best = (gcv, a)
            self.alpha_ = float(best[1])
            self.gcv_ = float(best[0])
            self.coef_, self.intercept_ = self._solve(Z, y, self.alpha_)
        elif self.variant == "robust_loss":
            w = np.ones(n)
            lam = self.alpha_
            for _ in range(30):
                coef, icpt = self._solve(Z, y, lam, w)
                r = y - Z @ coef - icpt
                sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
                c = self.huber_k * sigma
                w_new = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-12))
                done = np.max(np.abs(w_new - w)) < 1e-6
                w = w_new
                if done:
                    break
            self.coef_, self.intercept_ = coef, icpt
            self.sample_weights_ = w
        else:
            self.coef_, self.intercept_ = self._solve(Z, y, self.alpha_)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_fitted(self)
        X = check_X(X, n_features=self.n_features_in_)
        p = self.scaler_.transform(X) @ self.coef_ + self.intercept_
        return self.calib_[0] + self.calib_[1] * p

    def predict_interval(self, X, level=0.9):
        """Split-conformal prediction interval (risk_calibrated variant only)."""
        if self.variant != "risk_calibrated":
            raise ValueError("predict_interval requires variant='risk_calibrated'")
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        p = self.predict(X)
        m = len(self.conformal_res_)
        k = min(int(np.ceil((m + 1) * level)), m) - 1
        q = float(self.conformal_res_[max(k, 0)])
        return p - q, p + q

    def evaluate(self, X, y):
        X, y = check_Xy(X, y, numeric_y=True, min_samples=1)
        return regression_metrics(y, self.predict(X))
