"""k-nearest-neighbour classifier / regressor (pure NumPy) with reliability variants."""

from __future__ import annotations

import numpy as np

from .._common import (Standardizer, check_fitted, check_variant, check_X, check_Xy,
                       pairwise_sq_dists, softmax)
from ..evaluation.metrics import classification_metrics, regression_metrics
from ..kmeans.kmeans import KMeans

VARIANTS = ("base", "prototype_compressed", "learnable_metric", "stat_k_selection")


def _neighbour_average(D2, T, ref_w, k, weights):
    """Average targets T of the k nearest references (D2: query x reference sq. distances)."""
    k = min(k, D2.shape[1])
    idx = np.argsort(D2, axis=1, kind="stable")[:, :k]
    rows = np.arange(D2.shape[0])[:, None]
    w = ref_w[idx]
    if weights == "distance":
        w = w / (np.sqrt(D2[rows, idx]) + 1e-9)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
    return np.einsum("qk,qkc->qc", w, T[idx])


class _KNNBase:
    _is_classifier = True

    def __init__(self, n_neighbors=5, variant="base", weights="uniform", compression=0.2,
                 k_grid=None, metric_iters=60, metric_lr=0.1, random_state=0):
        check_variant(variant, VARIANTS, type(self).__name__)
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        if not 0 < compression <= 1:
            raise ValueError("compression must be in (0, 1]")
        self.n_neighbors = n_neighbors
        self.variant = variant
        self.weights = weights
        self.compression = compression
        self.k_grid = k_grid
        self.metric_iters = metric_iters
        self.metric_lr = metric_lr
        self.random_state = random_state
        self.is_fitted_ = False

    # --------------------------------------------------------- targets
    def _targets(self, y):
        if self._is_classifier:
            self.classes_, yi = np.unique(y, return_inverse=True)
            if len(self.classes_) < 2:
                raise ValueError("KNNClassifier needs at least two classes")
            return np.eye(len(self.classes_))[yi]
        return y.astype(float)[:, None]

    # --------------------------------------------------------- variants
    def _compress(self, Z, T):
        rng_seed = self.random_state
        protos, ptargets, counts = [], [], []
        if self._is_classifier:
            groups = [np.flatnonzero(T[:, c] == 1) for c in range(T.shape[1])]
        else:
            groups = [np.arange(len(Z))]
        for idx in groups:
            m = max(1, int(np.ceil(self.compression * len(idx))))
            if m >= len(idx):
                protos.append(Z[idx]); ptargets.append(T[idx]); counts.append(np.ones(len(idx)))
                continue
            if self._is_classifier:
                feats = Z[idx]
            else:
                yz = (T[idx] - T[idx].mean()) / (T[idx].std() + 1e-12)
                feats = np.hstack([Z[idx], yz])
            km = KMeans(m, variant="better_init", n_init=2, max_iter=50, random_state=rng_seed).fit(feats)
            for j in range(m):
                mem = idx[km.labels_ == j]
                if len(mem) == 0:
                    continue
                protos.append(Z[mem].mean(0, keepdims=True))
                ptargets.append(T[mem].mean(0, keepdims=True))
                counts.append(np.array([float(len(mem))]))
        return np.vstack(protos), np.vstack(ptargets), np.concatenate(counts)

    def _learn_metric(self, Z, T):
        n, d = Z.shape
        rng = np.random.default_rng(self.random_state)
        sub = rng.permutation(n)[:400]
        Zs, Ts = Z[sub], T[sub]
        m = len(sub)
        Dk = (Zs[:, None, :] - Zs[None, :, :]) ** 2
        theta = np.full(d, np.log(1.0 / d))
        mom, vel = np.zeros(d), np.zeros(d)
        ys = None if self._is_classifier else (Ts[:, 0] - Ts[:, 0].mean()) / (Ts[:, 0].std() + 1e-12)
        same = (Ts @ Ts.T) > 0.5 if self._is_classifier else None
        for t in range(1, self.metric_iters + 1):
            a = np.exp(theta)
            S = -(Dk @ a)
            np.fill_diagonal(S, -np.inf)
            P = softmax(S)
            E = np.einsum("ij,ijk->ik", P, Dk)
            if self._is_classifier:
                A = np.einsum("ij,ijk->ik", P * same, Dk)
                q = (P * same).sum(1)
                grad_a = -np.sum(-A + q[:, None] * E, axis=0) / m  # minimise -mean(q)
            else:
                yhat = P @ ys
                dy = -np.einsum("ij,ijk->ik", P * (ys[None, :] - yhat[:, None]), Dk)
                grad_a = np.mean(2 * (yhat - ys)[:, None] * dy, axis=0)
            g = grad_a * a
            mom = 0.9 * mom + 0.1 * g
            vel = 0.999 * vel + 0.001 * g ** 2
            theta -= self.metric_lr * (mom / (1 - 0.9 ** t)) / (np.sqrt(vel / (1 - 0.999 ** t)) + 1e-8)
            theta = np.clip(theta, -8.0, 3.0)
        a = np.exp(theta)
        return a / a.mean()

    def _select_k(self, Z, T):
        n = len(Z)
        grid = self.k_grid
        if grid is None:
            grid = [k for k in range(1, min(31, n - 1) + 1, 2)]
        grid = [int(k) for k in grid if 1 <= int(k) <= n - 1]
        if not grid:
            raise ValueError("k_grid has no valid entries")
        D2 = pairwise_sq_dists(Z, Z)
        np.fill_diagonal(D2, np.inf)
        ones = np.ones(n)
        best, best_k = np.inf, grid[0]
        self.cv_scores_ = {}
        for k in grid:
            P = _neighbour_average(D2, T, ones, k, self.weights)
            if self._is_classifier:
                loss = 1.0 - np.mean(P.argmax(1) == T.argmax(1))
                loss += 1e-3 * np.mean(-np.log(np.clip(np.sum(P * T, 1), 1e-3, None)))
            else:
                loss = np.mean((P[:, 0] - T[:, 0]) ** 2)
            self.cv_scores_[k] = float(loss)
            if loss < best - 1e-12:
                best, best_k = loss, k
        return best_k

    # --------------------------------------------------------------- API
    def fit(self, X, y):
        X, y = check_Xy(X, y, numeric_y=not self._is_classifier)
        if not isinstance(self.n_neighbors, (int, np.integer)) or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        T = self._targets(y)
        self.n_features_in_ = X.shape[1]
        self.scaler_ = Standardizer().fit(X)
        Z = self.scaler_.transform(X)
        self.metric_diag_ = np.ones(X.shape[1])
        self.k_ = int(self.n_neighbors)
        if self.variant == "learnable_metric":
            self.metric_diag_ = self._learn_metric(Z, T)
            Z = Z * np.sqrt(self.metric_diag_)
        if self.variant == "stat_k_selection":
            self.k_ = self._select_k(Z, T)
        ref_w = np.ones(len(Z))
        if self.variant == "prototype_compressed":
            Z, T, ref_w = self._compress(Z, T)
        self._Z, self._T, self._w = Z, T, ref_w
        self.n_prototypes_ = len(Z)
        self.is_fitted_ = True
        return self

    def _raw(self, X):
        check_fitted(self)
        X = check_X(X, n_features=self.n_features_in_)
        Z = self.scaler_.transform(X) * np.sqrt(self.metric_diag_)
        return _neighbour_average(pairwise_sq_dists(Z, self._Z), self._T, self._w, self.k_, self.weights)


class KNNClassifier(_KNNBase):
    """k-NN classifier on standardised features.

    Variants
    --------
    base : Euclidean k-NN, uniform or distance-weighted votes.
    prototype_compressed : per-class k-means prototypes (``compression`` fraction of the data,
        weighted by cluster size) replace the training set; faster and denoised.
    learnable_metric : learns a diagonal Mahalanobis metric by maximising the leave-one-out
        soft-neighbour (NCA) objective with Adam.
    stat_k_selection : picks ``k`` from ``k_grid`` by leave-one-out cross-validation.
    """

    _is_classifier = True

    def predict_proba(self, X):
        return self._raw(X)

    def predict(self, X):
        check_fitted(self)
        return self.classes_[np.argmax(self._raw(X), axis=1)]

    def evaluate(self, X, y):
        X, y = check_Xy(X, y, min_samples=1)
        return classification_metrics(y, self.predict(X), self.predict_proba(X), classes=self.classes_)


class KNNRegressor(_KNNBase):
    """k-NN regressor on standardised features.

    Variants
    --------
    base : Euclidean k-NN mean (uniform or distance-weighted).
    prototype_compressed : k-means prototypes in (x, y) space with cluster-mean targets,
        weighted by cluster size.
    learnable_metric : diagonal metric learned by minimising leave-one-out soft-neighbour
        squared error.
    stat_k_selection : picks ``k`` from ``k_grid`` by leave-one-out cross-validated MSE.
    """

    _is_classifier = False

    def predict(self, X):
        return self._raw(X)[:, 0]

    def evaluate(self, X, y):
        X, y = check_Xy(X, y, numeric_y=True, min_samples=1)
        return regression_metrics(y, self.predict(X))
