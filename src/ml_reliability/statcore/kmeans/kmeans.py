"""K-means family (pure NumPy) with reliability-oriented variants."""

from __future__ import annotations

import numpy as np

from .._common import check_fitted, check_variant, check_X, pairwise_sq_dists
from ..evaluation.metrics import clustering_metrics

VARIANTS = ("base", "topology_aware", "stat_model_selection", "robust", "better_init")


def silhouette(X, labels):
    """Mean silhouette coefficient (0 if fewer than 2 clusters)."""
    ul = np.unique(labels)
    if len(ul) < 2:
        return 0.0
    D = np.sqrt(pairwise_sq_dists(X, X))
    n = len(X)
    a = np.zeros(n)
    b = np.full(n, np.inf)
    for c in ul:
        m = labels == c
        cnt = m.sum()
        sums = D[:, m].sum(1)
        own = m
        a[own] = sums[own] / max(cnt - 1, 1) if cnt > 1 else 0.0
        other = ~m
        b[other] = np.minimum(b[other], sums[other] / cnt)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    counts = np.bincount(np.unique(labels, return_inverse=True)[1])
    s[counts[np.unique(labels, return_inverse=True)[1]] == 1] = 0.0
    return float(s.mean())


class KMeans:
    """K-means clustering with selectable variants.

    Variants
    --------
    base : Lloyd's algorithm, random-point initialisation, best of ``n_init`` restarts.
    better_init : Lloyd's algorithm with k-means++ (D^2-weighted) seeding.
    robust : k-medians (L1 assignment, coordinate-wise median update) that additionally
        ignores the ``trim`` fraction of farthest points when updating centres.
    topology_aware : k-means++ seeding plus kNN-graph smoothing; assignment cost is distance minus
        ``topology_weight`` times the fraction of a point's neighbours in that cluster, so
        clusters follow the local data manifold.
    stat_model_selection : chooses the number of clusters in ``k_range`` by BIC (spherical
        Gaussian likelihood, ``criterion='bic'``) or mean silhouette (``criterion='silhouette'``).
    """

    def __init__(self, n_clusters=3, variant="base", max_iter=100, n_init=5, tol=1e-6,
                 k_range=(2, 8), criterion="bic", trim=0.05, n_neighbors=10,
                 topology_weight=0.5, random_state=0):
        check_variant(variant, VARIANTS, "KMeans")
        if criterion not in ("bic", "silhouette"):
            raise ValueError("criterion must be 'bic' or 'silhouette'")
        self.n_clusters = n_clusters
        self.variant = variant
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.k_range = k_range
        self.criterion = criterion
        self.trim = trim
        self.n_neighbors = n_neighbors
        self.topology_weight = topology_weight
        self.random_state = random_state
        self.is_fitted_ = False

    # --------------------------------------------------------------- inits
    @staticmethod
    def _init_random(X, k, rng):
        return X[rng.choice(len(X), size=k, replace=False)].copy()

    @staticmethod
    def _init_pp(X, k, rng):
        n = len(X)
        centers = [X[rng.integers(n)]]
        d2 = pairwise_sq_dists(X, centers[0][None]).ravel()
        for _ in range(1, k):
            tot = d2.sum()
            idx = rng.choice(n, p=d2 / tot) if tot > 0 else rng.integers(n)
            centers.append(X[idx])
            d2 = np.minimum(d2, pairwise_sq_dists(X, X[idx][None]).ravel())
        return np.array(centers)

    # ------------------------------------------------------------- lloyd
    def _lloyd(self, X, centers, rng, knn=None):
        k = len(centers)
        robust = self.variant == "robust"
        prev = None
        for _ in range(self.max_iter):
            if robust:
                cost = np.abs(X[:, None, :] - centers[None]).sum(2)
            else:
                cost = pairwise_sq_dists(X, centers)
            if knn is not None:
                cost = cost / (cost.mean() + 1e-12)
                if getattr(self, "_labels_prev", None) is not None:
                    onehot = np.eye(k)[self._labels_prev]
                    cost = cost - self.topology_weight * onehot[knn].mean(1)
            labels = cost.argmin(1)
            if knn is not None:
                self._labels_prev = labels
            own = cost[np.arange(len(X)), labels]
            keep = np.ones(len(X), dtype=bool)
            if robust and self.trim > 0:
                keep = own <= np.quantile(own, 1 - self.trim)
            new = centers.copy()
            for j in range(k):
                m = (labels == j) & keep
                if not m.any():
                    m = labels == j
                if not m.any():
                    new[j] = X[np.argmax(own)]  # re-seed empty cluster at worst-fit point
                    continue
                new[j] = np.median(X[m], axis=0) if robust else X[m].mean(0)
            shift = np.sum((new - centers) ** 2)
            centers = new
            if shift <= self.tol and (prev is not None and np.array_equal(prev, labels)):
                break
            prev = labels
        if robust:
            cost = np.abs(X[:, None, :] - centers[None]).sum(2)
        else:
            cost = pairwise_sq_dists(X, centers)
        labels = cost.argmin(1)
        obj = float(cost[np.arange(len(X)), labels].sum())
        return centers, labels, obj

    def _fit_k(self, X, k, seed):
        rng = np.random.default_rng(seed)
        knn = None
        if self.variant == "topology_aware":
            D = pairwise_sq_dists(X, X)
            np.fill_diagonal(D, np.inf)
            m = min(self.n_neighbors, len(X) - 1)
            knn = np.argsort(D, axis=1, kind="stable")[:, :m]
        best = None
        for r in range(self.n_init):
            init = self._init_random if self.variant in ("base", "robust") else self._init_pp
            centers = init(X, k, rng)
            self._labels_prev = None
            res = self._lloyd(X, centers, rng, knn)
            if best is None or res[2] < best[2]:
                best = res
        self._labels_prev = None
        return best

    @staticmethod
    def _bic(X, centers, labels, k):
        n, d = X.shape
        sse = float(np.sum((X - centers[labels]) ** 2))
        var = max(sse / max(n - k, 1) / d, 1e-12)
        counts = np.bincount(labels, minlength=k).astype(float)
        nz = counts[counts > 0]
        loglik = float(np.sum(nz * np.log(nz / n)) - 0.5 * n * d * np.log(2 * np.pi * var)
                       - 0.5 * (n - k) * d)
        n_params = (k - 1) + k * d + 1
        return -2 * loglik + n_params * np.log(n)

    # ---------------------------------------------------------------- API
    def fit(self, X, y=None):
        X = check_X(X, min_samples=2)
        n = len(X)
        if self.variant == "stat_model_selection":
            lo, hi = self.k_range
            if lo < 2 or hi < lo:
                raise ValueError("k_range must satisfy 2 <= low <= high")
            hi = min(hi, n - 1)
            self.scores_ = {}
            best = None
            for k in range(lo, hi + 1):
                c, l, o = self._fit_k(X, k, self.random_state + k)
                if self.criterion == "bic":
                    score = -self._bic(X, c, l, k)
                else:
                    score = silhouette(X, l)
                self.scores_[k] = float(-score if self.criterion == "bic" else score)
                if best is None or score > best[0]:
                    best = (score, k, c, l, o)
            _, k, centers, labels, obj = best
            self.n_clusters_ = k
        else:
            k = self.n_clusters
            if not isinstance(k, (int, np.integer)) or k < 1 or k > n:
                raise ValueError(f"n_clusters must be an integer in [1, {n}]")
            centers, labels, obj = self._fit_k(X, k, self.random_state)
            self.n_clusters_ = k
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.objective_ = obj
        self.inertia_ = float(np.sum((X - centers[labels]) ** 2))
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def _dist(self, X):
        if self.variant == "robust":
            return np.abs(X[:, None, :] - self.cluster_centers_[None]).sum(2)
        return pairwise_sq_dists(X, self.cluster_centers_)

    def predict(self, X):
        check_fitted(self)
        X = check_X(X, n_features=self.n_features_in_)
        return self._dist(X).argmin(1)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def transform(self, X):
        """Distances (L1 for the robust variant, Euclidean otherwise) to each centre."""
        check_fitted(self)
        X = check_X(X, n_features=self.n_features_in_)
        d = self._dist(X)
        return d if self.variant == "robust" else np.sqrt(d)

    def evaluate(self, X, y=None):
        X = check_X(X)
        labels = self.predict(X)
        out = clustering_metrics(X, labels, centers=self.cluster_centers_)
        return out
