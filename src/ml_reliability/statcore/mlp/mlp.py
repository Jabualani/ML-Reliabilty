"""Feed-forward neural networks (pure NumPy): He init, AdamW, early stopping, reliability variants."""

from __future__ import annotations

import numpy as np

from .._common import (Standardizer, check_fitted, check_variant, check_X, check_Xy, holdout_split,
                       softmax)
from ..evaluation.metrics import classification_metrics, regression_metrics
from ..linear.linear import LogisticRegression

VARIANTS = ("base", "stat_regularised", "reliability_head", "uncertainty_head")


class _Net:
    def __init__(self, sizes, rng):
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def copy_params(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set_params(self, params):
        self.W, self.b = [w.copy() for w in params[0]], [b.copy() for b in params[1]]

    def forward(self, X, dropout=0.0, rng=None):
        acts, masks = [X], []
        h = X
        for i in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[i] + self.b[i], 0.0)
            if dropout > 0.0:
                m = (rng.random(h.shape) >= dropout) / (1.0 - dropout)
                h = h * m
                masks.append(m)
            else:
                masks.append(None)
            acts.append(h)
        return h @ self.W[-1] + self.b[-1], acts, masks

    def backward(self, dout, acts, masks):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = d @ self.W[i].T
                if masks[i - 1] is not None:
                    d = d * masks[i - 1]
                d = d * (acts[i] > 0)
        return gW, gb


class _MLPBase:
    _is_classifier = True

    def __init__(self, hidden_layer_sizes=(32, 32), variant="base", learning_rate=0.01,
                 weight_decay=1e-2, max_epochs=150, batch_size=64, patience=15,
                 validation_fraction=0.15, dropout=0.1, mc_samples=30, head_fraction=0.2,
                 wd_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), random_state=0):
        check_variant(variant, VARIANTS, type(self).__name__)
        if not len(hidden_layer_sizes) or any(int(h) < 1 for h in hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain positive integers")
        if not 0 < validation_fraction < 0.9:
            raise ValueError("validation_fraction must be in (0, 0.9)")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        self.variant = variant
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.dropout = dropout
        self.mc_samples = mc_samples
        self.head_fraction = head_fraction
        self.wd_grid = wd_grid
        self.random_state = random_state
        self.is_fitted_ = False

    # ----------------------------------------------------------- losses
    @property
    def _hetero(self):
        return (not self._is_classifier) and self.variant == "uncertainty_head"

    @property
    def _p_drop(self):
        return self.dropout if self.variant == "uncertainty_head" else 0.0

    def _loss_and_grad(self, out, T):
        n = len(T)
        if self._is_classifier:
            P = softmax(out)
            return -np.mean(np.sum(T * np.log(np.clip(P, 1e-12, None)), 1)), (P - T) / n
        if self._hetero:
            mu, lv = out[:, :1], np.clip(out[:, 1:], -6.0, 6.0)
            var = np.exp(lv)
            r = T - mu
            loss = np.mean(0.5 * (lv + r ** 2 / var))
            return loss, np.hstack([-r / var, 0.5 * (1 - r ** 2 / var)]) / n
        r = out - T
        return 0.5 * np.mean(r ** 2), r / n

    def _val_loss(self, net, X, T):
        out = net.forward(X)[0]
        return self._loss_and_grad(out, T)[0]

    # ---------------------------------------------------------- training
    def _train(self, Xtr, Ttr, Xva, Tva, wd, seed):
        rng = np.random.default_rng(seed)
        out_dim = Ttr.shape[1] if self._is_classifier else (2 if self._hetero else 1)
        net = _Net([Xtr.shape[1], *self.hidden_layer_sizes, out_dim], rng)
        mW = [np.zeros_like(w) for w in net.W]; vW = [np.zeros_like(w) for w in net.W]
        mb = [np.zeros_like(b) for b in net.b]; vb = [np.zeros_like(b) for b in net.b]
        best, best_params, wait, t = np.inf, net.copy_params(), 0, 0
        n = len(Xtr)
        bs = max(1, min(int(self.batch_size), n))
        lr = self.learning_rate
        for epoch in range(self.max_epochs):
            perm = rng.permutation(n)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                out, acts, masks = net.forward(Xtr[idx], self._p_drop, rng)
                _, dout = self._loss_and_grad(out, Ttr[idx])
                gW, gb = net.backward(dout, acts, masks)
                t += 1
                for i in range(len(net.W)):
                    mW[i] = 0.9 * mW[i] + 0.1 * gW[i]; vW[i] = 0.999 * vW[i] + 0.001 * gW[i] ** 2
                    mb[i] = 0.9 * mb[i] + 0.1 * gb[i]; vb[i] = 0.999 * vb[i] + 0.001 * gb[i] ** 2
                    c1, c2 = 1 - 0.9 ** t, 1 - 0.999 ** t
                    net.W[i] -= lr * (mW[i] / c1) / (np.sqrt(vW[i] / c2) + 1e-8) + lr * wd * net.W[i]
                    net.b[i] -= lr * (mb[i] / c1) / (np.sqrt(vb[i] / c2) + 1e-8)
            v = self._val_loss(net, Xva, Tva)
            if v < best - 1e-5:
                best, best_params, wait = v, net.copy_params(), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.set_params(best_params)
        return net, best

    # --------------------------------------------------------------- fit
    def _prepare_targets(self, y):
        if self._is_classifier:
            self.classes_, yi = np.unique(y, return_inverse=True)
            if len(self.classes_) < 2:
                raise ValueError("MLPClassifier needs at least two classes")
            self._yi = yi
            return np.eye(len(self.classes_))[yi]
        self._y_mean, self._y_std = float(y.mean()), float(y.std()) or 1.0
        self._yi = None
        return ((y - self._y_mean) / self._y_std)[:, None]

    def fit(self, X, y):
        X, y = check_Xy(X, y, numeric_y=not self._is_classifier, min_samples=6)
        T = self._prepare_targets(y)
        self.n_features_in_ = X.shape[1]
        self.scaler_ = Standardizer().fit(X)
        Z = self.scaler_.transform(X)
        n = len(Z)
        strat = self._yi
        self.weight_decay_ = self.weight_decay
        rest = np.arange(n)
        head_idx = None
        if self.variant == "reliability_head":
            rest, head_idx = holdout_split(n, self.head_fraction, self.random_state, stratify=strat)
        tr_rel, va_rel = holdout_split(len(rest), self.validation_fraction, self.random_state + 1,
                                       stratify=None if strat is None else strat[rest])
        tr, va = rest[tr_rel], rest[va_rel]

        if self.variant == "stat_regularised":
            best = None
            self.wd_scores_ = {}
            for wd in self.wd_grid:
                net, v = self._train(Z[tr], T[tr], Z[va], T[va], float(wd), self.random_state)
                self.wd_scores_[float(wd)] = float(v)
                if best is None or v < best[0]:
                    best = (v, float(wd), net)
            self.weight_decay_ = best[1]
            self.net_ = best[2]
        else:
            self.net_, _ = self._train(Z[tr], T[tr], Z[va], T[va], self.weight_decay, self.random_state)
        self._train_mean_sq = float(np.mean(Z[tr] ** 2))
        self.is_fitted_ = True

        self.head_ = None
        if head_idx is not None:
            self._fit_reliability_head(Z[head_idx], T[head_idx])
        return self

    # ---------------------------------------------------------- inference
    def _mc_outputs(self, Z):
        rng = np.random.default_rng(self.random_state)
        outs = [self.net_.forward(Z, self._p_drop, rng)[0] for _ in range(self.mc_samples)]
        return np.array(outs)

    def _outputs(self, Z):
        """Mean prediction (probabilities or standardised mean) for already-scaled Z."""
        if self.variant == "uncertainty_head":
            outs = self._mc_outputs(Z)
            if self._is_classifier:
                return softmax(outs, axis=2).mean(0)
            return outs[:, :, :1].mean(0)
        out = self.net_.forward(Z)[0]
        return softmax(out) if self._is_classifier else out[:, :1]

    def _scale(self, X):
        check_fitted(self)
        X = check_X(X, n_features=self.n_features_in_)
        return self.scaler_.transform(X)

    def _head_features(self, Z, mean_out):
        h = self.net_.forward(Z)[1][-1]
        cols = [np.linalg.norm(h, axis=1), np.mean(Z ** 2, axis=1) / (self._train_mean_sq + 1e-12)]
        if self._is_classifier:
            P = mean_out
            Ps = np.sort(P, axis=1)
            cols += [Ps[:, -1], -np.sum(P * np.log(np.clip(P, 1e-12, None)), 1), Ps[:, -1] - Ps[:, -2]]
        else:
            cols.append(np.abs(mean_out[:, 0]))
        return np.column_stack(cols)

    def _fit_reliability_head(self, Zh, Th):
        mean_out = self._outputs(Zh)
        F = self._head_features(Zh, mean_out)
        if self._is_classifier:
            target = (mean_out.argmax(1) == Th.argmax(1)).astype(int)
        else:
            err = np.abs(mean_out[:, 0] - Th[:, 0])
            target = (err <= np.median(err)).astype(int)
        if len(np.unique(target)) < 2:
            self.head_ = ("const", float(target.mean()))
        else:
            self.head_ = ("lr", LogisticRegression(alpha=1.0, random_state=self.random_state).fit(F, target))

    def predict_reliability(self, X):
        """Probability in [0, 1] that the prediction is reliable (reliability_head variant)."""
        if self.variant != "reliability_head":
            raise ValueError("predict_reliability requires variant='reliability_head'")
        Z = self._scale(X)
        kind, obj = self.head_
        if kind == "const":
            return np.full(len(Z), obj)
        F = self._head_features(Z, self._outputs(Z))
        return obj.predict_proba(F)[:, list(obj.classes_).index(1)]

    def predict_uncertainty(self, X):
        """Predictive uncertainty (uncertainty_head): normalised entropy for classifiers,
        predictive standard deviation in target units for regressors."""
        if self.variant != "uncertainty_head":
            raise ValueError("predict_uncertainty requires variant='uncertainty_head'")
        Z = self._scale(X)
        outs = self._mc_outputs(Z)
        if self._is_classifier:
            P = softmax(outs, axis=2).mean(0)
            H = -np.sum(P * np.log(np.clip(P, 1e-12, None)), axis=1)
            return H / np.log(P.shape[1])
        mu = outs[:, :, 0]
        alea = np.exp(np.clip(outs[:, :, 1], -6.0, 6.0))
        return np.sqrt(alea.mean(0) + mu.var(0)) * self._y_std

    def reliability_score(self, X):
        """Higher = more trustworthy. Head output, 1 - uncertainty, or max-probability for the
        base/stat_regularised classifier."""
        if self.variant == "reliability_head":
            return self.predict_reliability(X)
        if self.variant == "uncertainty_head":
            u = self.predict_uncertainty(X)
            return 1.0 - u if self._is_classifier else 1.0 / (1.0 + u / self._y_std)
        if self._is_classifier:
            return self.predict_proba(X).max(axis=1)
        raise ValueError("regressor reliability_score needs variant 'reliability_head' or 'uncertainty_head'")


class MLPClassifier(_MLPBase):
    """MLP classifier (ReLU, He init, AdamW weight decay, early stopping on a validation split).

    Variants
    --------
    base : plain network with fixed ``weight_decay``.
    stat_regularised : selects ``weight_decay`` from ``wd_grid`` by validation cross-entropy.
    reliability_head : a held-out split trains a small logistic head that predicts whether the
        network is correct from confidence, entropy, margin, feature norm and input novelty
        (``predict_reliability``).
    uncertainty_head : MC-dropout network; probabilities average ``mc_samples`` stochastic passes
        and ``predict_uncertainty`` returns normalised predictive entropy.
    """

    _is_classifier = True

    def predict_proba(self, X):
        return self._outputs(self._scale(X))

    def predict(self, X):
        check_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def evaluate(self, X, y):
        X, y = check_Xy(X, y, min_samples=1)
        return classification_metrics(y, self.predict(X), self.predict_proba(X), classes=self.classes_)


class MLPRegressor(_MLPBase):
    """MLP regressor (ReLU, He init, AdamW weight decay, early stopping, standardised target).

    Variants
    --------
    base : plain network with fixed ``weight_decay`` and squared error.
    stat_regularised : selects ``weight_decay`` from ``wd_grid`` by validation MSE.
    reliability_head : held-out split trains a logistic head predicting whether the absolute
        error is below the median (``predict_reliability``).
    uncertainty_head : heteroscedastic Gaussian-NLL output (mean, log-variance) with MC dropout;
        ``predict_uncertainty`` returns total predictive standard deviation.
    """

    _is_classifier = False

    def predict(self, X):
        return self._outputs(self._scale(X))[:, 0] * self._y_std + self._y_mean

    def evaluate(self, X, y):
        X, y = check_Xy(X, y, numeric_y=True, min_samples=1)
        return regression_metrics(y, self.predict(X))
