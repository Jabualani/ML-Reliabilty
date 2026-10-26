"""ReliableML Data Intelligence: one module, three layers.

``DataIntelligence``    the orchestrator and public entry point (``analyze`` routes by ``mode``):
                        comprehensive_eda, target_intelligence, feature_intelligence,
                        field_investigation, auto_question_generator, llm_augmented, autonomous.
``TargetIntelligence``  target profile, leakage/linkage screening, causal *screening*, explainability,
                        alternative targets and feature-selection experiments (``scope="full"``), or
                        collinearity only (``scope="multicollinearity"``; never reads the target, so it
                        is safe on an AutoML training partition).
``FeatureIntelligence`` hazard filter, correlation and VIF pruning, multi-method supervised ranking.

Rules of this module: every model is StatCore or NumPy (no scikit-learn); scipy/statsmodels are used
for classical statistics only; matplotlib/seaborn are used for figures, in the library house style
shared with ``visualization.py``. Only a statistical summary (never raw rows) reaches an LLM backend,
and only when one is configured.
"""

from __future__ import annotations

import copy
import json
import os
import warnings
from itertools import combinations
from math import erf, pi
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from .statcore._common import check_variant, stratified_folds, kfold_ids
from .statcore.kmeans.kmeans import KMeans
from .statcore.knn.knn import KNNClassifier, KNNRegressor
from .statcore.linear.linear import LogisticRegression, RidgeRegression

SCOPES = ("full", "multicollinearity")
MODES = ("comprehensive_eda", "target_intelligence", "feature_intelligence", "field_investigation",
         "auto_question_generator", "llm_augmented", "autonomous")
METHODS = ("mutual_info", "anova", "chi2", "rfe", "model_importance")
_ALIASES = {"eda": "comprehensive_eda"}
_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
_ID_WORDS = ("target", "label", "outcome", "result", "response", "leak", "future", "after")
_MAX_LEVELS = 8
_ERF = np.vectorize(erf)


# --------------------------------------------------------------------------- data prep
class _Frame:
    """Numeric/categorical view of X with cached model-matrix blocks."""

    def __init__(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.reset_index(drop=True)
        else:
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError("X must be 2-D")
            df = pd.DataFrame(arr, columns=[f"feature_{i + 1}" for i in range(arr.shape[1])])
        if df.shape[1] < 1 or df.shape[0] < 8:
            raise ValueError("X needs at least 8 rows and 1 column")
        self.names = [str(c) for c in df.columns]
        if len(set(self.names)) != len(self.names):
            raise ValueError("X column names must be unique")
        self.n = len(df)
        self.kind: dict[str, str] = {}
        self.values: dict[str, np.ndarray] = {}
        self.missing_rate: dict[str, float] = {}
        self.codes: dict[str, np.ndarray] = {}
        self._blocks: dict[str, np.ndarray] = {}
        for name, col in zip(self.names, df.columns):
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                v = np.array(pd.to_numeric(s, errors="coerce"), dtype=float)
                v[~np.isfinite(v)] = np.nan
                miss = np.isnan(v)
                fill = np.nanmedian(v) if (~miss).any() else 0.0
                self.values[name] = np.where(miss, fill, v)
                self.kind[name] = "numeric"
            else:
                miss = s.isna().to_numpy()
                v = s.astype(object).where(~miss, "__missing__").astype(str).to_numpy()
                self.values[name] = v
                self.kind[name] = "categorical"
            self.missing_rate[name] = float(miss.mean())
            self.codes[name] = self._discretise(name)

    def _discretise(self, name):
        v = self.values[name]
        if self.kind[name] == "categorical":
            return pd.factorize(v)[0].astype(int)
        qs = np.unique(np.quantile(v, np.linspace(0, 1, 11)[1:-1]))
        return np.searchsorted(qs, v, side="right").astype(int)

    def block(self, name):
        if name not in self._blocks:
            v = self.values[name]
            if self.kind[name] == "numeric":
                self._blocks[name] = v[:, None]
            else:
                vc = pd.Series(v).value_counts()
                keep = list(vc.index[:_MAX_LEVELS])
                cols = [(v == k).astype(float) for k in keep]
                if len(vc) > _MAX_LEVELS:
                    cols.append((~np.isin(v, keep)).astype(float))
                self._blocks[name] = np.column_stack(cols)
        return self._blocks[name]

    def matrix(self, names):
        return np.hstack([self.block(n) for n in names]) if names else np.zeros((self.n, 0))

    def numeric(self):
        return [n for n in self.names if self.kind[n] == "numeric"
                and np.std(self.values[n]) > 1e-12]

    def subset(self, rows):
        out = copy.copy(self)
        out.n = len(rows)
        out.values = {k: v[rows] for k, v in self.values.items()}
        out.codes = {k: v[rows] for k, v in self.codes.items()}
        out._blocks = {}
        return out


# ----------------------------------------------------------------------- small stats
def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _contingency(a, b):
    na, nb = a.max() + 1, b.max() + 1
    return np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(float)


def _mutual_info(a, b):
    p = _contingency(a, b)
    p = p / p.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())


def _cramers_v(a, b):
    tab = _contingency(a, b)
    if min(tab.shape) < 2:
        return 0.0
    n = tab.sum()
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    chi2 = float(((tab - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())
    return float(np.sqrt(min(1.0, chi2 / (n * (min(tab.shape) - 1)))))


def _eta(values, groups):
    """Correlation ratio (square root of eta^2) of numeric ``values`` over ``groups``."""
    sst = float(((values - values.mean()) ** 2).sum())
    if sst <= 1e-12:
        return 0.0
    g = groups.max() + 1
    cnt = np.bincount(groups, minlength=g).astype(float)
    mean = np.bincount(groups, weights=values, minlength=g) / np.maximum(cnt, 1)
    ssb = float((cnt * (mean - values.mean()) ** 2).sum())
    return float(np.sqrt(min(1.0, ssb / sst)))


def _rank(v):
    return pd.Series(v).rank(method="average").to_numpy()


def _corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _balanced_accuracy(y, pred, n_classes):
    rec = [float((pred[y == c] == c).mean()) for c in range(n_classes) if (y == c).any()]
    return float(np.mean(rec))


def _r2(y, pred):
    sst = float(((y - y.mean()) ** 2).sum())
    return 0.0 if sst <= 1e-12 else 1.0 - float(((y - pred) ** 2).sum()) / sst


def _skill(task, y, pred, n_classes):
    if task == "classification":
        return (_balanced_accuracy(y, pred, n_classes) - 1 / n_classes) / (1 - 1 / n_classes)
    return _r2(y, pred)


def _erfinv(x):
    """Inverse error function: Winitzki start refined by Newton steps."""
    x = np.clip(np.asarray(x, dtype=float), -1 + 1e-12, 1 - 1e-12)
    a = 0.147
    ln = np.log(1 - x ** 2)
    t = 2 / (pi * a) + ln / 2
    y = np.sign(x) * np.sqrt(np.sqrt(t ** 2 - ln / a) - t)
    for _ in range(3):
        y = y - (_ERF(y) - x) / (2 / np.sqrt(pi) * np.exp(-y ** 2))
    return y


# ----------------------------------------------------------------- fast linear CV tools
def _fold_ids(y, task, n_folds, seed):
    if task == "classification":
        return stratified_folds(y, n_folds, seed)
    return kfold_ids(len(y), n_folds, seed)


def _targets_matrix(y, task, n_classes):
    return np.eye(n_classes)[y] if task == "classification" else y[:, None].astype(float)


def _ridge_fit(Z, T, lam):
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Zs = (Z - mu) / sd
    tm = T.mean(0)
    A = Zs.T @ Zs + (lam + 1e-10) * np.eye(Z.shape[1])
    return mu, sd, np.linalg.solve(A, Zs.T @ (T - tm)), tm


def _ridge_pred(fit, Z):
    mu, sd, W, tm = fit
    return ((Z - mu) / sd) @ W + tm


def _fast_oof(Z, T, folds, lam=1.0):
    P = np.zeros_like(T, dtype=float)
    for f in np.unique(folds):
        te = folds == f
        P[te] = _ridge_pred(_ridge_fit(Z[~te], T[~te], lam), Z[te]) if Z.shape[1] else T[~te].mean(0)
    return P


def _fast_skill(Z, y, T, folds, task, n_classes, lam=1.0):
    P = _fast_oof(Z, T, folds, lam)
    pred = P.argmax(1) if task == "classification" else P[:, 0]
    return _skill(task, y, pred, n_classes)


# --------------------------------------------------------------------- report container
class TargetIntelligenceReport:
    """Dictionary-backed report with text and JSON renderings."""

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def to_dict(self):
        return copy.deepcopy(self.data)

    def to_json(self, path=None, indent=2):
        text = json.dumps(self.data, indent=indent, default=_json_default)
        if path is not None:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text

    def export(self, directory, formats=("json", "txt", "html"), name="target_intelligence"):
        """Save the report to ``directory`` as json / txt / html. Returns ``{format: path}``."""
        return export_report(self, directory, name=name, formats=formats, text=self.to_text(),
                             title="ReliableML Target Intelligence report")

    def to_text(self):
        d = self.data
        out = [f"TARGET INTELLIGENCE REPORT  (scope={d['scope']}, score={d.get('summary_score', 0):.0f}/100)", "=" * 60]
        mc = d["multicollinearity"]
        if d["scope"] == "full":
            tp = d["target_profile"]
            out += ["", "1. Target profile", f"   task={tp['task']}  n={tp['n_rows']}  features={tp['n_features']}"]
            for k, v in tp.items():
                if k not in ("task", "n_rows", "n_features"):
                    out.append(f"   {k}: {_fmt(v)}")
            ex = d["explainability"]
            out += ["", "2. Can the target be explained from X?",
                    f"   verdict: {ex['verdict']}",
                    f"   linear skill={ex['linear_skill']:.3f}  knn skill={ex['knn_skill']:.3f}  "
                    f"permutation p={ex['permutation_p_value']:.3f}",
                    f"   learning-curve gain (75%->100% rows): {ex['learning_curve_gain']:+.3f}",
                    f"   nonlinearity gap (knn - linear): {ex['nonlinearity_gap']:+.3f}"]
            for k in ("estimated_label_noise", "events_per_variable"):
                if k in ex:
                    out.append(f"   {k}: {_fmt(ex[k])}")
            lk = d["leakage"]
            out += ["", f"3. Leakage / target-linkage screening: {len(lk['flags'])} flag(s)"]
            for f in lk["flags"]:
                out.append(f"   [{f['severity']}] {f['feature']}: {f['reason']}")
            out.append("   top associations with the target:")
            for row in lk["associations"][:8]:
                out.append(f"     {row['feature']:<24} strength={row['strength']:.3f} "
                           f"single-feature skill={row['single_feature_skill']:+.3f}")
            cs = d["causal_screening"]
            out += ["", "4. Causal screening (associational evidence only)", f"   {cs['note']}"]
            for row in cs["suppression_or_confounding"][:6]:
                out.append(f"   {row['feature']}: marginal r={row['marginal_r']:+.2f}, "
                           f"partial r={row['partial_r']:+.2f}  ({row['reason']})")
            out += ["", "5. Candidate targets (compare only within a group)"]
            for row in d["target_candidates"]:
                out.append(f"   [{row['group']}] {row['name']:<30} skill linear={row['linear_skill']:+.3f} "
                           f"knn={row['knn_skill']:+.3f}  excluded={row['excluded_features'] or '-'}")
            out.append(f"   best in original group: {d['best_target']}")
            fs = d["feature_selection"]
            out += ["", "6. Feature-selection experiments (all combinations)",
                    f"   {len(fs['subsets'])} unique subsets tried; best: {fs['best_method']} "
                    f"({len(fs['best_features'])} features, skill {fs['best_skill']:.3f}, "
                    f"all-features skill {fs['all_features_skill']:.3f})"]
            for row in fs["subsets"][:8]:
                out.append(f"     {row['method']:<34} k={row['n_features']:<3} fast skill={row['fast_skill']:+.3f}")
        out += ["", "7. Multicollinearity",
                f"   condition number={mc['condition_number']:.1f}  max VIF={mc['max_vif']:.1f}  "
                f"redundant pairs={len(mc['redundant_pairs'])}"]
        for p in mc["redundant_pairs"][:8]:
            out.append(f"     {p['a']} ~ {p['b']}  assoc={p['association']:.3f}")
        out.append(f"   VIF-pruned drops (threshold {mc['vif_threshold']}): {mc['recommended_drop'] or 'none'}")
        if d["scope"] == "full":
            out += ["", "8. Recommendations"] + [f"   - {r}" for r in d["recommendations"]]
        out += ["", "Note: association is not causation; verify timing of every flagged feature."]
        return "\n".join(out)

    def __repr__(self):
        return f"TargetIntelligenceReport(scope={self.data['scope']!r})"


def _summary_score(d):
    """0-100 reliability score: penalties for leakage, noise-level targets, collinearity, label noise."""
    mc = d["multicollinearity"]
    score = 100.0 - min(25.0, 2.0 * len(mc["recommended_drop"])) - (10.0 if mc["condition_warning"] else 0.0)
    if d["scope"] == "full":
        high = sum(f["severity"] == "high" for f in d["leakage"]["flags"])
        score -= min(60.0, 25.0 * high)
        verdict = d["explainability"]["verdict"]
        score -= 25.0 if verdict.startswith("not distinguishable") else (8.0 if verdict == "weakly explained" else 0.0)
        score -= 5.0 if d["target_profile"]["duplicate_rows"] else 0.0
        score -= 5.0 if d["explainability"].get("estimated_label_noise", 0) > 0.25 else 0.0
    return float(np.clip(score, 0.0, 100.0))


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


EXPORT_FORMATS = ("json", "txt", "html")


def export_report(data, directory, name="report", formats=("json", "txt", "html"), text=None, title=None,
                  figure_paths=None):
    """Write a report (a dict, or any object with ``to_dict``) to ``directory``.

    ``formats`` is any of ``json``, ``txt`` and ``html``. The HTML file is self-contained: figures listed in
    ``figure_paths`` ({name: png path}) are embedded as base64. Returns ``{format: path}``.
    """
    import base64
    import html as _html

    formats = (formats,) if isinstance(formats, str) else tuple(formats)
    bad = [f for f in formats if f not in EXPORT_FORMATS]
    if bad:
        raise ValueError(f"formats must be a subset of {EXPORT_FORMATS}; got {bad}")
    payload = data.to_dict() if hasattr(data, "to_dict") else data
    payload = json.loads(json.dumps(payload, default=_json_default))
    os.makedirs(directory, exist_ok=True)
    text = text if text is not None else json.dumps(payload, indent=2)
    written: dict[str, str] = {}
    for fmt in formats:
        path = os.path.join(directory, f"{name}.{fmt}")
        if fmt == "json":
            body = json.dumps(payload, indent=2)
        elif fmt == "txt":
            body = text
        else:
            imgs = ""
            for fig_name, fig_path in (figure_paths or {}).items():
                if os.path.exists(fig_path):
                    with open(fig_path, "rb") as handle:
                        enc = base64.b64encode(handle.read()).decode("ascii")
                    imgs += (f"<h3>{_html.escape(fig_name)}</h3>"
                             f"<img alt='{_html.escape(fig_name)}' style='max-width:100%' "
                             f"src='data:image/png;base64,{enc}'>")
            heading = _html.escape(title or name)
            body = (f"<!doctype html><html><head><meta charset='utf-8'><title>{heading}</title>"
                    "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto}"
                    "pre{background:#f5f5f5;padding:1em;overflow:auto}</style></head>"
                    f"<body><h1>{heading}</h1><pre>{_html.escape(text)}</pre>{imgs}</body></html>")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
        written[fmt] = path
    return written


# ------------------------------------------------------------------- multicollinearity
def _association_matrix(fr: _Frame, names):
    p = len(names)
    A = np.eye(p)
    for i, j in combinations(range(p), 2):
        a, b = names[i], names[j]
        ka, kb = fr.kind[a], fr.kind[b]
        if ka == "numeric" and kb == "numeric":
            s = abs(_corr(fr.values[a], fr.values[b]))
        elif ka == "categorical" and kb == "categorical":
            s = _cramers_v(fr.codes[a], fr.codes[b])
        else:
            num, cat = (a, b) if ka == "numeric" else (b, a)
            s = _eta(fr.values[num], fr.codes[cat])
        A[i, j] = A[j, i] = s
    return A


def _vif_values(V):
    R = np.corrcoef(V, rowvar=False) if V.shape[1] > 1 else np.ones((1, 1))
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    inv = np.linalg.inv(R + 1e-9 * np.eye(len(R)))
    return np.clip(np.diag(inv), 1.0, 1e6), R


def _vif_prune(fr: _Frame, names, threshold, priority=None):
    """Iteratively drop the column with the largest VIF (lowest priority breaks near-ties)."""
    keep = [n for n in names if fr.kind[n] == "numeric" and np.std(fr.values[n]) > 1e-12]
    prio = {n: (priority or {}).get(n, 0.0) for n in keep}
    dropped, steps = [], []
    while len(keep) > 1:
        vif, _ = _vif_values(np.column_stack([fr.values[n] for n in keep]))
        top = vif.max()
        if top <= threshold:
            break
        cand = [i for i in range(len(keep)) if vif[i] >= 0.95 * top]
        j = min(cand, key=lambda i: (prio[keep[i]], -vif[i]))
        steps.append({"dropped": keep[j], "vif": float(vif[j])})
        dropped.append(keep.pop(j))
    return keep, dropped, steps


def _corr_prune(fr: _Frame, names, threshold, priority):
    """Drop the lower-priority member of every numeric pair with |r| above threshold."""
    num = [n for n in names if fr.kind[n] == "numeric" and np.std(fr.values[n]) > 1e-12]
    drop = set()
    for a, b in combinations(num, 2):
        if a in drop or b in drop:
            continue
        if abs(_corr(fr.values[a], fr.values[b])) >= threshold:
            drop.add(a if priority.get(a, 0.0) < priority.get(b, 0.0) else b)
    return [n for n in names if n not in drop]


def multicollinearity_report(X, *, vif_threshold=10.0, corr_threshold=0.9, max_columns=150):
    """Collinearity intelligence for ``X``; never touches a target.

    Returns a plain dict with redundant pairs and clusters, VIF per numeric column,
    the condition number of the correlation matrix, and a VIF-pruned drop list.
    """
    fr = X if isinstance(X, _Frame) else _Frame(X)
    names = fr.names[:max_columns]
    A = _association_matrix(fr, names)
    pairs = [{"a": names[i], "b": names[j], "association": float(A[i, j])}
             for i, j in combinations(range(len(names)), 2) if A[i, j] >= corr_threshold]
    pairs.sort(key=lambda r: -r["association"])
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pr in pairs:
        parent[find(names.index(pr["a"]))] = find(names.index(pr["b"]))
    groups: dict[int, list[str]] = {}
    for i, n in enumerate(names):
        groups.setdefault(find(i), []).append(n)
    num = [n for n in names if fr.kind[n] == "numeric" and np.std(fr.values[n]) > 1e-12]
    vif, cond = {}, 1.0
    if len(num) >= 2:
        v, R = _vif_values(np.column_stack([fr.values[n] for n in num]))
        vif = {n: float(x) for n, x in zip(num, v)}
        ev = np.clip(np.linalg.eigvalsh(R), 1e-12, None)
        cond = float(np.sqrt(ev.max() / ev.min()))
    _, dropped, steps = _vif_prune(fr, names, vif_threshold)
    return {
        "n_columns": len(names), "n_numeric": len(num), "vif": vif,
        "max_vif": float(max(vif.values())) if vif else 1.0,
        "condition_number": cond, "condition_warning": cond > 30,
        "redundant_pairs": pairs[:50],
        "redundant_clusters": [g for g in groups.values() if len(g) > 1],
        "constant_columns": [n for n in names if fr.kind[n] == "numeric" and np.std(fr.values[n]) <= 1e-12],
        "vif_threshold": vif_threshold, "corr_threshold": corr_threshold,
        "recommended_drop": dropped, "vif_steps": steps,
        "kept_after_vif": [n for n in names if n not in dropped],
        "note": "VIF and condition number use numeric columns; categorical links use "
                "Cramer's V and the correlation ratio.",
    }


# ---------------------------------------------------------------------- main estimator
class TargetIntelligence:
    """Experiment with the target, its leakage risk, its explainability and the features.

    Parameters
    ----------
    scope : ``"full"`` or ``"multicollinearity"`` (the target is never read).
    task : ``"auto"``, ``"classification"`` or ``"regression"``.
    vif_threshold, corr_threshold : collinearity pruning thresholds.
    leakage_threshold : single-feature skill at or above this raises a leakage flag.
    n_folds, max_rows, random_state : cross-validation settings and row cap.
    n_permutations : label permutations for the explainability p-value.
    search_targets, search_features : switch the two expensive searches.
    max_selection_features : the feature searches use the strongest this many.
    min_gain : minimum skill gain for forward selection to keep adding features.
    """

    def __init__(self, scope="full", task="auto", vif_threshold=10.0, corr_threshold=0.9,
                 leakage_threshold=0.95, n_folds=5, max_rows=4000, n_permutations=20,
                 search_targets=True, search_features=True, max_selection_features=30,
                 min_gain=0.002, random_state=0):
        check_variant(scope, SCOPES, "TargetIntelligence scope")
        check_variant(task, ("auto", "classification", "regression"), "TargetIntelligence task")
        if vif_threshold <= 1 or not 0 < corr_threshold <= 1:
            raise ValueError("vif_threshold must exceed 1 and corr_threshold be in (0, 1]")
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        self.scope = scope
        self.task = task
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold
        self.leakage_threshold = leakage_threshold
        self.n_folds = n_folds
        self.max_rows = max_rows
        self.n_permutations = n_permutations
        self.search_targets = search_targets
        self.search_features = search_features
        self.max_selection_features = max_selection_features
        self.min_gain = min_gain
        self.random_state = random_state
        self.is_fitted_ = False

    # ------------------------------------------------------------------ public API
    def fit(self, X, y=None, target_name=None):
        fr = _Frame(X)
        self.feature_names_in_ = list(fr.names)
        if self.scope == "multicollinearity":
            mc = multicollinearity_report(fr, vif_threshold=self.vif_threshold,
                                          corr_threshold=self.corr_threshold)
            self.best_features_ = list(mc["kept_after_vif"])
            self.best_target_ = None
            self.report_ = TargetIntelligenceReport({
                "scope": "multicollinearity", "multicollinearity": mc,
                "summary_score": _summary_score({"scope": "multicollinearity", "multicollinearity": mc}),
            })
            self.is_fitted_ = True
            return self
        if y is None:
            raise ValueError("scope='full' needs y; use scope='multicollinearity' to ignore the target")
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != fr.n:
            raise ValueError("y must be 1-D with one value per row of X")
        if pd.isna(pd.Series(y)).any():
            raise ValueError("y contains missing values; drop or impute them first")
        self._target_name = target_name or "target"
        task = self._detect_task(y)
        rows = self._row_sample(y, task, np.random.default_rng(self.random_state))
        self._fit_full(fr.subset(rows), y[rows], task)
        self.is_fitted_ = True
        return self

    @property
    def report(self) -> TargetIntelligenceReport:
        self._check()
        return self.report_

    def transform(self, X):
        """Keep only ``best_features_`` (DataFrame in, DataFrame out)."""
        self._check()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("transform expects a pandas DataFrame")
        return X[self.best_features_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def automl_hook(self, X, y=None):
        """Multicollinearity-only payload for AutoML (never reads ``y``)."""
        mc = multicollinearity_report(X, vif_threshold=self.vif_threshold, corr_threshold=self.corr_threshold)
        return {"multicollinearity_report": mc,
                "summary_score": _summary_score({"scope": "multicollinearity", "multicollinearity": mc})}

    def recommend_models(self):
        """Suggest StatCore model families from the measured linear/non-linear gap."""
        self._check()
        if self.scope != "full":
            raise ValueError("recommend_models needs scope='full'")
        ex = self.report_["explainability"]
        if ex["verdict"].startswith("not distinguishable"):
            return {"families": [], "reason": "target not distinguishable from noise with these features"}
        gap = ex["nonlinearity_gap"]
        if gap > 0.05:
            return {"families": ["neural (StatCoreMLPClassifier)", "instance-based (StatCoreKNNClassifier)",
                                 "meta/stacking (StatCoreMetaClassifier)"],
                    "reason": f"non-linear gap {gap:+.3f}: a linear model misses signal"}
        return {"families": ["linear (StatCoreLogisticClassifier)", "discriminant (StatCoreLDAClassifier)",
                             "meta/stacking (StatCoreMetaClassifier)"],
                "reason": f"non-linear gap {gap:+.3f}: linear is already near the best probe"}

    def evaluate_estimator(self, estimator, X, y, n_folds=None):
        """Cross-validate ``estimator`` on all features vs ``best_features_``.

        The selection was learned on the data given to ``fit``; pass fresh data (or a
        training partition) here to keep the comparison honest.
        """
        self._check()
        fr = _Frame(X)
        y = np.asarray(y)
        task = self._detect_task(y)
        enc = pd.factorize(y, sort=True)[0] if task == "classification" else y.astype(float)
        folds = _fold_ids(enc, task, n_folds or self.n_folds, self.random_state)
        n_classes = int(enc.max()) + 1 if task == "classification" else 0
        out = {}
        for label, cols in (("all_features", fr.names), ("selected_features", self.best_features_)):
            M = fr.matrix(cols)
            pred = np.zeros(len(enc), dtype=enc.dtype)
            for f in np.unique(folds):
                te = folds == f
                pred[te] = copy.deepcopy(estimator).fit(M[~te], enc[~te]).predict(M[te])
            out[label] = {"n_features": len(cols), "skill": _skill(task, enc, pred, n_classes)}
        out["gain"] = out["selected_features"]["skill"] - out["all_features"]["skill"]
        return out

    # ----------------------------------------------------------------- internals
    def _check(self):
        if not self.is_fitted_:
            raise RuntimeError("TargetIntelligence is not fitted; call fit() first")

    def _detect_task(self, y):
        if self.task != "auto":
            return self.task
        s = pd.Series(y)
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            return "classification"
        nun = s.nunique()
        if nun <= 2 or (nun <= 15 and np.allclose(s, np.round(s)) and nun / len(s) < 0.05):
            return "classification"
        return "regression"

    def _row_sample(self, y, task, rng):
        n = len(y)
        if n <= self.max_rows:
            return np.arange(n)
        if task == "classification":
            idx = []
            for c in np.unique(y):
                m = np.flatnonzero(y == c)
                k = max(2, int(round(len(m) * self.max_rows / n)))
                idx.append(rng.choice(m, size=min(k, len(m)), replace=False))
            return np.sort(np.concatenate(idx))
        return np.sort(rng.choice(n, size=self.max_rows, replace=False))

    @staticmethod
    def _encode_target(y, task):
        if task == "classification":
            codes, classes = pd.factorize(pd.Series(y), sort=True)
            return codes.astype(int), list(classes)
        return np.asarray(y, dtype=float), None

    def _model_cv(self, M, y, task, n_classes, kind, folds):
        """Out-of-fold skill (and fold spread) of a StatCore model: ``linear`` or ``knn``."""
        pred = np.zeros(len(y), dtype=float if task == "regression" else int)
        scores = []
        for f in np.unique(folds):
            te = folds == f
            try:
                if kind == "linear":
                    model = (LogisticRegression(variant="base", max_iter=100, random_state=self.random_state)
                             if task == "classification" else RidgeRegression(alpha=1.0))
                else:
                    k = int(np.clip(round(np.sqrt((~te).sum()) / 2), 3, 25))
                    model = KNNClassifier(n_neighbors=k) if task == "classification" else KNNRegressor(n_neighbors=k)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(M[~te], y[~te])
                    pred[te] = model.predict(M[te])
            except Exception:
                return float("nan"), float("nan")
            scores.append(_skill(task, y[te], pred[te], n_classes))
        return _skill(task, y, pred, n_classes), float(np.std(scores))

    # -------------------------------------------------------------------- full run
    def _fit_full(self, fr, y_raw, task):
        y, classes = self._encode_target(y_raw, task)
        n_classes = len(classes) if classes else 0
        folds_n, notes = self.n_folds, []
        if task == "classification":
            counts = np.bincount(y)
            if counts.min() < 2:
                raise ValueError("every class needs at least 2 rows")
            if counts.min() < folds_n:
                folds_n = int(counts.min())
                notes.append(f"n_folds reduced to {folds_n} (smallest class has {folds_n} rows)")
        folds = _fold_ids(y, task, folds_n, self.random_state)
        T = _targets_matrix(y, task, n_classes)
        names = fr.names

        profile = self._profile(fr, y, task, classes)
        assoc = self._associations(fr, y, task, n_classes, folds)
        priority = {a["feature"]: a["strength"] for a in assoc}
        leakage = self._leakage(fr, assoc, task, y, n_classes)
        causal = self._causal(fr, y)
        mc = multicollinearity_report(fr, vif_threshold=self.vif_threshold, corr_threshold=self.corr_threshold)

        Mall = fr.matrix(names)
        lin, lin_std = self._model_cv(Mall, y, task, n_classes, "linear", folds)
        knn, _ = self._model_cv(Mall, y, task, n_classes, "knn", folds)
        explain = self._explainability(Mall, y, T, folds, task, n_classes, lin, lin_std, knn)
        explain["notes"] = notes

        candidates, best_target = [], self._target_name
        if self.search_targets:
            candidates, best_target = self._target_search(fr, y, task, n_classes, assoc)
        if self.search_features:
            selection = self._feature_search(fr, y, T, task, n_classes, folds, assoc, priority, lin, knn)
        else:
            selection = {"subsets": [], "best_method": "all", "best_features": list(names), "pool_size": len(names),
                         "best_skill": float(lin), "all_features_skill": float(lin)}
        self.best_features_ = list(selection["best_features"])
        self.best_target_ = best_target
        recs = self._recommendations(profile, explain, leakage, mc, selection, best_target, causal)
        data = {
            "scope": "full", "target_profile": profile, "leakage": leakage, "causal_screening": causal,
            "explainability": explain, "target_candidates": candidates, "best_target": best_target,
            "feature_selection": selection, "multicollinearity": mc, "recommendations": recs,
        }
        data["summary_score"] = _summary_score(data)
        self.report_ = TargetIntelligenceReport(data)

    # --------------------------------------------------------------------- sections
    def _profile(self, fr, y, task, classes):
        p = {"task": task, "n_rows": fr.n, "n_features": len(fr.names),
             "n_numeric": sum(k == "numeric" for k in fr.kind.values()),
             "n_categorical": sum(k == "categorical" for k in fr.kind.values()),
             "max_missing_rate": max(fr.missing_rate.values())}
        if task == "classification":
            cnt = np.bincount(y)
            pr = cnt / cnt.sum()
            p.update({"classes": [str(c) for c in classes], "class_counts": cnt.tolist(),
                      "imbalance_ratio": float(cnt.max() / cnt.min()),
                      "normalised_entropy": _entropy(pr) / np.log(len(cnt)) if len(cnt) > 1 else 0.0,
                      "majority_baseline_accuracy": float(pr.max())})
        else:
            mad = np.median(np.abs(y - np.median(y))) * 1.4826
            z = (y - y.mean()) / (y.std() + 1e-12)
            p.update({"mean": float(y.mean()), "std": float(y.std()), "skewness": float((z ** 3).mean()),
                      "excess_kurtosis": float((z ** 4).mean() - 3),
                      "outlier_fraction_mad": float((np.abs(y - np.median(y)) > 3.5 * mad).mean()) if mad > 0 else 0.0,
                      "zero_fraction": float((y == 0).mean()), "n_unique": int(len(np.unique(y)))})
        p["duplicate_rows"] = int(pd.DataFrame({n: fr.values[n] for n in fr.names}).duplicated().sum())
        return p

    def _associations(self, fr, y, task, n_classes, folds):
        yb = y if task == "classification" else self._quantile_codes(y)
        hy = _entropy(np.bincount(yb) / len(yb)) or 1.0
        rows = []
        for n in fr.names:
            v, kind = fr.values[n], fr.kind[n]
            spearman = None
            if task == "classification":
                s = _eta(v, y) if kind == "numeric" else _cramers_v(fr.codes[n], y)
            elif kind == "numeric":
                s, spearman = abs(_corr(v, y)), _corr(_rank(v), _rank(y))
            else:
                s = _eta(y, fr.codes[n])
            rows.append({"feature": n, "kind": kind, "strength": float(s), "spearman": spearman,
                         "normalised_mutual_info": float(_mutual_info(fr.codes[n], yb) / hy),
                         "single_feature_skill": float(self._single_skill(fr.codes[n], y, folds, task, n_classes))})
        rows.sort(key=lambda r: -max(r["strength"], r["single_feature_skill"]))
        return rows

    @staticmethod
    def _quantile_codes(v, q=10):
        qs = np.unique(np.quantile(v, np.linspace(0, 1, q + 1)[1:-1]))
        return np.searchsorted(qs, v, side="right").astype(int)

    @staticmethod
    def _single_skill(codes, y, folds, task, n_classes):
        nb = codes.max() + 1
        pred = np.zeros(len(y), dtype=float if task == "regression" else int)
        for f in np.unique(folds):
            te, tr = folds == f, folds != f
            if task == "classification":
                tab = np.bincount(codes[tr] * n_classes + y[tr], minlength=nb * n_classes).reshape(nb, n_classes)
                prior = np.bincount(y[tr], minlength=n_classes).argmax()
                pred[te] = np.where(tab.sum(1) > 0, tab.argmax(1), prior)[codes[te]]
            else:
                cnt = np.bincount(codes[tr], minlength=nb)
                mean = np.bincount(codes[tr], weights=y[tr], minlength=nb) / np.maximum(cnt, 1)
                pred[te] = np.where(cnt > 0, mean, y[tr].mean())[codes[te]]
        return _skill(task, y, pred, n_classes)

    def _leakage(self, fr, assoc, task, y, n_classes):
        flags = []
        tname = self._target_name.lower()
        for a in assoc:
            n = a["feature"]
            lname = n.lower()
            ratio = len(np.unique(fr.values[n])) / fr.n
            if a["single_feature_skill"] >= self.leakage_threshold:
                flags.append({"feature": n, "severity": "high",
                              "reason": f"one feature alone predicts the target out-of-fold (skill {a['single_feature_skill']:.3f}); "
                                        "check whether it is only known after the outcome"})
            elif a["strength"] >= self.leakage_threshold:
                flags.append({"feature": n, "severity": "high",
                              "reason": f"association with the target is {a['strength']:.3f}; suspected duplicate or derived target"})
            if task == "regression" and a["spearman"] is not None and abs(a["spearman"]) >= 0.999:
                flags.append({"feature": n, "severity": "high", "reason": "monotone copy of the target (|Spearman| ~ 1)"})
            integer_like = fr.kind[n] == "categorical" or bool(np.allclose(fr.values[n], np.round(fr.values[n])))
            if ratio >= 0.95 and fr.n > 20 and integer_like:
                flags.append({"feature": n, "severity": "medium",
                              "reason": f"almost every row has a unique value ({ratio:.0%}): identifier-like, can memorise rows"})
            if tname != "target" and tname in lname:
                flags.append({"feature": n, "severity": "medium", "reason": "column name contains the target name"})
            elif any(w in lname for w in _ID_WORDS) and a["single_feature_skill"] > 0.3:
                flags.append({"feature": n, "severity": "low",
                              "reason": "name suggests an outcome/after-the-fact field and it is strongly predictive"})
            if fr.kind[n] == "categorical":
                miss = fr.values[n] == "__missing__"
                if miss.any() and not miss.all():
                    s = self._single_skill(miss.astype(int), y, _fold_ids(y, task, 3, self.random_state), task, n_classes)
                    if s >= 0.2:
                        flags.append({"feature": n, "severity": "medium",
                                      "reason": f"missingness itself predicts the target (skill {s:.2f})"})
        return {"flags": flags, "associations": assoc, "threshold": self.leakage_threshold}

    def _causal(self, fr, y):
        out = {"note": "Partial correlation separates direct from shared association but cannot orient "
                       "cause and effect; time order and domain knowledge are required.",
               "suppression_or_confounding": [], "direct_evidence": []}
        num = fr.numeric()
        if len(num) < 2:
            return out
        R = np.nan_to_num(np.corrcoef(np.column_stack([fr.values[n] for n in num] + [y.astype(float)]), rowvar=False))
        np.fill_diagonal(R, 1.0)
        P = np.linalg.inv(R + 1e-6 * np.eye(len(R)))
        d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
        partial = -P / np.outer(d, d)
        rows = []
        for i, n in enumerate(num):
            m, pr = float(R[i, -1]), float(partial[i, -1])
            if abs(m) < 0.1 and abs(pr) < 0.1:
                continue
            if m * pr < 0 and abs(m) > 0.1 and abs(pr) > 0.1:
                reason = "sign flips once other features are held fixed (suppression)"
            elif abs(m) > 0.3 and abs(pr) < 0.3 * abs(m):
                reason = "association vanishes given the others (shared cause or proxy)"
            elif abs(pr) > 1.5 * abs(m) and abs(pr) > 0.2:
                reason = "association strengthens given the others (suppressor effect)"
            else:
                continue
            rows.append({"feature": n, "marginal_r": m, "partial_r": pr, "reason": reason})
        rows.sort(key=lambda r: -abs(r["marginal_r"] - r["partial_r"]))
        out["suppression_or_confounding"] = rows
        order = sorted(range(len(num)), key=lambda i: -abs(partial[i, -1]))[:10]
        out["direct_evidence"] = [{"feature": num[i], "partial_r": float(partial[i, -1])} for i in order]
        return out

    def _explainability(self, M, y, T, folds, task, n_classes, lin, lin_std, knn):
        rng = np.random.default_rng(self.random_state + 1)
        obs = _fast_skill(M, y, T, folds, task, n_classes)
        null = np.asarray([_fast_skill(M, y[p], T[p], folds, task, n_classes)
                           for p in (rng.permutation(len(y)) for _ in range(self.n_permutations))])
        p_value = float((1 + (null >= obs).sum()) / (1 + len(null)))
        curve = []
        for frac in (0.75, 1.0):
            sub = np.sort(rng.choice(len(y), size=int(len(y) * frac), replace=False)) if frac < 1 else np.arange(len(y))
            curve.append(_fast_skill(M[sub], y[sub], T[sub], _fold_ids(y[sub], task, min(self.n_folds, 5), self.random_state),
                                     task, n_classes))
        best = float(np.nanmax([lin, knn]))
        beats_null = bool(obs > null.max())
        if (p_value > 0.05 and not beats_null) or best < 0.02:
            verdict = "not distinguishable from noise with these features"
        elif best < 0.15:
            verdict = "weakly explained"
        elif best < 0.45:
            verdict = "moderately explained"
        else:
            verdict = "strongly explained"
        gap = float(knn - lin) if np.isfinite(knn) and np.isfinite(lin) else 0.0
        out = {"linear_skill": float(lin), "linear_skill_fold_std": float(lin_std), "knn_skill": float(knn),
               "best_skill": best, "nonlinearity_gap": gap, "permutation_null_mean": float(null.mean()),
               "permutation_null": [float(x) for x in null],
               "permutation_null_max": float(null.max()), "permutation_p_value": p_value,
               "learning_curve_gain": float(curve[1] - curve[0]), "verdict": verdict,
               "beats_all_permutations": beats_null}
        if task == "classification":
            out["events_per_variable"] = float(np.bincount(y).min() / max(M.shape[1], 1))
            out["estimated_label_noise"] = self._label_noise(M, y)
        return out

    @staticmethod
    def _label_noise(M, y, k=5, cap=1500):
        """Share of rows whose label disagrees with the majority of their k nearest neighbours."""
        idx = np.arange(len(y)) if len(y) <= cap else np.random.default_rng(0).choice(len(y), cap, replace=False)
        Z = M[idx]
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 1e-12, Z.std(0), 1.0)
        d = (Z ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2 * Z @ Z.T
        np.fill_diagonal(d, np.inf)
        votes = y[idx][np.argsort(d, axis=1)[:, :k]]
        return float((np.array([np.bincount(r).argmax() for r in votes]) != y[idx]).mean())

    # ------------------------------------------------------------ alternative targets
    def _target_search(self, fr, y, task, n_classes, assoc):
        rows = []

        def add(name, group, yy, t, nc, excluded=(), note=""):
            cols = [n for n in fr.names if n not in excluded]
            if not cols or (t == "classification" and np.bincount(yy).min() < 2):
                return
            nf = self.n_folds if t == "regression" else max(2, min(self.n_folds, int(np.bincount(yy).min())))
            f = _fold_ids(yy, t, nf, self.random_state)
            M = fr.matrix(cols)
            ls, _ = self._model_cv(M, yy, t, nc, "linear", f)
            ks, _ = self._model_cv(M, yy, t, nc, "knn", f)
            rows.append({"name": name, "group": group, "task": t, "linear_skill": float(ls),
                         "knn_skill": float(ks), "excluded_features": list(excluded), "note": note})

        if task == "regression":
            add("original", "regression-scale", y, "regression", 0)
            add("log(y - min + 1)", "regression-scale", np.log(y - y.min() + 1.0), "regression", 0, note="monotone transform")
            add("sqrt(y - min)", "regression-scale", np.sqrt(y - y.min()), "regression", 0, note="monotone transform")
            add("rank-gauss(y)", "regression-scale", self._rank_gauss(y), "regression", 0, note="monotone transform")
            for c in [a["feature"] for a in assoc if a["kind"] == "numeric"][:4]:
                v = fr.values[c]
                if np.std(v) < 1e-12:
                    continue
                b = np.cov(v, y, bias=True)[0, 1] / np.var(v)
                note = "target built from a column: that column is removed from X"
                add(f"y residual after {c}", "derived-from-column", y - (y.mean() + b * (v - v.mean())),
                    "regression", 0, (c,), note)
                if (v > 0).all():
                    add(f"y / {c}", "derived-from-column", y / v, "regression", 0, (c,), note)
            for q in (2, 3, 4):
                add(f"{q} quantile classes", "classification-bins", self._quantile_codes(y, q), "classification", q,
                    note="regression turned into bands")
            return rows, self._best_in_group(rows, "regression-scale")

        add("original", "classification-labels", y, "classification", n_classes)
        if n_classes > 2:
            for c in np.argsort(-np.bincount(y))[: min(n_classes, 5)]:
                add(f"class {c} vs rest", "binary-views", (y == c).astype(int), "classification", 2, note="one-vs-rest view")
            folds = _fold_ids(y, task, min(self.n_folds, int(np.bincount(y).min())), self.random_state)
            pred = self._oof_labels(fr.matrix(fr.names), y, folds)
            conf = np.zeros((n_classes, n_classes))
            for a, b in zip(y, pred):
                conf[a, b] += 1
            cur, labels = y.copy(), list(range(n_classes))
            while len(labels) > 2:
                cc = np.array([[conf[a, b] + conf[b, a] if a != b else -1 for b in labels] for a in labels])
                i, j = np.unravel_index(cc.argmax(), cc.shape)
                a, b = labels[i], labels[j]
                cur = np.where(cur == b, a, cur)
                conf[a, :] += conf[b, :]
                conf[:, a] += conf[:, b]
                labels.remove(b)
                add(f"merge most-confused -> {len(labels)} classes", "merged-classes",
                    pd.factorize(cur, sort=True)[0], "classification", len(labels),
                    note="merges the pair the model confuses most")
        return rows, "original"

    @staticmethod
    def _rank_gauss(y):
        r = (pd.Series(y).rank(method="average").to_numpy() - 0.5) / len(y)
        return np.sqrt(2) * _erfinv(2 * r - 1)

    def _oof_labels(self, M, y, folds):
        pred = np.zeros(len(y), dtype=int)
        for f in np.unique(folds):
            te = folds == f
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred[te] = LogisticRegression(variant="base", max_iter=100).fit(M[~te], y[~te]).predict(M[te])
        return pred

    @staticmethod
    def _best_in_group(rows, group):
        g = [r for r in rows if r["group"] == group]
        if not g:
            return "original"
        value = lambda r: float(np.nanmax([r["linear_skill"], r["knn_skill"]]))
        orig = next((r for r in g if r["name"] == "original"), g[0])
        top = max(g, key=value)
        return top["name"] if value(top) - value(orig) > 0.03 else "original"

    # --------------------------------------------------------- feature-selection search
    def _feature_search(self, fr, y, T, task, n_classes, folds, assoc, priority, lin, knn):
        names = list(fr.names)
        strength = {a["feature"]: max(a["strength"], a["single_feature_skill"], a["normalised_mutual_info"]) for a in assoc}
        pool = sorted(names, key=lambda n: -strength[n])[: self.max_selection_features]
        mi = {a["feature"]: a["normalised_mutual_info"] for a in assoc}
        lasso_cache: dict[frozenset, list[str]] = {}

        def score(cols):
            return _fast_skill(fr.matrix(cols), y, T, folds, task, n_classes) if cols else -1.0

        stage1 = {
            "none": list(pool),
            "variance": [n for n in pool if fr.kind[n] == "categorical" or np.std(fr.values[n]) > 1e-12],
            "corr_prune": _corr_prune(fr, pool, self.corr_threshold, priority),
            "vif_prune": [n for n in pool if n not in set(_vif_prune(fr, pool, self.vif_threshold, priority)[1])],
        }

        def top_by(table):
            return lambda cols: sorted(cols, key=lambda n: -table.get(n, 0))[: max(3, len(cols) // 2)]

        def sel_lasso(cols):
            key = frozenset(cols)
            if key not in lasso_cache:
                lasso_cache[key] = self._lasso(fr, cols, y, task, n_classes)
            return lasso_cache[key]

        def sel_forward(cols):
            chosen, best = [], -1.0
            while len(chosen) < len(cols):
                s, c = max((score(chosen + [c]), c) for c in cols if c not in chosen)
                if s < best + self.min_gain:
                    break
                chosen.append(c)
                best = s
            return chosen or cols[:1]

        def sel_backward(cols):
            cur, best = list(cols), score(cols)
            while len(cur) > 1:
                s, c = max((score([x for x in cur if x != c]), c) for c in cur)
                if s < best - 0.0005:
                    break
                cur.remove(c)
                best = max(best, s)
            return cur

        selectors = {"none": list, "assoc_top": top_by(priority), "mutual_info_top": top_by(mi), "lasso": sel_lasso,
                     "forward": sel_forward, "backward": sel_backward,
                     "permutation": lambda c: self._permutation_select(fr, c, y, T, folds, task, n_classes)}
        seen: dict[frozenset, dict] = {}
        for s1, cols1 in stage1.items():
            if not cols1:
                continue
            for s2, fn in selectors.items():
                chosen = fn(cols1)
                key = frozenset(chosen)
                label = s2 if s1 == "none" else (s1 if s2 == "none" else f"{s1} -> {s2}")
                if key in seen:
                    seen[key]["also_found_by"].append(label)
                else:
                    seen[key] = {"method": label, "features": [n for n in names if n in key], "n_features": len(key),
                                 "fast_skill": float(score(chosen)), "also_found_by": []}
        subsets = sorted(seen.values(), key=lambda r: (-r["fast_skill"], r["n_features"]))
        top = subsets[:6]
        full = next((r for r in subsets if r["n_features"] == len(pool)), None)
        if full is not None and full not in top:
            top.append(full)
        for r in top:
            M = fr.matrix(r["features"])
            r["linear_skill"] = float(self._model_cv(M, y, task, n_classes, "linear", folds)[0])
            r["knn_skill"] = float(self._model_cv(M, y, task, n_classes, "knn", folds)[0])
        value = lambda r: float(np.nanmax([r["linear_skill"], r["knn_skill"]]))
        best = max(value(r) for r in top)
        chosen = min((r for r in top if value(r) >= best - 0.005), key=lambda r: r["n_features"])
        return {"pool_size": len(pool), "stage1_filters": {k: len(v) for k, v in stage1.items()},
                "subsets": subsets, "best_method": chosen["method"], "best_features": chosen["features"],
                "best_skill": value(chosen),
                "all_features_skill": float(np.nanmax([lin, knn])),
                "note": "searches use the strongest features by association (pool_size) and reuse one set of "
                        "cross-validation folds, so treat gains as optimistic and confirm on held-out data."}

    def _lasso(self, fr, cols, y, task, n_classes):
        M = fr.matrix(cols)
        owner = np.concatenate([[i] * fr.block(c).shape[1] for i, c in enumerate(cols)])
        Z = (M - M.mean(0)) / np.where(M.std(0) > 1e-12, M.std(0), 1.0)
        n = len(Z)
        if task == "classification":
            onehot = _targets_matrix(y, task, n_classes)
            targets = [onehot[:, 1]] if n_classes == 2 else [onehot[:, c] for c in np.argsort(-onehot.sum(0))[:4]]
        else:
            targets = [y.astype(float)]
        targets = [t - t.mean() for t in targets]
        grid = max(np.abs(Z.T @ t).max() / n for t in targets) * np.geomspace(1.0, 1e-2, 8)
        folds = kfold_ids(n, 3, self.random_state)

        def descend(Zs, t, lam, w):
            r = t - Zs @ w
            nn = len(Zs)
            sq = (Zs ** 2).sum(0) / nn
            for _ in range(30):
                delta = 0.0
                for j in range(Zs.shape[1]):
                    if sq[j] < 1e-12:
                        continue
                    r += Zs[:, j] * w[j]
                    rho = Zs[:, j] @ r / nn
                    new = np.sign(rho) * max(abs(rho) - lam, 0.0) / sq[j]
                    delta = max(delta, abs(new - w[j]))
                    w[j] = new
                    r -= Zs[:, j] * w[j]
                if delta < 1e-4:
                    break
            return w

        errs = np.zeros(len(grid))
        for f in range(3):
            te = folds == f
            for t in targets:
                w = np.zeros(Z.shape[1])
                for gi, lam in enumerate(grid):
                    w = descend(Z[~te], t[~te], lam, w)
                    errs[gi] += float(((t[te] - Z[te] @ w) ** 2).mean())
        best = int(errs.argmin())
        gi = next(i for i in range(len(grid)) if errs[i] <= errs[best] + errs.std() / np.sqrt(len(errs)))
        used = np.zeros(Z.shape[1], dtype=bool)
        for t in targets:
            w = np.zeros(Z.shape[1])
            for lam in grid[: gi + 1]:
                w = descend(Z, t, lam, w)
            used |= np.abs(w) > 1e-8
        return [cols[i] for i in sorted(set(owner[used].tolist()))] or cols[:1]

    def _permutation_select(self, fr, cols, y, T, folds, task, n_classes):
        rng = np.random.default_rng(self.random_state + 2)
        M = fr.matrix(cols)
        owner = np.concatenate([[i] * fr.block(c).shape[1] for i, c in enumerate(cols)])
        fits, base_pred = [], np.zeros_like(T, dtype=float)
        for f in np.unique(folds):
            te = folds == f
            fit = _ridge_fit(M[~te], T[~te], 1.0)
            fits.append((te, fit))
            base_pred[te] = _ridge_pred(fit, M[te])

        def sc(P):
            return _skill(task, y, P.argmax(1) if task == "classification" else P[:, 0], n_classes)

        base = sc(base_pred)
        drop = np.zeros(len(cols))
        for i in range(len(cols)):
            P = np.zeros_like(base_pred)
            idx = np.flatnonzero(owner == i)
            for te, fit in fits:
                Zt = M[te].copy()
                Zt[:, idx] = Zt[rng.permutation(len(Zt))][:, idx]
                P[te] = _ridge_pred(fit, Zt)
            drop[i] = base - sc(P)
        return [c for c, d in zip(cols, drop) if d > 1e-4] or [cols[int(drop.argmax())]]

    # -------------------------------------------------------------- recommendations
    def _recommendations(self, profile, explain, leakage, mc, selection, best_target, causal):
        recs = []
        high = sorted({f["feature"] for f in leakage["flags"] if f["severity"] == "high"})
        if high:
            recs.append("Investigate possible leakage before modelling: " + ", ".join(high)
                        + ". Remove them unless they are known at prediction time.")
        if explain["verdict"].startswith("not distinguishable"):
            recs.append("The target is not distinguishable from noise with these features: collect better features "
                        "or redefine the target before tuning models.")
        elif explain["verdict"] == "weakly explained":
            recs.append("The target is only weakly explained: expect modest skill; check label quality and consider "
                        "a coarser or derived target.")
        if explain["nonlinearity_gap"] > 0.05:
            recs.append(f"A non-linear probe beats the linear one by {explain['nonlinearity_gap']:.2f} skill: use neural, "
                        "instance-based or meta/stacking models.")
        else:
            recs.append("A linear model is already close to the best probe: prefer simple, well-calibrated models first.")
        if explain["learning_curve_gain"] > 0.02:
            recs.append("Skill is still rising with more rows: collecting more data should help.")
        if profile["task"] == "classification":
            if profile["imbalance_ratio"] > 5:
                recs.append(f"Classes are imbalanced ({profile['imbalance_ratio']:.1f}:1): use balanced accuracy, "
                            "per-class recall and stratified splits.")
            if explain.get("estimated_label_noise", 0) > 0.25:
                recs.append(f"About {explain['estimated_label_noise']:.0%} of labels disagree with their neighbours: "
                            "audit label quality.")
            if explain.get("events_per_variable", 99) < 10:
                recs.append("Fewer than 10 rows of the rarest class per feature: reduce features or use regularised models.")
        if mc["recommended_drop"]:
            recs.append("Drop redundant columns (VIF): " + ", ".join(mc["recommended_drop"]) + ".")
        if mc["condition_warning"]:
            recs.append(f"Condition number {mc['condition_number']:.0f} signals multicollinearity: prefer regularised "
                        "or instance-based models.")
        if len(selection["best_features"]) < selection["pool_size"] if "pool_size" in selection else False:
            recs.append(f"Feature search ({selection['best_method']}) keeps {len(selection['best_features'])} features "
                        f"with skill {selection['best_skill']:.3f} vs {selection['all_features_skill']:.3f} for all.")
        if best_target not in ("original", self._target_name):
            recs.append(f"Alternative target '{best_target}' is clearly more predictable than the original: consider it.")
        if causal["suppression_or_confounding"]:
            recs.append("Some features show suppression/confounding patterns: interpret their coefficients with care.")
        if profile["duplicate_rows"]:
            recs.append(f"{profile['duplicate_rows']} duplicate rows in X can inflate cross-validation scores.")
        return recs


# =========================================================================== feature layer
class FeatureIntelligenceReport:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def to_dict(self):
        return json.loads(json.dumps(self.data, default=_json_default))

    def to_json(self, path=None, indent=2):
        text = json.dumps(self.data, indent=indent, default=_json_default)
        if path is not None:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text

    def export(self, directory, formats=("json", "txt", "html"), name="feature_intelligence"):
        """Save the report to ``directory`` as json / txt / html. Returns ``{format: path}``."""
        return export_report(self, directory, name=name, formats=formats, text=self.to_text(),
                             title="ReliableML Feature Intelligence report")

    def to_text(self):
        d = self.data
        out = [f"FEATURE INTELLIGENCE REPORT  (supervised={d['supervised']}, score={d['summary_score']:.0f}/100)",
               "=" * 60,
               f"features in: {d['n_features_in']}   selected: {len(d['selected_features'])}"]
        for label, key in (("Hazards removed", "hazards"), ("Redundant (|corr|)", "correlation_dropped"),
                           ("VIF-pruned", "vif_dropped")):
            v = d[key]
            out.append(f"{label}: {v if v else 'none'}")
        if d["supervised"]:
            out.append("Aggregated ranking (higher = better): ")
            for row in d["ranking"][:12]:
                out.append(f"   {row['feature']:<26} {row['aggregate']:.3f}  "
                           + "  ".join(f"{m}={row[m]:.2f}" for m in d["methods"]))
        out.append(f"Selected: {d['selected_features']}")
        out += ["Recommendations:"] + [f"   - {r}" for r in d["recommendations"]]
        return "\n".join(out)


class FeatureIntelligence:
    """Prune and select features, optionally guided by a target.

    Methods (all StatCore/NumPy/scipy, no scikit-learn): ``mutual_info`` (binned MI), ``anova``
    (|r|, correlation ratio, Cramer's V), ``chi2`` (independence on binned features, with p-values),
    ``rfe`` (recursive elimination on a ridge model-matrix) and ``model_importance`` (out-of-fold
    permutation importance of a ridge model). Without ``y`` only the unsupervised layers run.
    """

    def __init__(self, methods=METHODS, n_features=None, vif_threshold=10.0, corr_threshold=0.90,
                 variance_threshold=1e-8, quasi_constant=0.99, n_folds=5, max_rows=5000, random_state=0):
        bad = [m for m in methods if m not in METHODS]
        if bad or not methods:
            raise ValueError(f"methods must be a non-empty subset of {METHODS}; got {bad or methods}")
        if n_features is not None and n_features < 1:
            raise ValueError("n_features must be at least 1")
        self.methods = tuple(methods)
        self.n_features = n_features
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        self.quasi_constant = quasi_constant
        self.n_folds = n_folds
        self.max_rows = max_rows
        self.random_state = random_state
        self.is_fitted_ = False

    def fit(self, X, y=None, task="auto"):
        fr = _Frame(X)
        names = list(fr.names)
        yv = None
        if y is not None:
            y = np.asarray(y)
            if len(y) != fr.n or pd.isna(pd.Series(y)).any():
                raise ValueError("y must have one non-missing value per row of X")
            rows = np.arange(fr.n)
            if fr.n > self.max_rows:
                rows = np.sort(np.random.default_rng(self.random_state).choice(fr.n, self.max_rows, replace=False))
            fr, y = fr.subset(rows), y[rows]
            helper = TargetIntelligence(task=task)
            task = helper._detect_task(y)
            yv, _ = helper._encode_target(y, task)

        hazards = self._hazards(fr)
        dropped_hazard = sorted(set(hazards["constant"] + hazards["quasi_constant"] + hazards["low_variance"]))
        alive = [n for n in names if n not in dropped_hazard]

        priority = {n: -fr.missing_rate[n] for n in names}
        if yv is not None and alive:
            assoc = self._anova(fr, yv, task, alive)
            priority = {n: assoc.get(n, 0.0) for n in names}
        kept = _corr_prune(fr, alive, self.corr_threshold, priority)
        corr_dropped = [n for n in alive if n not in kept]
        vif_dropped = list(_vif_prune(fr, kept, self.vif_threshold, priority)[1])
        survivors = [n for n in kept if n not in vif_dropped] or alive[:1] or names[:1]

        data = {"supervised": yv is not None, "task": task if yv is not None else None, "n_features_in": len(names),
                "hazards": dropped_hazard, "hazard_detail": hazards, "correlation_dropped": corr_dropped,
                "vif_dropped": vif_dropped, "survivors": survivors, "methods": list(self.methods),
                "ranking": [], "scores": {}}
        if yv is not None:
            ranking, scores = self._rank(fr, yv, task, survivors)
            data["ranking"], data["scores"] = ranking, scores
            k = self.n_features or max(1, int(np.ceil(len(survivors) / 2)))
            selected = [r["feature"] for r in ranking[:k]]
        else:
            selected = list(survivors if self.n_features is None else survivors[: self.n_features])
        data["selected_features"] = [n for n in names if n in set(selected)]
        data["recommendations"] = self._recommend(data)
        data["summary_score"] = float(np.clip(
            100 - min(40, 8 * len(dropped_hazard)) - min(30, 3 * len(corr_dropped)) - min(30, 4 * len(vif_dropped)), 0, 100))
        self.selected_features_ = data["selected_features"]
        self.report_ = FeatureIntelligenceReport(data)
        self.is_fitted_ = True
        return self

    @property
    def report(self):
        if not self.is_fitted_:
            raise RuntimeError("FeatureIntelligence is not fitted; call fit() first")
        return self.report_

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("FeatureIntelligence is not fitted; call fit() first")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("transform expects a pandas DataFrame")
        return X[self.selected_features_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _hazards(self, fr):
        out = {"constant": [], "quasi_constant": [], "low_variance": [], "high_cardinality": []}
        for n in fr.names:
            v = fr.values[n]
            share = pd.Series(v).value_counts(normalize=True).iloc[0]
            if share >= 1.0:
                out["constant"].append(n)
            elif share >= self.quasi_constant:
                out["quasi_constant"].append(n)
            elif fr.kind[n] == "numeric" and np.var(v) < self.variance_threshold:
                out["low_variance"].append(n)
            if fr.kind[n] == "categorical" and len(np.unique(v)) > 50 and len(np.unique(v)) / fr.n > 0.5:
                out["high_cardinality"].append(n)
        return out

    @staticmethod
    def _ycodes(yv, task):
        return yv if task == "classification" else TargetIntelligence._quantile_codes(yv)

    def _anova(self, fr, yv, task, cols):
        out = {}
        for n in cols:
            v = fr.values[n]
            if fr.kind[n] == "numeric":
                out[n] = _eta(v, yv) if task == "classification" else abs(_corr(v, yv))
            else:
                out[n] = _cramers_v(fr.codes[n], yv) if task == "classification" else _eta(yv, fr.codes[n])
        return out

    def _mutual_info(self, fr, yv, task, cols):
        yc = self._ycodes(yv, task)
        return {n: _mutual_info(fr.codes[n], yc) for n in cols}

    def _chi2(self, fr, yv, task, cols):
        yc = self._ycodes(yv, task)
        out, pvals = {}, {}
        for n in cols:
            tab = _contingency(fr.codes[n], yc)
            tab = tab[tab.sum(1) > 0][:, tab.sum(0) > 0]
            if min(tab.shape) < 2:
                out[n], pvals[n] = 0.0, 1.0
                continue
            chi2, p, _, _ = stats.chi2_contingency(tab)
            out[n], pvals[n] = float(chi2 / (tab.sum() * (min(tab.shape) - 1))), float(p)
        self.chi2_p_values_ = pvals
        return out

    def _rfe(self, fr, yv, task, cols, n_classes):
        T = _targets_matrix(yv, task, n_classes)
        remaining, order = list(cols), []
        while len(remaining) > 1:
            M = fr.matrix(remaining)
            owner = np.concatenate([[i] * fr.block(c).shape[1] for i, c in enumerate(remaining)])
            W = np.abs(_ridge_fit(M, T, 1.0)[2]).sum(1)
            imp = np.bincount(owner, weights=W, minlength=len(remaining))
            for i in sorted(np.argsort(imp)[: max(1, len(remaining) // 10)], reverse=True):
                order.append(remaining.pop(i))
        order.append(remaining[0])
        return {n: float(i) for i, n in enumerate(order)}

    def _model_importance(self, fr, yv, task, cols, n_classes):
        rng = np.random.default_rng(self.random_state)
        T = _targets_matrix(yv, task, n_classes)
        folds = _fold_ids(yv, task, self.n_folds, self.random_state)
        M = fr.matrix(cols)
        owner = np.concatenate([[i] * fr.block(c).shape[1] for i, c in enumerate(cols)])
        drop = np.zeros(len(cols))
        for f in np.unique(folds):
            te = folds == f
            fit = _ridge_fit(M[~te], T[~te], 1.0)
            base = self._fold_skill(_ridge_pred(fit, M[te]), yv[te], task, n_classes)
            for i in range(len(cols)):
                Zt = M[te].copy()
                idx = np.flatnonzero(owner == i)
                Zt[:, idx] = Zt[rng.permutation(len(Zt))][:, idx]
                drop[i] += base - self._fold_skill(_ridge_pred(fit, Zt), yv[te], task, n_classes)
        return {c: float(d / len(np.unique(folds))) for c, d in zip(cols, drop)}

    @staticmethod
    def _fold_skill(P, y, task, n_classes):
        return _skill(task, y, P.argmax(1) if task == "classification" else P[:, 0], n_classes)

    def _rank(self, fr, yv, task, cols):
        n_classes = int(yv.max()) + 1 if task == "classification" else 0
        fns = {"mutual_info": lambda: self._mutual_info(fr, yv, task, cols),
               "anova": lambda: self._anova(fr, yv, task, cols),
               "chi2": lambda: self._chi2(fr, yv, task, cols),
               "rfe": lambda: self._rfe(fr, yv, task, cols, n_classes),
               "model_importance": lambda: self._model_importance(fr, yv, task, cols, n_classes)}
        scores = {m: fns[m]() for m in self.methods}
        ranks = pd.DataFrame({m: pd.Series(s).rank(pct=True) for m, s in scores.items()})
        agg = ranks.mean(axis=1)
        rows = [{"feature": n, "aggregate": float(agg[n]), **{m: float(ranks.loc[n, m]) for m in self.methods},
                 **{f"raw_{m}": float(scores[m][n]) for m in self.methods}} for n in cols]
        rows.sort(key=lambda r: -r["aggregate"])
        return rows, {m: {k: float(v) for k, v in s.items()} for m, s in scores.items()}

    @staticmethod
    def _recommend(d):
        recs = []
        if d["hazards"]:
            recs.append("Remove constant/quasi-constant/low-variance columns: " + ", ".join(d["hazards"]) + ".")
        hc = d["hazard_detail"]["high_cardinality"]
        if hc:
            recs.append("High-cardinality categoricals can memorise rows; encode or drop: " + ", ".join(hc) + ".")
        if d["correlation_dropped"]:
            recs.append("Near-duplicate columns (pairwise correlation): " + ", ".join(d["correlation_dropped"]) + ".")
        if d["vif_dropped"]:
            recs.append("VIF-pruned for multicollinearity: " + ", ".join(d["vif_dropped"]) + ".")
        if d["supervised"]:
            recs.append(f"Keep {len(d['selected_features'])} of {d['n_features_in']} features by aggregated "
                        f"{'/'.join(d['methods'])} rank; confirm gains on held-out data.")
        else:
            recs.append("No target given: only unsupervised pruning ran; pass y for relevance ranking.")
        return recs


# ============================================================= outlier and structure detectors
# Pure NumPy: isolation forest, local outlier factor, self-organising map, robust Mahalanobis.
_EULER = 0.5772156649015329


def _c_factor(n):
    """Average path length of an unsuccessful BST search over n points (Liu et al., 2008)."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + _EULER) - 2.0 * (n[big] - 1.0) / n[big]
    out[n == 2] = 1.0
    return out


def isolation_forest_scores(Z, n_trees=100, sample_size=256, random_state=0):
    """Anomaly score in (0, 1) per row: ``2 ** (-E[h(x)] / c(psi))``. Scores near 1 are easy to isolate."""
    Z = np.asarray(Z, dtype=float)
    n, p = Z.shape
    psi = int(min(sample_size, n))
    height_limit = int(np.ceil(np.log2(max(psi, 2))))
    rng = np.random.default_rng(random_state)
    total = np.zeros(n)
    rows = np.arange(n)
    for _ in range(n_trees):
        sub = Z[rng.choice(n, psi, replace=False)]
        feat, thr, left, right, size = [], [], [], [], []

        def new_node():
            feat.append(-1), thr.append(0.0), left.append(-1), right.append(-1), size.append(0)
            return len(feat) - 1

        stack = [(np.arange(psi), 0, new_node())]
        while stack:
            idx, depth, node = stack.pop()
            size[node] = len(idx)
            if depth >= height_limit or len(idx) <= 1:
                continue
            part = sub[idx]
            lo, hi = part.min(0), part.max(0)
            live = np.flatnonzero(hi > lo)
            if live.size == 0:
                continue
            q = int(rng.choice(live))
            cut = rng.uniform(lo[q], hi[q])
            go_left = part[:, q] < cut
            feat[node], thr[node] = q, cut
            left[node], right[node] = new_node(), new_node()
            stack.append((idx[go_left], depth + 1, left[node]))
            stack.append((idx[~go_left], depth + 1, right[node]))
        feat_a, thr_a = np.array(feat), np.array(thr)
        left_a, right_a, size_a = np.array(left), np.array(right), np.array(size)
        node = np.zeros(n, dtype=int)
        depth = np.zeros(n)
        for _ in range(height_limit + 1):
            f = feat_a[node]
            internal = f >= 0
            if not internal.any():
                break
            x = Z[rows, np.where(internal, f, 0)]
            nxt = np.where(x < thr_a[node], left_a[node], right_a[node])
            node = np.where(internal, nxt, node)
            depth += internal
        total += depth + _c_factor(size_a[node])
    return 2.0 ** (-(total / n_trees) / max(float(_c_factor(np.array([psi]))[0]), 1e-12))


def local_outlier_factor(Z, k=20):
    """LOF (Breunig et al., 2000). Values near 1 are inliers; clearly above 1 are locally sparse points."""
    Z = np.asarray(Z, dtype=float)
    n = len(Z)
    k = int(max(1, min(k, n - 1)))
    sq = (Z ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0))
    np.fill_diagonal(D, np.inf)
    nbr = np.argpartition(D, k - 1, axis=1)[:, :k]
    dist = np.take_along_axis(D, nbr, 1)
    kdist = dist.max(1)
    reach = np.maximum(kdist[nbr], dist)
    lrd = 1.0 / (reach.mean(1) + 1e-12)
    return lrd[nbr].mean(1) / lrd


def robust_mahalanobis(Z, quantile=0.999):
    """Squared Mahalanobis distance on median/MAD-scaled data, with its chi-square cut-off."""
    Z = np.asarray(Z, dtype=float)
    med = np.median(Z, 0)
    mad = np.median(np.abs(Z - med), 0) * 1.4826
    Zr = (Z - med) / np.where(mad > 1e-12, mad, 1.0)
    cov = np.cov(Zr, rowvar=False).reshape(Z.shape[1], Z.shape[1]) + 1e-6 * np.eye(Z.shape[1])
    d2 = np.einsum("ij,jk,ik->i", Zr, np.linalg.inv(cov), Zr)
    return d2, float(stats.chi2.ppf(quantile, Z.shape[1])), Zr


class SelfOrganisingMap:
    """Batch self-organising map (Kohonen) on a rectangular grid, initialised on the PCA plane.

    Update: ``w_u <- sum_i h(u, b_i) x_i / sum_i h(u, b_i)`` with ``h = exp(-||r_u - r_b||^2 / (2 sigma_t^2))``
    and ``sigma_t`` shrinking from half the grid to 0.7 cells.
    """

    def __init__(self, grid=None, epochs=40, random_state=0):
        self.grid = grid
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, Z):
        Z = np.asarray(Z, dtype=float)
        n, p = Z.shape
        rows, cols = self.grid or (lambda m: (m, m))(int(np.clip(np.ceil(np.sqrt(5 * np.sqrt(n))), 4, 12)))
        self.shape_ = (rows, cols)
        coords = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=float)
        self.coords_ = coords
        mean = Z.mean(0)
        _, S, Vt = np.linalg.svd(Z - mean, full_matrices=False)
        sd = S / np.sqrt(max(n - 1, 1))
        a = np.linspace(-2, 2, rows)[:, None, None] * sd[0] * Vt[0]
        b = np.linspace(-2, 2, cols)[None, :, None] * (sd[1] if len(sd) > 1 else 0.0) * (Vt[1] if len(Vt) > 1 else 0.0)
        W = (mean + a + b).reshape(rows * cols, p)
        G = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        s0, s1 = max(rows, cols) / 2.0, 0.7
        for e in range(self.epochs):
            sigma = s0 * (s1 / s0) ** (e / max(self.epochs - 1, 1))
            bmu = self._bmu(Z, W)
            onehot = np.zeros((n, len(W)))
            onehot[np.arange(n), bmu] = 1.0
            H = np.exp(-G / (2.0 * sigma ** 2))
            den = H @ onehot.sum(0)
            W = np.where(den[:, None] > 1e-12, (H @ (onehot.T @ Z)) / np.maximum(den[:, None], 1e-12), W)
        self.weights_ = W
        self.unit_distance_ = G
        d = self._dist(Z, W)
        order = np.argsort(d, 1)
        self.bmu_ = order[:, 0]
        self.quantization_error_row_ = d[np.arange(n), self.bmu_]
        self.topographic_error_ = float((G[order[:, 0], order[:, 1]] > 2.0).mean()) if W.shape[0] > 1 else 0.0
        self.hits_ = np.bincount(self.bmu_, minlength=len(W)).reshape(rows, cols)
        return self

    @staticmethod
    def _dist(Z, W):
        return np.sqrt(np.maximum((Z ** 2).sum(1)[:, None] + (W ** 2).sum(1)[None, :] - 2.0 * Z @ W.T, 0.0))

    def _bmu(self, Z, W):
        return self._dist(Z, W).argmin(1)

    @property
    def quantization_error_(self):
        return float(self.quantization_error_row_.mean())

    def u_matrix(self):
        """Mean weight-space distance from each unit to its grid neighbours (high = cluster border)."""
        rows, cols = self.shape_
        W = self.weights_.reshape(rows, cols, -1)
        U = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                ds = [np.linalg.norm(W[r, c] - W[rr, cc]) for rr, cc in
                      ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if 0 <= rr < rows and 0 <= cc < cols]
                U[r, c] = np.mean(ds)
        return U

    def component_planes(self):
        rows, cols = self.shape_
        return self.weights_.reshape(rows, cols, -1)

    def unit_quantization_error(self):
        rows, cols = self.shape_
        err = np.zeros(rows * cols)
        counts = np.bincount(self.bmu_, minlength=rows * cols)
        np.add.at(err, self.bmu_, self.quantization_error_row_)
        return np.where(counts > 0, err / np.maximum(counts, 1), np.nan).reshape(rows, cols)


def _robust_flag(score, cut=3.5):
    med = np.median(score)
    mad = np.median(np.abs(score - med)) * 1.4826
    if mad <= 1e-12:
        return np.zeros(len(score), dtype=bool)
    return (score - med) / mad > cut


# ========================================================================= orchestrator
class DataIntelligenceReport:
    """Standardised report: ``summary_score`` (0-100) plus a details dictionary."""

    def __init__(self, summary_score: float, details: dict[str, Any]):
        self.summary_score = summary_score
        self.details = details

    def __getitem__(self, key):
        return self.details[key]

    def __getattr__(self, name):
        details = self.__dict__.get("details", {})
        if name in details:
            return details[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps({"summary_score": self.summary_score, **self.details}, default=_json_default))

    def to_json(self, path=None, indent=2):
        text = json.dumps({"summary_score": self.summary_score, **self.details}, indent=indent, default=_json_default)
        if path is not None:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text

    def export(self, directory, formats=("json", "txt", "html"), name="data_intelligence", figure_paths=None):
        """Save the report to ``directory`` as json / txt / html. Returns ``{format: path}``."""
        figure_paths = figure_paths or self.details.get("visualizations")
        return export_report(self, directory, name=name, formats=formats, text=self.to_text(),
                             title="ReliableML Data Intelligence report", figure_paths=figure_paths)

    def to_text(self) -> str:
        d = self.details
        out = [f"DATA INTELLIGENCE REPORT  (mode={d['mode']}, score={self.summary_score:.0f}/100)", "=" * 60]
        ov = d["overview"]
        out.append(f"rows={ov['rows']}  columns={ov['columns']}  missing cells={ov['missing_cells']}  "
                   f"target={ov['target_column']}  time={ov['time_column']}")
        if d.get("narrative"):
            out += ["", "Narrative", d["narrative"]]
        if d.get("question_plan"):
            out += ["", "Questions the data raises (highest priority first)"]
            for i, q in enumerate(d["question_plan"]):
                out += [f"  {i + 1}. [{q['priority']}/{q['category']}] {q['question']}", f"       next: {q['next_step']}"]
        elif d.get("questions"):
            out += ["", "Questions the data raises"] + [f"  {i + 1}. {q}" for i, q in enumerate(d["questions"])]
        if d.get("target_intelligence"):
            out += ["", d["target_intelligence_text"]]
        if d.get("feature_intelligence"):
            out += ["", d["feature_intelligence_text"]]
        out += ["", "Recommendations"] + [f"  - {r}" for r in d["recommendations"]]
        return "\n".join(out)

    def __repr__(self):
        return f"DataIntelligenceReport(mode={self.details['mode']!r}, summary_score={self.summary_score:.1f})"


# =================================================================== analysis toolkit
# Power analysis, group comparison, A/B testing, root cause, time series, feature construction and
# active learning. Every result is a plain dict with a "kind" and plain-language "findings".
def _binary01(s):
    s = pd.Series(s).dropna()
    if pd.api.types.is_bool_dtype(s):
        return True
    return bool(pd.api.types.is_numeric_dtype(s) and len(s) and set(s.unique()) <= {0, 1})


def _holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def _bh(p):
    p = np.asarray(p, float)
    m = len(p)
    adj, run = np.empty(m), 1.0
    for k, i in enumerate(np.argsort(p)[::-1]):
        run = min(run, p[i] * m / (m - k))
        adj[i] = min(1.0, run)
    return adj


def _cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1))
    if sp <= 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def _f_power(f2, p, n, alpha=0.05):
    df2 = n - p - 1
    if df2 < 1:
        return float("nan")
    return float(stats.ncf.sf(stats.f.isf(alpha, p, df2), p, df2, f2 * n))


def _required_n(f2, p, power=0.8, alpha=0.05):
    lo, hi = p + 3, p + 3
    while _f_power(f2, p, hi, alpha) < power:
        lo, hi = hi, hi * 2
        if hi > 10_000_000:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if _f_power(f2, p, mid, alpha) < power else (lo, mid)
    return int(hi)


def _min_detectable_f2(p, n, power=0.8, alpha=0.05):
    lo, hi = 1e-6, 20.0
    if not _f_power(hi, p, n, alpha) >= power:
        return float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _f_power(mid, p, n, alpha) < power else (lo, mid)
    return float(hi)


def _strength(fr, name, yy, task, k):
    """Univariate (strength in [0,1], p-value) of one feature against the target."""
    n, v = fr.n, fr.values[name]
    if fr.kind[name] == "numeric" and task == "regression":
        if np.std(v) < 1e-12 or np.std(yy) < 1e-12:
            return 0.0, 1.0
        r = _corr(v, yy)
        return r * r, float(stats.pearsonr(v, yy)[1])
    if task == "regression":
        codes = fr.codes[name]
        g, e2 = codes.max() + 1, _eta(yy, fr.codes[name]) ** 2
    elif fr.kind[name] == "numeric":
        g, e2 = k, _eta(v, yy) ** 2
    else:
        tab = _contingency(fr.codes[name], yy)
        if min(tab.shape) < 2:
            return 0.0, 1.0
        chi2, p, _, _ = stats.chi2_contingency(tab)
        return float(chi2 / (tab.sum() * (min(tab.shape) - 1))), float(p)
    if g < 2 or n - g < 1 or e2 >= 1:
        return float(e2), 0.0 if e2 >= 1 else 1.0
    return float(e2), float(stats.f.sf((e2 / (g - 1)) / ((1 - e2) / (n - g)), g - 1, n - g))


def _ols_r2(Z, y):
    A = np.column_stack([np.ones(len(y)), Z])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    sst = float(((y - y.mean()) ** 2).sum())
    return 0.0 if sst <= 1e-12 else max(0.0, 1 - float(((y - A @ beta) ** 2).sum()) / sst)


def power_analysis(X, y, task="auto", n_folds=5, alpha=0.05, power=0.8, random_state=0):
    """Predictive power vs explanatory power, plus the statistical power of the data you have.

    *Predictive* power is out-of-fold skill (ridge and KNN). *Explanatory* power is in-sample: the share
    of the target the features account for, with F and per-feature tests. A large gap means the model
    story is partly memorisation of noise. *Statistical* power asks whether n is large enough to detect
    an effect of a given size (Cohen's f2) with the F test.
    """
    y = np.asarray(y)
    task = TargetIntelligence(task=task)._detect_task(y)
    fr = _Frame(X)
    if len(y) != fr.n:
        raise ValueError("y must have one value per row of X")
    yy, classes = TargetIntelligence._encode_target(y, task)
    k, n = (len(classes) if classes else 0), fr.n
    folds_n = n_folds
    if task == "classification":
        folds_n = max(2, min(n_folds, int(np.bincount(yy).min())))
    folds = _fold_ids(yy, task, folds_n, random_state)
    T = _targets_matrix(yy, task, k)
    names = fr.names
    Z = fr.matrix(names)
    ti = TargetIntelligence(random_state=random_state)
    pred_lin = float(_fast_skill(Z, yy, T, folds, task, k))
    pred_knn = float(ti._model_cv(Z, yy, task, k, "knn", folds)[0])
    stren = {c: _strength(fr, c, yy, task, k) for c in names}
    order = sorted(names, key=lambda c: -stren[c][0])
    keep, width = [], 0
    for c in order:
        w = fr.block(c).shape[1]
        if width + w > max(1, n // 5):
            continue
        keep.append(c)
        width += w
    Zk = fr.matrix(keep)
    p = Zk.shape[1]
    fit = _ridge_fit(Zk, T, 1e-6)
    P = _ridge_pred(fit, Zk)
    pred_in = P.argmax(1) if task == "classification" else P[:, 0]
    in_skill = float(_skill(task, yy, pred_in, k))
    r2 = float(_ols_r2(Zk, yy)) if task == "regression" else float(max(0.0, in_skill))
    adj = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
    f_p = float("nan")
    if task == "regression" and n - p - 1 > 0 and r2 < 1:
        f_p = float(stats.f.sf((r2 / p) / ((1 - r2) / (n - p - 1)), p, n - p - 1))
    pvals = [stren[c][1] for c in names]
    adjp = _bh(pvals)
    per = []
    for c, ap in zip(names, adjp):
        row = {"feature": c, "explanatory_strength": float(stren[c][0]), "p_value": float(stren[c][1]),
               "p_adjusted_bh": float(ap), "significant": bool(ap < alpha)}
        if task == "regression" and c in keep and len(keep) > 1:
            rest = fr.matrix([x for x in keep if x != c])
            row["partial_r2"] = float(max(0.0, r2 - _ols_r2(rest, yy)))
        per.append(row)
    per.sort(key=lambda r: -r["explanatory_strength"])
    gap = in_skill - pred_lin
    best_pred = max(pred_lin, pred_knn if np.isfinite(pred_knn) else -1)
    if best_pred < 0.05 and in_skill < 0.10:
        verdict = "weak on both: the features neither explain nor predict this target"
    elif best_pred < 0.05:
        verdict = "explains but does not predict: in-sample structure that does not generalise (overfitting risk)"
    elif gap > 0.15:
        verdict = "predicts, with a large explanatory-minus-predictive gap: expect optimism, validate carefully"
    elif pred_knn > pred_lin + 0.05:
        verdict = "predicts better non-linearly than linearly: a model beats a simple explanation"
    else:
        verdict = "explains and predicts consistently: the relationship is simple and stable"
    f2 = r2 / (1 - r2) if r2 < 1 else float("inf")
    stat = {"n": n, "predictors": p, "alpha": alpha, "target_power": power, "observed_f2": float(f2),
            "power_at_observed_effect": _f_power(f2, p, n, alpha) if np.isfinite(f2) else 1.0,
            "min_detectable_f2": _min_detectable_f2(p, n, power, alpha),
            "required_n": {name: _required_n(v, p, power, alpha) for name, v in
                           (("small_f2_0.02", 0.02), ("medium_f2_0.15", 0.15), ("large_f2_0.35", 0.35))}}
    findings = [f"Predictive skill (cross-validated): ridge {pred_lin:.3f}, KNN {pred_knn:.3f}.",
                f"Explanatory skill (in-sample, {p} columns): {in_skill:.3f}; adjusted R2 {adj:.3f}.",
                f"Verdict: {verdict}.",
                f"With n={n} and {p} predictors you can reliably detect f2 >= {stat['min_detectable_f2']:.3f} "
                f"(power {power:.0%}); a medium effect needs n >= {stat['required_n']['medium_f2_0.15']}."]
    return {"kind": "power", "task": task, "predictive": {"ridge_cv_skill": pred_lin, "knn_cv_skill": pred_knn},
            "explanatory": {"in_sample_skill": in_skill, "r_squared": r2, "adjusted_r_squared": float(adj),
                            "f_test_p_value": f_p, "columns_used": keep, "features": per},
            "gap_explanatory_minus_predictive": float(gap), "verdict": verdict, "statistical_power": stat,
            "findings": findings}


def compare_groups(df, value, group, alpha=0.05, min_group=3):
    """Group-by comparison with the right hypothesis test, effect sizes and Holm-corrected pairwise tests."""
    d = df[[value, group]].dropna()
    g = d[group].astype(str)
    if pd.api.types.is_numeric_dtype(d[group]) and d[group].nunique() > 12:
        raise ValueError(f"'{group}' has too many distinct numeric values to act as a group; bin it first")
    if not pd.api.types.is_numeric_dtype(d[value]) or pd.api.types.is_bool_dtype(d[value]):
        tab = pd.crosstab(g, d[value].astype(str))
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        v = float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))) if min(tab.shape) > 1 else 0.0
        return {"kind": "group_comparison", "value": value, "group": group, "test": "chi-square",
                "omnibus": {"statistic": float(chi2), "p_value": float(p), "dof": int(dof), "cramers_v": v},
                "table": tab.to_dict(), "significant": bool(p < alpha),
                "findings": [f"{value} is {'' if p < alpha else 'not '}associated with {group} "
                             f"(chi-square p={p:.3g}, Cramer's V={v:.2f})."]}
    v = d[value].astype(float)
    groups = {k: x.to_numpy() for k, x in v.groupby(g) if len(x) >= min_group}
    if len(groups) < 2:
        raise ValueError("need at least two groups with enough rows")
    rows = []
    for k, x in groups.items():
        se = x.std(ddof=1) / np.sqrt(len(x))
        t = stats.t.ppf(0.975, len(x) - 1)
        rows.append({"group": k, "n": len(x), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
                     "median": float(np.median(x)), "ci95_low": float(x.mean() - t * se),
                     "ci95_high": float(x.mean() + t * se)})
    normal = True
    for x in groups.values():
        if len(x) >= 8 and x.std() > 0 and stats.shapiro(x[:5000])[1] < 0.05:
            normal = False
    names = list(groups)
    out = {"kind": "group_comparison", "value": value, "group": group, "table": rows,
           "normality_ok": normal, "alpha": alpha}
    if len(groups) == 2:
        a, b = groups[names[0]], groups[names[1]]
        welch = stats.ttest_ind(a, b, equal_var=False)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        out["omnibus"] = {"welch_t": {"statistic": float(welch.statistic), "p_value": float(welch.pvalue)},
                          "mann_whitney": {"statistic": float(mw.statistic), "p_value": float(mw.pvalue)},
                          "cohens_d": _cohen_d(a, b)}
        out["test"] = "welch-t" if normal else "mann-whitney"
        p = welch.pvalue if normal else mw.pvalue
    else:
        an, kw = stats.f_oneway(*groups.values()), stats.kruskal(*groups.values())
        allv = np.concatenate(list(groups.values()))
        ssb = sum(len(x) * (x.mean() - allv.mean()) ** 2 for x in groups.values())
        eta2 = float(ssb / ((allv - allv.mean()) ** 2).sum()) if allv.std() > 0 else 0.0
        out["omnibus"] = {"anova": {"statistic": float(an.statistic), "p_value": float(an.pvalue)},
                          "kruskal": {"statistic": float(kw.statistic), "p_value": float(kw.pvalue)},
                          "eta_squared": eta2}
        out["test"] = "anova" if normal else "kruskal"
        p = an.pvalue if normal else kw.pvalue
        pairs = list(combinations(names, 2))
        raw = [stats.ttest_ind(groups[a], groups[b], equal_var=False).pvalue for a, b in pairs]
        adj = _holm(raw)
        out["pairwise"] = [{"a": a, "b": b, "mean_difference": float(groups[a].mean() - groups[b].mean()),
                            "p_value": float(r), "p_holm": float(h), "cohens_d": _cohen_d(groups[a], groups[b]),
                            "significant": bool(h < alpha)} for (a, b), r, h in zip(pairs, raw, adj)]
    out["p_value"], out["significant"] = float(p), bool(p < alpha)
    top = max(rows, key=lambda r: r["mean"])
    out["findings"] = [f"{out['test']} on {value} by {group}: p={p:.3g} "
                       f"({'a real difference' if p < alpha else 'no detectable difference'}).",
                       f"Highest mean: {top['group']} ({top['mean']:.3g}, n={top['n']})."]
    return out


def scan_groups(df, target, alpha=0.05, max_levels=12, top=10):
    """Why is the target different? Test every categorical column against it, BH-corrected."""
    rows = []
    for c in df.columns:
        s = df[c]
        if c == target or not 2 <= s.nunique() <= max_levels:
            continue
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
            continue
        try:
            r = compare_groups(df, target, c, alpha)
        except Exception:
            continue
        om = r["omnibus"]
        effect = om.get("cramers_v", abs(om.get("cohens_d", om.get("eta_squared", 0.0))))
        p = r.get("p_value", om.get("p_value", 1.0))
        rows.append({"column": c, "test": r["test"], "effect": float(effect), "p_value": float(p)})
    if rows:
        for row, a in zip(rows, _bh([r["p_value"] for r in rows])):
            row["p_adjusted_bh"], row["significant"] = float(a), bool(a < alpha)
    rows.sort(key=lambda r: r["p_adjusted_bh"])
    sig = [r["column"] for r in rows if r["significant"]]
    return {"kind": "group_scan", "target": target, "columns": rows[:top],
            "findings": [f"{len(sig)} of {len(rows)} grouping columns separate '{target}': {sig[:6]}."]}


def ab_test(df, group, outcome, control=None, treatment=None, alpha=0.05, power=0.8):
    """Compare two variants (e.g. which page produces more leads). Binary outcomes use z, chi-square and
    Fisher; numeric outcomes use Welch and Mann-Whitney. Includes sample-ratio and sample-size checks."""
    d = df[[group, outcome]].dropna()
    levels = list(pd.unique(d[group]))
    if control is None or treatment is None:
        if len(levels) != 2:
            raise ValueError("pass control= and treatment= when the group column has more than two levels")
        control, treatment = sorted(levels, key=str)
    a = d.loc[d[group] == control, outcome]
    b = d.loc[d[group] == treatment, outcome]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each variant needs at least 2 rows")
    srm = float(stats.chisquare([len(a), len(b)])[1])
    out = {"kind": "ab_test", "control": str(control), "treatment": str(treatment), "n_control": len(a),
           "n_treatment": len(b), "sample_ratio_p": srm, "alpha": alpha, "warnings": []}
    if srm < 0.001:
        out["warnings"].append("sample-ratio mismatch (arms are unevenly sized): check assignment before trusting results")
    za, zb = stats.norm.isf(alpha / 2), stats.norm.isf(1 - power)
    if _binary01(d[outcome]):
        xa, xb = float(a.astype(float).sum()), float(b.astype(float).sum())
        pa, pb = xa / len(a), xb / len(b)
        pool = (xa + xb) / (len(a) + len(b))
        se0 = np.sqrt(pool * (1 - pool) * (1 / len(a) + 1 / len(b)))
        z = (pb - pa) / se0 if se0 > 0 else 0.0
        p_z = float(2 * stats.norm.sf(abs(z)))
        se1 = np.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))
        tab = np.array([[xa, len(a) - xa], [xb, len(b) - xb]])
        exp_min = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / tab.sum()
        p_fisher = float(stats.fisher_exact(tab)[1])
        p = p_fisher if exp_min.min() < 5 else p_z
        n_arm = min(len(a), len(b))
        mde = (za + zb) * np.sqrt(2 * max(pa * (1 - pa), 1e-9) / n_arm)
        need = None
        if abs(pb - pa) > 1e-9:
            need = int(np.ceil((za * np.sqrt(2 * pool * (1 - pool)) + zb * np.sqrt(pa * (1 - pa) + pb * (1 - pb))) ** 2
                               / (pb - pa) ** 2))
        out.update({"metric": "conversion rate", "rate_control": pa, "rate_treatment": pb,
                    "absolute_lift": pb - pa, "relative_lift": (pb - pa) / pa if pa > 0 else float("nan"),
                    "ci95_difference": [float(pb - pa - 1.96 * se1), float(pb - pa + 1.96 * se1)],
                    "p_value": p, "p_z": p_z, "p_fisher": p_fisher,
                    "p_chi2": float(stats.chi2_contingency(tab)[1]), "test": "fisher" if exp_min.min() < 5 else "two-proportion z",
                    "min_detectable_absolute_lift": float(mde), "n_per_arm_for_observed_lift": need})
    else:
        av, bv = a.astype(float).to_numpy(), b.astype(float).to_numpy()
        w = stats.ttest_ind(bv, av, equal_var=False)
        diff = float(bv.mean() - av.mean())
        se = float(np.sqrt(av.var(ddof=1) / len(av) + bv.var(ddof=1) / len(bv)))
        sd = float(np.sqrt((av.var(ddof=1) + bv.var(ddof=1)) / 2))
        n_arm = min(len(a), len(b))
        out.update({"metric": f"mean {outcome}", "mean_control": float(av.mean()), "mean_treatment": float(bv.mean()),
                    "absolute_lift": diff, "relative_lift": diff / av.mean() if av.mean() else float("nan"),
                    "ci95_difference": [diff - 1.96 * se, diff + 1.96 * se], "p_value": float(w.pvalue),
                    "test": "welch-t", "p_mann_whitney": float(stats.mannwhitneyu(av, bv).pvalue),
                    "cohens_d": _cohen_d(bv, av), "min_detectable_absolute_lift": float((za + zb) * sd * np.sqrt(2 / n_arm)),
                    "n_per_arm_for_observed_lift": int(np.ceil(2 * ((za + zb) * sd / diff) ** 2)) if abs(diff) > 1e-12 else None})
        p = float(w.pvalue)
    out["significant"] = bool(p < alpha)
    if p < alpha:
        out["decision"] = f"'{treatment}' is {'better' if out['absolute_lift'] > 0 else 'worse'} than '{control}'"
    else:
        out["decision"] = "no detectable difference: keep collecting data or accept the variants are equal"
    out["warnings"].append("fix the sample size in advance; repeatedly checking p-values and stopping early inflates false positives")
    out["findings"] = [f"{out['metric']}: {control} vs {treatment}, lift {out['absolute_lift']:+.4g} "
                       f"(95% CI {out['ci95_difference'][0]:+.3g} to {out['ci95_difference'][1]:+.3g}), p={p:.3g}.",
                       out["decision"] + ".",
                       f"At n={n_arm} per arm the smallest reliably detectable lift is {out['min_detectable_absolute_lift']:.3g}."]
    return out


def root_cause(df, outcome, bad=None, direction="high", features=None, max_depth=2, top=10, alpha=0.05, max_levels=8):
    """Which features and segments concentrate the bad outcome? Associational, not causal."""
    y = df[outcome]
    if bad is not None:
        isbad = (y.astype(str) == str(bad)).to_numpy()
        rule = f"{outcome} == {bad}"
    elif _binary01(y):
        isbad, rule = (y == 1).to_numpy(), f"{outcome} == 1"
    elif pd.api.types.is_numeric_dtype(y):
        q = float(np.nanquantile(y, 0.75 if direction == "high" else 0.25))
        isbad = (y >= q).to_numpy() if direction == "high" else (y <= q).to_numpy()
        rule = f"{outcome} {'>=' if direction == 'high' else '<='} {q:.4g}"
    else:
        raise ValueError("pass bad= for a categorical outcome")
    n, p0 = len(df), float(isbad.mean())
    if not 0 < p0 < 1:
        raise ValueError("the bad outcome must occur in some, but not all, rows")
    feats = [c for c in (features or df.columns) if c != outcome]
    contrasts, conds = [], []
    for c in feats:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique() > max_levels:
            v = s.to_numpy(float)
            ok = np.isfinite(v)
            if isbad[ok].sum() > 2 and (~isbad[ok]).sum() > 2:
                u = stats.mannwhitneyu(v[ok][isbad[ok]], v[ok][~isbad[ok]])
                contrasts.append({"feature": c, "auc_bad_vs_good": float(u.statistic / (isbad[ok].sum() * (~isbad[ok]).sum())),
                                  "p_value": float(u.pvalue), "median_bad": float(np.median(v[ok][isbad[ok]])),
                                  "median_good": float(np.median(v[ok][~isbad[ok]]))})
            edges = np.unique(np.nanquantile(v, [1 / 3, 2 / 3]))
            bins = np.searchsorted(edges, v, side="left")
            for j in range(len(edges) + 1):
                conds.append((c, f"{c} in tercile {j + 1}/{len(edges) + 1}", (bins == j) & ok))
        else:
            lv = s.astype(str).value_counts().index[:max_levels]
            for val in lv:
                conds.append((c, f"{c} == {val}", (s.astype(str) == val).to_numpy()))
    min_support = max(20, int(0.02 * n))
    segs = []

    def score(label, mask, cols):
        cnt = int(mask.sum())
        if cnt < min_support:
            return
        kk = int(isbad[mask].sum())
        segs.append({"segment": label, "columns": cols, "support": cnt, "bad_rate": kk / cnt, "lift": kk / cnt / p0,
                     "excess_bad_rows": float(kk - p0 * cnt),
                     "p_value": float(stats.binomtest(kk, cnt, p0, alternative="greater").pvalue)})

    for c, label, m in conds:
        score(label, m, [c])
    if max_depth >= 2:
        best = sorted([s for s in segs], key=lambda s: -s["lift"])[:25]
        lookup = {label: (c, m) for c, label, m in conds}
        for s1, s2 in combinations(best, 2):
            (c1, m1), (c2, m2) = lookup[s1["segment"]], lookup[s2["segment"]]
            if c1 != c2:
                score(f"{s1['segment']} AND {s2['segment']}", m1 & m2, [c1, c2])
    if segs:
        for s, a in zip(segs, _bh([s["p_value"] for s in segs])):
            s["p_adjusted_bh"], s["significant"] = float(a), bool(a < alpha)
    good = sorted([s for s in segs if s["significant"] and s["lift"] > 1], key=lambda s: -s["excess_bad_rows"])[:top]
    contrasts.sort(key=lambda r: -abs(r["auc_bad_vs_good"] - 0.5))
    findings = [f"Bad outcome: {rule} ({p0:.1%} of rows).",
                (f"Strongest segment: {good[0]['segment']} (bad rate {good[0]['bad_rate']:.1%}, lift {good[0]['lift']:.1f}x, "
                 f"n={good[0]['support']}).") if good else "No segment concentrates the bad outcome beyond chance.",
                "These are associations; confirm a cause with an experiment or domain knowledge."]
    return {"kind": "root_cause", "outcome_rule": rule, "base_rate": p0, "segments": good,
            "feature_contrasts": contrasts[:top], "findings": findings}


def _mann_kendall(x, cap=2000):
    x = np.asarray(x, float)
    spacing = 1.0
    if len(x) > cap:
        spacing = (len(x) - 1) / (cap - 1)
        x = x[np.linspace(0, len(x) - 1, cap).astype(int)]
    n = len(x)
    s = sum(float(np.sign(x[i + 1:] - x[i]).sum()) for i in range(n - 1))
    _, cnt = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - float((cnt * (cnt - 1) * (2 * cnt + 5)).sum())) / 18
    z = 0.0 if s == 0 or var <= 0 else (s - np.sign(s)) / np.sqrt(var)
    iu = np.triu_indices(n, 1)
    slopes = np.subtract.outer(x, x).T[iu] / np.subtract.outer(np.arange(n), np.arange(n)).T[iu]
    return float(z), float(2 * stats.norm.sf(abs(z))), float(np.median(slopes) / spacing)


def time_series_analysis(df, time_column, value_column, period=None, alpha=0.05, random_state=0):
    """Trend (Mann-Kendall, Sen slope), seasonality (periodogram + Fisher g), decomposition, stationarity
    (ADF/KPSS), residual autocorrelation (Ljung-Box) and a CUSUM change point."""
    t = pd.to_datetime(df[time_column], errors="coerce")
    v = pd.to_numeric(df[value_column], errors="coerce")
    ok = t.notna() & v.notna()
    if ok.sum() < 24:
        raise ValueError("need at least 24 valid observations for time-series analysis")
    order = np.argsort(t[ok].to_numpy())
    x = v[ok].to_numpy(float)[order]
    times = t[ok].to_numpy()[order]
    n = len(x)
    z, p_mk, sen_per_step = _mann_kendall(x)
    idx = np.arange(n)
    lin = np.polyfit(idx, x, 1)
    detr = x - np.polyval(lin, idx)
    out = {"kind": "time_series", "value": value_column, "n": n, "alpha": alpha,
           "trend": {"mann_kendall_z": z, "p_value": p_mk, "sen_slope_per_step": float(sen_per_step),
                     "direction": "increasing" if z > 0 else "decreasing", "significant": bool(p_mk < alpha)}}
    spec = np.abs(np.fft.rfft(detr - detr.mean())[1:]) ** 2
    periods = n / np.arange(1, len(spec) + 1)
    valid = (periods >= 2) & (periods <= n / 2)
    found = None
    if valid.any() and spec.sum() > 0:
        i = int(np.argmax(np.where(valid, spec, -1)))
        g = float(spec[i] / spec.sum())
        m = len(spec)
        p_g = float(min(1.0, m * (1 - g) ** (m - 1)))
        found = int(round(periods[i]))
        out["seasonality"] = {"dominant_period": found, "fisher_g": g, "p_value": p_g,
                              "significant": bool(p_g < alpha)}
    per = period or (found if out.get("seasonality", {}).get("significant") else None)
    if per and per >= 2 and n >= 2 * per:
        k = int(per)
        kernel = np.ones(k) / k if k % 2 else np.r_[0.5, np.ones(k - 1), 0.5] / k
        trend = np.convolve(x, kernel, mode="same")
        half = len(kernel) // 2
        trend[:half] = trend[half]
        trend[-half:] = trend[-half - 1]
        phase = idx % k
        seas_means = np.array([(x - trend)[phase == j].mean() for j in range(k)])
        seas = (seas_means - seas_means.mean())[phase]
        resid = x - trend - seas
        out["decomposition"] = {"period": k, "seasonal_strength": float(max(0, 1 - resid.var() / (seas + resid).var())),
                                "trend_strength": float(max(0, 1 - resid.var() / (trend - trend.mean() + resid).var()))}
    else:
        trend, seas = np.polyval(lin, idx), np.zeros(n)
        resid = detr
    keep = np.linspace(0, n - 1, min(n, 1500)).astype(int)
    out["series"] = {"time": [str(pd.Timestamp(q)) for q in times[keep]], "value": x[keep].tolist(),
                     "trend": trend[keep].tolist(), "seasonal": seas[keep].tolist(), "residual": resid[keep].tolist()}
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        from statsmodels.stats.diagnostic import acorr_ljungbox
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = adfuller(x, autolag="AIC")
            kp = kpss(x, regression="c", nlags="auto")
            lb = acorr_ljungbox(resid, lags=[min(10, n // 5)])
        out["stationarity"] = {"adf_p": float(adf[1]), "kpss_p": float(kp[1]),
                               "stationary": bool(adf[1] < alpha and kp[1] >= alpha)}
        out["residual_autocorrelation"] = {"ljung_box_p": float(lb["lb_pvalue"].iloc[0]),
                                           "autocorrelated": bool(lb["lb_pvalue"].iloc[0] < alpha)}
    except Exception as exc:
        out["stationarity"] = {"error": str(exc)}
    cs = np.cumsum(x - x.mean())
    kcp = int(np.argmax(np.abs(cs)))
    rng = np.random.default_rng(random_state)
    stat = np.abs(cs).max()
    perm = [np.abs(np.cumsum(rng.permutation(x) - x.mean())).max() for _ in range(300)]
    p_cp = float((1 + sum(q >= stat for q in perm)) / 301)
    out["change_point"] = {"index": kcp, "time": str(pd.Timestamp(times[kcp])), "p_value": p_cp,
                           "significant": bool(p_cp < alpha),
                           "note": "permutation test ignores autocorrelation: treat as a lead, not proof"}
    f = [f"Trend: {out['trend']['direction']}, {'significant' if p_mk < alpha else 'not significant'} (Mann-Kendall p={p_mk:.3g})."]
    if "seasonality" in out:
        s = out["seasonality"]
        f.append(f"Seasonality: dominant period {s['dominant_period']} steps ({'significant' if s['significant'] else 'not significant'}, p={s['p_value']:.3g}).")
    if "stationary" in out.get("stationarity", {}):
        f.append("Series is " + ("stationary." if out["stationarity"]["stationary"] else "not stationary: difference or detrend before modelling."))
    if out["change_point"]["significant"]:
        f.append(f"A level shift is likely near {out['change_point']['time']}.")
    out["findings"] = f
    return out


# ----------------------------------------------------------------- feature construction
def _skew(v):
    sd = v.std()
    return 0.0 if sd < 1e-12 else float(((v - v.mean()) ** 3).mean() / sd ** 3)


def _valley_threshold(v, rng):
    """Lowest-density point between the quartiles when it is clearly below the neighbouring modes."""
    s = v if len(v) <= 2000 else rng.choice(v, 2000, replace=False)
    if np.std(s) < 1e-12:
        return None
    try:
        kde = stats.gaussian_kde(s)
    except Exception:
        return None
    grid = np.linspace(*np.quantile(s, [0.1, 0.9]), 80)
    dens = kde(grid)
    i = int(np.argmin(dens[10:-10])) + 10
    if dens[i] < 0.6 * min(dens[:i].max(), dens[i:].max()):
        return float(grid[i])
    return None


class FeatureConstructor:
    """Iterative, sample-distribution-aware feature construction.

    Candidates are generated from each column's empirical distribution (skew, tails, zeros, density
    valleys, missingness) and from pairs of the strongest columns. Each round every candidate is
    *screened* on one resampling, then the best few are *confirmed* across ``n_repeats`` fresh fold
    assignments. A candidate is accepted only if its paired gain is stable: ``mean - std > min_gain``.
    The accepted feature joins the pool, so the next round can build on it (up to ``max_depth``).
    All constructions are target-free (thresholds and ranks come from X); only selection uses y. A
    ``holdout`` fraction of rows is never touched by the search and measures the selection optimism.
    """

    def __init__(self, task="auto", max_steps=5, max_depth=2, n_repeats=5, n_folds=4, min_gain=0.002,
                 holdout=0.25, top_pairs=6, confirm_top=6, leakage_corr=0.95, random_state=0):
        self.task, self.max_steps, self.max_depth = task, max_steps, max_depth
        self.n_repeats, self.n_folds, self.min_gain, self.holdout = n_repeats, n_folds, min_gain, holdout
        self.top_pairs, self.confirm_top, self.leakage_corr, self.random_state = top_pairs, confirm_top, leakage_corr, random_state

    def _candidates(self, cols, depth, y, rng, have):
        out = []
        for name, v in cols.items():
            sk, mn = _skew(v), float(v.min())
            if sk > 1 and mn >= 0:
                out.append((f"log1p({name})", "log1p", [name], {}))
            elif abs(sk) > 1:
                out.append((f"signedlog({name})", "signedlog", [name], {}))
            if len(np.unique(v)) >= 10 and abs(stats.kurtosis(v)) > 5:
                out.append((f"rankgauss({name})", "rankgauss", [name], {"ref": np.sort(v).tolist()}))
            if len(np.unique(v)) >= 10:
                thr = _valley_threshold(v, rng)
                if thr is not None:
                    out.append((f"{name}>{thr:.4g}", "threshold", [name], {"t": thr}))
        ranked = sorted(cols, key=lambda c: -abs(_corr(cols[c], y)))[: self.top_pairs]
        for a, b in combinations(ranked, 2):
            va, vb = cols[a], cols[b]
            out += [(f"{a}*{b}", "product", [a, b], {}), (f"{a}-{b}", "diff", [a, b], {}),
                    (f"{a}+{b}", "sum", [a, b], {})]
            for p_, q_ in ((a, b), (b, a)):
                den = cols[q_]
                if (den > 0).all() or (den < 0).all():
                    out.append((f"{p_}/{q_}", "ratio", [p_, q_], {}))
        return [c for c in out if c[0] not in have and depth.get(c[2][0], 0) < self.max_depth
                and (len(c[2]) == 1 or depth.get(c[2][1], 0) < self.max_depth)]

    @staticmethod
    def _apply(op, args, params):
        a = args[0]
        if op == "log1p":
            return np.log1p(np.maximum(a - 0, 0))
        if op == "signedlog":
            return np.sign(a) * np.log1p(np.abs(a))
        if op == "rankgauss":
            ref = np.asarray(params["ref"])
            r = (np.searchsorted(ref, a, side="left") + np.searchsorted(ref, a, side="right")) / 2
            return stats.norm.ppf(np.clip((r + 0.5) / (len(ref) + 1), 1e-6, 1 - 1e-6))
        if op == "threshold":
            return (a > params["t"]).astype(float)
        b = args[1]
        return {"product": a * b, "diff": a - b, "sum": a + b, "ratio": a / np.where(np.abs(b) < 1e-12, 1e-12, b)}[op]

    def _gains(self, base, extra, yy, T, task, k, seeds, base_cache):
        res = []
        for i, s in enumerate(seeds):
            f = _fold_ids(yy, task, self.n_folds, s)
            if i not in base_cache:
                base_cache[i] = _fast_skill(base, yy, T, f, task, k)
            res.append(_fast_skill(np.column_stack([base, extra]), yy, T, f, task, k) - base_cache[i])
        return np.array(res)

    def fit(self, X, y):
        y = np.asarray(y)
        task = TargetIntelligence(task=self.task)._detect_task(y)
        fr = _Frame(X)
        if len(y) != fr.n:
            raise ValueError("y must have one value per row of X")
        yy, classes = TargetIntelligence._encode_target(y, task)
        k = len(classes) if classes else 0
        rng = np.random.default_rng(self.random_state)
        names = fr.names
        raw_df = X if isinstance(X, pd.DataFrame) else None
        cols = {c: np.asarray(fr.values[c], float) for c in fr.numeric()}
        cols = {c: v for c, v in cols.items() if abs(_corr(v, yy)) < self.leakage_corr}
        depth = {c: 0 for c in cols}
        notes = []
        if len(cols) < 1:
            raise ValueError("feature construction needs at least one numeric, non-leaky column")
        hold_fold = _fold_ids(yy, task, max(2, int(round(1 / self.holdout))), self.random_state)
        hold = hold_fold == 0
        dev = ~hold
        Ydev, Tdev = yy[dev], _targets_matrix(yy[dev], task, k)
        base = fr.matrix(names)[dev]
        seeds = [self.random_state + 11 * i for i in range(self.n_repeats)]
        steps, recipes, have, built = [], [], set(cols), {}
        base_now = base.copy()
        cache: dict = {}
        base_cv = float(_fast_skill(base, Ydev, Tdev, _fold_ids(Ydev, task, self.n_folds, seeds[0]), task, k))
        for _ in range(self.max_steps):
            cands = self._candidates({c: v[dev] for c, v in cols.items()}, depth, Ydev, rng, have)
            if not cands:
                break
            built_now = {}
            for name, op, inp, par in cands:
                try:
                    with np.errstate(all="ignore"):
                        full = self._apply(op, [cols[i] for i in inp], par)
                except Exception:
                    continue
                if np.isfinite(full).all() and np.std(full[dev]) > 1e-12 and abs(_corr(full[dev], Ydev)) < self.leakage_corr:
                    built_now[name] = (full, op, inp, par)
            cache.clear()
            screen = sorted(((self._gains(base_now, f[dev], Ydev, Tdev, task, k, seeds[:1], cache)[0], n)
                             for n, (f, *_rest) in built_now.items()), reverse=True)[: self.confirm_top]
            best = None
            for _, n in screen:
                g = self._gains(base_now, built_now[n][0][dev], Ydev, Tdev, task, k, seeds, cache)
                stable = float(g.mean() - g.std())
                if stable > self.min_gain and (best is None or stable > best[1]):
                    best = (n, stable, float(g.mean()), float(g.std()))
            if best is None:
                break
            n, stable, gm, gs = best
            full, op, inp, par = built_now[n]
            cols[n], have = full, have | {n}
            depth[n] = 1 + max(depth.get(i, 0) for i in inp)
            base_now = np.column_stack([base_now, full[dev]])
            recipes.append({"name": n, "op": op, "inputs": inp, "params": par, "depth": depth[n]})
            steps.append({"name": n, "op": op, "inputs": inp, "depth": depth[n], "gain_mean": gm,
                          "gain_std": gs, "stable_gain": stable})
        final_cv = float(_fast_skill(base_now, Ydev, Tdev, _fold_ids(Ydev, task, self.n_folds, seeds[0]), task, k))
        Zall = fr.matrix(names)
        Zfin = np.column_stack([Zall] + [cols[r["name"]] for r in recipes]) if recipes else Zall

        def hold_skill(Z):
            fit = _ridge_fit(Z[dev], Tdev, 1.0)
            P = _ridge_pred(fit, Z[hold])
            return float(_skill(task, yy[hold], P.argmax(1) if task == "classification" else P[:, 0], k))

        hb, hf = hold_skill(Zall), hold_skill(Zfin)
        self.recipes_, self.task_ = recipes, task
        self.base_names_ = names
        verdict = ("no construction beat the baseline with stable gain" if not recipes else
                   "confirmed on untouched rows" if hf - hb > 0 else
                   "gain did not hold on untouched rows: treat as selection optimism")
        self.report_ = {"kind": "feature_construction", "task": task, "steps": steps, "recipes": recipes,
                       "cv_skill_baseline": base_cv, "cv_skill_final": final_cv, "holdout_skill_baseline": hb,
                       "holdout_skill_final": hf, "holdout_gain": hf - hb, "verdict": verdict, "notes": notes,
                       "findings": [f"Accepted {len(recipes)} constructed feature(s): {[r['name'] for r in recipes]}.",
                                    f"Search CV skill {base_cv:.3f} -> {final_cv:.3f}; untouched holdout {hb:.3f} -> {hf:.3f}.",
                                    f"Verdict: {verdict}."]}
        return self

    def transform(self, X):
        df = (X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=self.base_names_)).copy()
        df.columns = [str(c) for c in df.columns]
        cols = {c: pd.to_numeric(df[c], errors="coerce").fillna(pd.to_numeric(df[c], errors="coerce").median()).to_numpy(float)
                for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
        for r in self.recipes_:
            with np.errstate(all="ignore"):
                cols[r["name"]] = self._apply(r["op"], [cols[i] for i in r["inputs"]], r["params"])
            df[r["name"]] = cols[r["name"]]
        return df

    @property
    def report(self):
        return self.report_


# --------------------------------------------------------------------- active learning
def _al_factory(learner, seed):
    from .statcore.lda.lda import LDA
    from .statcore.mlp.mlp import MLPClassifier

    if callable(learner) and not hasattr(learner, "fit"):
        return learner
    if hasattr(learner, "fit"):
        return lambda: copy.deepcopy(learner)
    table = {"logistic": lambda: LogisticRegression(variant="base", max_iter=100, random_state=seed),
             "knn": lambda: KNNClassifier(n_neighbors=5),
             "lda": lambda: LDA(variant="base"),
             "mlp": lambda: MLPClassifier(hidden_layer_sizes=(16,), max_epochs=60, random_state=seed)}
    if learner not in table:
        raise ValueError(f"learner must be one of {tuple(table)}, a factory, or an estimator with fit/predict_proba")
    return table[learner]


def _al_proba(factory, Xl, yl, Xq, k):
    model = factory()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xl, yl)
        P = np.asarray(model.predict_proba(Xq), float)
    full = np.zeros((len(Xq), k))
    full[:, np.asarray(model.classes_, int)] = P
    return full


def _al_skill(factory, Xl, yl, Xt, yt, k):
    try:
        return float(_skill("classification", yt, _al_proba(factory, Xl, yl, Xt, k).argmax(1), k))
    except Exception:
        return float("nan")


def evaluate_row_subset(X, y, idx, learner="logistic", test_size=0.3, random_state=0):
    """Score any row subset (e.g. from your own selector): skill retained vs the full training rows, plus
    how faithfully the subset keeps each column's distribution (KS) and the class balance (TV distance)."""
    fr = _Frame(X)
    yy, classes = TargetIntelligence._encode_target(np.asarray(y), "classification")
    k = len(classes)
    M = fr.matrix(fr.names)
    M = (M - M.mean(0)) / np.where(M.std(0) > 1e-12, M.std(0), 1.0)
    te = _fold_ids(yy, "classification", max(2, int(round(1 / test_size))), random_state) == 0
    idx = np.unique(np.asarray(idx, int))
    if len(idx) == 0 or idx.min() < 0 or idx.max() >= fr.n:
        raise ValueError("idx must be valid row positions of X")
    sub = idx[~te[idx]]
    train = np.where(~te)[0]
    factory = _al_factory(learner, random_state)
    s_sub = _al_skill(factory, M[sub], yy[sub], M[te], yy[te], k)
    s_full = _al_skill(factory, M[train], yy[train], M[te], yy[te], k)
    ks = [float(stats.ks_2samp(M[sub][:, j], M[train][:, j]).statistic) for j in range(M.shape[1]) if M[:, j].std() > 0]
    pc, ps = np.bincount(yy[train], minlength=k) / len(train), np.bincount(yy[sub], minlength=k) / max(len(sub), 1)
    return {"kind": "row_subset", "n_subset": int(len(sub)), "fraction_of_train": float(len(sub) / len(train)),
            "skill_subset": s_sub, "skill_full": s_full,
            "skill_retained": float(s_sub / s_full) if s_full and np.isfinite(s_full) and s_full > 0 else float("nan"),
            "ks_mean": float(np.mean(ks)) if ks else 0.0, "class_balance_tv": float(0.5 * np.abs(pc - ps).sum())}


class ActiveLearningStudy:
    """Simulate label-efficient learning: which rows should be labelled (or kept) first?

    Strategies: ``random`` (baseline), ``least_confidence``, ``margin``, ``entropy``, ``committee``
    (bootstrap vote entropy), ``kcenter`` (diversity), ``density_margin`` (uncertainty x representativeness),
    ``coverage_margin`` (own variant: margin uncertainty down-weighted near already-labelled rows, batch-wise),
    ``diversity_static`` (label-free farthest-first prefix). Pass ``row_selector(X, n) -> row positions`` to
    add any external selector of your own as one more, static curve;
    only its returned indices are used. Classification only.
    """

    STRATEGIES = ("random", "least_confidence", "margin", "entropy", "committee", "kcenter", "density_margin",
                  "coverage_margin", "diversity_static")

    def __init__(self, strategies=None, learner="logistic", seed_size=None, batch_size=None, n_steps=8,
                 test_size=0.3, n_repeats=3, committee_size=4, max_rows=2500, row_selector=None, random_state=0):
        strategies = tuple(strategies or self.STRATEGIES)
        bad = [s for s in strategies if s not in self.STRATEGIES]
        if bad:
            raise ValueError(f"unknown strategies {bad}; choose from {self.STRATEGIES}")
        self.strategies, self.learner, self.seed_size, self.batch_size = strategies, learner, seed_size, batch_size
        self.n_steps, self.test_size, self.n_repeats, self.committee_size = n_steps, test_size, n_repeats, committee_size
        self.max_rows, self.row_selector, self.random_state = max_rows, row_selector, random_state

    def _pick(self, strat, factory, M, yy, lab, pool, k, b, rng, D, committee_size):
        unl = np.setdiff1d(pool, lab)
        b = min(b, len(unl))
        if strat == "random":
            return rng.choice(unl, b, replace=False)
        if strat in ("kcenter", "coverage_margin"):
            dmin = D[np.ix_(unl, lab)].min(1)
        P = _al_proba(factory, M[lab], yy[lab], M[unl], k)
        srt = np.sort(P, axis=1)
        if strat == "least_confidence":
            sc = 1 - srt[:, -1]
        elif strat == "entropy":
            sc = -(P * np.log(P + 1e-12)).sum(1)
        elif strat == "committee":
            votes = np.zeros((len(unl), k))
            for _ in range(committee_size):
                bs = rng.choice(lab, len(lab), replace=True)
                if len(np.unique(yy[bs])) < 2:
                    bs = lab
                votes[np.arange(len(unl)), _al_proba(factory, M[bs], yy[bs], M[unl], k).argmax(1)] += 1
            q = votes / committee_size
            sc = -(q * np.log(q + 1e-12)).sum(1)
        elif strat == "kcenter":
            chosen = []
            for _ in range(b):
                j = int(np.argmax(dmin))
                chosen.append(j)
                dmin = np.minimum(dmin, D[unl, unl[j]])
            return unl[chosen]
        elif strat == "margin":
            sc = -(srt[:, -1] - srt[:, -2])
        else:
            unc = 1 - (srt[:, -1] - srt[:, -2])
            if strat == "density_margin":
                near = np.sort(D[np.ix_(unl, unl)], axis=1)[:, 1:11].mean(1)
                sc = unc * (1 / (1e-9 + near)) ** 1.0
            else:
                ref = np.median(dmin) + 1e-9
                chosen = []
                for _ in range(b):
                    j = int(np.argmax(unc * dmin / (dmin + ref)))
                    chosen.append(j)
                    dmin = np.minimum(dmin, D[unl, unl[j]])
                    unc[j] = -1
                return unl[chosen]
        sc = sc + 1e-9 * rng.random(len(sc))
        return unl[np.argsort(-sc)[:b]]

    def run(self, X, y):
        fr = _Frame(X)
        yy, classes = TargetIntelligence._encode_target(np.asarray(y), "classification")
        k = len(classes)
        if k < 2:
            raise ValueError("active learning needs at least two classes")
        M = fr.matrix(fr.names)
        M = (M - M.mean(0)) / np.where(M.std(0) > 1e-12, M.std(0), 1.0)
        notes = []
        if fr.n > self.max_rows:
            keep = np.sort(np.random.default_rng(self.random_state).choice(fr.n, self.max_rows, replace=False))
            M, yy = M[keep], yy[keep]
            notes.append(f"subsampled to {self.max_rows} rows")
        factory = _al_factory(self.learner, self.random_state)
        curves = {s: [] for s in self.strategies}
        sel_name = "row_selector"
        if self.row_selector is not None:
            curves[sel_name] = []
        ceilings, sizes_ref = [], None
        for r in range(self.n_repeats):
            seed = self.random_state + r
            rng = np.random.default_rng(seed)
            te = _fold_ids(yy, "classification", max(2, int(round(1 / self.test_size))), seed) == 0
            pool = np.where(~te)[0]
            Xt, yt = M[te], yy[te]
            n0 = self.seed_size or max(2 * k, 10)
            b = self.batch_size or max(5, len(pool) // 40)
            sizes = [min(len(pool), n0 + i * b) for i in range(self.n_steps + 1)]
            sizes = sorted(set(sizes))
            sizes_ref = sizes
            first = []
            for c in range(k):
                members = pool[yy[pool] == c]
                first += list(rng.choice(members, min(2, len(members)), replace=False))
            rest = np.setdiff1d(pool, first)
            seed_set = np.array(first + list(rng.choice(rest, max(0, n0 - len(first)), replace=False)), int)
            D = np.sqrt(np.maximum(((M[:, None, :] - M[None, :, :]) ** 2).sum(2), 0)) if len(M) <= 3000 else None
            ceilings.append(_al_skill(factory, M[pool], yy[pool], Xt, yt, k))
            for s in self.strategies:
                if s == "diversity_static":
                    order = [int(rng.choice(pool))]
                    dm = D[pool, order[0]].copy()
                    for _ in range(sizes[-1] - 1):
                        j = int(np.argmax(dm))
                        order.append(int(pool[j]))
                        dm = np.minimum(dm, D[pool, pool[j]])
                    curves[s].append([_al_skill(factory, M[order[:z]], yy[order[:z]], Xt, yt, k) for z in sizes])
                    continue
                lab, cur = seed_set.copy(), []
                for z in sizes:
                    if len(lab) < z:
                        lab = np.concatenate([lab, self._pick(s, factory, M, yy, lab, pool, k, z - len(lab), rng, D, self.committee_size)])
                    cur.append(_al_skill(factory, M[lab], yy[lab], Xt, yt, k))
                curves[s].append(cur)
            if self.row_selector is not None:
                cur = []
                for z in sizes:
                    loc = np.unique(np.asarray(self.row_selector(M[pool], z), int))
                    if len(loc) == 0 or loc.min() < 0 or loc.max() >= len(pool):
                        raise ValueError("row_selector must return valid row positions of the array it is given")
                    sel = pool[loc]
                    cur.append(_al_skill(factory, M[sel], yy[sel], Xt, yt, k))
                curves[sel_name].append(cur)
        ceiling = float(np.nanmean(ceilings))
        out, rnd = {}, np.nanmean(curves["random"], axis=0) if "random" in curves else None
        for s, runs in curves.items():
            arr = np.array(runs, float)
            mean = np.nanmean(arr, axis=0)
            alc = float(np.nanmean(mean))
            hit = [int(z) for z, v in zip(sizes_ref, mean) if v >= 0.95 * ceiling] if ceiling > 0 else []
            ent = {"sizes": sizes_ref, "mean_skill": mean.tolist(), "std_skill": np.nanstd(arr, axis=0).tolist(),
                   "area_under_curve": alc, "labels_to_95pct_of_ceiling": hit[0] if hit else None}
            if "random" in curves and s != "random":
                d = np.nanmean(arr, axis=1) - np.nanmean(np.array(curves["random"], float), axis=1)
                ent["alc_gain_vs_random_mean"], ent["alc_gain_vs_random_std"] = float(d.mean()), float(d.std())
            out[s] = ent
        ranking = sorted(out, key=lambda s: -out[s]["area_under_curve"])
        beat = [s for s in ranking if out[s].get("alc_gain_vs_random_mean", 0) > out[s].get("alc_gain_vs_random_std", 0) and s != "random"]
        rec = (f"best average curve: {ranking[0]}; clearly better than random: {beat or 'none'}"
               + ("" if beat else " (random is competitive here: sophisticated selection is not justified)"))
        self.report_ = {"kind": "active_learning", "learner": str(self.learner), "n_classes": k, "ceiling_skill": ceiling,
                        "label_sizes": sizes_ref, "strategies": out, "ranking": ranking, "recommendation": rec, "notes": notes,
                        "findings": [f"Learner {self.learner}: ceiling skill with all pool rows {ceiling:.3f}.",
                                     f"Ranking by area under the learning curve: {ranking[:4]}.", rec.capitalize() + "."]}
        return self.report_


# --------------------------------------------------------------- sections and figures
SECTIONS = ("power", "group_scan", "root_cause", "construction", "active_learning", "time_series")


def run_sections(df, target_column, sections=("power",), time_column=None, random_state=0, **options):
    """Run chosen toolkit sections on ``df``. Failures become ``{"error": ...}`` instead of exceptions.
    ``options`` may hold ``construction={...}`` / ``active_learning={...}`` keyword dicts."""
    bad = [s for s in sections if s not in SECTIONS]
    if bad:
        raise ValueError(f"unknown sections {bad}; choose from {SECTIONS}")
    X, y = df.drop(columns=[target_column]), df[target_column].to_numpy()
    res = {}
    for s in sections:
        try:
            if s == "power":
                res[s] = power_analysis(X, y, random_state=random_state, **options.get("power", {}))
            elif s == "group_scan":
                res[s] = scan_groups(df, target_column, **options.get("group_scan", {}))
            elif s == "root_cause":
                res[s] = root_cause(df, target_column, **options.get("root_cause", {}))
            elif s == "construction":
                res[s] = FeatureConstructor(random_state=random_state, **options.get("construction", {})).fit(X, y).report_
            elif s == "active_learning":
                res[s] = ActiveLearningStudy(random_state=random_state, **options.get("active_learning", {})).run(X, y)
            elif s == "time_series":
                if not time_column:
                    raise ValueError("time_column is required for the time_series section")
                res[s] = time_series_analysis(df, time_column, target_column, random_state=random_state,
                                              **options.get("time_series", {}))
        except Exception as exc:
            res[s] = {"kind": s, "error": f"{type(exc).__name__}: {exc}"}
    return res


def describe(result):
    """Plain-language summary of one result or of a ``run_sections`` dict."""
    if "kind" not in result:
        return "\n\n".join(f"[{k}]\n" + describe(v) for k, v in result.items())
    if "error" in result:
        return f"error: {result['error']}"
    return "\n".join(f"- {f}" for f in result.get("findings", []))


def plot_result(result):
    """Figure for a toolkit result (power, group_comparison, ab_test, time_series, feature_construction,
    active_learning, root_cause). Nothing is saved; call ``fig.savefig`` yourself."""
    from matplotlib.figure import Figure

    from .visualization import apply_house_style

    pal = apply_house_style()
    kind = result.get("kind")
    fig = Figure(figsize=(8, 4.6), constrained_layout=True)
    if kind == "power":
        a1, a2 = fig.subplots(1, 2)
        vals = [result["predictive"]["ridge_cv_skill"], result["predictive"]["knn_cv_skill"],
                result["explanatory"]["in_sample_skill"]]
        a1.bar(["predictive\nridge CV", "predictive\nKNN CV", "explanatory\nin-sample"], vals, color=[pal[0], pal[1], pal[3]])
        a1.set_title("Predictive vs explanatory skill", fontsize=10)
        f = result["explanatory"]["features"][:10][::-1]
        a2.barh([r["feature"] for r in f], [r["explanatory_strength"] for r in f],
                color=[pal[0] if r["significant"] else "#bbbbbb" for r in f])
        a2.set_title("Feature strength (blue = significant, BH)", fontsize=10)
    elif kind == "group_comparison" and "table" in result and isinstance(result["table"], list):
        ax = fig.subplots()
        t = result["table"]
        m = np.array([r["mean"] for r in t])
        ax.errorbar([r["group"] for r in t], m, yerr=[m - [r["ci95_low"] for r in t], [r["ci95_high"] for r in t] - m],
                    fmt="o", color=pal[0], capsize=4)
        ax.set_title(f"{result['value']} by {result['group']} ({result['test']}, p={result['p_value']:.3g})", fontsize=10)
    elif kind == "ab_test":
        ax = fig.subplots()
        ci = result["ci95_difference"]
        ax.errorbar([result["absolute_lift"]], [0], xerr=[[result["absolute_lift"] - ci[0]], [ci[1] - result["absolute_lift"]]],
                    fmt="o", color=pal[0], capsize=5)
        ax.axvline(0, color=pal[3], ls="--")
        ax.set_yticks([])
        ax.set_title(f"{result['treatment']} - {result['control']}: lift with 95% CI (p={result['p_value']:.3g})", fontsize=10)
    elif kind == "time_series":
        axes = fig.subplots(3, 1, sharex=True)
        s = result["series"]
        t = pd.to_datetime(s["time"])
        for ax, key, col in zip(axes, ("value", "trend", "seasonal"), (pal[0], pal[1], pal[2])):
            ax.plot(t, s[key], color=col, lw=1)
            ax.set_ylabel(key, fontsize=8)
        axes[0].set_title(f"{result['value']}: trend {result['trend']['direction']} "
                          f"(p={result['trend']['p_value']:.3g})", fontsize=10)
        if result["change_point"]["significant"]:
            axes[0].axvline(pd.Timestamp(result["change_point"]["time"]), color=pal[3], ls="--")
    elif kind == "feature_construction":
        ax = fig.subplots()
        labels = ["search CV", "holdout"]
        ax.bar(np.arange(2) - 0.18, [result["cv_skill_baseline"], result["holdout_skill_baseline"]], 0.36, label="baseline", color=pal[1])
        ax.bar(np.arange(2) + 0.18, [result["cv_skill_final"], result["holdout_skill_final"]], 0.36, label="constructed", color=pal[0])
        ax.set_xticks(range(2), labels)
        ax.legend(fontsize=8)
        ax.set_title("Construction: " + result["verdict"], fontsize=9)
    elif kind == "active_learning":
        ax = fig.subplots()
        for name, st in result["strategies"].items():
            ax.plot(st["sizes"], st["mean_skill"], label=name, lw=2.2 if name == "random" else 1.3,
                    ls="--" if name in ("random", "row_selector", "diversity_static") else "-")
        ax.axhline(result["ceiling_skill"], color="k", lw=0.8, ls=":")
        ax.set_xlabel("labelled rows")
        ax.set_ylabel("skill")
        ax.legend(fontsize=7, ncol=2)
        ax.set_title("Active learning curves (dotted = ceiling)", fontsize=10)
    elif kind == "root_cause":
        ax = fig.subplots()
        seg = result["segments"][:8][::-1]
        if not seg:
            return None
        ax.barh([s["segment"][:50] for s in seg], [s["lift"] for s in seg], color=pal[0])
        ax.axvline(1, color=pal[3], ls="--")
        ax.set_title(f"Segments concentrating the bad outcome (base {result['base_rate']:.1%})", fontsize=10)
    else:
        return None
    return fig


class DataIntelligence:
    """Autonomous data investigator (the public entry point).

    Parameters
    ----------
    mode : one of ``MODES``.
    plot_dir : folder where ``analyze`` saves figures; ``None`` (default) writes nothing. ``plot()`` always works.
    vif_threshold, corr_threshold, leakage_threshold : pruning and leakage thresholds.
    llm_backend : ``None``, a callable ``prompt -> str``, ``"ollama"``, ``"custom_url"`` or ``"scikit-llm"``.
    max_rows : rows used by the expensive analyses (a seeded sample).
    """

    def __init__(self, mode="comprehensive_eda", plot_dir=None, vif_threshold=10.0, corr_threshold=0.90,
                 leakage_threshold=0.95, llm_backend: str | Callable | None = None, llm_url=None,
                 llm_model="llama3", llm_api_key=None, max_rows=5000, random_state=0, verbose=False,
                 export_dir=None, export_formats=("json", "txt", "html"), outlier_max_rows=2000, som_grid=None):
        mode = _ALIASES.get(mode, mode)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
        self.outlier_max_rows = outlier_max_rows
        self.som_grid = som_grid
        self._outlier_viz = None
        self._som_viz = None
        self.export_dir = export_dir
        self.export_formats = export_formats
        self.mode = mode
        self.plot_dir = plot_dir
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold
        self.leakage_threshold = leakage_threshold
        self.llm_backend = llm_backend
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.max_rows = max_rows
        self.random_state = random_state
        self.verbose = verbose
        self._cluster_viz = None
        self._ctx = None

    # ----------------------------------------------------------------- public API
    def analyze(self, df, target_column=None, time_column=None, user_question=None, y=None):
        df = df.reset_index(drop=True).copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        df.columns = [str(c) for c in df.columns]
        if y is not None:
            df["target"] = np.asarray(y)
            target_column = target_column or "target"
        if target_column is not None and target_column not in df.columns:
            raise KeyError(f"target_column {target_column!r} not in data")
        if time_column is not None and time_column not in df.columns:
            raise KeyError(f"time_column {time_column!r} not in data")
        mode = self.mode
        if mode in ("target_intelligence", "field_investigation") and target_column is None:
            raise ValueError(f"mode={mode!r} needs target_column (or y)")
        time_column = time_column or self._detect_time_column(df, target_column)
        features = [c for c in df.columns if c not in (target_column, time_column)]
        X = df[features]
        yv = df[target_column] if target_column else None
        sample = self._sample(df)
        self._cluster_viz = None
        self._outlier_viz = None
        self._som_viz = None
        self._candidates = []
        details: dict[str, Any] = {"mode": mode, "overview": {
            "rows": len(df), "columns": df.shape[1], "missing_cells": int(df.isna().sum().sum()),
            "memory_kb": float(df.memory_usage(deep=True).sum() / 1024), "target_column": target_column,
            "time_column": time_column, "n_features": len(features), "user_question": user_question}}
        want = lambda *modes: mode in modes or mode == "autonomous"

        if want("comprehensive_eda", "llm_augmented", "auto_question_generator", "field_investigation",
                "feature_intelligence", "target_intelligence"):
            details["what_happened"] = self._what_happened(df)
        if want("comprehensive_eda", "llm_augmented", "field_investigation"):
            details["why"] = self._why(sample, features, target_column, time_column)
        if want("comprehensive_eda", "llm_augmented", "field_investigation") and time_column:
            details["what_then"] = self._what_then(df, time_column, target_column, features)
        if want("target_intelligence") and target_column:
            ti = TargetIntelligence(scope="full", vif_threshold=self.vif_threshold, corr_threshold=self.corr_threshold,
                                    leakage_threshold=self.leakage_threshold, max_rows=self.max_rows,
                                    random_state=self.random_state).fit(X, yv.to_numpy(), target_name=target_column)
            details["target_intelligence"] = ti.report.to_dict()
            details["target_intelligence_text"] = ti.report.to_text()
            self.target_intelligence_ = ti
        if want("feature_intelligence"):
            fi = FeatureIntelligence(vif_threshold=self.vif_threshold, corr_threshold=self.corr_threshold,
                                     max_rows=self.max_rows, random_state=self.random_state)
            fi.fit(X, yv.to_numpy() if yv is not None else None)
            details["feature_intelligence"] = fi.report.to_dict()
            details["feature_intelligence_text"] = fi.report.to_text()
            self.feature_intelligence_ = fi
        plan = self.question_plan(df, target_column=target_column, time_column=time_column, details=details)
        details["question_plan"] = plan
        if target_column is None:
            details["candidate_targets"] = getattr(self, "_candidates", [])
        details["questions"] = [user_question] if user_question else [q["question"] for q in plan]
        details["recommendations"] = self._recommendations(details)
        score = self._score(details)
        details["narrative"] = self._narrative(details, score)
        if want("llm_augmented"):
            details.update(self._llm(details, score, user_question))
        self.report_ = DataIntelligenceReport(score, details)
        self._ctx = {"df": df, "features": features, "target": target_column, "time": time_column,
                     "outlier_viz": self._outlier_viz, "som_viz": self._som_viz}
        if self.plot_dir:
            self.report_.details["visualizations"] = {k: v for k, v in self.plot(save_to=self.plot_dir)["paths"].items()}
        if self.export_dir:
            self.export(self.export_dir, formats=self.export_formats)
        return self.report_

    def export(self, directory=None, formats=None, include_figures=True, name="data_intelligence"):
        """Save the last report (json / txt / html) plus its figures. Returns ``{format: path, "figures": {...}}``."""
        report = getattr(self, "report_", None)
        if report is None:
            raise RuntimeError("run analyze() first")
        directory = directory or self.export_dir
        if not directory:
            raise ValueError("give a directory (or set export_dir)")
        figure_paths = report.details.get("visualizations") or {}
        if include_figures and not figure_paths:
            figure_paths = self.plot(save_to=os.path.join(directory, "figures"))["paths"]
        written = report.export(directory, formats=formats or self.export_formats, name=name,
                                figure_paths=figure_paths if include_figures else {})
        written["figures"] = dict(figure_paths) if include_figures else {}
        return written

    def plot(self, report=None, save_to=None):
        """Build the figures for a report. Returns ``{"figures": {name: Figure}, "paths": {name: path}}``.

        Panels appear only when their section exists (e.g. association/leakage panels need a target).
        """
        report = report or getattr(self, "report_", None)
        if report is None or self._ctx is None:
            raise RuntimeError("run analyze() first")
        figures = _build_figures(report.details, report.summary_score, self._ctx, self._cluster_viz,
                                 self.leakage_threshold, self.vif_threshold)
        paths: dict[str, str] = {}
        if save_to:
            os.makedirs(save_to, exist_ok=True)
            for name, fig in figures.items():
                paths[name] = os.path.join(save_to, f"{name}.png")
                fig.savefig(paths[name], dpi=150)
        return {"figures": figures, "paths": paths}

    # ------------------------------------------------------------ analysis toolkit
    def power_analysis(self, df, target_column, **kw):
        """Predictive vs explanatory power and statistical power. See ``power_analysis``."""
        return power_analysis(df.drop(columns=[target_column]), df[target_column].to_numpy(),
                              random_state=self.random_state, **kw)

    def compare_groups(self, df, value, group, **kw):
        return compare_groups(df, value, group, **kw)

    def scan_groups(self, df, target_column, **kw):
        return scan_groups(df, target_column, **kw)

    def ab_test(self, df, group, outcome, **kw):
        return ab_test(df, group, outcome, **kw)

    def root_cause(self, df, outcome, **kw):
        return root_cause(df, outcome, **kw)

    def time_series_analysis(self, df, time_column, value_column, **kw):
        return time_series_analysis(df, time_column, value_column, random_state=self.random_state, **kw)

    def construct_features(self, df, target_column, **kw):
        """Iterative, distribution-aware feature construction. Returns the fitted ``FeatureConstructor``."""
        return FeatureConstructor(random_state=self.random_state, **kw).fit(df.drop(columns=[target_column]),
                                                                           df[target_column].to_numpy())

    def active_learning(self, df, target_column, **kw):
        """Active-learning simulation. Returns the report dict. ``row_selector=`` plugs in your own selector."""
        return ActiveLearningStudy(random_state=self.random_state, **kw).run(df.drop(columns=[target_column]),
                                                                            df[target_column].to_numpy())

    def run_sections(self, df, target_column, sections=("power",), time_column=None, **options):
        """Run selected toolkit sections: any of ``SECTIONS``."""
        res = run_sections(df, target_column, sections, time_column, self.random_state, **options)
        if self.export_dir:
            export_report(res, self.export_dir, name="toolkit", formats=self.export_formats,
                          text=describe(res), title="ReliableML toolkit")
        return res

    def auto_generate_data_questions(self, df, target_column=None, time_column=None):
        """Formulate analytical questions from variance, correlation, tails, missingness and time."""
        return [item["question"] for item in self.question_plan(df, target_column, time_column)]

    def question_plan(self, df, target_column=None, time_column=None, details=None):
        """Turn the data into a prioritised list of questions worth asking, for someone with data but no question.

        Each item has ``category``, ``priority`` (high/medium/low), ``question``, ``evidence`` (the numbers that
        triggered it) and ``next_step`` (what to run to answer it). Without a target the plan starts with
        ``candidate_targets``: which columns the rest of the data can predict. ``details`` (an ``analyze`` result)
        adds cluster and outlier-consensus questions.
        """
        df = pd.DataFrame(df)
        details = details or {}
        feats = [c for c in df.columns if c not in (target_column, time_column)]
        num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
               and df[c].nunique() > 1]
        cat = [c for c in feats if c not in num and 2 <= df[c].nunique() <= 30]
        plan: list[dict[str, Any]] = []

        def add(category, priority, question, evidence, next_step):
            plan.append({"category": category, "priority": priority, "question": question,
                         "evidence": evidence, "next_step": next_step})

        if target_column:
            add("target", "high", f"Which features carry genuine signal for '{target_column}', and is any of them a "
                "post-outcome proxy?", "target given", f"DataIntelligence(mode='target_intelligence').analyze(df, '{target_column}')")
            add("target", "high", f"Is '{target_column}' explainable from these features, or would a different target "
                "definition do better?", "target given", "see the target-intelligence explainability and alternative-target sections")
        else:
            cands = self._candidate_targets(df, num, cat)
            self._candidates = cands
            if cands:
                best = cands[0]
                add("target", "high", "What do you want to predict or explain? These columns are the most learnable from "
                    "the rest of the data: " + ", ".join(f"'{c['column']}' ({c['skill']:.2f})" for c in cands[:3]) + ".",
                    {"skills": {c["column"]: c["skill"] for c in cands[:5]}},
                    f"DataIntelligence(mode='target_intelligence').analyze(df, '{best['column']}')")
                for c in cands:
                    if c.get("derived_from"):
                        add("target", "low", f"'{c['column']}' looks predictable only because it is almost a copy of "
                            f"'{c['derived_from']}': is it a unit change or a derived field rather than a real outcome?",
                            c, "exclude it as a target; keep one of the two as a feature")
            else:
                add("target", "medium", "No column is clearly predictable from the others: is there an outcome column "
                    "missing from this table?", "no candidate reached skill 0.1", "join an outcome or label column")

        clusters = (details.get("why", {}) or {}).get("clusters", {})
        if isinstance(clusters, dict) and clusters.get("n_clusters", 0) >= 2 and clusters.get("silhouette", 0) >= 0.25:
            big = max(clusters["clusters"], key=lambda c: c["share"])
            small = min(clusters["clusters"], key=lambda c: c["share"])
            sig = ", ".join(f"{k} ({v:+.1f} sd)" for k, v in small["signature"].items())
            add("structure", "high", f"The rows fall into {clusters['n_clusters']} groups (silhouette "
                f"{clusters['silhouette']:.2f}). Who is the smallest group ({small['share']:.0%} of rows, distinguished by "
                f"{sig}), and should it be modelled separately?", {"largest_share": big["share"], "smallest_share": small["share"]},
                "compare the clusters in the 'clusters' figure; fit one model per group and compare")
        ens = (details.get("why", {}) or {}).get("outliers", {}).get("ensemble") if details else None
        if ens and ens["consensus"]["n_rows"]:
            top = ens["consensus"]["top_examples"][0]
            cols = ", ".join(top["driving_columns"])
            add("outliers", "high", f"{ens['consensus']['n_rows']} rows ({ens['consensus']['fraction']:.1%}) are flagged by "
                f"at least two detectors, mostly because of {cols}. Are they recording errors or the most interesting cases?",
                ens["consensus"]["top_examples"][:3], "inspect df.iloc[row_positions] and the 'outlier_consensus' and 'som' figures")
        elif num:
            tails = {}
            for c in num:
                s = df[c].dropna()
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                tails[c] = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()) if iqr > 0 else 0.0
            c, v = max(tails.items(), key=lambda kv: kv[1])
            if v > 0.02:
                add("outliers", "medium", f"{v:.1%} of rows lie in the tails of '{c}': errors or a real sub-population?",
                    {"tail_fraction": v}, "plot the column; compare tail rows with the rest")

        if len(num) > 1:
            corr = df[num].corr(numeric_only=True).abs()
            pairs = sorted(((float(corr.loc[a, b]), a, b) for a, b in combinations(num, 2) if corr.loc[a, b] > 0.7),
                           reverse=True)
            groups: dict[str, set] = {}
            for r, a, b in pairs:
                g = groups.get(a) or groups.get(b) or {a, b}
                g.update((a, b))
                groups[a] = groups[b] = g
            seen = []
            for g in groups.values():
                if g not in seen:
                    seen.append(g)
            for g in sorted(seen, key=len, reverse=True)[:2]:
                if len(g) >= 3:
                    add("relationships", "medium", f"{len(g)} columns move together ({', '.join(sorted(g)[:5])}): do they "
                        "measure one underlying quantity that you could summarise with a single feature?",
                        {"columns": sorted(g)}, "multicollinearity_report(X), then keep one column or use a combined score")
            for r, a, b in pairs[:3]:
                add("relationships", "medium" if r > 0.9 else "low", f"'{a}' and '{b}' move together (|r|={r:.2f}): is one "
                    "derived from the other, or do they share a driver?", {"abs_r": r}, "check how each column is measured")

            skew = {c: float(stats.skew(df[c].dropna())) for c in num if df[c].nunique() > 10}
            for c, v in sorted(skew.items(), key=lambda kv: -abs(kv[1]))[:1]:
                if abs(v) > 2:
                    add("distribution", "low", f"'{c}' is strongly skewed (skewness {v:+.1f}): does a log or rank scale "
                        "describe it better, and are the extremes real?", {"skewness": v}, f"compare models on {c} and log({c})")
        cv = {c: abs(df[c].std() / df[c].mean()) for c in num if df[c].mean() != 0}
        for c, v in sorted(cv.items(), key=lambda kv: -kv[1])[:1]:
            if v > 0.5:
                add("distribution", "low", f"What drives the high variability of '{c}' (coefficient of variation {v:.2f})?",
                    {"cv": v}, "split the column by categorical columns or clusters and compare")

        for c in cat:
            top = df[c].value_counts(normalize=True)
            if len(top) >= 2 and top.iloc[0] > 0.8:
                add("categories", "medium", f"'{c}' is {top.iloc[0]:.0%} '{top.index[0]}': are the rare levels rare by nature "
                    "or under-sampled?", {"top_share": float(top.iloc[0])}, "stratify any split on this column")
        best_eta = None
        for c in cat[:6]:
            codes = pd.factorize(df[c].astype(str))[0]
            for n in num[:12]:
                ok = df[n].notna().to_numpy()
                if ok.sum() > 20:
                    e = float(_eta(df[n].to_numpy(float)[ok], codes[ok]))
                    if best_eta is None or e > best_eta[0]:
                        best_eta = (e, c, n)
        if best_eta and best_eta[0] >= 0.3:
            add("categories", "medium", f"'{best_eta[2]}' differs strongly between levels of '{best_eta[1]}' (eta "
                f"{best_eta[0]:.2f}): is this a real group effect or a different measurement process per group?",
                {"eta": best_eta[0]}, f"compare {best_eta[2]} by {best_eta[1]}; consider group-wise models and fairness checks")

        miss = df[feats].isna().mean()
        for c in miss[miss > 0.05].sort_values(ascending=False).index[:2]:
            flag = df[c].isna().to_numpy(float)
            link = None
            for n in num:
                if n != c and df[n].notna().all() and flag.std() > 0:
                    r = abs(float(np.corrcoef(flag, df[n].to_numpy(float))[0, 1]))
                    if link is None or r > link[0]:
                        link = (r, n)
            if link and link[0] >= 0.3:
                add("missingness", "high", f"'{c}' is missing in {miss[c]:.0%} of rows and the gaps track '{link[1]}' "
                    f"(|r|={link[0]:.2f}): why is it recorded only for some rows?", {"abs_r": link[0]},
                    "do not impute with the mean; model the missingness or use the missing flag as a feature")
            else:
                add("missingness", "medium", f"'{c}' is missing in {miss[c]:.0%} of rows with no visible link to the numeric "
                    "columns: is it random, or tied to the outcome?", {"missing": float(miss[c])},
                    "compare the outcome between missing and present rows")
        if time_column:
            add("time", "medium", f"Is there a trend or seasonality over '{time_column}', and has the relationship between "
                "the features and the outcome changed over time?", "time column detected",
                "DriftMonitor on early vs late rows; assess_deterioration on batches")
        dup = int(df.duplicated().sum())
        if dup:
            add("hygiene", "medium", f"{dup} rows are exact duplicates: genuine repeats or a data-collection fault?",
                {"duplicates": dup}, "df.drop_duplicates() before any train/test split")
        order = {"high": 0, "medium": 1, "low": 2}
        plan.sort(key=lambda p: order[p["priority"]])
        return plan or [{"category": "structure", "priority": "low", "question": "What is the main structure of this "
                         "dataset?", "evidence": "nothing stood out", "next_step": "start with the distributions figure"}]

    def _candidate_targets(self, df, num, cat, max_numeric=8, max_categorical=4):
        """Rank columns by how well the remaining numeric columns predict them (fast cross-validated ridge)."""
        if len(num) < 2:
            return []
        sample = self._sample(df)
        base = sample[num].apply(lambda s: s.fillna(s.median()))
        spread = {c: abs(base[c].std() / (abs(base[c].mean()) + 1e-12)) for c in num}
        picks = sorted(num, key=lambda c: -spread[c])[:max_numeric] + [c for c in cat if df[c].nunique() <= 10][:max_categorical]
        out = []
        for c in picks:
            others = [x for x in num if x != c]
            derived = None
            if c in num:
                y = base[c].to_numpy(float)
                task, k = "regression", 0
                r = {x: abs(float(np.corrcoef(y, base[x].to_numpy(float))[0, 1])) for x in others if base[x].std() > 0}
                derived = max(r, key=r.get) if r and max(r.values()) >= 0.98 else None
                if derived:
                    others = [x for x in others if abs(r.get(x, 0)) < 0.98]
            else:
                codes, uniques = pd.factorize(sample[c].astype(str))
                counts = np.bincount(codes)
                if counts.min() < 5:
                    continue
                y, task, k = codes.astype(int), "classification", len(uniques)
            if not others:
                continue
            Z = base[others].to_numpy(float)
            folds = _fold_ids(y, task, 3, self.random_state)
            skill = float(_fast_skill(Z, y, _targets_matrix(y, task, k), folds, task, k))
            verdict = ("strongly explained" if skill >= 0.6 else "moderately explained" if skill >= 0.3
                       else "weakly explained" if skill >= 0.1 else "not explainable from the other columns")
            out.append({"column": c, "kind": "numeric" if c in num else "categorical", "task": task,
                        "skill": skill, "verdict": verdict, "derived_from": derived})
        return sorted(out, key=lambda r: -r["skill"])

    def detect_target_leakage(self, df, target_column):
        """Leakage and linkage screening for ``target_column``."""
        df = pd.DataFrame(df)
        ti = TargetIntelligence(scope="full", leakage_threshold=self.leakage_threshold, search_targets=False,
                                search_features=False, random_state=self.random_state)
        ti.fit(df.drop(columns=[target_column]), df[target_column].to_numpy(), target_name=target_column)
        return ti.report["leakage"]

    def select_features(self, df, target_column=None, n_features_to_select=None):
        df = pd.DataFrame(df)
        X = df.drop(columns=[target_column]) if target_column else df
        fi = FeatureIntelligence(n_features=n_features_to_select, vif_threshold=self.vif_threshold,
                                 corr_threshold=self.corr_threshold, random_state=self.random_state)
        fi.fit(X, df[target_column].to_numpy() if target_column else None)
        return fi.report.to_dict()

    # ------------------------------------------------------------------ internals
    def _sample(self, df):
        if len(df) <= self.max_rows:
            return df
        return df.iloc[np.sort(np.random.default_rng(self.random_state).choice(len(df), self.max_rows, replace=False))]

    @staticmethod
    def _detect_time_column(df, target_column):
        for c in df.columns:
            if c != target_column and pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        return None

    def _what_happened(self, df):
        profiles, hazards = {}, {"constant": [], "quasi_constant": [], "high_cardinality": []}
        rng = np.random.default_rng(self.random_state)
        for c in df.columns:
            s = df[c]
            miss = int(s.isna().sum())
            p = {"dtype": str(s.dtype), "missing_count": miss, "missing_percentage": 100.0 * miss / len(df),
                 "unique_values": int(s.nunique()), "memory_bytes": int(s.memory_usage(deep=True))}
            top = s.value_counts(normalize=True, dropna=True)
            if p["unique_values"] <= 1:
                hazards["constant"].append(c)
            elif len(top) and top.iloc[0] >= 0.99:
                hazards["quasi_constant"].append(c)
            if not pd.api.types.is_numeric_dtype(s) and p["unique_values"] > 50 and p["unique_values"] / len(df) > 0.5:
                hazards["high_cardinality"].append(c)
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                v = s.dropna().to_numpy(float)
                if len(v) > 2:
                    p.update({"mean": float(v.mean()), "std": float(v.std(ddof=1)), "median": float(np.median(v)),
                              "min": float(v.min()), "max": float(v.max()),
                              "skewness": float(stats.skew(v)), "excess_kurtosis": float(stats.kurtosis(v)),
                              "percentiles": {f"p{q}": float(np.percentile(v, q)) for q in _PERCENTILES}})
                    if v.std() > 0:
                        sub = rng.choice(v, 5000, replace=False) if len(v) > 5000 else v
                        sw = stats.shapiro(sub) if len(sub) >= 3 else None
                        ks = stats.kstest((sub - sub.mean()) / sub.std(), "norm")
                        p["normality"] = {"shapiro_p": float(sw.pvalue) if sw else None, "ks_p": float(ks.pvalue),
                                          "looks_normal": bool(sw is not None and sw.pvalue > 0.05 and ks.pvalue > 0.05)}
            profiles[c] = p
        return {"column_profiles": profiles, "hazards": hazards}

    def _why(self, df, features, target_column, time_column):
        num = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
               and df[c].nunique() > 1]
        out: dict[str, Any] = {"numeric_columns": num}
        field = target_column if target_column and pd.api.types.is_numeric_dtype(df[target_column]) else None
        if field is None and num:
            field = max(num, key=lambda c: df[c].std() / (abs(df[c].mean()) + 1e-12))
            out["field_choice"] = "highest coefficient of variation (no numeric target given)"
        out["investigated_field"] = field
        if len(num) >= 2:
            sub = df[num + ([target_column] if field == target_column and target_column not in num else [])]
            pear = sub.corr(method="pearson")
            spear = sub.corr(method="spearman")
            kend = sub.sample(min(len(sub), 1500), random_state=self.random_state).corr(method="kendall")
            out["pearson"], out["spearman"], out["kendall"] = pear.to_dict(), spear.to_dict(), kend.to_dict()
            cols = list(sub.columns)
            out["strong_pairs"] = sorted(
                ({"a": a, "b": b, "pearson": float(pear.loc[a, b]), "spearman": float(spear.loc[a, b])}
                 for a, b in combinations(cols, 2) if abs(pear.loc[a, b]) >= 0.7 or abs(spear.loc[a, b]) >= 0.7),
                key=lambda r: -abs(r["pearson"]))[:25]
            if len(cols) <= 30:
                codes = {c: self._codes(sub[c]) for c in cols}
                out["mutual_info_matrix"] = {a: {b: float(_mutual_info(codes[a], codes[b])) for b in cols} for a in cols}
            if field:
                drivers = []
                for c in cols:
                    if c == field:
                        continue
                    ok = sub[[c, field]].dropna()
                    r, p = stats.pearsonr(ok[c], ok[field]) if ok[c].std() > 0 and ok[field].std() > 0 else (0.0, 1.0)
                    drivers.append({"feature": c, "pearson": float(r), "p_value": float(p),
                                    "spearman": float(spear.loc[c, field]),
                                    "mutual_info": float(_mutual_info(codes[c], codes[field])) if len(cols) <= 30 else None})
                drivers.sort(key=lambda r: -max(abs(r["pearson"]), abs(r["spearman"])))
                out["drivers"] = drivers[:10]
                chains = []
                for d in drivers[:3]:
                    sec = [{"feature": c, "pearson_with_driver": float(pear.loc[d["feature"], c])}
                           for c in cols if c not in (field, d["feature"]) and abs(pear.loc[d["feature"], c]) >= 0.5]
                    chains.append({"primary_driver": d["feature"], "secondary_links": sec[:5]})
                out["multi_hop_chains"] = chains
        out["outliers"] = self._outliers(df, num)
        out["clusters"] = self._clusters(df, num)
        return out

    @staticmethod
    def _codes(s):
        v = s.to_numpy(float)
        v = np.where(np.isfinite(v), v, np.nanmedian(v))
        qs = np.unique(np.quantile(v, np.linspace(0, 1, 11)[1:-1]))
        return np.searchsorted(qs, v, side="right").astype(int)

    def _outliers(self, df, num):
        out: dict[str, Any] = {"iqr": {}, "multivariate": None}
        for c in num:
            s = df[c].dropna()
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                out["iqr"][c] = {"low": int((s < lo).sum()), "high": int((s > hi).sum()),
                                 "fraction": float(((s < lo) | (s > hi)).mean()), "bounds": [float(lo), float(hi)]}
        if len(num) >= 2:
            Z = df[num].apply(lambda s: s.fillna(s.median())).to_numpy(float)
            d2, cut, Zr = robust_mahalanobis(Z)
            idx = np.flatnonzero(d2 > cut)
            why = {}
            if len(idx):
                contrib = np.abs(Zr[idx]).mean(0)
                why = {num[i]: float(contrib[i]) for i in np.argsort(-contrib)[:5]}
            out["multivariate"] = {"method": "robust Mahalanobis (median/MAD), chi-square 0.999 cut",
                                   "n_outlier_rows": int(len(idx)), "fraction": float(len(idx) / len(Z)),
                                   "row_positions": idx[:50].tolist(), "columns_driving_outliers": why}
            if len(df) >= 30:
                out["ensemble"] = self._outlier_ensemble(Z, num)
        return out

    def _outlier_ensemble(self, Z, num):
        """Isolation forest + LOF + robust Mahalanobis + SOM quantization error on one seeded sample."""
        n = len(Z)
        rng = np.random.default_rng(self.random_state)
        rows = np.arange(n) if n <= self.outlier_max_rows else np.sort(
            rng.choice(n, self.outlier_max_rows, replace=False))
        S = Z[rows]
        sd = S.std(0)
        std = (S - S.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
        iso = isolation_forest_scores(std, random_state=self.random_state)
        lof = local_outlier_factor(std, k=min(20, len(std) - 1))
        d2, cut, _ = robust_mahalanobis(S)
        som = SelfOrganisingMap(grid=self.som_grid, random_state=self.random_state).fit(std)
        qe = som.quantization_error_row_
        flags = {"isolation_forest": _robust_flag(iso), "lof": lof > 1.5, "mahalanobis": d2 > cut,
                 "som_quantization": _robust_flag(qe)}
        votes = np.sum(list(flags.values()), axis=0)
        consensus = votes >= 2
        ranks = np.mean([stats.rankdata(v) / len(v) for v in (iso, lof, d2, qe)], axis=0)
        top = np.argsort(-ranks)[:10]
        z = np.abs(std)
        examples = [{"row": int(rows[i]), "rank_score": float(ranks[i]), "votes": int(votes[i]),
                     "flagged_by": [k for k, f in flags.items() if f[i]],
                     "driving_columns": {num[j]: float(std[i, j]) for j in np.argsort(-z[i])[:3]}} for i in top]
        pcs = np.linalg.svd(std - std.mean(0), full_matrices=False)
        xy = (std - std.mean(0)) @ pcs[2][:2].T
        self._outlier_viz = {"xy": xy, "iso": iso, "consensus": consensus, "votes": votes,
                             "flags": {k: int(v.sum()) for k, v in flags.items()}}
        rows_g, cols_g = som.shape_
        order = np.argsort(-som.component_planes().reshape(rows_g * cols_g, -1).var(0))[:3]
        self._som_viz = {"u": som.u_matrix(), "hits": som.hits_, "unit_qe": som.unit_quantization_error(),
                         "planes": [(num[j], som.component_planes()[:, :, j]) for j in order]}
        return {
            "rows_analysed": int(len(rows)), "sampled": bool(n > len(rows)),
            "methods": {"isolation_forest": {"n_trees": 100, "sample_size": min(256, len(std)),
                                             "flagged": int(flags["isolation_forest"].sum()),
                                             "rule": "robust z of score > 3.5"},
                        "lof": {"k": min(20, len(std) - 1), "flagged": int(flags["lof"].sum()), "rule": "LOF > 1.5"},
                        "mahalanobis": {"flagged": int(flags["mahalanobis"].sum()), "rule": "chi-square 0.999"},
                        "som_quantization": {"grid": list(som.shape_), "flagged": int(flags["som_quantization"].sum()),
                                             "rule": "robust z of quantization error > 3.5"}},
            "consensus": {"rule": "flagged by at least 2 of 4 methods", "n_rows": int(consensus.sum()),
                          "fraction": float(consensus.mean()),
                          "row_positions": rows[consensus][:50].tolist(), "top_examples": examples},
            "single_detector_only": {k: int((f & (votes == 1)).sum()) for k, f in flags.items()},
            "som": {"grid": list(som.shape_), "quantization_error": som.quantization_error_,
                    "topographic_error": som.topographic_error_,
                    "empty_units": int((som.hits_ == 0).sum()),
                    "u_matrix": som.u_matrix().round(4).tolist(), "hits": som.hits_.tolist()},
        }

    def _clusters(self, df, num):
        if len(num) < 2 or len(df) < 30:
            return {"status": "skipped: needs at least 2 numeric columns and 30 rows"}
        Z = df[num].apply(lambda s: s.fillna(s.median())).to_numpy(float)
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 1e-12, Z.std(0), 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(variant="stat_model_selection", criterion="silhouette", k_range=(2, 6),
                        random_state=self.random_state).fit(Z)
        labels = km.labels_
        _, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        self._cluster_viz = {"xy": Z @ Vt[:2].T, "labels": labels, "explained": (S ** 2 / (S ** 2).sum())[:2]}
        raw = df[num].to_numpy(float)
        grand_mean, grand_std = np.nanmean(raw, 0), np.nanstd(raw, 0) + 1e-12
        clusters = []
        for k in np.unique(labels):
            m = labels == k
            z = (np.nanmean(raw[m], 0) - grand_mean) / grand_std
            top = np.argsort(-np.abs(z))[:3]
            clusters.append({"cluster": int(k), "size": int(m.sum()), "share": float(m.mean()),
                             "signature": {num[i]: float(z[i]) for i in top}})
        return {"n_clusters": len(clusters), "silhouette": self._silhouette(Z, labels), "clusters": clusters,
                "note": "signature = cluster mean minus overall mean, in standard deviations"}

    @staticmethod
    def _silhouette(Z, labels, cap=1000):
        idx = np.arange(len(Z)) if len(Z) <= cap else np.random.default_rng(0).choice(len(Z), cap, replace=False)
        Zs, ls = Z[idx], labels[idx]
        d = np.sqrt(np.maximum((Zs ** 2).sum(1)[:, None] + (Zs ** 2).sum(1)[None, :] - 2 * Zs @ Zs.T, 0))
        sc = []
        for i in range(len(Zs)):
            same = ls == ls[i]
            if same.sum() < 2:
                continue
            a = d[i, same].sum() / (same.sum() - 1)
            b = min(d[i, ls == k].mean() for k in np.unique(ls) if k != ls[i])
            sc.append((b - a) / max(a, b))
        return float(np.mean(sc)) if sc else 0.0

    def _what_then(self, df, time_column, target_column, features):
        t = pd.to_datetime(df[time_column], errors="coerce")
        if t.notna().sum() < 8:
            return {"status": f"'{time_column}' is not a usable datetime column"}
        order = np.argsort(t.to_numpy())
        cols = [c for c in ([target_column] if target_column else []) + features
                if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])][:4]
        out = {"time_column": time_column, "series": {}}
        for c in cols:
            v = df[c].to_numpy(float)[order]
            ok = np.isfinite(v)
            if ok.sum() < 8 or v[ok].std() == 0:
                continue
            x = np.arange(len(v))[ok]
            lr = stats.linregress(x, v[ok])
            res = {"trend_slope_per_step": float(lr.slope), "r_squared": float(lr.rvalue ** 2),
                   "p_value": float(lr.pvalue), "direction": "increasing" if lr.slope > 0 else "decreasing",
                   "next_step_forecast": float(lr.intercept + lr.slope * (len(v))),
                   "significant_trend": bool(lr.pvalue < 0.05)}
            try:
                from statsmodels.tsa.stattools import adfuller
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adf = adfuller(v[ok], autolag="AIC")
                res["adf"] = {"statistic": float(adf[0]), "p_value": float(adf[1]), "stationary": bool(adf[1] < 0.05)}
            except Exception as exc:
                res["adf"] = {"error": str(exc)}
            detr = v[ok] - (lr.intercept + lr.slope * x)
            nlag = min(len(detr) // 3, 60)
            if nlag >= 3 and detr.std() > 1e-9 * max(1.0, abs(v[ok]).max()):
                ac = np.array([np.corrcoef(detr[:-k], detr[k:])[0, 1] for k in range(1, nlag + 1)])
                best = int(np.nanargmax(ac[1:])) + 2
                res["seasonality"] = {"peak_lag": best, "autocorrelation": float(ac[best - 1]),
                                      "seasonal": bool(ac[best - 1] > 0.4)}
            out["series"][c] = res
        return out

    def _recommendations(self, d):
        recs = []
        haz = d.get("what_happened", {}).get("hazards", {})
        for k, label in (("constant", "constant"), ("quasi_constant", "quasi-constant"),
                         ("high_cardinality", "high-cardinality")):
            if haz.get(k):
                recs.append(f"Review {label} columns: {', '.join(haz[k])}.")
        prof = d.get("what_happened", {}).get("column_profiles", {})
        miss = [c for c, p in prof.items() if p["missing_percentage"] > 20]
        if miss:
            recs.append("More than 20% missing in: " + ", ".join(miss) + "; impute or drop deliberately.")
        skew = [c for c, p in prof.items() if abs(p.get("skewness", 0)) > 2]
        if skew:
            recs.append("Heavily skewed columns (|skew|>2), consider a log or rank transform: " + ", ".join(skew) + ".")
        pairs = d.get("why", {}).get("strong_pairs", [])
        if pairs:
            recs.append(f"{len(pairs)} strongly related pairs (|r|>=0.7); check redundancy before modelling.")
        mv = (d.get("why", {}).get("outliers") or {}).get("multivariate")
        if mv and mv["fraction"] > 0.02:
            recs.append(f"{mv['fraction']:.1%} of rows are multivariate outliers; inspect before training.")
        cons = ((d.get("why", {}).get("outliers") or {}).get("ensemble") or {}).get("consensus")
        if cons and cons["n_rows"]:
            recs.append(f"{cons['n_rows']} rows ({cons['fraction']:.1%}) are flagged by at least two of isolation "
                        "forest, LOF, Mahalanobis and SOM: they are either errors or the interesting cases; inspect them.")
        for c, s in (d.get("what_then", {}).get("series") or {}).items():
            if s.get("significant_trend"):
                recs.append(f"'{c}' trends {s['direction']} over time; use time-aware validation.")
            if s.get("adf", {}).get("stationary") is False:
                recs.append(f"'{c}' is non-stationary (ADF); difference or detrend before time-series models.")
        if d.get("target_intelligence"):
            recs += d["target_intelligence"]["recommendations"]
        if d.get("feature_intelligence"):
            recs += d["feature_intelligence"]["recommendations"]
        return recs or ["No structural problems detected."]

    @staticmethod
    def _score(d):
        prof = d.get("what_happened", {}).get("column_profiles", {})
        score = 100.0
        if prof:
            score -= min(15.0, float(np.mean([p["missing_percentage"] for p in prof.values()])))
        haz = d.get("what_happened", {}).get("hazards", {})
        score -= min(15.0, 3.0 * sum(len(v) for v in haz.values()))
        score -= min(10.0, 2.0 * len(d.get("why", {}).get("strong_pairs", [])))
        mv = (d.get("why", {}).get("outliers") or {}).get("multivariate")
        if mv:
            score -= min(10.0, 100.0 * mv["fraction"])
        parts = [score]
        for key in ("target_intelligence", "feature_intelligence"):
            if d.get(key):
                parts.append(d[key]["summary_score"])
        return float(np.clip(np.mean(parts) if len(parts) > 1 else score, 0, 100))

    @staticmethod
    def _narrative(d, score):
        ov = d["overview"]
        lines = [f"The dataset has {ov['rows']} rows and {ov['n_features']} features; overall reliability score {score:.0f}/100."]
        why = d.get("why", {})
        if why.get("drivers"):
            top = why["drivers"][0]
            lines.append(f"The strongest driver of '{why['investigated_field']}' is '{top['feature']}' "
                         f"(r={top['pearson']:+.2f}); association is not causation.")
        if why.get("clusters", {}).get("n_clusters"):
            lines.append(f"{why['clusters']['n_clusters']} sub-populations were found (silhouette {why['clusters']['silhouette']:.2f}).")
        ti = d.get("target_intelligence")
        if ti:
            lines.append(f"Target check: {ti['explainability']['verdict']}; {len(ti['leakage']['flags'])} leakage flag(s).")
        lines.append(f"{len(d['recommendations'])} recommendation(s) follow.")
        return " ".join(lines)

    def _llm(self, d, score, question):
        if self.llm_backend is None:
            return {"llm_status": "no llm_backend configured; rule-based narrative only"}
        summary = json.dumps({"score": score, "overview": d["overview"], "recommendations": d["recommendations"],
                              "questions": d["questions"], "narrative": d["narrative"]}, default=_json_default)[:6000]
        prompt = ("You are a careful senior data scientist. From this statistical summary write a short executive "
                  f"summary and the three most important next steps. {('Answer this question too: ' + question) if question else ''}\n"
                  + summary)
        try:
            return {"llm_synthesis": self._call_llm(prompt), "llm_status": "ok"}
        except Exception as exc:
            return {"llm_status": f"backend failed: {exc}"}

    def _call_llm(self, prompt):
        b = self.llm_backend
        if callable(b):
            return str(b(prompt))
        import urllib.request
        if b == "ollama":
            url = (self.llm_url or "http://localhost:11434") + "/api/generate"
            body = {"model": self.llm_model, "prompt": prompt, "stream": False}
            key = "response"
        elif b == "custom_url":
            if not self.llm_url:
                raise ValueError("custom_url backend needs llm_url")
            url, body, key = self.llm_url, {"prompt": prompt, "model": self.llm_model}, None
        elif b == "scikit-llm":
            from skllm.config import SKLLMConfig
            from skllm.models.gpt.text2text.summarization import GPTSummarizer
            SKLLMConfig.set_openai_key(self.llm_api_key)
            return str(GPTSummarizer(model=self.llm_model).fit_transform([prompt])[0])
        else:
            raise ValueError(f"unknown llm_backend {b!r}")
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
        return str(payload[key] if key else payload.get("response") or payload.get("text") or payload)


# ============================================================================== figures
# Library house style (shared with visualization.py): seaborn whitegrid, colorblind palette.
def _panel_scorecard(ax, d, score, ctx, pal, **_):
    import textwrap

    ax.axis("off")
    colour = pal[2] if score >= 80 else (pal[1] if score >= 60 else pal[3])
    ax.text(0.0, 0.98, f"{score:.0f}", fontsize=54, color=colour, va="top", fontweight="bold")
    ax.text(0.42, 0.90, "/ 100\nreliability score", fontsize=11, va="top")
    ov = d["overview"]
    lines = [f"{ov['rows']} rows, {ov['n_features']} features, {ov['missing_cells']} missing cells"]
    ti = d.get("target_intelligence")
    if ti:
        lines += [f"target: {ti['explainability']['verdict']}", f"leakage flags: {len(ti['leakage']['flags'])}"]
    lines += ["Top recommendations:"] + ["- " + textwrap.shorten(r, 70) for r in d["recommendations"][:4]]
    ax.text(0.0, 0.58, "\n".join(textwrap.fill(x, 48) for x in lines), fontsize=8.5, va="top", linespacing=1.35)
    ax.set_title("Scorecard", fontsize=10)
    return True


def _panel_association(ax, d, score, ctx, pal, leakage_threshold, **_):
    ti = d.get("target_intelligence")
    if not ti:
        return False
    rows = ti["leakage"]["associations"][:10][::-1]
    y = np.arange(len(rows))
    ax.barh(y + 0.2, [r["strength"] for r in rows], height=0.38, color=pal[0], label="association")
    ax.barh(y - 0.2, [max(r["single_feature_skill"], 0) for r in rows], height=0.38, color=pal[1],
            label="single-feature skill")
    ax.axvline(leakage_threshold, color=pal[3], ls="--", lw=1.2, label=f"leakage threshold {leakage_threshold:g}")
    ax.set_yticks(y, [r["feature"] for r in rows], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_title("Target association and leakage screen", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    return True


def _panel_vif(ax, d, score, ctx, pal, vif_threshold, **_):
    mc = (d.get("target_intelligence") or {}).get("multicollinearity")
    vif = (mc or {}).get("vif")
    if not vif:
        return False
    items = sorted(vif.items(), key=lambda kv: kv[1])[-12:]
    ax.barh([k for k, _ in items], [v for _, v in items],
            color=[pal[3] if v > vif_threshold else pal[0] for _, v in items])
    ax.axvline(vif_threshold, color="k", ls="--", lw=1, label=f"VIF = {vif_threshold:g}")
    ax.set_xscale("log")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(f"Multicollinearity (condition number {mc['condition_number']:.0f})", fontsize=10)
    ax.legend(fontsize=7)
    return True


def _panel_explain(ax, d, score, ctx, pal, **_):
    ti = d.get("target_intelligence")
    if not ti:
        return False
    ex = ti["explainability"]
    null = ex.get("permutation_null") or []
    if null:
        ax.hist(null, bins=min(12, max(4, len(null) // 2)), color=pal[7], alpha=0.8, label="label-shuffled skill")
    best = max(ex["linear_skill"], ex["knn_skill"])
    ax.axvline(best, color=pal[2], lw=2.5, label=f"real skill {best:.2f}")
    ax.axvline(ex["linear_skill"], color=pal[0], ls="--", label=f"linear {ex['linear_skill']:.2f}")
    ax.axvline(ex["knn_skill"], color=pal[1], ls="--", label=f"KNN {ex['knn_skill']:.2f}")
    ax.set_xlabel("cross-validated skill")
    ax.set_title(f"Explainability: {ex['verdict']}", fontsize=10)
    ax.legend(fontsize=7)
    return True


def _panel_selection(ax, d, score, ctx, pal, **_):
    ti = d.get("target_intelligence")
    fs = (ti or {}).get("feature_selection") or {}
    if not fs.get("subsets"):
        return False
    sub = fs["subsets"]
    ax.scatter([r["n_features"] for r in sub], [r["fast_skill"] for r in sub], s=34, color=pal[0], alpha=0.8,
               label="subset tried")
    best = fs["best_features"]
    ax.scatter([len(best)], [fs["best_skill"]], s=150, marker="*", color=pal[2], label=f"chosen: {fs['best_method']}")
    ax.axhline(fs["all_features_skill"], color=pal[3], ls="--", lw=1.2, label="all features")
    ax.set_xlabel("number of features")
    ax.set_ylabel("skill")
    ax.set_title(f"Feature-selection experiments ({len(sub)} subsets)", fontsize=10)
    ax.legend(fontsize=7)
    return True


def _panel_targets(ax, d, score, ctx, pal, **_):
    cands = (d.get("target_intelligence") or {}).get("target_candidates") or []
    if len(cands) < 2:
        return False
    cands = cands[:12][::-1]
    y = np.arange(len(cands))
    ax.barh(y + 0.2, [c["linear_skill"] for c in cands], height=0.38, color=pal[0], label="linear")
    ax.barh(y - 0.2, [c["knn_skill"] for c in cands], height=0.38, color=pal[1], label="KNN")
    ax.set_yticks(y, [f"[{c['group'][:10]}] {c['name'][:26]}" for c in cands], fontsize=7)
    ax.set_title("Alternative targets (compare within a group)", fontsize=10)
    ax.legend(fontsize=7)
    return True


def _panel_correlation(ax, d, score, ctx, pal, **_):
    import seaborn as sns

    pear = (d.get("why") or {}).get("pearson")
    if not pear:
        return False
    corr = pd.DataFrame(pear)
    cols = list(corr.columns)[:14]
    corr = corr.loc[cols, cols]
    sns.heatmap(corr, cmap="vlag", vmin=-1, vmax=1, center=0, annot=len(cols) <= 10, fmt=".2f",
                annot_kws={"size": 7}, square=True, linewidths=0.4, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.tick_params(labelsize=7)
    ax.set_title("Correlation matrix (Pearson)", fontsize=10)
    return True


def _panel_clusters(ax, d, score, ctx, pal, cluster_viz, **_):
    if cluster_viz is None:
        return False
    xy, lab = cluster_viz["xy"], cluster_viz["labels"]
    for k in np.unique(lab):
        ax.scatter(xy[lab == k, 0], xy[lab == k, 1], s=14, alpha=0.7, color=pal[int(k) % len(pal)],
                   label=f"cluster {k} ({(lab == k).mean():.0%})")
    ev = cluster_viz["explained"]
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    cl = d["why"]["clusters"]
    ax.set_title(f"Sub-populations (silhouette {cl['silhouette']:.2f})", fontsize=10)
    ax.legend(fontsize=7)
    return True


def _panel_ranking(ax, d, score, ctx, pal, **_):
    import seaborn as sns

    fi = d.get("feature_intelligence")
    if not fi or not fi.get("ranking"):
        return False
    rows = fi["ranking"][:15]
    frame = pd.DataFrame([{m: r[m] for m in fi["methods"]} for r in rows], index=[r["feature"] for r in rows])
    sns.heatmap(frame, cmap="crest", vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax,
                cbar_kws={"label": "rank percentile", "shrink": 0.7})
    ax.tick_params(labelsize=7)
    ax.set_title("Feature ranking by method (aggregate order)", fontsize=10)
    return True


def _panel_outliers(ax, d, score, ctx, pal, **_):
    import seaborn as sns

    num = (d.get("why") or {}).get("numeric_columns") or []
    if not num:
        return False
    df = ctx["df"][num[:12]]
    sns.boxplot(data=(df - df.mean()) / (df.std() + 1e-12), orient="h", fliersize=2, linewidth=0.8, ax=ax,
                color=pal[0])
    ax.tick_params(labelsize=7)
    ax.set_title("Tails (standardised boxplots)", fontsize=10)
    return True


def _panel_missing(ax, d, score, ctx, pal, **_):
    import seaborn as sns

    df = ctx["df"]
    if not df.isna().any().any():
        return False
    sns.heatmap(df.isna().T, cbar=False, cmap=["#f2f2f2", pal[3]], ax=ax, xticklabels=False)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Missingness (dark = missing)", fontsize=10)
    return True


def _panel_trend(ax, d, score, ctx, pal, **_):
    series = (d.get("what_then") or {}).get("series") or {}
    if not series:
        return False
    col = ctx["target"] if ctx["target"] in series else next(iter(series))
    df = ctx["df"]
    t = pd.to_datetime(df[ctx["time"]], errors="coerce")
    order = np.argsort(t.to_numpy())
    v = df[col].to_numpy(float)[order]
    s = series[col]
    x = np.arange(len(v))
    ax.plot(t.to_numpy()[order], v, color=pal[0], lw=1, label=col)
    fit = (np.nanmean(v) - s["trend_slope_per_step"] * x.mean()) + s["trend_slope_per_step"] * x
    ax.plot(t.to_numpy()[order], fit, color=pal[3], ls="--", label=f"trend {s['trend_slope_per_step']:+.3g}/step")
    adf = s.get("adf", {})
    ax.set_title(f"{col} over time (R2 {s['r_squared']:.2f}, "
                 f"{'stationary' if adf.get('stationary') else 'non-stationary'})", fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.legend(fontsize=7)
    return True


def _distribution_figure(d, ctx, pal):
    import seaborn as sns
    from matplotlib.figure import Figure

    num = ((d.get("why") or {}).get("numeric_columns") or
           [c for c in ctx["features"] if pd.api.types.is_numeric_dtype(ctx["df"][c])])[:6]
    if not num:
        return None
    ncols = min(3, len(num))
    nrows = int(np.ceil(len(num) / ncols))
    fig = Figure(figsize=(5 * ncols, 3.6 * nrows), constrained_layout=True)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    prof = d.get("what_happened", {}).get("column_profiles", {})
    for i, c in enumerate(num):
        ax = axes[i // ncols][i % ncols]
        v = ctx["df"][c].dropna().to_numpy(float)
        sns.histplot(v, bins=30, stat="density", kde=True, color=pal[0], alpha=0.35, ax=ax)
        cdf = ax.twinx()
        sns.ecdfplot(x=v, ax=cdf, color=pal[1], linestyle="--")
        cdf.set_ylim(0, 1)
        cdf.set_ylabel("")
        cdf.grid(False)
        ax.axvline(v.mean(), color=pal[3], lw=1.3)
        p = prof.get(c, {})
        norm = p.get("normality", {})
        ax.set_title(f"{c}  skew {p.get('skewness', 0):+.2f}"
                     f"{'  ~normal' if norm.get('looks_normal') else ''}", fontsize=9)
        ax.set_xlabel("")
    for j in range(len(num), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Distributions: density + KDE, dashed ECDF (right axis), solid line = mean", fontsize=10)
    return fig


def _outlier_figure(d, ctx, pal):
    from matplotlib.figure import Figure

    viz = ctx.get("outlier_viz")
    ens = ((d.get("why") or {}).get("outliers") or {}).get("ensemble")
    if not viz or not ens:
        return None
    fig = Figure(figsize=(16, 4.8), constrained_layout=True)
    a, b, c = fig.subplots(1, 3)
    xy, iso, cons = viz["xy"], viz["iso"], viz["consensus"]
    sc = a.scatter(xy[:, 0], xy[:, 1], c=iso, cmap="viridis_r", s=14, alpha=0.8)
    a.scatter(xy[cons, 0], xy[cons, 1], s=70, facecolors="none", edgecolors=pal[3], linewidths=1.4,
              label=f"consensus outliers ({int(cons.sum())})")
    fig.colorbar(sc, ax=a, label="isolation score (high = isolated)", shrink=0.8)
    a.set_title("Where the odd rows sit (PCA plane)", fontsize=10)
    a.set_xlabel("PC1")
    a.set_ylabel("PC2")
    a.legend(fontsize=7)
    names = {"isolation_forest": "isolation\nforest", "lof": "LOF", "mahalanobis": "Mahalanobis",
             "som_quantization": "SOM\nquant. error"}
    counts = [viz["flags"][k] for k in names]
    b.bar(list(names.values()), counts, color=pal[0])
    b.bar(["consensus\n(>=2 votes)"], [int(cons.sum())], color=pal[3])
    b.set_ylabel("rows flagged")
    b.set_title("Detector agreement", fontsize=10)
    b.tick_params(axis="x", labelsize=8)
    votes = np.bincount(viz["votes"], minlength=5)
    c.bar([str(i) for i in range(len(votes))], votes, color=[pal[0]] * 2 + [pal[3]] * (len(votes) - 2))
    c.set_yscale("symlog")
    c.set_xlabel("number of detectors that flag the row")
    c.set_ylabel("rows (log scale)")
    c.set_title("Votes per row", fontsize=10)
    fig.suptitle("Outlier consensus: isolation forest, LOF, robust Mahalanobis and SOM", fontsize=11)
    return fig


def _som_figure(d, ctx, pal):
    import seaborn as sns
    from matplotlib.figure import Figure

    viz = ctx.get("som_viz")
    som = (((d.get("why") or {}).get("outliers") or {}).get("ensemble") or {}).get("som")
    if not viz or not som:
        return None
    fig = Figure(figsize=(15, 8.6), constrained_layout=True)
    axes = fig.subplots(2, 3).ravel()
    sns.heatmap(viz["u"], cmap="magma_r", ax=axes[0], cbar_kws={"label": "mean neighbour distance"})
    axes[0].set_title("U-matrix (ridges = cluster borders)", fontsize=10)
    sns.heatmap(viz["hits"], cmap="Blues", annot=viz["hits"].shape[0] <= 10, fmt="d", annot_kws={"size": 7},
                ax=axes[1], cbar_kws={"label": "rows"})
    axes[1].set_title("Hit map (empty cells = gaps in the data)", fontsize=10)
    sns.heatmap(viz["unit_qe"], cmap="rocket_r", ax=axes[2], cbar_kws={"label": "mean quantization error"})
    axes[2].set_title("Where the map fits poorly (anomaly zones)", fontsize=10)
    for ax, (name, plane) in zip(axes[3:], viz["planes"]):
        sns.heatmap(plane, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "standardised value"})
        ax.set_title(f"Component plane: {name}", fontsize=10)
    for ax in axes[3 + len(viz["planes"]):]:
        ax.set_visible(False)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Self-organising map {som['grid'][0]}x{som['grid'][1]}   quantization error "
                 f"{som['quantization_error']:.3f}   topographic error {som['topographic_error']:.3f}", fontsize=11)
    return fig


def _build_figures(d, score, ctx, cluster_viz, leakage_threshold, vif_threshold):
    from matplotlib.figure import Figure

    from .visualization import apply_house_style

    pal = apply_house_style()
    kw = dict(cluster_viz=cluster_viz, leakage_threshold=leakage_threshold, vif_threshold=vif_threshold)
    panels = {
        "association": _panel_association, "vif": _panel_vif, "explainability": _panel_explain,
        "feature_selection": _panel_selection, "target_candidates": _panel_targets,
        "correlation": _panel_correlation, "clusters": _panel_clusters, "feature_ranking": _panel_ranking,
        "tails": _panel_outliers, "missingness": _panel_missing, "trend": _panel_trend,
    }
    figures: dict[str, Any] = {}
    dist = _distribution_figure(d, ctx, pal)
    if dist is not None:
        figures["distributions"] = dist
    for extra, fn in (("outlier_consensus", _outlier_figure), ("som", _som_figure)):
        fig = fn(d, ctx, pal)
        if fig is not None:
            figures[extra] = fig
    for name, fn in panels.items():
        fig = Figure(figsize=(7, 4.6), constrained_layout=True)
        if fn(fig.subplots(), d, score, ctx, pal, **kw):
            figures[name] = fig
    order = ["scorecard", "association", "explainability", "vif", "feature_selection", "correlation", "clusters",
             "feature_ranking", "target_candidates", "tails", "trend", "missingness"]
    all_panels = {"scorecard": _panel_scorecard, **panels}
    dash = Figure(figsize=(17, 9.5), constrained_layout=True)
    grid = dash.add_gridspec(2, 3)
    drawn = 0
    for name in order:
        if drawn == 6:
            break
        ax = dash.add_subplot(grid[drawn // 3, drawn % 3])
        if all_panels[name](ax, d, score, ctx, pal, **kw):
            drawn += 1
        else:
            ax.remove()
    dash.suptitle(f"ReliableML Data Intelligence dashboard  |  mode: {d['mode']}", fontsize=13)
    figures = {"dashboard": dash, **figures}
    return figures
