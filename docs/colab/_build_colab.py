"""Builds the StatCore Colab notebooks (plain-json nbformat 4, no extra dependencies).

Run from anywhere:  python docs/colab/_build_colab.py
Each notebook mirrors docs/latex/statcore-<name>.tex and docs/latex/experiments/*_experiment.py
with a reduced seed count so it finishes in a few minutes on Colab.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = "https://github.com/Jabualani/ML-Reliabilty"
INSTALL = 'pip install "git+https://github.com/Jabualani/ML-Reliabilty.git"'

_counter = [0]


def _src(text: str) -> list[str]:
    text = text.strip("\n")
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def _id() -> str:
    _counter[0] += 1
    return f"cell-{_counter[0]:03d}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": _src(text)}


def code(text: str, tags: list[str] | None = None) -> dict:
    meta = {"tags": tags} if tags else {}
    return {"cell_type": "code", "id": _id(), "metadata": meta, "execution_count": None,
            "outputs": [], "source": _src(text)}


def write(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"name": f"statcore-{name}.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUT / f"statcore-{name}.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)


def header(name: str, title: str, topic: str, seeds: str, extra_param: str = "") -> list[dict]:
    """Title, install (parameter) cells and the imports/parameters cell shared by every notebook."""
    _counter[0] = 0
    return [
        md(f"""
# {title}

[Open in Colab](https://colab.research.google.com/github/Jabualani/ML-Reliabilty/blob/main/docs/colab/statcore-{name}.ipynb)

**Paper topic:** {topic} — `docs/latex/statcore-{name}.tex` ([repository]({REPO}/blob/main/docs/latex/statcore-{name}.tex)).
**Experiment script reproduced here:** `docs/latex/experiments/{name}_experiment.py` (reduced seed count so the notebook runs in a few minutes).

StatCore is a pure-NumPy/SciPy model family in `ml_reliability.statcore`. Every model has a `variant`
argument; each variant below is compared with the standard reference implementation.
**scikit-learn is used only as the baseline/reference** (and for loading public datasets), never inside StatCore.
"""),
        md(f"""
## 1. Install (parameter cell)

> **Note:** StatCore has **not been pushed / published yet**. Until the repository is public, the
> `git+https` install below will fail. In that case use the **fallback cell**: zip the repository folder,
> upload the zip to Colab (left sidebar → Files, or the `files.upload()` helper) and `pip install` it.
> Run exactly one of the two cells.
"""),
        code(f"""
# @title Install StatCore (PARAMETER: change the source if needed)
# Option A - once the repository is published:
!{INSTALL}
""", tags=["parameters", "install"]),
        code("""
# @title FALLBACK install from an uploaded zip of the repository (use if Option A fails)
# Zip the repo folder (ML-Reliabilty.zip), then uncomment and run:
# from google.colab import files
# files.upload()                       # pick ML-Reliabilty.zip
# !unzip -q -o ML-Reliabilty.zip -d repo
# !pip install ./repo/ML-Reliabilty     # adjust to the folder that contains pyproject.toml
""", tags=["install"]),
        code(f"""
# @title Parameters and imports
SEEDS = {seeds}   # the paper uses more seeds; fewer here to keep the run short
{extra_param}
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
""", tags=["parameters"]),
    ]


def summary_md(text: str) -> dict:
    return md(text)


# ---------------------------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------------------------
def build_pca() -> None:
    c = header("pca", "StatCore PCA: reliability-oriented principal component analysis",
               "Reliability-Oriented Principal Component Analysis in StatCore", "(11, 23, 37)")
    c += [
        md(r"""
## 2. Method summary

Notation: $X\in\mathbb{R}^{n\times d}$, $k$ components, centred data $X_c = X-\mu$, weights $w_i\ge 0$.
`transform` returns $Z=(x-\mu)\,V_k^{+}$ and `inverse_transform` returns $\hat x = z\,V_k+\mu$, where $V_k\in\mathbb{R}^{k\times d}$ holds the components as rows (`components_`).

**Reference / `base`.** Thin SVD $X_c=U\Sigma V^\top$, $V_k$ = first $k$ right singular vectors, $\mu=\bar x$. This minimises the *training* squared reconstruction error $\|X_c-X_cV_k^\top V_k\|_F^2$.

**`supervised_row_aware`** (weighted PCA). With scores $s_i=V_k(x_i-\bar x)$ from a plain SVD and $\lambda_j=\sigma_j^2/n$:
$$d_i^2=\sum_{j\le k}\frac{s_{ij}^2}{\lambda_j},\qquad c=2.5\,\mathrm{median}(d^2),\qquad w_i=\begin{cases}1 & d_i^2\le c\\ c/d_i^2 & d_i^2>c\end{cases}$$
If labels are given, $w_i\leftarrow w_i\,\dfrac{n}{C\,n_{y_i}}$ (inverse class frequency). Then $w\leftarrow nw/\sum_i w_i$, the mean is $\mu_w=\sum_i w_ix_i/\sum_i w_i$ and components are the right singular vectors of $\mathrm{diag}(\sqrt w)(X-\mu_w)$.

**`robust`** (median start, Huber-type IRLS on orthogonal residuals). $\mu_0=\mathrm{median}(X)$; initial scaled norms $r_i=\|(x_i-\mu_0)/(1.4826\,\mathrm{MAD}_j)\|$, $c_0=\mathrm{med}(r)+1.345\cdot1.4826\,\mathrm{MAD}(r)$, $w_i=\min(1,c_0/r_i)^2$. For at most 10 iterations: weighted mean $\mu_w$, weighted SVD, residual norms $r_i=\|(x_i-\mu_w)-(x_i-\mu_w)V_k^\top V_k\|$, $c=\mathrm{med}(r)+1.345\cdot1.4826\,\mathrm{MAD}(r)$, $w_i=\min(1,c/r_i)$. *Implementation detail:* the stored `mean_` used by `transform` is the coordinate-wise **median** $\mu_0$.

**`sparse`** (L1 power iterations with deflation). For each component, start from the leading singular vector of the residual matrix $R$ (plus tiny noise) and iterate
$u=Rv/\|Rv\|,\; z=R^\top u,\; v\leftarrow \mathrm{soft}(z,\alpha\,\|z\|_\infty)/\|\cdot\|$, then deflate $R\leftarrow R-(Rv)v^\top$. Components are sparse but **not orthogonal**, so `transform` uses the pseudo-inverse of the component matrix.

**`probabilistic`** (closed-form PPCA, Tipping & Bishop). With eigenvalues $\ell_1\ge\dots\ge\ell_d$ of the sample covariance:
$$\sigma^2=\frac{1}{d-k}\sum_{j>k}\ell_j,\qquad W=V_k^\top(\Lambda_k-\sigma^2I)^{1/2},\qquad M=W^\top W+\sigma^2I,\qquad z=M^{-1}W^\top(x-\mu).$$
Reconstruction is the orthogonal projection onto $\mathrm{span}(W)$: $\hat x=\mu+zMW^{+}$.

Because the reference PCA is the *exact minimiser of training reconstruction error*, no variant can beat it in-sample on clean data; variants trade a little clean accuracy for robustness to contamination, class balance, sparsity or a noise model.
"""),
        md("## 3. Experiment (mirrors `pca_experiment.py`)\n\nHeld-out reconstruction MSE with $k=2$. *synthetic*: rank-2 latent data plus noise, **24 gross outliers injected into the training set only** (test rows are clean). *wine*: 75/25 split of `load_wine`. All data are standardised with training statistics."),
        code("""
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA as ReferencePCA      # baseline only
from ml_reliability.statcore import PCA
from ml_reliability.statcore.pca.pca import VARIANTS

METHODS = ("reference",) + tuple(VARIANTS)

def standardize(train, test):
    mean = train.mean(0); scale = train.std(0); scale[scale <= 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale

def synthetic(seed):
    rng = np.random.default_rng(seed); latent = rng.normal(size=(400, 2))
    loading = np.array([[1.5, 0], [0, 1.2], [.8, .3], [0, .7], [.4, -.4], [.2, .5]])
    clean = latent @ loading.T + .12 * rng.normal(size=(400, 6)); order = rng.permutation(400)
    tr, te = order[:320], order[320:]; train = clean[tr].copy()
    idx = rng.choice(len(train), 24, replace=False); train[idx] += rng.normal(0, 8, (24, 6))
    xtr, xte = standardize(train, clean[te]); y = (latent[tr, 0] > 0).astype(int)
    return xtr, xte, y

def wine(seed):
    data = load_wine(); order = np.random.default_rng(seed).permutation(len(data.data)); split = int(.75 * len(order))
    tr, te = order[:split], order[split:]; xtr, xte = standardize(data.data[tr], data.data[te])
    return xtr, xte, data.target[tr]

def fit_method(method, xtr, y, seed):
    if method == "reference":
        return ReferencePCA(n_components=2, svd_solver="full").fit(xtr)
    return PCA(n_components=2, variant=method, random_state=seed).fit(xtr, y=y)

def score(method, xtr, xte, y, seed):
    model = fit_method(method, xtr, y, seed)
    recon = model.inverse_transform(model.transform(xte))
    return float(np.mean((xte - recon) ** 2))

rows = []
for ds, factory in (("synthetic", synthetic), ("wine", wine)):
    for method in METHODS:
        for seed in SEEDS:
            rows.append(dict(dataset=ds, method=method, seed=seed, mse=score(method, *factory(seed), seed)))
res = pd.DataFrame(rows)
tab = res.groupby(["dataset", "method"], sort=False)["mse"].agg(["mean", "std"])
tab["mean +/- sd"] = tab.apply(lambda r: f"{r['mean']:.4f} +/- {r['std']:.4f}", axis=1)
print(f"Held-out reconstruction MSE over seeds {SEEDS}")
tab[["mean +/- sd"]]
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, ds in zip(axes, ("synthetic", "wine")):
    t = tab.loc[ds].loc[list(METHODS)]
    ax.bar(range(len(t)), t["mean"], yerr=t["std"], capsize=3, color=["gray"] + ["C0"] * (len(t) - 1))
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t.index, rotation=35, ha="right")
    ax.set_title(f"{ds}: held-out reconstruction MSE (lower is better)"); ax.set_ylabel("MSE")
plt.tight_layout(); plt.show()
"""),
        md("## 4. Look inside each variant\n\nOne fit per variant on the contaminated synthetic training set: explained variance, sparsity of the loadings, PPCA noise variance and how strongly the weighted variants down-weight rows."),
        code("""
xtr, xte, y = synthetic(SEEDS[0])
info = []
models = {m: fit_method(m, xtr, y, SEEDS[0]) for m in METHODS}
for m, model in models.items():
    comps = model.components_
    row = dict(method=m, expl_var_ratio=np.round(model.explained_variance_ratio_, 3).tolist(),
               zero_loadings=int(np.sum(np.abs(comps) < 1e-12)),
               orthonormal_err=float(np.abs(comps @ comps.T - np.eye(2)).max()))
    row["noise_var"] = round(float(getattr(model, "noise_variance_", np.nan)), 4) if m == "probabilistic" else np.nan
    w = getattr(model, "row_weights_", None)
    row["min_row_weight"] = float(w.min()) if w is not None else np.nan
    row["rows_downweighted"] = int(np.sum(w < 0.999)) if w is not None else 0
    info.append(row)
pd.DataFrame(info).set_index("method")
"""),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, m in zip(axes, ("reference", "supervised_row_aware", "robust")):
    Z = models[m].transform(xtr)
    ax.scatter(Z[:, 0], Z[:, 1], c=y, s=10, cmap="coolwarm"); ax.set_title(f"{m}: training scores")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.tight_layout(); plt.show()

# per-sample reconstruction error as an outlier / reliability score (StatCore-only method)
err = models["robust"].score_samples(xtr)
print("robust PCA: mean per-sample error, top-5% rows vs rest ->",
      round(float(err[err >= np.quantile(err, .95)].mean()), 3), "vs", round(float(err[err < np.quantile(err, .95)].mean()), 3))
"""),
        md("## 5. Summary and limitations"),
        code("""
best = tab.reset_index().sort_values("mean").groupby("dataset").first()[["method", "mean"]]
ref = tab.xs("reference", level="method")["mean"]
print("Best method per dataset (lowest held-out MSE):")
print(best.assign(reference_mse=ref, ratio_to_reference=best["mean"] / ref).round(4))
"""),
        md("""
**What to read from the table.** On the contaminated synthetic set, estimators that limit the influence of the 24 training outliers (`robust`, `supervised_row_aware`) should reconstruct the *clean* test rows better than the reference; on clean real data (wine) the reference is essentially optimal and the variants are equal or worse.

**Honest limitations**
- Only 2 datasets, $k=2$, and a few seeds here (5 in the paper): differences within one standard deviation are not evidence of a real gap.
- The reference PCA is the optimal in-sample reconstruction; variants improve *robustness*, not clean-data error.
- `sparse` components are non-orthogonal (max deviation ≈ 1e-2 here) and its reconstruction is essentially the reference's (marginally worse on wine); the threshold `alpha` is not tuned and it is **not** robust to the injected outliers. `probabilistic` has the same subspace as the reference (so identical reconstruction MSE up to numerical error) and adds a noise model; it is not a better reconstructor.
- On clean wine, `robust` is the worst variant and `supervised_row_aware` is only ~0.5 % better than the reference — within one SD.
- `robust` stores the coordinate-wise median as the centring used by `transform`, while the components come from the weighted mean — a deliberate implementation detail, not a theorem.
- Outlier contamination in this study is synthetic and gross; subtle contamination, missing data and large $d$ were not studied.
"""),
    ]
    write("pca", c)


# ---------------------------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------------------------
def build_lda() -> None:
    c = header("lda", "StatCore LDA: reliability-oriented discriminant analysis",
               "Reliability-Oriented Linear Discriminant Analysis in StatCore", "(11, 23, 37)")
    c += [
        md(r"""
## 2. Method summary

Data $(x_i,y_i)$, classes $k=1..C$, sample weights $w_i$ ($=1$ except `shift_corrected`), class weight $W_k=\sum_{i:y_i=k}w_i$.
Priors $\pi_k=W_k/\sum W$, means $\mu_k=\sum_{i:y_i=k}w_ix_i/W_k$, pooled within-class covariance
$$\Sigma=\frac{\sum_i w_i\,(x_i-\mu_{y_i})(x_i-\mu_{y_i})^\top}{\sum_i w_i}+\texttt{reg}\cdot\frac{\mathrm{tr}\,\Sigma}{d}I.$$
Linear score (all variants except `heteroscedastic`): $\delta_k(x)=x^\top\Sigma^{-1}\mu_k-\tfrac12\mu_k^\top\Sigma^{-1}\mu_k+\log\pi_k$; probabilities are $\mathrm{softmax}_k\,\delta_k(x)$.

**`base`** — the classic pooled LDA above (reference: sklearn `LinearDiscriminantAnalysis(solver="lsqr")`).

**`shift_corrected`** (covariate shift). A logistic domain classifier $g(x)=P(\text{target}\mid x)$ on standardised features (L2 = 1, Newton) gives
$$w_i=\min\!\Big(\tfrac{g(x_i)}{1-g(x_i)}\cdot\tfrac{n_s}{n_t},\,20\Big),\qquad w\leftarrow w\,n_s/\textstyle\sum w,$$
then weighted priors, means and covariance. `fit(X, y, X_target=...)` requires unlabeled target features.

**`shrinkage`** — $\Sigma_s=(1-s)\Sigma+s\,\frac{\mathrm{tr}\Sigma}{d}I$ with the analytic Ledoit–Wolf intensity $s=\min(\hat b^2,\hat\delta^2)/\hat\delta^2\in[0,1]$ (or fixed `shrinkage`).

**`sparse`** (nearest shrunken centroids). With $s_j=\sqrt{\Sigma_{jj}}$, $s_0=\mathrm{median}_j\,s_j$, $m_k=\sqrt{1/n_k-1/n}$:
$$d_{kj}=\frac{\mu_{kj}-\bar x_j}{m_k(s_j+s_0)},\quad d'_{kj}=\mathrm{sign}(d_{kj})\,(|d_{kj}|-\Delta)_+,\quad \mu'_{kj}=\bar x_j+m_k(s_j+s_0)d'_{kj},$$
with diagonal covariance $\mathrm{diag}(\Sigma)$; features with all $d'_{kj}=0$ are discarded (`selected_features_`).

**`heteroscedastic`** (regularised QDA). $\Sigma_k=(1-\rho)\hat\Sigma_k+\rho\Sigma_{\text{pooled}}+10^{-8}I$ with `reg_param`$=\rho$, and
$\delta_k(x)=-\tfrac12\big[(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)+\log|\Sigma_k|\big]+\log\pi_k$.
"""),
        md("## 3. Experiment (mirrors `lda_experiment.py`)\n\nSeven cases: `clean`, covariate `shifted` test set (the `shift_corrected` variant gets an unlabeled sample from the shifted domain; for other cases it receives a copy of the training features), 15 % `label_noise`, 8 % gross `outliers`, `high_dim` ($d=80$, $n=120$), `heteroscedastic` class scales, and the real `wine` data. Metrics: accuracy, log-loss and a 10-bin confidence ECE."),
        code("""
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis   # baseline only
from ml_reliability.statcore import LDA
from ml_reliability.statcore.lda.lda import VARIANTS

CASES = ("clean", "shifted", "label_noise", "outliers", "high_dim", "heteroscedastic", "wine")

def stratified_split(y, fraction, seed):
    rng = np.random.default_rng(seed); tr, te = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls); rng.shuffle(idx)
        n_test = min(max(1, int(round(len(idx) * fraction))), len(idx) - 1)
        te.extend(idx[:n_test]); tr.extend(idx[n_test:])
    return np.array(sorted(tr)), np.array(sorted(te))

def synthetic(case, seed):
    rng = np.random.default_rng(seed)
    ntr, nte = (120, 300) if case == "high_dim" else (360, 600)
    d = 80 if case == "high_dim" else 12
    informative = 10 if case == "high_dim" else min(8, d)
    classes = 3
    centers = np.zeros((classes, d))
    centers[:, :informative] = np.array([
        np.r_[-2.0, 0.0, np.zeros(informative - 2)],
        np.r_[2.0, 0.0, np.zeros(informative - 2)],
        np.r_[0.0, 2.2, np.zeros(informative - 2)]])
    scale = np.ones((classes, d))
    if case == "heteroscedastic":
        scale[0, :informative] = 0.55; scale[1, :informative] = 1.8
    shift = np.zeros(d)
    if case == "shifted":
        shift[:informative] = np.linspace(0.5, 1.1, informative)
    def sample(n, offset):
        y = rng.integers(classes, size=n)
        return centers[y] + rng.normal(size=(n, d)) * scale[y] + offset, y
    xtr, ytr = sample(ntr, np.zeros(d)); xte, yte = sample(nte, shift)
    if case == "label_noise":
        noisy = rng.random(ntr) < 0.15
        ytr[noisy] = (ytr[noisy] + rng.integers(1, classes, noisy.sum())) % classes
    elif case == "outliers":
        out = rng.random(ntr) < 0.08
        xtr[out] = rng.normal(0, 6, size=(out.sum(), d))
    xadapt, _ = sample(500, shift)
    return xtr, ytr, xte, yte, (xadapt if case == "shifted" else None)

def dataset(case, seed):
    if case != "wine":
        return synthetic(case, seed)
    ds = load_wine(); tr, te = stratified_split(ds.target, 0.25, seed)
    return ds.data[tr], ds.target[tr], ds.data[te], ds.target[te], None

def metrics(y, p):
    pred = p.argmax(axis=1); acc = float(np.mean(pred == y))
    logloss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)).mean())
    conf = p.max(axis=1); correct = pred == y; edges = np.linspace(0, 1, 11); ece = 0.0
    for j in range(10):
        m = (conf >= edges[j]) & (conf < edges[j + 1] if j < 9 else conf <= edges[j + 1])
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return dict(accuracy=acc, logloss=logloss, ece=float(ece))

rows = []
for case in CASES:
    for seed in SEEDS:
        xtr, ytr, xte, yte, xadapt = dataset(case, seed)
        ref = LinearDiscriminantAnalysis(solver="lsqr").fit(xtr, ytr)
        rows.append(dict(case=case, method="sklearn_LDA", seed=seed, **metrics(yte, ref.predict_proba(xte))))
        for variant in VARIANTS:
            model = LDA(variant=variant, random_state=seed)
            if variant == "shift_corrected":
                model.fit(xtr, ytr, X_target=xadapt if xadapt is not None else xtr.copy())
            else:
                model.fit(xtr, ytr)
            rows.append(dict(case=case, method=variant, seed=seed, **metrics(yte, model.predict_proba(xte))))
res = pd.DataFrame(rows)
METHODS = ["sklearn_LDA", *VARIANTS]
mean = res.groupby(["case", "method"])[["accuracy", "logloss", "ece"]].mean()
sd = res.groupby(["case", "method"])[["accuracy", "logloss", "ece"]].std()
def pivot(metric):
    return mean[metric].unstack("method").loc[list(CASES), METHODS].round(4)
print(f"Accuracy (mean over seeds {SEEDS})")
pivot("accuracy")
"""),
        code("""
print("Log-loss (lower is better)"); print(pivot("logloss").to_string())
print("\\nECE (lower is better)"); print(pivot("ece").to_string())
print("\\nAccuracy sample SD over seeds"); print(sd["accuracy"].unstack("method").loc[list(CASES), METHODS].round(4).to_string())
"""),
        code("""
acc = pivot("accuracy")
ax = acc.plot.bar(figsize=(12, 4), width=0.85)
ax.set_ylabel("accuracy"); ax.set_ylim(max(0.0, acc.values.min() - 0.05), 1.0)
ax.set_title("Held-out accuracy by case (mean over seeds)"); plt.xticks(rotation=25)
plt.legend(ncol=3, fontsize=8); plt.tight_layout(); plt.show()
"""),
        md("## 4. Look inside each variant"),
        code("""
seed = SEEDS[0]
# shift_corrected: importance weights
xtr, ytr, xte, yte, xadapt = dataset("shifted", seed)
sc = LDA(variant="shift_corrected").fit(xtr, ytr, X_target=xadapt)
w = sc.sample_weights_
print(f"shift_corrected weights: min={w.min():.3f} mean={w.mean():.3f} max={w.max():.3f} (clipped at 20, renormalised)")
# shrinkage intensity
xtr, ytr, *_ = dataset("high_dim", seed)
print("Ledoit-Wolf shrinkage intensity (high_dim):", round(LDA(variant="shrinkage").fit(xtr, ytr).shrinkage_, 3),
      "| (clean):", round(LDA(variant="shrinkage").fit(*dataset("clean", seed)[:2]).shrinkage_, 3))
# sparse: selected features out of 80
sp = LDA(variant="sparse", threshold=1.0).fit(xtr, ytr)
print(f"sparse: {len(sp.selected_features_)} of {xtr.shape[1]} features retained (10 informative): {sp.selected_features_[:15]}")
# heteroscedastic: class covariance traces
xh, yh, *_ = dataset("heteroscedastic", seed)
het = LDA(variant="heteroscedastic").fit(xh, yh)
print("heteroscedastic class-covariance traces:", [round(float(np.trace(S)), 1) for S in het.class_covariances_])

# discriminant projection (wine) with the base variant
xw, yw, xwt, ywt, _ = dataset("wine", seed)
Z = LDA(variant="base").fit(xw, yw).transform(xw)
plt.figure(figsize=(4.8, 3.8)); plt.scatter(Z[:, 0], Z[:, 1], c=yw, cmap="viridis", s=14)
plt.title("StatCore LDA projection (wine)"); plt.show()
"""),
        md("## 5. Summary and limitations"),
        code("""
base = mean["accuracy"].unstack("method")
summary = pd.DataFrame({
    "best_method": base[METHODS].idxmax(axis=1),
    "best_acc": base[METHODS].max(axis=1),
    "sklearn_LDA_acc": base["sklearn_LDA"],
    "statcore_base_acc": base["base"]}).loc[list(CASES)].round(4)
print(summary.to_string())
"""),
        md("""
**What to read from the tables.** `base` should match sklearn's LDA closely (same model, different numerical path). `shrinkage` is expected to help when $n\\approx d$ (`high_dim`), `sparse` when most features are noise, `heteroscedastic` when class covariances really differ, and `shift_corrected` only when the shift is genuine and the domain classifier is informative.

**Honest limitations**
- In our local 3-seed run: `shift_corrected` was *worse* than plain LDA on the shifted case (accuracy ≈ 0.80 vs 0.87, log-loss ≈ 0.72 vs 0.32) — importance weights with high variance (some > 10 after renormalisation) hurt; and `shrinkage` was clearly worse on the raw (unstandardised) wine features because shrinking towards a scaled identity is scale-sensitive. Standardise features first if you use it.
- `heteroscedastic` wins on wine/heteroscedastic data but collapses when $n\\approx d$ (`high_dim`, accuracy ≈ 0.54, log-loss > 12).
- Synthetic Gaussian data favour LDA-type models by construction; wine is small (178 rows), so with a handful of seeds many differences are within noise.
- `shift_corrected` needs unlabeled target data, only corrects *covariate* shift (not label or concept shift), and clips weights at 20, which biases the correction when the shift is severe. In non-shifted cases it is given a copy of the training features (weights ≈ 1), so it only tests that it does no harm.
- `sparse` is a diagonal-covariance model: it ignores feature correlations and its threshold `threshold` is not tuned.
- `heteroscedastic` needs enough samples per class to estimate $\\Sigma_k$; its regularisation weight `reg_param` is fixed at 0.3 here.
- Probabilities come from a Gaussian model and are not post-hoc calibrated; ECE is a noisy estimate with 10 bins on a few hundred test points.
"""),
    ]
    write("lda", c)


def build_kmeans() -> None:
    c = header("kmeans", "StatCore K-Means: robust, topology-aware and model-selecting variants",
               "Reliability-Oriented K-Means in StatCore", "(11, 23, 37)")
    c += [
        md(r"""
## 2. Method summary

Data $x_i\in\mathbb{R}^d$, centres $c_1..c_k$. Lloyd iterations alternate assignment $z_i=\arg\min_j D(x_i,c_j)$ and centre updates; `n_init` restarts keep the lowest objective $\sum_i D(x_i,c_{z_i})$. Reference: sklearn `KMeans(n_init=5, max_iter=100, tol=1e-6)`.

**`base`** — squared Euclidean $D=\|x-c\|^2$, restarts seeded with $k$ distinct random data rows, mean update.

**`better_init`** — same Lloyd, seeded by k-means++: first centre uniform, then $P(x_i)\propto\min_j\|x_i-c_j\|^2$.

**`robust`** (trimmed coordinate-wise k-medians) — $D=\|x-c\|_1$; at each step the `trim` fraction (5 %) of points with the largest assignment cost is ignored, and each centre becomes the coordinate-wise **median** of its remaining members (random-row seeding as in `base`).

**`topology_aware`** — k-means++ seeding plus kNN-graph smoothing. With $N_m(i)$ the $m$ nearest neighbours (`n_neighbors`) and previous assignments $z^{\text{prev}}$,
$$\tilde D(x_i,c_j)=\frac{\|x_i-c_j\|^2}{\overline{\|\cdot\|^2}}-\lambda\cdot\frac{|\{l\in N_m(i): z^{\text{prev}}_l=j\}|}{m},\qquad \lambda=\texttt{topology\_weight}$$
(the smoothing term is only active from the second iteration on), so points prefer the cluster of their neighbours.

**`stat_model_selection`** — fits every $k\in[k_{\min},k_{\max}]$ and keeps the best by BIC or mean silhouette. For the spherical-Gaussian BIC with $\hat\sigma^2=\mathrm{SSE}/((n-k)d)$:
$$\log L=\sum_j n_j\log\frac{n_j}{n}-\frac{nd}{2}\log(2\pi\hat\sigma^2)-\frac{(n-k)d}{2},\qquad \mathrm{BIC}=-2\log L+(k-1+kd+1)\log n .$$
`scores_` stores BIC (lower is better) or silhouette (higher is better) per $k$.
"""),
        md("## 3. Experiment (mirrors `kmeans_experiment.py`)\n\n*synthetic*: three Gaussian blobs in 4-D ($\\sigma=0.48$), 18 gross outliers injected into the training set only; *iris*: 75/25 split. Data are standardised with training statistics; the model is fitted on the training part and evaluated on the held-out part: adjusted Rand index (ARI) against the true labels, silhouette and held-out SSE (squared distance to the assigned centre). All methods get $k=3$ except `stat_model_selection`, which searches $k\\in[2,6]$ by BIC."),
        code("""
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans as ReferenceKMeans        # baseline only
from ml_reliability.statcore import KMeans
from ml_reliability.statcore.kmeans.kmeans import VARIANTS

METHODS = ("reference",) + tuple(VARIANTS)

def standardize(train, test):
    mean = train.mean(0); scale = train.std(0); scale[scale <= 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale

def synthetic(seed):
    rng = np.random.default_rng(seed)
    means = np.array([[-3, -2, 0, 1], [3, -2, 1, -1], [0, 3, -1, 0]])
    clean = np.vstack([rng.normal(loc=m, scale=.48, size=(100, 4)) for m in means]); labels = np.repeat(np.arange(3), 100)
    order = rng.permutation(len(clean)); tr, te = order[:240], order[240:]; train = clean[tr].copy()
    idx = rng.choice(len(train), 18, replace=False); train[idx] += rng.normal(0, 6, (18, 4))
    xtr, xte = standardize(train, clean[te]); return xtr, xte, labels[te], labels[tr]

def iris(seed):
    data = load_iris(); order = np.random.default_rng(seed).permutation(len(data.data)); split = int(.75 * len(order))
    tr, te = order[:split], order[split:]; xtr, xte = standardize(data.data[tr], data.data[te])
    return xtr, xte, data.target[te], data.target[tr]

def ari(a, b):
    _, ai = np.unique(a, return_inverse=True); _, bi = np.unique(b, return_inverse=True)
    tab = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64); np.add.at(tab, (ai, bi), 1)
    c2 = lambda x: x * (x - 1) / 2.0
    index = c2(tab).sum(); rows = c2(tab.sum(1)).sum(); cols = c2(tab.sum(0)).sum(); total = c2(len(a))
    expected = rows * cols / total if total else 0.; maximum = .5 * (rows + cols)
    return float((index - expected) / (maximum - expected)) if maximum != expected else 1.

def silhouette(X, labels):
    classes = np.unique(labels)
    if len(classes) < 2: return 0.
    D = np.sqrt(np.maximum((X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2 * X @ X.T, 0)); vals = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]; count = int(own.sum()); a = D[i, own].sum() / (count - 1) if count > 1 else 0.
        b = min(D[i, labels == c].mean() for c in classes if c != labels[i])
        vals[i] = (b - a) / max(a, b, 1e-12) if count > 1 else 0.
    return float(vals.mean())

def fit_method(method, xtr, k, seed):
    if method == "reference":
        return ReferenceKMeans(n_clusters=k, n_init=5, random_state=seed, max_iter=100, tol=1e-6).fit(xtr)
    kw = dict(variant=method, random_state=seed, max_iter=100, n_init=5, tol=1e-6)
    if method == "stat_model_selection":
        kw.update(k_range=(2, 6), criterion="bic")
    else:
        kw.update(n_clusters=k)
    return KMeans(**kw).fit(xtr)

def run(method, xtr, xte, k, seed):
    model = fit_method(method, xtr, k, seed)
    labels = model.predict(xte); centers = model.cluster_centers_
    return labels, float(np.sum((xte - centers[labels]) ** 2)), len(centers)

rows = []
for ds, factory in (("synthetic", synthetic), ("iris", iris)):
    for method in METHODS:
        for seed in SEEDS:
            xtr, xte, truth, _ = factory(seed)
            labels, sse, k_used = run(method, xtr, xte, 3, seed)
            rows.append(dict(dataset=ds, method=method, seed=seed, ARI=ari(truth, labels),
                             silhouette=silhouette(xte, labels), SSE=sse, k_used=k_used))
res = pd.DataFrame(rows)
mean = res.groupby(["dataset", "method"], sort=False)[["ARI", "silhouette", "SSE", "k_used"]].mean()
sd = res.groupby(["dataset", "method"], sort=False)[["ARI", "silhouette", "SSE"]].std()
print(f"Held-out metrics, mean over seeds {SEEDS}")
mean.round(4)
"""),
        code("""
print("sample SD over seeds"); print(sd.round(4).to_string())
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
for ax, metric in zip(axes, ("ARI", "silhouette", "SSE")):
    mean[metric].unstack("dataset").loc[list(METHODS)].plot.bar(ax=ax, rot=30); ax.set_title(f"{metric} (mean over seeds)")
plt.tight_layout(); plt.show()
"""),
        md("## 4. Look inside each variant\n\nCluster assignments on the first two standardised features of the contaminated synthetic set, and the BIC / silhouette curves used by `stat_model_selection`."),
        code("""
xtr, xte, truth, _ = synthetic(SEEDS[0])
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, m in zip(axes, ("reference", "base", "robust", "topology_aware")):
    model = fit_method(m, xtr, 3, SEEDS[0]); lab = model.predict(xte)
    ax.scatter(xte[:, 0], xte[:, 1], c=lab, s=10, cmap="viridis"); ax.scatter(*model.cluster_centers_[:, :2].T, c="red", marker="x", s=80)
    ax.set_title(f"{m}  ARI={ari(truth, lab):.2f}")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, crit in zip(axes, ("bic", "silhouette")):
    sel = KMeans(variant="stat_model_selection", k_range=(2, 6), criterion=crit, random_state=SEEDS[0]).fit(xtr)
    ks = sorted(sel.scores_); ax.plot(ks, [sel.scores_[k] for k in ks], "o-"); ax.axvline(sel.n_clusters_, color="r", ls="--")
    ax.set_title(f"criterion={crit}: chosen k={sel.n_clusters_} (true k=3)"); ax.set_xlabel("k")
plt.tight_layout(); plt.show()
"""),
        md("## 5. Summary and limitations"),
        code("""
t = mean.reset_index()
for metric, better in (("ARI", "max"), ("SSE", "min")):
    idx = t.groupby("dataset")[metric].transform(better) == t[metric]
    print(f"best {metric}:", dict(zip(t[idx]["dataset"], t[idx]["method"])))
"""),
        md("""
**What to read from the tables.** In our local 3-seed run, `robust` was the best on ARI for both datasets, while `better_init` and `topology_aware` were *worse* than random-row seeding on the contaminated synthetic set (ARI ≈ 0.88 vs 1.0, large SD): $D^2$-weighted k-means++ seeding preferentially picks the injected outliers as centres. `stat_model_selection` chose $k\\approx5$–$6$ (not 3) on both datasets, because BIC treats the outlier points as extra clusters (synthetic) and the overlapping iris classes need extra spherical components.

**Honest limitations**
- Two small datasets and a few seeds: ARI/SSE differences between several variants are often within one SD; iris clusters (versicolor/virginica) overlap, which caps every method.
- SSE (squared distance to centre) is *not* the objective of `robust` (L1), so reference-style SSE can favour non-robust methods even when ARI favours the robust one; compare metrics, do not rank on one.
- `topology_aware` adds a heuristic penalty (weights fixed at `topology_weight=0.5`, `n_neighbors=10`) and costs $O(n^2)$ for the neighbour graph; it can blur nearby clusters.
- BIC uses a spherical-Gaussian model, so it is misleading for elongated, overlapping or contaminated clusters (as seen above); `stat_model_selection` is not a reliable way to recover the true $k$ here, and its SSE is lower simply because it uses more centres.
- Silhouette is evaluated on the held-out part with held-out predictions only; no significance tests were run.
"""),
    ]
    write("kmeans", c)


def build_linear() -> None:
    c = header("linear", "StatCore linear models: calibrated, regularised and robust logistic / ridge regression",
               "Regularized, Calibrated, and Robust Linear Models in StatCore", "(11, 23, 37)")
    c += [
        md(r"""
## 2. Method summary

Features are standardised with training statistics, $z=(x-m)/s$. Notation: $\ell_i=-\log p_{y_i}(z_i)$.

### Multinomial logistic regression
Softmax model $p(c\mid z)=\mathrm{softmax}(Wz+b)_c$, trained with L-BFGS on the weighted mean loss with L2 penalty on $W$ only:
$$\min_{W,b}\ \sum_i \tilde w_i\,\ell_i+\tfrac{\alpha}{2}\|W\|_F^2,\qquad \tilde w_i=w_i/\textstyle\sum w .$$
Reference: sklearn `Pipeline(StandardScaler, LogisticRegression(C=1/(n\alpha)))` (same penalty scale).

- **`base`** — fixed $\alpha$ (default $10^{-2}$).
- **`risk_calibrated`** — fit on a stratified split (`calib_fraction`=25 % held out), then a scalar temperature $T=e^{\tau}$, $\tau\in[-3,3]$, minimises the held-out NLL of $\mathrm{softmax}(\text{logits}/T)$ (golden-section search); `decision_function` divides logits by $T$.
- **`stat_regularised`** — stratified $K$-fold CV log-loss over $\alpha\in\{10^{-4},..,10\}$ (6 log-spaced values), refit on all data with the best $\alpha$ (`cv_scores_`).
- **`robust_loss`** — 6 reweighting rounds: $\mathrm{cut}=\mathrm{median}(\ell)+1.345\cdot1.4826\,\mathrm{MAD}(\ell)$, $w_i=\min(1,\mathrm{cut}/\ell_i)$ (Huber-type bounded influence for high-loss, e.g. mislabelled, points).

### Ridge regression
Closed form with unpenalised intercept: $\beta=(Z_a^\top WZ_a+\lambda P)^{-1}Z_a^\top Wy$, $P=\mathrm{diag}(0,1,..,1)$. Reference: sklearn `Pipeline(StandardScaler, Ridge(alpha=1))`.

- **`base`** — fixed $\lambda=\alpha$.
- **`risk_calibrated`** — fit on the training part of a split; linear recalibration $\hat y\leftarrow a+b\,\hat y$ fitted on the held-out part ($b=\mathrm{cov}(\hat y,y)/\mathrm{var}(\hat y)$); absolute held-out residuals are stored and `predict_interval(X, level)` returns the split-conformal band $\hat y\pm q$ with $q$ the $\lceil(m+1)\,\text{level}\rceil$-th smallest residual.
- **`stat_regularised`** — generalised cross-validation, with singular values $s_j$ of the centred design: $\mathrm{shrink}_j=\frac{s_j^2}{s_j^2+\lambda}$, $\mathrm{dof}=1+\sum_j\mathrm{shrink}_j$,
$$\mathrm{GCV}(\lambda)=\frac{n\,\mathrm{RSS}(\lambda)}{(n-\mathrm{dof})^2},$$
minimised on a 46-point log grid in $[10^{-6},10^{3}]$.
- **`robust_loss`** — Huber IRLS (up to 30 iterations): $\sigma=1.4826\,\mathrm{MAD}(r)$, $c=\kappa\sigma$ ($\kappa=1.345$), $w_i=\min(1,c/|r_i|)$.
"""),
        md("## 3. Experiments (mirror `linear_experiment.py`)\n\nClassification cases: `clean`, covariate `shifted` test set, 15 % `label_noise`, 8 % gross `outliers`, `high_dim` ($d=80$, $n=140$), and real `breast_cancer`. Regression cases: `clean`, `shifted`, `noisy` (extra training noise), `outliers` (8 % of $y$ corrupted), `high_dim`, and `diabetes`. Metrics: accuracy / log-loss / 10-bin ECE and MSE / MAE / $R^2$."),
        code("""
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression      # baseline only
from sklearn.linear_model import Ridge as SklearnRidge                                # baseline only
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml_reliability.statcore import LogisticRegression, RidgeRegression
from ml_reliability.statcore.linear.linear import VARIANTS

CLASS_CASES = ("clean", "shifted", "label_noise", "outliers", "high_dim", "breast_cancer")
REG_CASES = ("clean", "shifted", "noisy", "outliers", "high_dim", "diabetes")

def split_indices(n, seed):
    rng = np.random.default_rng(seed); idx = rng.permutation(n); k = int(round(0.25 * n))
    return np.sort(idx[k:]), np.sort(idx[:k])

def class_data(case, seed):
    rng = np.random.default_rng(seed)
    if case == "breast_cancer":
        ds = load_breast_cancer(); tr, te = split_indices(len(ds.target), seed)
        return ds.data[tr], ds.target[tr], ds.data[te], ds.target[te]
    ntr, nte = (140, 300) if case == "high_dim" else (360, 600)
    d = 80 if case == "high_dim" else 12
    info = 10 if case == "high_dim" else min(d, 8)
    beta = np.zeros(d); beta[:info] = np.array([1.2, -1.0, 0.8, -0.7, 0.5, -0.4, 0.3, -0.2, 0.15, -0.1])[:info]
    shift = np.zeros(d)
    if case == "shifted":
        shift[:info] = np.linspace(0.35, 0.9, info)
    def sample(n, offset):
        x = rng.normal(size=(n, d)) + offset
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
        return x, rng.binomial(1, p)
    xtr, ytr = sample(ntr, np.zeros(d)); xte, yte = sample(nte, shift)
    if case == "label_noise":
        flip = rng.random(ntr) < 0.15; ytr[flip] = 1 - ytr[flip]
    elif case == "outliers":
        out = rng.random(ntr) < 0.08; xtr[out] = rng.normal(0, 6, size=(out.sum(), d))
    return xtr, ytr, xte, yte

def regression_data(case, seed):
    rng = np.random.default_rng(seed)
    if case == "diabetes":
        ds = load_diabetes(); tr, te = split_indices(len(ds.target), seed)
        return ds.data[tr], ds.target[tr], ds.data[te], ds.target[te]
    ntr, nte = (100, 300) if case == "high_dim" else (360, 600)
    d = 80 if case == "high_dim" else 12
    info = 10 if case == "high_dim" else min(d, 8)
    beta = np.zeros(d); beta[:info] = np.linspace(1.5, -0.7, info)
    offset = np.zeros(d)
    if case == "shifted":
        offset[:info] = np.linspace(0.4, 0.9, info)
    xtr = rng.normal(size=(ntr, d)); xte = rng.normal(size=(nte, d)) + offset
    ytr = xtr @ beta + rng.normal(0, 1.0, ntr); yte = xte @ beta + rng.normal(0, 1.0, nte)
    if case == "noisy":
        ytr += rng.normal(0, 3.0, ntr)
    elif case == "outliers":
        out = rng.random(ntr) < 0.08; ytr[out] += rng.normal(0, 20.0, out.sum())
    return xtr, ytr, xte, yte

def class_scores(y, p):
    pred = p.argmax(axis=1); acc = float(np.mean(pred == y))
    loss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())
    conf, correct = p.max(axis=1), pred == y; edges = np.linspace(0, 1, 11); ece = 0.0
    for j in range(10):
        m = (conf >= edges[j]) & (conf < edges[j + 1] if j < 9 else conf <= edges[j + 1])
        if m.any(): ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return dict(accuracy=acc, logloss=loss, ece=float(ece))

def reg_scores(y, pred):
    err = pred - y
    return dict(mse=float(np.mean(err ** 2)), mae=float(np.mean(np.abs(err))),
                r2=float(1.0 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)))

crow, rrow = [], []
for case in CLASS_CASES:
    for seed in SEEDS:
        xtr, ytr, xte, yte = class_data(case, seed)
        base = make_pipeline(StandardScaler(), SklearnLogisticRegression(C=1.0 / (len(ytr) * 1e-2), max_iter=1000, random_state=seed)).fit(xtr, ytr)
        crow.append(dict(case=case, method="sklearn_standard", seed=seed, **class_scores(yte, base.predict_proba(xte))))
        for v in VARIANTS:
            with np.errstate(divide="ignore", invalid="ignore"):
                m = LogisticRegression(variant=v, random_state=seed).fit(xtr, ytr)
            crow.append(dict(case=case, method=v, seed=seed, **class_scores(yte, m.predict_proba(xte))))
for case in REG_CASES:
    for seed in SEEDS:
        xtr, ytr, xte, yte = regression_data(case, seed)
        base = make_pipeline(StandardScaler(), SklearnRidge(alpha=1.0)).fit(xtr, ytr)
        rrow.append(dict(case=case, method="sklearn_standard", seed=seed, **reg_scores(yte, base.predict(xte))))
        for v in VARIANTS:
            m = RidgeRegression(variant=v, random_state=seed).fit(xtr, ytr)
            rrow.append(dict(case=case, method=v, seed=seed, **reg_scores(yte, m.predict(xte))))
cres, rres = pd.DataFrame(crow), pd.DataFrame(rrow)
METHODS = ["sklearn_standard", *VARIANTS]
def pivot(df, metric, cases):
    return df.groupby(["case", "method"])[metric].mean().unstack("method").loc[list(cases), METHODS].round(4)
print(f"CLASSIFICATION, mean over seeds {SEEDS}")
for metric in ("accuracy", "logloss", "ece"):
    print(f"\\n[{metric}]"); print(pivot(cres, metric, CLASS_CASES).to_string())
"""),
        code("""
print("REGRESSION")
for metric in ("mse", "mae", "r2"):
    print(f"\\n[{metric}]"); print(pivot(rres, metric, REG_CASES).to_string())
"""),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
pivot(cres, "logloss", CLASS_CASES).plot.bar(ax=axes[0], rot=30, legend=False); axes[0].set_title("classification log-loss"); axes[0].set_yscale("log")
pivot(cres, "ece", CLASS_CASES).plot.bar(ax=axes[1], rot=30, legend=False); axes[1].set_title("classification ECE")
pivot(rres, "mse", REG_CASES).plot.bar(ax=axes[2], rot=30); axes[2].set_title("regression MSE"); axes[2].set_yscale("log")
axes[2].legend(fontsize=7); plt.tight_layout(); plt.show()
"""),
        md("## 4. Look inside each variant"),
        code("""
seed = SEEDS[0]
xtr, ytr, xte, yte = class_data("label_noise", seed)
rc = LogisticRegression(variant="risk_calibrated", random_state=seed).fit(xtr, ytr)
sr = LogisticRegression(variant="stat_regularised", random_state=seed).fit(xtr, ytr)
rl = LogisticRegression(variant="robust_loss", random_state=seed).fit(xtr, ytr)
flipped_weight = rl.sample_weights_
print(f"risk_calibrated temperature T = {rc.temperature_:.3f}  (T>1 softens over-confident logits)")
print("stat_regularised CV log-loss by alpha:", {k: round(float(v), 4) for k, v in sr.cv_scores_.items()}, "-> alpha =", sr.alpha_)
print(f"robust_loss: {int((flipped_weight < 0.999).sum())} of {len(flipped_weight)} samples down-weighted; min weight {flipped_weight.min():.3f}")

# reliability diagram (label_noise): plain vs risk_calibrated
plain = LogisticRegression(variant="base", random_state=seed).fit(xtr, ytr)
plt.figure(figsize=(4.6, 4))
for name, mdl in (("base", plain), ("risk_calibrated", rc)):
    p = mdl.predict_proba(xte); conf = p.max(1); corr = p.argmax(1) == yte
    bins = np.linspace(0.5, 1, 7); idx = np.clip(np.digitize(conf, bins) - 1, 0, 5)
    xs = [conf[idx == b].mean() for b in range(6) if (idx == b).any()]; ys = [corr[idx == b].mean() for b in range(6) if (idx == b).any()]
    plt.plot(xs, ys, "o-", label=name)
plt.plot([0.5, 1], [0.5, 1], "k--"); plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.legend(); plt.title("Reliability (label_noise)"); plt.show()

# Ridge: conformal interval, GCV alpha, Huber weights
xtr, ytr, xte, yte = regression_data("outliers", seed)
rcal = RidgeRegression(variant="risk_calibrated", random_state=seed).fit(xtr, ytr)
lo, hi = rcal.predict_interval(xte, level=0.9)
print(f"split-conformal 90% interval: coverage {np.mean((yte >= lo) & (yte <= hi)):.3f}, mean width {np.mean(hi - lo):.2f}  (calibration a,b = {np.round(rcal.calib_, 3)})")
print("GCV-selected ridge alpha:", RidgeRegression(variant="stat_regularised").fit(xtr, ytr).alpha_)
rrob = RidgeRegression(variant="robust_loss").fit(xtr, ytr)
print(f"Huber IRLS: {int((rrob.sample_weights_ < 0.999).sum())} of {len(ytr)} samples down-weighted (true outliers ~ 8%)")
"""),
        md("## 5. Summary and limitations"),
        code("""
def best(df, metric, cases, smaller=True):
    p = pivot(df, metric, cases); return (p.idxmin(axis=1) if smaller else p.idxmax(axis=1)).to_dict()
print("best log-loss per classification case:", best(cres, "logloss", CLASS_CASES))
print("best accuracy per classification case:", best(cres, "accuracy", CLASS_CASES, smaller=False))
print("best MSE per regression case:", best(rres, "mse", REG_CASES))
"""),
        md("""
**What to read from the tables.** `base` tracks the sklearn pipeline (same objective; small differences from solver details). In our local 3-seed run: `robust_loss` had the best log-loss/ECE under `label_noise` and `outliers` and the best accuracy there, and for ridge cut MSE on `outliers` by ~50 % (≈2.06 → 1.06); `stat_regularised` helped most in `high_dim` for both models (ridge MSE 4.34 → 2.61); `risk_calibrated` reduced ECE only in `high_dim` and was *worse* than `base` on most other cases (e.g. breast-cancer log-loss 0.37 vs 0.11).

**Honest limitations**
- `risk_calibrated` trains on only 75 % of the data and its temperature is fit on a small held-out set, so on small or already well-calibrated problems it can hurt (see breast cancer); a single scalar temperature cannot fix mis-specification or covariate shift. Ridge `risk_calibrated` likewise lost accuracy everywhere except as an interval generator.
- `robust_loss` (logistic) raised ECE/log-loss on the clean cases (it down-weights hard but legitimate points) and lowered breast-cancer accuracy.
- GCV selected $\\lambda=100$ on the outlier case (the grid spans $10^{-6}$–$10^{3}$), where GCV itself is distorted by the outliers.
- Mostly synthetic data with known structure and a few seeds; real datasets are small (breast cancer 569, diabetes 442). Gains of the order of one SD are not evidence.
- The conformal interval gives marginal (average) coverage under exchangeability only; it is not valid under covariate shift and exists only for ridge `risk_calibrated`. CV/GCV grids are fixed and coarse and ignore distribution shift.
"""),
    ]
    write("linear", c)


_NN_UTILS = '''
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def _synthetic(kind, seed, n=640):
    rng = np.random.default_rng(seed + (10000 if kind == "regression" else 0))
    x = rng.uniform(-2.0, 2.0, size=(n, 2)); noise = rng.normal(size=n)
    if kind == "classification":
        y = ((x[:, 0] ** 2 + 0.65 * x[:, 1] ** 2 + 0.25 * noise) > 1.65).astype(int)
        flip = rng.random(n) < 0.10; y[flip] = 1 - y[flip]
    else:
        y = np.sin(1.4 * x[:, 0]) + 0.35 * x[:, 1] ** 2 + 0.30 * noise
    return np.column_stack((x, rng.normal(size=(n, 18)))), y      # 2 relevant + 18 irrelevant features

def datasets(kind, seed):
    X, y = _synthetic(kind, seed)
    yield "synthetic_nonlinear_irrelevant", X, y
    bunch = load_breast_cancer() if kind == "classification" else load_diabetes()
    yield ("sklearn_breast_cancer" if kind == "classification" else "sklearn_diabetes"), bunch.data, bunch.target

def split_data(X, y, seed, stratify):
    rng = np.random.default_rng(seed); n_test = int(np.ceil(0.25 * len(y)))
    if not stratify:
        order = rng.permutation(len(y))
        return X[order[n_test:]], X[order[:n_test]], y[order[n_test:]], y[order[:n_test]]
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    raw = counts * (n_test / len(y)); allocation = np.floor(raw).astype(int)
    remaining = n_test - int(allocation.sum()); tie_break = rng.random(len(counts))
    order = np.lexsort((tie_break, -(raw - allocation))); allocation[order[:remaining]] += 1
    train_ids, test_ids = [], []
    for c, count in enumerate(allocation):
        ids = rng.permutation(np.flatnonzero(inverse == c)); test_ids.extend(ids[:count]); train_ids.extend(ids[count:])
    train_ids = rng.permutation(np.asarray(train_ids, dtype=int)); test_ids = rng.permutation(np.asarray(test_ids, dtype=int))
    return X[train_ids], X[test_ids], y[train_ids], y[test_ids]

def ece_equal_width(y, prob, bins=15):
    pred = prob.argmax(axis=1); conf = prob.max(axis=1); correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1); ans = 0.0
    for j in range(bins):
        mask = (conf >= edges[j]) & ((conf < edges[j + 1]) if j < bins - 1 else (conf <= edges[j + 1]))
        if mask.any():
            ans += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ans)

def classification_scores(y, prob):
    prob = np.clip(np.asarray(prob), 1e-12, 1.0); prob = prob / prob.sum(axis=1, keepdims=True)
    truth = np.searchsorted(np.unique(y), y)
    return {"accuracy": float(np.mean(truth == prob.argmax(axis=1))),
            "log_loss": float(-np.log(prob[np.arange(len(y)), truth]).mean()),
            "ece": ece_equal_width(truth, prob)}

def regression_scores(y, pred):
    residual = np.asarray(y) - np.asarray(pred); total = np.sum((y - np.mean(y)) ** 2)
    return {"mse": float(np.mean(residual ** 2)), "mae": float(np.mean(np.abs(residual))),
            "r2": float(1.0 - np.sum(residual ** 2) / total)}

def run_family(make_models, seeds):
    import time
    rows = []
    for task in ("classification", "regression"):
        for seed in seeds:
            for dataset, X, y in datasets(task, seed):
                Xtr, Xte, ytr, yte = split_data(X, y, seed=seed, stratify=(task == "classification"))
                for name, model in make_models(task, seed).items():
                    start = time.perf_counter(); model.fit(Xtr, ytr)
                    if task == "classification":
                        prob = model.predict_proba(Xte); classes = model.classes_
                        if not np.array_equal(classes, np.arange(len(classes))):
                            prob = prob[:, [int(np.flatnonzero(classes == c)[0]) for c in np.arange(len(classes))]]
                        metrics = classification_scores(yte, prob)
                    else:
                        metrics = regression_scores(yte, model.predict(Xte))
                    rows.append({"task": task, "dataset": dataset, "seed": seed, "model": name,
                                 "runtime_s": time.perf_counter() - start, **metrics})
    return pd.DataFrame(rows)

def summarise(res, order):
    for task, metrics in (("classification", ["accuracy", "log_loss", "ece", "runtime_s"]),
                          ("regression", ["mse", "mae", "r2", "runtime_s"])):
        sub = res[res.task == task]
        g = sub.groupby(["dataset", "model"], sort=False)[metrics]
        m, s = g.mean(), g.std()
        out = m.round(4).astype(str) + " +/- " + s.round(4).astype(str)
        print(f"\\n[{task}] mean +/- sample SD over seeds"); print(out.loc[pd.MultiIndex.from_product([sub.dataset.unique(), order])].to_string())
    return res
'''


def build_knn() -> None:
    c = header("knn", "StatCore k-NN: prototype compression, learnable metric and data-adaptive k",
               "StatCore Nearest Neighbours: Exact Baselines, Prototype Compression, Diagonal Metric Learning, and Data-Adaptive k",
               "(11, 23)")
    c += [
        md(r"""
## 2. Method summary

All variants standardise features, $z=(x-m)/s$, and predict by a (weighted) average of the targets of the $k$ nearest references under $d^2_a(z,z')=\sum_j a_j(z_j-z'_j)^2$ (with $a=\mathbf 1$ unless the metric is learned):
$$\hat t(z)=\frac{\sum_{j\in N_k(z)} r_j\,\omega_j\,t_j}{\sum_{j\in N_k(z)} r_j\,\omega_j},\qquad \omega_j=1\ \text{(uniform)}\ \text{or}\ 1/(d_j+10^{-9})\ \text{(distance)},$$
where $t_j$ is a one-hot label (classification; $\hat t$ are the class probabilities) or the response (regression) and $r_j$ are reference weights ($r_j=1$ except for prototypes). Reference: sklearn `Pipeline(StandardScaler, KNeighbors*(n_neighbors=5))`.

**`base`** — exact Euclidean k-NN with $k=$ `n_neighbors`.

**`prototype_compressed`** — per class (classification) or over the whole set (regression, clustering in $(z, \text{standardised } y)$), k-means (`better_init`, 2 restarts, 50 iterations) with $m=\lceil\gamma\,n_c\rceil$ clusters, $\gamma=$ `compression`. Each cluster $S$ becomes a prototype at the mean of its members' $z$ with target equal to the mean of its targets and reference weight $r=|S|$. Classes with $m\ge n_c$ keep all points.

**`learnable_metric`** — diagonal metric $a=e^{\theta}$ learned by Adam on a random sub-sample of at most 400 points. With soft neighbours $p_{ij}=\mathrm{softmax}_{j\ne i}\big(-d^2_a(z_i,z_j)\big)$:
- classification maximises the leave-one-out NCA objective $\frac1m\sum_i q_i$, $q_i=\sum_{j:\,y_j=y_i}p_{ij}$;
- regression minimises $\frac1m\sum_i(\hat y_i-y_i)^2$ with $\hat y_i=\sum_j p_{ij}y_j$ (standardised $y$).

Adam ($\beta=(0.9,0.999)$, `metric_lr`, `metric_iters` steps) acts on $\theta$ clipped to $[-8,3]$; the final metric is normalised to $\mathrm{mean}(a)=1$. Irrelevant features should receive small $a_j$.

**`stat_k_selection`** — leave-one-out (self excluded) over odd $k\le30$:
$$L_k=\Big(1-\mathrm{acc}_k\Big)+10^{-3}\,\overline{-\log\max(\hat p_{y},10^{-3})}\quad(\text{classification}),\qquad L_k=\overline{(\hat y-y)^2}\quad(\text{regression}),$$
and the smallest minimiser is used (ties go to the smaller $k$).
"""),
        md("## 3. Experiment (mirrors `knn_experiment.py` / `_experiment_utils.py`)\n\nClassification and regression, each on a *synthetic nonlinear* set (2 relevant + 18 irrelevant features, 640 rows, 10 % label flips for classification) and a real sklearn set (breast cancer / diabetes). 75/25 split (stratified for classification); $k=5$, uniform weights, `compression=0.35`, 40 metric iterations."),
        code(_NN_UTILS + """
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor      # baseline only
from ml_reliability.statcore import KNNClassifier, KNNRegressor

VARIANT_NAMES = ("base", "prototype_compressed", "learnable_metric", "stat_k_selection")

def make_models(task, seed):
    is_class = task == "classification"
    baseline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5) if is_class else KNeighborsRegressor(n_neighbors=5))
    cls = KNNClassifier if is_class else KNNRegressor
    models = {name: cls(variant=name, n_neighbors=5, weights="uniform", compression=0.35,
                        metric_iters=40, metric_lr=0.1, random_state=seed) for name in VARIANT_NAMES}
    return {"sklearn_standard": baseline, **models}

ORDER = ["sklearn_standard", *VARIANT_NAMES]
res = run_family(make_models, SEEDS)
print(f"k-NN experiment over seeds {SEEDS} (runtime = single fit + predict, no warm-up)")
res = summarise(res, ORDER)
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, (task, metric) in zip(axes, (("classification", "accuracy"), ("regression", "mse"))):
    t = res[res.task == task].groupby(["dataset", "model"])[metric].mean().unstack("model")[ORDER]
    t.plot.bar(ax=ax, rot=15, legend=(task == "regression")); ax.set_title(f"{task}: {metric}")
    if metric == "mse": ax.set_yscale("log")
plt.tight_layout(); plt.show()
"""),
        md("## 4. Look inside each variant"),
        code("""
seed = SEEDS[0]
Xtr, Xte, ytr, yte = split_data(*_synthetic("classification", seed), seed=seed, stratify=True)
lm = KNNClassifier(variant="learnable_metric", metric_iters=40, random_state=seed).fit(Xtr, ytr)
sk = KNNClassifier(variant="stat_k_selection", random_state=seed).fit(Xtr, ytr)
pc = KNNClassifier(variant="prototype_compressed", compression=0.35, random_state=seed).fit(Xtr, ytr)

fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
axes[0].bar(range(20), lm.metric_diag_, color=["C3"] * 2 + ["C0"] * 18)
axes[0].set_title("classification: learned a_j (red = 2 relevant)"); axes[0].set_xlabel("feature")
Rtr, Rte, rytr, ryte = split_data(*_synthetic("regression", seed), seed=seed, stratify=False)
lmr = KNNRegressor(variant="learnable_metric", metric_iters=40, random_state=seed).fit(Rtr, rytr)
axes[1].bar(range(20), lmr.metric_diag_, color=["C3"] * 2 + ["C0"] * 18)
axes[1].set_title("regression: learned a_j (red = 2 relevant)"); axes[1].set_xlabel("feature")
ks = sorted(sk.cv_scores_); axes[2].plot(ks, [sk.cv_scores_[k] for k in ks], "o-"); axes[2].axvline(sk.k_, color="r", ls="--")
axes[2].set_title(f"stat_k_selection LOO loss, chosen k={sk.k_}"); axes[2].set_xlabel("k")
plt.tight_layout(); plt.show()
print(f"relevant-feature weight share  classification: {lm.metric_diag_[:2].sum() / lm.metric_diag_.sum():.2f}   "
      f"regression: {lmr.metric_diag_[:2].sum() / lmr.metric_diag_.sum():.2f}   (uniform would be 0.10)")
print(f"prototype_compressed keeps {pc.n_prototypes_} of {len(ytr)} training rows")
"""),
        md("## 5. Summary and limitations"),
        code("""
for task, metric, smaller in (("classification", "accuracy", False), ("regression", "mse", True)):
    t = res[res.task == task].groupby(["dataset", "model"])[metric].mean().unstack("model")[ORDER]
    pick = t.idxmin(axis=1) if smaller else t.idxmax(axis=1)
    print(f"best {metric} per dataset:", pick.to_dict())
"""),
        md("""
**What to read from the tables.** In our local 2-seed run the learned diagonal metric gave a large gain for *regression* on the synthetic set (MSE ≈ 0.14 vs 0.56 for the baseline, relevant features up-weighted) but did **not** help *classification* there (log-loss worse than the baseline; the radial class boundary with 10 % label noise is hard for a diagonal NCA objective, and the learned weights even down-weighted the two relevant features). `prototype_compressed` and `stat_k_selection` improved log-loss on the noisy synthetic classification set, while on breast cancer `base`/sklearn were as good or better.

**Honest limitations**
- Two seeds here (3 in the paper) and four datasets: most differences are within one SD, and the real datasets are small. Runtime figures are single un-warmed fits on shared hardware — indicative only. The StatCore variants are pure NumPy with $O(nq)$ distance matrices and are not expected to be faster than sklearn's KD-tree/ball-tree implementation.
- The metric is learned on a random sub-sample of $\\le400$ points, is diagonal only (no feature interactions), and the NCA/soft-neighbour objective is non-convex (only a local optimum is reached; no restarts).
- Prototype compression is a heuristic: with a high `compression` ratio it is lossless, with a low ratio it can blur decision boundaries and noisy minority classes.
- `stat_k_selection` selects $k$ on the training data with leave-one-out, so it adds no information on shift; the $10^{-3}$ log-loss tiebreak is a heuristic.
- Probabilities from k-NN vote fractions are coarse (multiples of $1/k$); ECE/log-loss are therefore noisy and not directly comparable to smooth models.
"""),
    ]
    write("knn", c)


def build_mlp() -> None:
    c = header("mlp", "StatCore MLP: weight-decay selection, reliability head and predictive uncertainty",
               "StatCore Multilayer Perceptrons: NumPy Backpropagation, Weight-Decay Selection, Reliability Heads, and Predictive Uncertainty",
               "(11, 23)")
    c += [
        md(r"""
## 2. Method summary

Standardised inputs $h_0=z$; hidden layers $h_l=\mathrm{ReLU}(h_{l-1}W_l+b_l)$ (He initialisation $W\sim\mathcal N(0,2/n_{\text{in}})$), output $o=h_LW_{L+1}+b_{L+1}$. **Inverted dropout** with rate $p$ is applied to hidden activations in training only for `uncertainty_head`: $h\leftarrow h\odot m/(1-p)$, $m\sim\mathrm{Bernoulli}(1-p)$.

**Losses.** Classification: softmax cross-entropy $\mathcal L=-\frac1n\sum_i\log\mathrm{softmax}(o_i)_{y_i}$ (gradient $(P-T)/n$). Regression (standardised $y$): $\mathcal L=\frac1{2n}\sum_i(o_i-y_i)^2$. Heteroscedastic regression (`uncertainty_head`): two outputs $(\mu,\ell)$ with $\ell=\log\sigma^2$ clipped to $[-6,6]$:
$$\mathcal L=\frac1n\sum_i\tfrac12\Big(\ell_i+\frac{(y_i-\mu_i)^2}{e^{\ell_i}}\Big).$$

**Optimiser (AdamW-style).** Mini-batches (`batch_size`), $m\leftarrow0.9m+0.1g$, $v\leftarrow0.999v+0.001g^2$, bias-corrected $\hat m,\hat v$:
$$W\leftarrow W-\eta\,\frac{\hat m}{\sqrt{\hat v}+10^{-8}}-\eta\,\lambda\,W,$$
decoupled weight decay $\lambda$ on weight matrices only (not biases). A `validation_fraction`=15 % split drives early stopping: stop after `patience` epochs without improving the validation loss by $>10^{-5}$ and restore the best weights. Reference: sklearn `Pipeline(StandardScaler, MLP*(hidden=(32,32), alpha=1e-2, early_stopping=True,...))`.

**`base`** — fixed $\lambda$.

**`stat_regularised`** — trains one network per $\lambda\in$ `wd_grid` ($10^{-4}..1$) and keeps the one with the lowest validation loss (`wd_scores_`).

**`reliability_head`** — `head_fraction`=20 % of the data is held out from network training. A small logistic regression ($\alpha=1$) is trained on features of the held-out points to predict *whether the network is right*: $\|h_L\|$, the input energy $\overline{z^2}/\overline{z^2}_{\text{train}}$, and for classification $(\max p,\ H(p),\ p_{(1)}-p_{(2)})$, for regression $|\hat y|$. Target: correct prediction (classification) or $|{\rm error}|\le$ median error (regression). `predict_reliability(X)` returns the head's probability.

**`uncertainty_head`** — Monte-Carlo dropout with $S=$ `mc_samples` stochastic passes. Classification: $\bar p=\frac1S\sum_s\mathrm{softmax}(o_s)$, uncertainty $=H(\bar p)/\log C$. Regression: $\hat y=\frac1S\sum_s\mu_s$ and
$$\hat\sigma^2_{\text{pred}}=\underbrace{\frac1S\sum_s e^{\ell_s}}_{\text{aleatoric}}+\underbrace{\mathrm{Var}_s(\mu_s)}_{\text{epistemic}}\quad(\times\,\sigma_y^2\ \text{to original units}).$$
"""),
        md("## 3. Experiment (mirrors `mlp_experiment.py` / `_experiment_utils.py`)\n\nSame data protocol as the k-NN notebook. Networks: hidden $(32,32)$, lr $10^{-2}$, weight decay $10^{-2}$, batch 64, at most 100 epochs, patience 12, dropout 0.1, 20 MC samples, 20 % head split."),
        code(_NN_UTILS + """
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier         # baseline only
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor           # baseline only
from ml_reliability.statcore import MLPClassifier, MLPRegressor

VARIANT_NAMES = ("base", "stat_regularised", "reliability_head", "uncertainty_head")

def make_models(task, seed):
    is_class = task == "classification"
    sk_cls = SklearnMLPClassifier if is_class else SklearnMLPRegressor
    baseline = make_pipeline(StandardScaler(), sk_cls(hidden_layer_sizes=(32, 32), alpha=1e-2, batch_size=64,
                             learning_rate_init=1e-2, max_iter=100, early_stopping=True, n_iter_no_change=12, random_state=seed))
    cls = MLPClassifier if is_class else MLPRegressor
    models = {name: cls(hidden_layer_sizes=(32, 32), variant=name, learning_rate=1e-2, weight_decay=1e-2,
                        max_epochs=100, batch_size=64, patience=12, validation_fraction=0.15, dropout=0.1,
                        mc_samples=20, head_fraction=0.2, random_state=seed) for name in VARIANT_NAMES}
    return {"sklearn_standard": baseline, **models}

ORDER = ["sklearn_standard", *VARIANT_NAMES]
res = run_family(make_models, SEEDS)
print(f"MLP experiment over seeds {SEEDS} (runtime = single fit + predict, no warm-up)")
res = summarise(res, ORDER)
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, (task, metric) in zip(axes, (("classification", "log_loss"), ("regression", "mse"))):
    t = res[res.task == task].groupby(["dataset", "model"])[metric].mean().unstack("model")[ORDER]
    t.plot.bar(ax=ax, rot=15, legend=(task == "regression")); ax.set_title(f"{task}: {metric}")
    if metric == "mse": ax.set_yscale("log")
plt.tight_layout(); plt.show()
"""),
        md("## 4. Reliability head, uncertainty head and weight-decay selection\n\nDo the reliability / uncertainty scores actually separate good predictions from bad ones? We check on the held-out breast-cancer and diabetes splits."),
        code("""
def auroc(score, positive):
    # probability that a random positive outscores a random negative (ties count 1/2)
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))

seed = SEEDS[0]
Xtr, Xte, ytr, yte = split_data(load_breast_cancer().data, load_breast_cancer().target, seed, True)
kw = dict(hidden_layer_sizes=(32, 32), learning_rate=1e-2, weight_decay=1e-2, max_epochs=100, batch_size=64,
          patience=12, dropout=0.1, mc_samples=20, head_fraction=0.2, random_state=seed)
rows = []
for variant in ("base", "reliability_head", "uncertainty_head"):
    m = MLPClassifier(variant=variant, **kw).fit(Xtr, ytr)
    correct = m.predict(Xte) == yte
    rows.append(dict(task="classification", variant=variant, accuracy=correct.mean(),
                     AUROC_reliability_vs_correct=auroc(m.reliability_score(Xte), correct)))
Xtr, Xte, ytr, yte = split_data(load_diabetes().data, load_diabetes().target, seed, False)
for variant in ("reliability_head", "uncertainty_head"):
    m = MLPRegressor(variant=variant, **kw).fit(Xtr, ytr)
    err = np.abs(m.predict(Xte) - yte)
    score = m.reliability_score(Xte)
    rows.append(dict(task="regression", variant=variant, accuracy=np.nan,
                     AUROC_reliability_vs_correct=auroc(score, err <= np.median(err))))
    if variant == "uncertainty_head":
        sd = m.predict_uncertainty(Xte)
        lo, hi = m.predict(Xte) - 1.645 * sd, m.predict(Xte) + 1.645 * sd
        rows[-1]["coverage90_gaussian"] = float(np.mean((yte >= lo) & (yte <= hi)))
        rows[-1]["corr(sd,|err|)"] = float(np.corrcoef(sd, err)[0, 1])
print("AUROC 0.5 = uninformative, 1.0 = perfectly ranks trustworthy predictions first")
pd.DataFrame(rows).round(3)
"""),
        code("""
sr = MLPClassifier(variant="stat_regularised", **{k: v for k, v in kw.items() if k != "weight_decay"}).fit(*split_data(*_synthetic("classification", seed), seed, True)[::2])
ws = sorted(sr.wd_scores_)
plt.figure(figsize=(4.8, 3.4)); plt.semilogx(ws, [sr.wd_scores_[w] for w in ws], "o-"); plt.axvline(sr.weight_decay_, color="r", ls="--")
plt.xlabel("weight decay"); plt.ylabel("validation cross-entropy"); plt.title(f"stat_regularised: chosen {sr.weight_decay_:g}"); plt.show()
"""),
        md("## 5. Summary and limitations"),
        code("""
for task, metric, smaller in (("classification", "log_loss", True), ("regression", "mse", True)):
    t = res[res.task == task].groupby(["dataset", "model"])[metric].mean().unstack("model")[ORDER]
    print(f"best {metric} per dataset:", (t.idxmin(axis=1) if smaller else t.idxmax(axis=1)).to_dict())
"""),
        md("""
**What to read from the tables.** In our local 2-seed run `stat_regularised` and `uncertainty_head` gave the best regression MSE on the synthetic set (0.23 and 0.34 vs 0.40 for sklearn), whereas on the synthetic classification set the sklearn MLP had the highest accuracy. The heads are meant to *rank* predictions by trustworthiness, so judge them in the AUROC table: on breast cancer the MC-dropout score (`uncertainty_head`, AUROC ≈ 0.95) separated right from wrong predictions about as well as plain max-probability (`base`, ≈ 0.96), whereas the `reliability_head` score was **anti-informative** in that run (AUROC ≈ 0.17; the head was fitted on ~85 rows with only a handful of errors).

**Honest limitations**
- Two seeds (3 in the paper), small datasets, single architecture; differences between variants are mostly within one SD. Runtime is an un-warmed single fit.
- `reliability_head` removes 20 % of the data from network training and its head is trained on few points (~85 for breast cancer after the split, with very few errors to learn from): the AUROC is noisy, it can be below 0.5 (as above) and the head is near-constant when predictions are almost all correct. Test on one seed only here.
- MC dropout approximates epistemic uncertainty only roughly, is not calibrated by construction, and the Gaussian 90 % coverage above is a diagnostic, not a guarantee. The regression head assumes Gaussian noise.
- `stat_regularised` trains five networks (5x cost) and picks $\\lambda$ on a single validation split, which is itself noisy; early stopping already regularises, so the selected $\\lambda$ often matters little.
- The pure-NumPy training loop is single-threaded and slower than optimised libraries for large data; no GPU, no convolutional/recurrent layers.
"""),
    ]
    write("mlp", c)


def build_meta() -> None:
    c = header("meta", "StatCore meta-learning: leakage-free model combination, sklearn adapters and metrics",
               "StatCore Meta-Learning: Cross-Fitted Model Combination and NumPy Reliability Metrics",
               "(42, 7)")
    c += [
        md(r"""
## 2. Method summary

**Cross-fitting.** $K$ stratified folds. For each base model $m$ and fold $f$, the model is fitted on the other folds and predicts fold $f$, giving out-of-fold (OOF) probabilities $P^{(m)}_i\in\Delta^{C}$ for every sample. Base models used at prediction time are **refit on all data**. Every meta-model is trained on OOF predictions only, so a base model never predicts a sample it was trained on (this limits one form of leakage; it does not remove model-selection optimism or distribution shift). Models without `predict_proba` contribute one-hot predictions. Let $\ell_m$ be the OOF log-loss of model $m$.

**`performance_aware`** — softmax weights on OOF log-loss, temperature $\tau$ (`temperature`):
$$w_m=\mathrm{softmax}_m\!\Big(-\frac{\ell_m-\min_{m'}\ell_{m'}}{\tau}\Big),\qquad P(x)=\sum_m w_m P^{(m)}(x).$$

**`metrics_based`** — features built from base-model outputs: $f1_m\cdot P^{(m)}$ (OOF macro-F1 scaled), per-model $(\max P^{(m)},\ H(P^{(m)}),\ \text{top-2 margin})$, the fraction of models agreeing with the consensus (argmax of the mean), and $H(\bar P)$; logistic-regression meta-model ($\alpha=10^{-2}$).

**`oof_stacking`** — classical stacking: logistic regression ($\alpha=10^{-2}$) on the concatenated OOF probabilities $[P^{(1)},\dots,P^{(M)}]$.

**`statistical_meta`** — features $\log\mathrm{clip}(P^{(m)},10^{-3},1)$ concatenated; the final probability is the **average of three** meta-models: shrinkage LDA, a *cluster posterior* (k-means++ with $k=\max(C,\min(2C,\lfloor n/5\rfloor))$ clusters on the meta-features, class posterior per cluster with add-one smoothing), and logistic regression.

**`calibrated_stack`** — stacking logistic regression whose logits are divided by a temperature $T=e^{\tau}$, $\tau\in[-3,3]$, learned by minimising NLL on *nested* out-of-fold meta predictions ($\min(K,3)$ inner folds): $P(x)=\mathrm{softmax}(\text{logits}(x)/T)$.

### Evaluation metrics (`ml_reliability.statcore.evaluation.metrics`)
Accuracy; macro precision/recall/F1 over the union of classes; multiclass MCC (Gorodkin): $\mathrm{MCC}=\frac{c\,n-\sum_k p_kt_k}{\sqrt{(n^2-\sum p_k^2)(n^2-\sum t_k^2)}}$; log-loss $-\frac1n\sum_i\log\max(p_{i,y_i},10^{-12})$; ECE over 10 equal-width confidence bins $\sum_b\frac{|B_b|}{n}\big|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)\big|$; regression MSE/MAE/$R^2$; reconstruction MSE/MAE/relative Frobenius error; clustering inertia and number of clusters found.

### sklearn-compatible adapters
`StatCoreLogisticClassifier`, `StatCoreLDAClassifier`, `StatCoreKNNClassifier`, `StatCoreMLPClassifier`, `StatCoreMetaClassifier` (in `ml_reliability`) inherit `BaseEstimator`/`ClassifierMixin` (parameters are constructor fields → `clone`, `GridSearchCV`, pipelines work), validate input, expose `classes_`, `predict_proba`, `predict`; all numerics are done by StatCore. Only the *adapters* import scikit-learn.
"""),
        md("## 3. Experiment (mirrors `meta_experiment.py`)\n\nBase learners (all StatCore): logistic (`alpha=1e-2`), LDA (`shrinkage`), k-NN ($k=5$). 25 % stratified held-out split, 3-fold OOF. Reference baseline: sklearn `StackingClassifier` over sklearn logistic / LDA / k-NN with a logistic final estimator (`cv=3`)."),
        code("""
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier                        # baseline only
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml_reliability.statcore import KNNClassifier, LDA, LogisticRegression, MetaLearner
from ml_reliability.statcore.evaluation.metrics import classification_metrics
from ml_reliability.statcore.meta.meta_learning import VARIANTS

def datasets():
    for name, loader in (("wine", load_wine), ("breast_cancer", load_breast_cancer)):
        bunch = loader()
        yield name, np.asarray(bunch.data, dtype=float), np.asarray(bunch.target)

def base_models():
    return [("logistic", LogisticRegression(alpha=1e-2, random_state=0)),
            ("lda", LDA(variant="shrinkage", random_state=0)),
            ("knn", KNNClassifier(n_neighbors=5, variant="base", random_state=0))]

def sklearn_stack():
    return StackingClassifier(
        estimators=[("logistic", make_pipeline(StandardScaler(), SkLogisticRegression(max_iter=1500))),
                    ("lda", LinearDiscriminantAnalysis()), ("knn", KNeighborsClassifier(n_neighbors=5))],
        final_estimator=SkLogisticRegression(max_iter=1500), stack_method="predict_proba", cv=3, n_jobs=1)

rows = []
for dataset, X, y in datasets():
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
        models = {v: MetaLearner(base_models(), meta_variant=v, n_folds=3, random_state=seed).fit(Xtr, ytr) for v in VARIANTS}
        # individual StatCore base learners, for reference
        for name, est in base_models():
            models[f"base:{name}"] = est.fit(Xtr, ytr)
        models["sklearn_stacking_baseline"] = sklearn_stack().fit(Xtr, ytr)
        for name, mdl in models.items():
            rows.append(dict(dataset=dataset, seed=seed, model=name,
                             **classification_metrics(yte, mdl.predict(Xte), mdl.predict_proba(Xte), classes=mdl.classes_)))
res = pd.DataFrame(rows)
ORDER = list(res.model.unique())
cols = ["accuracy", "f1_macro", "mcc", "log_loss", "ece"]
print(f"Held-out metrics, mean over seeds {SEEDS} (25 % stratified test split)")
mean = res.groupby(["dataset", "model"], sort=False)[cols].mean()
mean.round(4)
"""),
        code("""
print("sample SD over seeds"); print(res.groupby(["dataset", "model"], sort=False)[cols].std().round(4).to_string())
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, metric in zip(axes, ("log_loss", "accuracy")):
    mean[metric].unstack("dataset").loc[ORDER].plot.barh(ax=ax); ax.set_title(metric); ax.invert_yaxis()
plt.tight_layout(); plt.show()
"""),
        md("## 4. Inside the meta-learners"),
        code("""
Xtr, Xte, ytr, yte = train_test_split(*load_wine(return_X_y=True), test_size=0.25, random_state=SEEDS[0], stratify=load_wine().target)
pa = MetaLearner(base_models(), meta_variant="performance_aware", n_folds=3, random_state=SEEDS[0]).fit(Xtr, ytr)
print("OOF log-loss per base model:", {k: round(v, 4) for k, v in pa.oof_log_loss_.items()})
print("performance_aware weights:  ", dict(zip([n for n, _ in pa.base_estimators], np.round(pa.weights_, 4).tolist())), "| best:", pa.best_estimator_)
cs = MetaLearner(base_models(), meta_variant="calibrated_stack", n_folds=3, random_state=SEEDS[0]).fit(Xtr, ytr)
print(f"calibrated_stack learned temperature T = {cs.temperature_:.3f}")
sm = MetaLearner(base_models(), meta_variant="statistical_meta", n_folds=3, random_state=SEEDS[0]).fit(Xtr, ytr)
print("statistical_meta cluster-posterior shape (clusters x classes):", sm.meta_model_["cluster_posterior"].shape)
print("OOF accuracy per base model:", {k: round(v, 3) for k, v in pa.performance_scores_.items()})
print("evaluate() on held-out data:", {k: round(v, 4) for k, v in cs.evaluate(Xte, yte).items()})
"""),
        md("## 5. sklearn-compatible adapters\n\nThe adapters plug StatCore models into the scikit-learn ecosystem (cross-validation, `clone`, grid search, pipelines). Here scikit-learn provides the *tooling* only; the models being evaluated are StatCore's."),
        code("""
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from ml_reliability import (StatCoreKNNClassifier, StatCoreLDAClassifier, StatCoreLogisticClassifier,
                            StatCoreMetaClassifier, StatCoreMLPClassifier)
from ml_reliability.statcore_estimators import STATCORE_VARIANTS

print("own variants available to the adapters:", STATCORE_VARIANTS)
X, y = load_wine(return_X_y=True)
adapters = {
    "StatCoreLogisticClassifier": StatCoreLogisticClassifier(variant="stat_regularised"),
    "StatCoreLDAClassifier": StatCoreLDAClassifier(variant="shrinkage"),
    "StatCoreKNNClassifier": StatCoreKNNClassifier(variant="stat_k_selection"),
    "StatCoreMLPClassifier": StatCoreMLPClassifier(variant="stat_regularised", max_epochs=60),
    "StatCoreMetaClassifier": StatCoreMetaClassifier(meta_variant="calibrated_stack", n_folds=3),
}
rows = []
for name, est in adapters.items():
    scores = cross_val_score(clone(est), X, y, cv=3, scoring="accuracy")
    rows.append(dict(adapter=name, cv_accuracy=scores.mean(), sd=scores.std(ddof=1)))
pd.DataFrame(rows).round(4)
"""),
        code("""
est = StatCoreMetaClassifier(meta_variant="oof_stacking", n_folds=3, random_state=0).fit(Xtr, ytr)
print("classes_:", est.classes_, "| predict_proba rows sum to 1:", np.allclose(est.predict_proba(Xte).sum(1), 1.0))
print("get_params keys:", sorted(est.get_params())[:5], "...")

grid = GridSearchCV(StatCoreLogisticClassifier(), {"variant": list(STATCORE_VARIANTS["logistic"]) + ["base"], "alpha": [1e-3, 1e-1]}, cv=3)
grid.fit(Xtr, ytr)
print("GridSearchCV best:", grid.best_params_, "cv acc =", round(grid.best_score_, 4), "| held-out acc =", round(grid.score(Xte, yte), 4))
"""),
        md("## 6. The evaluation metrics on their own\n\nAll five metric families are plain NumPy functions usable with any model."),
        code("""
from ml_reliability.statcore import (calibration_metrics, classification_metrics, clustering_metrics,
                                      reconstruction_metrics, regression_metrics, KMeans, PCA, RidgeRegression)
proba = est.predict_proba(Xte); pred = est.classes_[proba.argmax(1)]
print("classification:", {k: round(v, 4) for k, v in classification_metrics(yte, pred, proba, classes=est.classes_).items()})
print("calibration:   ", {k: round(v, 4) for k, v in calibration_metrics(yte, proba, n_bins=10, classes=est.classes_).items()})
Xs = (X - X.mean(0)) / X.std(0)
reg = RidgeRegression().fit(Xs[:, 1:], Xs[:, 0])
print("regression:    ", {k: round(v, 4) for k, v in regression_metrics(Xs[:, 0], reg.predict(Xs[:, 1:])).items()})
pca = PCA(n_components=2).fit(Xs)
print("reconstruction:", {k: round(v, 4) for k, v in reconstruction_metrics(Xs, pca.inverse_transform(pca.transform(Xs))).items()})
km = KMeans(n_clusters=3, variant="better_init", random_state=0).fit(Xs)
print("clustering:    ", clustering_metrics(Xs, km.labels_, km.cluster_centers_))
"""),
        md("## 7. Summary and limitations"),
        code("""
summary = mean["log_loss"].unstack("dataset").loc[ORDER].round(4)
summary["rank_wine"] = summary["wine"].rank(); summary["rank_breast_cancer"] = summary["breast_cancer"].rank()
print("Held-out log-loss (lower is better) and ranks:"); print(summary.to_string())
"""),
        md("""
**What to read from the tables.** In our local 2-seed run `calibrated_stack` had the lowest held-out log-loss on both datasets (ahead of sklearn's `StackingClassifier`), `oof_stacking` was close, and `statistical_meta`/`metrics_based` were worse on wine. Note that the shrinkage-LDA base learner is poor on these *unscaled* features (OOF accuracy ≈ 0.77 on wine), which is why `performance_aware` puts almost all weight on the logistic model: stacking is only as good as its base learners.

**Honest limitations**
- Both datasets are small and nearly separable (wine 178 rows, breast cancer 569), accuracy is close to ceiling for every method, and only two seeds are used here (the paper reports a single seed 42): the ranking among meta variants is **not statistically established**.
- The sklearn baseline uses default `LinearDiscriminantAnalysis` and a different logistic regularisation, so the comparison is "natural defaults on each side", not an identical-hyper-parameter test.
- OOF stacking removes one leakage path but not model-selection optimism: the number of folds, base learners and variant were chosen without nested evaluation here. Cross-fitting costs $K+1$ fits per base model.
- `statistical_meta` and `metrics_based` have many meta-features relative to $n$ on small data; `calibrated_stack` learns one scalar from OOF logits that are themselves noisy at small $n$.
- Metrics: ECE with 10 bins on ~45–143 test rows is very noisy; MCC/F1 are reported on a single split. None of this covers distribution shift.
- The adapters need scikit-learn at import time (the numeric core does not); `MLP` adapters are comparatively slow.
"""),
    ]
    write("meta", c)


# --- MORE NOTEBOOKS BELOW ---
if __name__ == "__main__":
    build_pca()
    build_lda()
    build_kmeans()
    build_linear()
    build_knn()
    build_mlp()
    build_meta()