"""Build docs/colab/automl.ipynb (family-based AutoML comparison).

Run:  python docs/colab/_build_automl_colab.py
The notebook embeds docs/latex/experiments/automl_comparison.py via %%writefile
so the notebook and the paper protocol cannot drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = (HERE.parent / "latex" / "experiments" / "automl_comparison.py").read_text(
    encoding="utf-8")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# ReliableAutoML vs. other AutoML frameworks (family-based comparison)

This notebook runs the comparison described in the paper *automl-comparison*
(`docs/latex/automl-comparison.tex`). Competitors are grouped by method family:

| Family | Frameworks | Role |
|---|---|---|
| (a) stacking / meta-ensemble | AutoGluon-Tabular, H2O AutoML, auto-sklearn | closest family to ReliableAutoML (meta-learned ensembles) |
| (b) cost-frugal search | FLAML | contrast: single tuned model, small budget |
| (c) evolutionary pipeline search | TPOT | contrast: genetic programming over sklearn pipelines |

**Outputs compared** (not just accuracy): selected model, held-out balanced accuracy / accuracy /
log-loss, wall-clock time, leaderboard size, and which reports/artefacts each framework returns.

**Protocol.** For each dataset $D$ and seed $s$ one stratified split
$D=D_{\text{dev}}\cup D_{\text{test}}$ (25% test, `random_state=s`) is drawn. Every framework sees
the same $D_{\text{dev}}$ and is scored once on the same $D_{\text{test}}$:

$$\mathrm{BA}=\frac1C\sum_{c=1}^{C}\frac{TP_c}{TP_c+FN_c},\qquad
\mathrm{LL}=-\frac1n\sum_{i=1}^{n}\log \hat p_{i,y_i}$$

(probabilities clipped to $[10^{-15},1]$ and renormalised). Competitors get a wall-clock budget
`BUDGET` seconds; ReliableAutoML has no time-budget parameter, so its measured wall time is
reported instead. Budgets, internal metrics and CV schemes differ between frameworks (see the
paper's feature table), so the numbers compare *delivered outputs under a stated budget*, not a
controlled ablation. Report mean ± std over seeds; with few datasets/seeds the result is
descriptive.
""")

md("## 1. Parameters (edit this cell)")
code(r'''
# ---- PARAMETER CELL -------------------------------------------------------
# StatCore is not yet pushed/published, so the GitHub install below may fail or
# give an older ReliableML. If so set INSTALL_MODE = "zip" and upload a zip of
# the repository (containing pyproject.toml and the automl/ folder) in step 2b.
INSTALL_MODE = "git"            # "git" or "zip"
REPO_GIT = "git+https://github.com/Jabualani/ML-Reliabilty.git"
AUTOML_GIT = "git+https://github.com/Jabualani/ML-Reliabilty.git#subdirectory=automl"

FRAMEWORKS = ["reliableml", "sklearn_logistic", "autogluon", "h2o", "autosklearn",
              "flaml", "tpot"]
DATASETS = ["synthetic_mixed", "wine", "breast_cancer", "digits"]
SEEDS = [42, 43, 44]
BUDGET = 60                     # seconds per competitor fit
INSTALL_COMPETITORS = True
# ---------------------------------------------------------------------------
''')

md("## 2a. Install ReliableML and ReliableML-AutoML")
code(r'''
import subprocess, sys
def pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                       capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + " ".join(args))
    if r.returncode != 0:
        print(r.stdout[-800:], r.stderr[-800:])
    return r.returncode == 0

if INSTALL_MODE == "git":
    pip(REPO_GIT)
    pip(AUTOML_GIT)
else:
    print("INSTALL_MODE='zip': run cell 2b")
''')
md("## 2b. Fallback: upload a zip of the repository\nOnly used when `INSTALL_MODE = \"zip\"`.")
code(r'''
if INSTALL_MODE == "zip":
    import zipfile, pathlib
    from google.colab import files
    up = files.upload()
    name = next(iter(up))
    zipfile.ZipFile(name).extractall("repo_src")
    root = next(p.parent for p in pathlib.Path("repo_src").rglob("pyproject.toml")
                if p.parent.name != "automl")
    pip(str(root))
    pip(str(root / "automl"))
''')

md("""
## 3. Install competitors (each independently; failures are recorded, not fatal)

auto-sklearn is Linux-only and its current releases require an older Python (3.10 or below);
a recent Colab runtime with Python 3.12 will usually fail to install it. In that case use a
Python-3.10 runtime or accept an `error` row for `autosklearn` - do not substitute numbers.
Installing several frameworks into one runtime can produce version conflicts; if a framework
fails, restart the runtime and install only that one.
""")
code(r'''
installed = {}
if INSTALL_COMPETITORS:
    pkgs = {"flaml": "flaml[automl]", "tpot": "tpot", "autogluon": "autogluon.tabular",
            "h2o": "h2o", "autosklearn": "auto-sklearn"}
    for fw, pkg in pkgs.items():
        if fw in FRAMEWORKS:
            installed[fw] = pip(pkg)
print(installed)
''')

md("## 4. Experiment script (identical to `docs/latex/experiments/automl_comparison.py`)")
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": ("%%writefile automl_comparison.py\n" + SCRIPT).splitlines(keepends=True)})

md("""
## 5. Run

Each framework runs in its own subprocess so a crash in one cannot abort the others.
Expected runtime is roughly `len(DATASETS) * len(SEEDS) * BUDGET` seconds per competitor.
""")
code(r'''
import json, subprocess, sys, pathlib
parts = []
for fw in FRAMEWORKS:
    out = f"results_{fw}.json"
    cmd = [sys.executable, "automl_comparison.py", "--frameworks", fw,
           "--datasets", *DATASETS, "--seeds", *map(str, SEEDS),
           "--budget", str(BUDGET), "--output", out]
    print(">>", fw, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:]); 
    if r.returncode != 0:
        print(r.stderr[-1500:])
    if pathlib.Path(out).exists():
        parts.append(json.loads(pathlib.Path(out).read_text()))
records = [rec for p in parts for rec in p["records"]]
metadata = parts[0]["metadata"] if parts else {}
pathlib.Path("automl_comparison_results.json").write_text(
    json.dumps({"metadata": metadata, "records": records}, indent=1))
print(len(records), "records")
''')

md("""
## 6. Results table (paste this back into the paper)

Columns match the paper tables: balanced accuracy, accuracy, log-loss, wall seconds (mean ± std
over seeds) and the number of successful / failed runs. Failed runs are listed separately and are
excluded from the mean.
""")
code(r'''
import pandas as pd, numpy as np
df = pd.DataFrame(records)
if "error" not in df: df["error"] = np.nan
ok = df[df["error"].isna()]
def ms(x): return f"{x.mean():.4f} ± {x.std(ddof=1) if len(x) > 1 else float('nan'):.4f}"
rows = []
for (ds, fw), g in ok.groupby(["dataset", "framework"]):
    rows.append({"dataset": ds, "framework": fw, "n_ok": len(g),
                 "n_failed": int(((df.dataset == ds) & (df.framework == fw) & df.error.notna()).sum()),
                 "balanced_accuracy": ms(g.balanced_accuracy), "accuracy": ms(g.accuracy),
                 "log_loss": ms(g.log_loss), "wall_seconds": f"{g.wall_seconds.mean():.1f} ± {g.wall_seconds.std(ddof=1) if len(g)>1 else float('nan'):.1f}"})
table = pd.DataFrame(rows).sort_values(["dataset", "framework"])
table.to_csv("automl_comparison_table.csv", index=False)
print(table.to_string(index=False))
print("\nFAILED RUNS")
print(df[df.error.notna()][["framework", "dataset", "seed", "error"]].drop_duplicates(["framework", "error"]).to_string(index=False)
      if df.error.notna().any() else "none")
''')

md("## 7. Selected models and reports returned (qualitative outputs)")
code(r'''
for fw, g in ok.groupby("framework"):
    sel = g["info"].apply(lambda d: d.get("selected_model") if isinstance(d, dict) else None)
    print(f"{fw:18s}", sorted(set(map(str, sel)))[:6])
raml = [r for r in records if r["framework"] == "reliableml" and "info" in r]
if raml:
    info = raml[0]["info"]
    print("\nReliableAutoML extra outputs:", {k: info[k] for k in info if k != "leaderboard"})
''')

md("""
## 8. Download

Download `automl_comparison_results.json` and `automl_comparison_table.csv` and paste the table
into the *to be filled from the Colab run* cells of `automl-comparison.tex`.
""")
code(r'''
from google.colab import files
files.download("automl_comparison_results.json")
files.download("automl_comparison_table.csv")
''')

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}}
out = HERE / "automl.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
