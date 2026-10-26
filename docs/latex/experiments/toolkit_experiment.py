"""Measured experiments for the toolkit paper section and README. Run from the repo root:

    PYTHONPATH=src python docs/latex/experiments/toolkit_experiment.py
"""
import json
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd

from ml_reliability import (ActiveLearningStudy, FeatureConstructor, ab_test, plot_result, power_analysis,
                            root_cause, time_series_analysis)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
out = {}


def timed(name, fn):
    t = time.time()
    r = fn()
    out.setdefault("seconds", {})[name] = round(time.time() - t, 2)
    return r


# 1. power analysis: three regimes
rng = np.random.default_rng(0)
n = 500
X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
regimes = {
    "signal": 2 * X.a - X.b + 0.5 * rng.normal(size=n),
    "pure_noise": rng.normal(size=n),
}
Xo = pd.DataFrame(rng.normal(size=(60, 30)), columns=[f"f{i}" for i in range(30)])
regimes_X = {"signal": X, "pure_noise": X, "overfit_n60_p30": Xo}
regimes["overfit_n60_p30"] = rng.normal(size=60)
out["power"] = {}
for k, y in regimes.items():
    r = timed("power_" + k, lambda: power_analysis(regimes_X[k], np.asarray(y)))
    out["power"][k] = {"ridge_cv": r["predictive"]["ridge_cv_skill"], "knn_cv": r["predictive"]["knn_cv_skill"],
                       "in_sample": r["explanatory"]["in_sample_skill"], "gap": r["gap_explanatory_minus_predictive"],
                       "verdict": r["verdict"], "min_detectable_f2": r["statistical_power"]["min_detectable_f2"],
                       "required_n_medium": r["statistical_power"]["required_n"]["medium_f2_0.15"]}
plot_result(power_analysis(X, np.asarray(regimes["signal"]))).savefig(os.path.join(FIG, "toolkit_power.png"), dpi=150)

# 2. A/B test calibration: false-positive rate under the null, power under a real lift
fp = 0
tp = 0
reps = 200
for i in range(reps):
    r = np.random.default_rng(100 + i)
    g = np.repeat(["A", "B"], 1000)
    null = pd.DataFrame({"g": g, "y": (r.random(2000) < 0.10).astype(int)})
    alt = pd.DataFrame({"g": g, "y": np.r_[r.random(1000) < 0.10, r.random(1000) < 0.14].astype(int)})
    fp += ab_test(null, "g", "y")["significant"]
    tp += ab_test(alt, "g", "y")["significant"]
out["ab_test"] = {"reps": reps, "false_positive_rate_alpha_0.05": fp / reps, "power_10pct_to_14pct_n1000_per_arm": tp / reps}

# 3. root cause recovery
hits = 0
for i in range(20):
    r = np.random.default_rng(200 + i)
    m = 3000
    df = pd.DataFrame({"region": r.choice(list("nsew"), m), "age": r.integers(18, 80, m), "plan": r.choice(list("ab"), m),
                       "noise": r.normal(size=m)})
    p = np.where((df.region == "s") & (df.age > 60), 0.6, 0.05)
    df["churn"] = (r.random(m) < p).astype(int)
    segs = root_cause(df, "churn")["segments"]
    hits += bool(segs and "region == s" in segs[0]["segment"])
out["root_cause"] = {"reps": 20, "planted_segment_ranked_first": hits / 20}

# 4. time series
r = np.random.default_rng(0)
m = 365
t = pd.date_range("2022-01-01", periods=m)
v = 0.03 * np.arange(m) + 3 * np.sin(2 * np.pi * np.arange(m) / 7) + r.normal(0, 0.6, m)
v[220:] += 3
ts = timed("time_series", lambda: time_series_analysis(pd.DataFrame({"t": t, "v": v}), "t", "v"))
out["time_series"] = {"trend_p": ts["trend"]["p_value"], "sen_slope_true_0.03": ts["trend"]["sen_slope_per_step"],
                      "period_true_7": ts["seasonality"]["dominant_period"],
                      "seasonal_strength": ts["decomposition"]["seasonal_strength"],
                      "change_point_index_true_220": ts["change_point"]["index"], "stationary": ts["stationarity"].get("stationary")}
plot_result(ts).savefig(os.path.join(FIG, "toolkit_timeseries.png"), dpi=150)

# 5. feature construction: planted ratio, planted bin, pure noise
out["construction"] = {}
for scen in ("planted_ratio", "planted_step_heavy_tail", "noise"):
    gains, recipes = [], []
    for i in range(8):
        r = np.random.default_rng(300 + i)
        nn = 600
        D = pd.DataFrame({"price": r.lognormal(3, 0.6, nn), "area": r.lognormal(4, 0.5, nn), "age": r.normal(size=nn),
                          "z": r.normal(size=nn)})
        if scen == "planted_ratio":
            y = np.log(D.price / D.area) * 3 + 0.2 * r.normal(size=nn)
        elif scen == "planted_step_heavy_tail":
            low = r.normal(0, 1, int(nn * 0.6))
            D = D.assign(b=r.permutation(np.r_[low, 8 + r.lognormal(2, 1, nn - len(low))]))
            y = 2.0 * (D.b > 4) + 0.3 * r.normal(size=nn)
        else:
            y = r.normal(size=nn)
        rep = FeatureConstructor(random_state=i).fit(D, np.asarray(y)).report
        gains.append(rep["holdout_gain"])
        recipes.append([x["name"] for x in rep["recipes"]])
    out["construction"][scen] = {"mean_holdout_gain": float(np.mean(gains)), "runs_with_any_feature": int(sum(bool(x) for x in recipes)),
                                 "example_recipes": recipes[0]}
rep = FeatureConstructor(random_state=0).fit(
    pd.DataFrame({"price": np.random.default_rng(1).lognormal(3, .6, 600), "area": np.random.default_rng(2).lognormal(4, .5, 600)}),
    np.log(np.random.default_rng(1).lognormal(3, .6, 600) / np.random.default_rng(2).lognormal(4, .5, 600)) * 3).report
plot_result(rep).savefig(os.path.join(FIG, "toolkit_construction.png"), dpi=150)

# 6. active learning: nonlinear (two moons-like rings) and linear data, three learners
def rings(n, seed):
    r = np.random.default_rng(seed)
    a = r.uniform(0, np.pi, n // 2)
    c1 = np.c_[np.cos(a), np.sin(a)] + 0.2 * r.normal(size=(n // 2, 2))
    c2 = np.c_[1 - np.cos(a), 0.5 - np.sin(a)] + 0.2 * r.normal(size=(n // 2, 2))
    X = np.r_[c1, c2]
    X = np.c_[X, r.normal(size=(n, 4))]
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(6)]), np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)


def imbalanced(n, seed):
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
    y = ((X.x0 + X.x1 + 0.5 * r.normal(size=n)) > 1.6).astype(int).to_numpy()
    return X, y


out["active_learning"] = {}
for dname, gen in (("rings_nonlinear", rings), ("imbalanced_linear", imbalanced)):
    Xa, ya = gen(1200, 0)
    for learner in ("logistic", "knn", "lda"):
        rep = timed(f"al_{dname}_{learner}", lambda: ActiveLearningStudy(learner=learner, n_steps=8, n_repeats=5).run(Xa, ya))
        out["active_learning"][f"{dname}/{learner}"] = {
            "ceiling": rep["ceiling_skill"], "ranking": rep["ranking"],
            "alc": {s: round(v["area_under_curve"], 4) for s, v in rep["strategies"].items()},
            "gain_vs_random": {s: [round(v["alc_gain_vs_random_mean"], 4), round(v["alc_gain_vs_random_std"], 4)]
                               for s, v in rep["strategies"].items() if "alc_gain_vs_random_mean" in v}}
        if dname == "rings_nonlinear" and learner == "knn":
            plot_result(rep).savefig(os.path.join(FIG, "toolkit_active_learning.png"), dpi=150)

with open(os.path.join(HERE, "toolkit-results.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
