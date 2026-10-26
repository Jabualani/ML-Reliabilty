"""Reproducible StatCore PCA comparison; run from repository root."""
from pathlib import Path
import sys
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA as ReferencePCA
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from ml_reliability.statcore.pca.pca import PCA, VARIANTS
SEEDS = (11, 23, 37, 51, 71)
METHODS = ("reference",) + VARIANTS
def standardize(train, test):
    mean=train.mean(0); scale=train.std(0); scale[scale<=1e-12]=1.0
    return (train-mean)/scale, (test-mean)/scale
def synthetic(seed):
    rng=np.random.default_rng(seed); latent=rng.normal(size=(400,2))
    loading=np.array([[1.5,0],[0,1.2],[.8,.3],[0,.7],[.4,-.4],[.2,.5]])
    clean=latent@loading.T+.12*rng.normal(size=(400,6)); order=rng.permutation(400)
    tr,te=order[:320],order[320:]; train=clean[tr].copy()
    idx=rng.choice(len(train),24,replace=False); train[idx]+=rng.normal(0,8,(24,6))
    xtr,xte=standardize(train,clean[te]); y=(latent[tr,0]>0).astype(int)
    return xtr,xte,y
def wine(seed):
    data=load_wine(); order=np.random.default_rng(seed).permutation(len(data.data)); split=int(.75*len(order))
    tr,te=order[:split],order[split:]; xtr,xte=standardize(data.data[tr],data.data[te])
    return xtr,xte,data.target[tr]
def score(method,xtr,xte,y,seed):
    if method=="reference":
        model=ReferencePCA(n_components=2,svd_solver="full").fit(xtr); recon=model.inverse_transform(model.transform(xte))
    else:
        model=PCA(n_components=2,variant=method,random_state=seed).fit(xtr,y=y); recon=model.inverse_transform(model.transform(xte))
    return float(np.mean((xte-recon)**2))
def main():
    out=Path(__file__).resolve().parents[1]/"build"; out.mkdir(exist_ok=True)
    print("PCA held-out reconstruction MSE: mean +/- sample SD over seeds",SEEDS)
    print(f"{'dataset':<12} {'method':<25} {'mean':>12} {'sd':>12}")
    for ds,factory in (("synthetic",synthetic),("wine",wine)):
        for method in METHODS:
            values=[score(method,*factory(seed),seed) for seed in SEEDS]
            mean=float(np.mean(values)); sd=float(np.std(values,ddof=1))
            print(f"{ds:<12} {method:<25} {mean:12.6f} {sd:12.6f}")
            (out/f"pca-{ds}-{method.replace(chr(95), chr(45))}.tex").write_text(f"${mean:.4f}\\pm{sd:.4f}$",encoding="utf-8")
if __name__=="__main__": main()
