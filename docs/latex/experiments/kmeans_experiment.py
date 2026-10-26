"""Reproducible StatCore KMeans comparison; run from repository root."""
from pathlib import Path
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans as ReferenceKMeans
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT/"src"))
from ml_reliability.statcore.kmeans.kmeans import KMeans,VARIANTS
SEEDS=(11,23,37,51,71); METHODS=("reference",)+VARIANTS
def standardize(train,test):
    mean=train.mean(0); scale=train.std(0); scale[scale<=1e-12]=1.0
    return (train-mean)/scale,(test-mean)/scale
def synthetic(seed):
    rng=np.random.default_rng(seed)
    means=np.array([[-3,-2,0,1],[3,-2,1,-1],[0,3,-1,0]])
    clean=np.vstack([rng.normal(loc=m,scale=.48,size=(100,4)) for m in means]); labels=np.repeat(np.arange(3),100)
    order=rng.permutation(len(clean)); tr,te=order[:240],order[240:]; train=clean[tr].copy()
    idx=rng.choice(len(train),18,replace=False); train[idx]+=rng.normal(0,6,(18,4))
    xtr,xte=standardize(train,clean[te]); return xtr,xte,labels[te],labels[tr]
def iris(seed):
    data=load_iris(); order=np.random.default_rng(seed).permutation(len(data.data)); split=int(.75*len(order))
    tr,te=order[:split],order[split:]; xtr,xte=standardize(data.data[tr],data.data[te])
    return xtr,xte,data.target[te],data.target[tr]
def ari(a,b):
    _,ai=np.unique(a,return_inverse=True); _,bi=np.unique(b,return_inverse=True)
    tab=np.zeros((ai.max()+1,bi.max()+1),dtype=np.int64); np.add.at(tab,(ai,bi),1)
    c2=lambda x:x*(x-1)/2.0
    index=c2(tab).sum(); rows=c2(tab.sum(1)).sum(); cols=c2(tab.sum(0)).sum(); total=c2(len(a))
    expected=rows*cols/total if total else 0.; maximum=.5*(rows+cols)
    return float((index-expected)/(maximum-expected)) if maximum!=expected else 1.
def silhouette(X,labels):
    classes=np.unique(labels)
    if len(classes)<2:return 0.
    D=np.sqrt(np.maximum((X**2).sum(1)[:,None]+(X**2).sum(1)[None,:]-2*X@X.T,0)); vals=np.zeros(len(X))
    for i in range(len(X)):
        own=labels==labels[i]; count=int(own.sum()); a=D[i,own].sum()/(count-1) if count>1 else 0.
        b=min(D[i,labels==c].mean() for c in classes if c!=labels[i])
        vals[i]=(b-a)/max(a,b,1e-12) if count>1 else 0.
    return float(vals.mean())
def run(method,xtr,xte,k,seed):
    if method=="reference":
        model=ReferenceKMeans(n_clusters=k,n_init=5,random_state=seed,max_iter=100,tol=1e-6).fit(xtr)
        labels=model.predict(xte); centers=model.cluster_centers_
    else:
        kw=dict(variant=method,random_state=seed,max_iter=100,n_init=5,tol=1e-6)
        kw.update(k_range=(2,6),criterion="bic") if method=="stat_model_selection" else kw.update(n_clusters=k)
        model=KMeans(**kw).fit(xtr); labels=model.predict(xte); centers=model.cluster_centers_
    return labels,float(np.sum((xte-centers[labels])**2))
def main():
    out=Path(__file__).resolve().parents[1]/"build"; out.mkdir(exist_ok=True); results={}
    for ds,factory in (("synthetic",synthetic),("iris",iris)):
        for method in METHODS:
            vals=[]
            for seed in SEEDS:
                xtr,xte,truth,_=factory(seed); labels,sse=run(method,xtr,xte,3,seed)
                vals.append((ari(truth,labels),silhouette(xte,labels),sse))
            a=np.asarray(vals); results[(ds,method)]=[(float(a[:,j].mean()),float(a[:,j].std(ddof=1))) for j in range(3)]
    print("KMeans held-out metrics: mean +/- sample SD over seeds",SEEDS)
    print(f"{'dataset':<12} {'method':<25} {'ARI':>21} {'silhouette':>21} {'SSE':>21}")
    for ds in ("synthetic","iris"):
        for method in METHODS:
            metrics=results[(ds,method)]; cells=[]
            for name,(mean,sd) in zip(("ari","sil","sse"),metrics):
                cells.append(f"{mean:.4f}\\pm{sd:.4f}")
                (out/f"km-{ds}-{method.replace(chr(95), chr(45))}-{name}.tex").write_text(f"${mean:.4f}\\pm{sd:.4f}$",encoding="utf-8")
            print(f"{ds:<12} {method:<25} {cells[0]:>21} {cells[1]:>21} {cells[2]:>21}")
if __name__=="__main__":main()
