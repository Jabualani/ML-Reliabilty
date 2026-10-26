import time;t0=time.time()
import numpy as np, pandas as pd, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml_reliability import audit_model_reasons
rng=np.random.default_rng(0); n=700
X=pd.DataFrame(rng.normal(size=(n,4)),columns=["signal","weak","noise","negctl"])
y=(rng.random(n)<1/(1+np.exp(-(1.5*X.signal+.5*X.weak)))).astype(int)
X["leak"]=y+rng.normal(scale=.3,size=n)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.4,random_state=0)
m=RandomForestClassifier(25,random_state=0).fit(Xtr,ytr)
g=np.where(Xte.signal>0,"pos","neg"); t=np.tile(["t1","t2","t3"],len(Xte))[:len(Xte)]
r=audit_model_reasons(m,Xte,yte,positive_label=1,bootstrap_samples=100,explanation_samples=100,max_explained_instances=20,groups={"g":g},time_periods=t,counterfactuals={"signal":np.zeros(len(Xte))},negative_control_features=("negctl",),unavailable_features=("leak",))
d=r.to_dict()
print(list(d))
print(json.dumps(d["predictive_validity"])[:600])
print(json.dumps(d["feature_reliance"][:2])[:900])
print(json.dumps(d["stability"])[:1200])
print(json.dumps(d["counterfactual_sensitivity"])[:600])
print(json.dumps(d["logistic_association"])[:1000])
print(d["warnings"])

print('secs',time.time()-t0)
