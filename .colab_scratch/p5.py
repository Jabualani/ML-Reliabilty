import numpy as np, json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ml_reliability import *
d=load_breast_cancer(as_frame=True); X=d.data.iloc[:,:5]; y=d.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.4,random_state=0,stratify=y)
g=np.where(Xte["mean radius"]>Xte["mean radius"].median(),"large","small")
m=ReliablePipeline(LogisticRegression(max_iter=2000),plot_dir="p",bias_variance_folds=3,bias_variance_repeats=1).fit(Xtr,ytr)
f=m.evaluate(Xte,yte,sensitive_features={"size":g},positive_label=1).to_dict()["fairness"]
print(json.dumps(f,indent=1,default=str)[:3500])
