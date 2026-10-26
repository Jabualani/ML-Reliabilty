from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml_reliability import *
import json
d=load_wine(as_frame=True); Xtr,Xte,ytr,yte=train_test_split(d.data.iloc[:,:6],d.target,test_size=.4,random_state=0,stratify=d.target)
m=ReliablePipeline(RandomForestClassifier(50,random_state=0),plot_dir="p",bias_variance_folds=3,bias_variance_repeats=1).fit(Xtr,ytr)
e=m.evaluate(Xte,yte).to_dict()
c=e["classification"]
print(c["labels"],c["confusion_matrix"])
print(json.dumps(e["statistical_tests"],default=str)[:1500])
print(e["bias_variance"], e["fairness"])
