#!/usr/bin/env python3.8
"""A7: stronger heads on the SAME frozen DFN-CLIP features (no new data).
Tests whether the tuned linear probe (85.5%) is already near the frozen-feature
ceiling, or whether a nonlinear RBF-SVM / nearest-centroid head recovers more.
Matches the canonical protocol: 20% stratified validation (seed 42), select on
val, refit on 100% train, evaluate once on the 138 test images.
"""
import numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
z=np.load('data/clip_features_clean.npz')
Xtr,ytr,Xte,yte=z['train_feats'].astype(np.float32),z['train_labels'],z['test_feats'].astype(np.float32),z['test_labels']
# nearest centroid (no tuning)
# NOTE: `**` binds tighter than `-`, so the original `(X[:,None,:]-cent[None,:,:]**2)`
# parsed as `X - cent**2` and reduced to argmax ||cent||^2 -- a constant predictor
# that scored 8.7% (chance 6.25%). The parentheses below are load-bearing.
# An absent class would yield a NaN centroid, which silently poisons nca() for
# every point (the NaN just propagates into the argmin, no warning).
_present = [c for c in range(16) if (ytr == c).any()]
assert len(_present) == 16, f"expected 16 classes with training samples, got {len(_present)}"
cent=np.stack([Xtr[ytr==c].mean(0) for c in _present])
def nca(X): return ((X[:,None,:]-cent[None,:,:])**2).sum(-1).argmin(1)
# RBF-SVM: select C on 20% stratified val (seed 42), refit on 100%, eval on test
idx_sub, idx_val = train_test_split(np.arange(len(ytr)), test_size=0.2, random_state=42, stratify=ytr)
Xs,Xv,ys,yv = Xtr[idx_sub], Xtr[idx_val], ytr[idx_sub], ytr[idx_val]
Cgrid=[0.5,1,5,10,50]
best=None
for C in Cgrid:
    m=SVC(C=C,kernel='rbf',gamma='scale',random_state=42).fit(Xs,ys)
    va=accuracy_score(yv,m.predict(Xv))
    if best is None or va>best[1]: best=(C,va)
C_sel=best[0]
m=SVC(C=C_sel,kernel='rbf',gamma='scale',random_state=42).fit(Xtr,ytr)
pt,ba=accuracy_score(yte,m.predict(Xte)),balanced_accuracy_score(yte,m.predict(Xte))
pn,ban=accuracy_score(yte,nca(Xte)),balanced_accuracy_score(yte,nca(Xte))
res={'rbf_svm':{'C_selected':C_sel,'val_acc':round(best[1],4),'test_acc':round(float(pt),4),'test_balanced_acc':round(float(ba),4)},
     'nearest_centroid':{'test_acc':round(float(pn),4),'test_balanced_acc':round(float(ban),4)},
     'reference_linear_C300':{'test_acc':0.855,'test_balanced_acc':0.774}}
print(json.dumps(res,indent=1))
json.dump(res,open('data/heads_a7.json','w'))
print("\nSaved data/heads_a7.json")
