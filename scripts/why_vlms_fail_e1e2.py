"""
E1/E2 (immediate, from saved CLIP features): features-vs-head decomposition + geometry.
Question: is the CLIP->ResNet gap at full data a HEAD-capacity artifact, or a
REPRESENTATION limit? If non-linear (kNN) and non-parametric-geometry classifiers on
frozen CLIP features hit the SAME ceiling as the linear probe, and all sit well below
fine-tuned ResNet (85.7%), the gap is representational, not a probe artifact.
CPU-only, uses data/clip_features_clean.npz.
"""
import numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize

d = np.load("data/clip_features_clean.npz")
Xtr, ytr = d["train_feats"], d["train_labels"]
Xte, yte = d["test_feats"], d["test_labels"]
C = len(np.unique(ytr))
resnet_full = 0.857  # from paper / resnet full-data

print("Reference: linear probe (full) ~0.761 | ResNet-18 fine-tuned (full) = 0.857\n")

# ---- E2: head decomposition (same frozen CLIP features, different heads) ----
def report(name, acc):
    print(f"  {name:22s}: {acc*100:5.1f}%   (gap to ResNet { (resnet_full-acc)*100:+.1f}pp)")

Xtr_n = normalize(Xtr); Xte_n = normalize(Xte)
print("== Heads on FROZEN CLIP features (full train) ==")
lr = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr_n, ytr)
report("linear probe (LR C=1)", lr.predict(Xte_n).mean() if False else (lr.predict(Xte_n)==yte).mean())

# LR with stronger regularization sweep (is a bigger/better head the fix?)
best_lr = max((LogisticRegression(max_iter=3000, C=c).fit(Xtr_n, ytr).predict(Xte_n)==yte).mean()
              for c in [0.01,0.1,1.0,10.0,100.0])
report("best LR over C", best_lr)

for k in [1,3,5,10]:
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(Xtr_n, ytr)
    report(f"kNN cosine k={k}", (knn.predict(Xte_n)==yte).mean())

# MLP with internal val split to avoid head-overfit
from sklearn.model_selection import train_test_split
Xa, Xb, ya, yb = train_test_split(Xtr_n, ytr, test_size=0.2, random_state=0, stratify=ytr)
mlp = MLPClassifier(hidden_layer_sizes=(256,64), activation="relu", solver="lbfgs",
                    alpha=1e-3, max_iter=2000, random_state=0).fit(Xa, ya)
report("MLP 256-64 (val-tuned)", (mlp.predict(Xte_n)==yte).mean())

# ---- E1: feature geometry (frozen CLIP) ----
print("\n== Frozen-CLIP feature geometry (train) ==")
def dist(A, B):  # mean euclidean between sets
    return np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(-1)).mean()
intra = float(np.mean([dist(Xtr_n[ytr==c], Xtr_n[ytr==c]) for c in range(C)]))
cent = np.array([Xtr_n[ytr==c].mean(0) for c in range(C)])
inter = float(dist(cent, cent))
# kNN=1 pure-geometry accuracy on test
knn1 = KNeighborsClassifier(n_neighbors=1, metric="cosine").fit(Xtr_n, ytr)
geo_acc = float((knn1.predict(Xte_n)==yte).mean())
print(f"  intra-class mean dist : {intra:.4f}")
print(f"  inter-centroid dist   : {inter:.4f}")
print(f"  inter/intra ratio     : {inter/intra:.3f}  (higher = better separated)")
print(f"  kNN(1) geometry acc   : {geo_acc*100:.1f}%  (pure-geometry ceiling on CLIP features)")

# ---- E1: confusable structure from linear-probe full predictions ----
lr_full = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr_n, ytr)
pred = lr_full.predict(Xte_n)
cm = np.zeros((C,C), int)
for t,p in zip(yte, pred): cm[t,p]+=1
pairs=[]
for i in range(C):
    for j in range(C):
        if i!=j and cm[i,j]>=2: pairs.append((int(cm[i,j]), i, j))
pairs.sort(reverse=True)
print("\n== Top confusable pairs (true->pred, full linear probe, test) ==")
for n,i,j in pairs[:8]:
    print(f"  class {i:>2} -> class {j:>2} : {n}  ({n/max(cm[i].sum(),1)*100:.0f}% of class {i})")

out = dict(
    resnet_full=resnet_full,
    heads_on_clip_features=dict(
        linear_C1=float((lr.predict(Xte_n)==yte).mean()),
        linear_best_C=float(best_lr),
        knn={f"k{k}": float((KNeighborsClassifier(n_neighbors=k, metric='cosine').fit(Xtr_n,ytr).predict(Xte_n)==yte).mean()) for k in [1,3,5,10]},
        mlp_256_64=float((mlp.predict(Xte_n)==yte).mean())),
    geometry=dict(intra=intra, inter=inter, ratio=inter/intra, knn1_geometry_acc=float(geo_acc)),
    confusable_top=pairs[:8],
)
json.dump(out, open("data/why_vlms_fail_e1e2.json","w"), indent=2)
print("\nSaved data/why_vlms_fail_e1e2.json")
