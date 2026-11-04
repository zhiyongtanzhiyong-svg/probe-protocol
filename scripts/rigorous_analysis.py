"""
Rigorous re-analysis from the trusted CLIP feature cache (clip_features_clean.npz).
No re-encoding needed for probe/stats/calibration/per-class. Replicates the original
probe EXACTLY (RandomState(seed+500) stratified K-shot + LogisticRegression(C=1.0)) so
numbers align with clip_fewshot.json, then adds the statistics the reviewers demanded.
Outputs: data/rigorous_analysis.json + printed summary.
"""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

DATA = "data"
d = np.load(f"{DATA}/clip_features_clean.npz")
Xtr, ytr = d["train_feats"], d["train_labels"]
Xte, yte = d["test_feats"], d["test_labels"]
C = len(np.unique(ytr)); n_test = len(yte)
rng_master = np.random.RandomState(0)

# ---------- baselines (test split) ----------
vals, cnts = np.unique(yte, return_counts=True)
uniform = 1.0 / C
majority = cnts.max() / n_test
maj_cls = int(vals[cnts.argmax()])

# ---------- replicate CLIP probe ----------
def probe_acc(Xs, ys):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xs, ys)
    return clf

tr_by_class = {c: np.where(ytr == c)[0] for c in range(C)}
def kshot_idx(K, seed):
    rng = np.random.RandomState(seed + 500)
    sel = []
    for c in range(C):
        idxs = tr_by_class[c]
        sel.extend(rng.choice(idxs, size=min(K, len(idxs)), replace=False))
    return np.array(sorted(sel))

N = 15
probe = {}
full_probs = None
for K in [1, 2, 5, 10, -1]:
    accs, baccs, f1s = [], [], []
    for seed in range(N):
        if K == -1:
            si = np.arange(len(ytr)); Xs, ys = Xtr, ytr
        else:
            si = kshot_idx(K, seed); Xs, ys = Xtr[si], ytr[si]
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(Xs, ys)
        pred = clf.predict(Xte)
        accs.append(accuracy_score(yte, pred))
        baccs.append(balanced_accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, average="macro"))
        if K == -1:
            full_probs = clf.predict_proba(Xte)
    lab = "full" if K == -1 else f"K={K}"
    probe[lab] = dict(n=N, acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs, ddof=1)),
                      accs=accs, bacc_mean=float(np.mean(baccs)), bacc_std=float(np.std(baccs, ddof=1)),
                      macro_f1_mean=float(np.mean(f1s)))

# ---------- CLIP vs ResNet statistics ----------
def load_resnet(K):
    p = f"{DATA}/resnet_k{K}_n15.json"
    try:
        r = json.load(open(p)); return np.array(r["accs"])
    except Exception:
        return None

def welch(a, b):
    na, nb = len(a), len(b); ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va/na + vb/nb
    if se2 == 0: return np.nan, np.nan
    t = (ma-mb)/np.sqrt(se2)
    df = se2**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    from scipy import stats
    p = 2*stats.t.sf(abs(t), df)
    return float(t), float(p)

def tost(a, b, m=0.05):
    diff = a - b; n = len(diff); md = diff.mean()
    sed = diff.std(ddof=1)/np.sqrt(n)
    from scipy import stats
    tcrit = stats.t.ppf(0.975, n-1)
    lo, hi = md - tcrit*sed, md + tcrit*sed
    return dict(margin=m, mean_diff=md, ci_lo=lo, ci_hi=hi, equivalent=bool(hi < m and lo > -m))

def bootstrap_gap_ci(a, b, nboot=4000, seed=0):
    rng = np.random.RandomState(seed); n = len(a)
    gaps = np.empty(nboot)
    for i in range(nboot):
        idx = rng.randint(0, n, n)
        gaps[i] = a[idx].mean() - b[idx].mean()
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))

from scipy import stats as _st
clip_vs_resnet = {}
for K in [1, 5, 10]:
    a = np.array(probe[f"K={K}"]["accs"]); b = load_resnet(K)
    if b is None: continue
    d = a - b; ddof = d.std(ddof=1)/np.sqrt(len(d))
    t, p = _st.ttest_1d(d, 0) if False else (float(np.mean(d)/ddof), float(2*_st.t.sf(abs(np.mean(d)/ddof), len(d)-1)))
    d_pooled = (a.std(ddof=1)+b.std(ddof=1))/2
    clip_vs_resnet[f"K={K}"] = dict(
        clip=float(a.mean()), resnet=float(b.mean()), gap_pp=float((a.mean()-b.mean())*100),
        paired_t=float(t), paired_p=float(p), cohens_dz=float(np.mean(d)/d.std(ddof=1)),
        cohens_d_pooled=float((a.mean()-b.mean())/d_pooled),
        welch=welch(a, b), tost5=tost(a, b, 0.05), boot_ci95=list(bootstrap_gap_ci(a, b)))

# ---------- calibration (full probe) ----------
def ece(probs, pred, y, nbins, strat="ew"):
    if strat == "ew":
        edges = np.linspace(0, 1, nbins+1)
        conf = probs.max(1); idx = np.clip(np.digitize(conf, edges[1:-1]), 0, nbins-1)
    else:
        conf = probs.max(1)
        order = np.argsort(conf)
        step = max(1, n_test//nbins); idx = np.zeros(n_test, int)
        # n_test=138, nbins=15 -> step=9 covers only order[0:135]; the last 3
        # samples kept idx=0 and were double-counted into bin 0 while bins 13/14
        # stayed short. Split exactly instead.
        for k, chunk in enumerate(np.array_split(order, nbins)): idx[chunk] = k
    e = 0.0; rows = []
    for b in range(nbins):
        m = idx == b
        if m.sum() == 0: continue
        c = probs[m].max(1).mean(); acc = (pred[m] == y[m]).mean()
        e += m.sum()/n_test * abs(acc - c)
        rows.append((b, int(m.sum()), float(c), float(acc)))
    return e, rows

def boot_ece_ci(probs, pred, y, nbins=15, nboot=1000, seed=0):
    rng = np.random.RandomState(seed); out = []
    for _ in range(nboot):
        i = rng.randint(0, n_test, n_test)
        e, _ = ece(probs[i], pred[i], y[i], nbins, "ef"); out.append(e)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

full_pred = np.argmax(full_probs, 1)
calib = dict(
    ece15_ef=float(ece(full_probs, full_pred, yte, 15, "ef")[0]),
    ece10_ew=float(ece(full_probs, full_pred, yte, 10, "ew")[0]),
    ece15_ef_ci95=boot_ece_ci(full_probs, full_pred, yte),
    mean_conf_correct=float(full_probs[full_pred==yte].max(1).mean()),
    mean_conf_incorrect=float(full_probs[full_pred!=yte].max(1).mean()),
)

# ---------- per-class WITH threshold (full probe) ----------
pc = {}
for c in range(C):
    m = yte == c; n_c = int(m.sum())
    acc_c = float((full_pred[m] == c).mean()) if n_c else np.nan
    pc[c] = dict(n_test=n_c, acc=acc_c)
n_ge5 = [c for c in range(C) if pc[c]["n_test"] >= 5]
perclass = dict(
    per_class=pc,
    ge50_all=int(sum(1 for c in range(C) if pc[c]["acc"] >= 0.5)),
    n_classes=int(C),
    ge50_among_nge5=int(sum(1 for c in n_ge5 if pc[c]["acc"] >= 0.5)),
    n_classes_nge5=len(n_ge5),
    n_test_per_class=[pc[c]["n_test"] for c in range(C)],
    one_image_classes=[c for c in range(C) if pc[c]["n_test"] == 1],
)

# ---------- reconcile the two zero-shot runs (no model) ----------
za = {x["template"]: x["accuracy"] for x in json.load(open(f"{DATA}/prompt_ablation.json"))}
zb = json.load(open(f"{DATA}/zpe_experiment.json"))["per_template_test_acc"]
zb_pct = {k: v*100 for k, v in zb.items()}
zero_reconcile = {
    "prompt_ablation.json (paper)": {k: round(v, 2) for k, v in sorted(za.items(), key=lambda x: -x[1])},
    "zpe_experiment.json (more complete)": {k: round(v, 2) for k, v in sorted(zb_pct.items(), key=lambda x: -x[1])},
    "best_in_paper_run": max(za, key=za.get),
    "best_in_zpe_run": max(zb_pct, key=zb_pct.get),
    "en_label_bug": {"prompt_ablation": za.get("EN-Label"), "zpe": zb_pct.get("EN-Label"),
                     "note": "0.7246 in one file = 0.007246 in the other (x100 decimal bug)"},
    "disagreement_pp": {k: round(za.get(k, np.nan) - zb_pct.get(k, np.nan), 2) for k in za},
}

out = dict(
    dataset=dict(n_train=int(len(ytr)), n_test=int(n_test)),
    class_distribution_test=dict(counts=[int(x) for x in cnts], order=sorted(vals.tolist()),
                                 min=int(cnts.min()), max=int(cnts.max())),
    baselines=dict(uniform=round(uniform*100, 3), majority=round(majority*100, 3), majority_class_idx=maj_cls),
    probe=probe, clip_vs_resnet=clip_vs_resnet, calibration=calib, perclass=perclass,
    zero_shot_reconciliation=zero_reconcile,
)
json.dump(out, open(f"{DATA}/rigorous_analysis.json", "w"), indent=2)

# ---------- print ----------
print("=== BASELINES ===")
print(f"  uniform {uniform*100:.2f}%   majority {majority*100:.2f}% (class {maj_cls})   [best zero-shot 15.2-15.9%]")
print("=== CLASS DIST (test, n=%d) ===" % n_test)
print("  counts:", [int(x) for x in cnts], " min/max:", int(cnts.min()), int(cnts.max()), " 1-image classes:", perclass["one_image_classes"])
print("=== CLIP PROBE (reproduced from npz) ===")
for k in ["K=1","K=2","K=5","K=10","full"]:
    p = probe[k]
    print(f"  {k:5s}: acc {p['acc_mean']*100:5.1f}±{p['acc_std']*100:4.1f}  balAcc {p['bacc_mean']*100:5.1f}  macroF1 {p['macro_f1_mean']*100:5.1f}")
print("=== CLIP vs RESNET (n=15) ===")
for k in ["K=1","K=5","K=10"]:
    if k not in clip_vs_resnet: continue
    r = clip_vs_resnet[k]
    print(f"  {k}: clip {r['clip']*100:.1f} resnet {r['resnet']*100:.1f} gap {r['gap_pp']:+.1f}pp  "
          f"paired_p={r['paired_p']:.3f} dz={r['cohens_dz']:.2f} welch_p={r['welch'][1]:.3f} "
          f"TOST±5pp_equiv={r['tost5']['equivalent']} boot95=[{r['boot_ci95'][0]*100:+.1f},{r['boot_ci95'][1]*100:+.1f}]")
print("=== CALIBRATION (full probe) ===")
print(f"  ECE(15,ef)={calib['ece15_ef']:.3f} CI95={calib['ece15_ef_ci95']}  ECE(10,ew)={calib['ece10_ew']:.3f}")
print(f"  mean-conf correct={calib['mean_conf_correct']:.3f} incorrect={calib['mean_conf_incorrect']:.3f}")
print("=== PER-CLASS (full probe) ===")
print(f"  >=50% on {perclass['ge50_all']}/{perclass['n_classes']} classes; restricted to n>=5: {perclass['ge50_among_nge5']}/{perclass['n_classes_nge5']}")
print("=== ZERO-SHOT RECONCILIATION ===")
print("  best (paper run):", zero_reconcile["best_in_paper_run"], "  best (zpe run):", zero_reconcile["best_in_zpe_run"])
print("  EN-Label bug:", zero_reconcile["en_label_bug"])
print("\nSaved data/rigorous_analysis.json")
