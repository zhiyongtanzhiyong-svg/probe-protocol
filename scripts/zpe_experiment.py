"""
ZPE (zero-shot prompt ensembling) for this paper.

Unifies the 14 prompt templates from prompt_ablation.py into an ensemble
whose per-template weights are "discriminative signal" -- zero-shot
accuracy of each template measured on a labeled split.

Protocol:
- Baseline (uniform): average the 14 templates' [n, 16] logits with equal
  weights, evaluate on the held-out test split.
- Option 1 (accuracy-weighted): weight = softmax(acc / T). T is swept on
  the signal source (validation) and chosen by valley. Templates at or
  below chance (1/16) get weight 0 (floor) so they cannot add noise.
- Option 2 (per-class discriminative): per-class softmax(acc_c / T)
  weights, so each template only votes on the classes it can actually
  separate.
- Signal sources for acc (all implemented, user: "都尝试一下"):
  (a) support-K: mean zero-shot acc of each template over the 15 K-shot
      support subsets (same KShotSubset seed+500 as the paper's probe).
  (b) static: accuracies stored in data/prompt_ablation.json.
  (c) val-holdout: balanced 25% held out of the train split.
- Final evaluation always uses the held-out test split features
  (clip_features_clean.npz, same clean cache as coop_experiment.py).
"""
import os
import sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ROOT + '/scripts')
import open_clip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
OUT = ROOT + '/data/zpe_experiment.json'
FEAT_CACHE = ROOT + '/data/clip_features_clean.npz'
ABLATION_JSON = ROOT + '/data/prompt_ablation.json'
KG_MAP = ROOT + '/data/tpd_to_kg_map.json'
KG_DATA = ROOT + '/data/tobacco_records.json'

CLASS_NAMES = ['anthracnose', 'black_shank', 'brown_spot', 'cmv', 'frog_eye',
               'genetic_abnormality', 'healthy', 'nematodes', 'potato_tuber_moth',
               'pvy', 'sunscald', 'target_spot', 'tmv', 'tswv', 'weather_fleck',
               'wildfire']
N_CLASS = 16
CHANCE = 1.0 / N_CLASS
N_SEEDS = 15
K_LIST = [1, 2, 5, 10, -1]
T_LIST = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
LOGIT_SCALE = 100.0
VAL_FRAC = 0.25
VAL_SEED = 2024

CN_MAP = {
    "anthracnose": "炭疽病", "black_shank": "黑胫病", "brown_spot": "赤星病",
    "cmv": "黄瓜花叶病毒病", "frog_eye": "蛙眼病", "genetic_abnormality": "遗传异常",
    "healthy": "健康", "nematodes": "根结线虫病", "potato_tuber_moth": "马铃薯块茎蛾",
    "pvy": "马铃薯Y病毒病", "sunscald": "日灼病", "target_spot": "靶斑病",
    "tmv": "烟草普通花叶病毒病", "tswv": "番茄斑萎病毒病", "weather_fleck": "气候斑点病",
    "wildfire": "野火病",
}

TEMPLATES = [
    ("EN-Label", "{class_name}"),
    ("CN-Label", "{cn_name}"),
    ("EN-Photo", "a photo of a tobacco leaf with {class_name} disease"),
    ("CN-Photo", "一张患有{cn_name}的烟草叶片照片"),
    ("EN-Symptom", "tobacco leaf showing {class_name} symptoms"),
    ("CN-Symptom", "烟草叶片表现出{cn_name}的症状"),
    ("EN-Closeup", "a close-up photo of {class_name} on tobacco plant"),
    ("CN-Closeup", "烟草植株上{cn_name}的特写照片"),
    ("EN-Disease", "{class_name} disease on tobacco leaf"),
    ("CN-Disease", "烟草叶片上的{cn_name}病害"),
    ("KG-Symptom", "tobacco leaf: {kg_symptom}"),
    ("KG-Condition", "tobacco leaf disease occurring when: {kg_condition}"),
    ("KG-Combined", "tobacco leaf with {kg_symptom}, occurs when {kg_condition}"),
    ("EN-Expert", "tobacco plant pathology: {class_name} disease identification"),
]

TEMPLATE_NAMES = [t[0] for t in TEMPLATES]
N_TPL = len(TEMPLATES)


def kshot_subset_idx(labels, K, subset_seed):
    rng = np.random.RandomState(subset_seed)
    by_class = {}
    for i, l in enumerate(labels):
        by_class.setdefault(int(l), []).append(i)
    out = []
    for c in sorted(by_class):
        n = len(by_class[c]) if K < 0 else min(K, len(by_class[c]))
        out.extend(rng.choice(np.array(by_class[c]), size=n, replace=False))
    return np.sort(np.array(out))


def accuracy(labels, preds):
    return float((preds == labels).mean())


def softmax_acc_weights(acc, T):
    acc = np.asarray(acc, dtype=float)
    w = np.exp(np.clip(acc, -50.0, 50.0) / T)
    w[acc <= CHANCE] = 0.0
    s = w.sum()
    if s <= 0:
        return np.full_like(w, 1.0 / N_TPL)
    return w / s


def per_class_weights(acc2d, T):
    acc2d = np.asarray(acc2d, dtype=float)
    W = np.exp(np.clip(acc2d, -50.0, 50.0) / T)
    W[acc2d <= CHANCE] = 0.0
    colsum = W.sum(axis=0, keepdims=True)
    safe = np.where(colsum > 0, colsum, 1.0)
    W = np.where(colsum > 0, W / safe, np.full_like(W, 1.0 / N_TPL))
    return W


def ensemble_acc(logits, w, labels):
    S = np.einsum('t,tni->ni', w, logits)
    return float((S.argmax(1) == labels).mean())


def ensemble_acc_pc(logits, W, labels):
    S = np.einsum('tc,tni->ni', W, logits)
    return float((S.argmax(1) == labels).mean())


def build_signals(model, tokenizer):
    tpd_map = json.load(open(KG_MAP))
    kg_data = json.load(open(KG_DATA))
    kg_symptoms = {}
    kg_conditions = {}
    for name, info in tpd_map.items():
        if name == 'healthy':
            kg_symptoms[name] = "healthy tobacco leaf without any disease spots or damage"
            kg_conditions[name] = "optimal growing conditions with no disease pressure"
            continue
        kg_ent = info.get('kg_entity')
        if kg_ent:
            kg_name = kg_ent.replace('D_', '')
            matched = False
            for r in kg_data:
                if kg_name.lower() in r.get('name', '').lower() or name.lower() in r.get('english_name', '').lower():
                    kg_symptoms[name] = r.get('symptoms', '').split('。')[0][:100]
                    kg_conditions[name] = r.get('occurrence', '').strip()[:100]
                    matched = True
                    break
            if not matched:
                cn = CN_MAP.get(name, name)
                kg_symptoms[name] = f"{cn} disease symptoms on tobacco leaf"
                kg_conditions[name] = f"conditions favoring {cn} development"
        else:
            cn = CN_MAP.get(name, name)
            kg_symptoms[name] = f"{cn} disease symptoms on tobacco leaf"
            kg_conditions[name] = f"conditions favoring {cn} development"

    tf_all = []
    with torch.no_grad():
        for _, template in TEMPLATES:
            prompts = []
            for name in CLASS_NAMES:
                s = template.replace('{class_name}', name).replace('{cn_name}', CN_MAP.get(name, name))
                s = s.replace('{kg_symptom}', kg_symptoms.get(name, ''))
                s = s.replace('{kg_condition}', kg_conditions.get(name, ''))
                prompts.append(s)
            tokens = tokenizer(prompts).cuda()
            tf = model.encode_text(tokens)
            tf_all.append(F.normalize(tf, dim=-1).cpu().numpy())
    return np.stack(tf_all)


def support_signal(train_feats, train_labels, tf_all, K, VACC_HOOK):
    per_templ = np.zeros(N_TPL)
    per_class = np.zeros((N_TPL, N_CLASS))
    total = 0
    tot_pc = np.zeros(N_CLASS)
    seed_sup = []
    for seed in range(N_SEEDS):
        sup_idx = kshot_subset_idx(train_labels, K, seed + 500)
        sup_f = train_feats[sup_idx]
        sup_l = train_labels[sup_idx]
        L = sup_f @ tf_all.transpose(0, 2, 1) * LOGIT_SCALE
        seed_sup.append((sup_f, sup_l, L))
        total += len(sup_l)
        for t in range(N_TPL):
            p = L[t].argmax(-1)
            per_templ[t] += (p == sup_l).sum()
            for c in range(N_CLASS):
                m = sup_l == c
                if m.any():
                    per_class[t, c] += (p[m] == c).sum()
                    tot_pc[c] += m.sum()
    per_templ /= max(total, 1)
    per_class = per_class / np.maximum(tot_pc[None, :], 1)

    val_by_T = {}
    val_pc_by_T = {}
    for T in T_LIST:
        w = softmax_acc_weights(per_templ, T)
        W = per_class_weights(per_class, T)
        accs = [ensemble_acc(L, w, sup_l) for _, sup_l, L in seed_sup]
        accs_pc = [ensemble_acc_pc(L, W, sup_l) for _, sup_l, L in seed_sup]
        val_by_T[T] = float(np.mean(accs))
        val_pc_by_T[T] = float(np.mean(accs_pc))
    best_T = max(T_LIST, key=lambda T: val_by_T[T])
    best_T_pc = max(T_LIST, key=lambda T: val_pc_by_T[T])
    return {
        "template_acc": {n: float(per_templ[i]) for i, n in enumerate(TEMPLATE_NAMES)},
        "per_class_acc": {n: {CLASS_NAMES[c]: float(per_class[i, c]) for c in range(N_CLASS)}
                          for i, n in enumerate(TEMPLATE_NAMES)},
        "val_by_T": {f"{T:.2f}": float(v) for T, v in val_by_T.items()},
        "val_pc_by_T": {f"{T:.2f}": float(v) for T, v in val_pc_by_T.items()},
        "best_T": best_T,
        "best_T_pc": best_T_pc,
        "signal_source": f"support-K{K}",
    }


def holdout_signal(train_feats, train_labels, tf_all, side):
    rng = np.random.RandomState(VAL_SEED)
    by_class = {}
    for i, l in enumerate(train_labels):
        by_class.setdefault(int(l), []).append(i)
    val_idx = []
    train_idx = []
    for c in sorted(by_class):
        idx = np.array(by_class[c])
        rng.shuffle(idx)
        n_val = int(round(len(idx) * VAL_FRAC))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    val_idx = np.sort(np.array(val_idx))
    train_idx = np.sort(np.array(train_idx))

    if side == 'train':
        s_f, s_l = train_feats[train_idx], train_labels[train_idx]
    else:
        s_f, s_l = train_feats[val_idx], train_labels[val_idx]

    L = s_f @ tf_all.transpose(0, 2, 1) * LOGIT_SCALE
    per_templ = np.array([(L[t].argmax(-1) == s_l).mean() for t in range(N_TPL)])
    per_class = np.zeros((N_TPL, N_CLASS))
    for t in range(N_TPL):
        p = L[t].argmax(-1)
        for c in range(N_CLASS):
            m = s_l == c
            if m.any():
                per_class[t, c] = (p[m] == c).mean()

    if side == 'val':
        val_logits, val_labels = L, s_l
    else:
        val_logits = train_feats[val_idx] @ tf_all.transpose(0, 2, 1) * LOGIT_SCALE
        val_labels = train_labels[val_idx]

    val_by_T = {}
    val_pc_by_T = {}
    for T in T_LIST:
        w = softmax_acc_weights(per_templ, T)
        W = per_class_weights(per_class, T)
        val_by_T[T] = ensemble_acc(val_logits, w, val_labels)
        val_pc_by_T[T] = ensemble_acc_pc(val_logits, W, val_labels)
    best_T = max(T_LIST, key=lambda T: val_by_T[T])
    best_T_pc = max(T_LIST, key=lambda T: val_pc_by_T[T])
    return {
        "template_acc": {n: float(per_templ[i]) for i, n in enumerate(TEMPLATE_NAMES)},
        "per_class_acc": {n: {CLASS_NAMES[c]: float(per_class[i, c]) for c in range(N_CLASS)}
                          for i, n in enumerate(TEMPLATE_NAMES)},
        "val_by_T": {f"{T:.2f}": float(v) for T, v in val_by_T.items()},
        "val_pc_by_T": {f"{T:.2f}": float(v) for T, v in val_pc_by_T.items()},
        "n_val": int(len(val_idx)),
        "best_T": best_T,
        "best_T_pc": best_T_pc,
    }


def static_signal():
    recs = json.load(open(ABLATION_JSON))
    by_name = {r['template']: r for r in recs}
    per_templ = np.array([float(by_name[n]['accuracy']) / 100.0 for n in TEMPLATE_NAMES])
    per_class = np.array([[float(by_name[n]['per_class'][c]) / 100.0 for c in CLASS_NAMES]
                          for n in TEMPLATE_NAMES])
    return {
        "template_acc": {n: float(per_templ[i]) for i, n in enumerate(TEMPLATE_NAMES)},
        "per_class_acc": {n: {CLASS_NAMES[c]: float(per_class[i, c]) for c in range(N_CLASS)}
                          for i, n in enumerate(TEMPLATE_NAMES)},
        "best_T": None,
        "best_T_pc": None,
        "per_template_note": "accuracy from prompt_ablation.json (test-derived, static weights)",
    }


def evaluate_signal(sig, test_logits, test_labels, tag):
    per_templ = np.array([sig['template_acc'][n] for n in TEMPLATE_NAMES])
    per_class = np.array([[sig['per_class_acc'][n][c] for c in CLASS_NAMES] for n in TEMPLATE_NAMES])
    out = {}
    for T in T_LIST:
        w = softmax_acc_weights(per_templ, T)
        W = per_class_weights(per_class, T)
        out[f"T={T:.2f}"] = {
            "ensemble": ensemble_acc(test_logits, w, test_labels),
            "per_class": ensemble_acc_pc(test_logits, W, test_labels),
        }
    best = sig.get('best_T')
    best_pc = sig.get('best_T_pc')
    chosen = {}
    if best is not None:
        w = softmax_acc_weights(per_templ, best)
        chosen["ensemble_validation_chosen"] = ensemble_acc(test_logits, w, test_labels)
    if best_pc is not None:
        W = per_class_weights(per_class, best_pc)
        chosen["per_class_validation_chosen"] = ensemble_acc_pc(test_logits, W, test_labels)
    if best is None:
        weights_uniform = np.full(N_TPL, 1.0 / N_TPL)
        chosen["ensemble_uniform"] = ensemble_acc(test_logits, weights_uniform, test_labels)
        W_uniform_col = np.full((N_TPL, N_CLASS), 1.0 / N_TPL)
        chosen["per_class_uniform"] = ensemble_acc_pc(test_logits, W_uniform_col, test_labels)
    out["chosen"] = chosen
    out["note"] = sig.get('per_template_note', f"validated on {sig.get('signal_source', tag)}")
    return out


def main():
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CKPT)
    model = model.eval().cuda()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    z = np.load(FEAT_CACHE)
    train_feats, train_labels = z['train_feats'], z['train_labels']
    test_feats, test_labels = z['test_feats'], z['test_labels']
    print(f"features: train {train_feats.shape}, test {test_feats.shape}", flush=True)
    assert len(set(train_labels)) == N_CLASS and len(set(test_labels)) == N_CLASS

    tf_all = build_signals(model, tokenizer)
    test_logits = test_feats @ tf_all.transpose(0, 2, 1) * LOGIT_SCALE

    per_templ_test = np.array([(test_logits[t].argmax(1) == test_labels).mean() for t in range(N_TPL)])
    weights_uniform = np.full(N_TPL, 1.0 / N_TPL)
    uniform_acc = ensemble_acc(test_logits, weights_uniform, test_labels)
    W_uniform_col = np.full((N_TPL, N_CLASS), 1.0 / N_TPL)
    uniform_pc_acc = ensemble_acc_pc(test_logits, W_uniform_col, test_labels)

    results = {
        "config": {
            "n_templates": N_TPL, "n_class": N_CLASS, "chance": CHANCE,
            "n_seeds": N_SEEDS, "k_list": K_LIST, "t_list": T_LIST,
            "val_frac": VAL_FRAC, "val_seed": VAL_SEED, "logit_scale": LOGIT_SCALE,
            "feature_cache": FEAT_CACHE,
        },
        "per_template_test_acc": {n: float(per_templ_test[i]) for i, n in enumerate(TEMPLATE_NAMES)},
        "uniform_ensemble_test_acc": uniform_acc,
        "uniform_per_class_ensemble_test_acc": uniform_pc_acc,
        "signals": {},
    }

    print(f"\nuniform baseline test acc: {uniform_acc*100:.1f}%", flush=True)

    for K in K_LIST:
        tag = f"support-K{K}"
        sig = support_signal(train_feats, train_labels, tf_all, K, None)
        ev = evaluate_signal(sig, test_logits, test_labels, tag)
        results['signals'][tag] = {"signal": sig, "eval": ev}
        print(f"{tag:12s} best_T={sig['best_T']} -> test {ev['chosen'].get('ensemble_validation_chosen', 0)*100:.1f}%", flush=True)
        json.dump(results, open(OUT, 'w'), indent=2)

    sig = holdout_signal(train_feats, train_labels, tf_all, 'val')
    sig["template_acc_on_holdout"] = sig["template_acc"]
    ev = evaluate_signal(sig, test_logits, test_labels, "val-holdout")
    results['signals']['val-holdout'] = {"signal": sig, "eval": ev}
    print(f"val-holdout best_T={sig['best_T']} -> test {ev['chosen'].get('ensemble_validation_chosen', 0)*100:.1f}%  "
          f"| per-class best_T={sig['best_T_pc']} -> test {ev['chosen'].get('per_class_validation_chosen', 0)*100:.1f}%", flush=True)
    json.dump(results, open(OUT, 'w'), indent=2)

    sig = static_signal()
    ev = evaluate_signal(sig, test_logits, test_labels, "static")
    results['signals']['static'] = {"signal": sig, "eval": ev}
    print("static (prompt_ablation.json):", flush=True)
    for T in T_LIST:
        r = ev[f"T={T:.2f}"]
        print(f"    T={T:.2f} ensemble {r['ensemble']*100:.1f}%  per-class {r['per_class']*100:.1f}%", flush=True)
    print(f"    uniform fallback ensemble {ev['chosen']['ensemble_uniform']*100:.1f}%  "
          f"per-class {ev['chosen']['per_class_uniform']*100:.1f}%", flush=True)
    json.dump(results, open(OUT, 'w'), indent=2)

    print("\nSaved to", OUT)


if __name__ == '__main__':
    main()