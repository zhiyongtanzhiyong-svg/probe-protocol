"""
CoOp FULL (reviewer #9 fix): the earlier CoOp control used only 4 context tokens / 10
epochs, which an under-trained context is not a fair test of "learnable prompts."
This reruns the SAME K-curve x 15-seed protocol with the standard fuller CoOp setup:
16 learnable context tokens and 20 epochs.

Reuses the exact helpers from coop_experiment.py (encode_images, encode_text_ctx,
kshot_subset_idx) and the SAME feature cache / split / test set, so the only change is
N_CTX 4 -> 16 and EPOCHS 10 -> 20. Outputs data/coop_full.json for direct comparison
against data/coop_experiment.json.
"""
import sys, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM
from coop_experiment import (
    CKPT, FEAT_CACHE, CLASS_NAMES,
    encode_images, encode_text_ctx, kshot_subset_idx,
)

N_CTX = int(os.environ.get("COOP_N_CTX", "16"))
LR = 1e-3
EPOCHS = int(os.environ.get("COOP_EPOCHS", "20"))
BS = 32
N_SEEDS = 15
OUT = os.environ.get("COOP_OUT", os.path.join(REPO, "data", "coop_full.json"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained=CKPT)
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    logit_scale = model.logit_scale.exp().item()

    train_ds = TPDDDataset("train", transform=EVAL_TRANSFORM)
    test_ds = TPDDDataset("test", transform=EVAL_TRANSFORM)

    if os.path.exists(FEAT_CACHE):
        z = np.load(FEAT_CACHE)
        train_feats, train_labels = z["train_feats"], z["train_labels"]
        test_feats, test_labels = z["test_feats"], z["test_labels"]
    else:
        train_feats, train_labels = encode_images(model, train_ds)
        test_feats, test_labels = encode_images(model, test_ds)
        np.savez_compressed(FEAT_CACHE, train_feats=train_feats, train_labels=train_labels,
                            test_feats=test_feats, test_labels=test_labels)
    print(f"features: train {train_feats.shape}, test {test_feats.shape}", flush=True)

    prompts = [f"a photo of a {n.replace('_', ' ')}" for n in CLASS_NAMES]
    text_ids = tokenizer(prompts).to(device)
    with torch.no_grad():
        base_ids = tokenizer(["a photo of a"])[0, 1:5]          # 4 real tokens
        ctx4 = model.token_embedding(base_ids.to(device)).clone().float()
        ctx_init = ctx4.repeat((N_CTX // ctx4.shape[0]), 1)       # (16, dim)
        ctx_init = ctx_init[:N_CTX]
    print(f"ctx_init shape: {tuple(ctx_init.shape)} | N_CTX={N_CTX} EPOCHS={EPOCHS}", flush=True)

    test_t = torch.tensor(test_feats, device=device)
    test_l = np.asarray(test_labels)
    results = {}
    for K in [1, 2, 5, 10, -1]:
        coop_accs, probe_accs = [], []
        for seed in range(N_SEEDS):
            sup_idx = np.arange(len(train_labels)) if K == -1 else kshot_subset_idx(train_labels, K, seed + 500)
            sup_f = train_feats[sup_idx]; sup_l = train_labels[sup_idx]

            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(sup_f, sup_l)
            probe_accs.append(float((clf.predict(test_feats) == test_l).mean()))

            torch.manual_seed(seed)
            ctx = nn.Parameter(ctx_init.clone().to(device))
            opt = torch.optim.Adam([ctx], lr=LR)
            sup_t = torch.tensor(sup_f, device=device)
            sup_lab = torch.tensor(sup_l, device=device)
            n = len(sup_t)
            for _ in range(EPOCHS):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, BS):
                    b = perm[i:i + BS]
                    tf = encode_text_ctx(model, text_ids, ctx, N_CTX)
                    loss = F.cross_entropy((sup_t[b] @ tf.T) * logit_scale, sup_lab[b])
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                tf = encode_text_ctx(model, text_ids, ctx, N_CTX)
                logit = (test_t @ tf.T) * logit_scale
                coop_accs.append(float((logit.argmax(1).cpu().numpy() == test_l).mean()))
        tag = f"K={K}" if K > 0 else "full"
        results[tag] = {
            "coop": {"mean": float(np.mean(coop_accs)), "std": float(np.std(coop_accs, ddof=1)),
                     "accs": coop_accs, "n_seeds": N_SEEDS},
            "probe_clean": {"mean": float(np.mean(probe_accs)), "std": float(np.std(probe_accs, ddof=1)),
                            "accs": probe_accs, "n_seeds": N_SEEDS},
        }
        print(f"{tag:5s}: CoOp {np.mean(coop_accs)*100:.1f}% +/- {np.std(coop_accs, ddof=1)*100:.1f}  |  "
              f"probe(clean) {np.mean(probe_accs)*100:.1f}% +/- {np.std(probe_accs, ddof=1)*100:.1f}", flush=True)
        json.dump({"N_CTX": N_CTX, "EPOCHS": EPOCHS, "results": results}, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
