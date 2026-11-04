"""
CoOp (learnable context, Zhou et al. 2022, IJCV) control experiment for this paper.
Answers: is CLIP weak because of features, or because of the (linear probe) usage?

Protocol:
- Same TPDD splits and same K-shot support subsets/seeds as the paper's probe
  (KShotSubset with seed+500, 15 seeds).
- Clean (eval) feature extraction for support and test.
- CoOp: 4 learnable context tokens (init = "a photo of a" token embeddings),
  Adam lr=1e-3, 10 epochs, batch 32, cross-entropy on support set vs. 16 class
  prompts ("a photo of a {class}"); then zero-shot eval on test.
- Control: standard logistic-regression linear probe on the same clean support
  features (C=1.0, max_iter=2000, random_state=seed) — isolates the classifier
  difference from the feature difference.
"""
import os
import sys, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

sys.path.insert(0, ROOT + '/scripts')
import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
OUT = ROOT + '/data/coop_experiment.json'
FEAT_CACHE = ROOT + '/data/clip_features_clean.npz'

CLASS_NAMES = ['anthracnose', 'black_shank', 'brown_spot', 'cmv', 'frog_eye',
               'genetic_abnormality', 'healthy', 'nematodes', 'potato_tuber_moth',
               'pvy', 'sunscald', 'target_spot', 'tmv', 'tswv', 'weather_fleck',
               'wildfire']
N_CTX = 4
LR = 1e-3
EPOCHS = 10
BS = 32
N_SEEDS = 15


def encode_images(model, ds):
    loader = DataLoader(ds, 64, shuffle=False, num_workers=2)
    feats, labs = [], []
    with torch.no_grad():
        for batch in loader:
            f = model.encode_image(batch['image'].cuda(), normalize=True)
            feats.append(f.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
    return np.concatenate(feats), np.array(labs)


def encode_text_ctx(model, tok_ids, ctx, n_ctx):
    from open_clip.transformer import text_global_pool
    cast_dtype = model.transformer.get_cast_dtype()
    x = model.token_embedding(tok_ids).to(cast_dtype)
    x = x + model.positional_embedding.to(cast_dtype)
    x = x.clone()
    x[:, 1:1 + n_ctx] = ctx.to(cast_dtype)
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = model.ln_final(x)
    x = text_global_pool(x, tok_ids, model.text_pool_type)
    if model.text_projection is not None:
        if isinstance(model.text_projection, nn.Linear):
            x = model.text_projection(x)
        else:
            x = x @ model.text_projection
    return F.normalize(x, dim=-1)


def kshot_subset_idx(labels, K, subset_seed):
    rng = np.random.RandomState(subset_seed)
    by_class = {}
    for i, l in enumerate(labels):
        by_class.setdefault(int(l), []).append(i)
    out = []
    for c in sorted(by_class):
        out.extend(rng.choice(np.array(by_class[c]), size=min(K, len(by_class[c])), replace=False))
    return np.sort(np.array(out))


def main():
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CKPT)
    model = model.eval().cuda()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    logit_scale = model.logit_scale.exp().item()

    train_ds = TPDDDataset('train', transform=EVAL_TRANSFORM)
    test_ds = TPDDDataset('test', transform=EVAL_TRANSFORM)

    if os.path.exists(FEAT_CACHE):
        z = np.load(FEAT_CACHE)
        train_feats, train_labels = z['train_feats'], z['train_labels']
        test_feats, test_labels = z['test_feats'], z['test_labels']
    else:
        train_feats, train_labels = encode_images(model, train_ds)
        test_feats, test_labels = encode_images(model, test_ds)
        np.savez_compressed(FEAT_CACHE, train_feats=train_feats, train_labels=train_labels,
                            test_feats=test_feats, test_labels=test_labels)
    print(f"features: train {train_feats.shape}, test {test_feats.shape}")
    assert len(set(train_labels)) == 16 and len(set(test_labels)) == 16

    prompts = [f"a photo of a {n.replace('_', ' ')}" for n in CLASS_NAMES]
    text_ids = tokenizer(prompts).cuda()
    with torch.no_grad():
        base_text = F.normalize(model.encode_text(text_ids), dim=-1)
        init_ids = tokenizer(["a photo of a"])[0, 1:1 + N_CTX]
        ctx_init = model.token_embedding(init_ids.cuda()).clone().float()

    results = {}
    for K in [1, 2, 5, 10, -1]:
        coop_accs, probe_accs = [], []
        for seed in range(N_SEEDS):
            if K == -1:
                sup_idx = np.arange(len(train_labels))
            else:
                sup_idx = kshot_subset_idx(train_labels, K, seed + 500)
            sup_f = train_feats[sup_idx]
            sup_l = train_labels[sup_idx]

            # --- linear probe control (same clean features) ---
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(sup_f, sup_l)
            probe_accs.append(float((clf.predict(test_feats) == test_labels).mean()))

            # --- CoOp ---
            torch.manual_seed(seed)
            ctx = nn.Parameter(ctx_init.clone().cuda())
            opt = torch.optim.Adam([ctx], lr=LR)
            sup_t = torch.tensor(sup_f, device='cuda')
            sup_lab = torch.tensor(sup_l, device='cuda')
            n = len(sup_t)
            for _ in range(EPOCHS):
                perm = torch.randperm(n, device='cuda')
                for i in range(0, n, BS):
                    b = perm[i:i + BS]
                    tf = encode_text_ctx(model, text_ids, ctx, N_CTX)
                    logit = (sup_t[b] @ tf.T) * logit_scale
                    loss = F.cross_entropy(logit, sup_lab[b])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                tf = encode_text_ctx(model, text_ids, ctx, N_CTX)
                logit = (torch.tensor(test_feats, device='cuda') @ tf.T) * logit_scale
                coop_accs.append(float((logit.argmax(1).cpu().numpy() == test_labels).mean()))
        tag = f"K={K}" if K > 0 else "full"
        results[tag] = {
            "coop": {"mean": float(np.mean(coop_accs)), "std": float(np.std(coop_accs, ddof=1)),
                     "accs": coop_accs, "n_seeds": N_SEEDS},
            "probe_clean": {"mean": float(np.mean(probe_accs)), "std": float(np.std(probe_accs, ddof=1)),
                            "accs": probe_accs, "n_seeds": N_SEEDS},
        }
        print(f"{tag:5s}: CoOp {np.mean(coop_accs)*100:.1f}% +/- {np.std(coop_accs, ddof=1)*100:.1f}  |  "
              f"probe(clean) {np.mean(probe_accs)*100:.1f}% +/- {np.std(probe_accs, ddof=1)*100:.1f}", flush=True)
        json.dump(results, open(OUT, 'w'), indent=2)
        print("Saved to", OUT, flush=True)


if __name__ == '__main__':
    main()
