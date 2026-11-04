"""
CoOp (Zhou et al. 2022, IJCV) OFFICIAL-RECIPE rerun for this paper.

Why this exists
---------------
1. The earlier 16-token "full recipe" run (coop_full.py) had an IMPLEMENTATION
   BUG: encode_text_ctx() overwrites token positions 1..1+N_CTX, which for
   N_CTX=16 clobbers the class-name tokens, making all 16 prompts identical.
   The resulting per-seed accuracies (10.9-13.8%) exactly match single-class
   test-set fractions (15-19 of 138 images), i.e. every seed was a constant
   prediction. That "collapse" was an artifact, not CoOp behavior.
2. The 4-token run (coop_experiment.py) was structurally valid (ctx replaced
   exactly the 4 prefix tokens, class names intact) but deviated from the
   official recipe: token-embedding init (official: random N(0,0.02)), Adam
   (official: SGD lr 0.002), 10 epochs (official: 200), no LR schedule /
   warmup / early stopping.

This script reimplements the official recipe from the available reference
implementation (reference_lpp/methods/coop.py; the original
github.com/renshenzhou/CoOp repository is no longer online):
  - 16 SHARED learnable ctx tokens, random init N(0, 0.02), class tokens
    AFTER the ctx (prompt = "X"*16 + " " + class + "."), EOS-position pooling
  - SGD lr=0.002 (batch 32), 1-epoch constant warmup (lr 1e-5) then cosine
    annealing over 200 epochs; the scheduler is stepped once per EPOCH, as in
    the original design (the reference port steps it per batch - a porting
    bug that would complete the cosine in ~5 epochs)
  - CE loss, no label smoothing (as in the reference implementation)
  - fresh augmented image views each epoch (official CoOp trains with data
    augmentation): 200 augmented views per train image are pre-extracted once
    (fp16, cached), epoch e uses view e - statistically identical to the
    official per-epoch re-augmentation, and fully reproducible
  - early stopping on validation accuracy; report test accuracy at the
    best-validation epoch
  - determinism: torch.manual_seed(seed) per run (the reference does not seed)
  - fp32 throughout (reference: fp16 AMP); no algorithmic difference

Protocol (mutually comparable with lp_plus_plus.py and the paper's probe):
  - full: train = 1295, val = 20% stratified (seed 42)
  - K-shot (K in 1,2,5,10; 15 seeds): support = stratified K-per-class
    subsets (seed+500, SAME as the paper's K-shot probe); val = disjoint
    K-per-class subset (seed+1500)
Outputs data/coop_official.json.
"""
import os
import sys, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, ROOT + "/scripts")
import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM, TRAIN_TRANSFORM, worker_init_fn
from lp_plus_plus import (
    CKPT, FEAT_CACHE, TEMPLATES, CN_MAP, tensor_to_pil,
    extract_clip, text_embeddings, disjoint_kshot_idx,
    load_feature_cache,
)
from coop_experiment import kshot_subset_idx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_CTX = 16
EPOCHS = 200
BS = 32
LR = 0.002
WARMUP_EPOCH = 1
WARMUP_CONS_LR = 1e-5
N_SEEDS = 15
N_VIEWS = int(os.environ.get("COOP_N_VIEWS", "200"))
OUT = os.environ.get("COOP_OFFICIAL_OUT", os.path.join(REPO, "data", "coop_official.json"))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AUG_CACHE = {
    'dfn_clip': os.path.join(REPO, 'data', 'clip_aug_views_dfn.npz'),
    'laion2b': os.path.join(REPO, 'data', 'clip_aug_views_laion.npz'),
}


class PromptLearner(nn.Module):
    """Faithful port of the reference PromptLearner (shared ctx, random init,
    class tokens after ctx, EOS pooling)."""

    def __init__(self, model, tokenizer, classnames, seed):
        super().__init__()
        torch.manual_seed(seed)          # reproducibility (reference does not seed)
        dim = model.ln_final.weight.shape[0]
        self.ctx = nn.Parameter(
            torch.empty(1, N_CTX, dim, device=DEVICE).to(model.ln_final.weight.dtype).normal_(mean=0, std=0.02))
        names = [n.replace('_', ' ') for n in classnames]
        prefix = " ".join(["X"] * N_CTX)
        prompts = [f"{prefix} {name}." for name in names]
        tok_ids = tokenizer(prompts).to(DEVICE)
        with torch.no_grad():
            emb = model.token_embedding(tok_ids)
        self.register_buffer('token_prefix', emb[:, :1].clone())
        self.register_buffer('token_suffix', emb[:, 1 + N_CTX:].clone())
        self.register_buffer('tokenized', tok_ids)

    def prompt_tokens(self):
        n = self.token_prefix.shape[0]
        return torch.cat([self.token_prefix,
                          self.ctx.expand(n, -1, -1),
                          self.token_suffix], dim=1)


def text_encode(model, prompts, tokenized):
    x = prompts + model.positional_embedding
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = model.ln_final(x)
    idx = tokenized.argmax(dim=-1)        # EOS position (49407 is the max id)
    x = x[torch.arange(x.shape[0]), idx]
    if model.text_projection is not None:
        if isinstance(model.text_projection, nn.Linear):
            x = model.text_projection(x)
        else:
            x = x @ model.text_projection
    return F.normalize(x, dim=-1)


class ConstantWarmupScheduler(lrs._LRScheduler):
    """As in the reference (1 epoch at cons_lr, then the successor)."""

    def __init__(self, optimizer, successor, warmup_epoch, cons_lr, last_epoch=-1):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        self.cons_lr = cons_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


def extract_aug_views(model, preprocess, train_ds, n_views):
    """Pre-extract n_views stochastic (TRAIN_TRANSFORM) augmented views per
    train image; fp16 to mirror the official fp16 training precision."""
    loader = DataLoader(train_ds, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    out = []
    for v in range(n_views):
        feats = []
        with torch.no_grad():
            for batch in loader:
                imgs = torch.stack([preprocess(tensor_to_pil(t)) for t in batch['image']]).to(DEVICE)
                f = model.encode_image(imgs)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu())
        out.append(torch.cat(feats, dim=0).half())
        print(f"  aug view {v + 1}/{n_views} done", flush=True)
    return torch.stack(out, dim=0).numpy().astype(np.float16)   # (V, n, d)


def main():
    t0 = time.time()
    print(f"Device: {DEVICE}", flush=True)
    train_ds = TPDDDataset(split='train', transform=TRAIN_TRANSFORM)
    train_ds_eval = TPDDDataset(split='train', transform=EVAL_TRANSFORM)
    test_ds = TPDDDataset(split='test', transform=EVAL_TRANSFORM)
    class_names = [str(n) for n in train_ds.class_names]
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | N_CTX={N_CTX} EPOCHS={EPOCHS} views={N_VIEWS}", flush=True)

    y_full = np.asarray(train_ds.label_ids)
    idx_sub, idx_val = train_test_split(np.arange(len(y_full)), test_size=0.2,
                                        random_state=42, stratify=y_full)

    R = {
        'protocol': "Official CoOp recipe (reference port; original repo offline): 16 shared ctx tokens random N(0,0.02), class tokens after ctx, EOS pooling; SGD lr 0.002 bs 32, 1-epoch warmup 1e-5 + cosine over 200 epochs (per-epoch step); CE; per-epoch augmented views (200 pre-extracted, fp16); early stop on val; full: val=20% seed42; K-shot: support seed+500 (paired with probe), val disjoint seed+1500; 15 seeds.",
        'n_train': int(len(train_ds)), 'n_test': int(len(test_ds)),
        'N_CTX': N_CTX, 'EPOCHS': EPOCHS, 'N_VIEWS': N_VIEWS, 'checkpoints': {},
    }

    for name, ckpt in CKPT.items():
        print(f"\n=== {name} ===", flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=ckpt)
        model = model.to(DEVICE).eval()
        tok = open_clip.get_tokenizer('ViT-L-14')

        cache = FEAT_CACHE[name]
        if os.path.exists(cache):
            # canonical stores use different npz key conventions (train/train_y vs.
            # Xtr/ytr); load_feature_cache handles both (scripts/lp_plus_plus.py)
            X, y, Xte, yte = load_feature_cache(cache)
            print(f"features from cache {cache} (train {X.shape})", flush=True)
        else:
            X, y = extract_clip(model, preprocess, train_ds_eval, DEVICE)
            Xte, yte = extract_clip(model, preprocess, test_ds, DEVICE)
            np.savez_compressed(cache, train_feats=X, train_labels=y,
                                test_feats=Xte, test_labels=yte)
            print(f"features extracted + cached: train {X.shape} test {Xte.shape}", flush=True)
        assert len(set(y.tolist())) == 16 and len(yte) == 138
        # The assert above only validates the CACHE labels; the next line then
        # silently replaced them with dataset-order labels, leaving X in cache
        # order. If the two orders ever differ -- or if len(y) < n_train, since
        # labels.json is class-grouped and a prefix would cover only the first
        # few classes -- every downstream use of X/y would be mispaired with no
        # error. Assert they are the SAME array instead of overwriting.
        assert np.array_equal(y, y_full[:len(y)]), (
            "cached feature labels differ from dataset-order labels; X and y "
            "would be mispaired")

        # augmented views (pre-extracted once per checkpoint)
        vcache = AUG_CACHE[name]
        if os.path.exists(vcache):
            views = np.load(vcache)['views']
            print(f"aug views from cache {vcache} {views.shape}", flush=True)
        else:
            print(f"pre-extracting {N_VIEWS} aug views x {len(train_ds)} images ...", flush=True)
            views = extract_aug_views(model, preprocess, train_ds, N_VIEWS)
            np.savez_compressed(vcache, views=views)
            print(f"aug views cached: {views.shape}", flush=True)

        T = text_embeddings(model, tok, class_names)
        zs_acc = float(((Xte @ T.T).argmax(1) == yte).mean())
        print(f"zero-shot (14-template ensemble) {zs_acc*100:.1f}%", flush=True)

        ck_out = {'zero_shot': round(zs_acc, 4)}
        for K in [1, 2, 5, 10, -1]:
            seeds = [0] if K == -1 else range(N_SEEDS)
            accs, eps_best, probe = [], [], []
            for seed in seeds:
                if K == -1:
                    sup = np.arange(len(y)); val = idx_val
                else:
                    sup = kshot_subset_idx(y, K, seed + 500)
                    val = disjoint_kshot_idx(y, K, sup, seed + 1500)

                torch.manual_seed(seed)
                pl = PromptLearner(model, tok, class_names, seed)
                opt = torch.optim.SGD(pl.parameters(), LR, momentum=0)
                base = lrs.CosineAnnealingLR(opt, T_max=EPOCHS)
                sched = ConstantWarmupScheduler(opt, base, WARMUP_EPOCH, WARMUP_CONS_LR)

                sv = torch.tensor(views[:, sup], device=DEVICE)    # (V, s, d) fp16
                sl = torch.tensor(np.asarray(y[sup]), device=DEVICE)
                svv = torch.tensor(np.asarray(X[val]), device=DEVICE)   # clean eval-transform val (fixed, per official CoOp)
                yv_t = torch.tensor(np.asarray(y[val]), device=DEVICE)
                Xte_t = torch.tensor(Xte, device=DEVICE)
                yte_t = torch.tensor(np.asarray(yte), device=DEVICE)

                best_val, best_test, best_ep = -1.0, 0.0, -1
                for ep in range(EPOCHS):
                    perm = torch.randperm(len(sup), device=DEVICE)
                    for i in range(0, len(sup), BS):
                        b = perm[i:i + BS]
                        imgf = sv[ep % N_VIEWS][b].float()
                        tf = text_encode(model, pl.prompt_tokens(), pl.tokenized)
                        logits = model.logit_scale.exp() * (imgf @ tf.T)
                        loss = F.cross_entropy(logits, sl[b])
                        opt.zero_grad(); loss.backward(); opt.step()
                    sched.step()
                    with torch.no_grad():
                        tf = text_encode(model, pl.prompt_tokens(), pl.tokenized)
                        lv = model.logit_scale.exp() * (svv.float() @ tf.T)
                        at_ = model.logit_scale.exp() * (Xte_t @ tf.T)
                        av = float(lv.argmax(1).eq(yv_t).float().mean())
                        at = float(at_.argmax(1).eq(yte_t).float().mean())
                    if av >= best_val:
                        best_val, best_test, best_ep = av, at, ep

                accs.append(round(float(best_test), 4))
                eps_best.append(int(best_ep))
                clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
                clf.fit(X[sup], y[sup])
                probe.append(round(float((clf.predict(Xte) == yte).mean()), 4))
                print(f"  K={K} seed {seed:2d}: best ep {best_ep:3d} (val {best_val*100:5.1f}%) -> test {best_test*100:5.1f}%", flush=True)

            tag = f"K={K}" if K > 0 else "full"
            ck_out[tag] = {
                'coop_official': {'mean': round(float(np.mean(accs)), 4),
                                  'std': round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0,
                                  'accs': accs, 'best_epochs': eps_best, 'n_seeds': len(accs)},
                'probe_C1': {'mean': round(float(np.mean(probe)), 4),
                             'std': round(float(np.std(probe, ddof=1)), 4) if len(probe) > 1 else 0.0,
                             'accs': probe},
            }
            print(f"{tag:5s}: CoOp-official {np.mean(accs)*100:5.1f}% +/- {np.std(accs, ddof=1)*100:4.1f}  |  "
                  f"probe(C=1) {np.mean(probe)*100:5.1f}%", flush=True)
        R['checkpoints'][name] = ck_out
        del model, views
        torch.cuda.empty_cache()

    R['elapsed_sec'] = round(time.time() - t0, 1)
    json.dump(R, open(OUT, 'w'), indent=2)
    print(f"\nDone in {R['elapsed_sec']}s -> {OUT}", flush=True)


if __name__ == '__main__':
    main()
