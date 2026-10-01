"""
CLIP Few-Shot Linear Probe — FIXED: inverse ImageNet normalization.
"""
import sys, json, torch, numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '/home/lizhuo/Work/TobAI/Paper2-CLIPvsResNet/scripts')

import open_clip
from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_pil(img_tensor):
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

class KShotSubset(Dataset):
    def __init__(self, full_ds, K, seed):
        self.transform = full_ds.transform
        rng = np.random.RandomState(seed)
        indices_by_class = {i: [] for i in range(len(full_ds.class_names))}
        for idx, lbl in enumerate(full_ds.label_ids):
            indices_by_class[lbl].append(idx)
        idxs = []
        for c, idx_list in indices_by_class.items():
            idxs.extend(rng.choice(idx_list, size=min(K, len(idx_list)), replace=False))
        self.full_ds = full_ds
        self.indices = sorted(idxs)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.full_ds[self.indices[idx]]

def main():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='/home/lizhuo/Work/TobAI/checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin')
    model = model.eval().cuda()

    @torch.no_grad()
    def get_features(dataset):
        loader = DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
        feats, labs = [], []
        for batch in loader:
            imgs = batch['image']
            clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).cuda()
            f = model.encode_image(clip_imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
        return np.concatenate(feats), np.array(labs)

    train_full = TPDDDataset('train', transform=TRAIN_TRANSFORM)
    test_ds = TPDDDataset('test', transform=EVAL_TRANSFORM)
    test_feats, test_labels = get_features(test_ds)

    N_SEEDS = 15
    results = {}
    for K in [1, 2, 5, 10, -1]:
        accs = []
        for seed in range(N_SEEDS):
            if K == -1:
                train_subset = train_full
            else:
                train_subset = KShotSubset(train_full, K, seed + 500)
            train_feats, train_labels = get_features(train_subset)
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(train_feats, train_labels)
            preds = clf.predict(test_feats)
            accs.append(float((preds == test_labels).mean()))
        accs = np.array(accs)
        label = f"K={K}" if K > 0 else "full"
        results[label] = {"mean": float(accs.mean()), "std": float(accs.std(ddof=1)),
                           "accs": accs.tolist(), "n_seeds": N_SEEDS}
        print(f"CLIP probe {label:5s}: {accs.mean()*100:.1f}% ± {accs.std(ddof=1)*100:.1f}")

    # Write BOTH names. Six consumers (generate_figures, rigorous_analysis,
    # two_model_e4, verify_claims, paired_kshot_a4, check_tex_vs_data) read
    # data/clip_fewshot.json, but this script only wrote the "_fixed" variant,
    # so the two files were kept in sync by hand. The next run would have
    # updated only "_fixed" and every consumer would silently have kept reading
    # a stale artifact. One producer, one canonical filename, plus the alias.
    for _name in ("clip_fewshot.json", "clip_fewshot_fixed.json"):
        json.dump(results, open(f'/home/lizhuo/Work/TobAI/Paper2-CLIPvsResNet/data/{_name}', 'w'), indent=2)
    print("Saved data/clip_fewshot.json (+ _fixed alias).")

if __name__ == '__main__':
    main()
