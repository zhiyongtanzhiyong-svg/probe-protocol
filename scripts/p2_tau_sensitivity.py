"""
P2-6: τ sensitivity analysis for zero-shot CLIP.
Tests how accuracy changes with different temperature values.
"""
import os
import sys, json, torch, numpy as np, time, os
sys.path.insert(0, ROOT + '/scripts')
import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM, worker_init_fn
from PIL import Image
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIP_CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

TEMPLATES = {
    'EN-Label': "a photo of a {class_name} tobacco leaf",
    'CN-Label': "一张{class_name}烟草叶片的照片",
    'EN-Photo': "tobacco leaf showing {class_name}",
    'CN-Photo': "显示{class_name}的烟草叶片",
    'EN-Symptom': "tobacco leaf with symptoms of {class_name}",
    'CN-Symptom': "具有{class_name}症状的烟草叶片",
    'KG-Symptom': "tobacco leaf infected with {class_name}, showing characteristic lesions",
}

def tensor_to_pil(img_tensor):
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    t0 = time.time()

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained=CLIP_CKPT)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model = model.eval().to(device)

    test_ds = TPDDDataset('test', transform=EVAL_TRANSFORM)
    class_names = test_ds.class_names
    test_labels = np.array(test_ds.label_ids)
    N_CLASSES = len(class_names)
    n_test = len(test_ds)
    majority_baseline = float(np.bincount(test_labels).max() / n_test)
    print(f'Test set: {n_test} images, {N_CLASSES} classes, majority baseline: {majority_baseline*100:.1f}%')

    # Extract test features
    loader = DataLoader(test_ds, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    test_feats = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image']
            clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).to(device)
            f = model.encode_image(clip_imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            test_feats.append(f.cpu().numpy())
    test_feats = np.concatenate(test_feats)

    # Get text features for each template
    text_feats = {}
    for tname, ttmpl in TEMPLATES.items():
        texts = [ttmpl.format(class_name=cn) for cn in class_names]
        toks = tokenizer(texts).to(device)
        with torch.no_grad():
            tf = model.encode_text(toks)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        text_feats[tname] = tf.cpu().numpy()

    # Sweep τ values
    tau_values = [10, 25, 50, 100, 200, 500, 1000]
    results = {
        'n_test': n_test,
        'n_classes': N_CLASSES,
        'majority_baseline': majority_baseline,
        'tau_sweep': {},
    }

    print(f'\n{"τ":>6s} | ' + ' | '.join(f'{t:>12s}' for t in TEMPLATES))
    print('-' * (8 + 15 * len(TEMPLATES)))

    for tau in tau_values:
        row = {}
        for tname in TEMPLATES:
            # Cosine similarity × τ, then softmax → argmax
            sim = test_feats @ text_feats[tname].T  # (n_test, n_classes)
            logits = sim * tau
            # Softmax
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            preds = probs.argmax(axis=1)
            acc = float((preds == test_labels).mean())
            row[tname] = acc

        results['tau_sweep'][str(tau)] = row
        print(f'{tau:6d} | ' + ' | '.join(f'{v*100:12.1f}%' for v in row.values()))

    # Find best τ for each template
    print('\n=== Best τ per template ===')
    for tname in TEMPLATES:
        best_tau = max(results['tau_sweep'], key=lambda t: results['tau_sweep'][t][tname])
        best_acc = results['tau_sweep'][best_tau][tname]
        tau100_acc = results['tau_sweep']['100'][tname]
        print(f'  {tname:15s}: best τ={best_tau:>4s} ({best_acc*100:.1f}%)  vs  τ=100 ({tau100_acc*100:.1f}%)  Δ={(best_acc-tau100_acc)*100:+.1f}pp')

    results['elapsed_s'] = float(time.time() - t0)
    results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'p2_tau_sensitivity.json')
    json.dump(results, open(out_path, 'w'), indent=2)
    print(f'\nSaved to {out_path}')
    print(f'Elapsed: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
