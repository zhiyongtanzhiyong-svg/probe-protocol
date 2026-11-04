"""
Confidence calibration analysis for CLIP zero-shot and probe.
FIXED: Chinese fallback for unmatched KG classes (matches prompt_ablation.py behavior).
"""
import os
import sys, json, torch, numpy as np
sys.path.insert(0, ROOT + '/scripts')

import open_clip
from sklearn.linear_model import LogisticRegression
from dataset import TPDDDataset, EVAL_TRANSFORM, TRAIN_TRANSFORM, worker_init_fn
from PIL import Image
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_pil(img_tensor):
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if i == n_bins - 1:
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            bin_size = in_bin.sum()
            ece += np.abs(bin_conf - bin_acc) * (bin_size / len(confidences))
            bin_data.append({'bin': i, 'range': f'{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}',
                             'acc': round(float(bin_acc), 3), 'conf': round(float(bin_conf), 3),
                             'size': int(bin_size), 'gap': round(float(abs(bin_conf - bin_acc)), 3)})
        else:
            bin_data.append({'bin': i, 'range': f'{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}',
                             'acc': 0, 'conf': 0, 'size': 0, 'gap': 0})
    return ece, bin_data


def main():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model = model.eval().cuda()

    tpd_map = json.load(open(ROOT + '/data/tpd_to_kg_map.json'))
    kg_data = json.load(open(ROOT + '/data/tobacco_records.json'))
    class_names = list(tpd_map.keys())

    cn_map = {
        "anthracnose":"炭疽病","black_shank":"黑胫病","brown_spot":"赤星病",
        "cmv":"黄瓜花叶病毒病","frog_eye":"蛙眼病","genetic_abnormality":"遗传异常",
        "healthy":"健康","nematodes":"根结线虫病","potato_tuber_moth":"马铃薯块茎蛾",
        "pvy":"马铃薯Y病毒病","sunscald":"日灼病","target_spot":"靶斑病",
        "tmv":"烟草普通花叶病毒病","tswv":"番茄斑萎病毒病","weather_fleck":"气候斑点病",
        "wildfire":"野火病"
    }

    # Build KG symptoms (exactly matching prompt_ablation.py)
    kg_symptoms = {}
    for name, info in tpd_map.items():
        if name == 'healthy':
            kg_symptoms[name] = "healthy tobacco leaf without any disease spots or damage"
            continue
        kg_ent = info.get('kg_entity')
        if kg_ent:
            kg_name = kg_ent.replace('D_', '')
            matched = False
            for r in kg_data:
                if kg_name.lower() in r.get('name', '').lower() or name.lower() in r.get('english_name', '').lower():
                    kg_symptoms[name] = r.get('symptoms', '').split('。')[0][:100]
                    matched = True
                    break
            if not matched:
                cn = cn_map.get(name, name)
                kg_symptoms[name] = f"{cn} disease symptoms on tobacco leaf"
        else:
            cn = cn_map.get(name, name)
            kg_symptoms[name] = f"{cn} disease symptoms on tobacco leaf"

    # Zero-shot with KG-Symptom prompts
    prompts = [f"tobacco leaf: {kg_symptoms.get(name, '')}" for name in class_names]
    tokens = tokenizer(prompts).cuda()
    with torch.no_grad():
        text_feats = model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    test_ds = TPDDDataset('test', transform=EVAL_TRANSFORM)
    loader = DataLoader(test_ds, 16, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)

    all_logits, all_labels = [], []
    for batch in loader:
        imgs, labels = batch['image'], batch['label']
        clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).cuda()
        with torch.no_grad():
            img_feats = model.encode_image(clip_imgs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = (img_feats @ text_feats.T) * 100
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    # Max-subtract before exp: logits are cosine*100 in fp32, whose overflow
    # threshold is exp(88.7), so any cosine above ~0.887 produced inf and then
    # inf/inf -> NaN rows. It has not overflowed at the current checkpoint, but
    # the fix is free and the correct forms already exist elsewhere in the repo
    # (p2_tau_sensitivity.py, verify_c2_probes.py).
    _lg = all_logits - all_logits.max(axis=1, keepdims=True)
    all_probs = np.exp(_lg) / np.exp(_lg).sum(axis=1, keepdims=True)

    ece_zs, bins_zs = compute_ece(all_probs, all_labels)
    zs_acc = (all_logits.argmax(1) == all_labels).mean() * 100
    zs_conf = all_probs.max(1)

    print(f"=== Zero-shot (KG-Symptom) ===")
    print(f"  Accuracy: {zs_acc:.1f}%")
    print(f"  ECE: {ece_zs:.4f}")
    print(f"  Mean confidence: {zs_conf.mean():.3f}")
    print(f"  Confidence histogram:")
    for b in bins_zs:
        if b['size'] > 0:
            marker = ' <-- overconfident' if b['gap'] > 0.05 else ' <-- underconfident' if b['gap'] < -0.05 else ''
            print(f"    [{b['range']}] acc={b['acc']:.3f} conf={b['conf']:.3f} gap={b['gap']:+.3f} n={b['size']}{marker}")

    # Confidence breakdown: correct vs incorrect
    zs_correct = all_logits.argmax(1) == all_labels
    print(f"\n  Confidence breakdown:")
    print(f"    Correct predictions: mean conf={zs_conf[zs_correct].mean():.3f}")
    print(f"    Incorrect predictions: mean conf={zs_conf[~zs_correct].mean():.3f}")

    # CLIP Full-data probe calibration.
    # CANONICAL PROTOCOL: the train split is featurised with EVAL_TRANSFORM, not
    # TRAIN_TRANSFORM. Using the stochastic training transform here made the
    # probe features run-dependent and put this script's accuracy at 73.9% --
    # the frozen-ResNet number, not the 76.8% CLIP probe the paper reports
    # (Appendix A documents the same class of bug elsewhere). The deterministic
    # transform reproduces canonical_results.json exactly.
    train_full = TPDDDataset('train', transform=EVAL_TRANSFORM)
    test_ds2 = TPDDDataset('test', transform=EVAL_TRANSFORM)

    def get_features(dataset):
        dl = DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
        feats, labs = [], []
        for batch in dl:
            imgs, labels = batch['image'], batch['label']
            clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).cuda()
            with torch.no_grad():
                f = model.encode_image(clip_imgs)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labs.extend(labels.numpy().tolist())
        return np.concatenate(feats), np.array(labs)

    train_feats, train_labels = get_features(train_full)
    test_feats, test_labels = get_features(test_ds2)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(train_feats, train_labels)
    probe_probs = clf.predict_proba(test_feats)
    probe_preds = clf.predict(test_feats)
    probe_acc = (probe_preds == test_labels).mean() * 100
    probe_conf = probe_probs.max(1)

    # Pad by CLASS INDEX, not by column position: sklearn's LogisticRegression
    # is fitted on whatever labels appear in the support set, so column j of
    # probe_probs is class clf.classes_[j]. Padding with [:n] would fabricate
    # zero-probability mass for absent classes. Dead today (all 16 classes are
    # present and classes_ == 0..15) but wrong the moment a class is absent.
    classes = getattr(clf, "classes_", np.arange(probe_probs.shape[1]))
    padded = np.zeros((probe_probs.shape[0], 16))
    padded[:, classes] = probe_probs
    probe_probs = padded

    ece_probe, bins_probe = compute_ece(probe_probs, test_labels)
    probe_correct = probe_preds == test_labels

    print(f"\n=== CLIP Full-data Probe ===")
    print(f"  Accuracy: {probe_acc:.1f}%")
    print(f"  ECE: {ece_probe:.4f}")
    print(f"  Mean confidence: {probe_conf.mean():.3f}")
    print(f"  Confidence breakdown:")
    print(f"    Correct predictions: mean conf={probe_conf[probe_correct].mean():.3f}")
    print(f"    Incorrect predictions: mean conf={probe_conf[~probe_correct].mean():.3f}")
    for b in bins_probe:
        if b['size'] > 0:
            print(f"    [{b['range']}] acc={b['acc']:.3f} conf={b['conf']:.3f} gap={b['gap']:+.3f} n={b['size']}")

    result = {
        "zero_shot": {"accuracy": round(zs_acc, 2), "ece": round(ece_zs, 4), "bins": bins_zs,
                       "mean_confidence": round(float(zs_conf.mean()), 3),
                       "correct_conf": round(float(zs_conf[zs_correct].mean()), 3),
                       "incorrect_conf": round(float(zs_conf[~zs_correct].mean()), 3)},
        "probe_full": {"accuracy": round(probe_acc, 2), "ece": round(ece_probe, 4), "bins": bins_probe,
                        "mean_confidence": round(float(probe_conf.mean()), 3),
                        "correct_conf": round(float(probe_conf[probe_correct].mean()), 3),
                        "incorrect_conf": round(float(probe_conf[~probe_correct].mean()), 3)},
    }
    json.dump(result, open(ROOT + '/data/calibration_results.json', 'w'), indent=2)
    print("\nSaved.")


if __name__ == '__main__':
    main()
