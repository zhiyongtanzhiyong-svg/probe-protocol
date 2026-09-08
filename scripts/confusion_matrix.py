"""
Generate confusion matrix figure comparing CLIP full-data probe vs ResNet-18 full-data.
"""
import sys, json, torch, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
sys.path.insert(0, '/home/broadsense/works/lizhuo/TobAI/Paper1-KGRegularizer/scripts')

import open_clip
from sklearn.linear_model import LogisticRegression
from dataset import TPDDDataset, EVAL_TRANSFORM, TRAIN_TRANSFORM, worker_init_fn
from PIL import Image
from torch.utils.data import DataLoader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_pil(img_tensor):
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

def get_features(model, preprocess, dataset):
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

def main():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='/home/broadsense/works/lizhuo/scb5-zeroshot/ckpts/dfn_clip_vitl14/open_clip_pytorch_model.bin')
    model = model.eval().cuda()

    train_full = TPDDDataset('train', transform=TRAIN_TRANSFORM)
    test_ds = TPDDDataset('test', transform=EVAL_TRANSFORM)
    tpd_map = json.load(open('/home/broadsense/works/lizhuo/TobAI/Paper1-KGRegularizer/data/tpd_to_kg_map.json'))
    class_names_human = [n.replace('_', ' ').title() for n in sorted(tpd_map.keys())]

    # CLIP probe predictions
    train_feats, train_labels = get_features(model, preprocess, train_full)
    test_feats, test_labels = get_features(model, preprocess, test_ds)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(train_feats, train_labels)
    clip_preds = clf.predict(test_feats)

    # ResNet full-data predictions (use exp_rn18_15 style)
    from model_zoo import ResNetBaseline, set_seed
    import torch.nn as nn
    from torch.utils.data import DataLoader

    DEVICE = torch.device('cuda')
    val_full = TPDDDataset('validation', transform=EVAL_TRANSFORM)
    NUM_CLASSES = len(train_full.class_names)

    val_loader = DataLoader(val_full, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_ds, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)

    # Train ResNet full-data with seed 0
    set_seed(0)
    model_rn = ResNetBaseline(NUM_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(model_rn.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    crit = nn.CrossEntropyLoss()

    best_val, best_state = 0, None
    for ep in range(30):
        model_rn.train()
        for batch in DataLoader(train_full, 32, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn):
            img, lbl = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
            opt.zero_grad()
            logits = model_rn(img)
            loss = crit(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_rn.parameters(), 1.0)
            opt.step()
        sch.step()

        model_rn.eval()
        val_correct, val_total = 0, 0
        for batch in val_loader:
            img, lbl = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
            logits = model_rn(img)
            _, p = logits.max(1)
            val_correct += (p == lbl).sum().item()
            val_total += lbl.size(0)
        val_acc = val_correct / val_total
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model_rn.state_dict().items()}

    model_rn.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model_rn.eval()
    resnet_preds = []
    for batch in test_loader:
        img = batch['image'].to(DEVICE)
        logits = model_rn(img)
        _, p = logits.max(1)
        resnet_preds.extend(p.cpu().numpy().tolist())
    resnet_preds = np.array(resnet_preds)

    # Plot confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    shorten = [c.replace('Potato Tuber Moth', 'Potato Tuber') for c in class_names_human]

    for ax, preds, title in zip(axes, [clip_preds, resnet_preds],
                                 ['CLIP Full-data Probe', 'ResNet-18 Full-data']):
        cm = confusion_matrix(test_labels, preds, labels=range(NUM_CLASSES))
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(shorten, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(shorten, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                       fontsize=6, color='white' if cm_norm[i,j] > 0.5 else 'black')

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    plt.tight_layout()
    out_path = '/home/broadsense/works/lizhuo/TobAI/Paper2-CLIPvsResNet/figures/fig_confusion.pdf'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved confusion matrix to {out_path}")

    # Per-class accuracy from this run
    clip_correct = clip_preds == test_labels
    resnet_correct = resnet_preds == test_labels
    print(f"\nCLIP probe acc: {clip_correct.mean()*100:.1f}%")
    print(f"ResNet acc: {resnet_correct.mean()*100:.1f}%")

    perclass = {"clip_full": {}, "resnet_full": {}}
    for i, name in enumerate(train_full.class_names):
        mask = test_labels == i
        if mask.sum() > 0:
            perclass["clip_full"][name] = round(float(clip_correct[mask].mean() * 100), 1)
            perclass["resnet_full"][name] = round(float(resnet_correct[mask].mean() * 100), 1)
        else:
            perclass["clip_full"][name] = 0
            perclass["resnet_full"][name] = 0

    json.dump(perclass, open('/home/broadsense/works/lizhuo/TobAI/Paper2-CLIPvsResNet/data/perclass_comparison_v2.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
