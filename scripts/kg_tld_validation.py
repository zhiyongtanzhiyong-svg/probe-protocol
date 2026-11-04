"""
KG prompt validation on TLD cross-dataset.
Uses TPDD->KG mapping via TLD_TO_TPDD to create KG-augmented prompts for TLD.
"""
import os
import sys, json, torch, numpy as np
from datasets import load_dataset
sys.path.insert(0, ROOT + '/scripts')

import open_clip
from dataset import EVAL_TRANSFORM, worker_init_fn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_pil(img_tensor):
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
EVAL_TRANSFORM_LOCAL = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    NORMALIZE,
])

TLD_CLASS_NAMES = [
    'Brown spot', 'Frogeye', 'Healthy', 'Mosaic', 'Potato virus Y',
    'Target spot', 'Weather fleck', 'alternaria alternata',
    'cercospora nicotianae', 'no cercospora nicotianae or alternaria alternata present'
]

# Mapping from TLD class index to TPDD class name for KG lookup
TLD_TO_TPDD = {
    0: 'brown_spot', 1: 'frog_eye', 2: 'healthy', 3: 'tmv', 4: 'pvy',
    5: 'target_spot', 6: 'weather_fleck', 7: 'brown_spot',
    8: 'frog_eye', 9: 'healthy',
}

class TLDDatasetSimple(Dataset):
    def __init__(self, hf_dataset, indices, transform=EVAL_TRANSFORM_LOCAL):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.hf_dataset[self.indices[idx]]
        img = item['image'].convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'label': int(item['label'])}

def main():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model = model.eval().cuda()

    # Load KG data
    tpd_map = json.load(open(ROOT + '/data/tpd_to_kg_map.json'))
    kg_data = json.load(open(ROOT + '/data/tobacco_records.json'))

    # Build KG symptom descriptions for TPDD class names
    def get_kg_symptom(tpdd_name):
        if tpdd_name == 'healthy':
            return "healthy tobacco leaf without any disease spots or damage"
        info = tpd_map.get(tpdd_name, {})
        kg_ent = info.get('kg_entity')
        if not kg_ent:
            return f"{tpdd_name} disease symptoms on tobacco leaf"
        kg_name = kg_ent.replace('D_', '')
        for r in kg_data:
            if kg_name.lower() in r.get('name', '').lower() or tpdd_name.lower() in r.get('english_name', '').lower():
                return r.get('symptoms', '').split('。')[0][:100]
        return f"{tpdd_name} disease symptoms on tobacco leaf"

    # Build TLD KG prompts via TPDD mapping
    tld_kg_prompts = []
    tld_en_photo_prompts = []
    for i, name in enumerate(TLD_CLASS_NAMES):
        tpdd_name = TLD_TO_TPDD.get(i, name.lower().replace(' ', '_'))
        kg_sym = get_kg_symptom(tpdd_name)
        tld_kg_prompts.append(f"tobacco leaf: {kg_sym}")
        tld_en_photo_prompts.append(f"a photo of a tobacco leaf with {name} disease")

    # Load TLD test set
    hf_ds = load_dataset('Joker5800/Tobacco_Leaf_Diseases', split='test', trust_remote_code=True)
    rng = np.random.RandomState(42)
    indices = list(range(len(hf_ds)))
    rng.shuffle(indices)
    n_train = int(len(indices) * 0.06)
    n_val = int(len(indices) * 0.04)
    test_idx = indices[n_train + n_val:]
    test_ds = TLDDatasetSimple(hf_ds, test_idx, transform=EVAL_TRANSFORM_LOCAL)

    loader = DataLoader(test_ds, 16, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)

    prompt_sets = {"EN-Photo (TLD)": tld_en_photo_prompts, "KG-Symptom (via TPDD)": tld_kg_prompts}

    results = []
    for set_name, prompts in prompt_sets.items():
        tokens = tokenizer(prompts).cuda()
        with torch.no_grad():
            tf = model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        correct, total = 0, 0
        for batch in loader:
            imgs, labels = batch['image'], batch['label']
            clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).cuda()
            with torch.no_grad():
                ife = model.encode_image(clip_imgs)
                ife = ife / ife.norm(dim=-1, keepdim=True)
                logits = (ife @ tf.T) * 100
            preds = logits.argmax(-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total * 100
        results.append({"template": set_name, "accuracy": round(acc, 2)})
        print(f"{set_name:30s}: {acc:.1f}%")

    kg_acc = results[1]["accuracy"]
    en_acc = results[0]["accuracy"]
    print(f"\nKG advantage on TLD (via TPDD mapping): {kg_acc - en_acc:+.1f} pp")
    print(f"Compare with TPDD KG advantage: +1.4 pp (KG-Symptom 15.2% vs EN-Photo 13.8%)")

    json.dump(results, open(ROOT + '/data/tld_kg_prompts.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
