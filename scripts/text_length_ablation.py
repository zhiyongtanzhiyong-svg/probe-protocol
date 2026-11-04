"""
Text-length ablation for CLIP zero-shot.
FIXED: No truncation. Uses a common target length, pads ALL prompts to that length.
"""
import os
import sys, json, torch, numpy as np
sys.path.insert(0, ROOT + '/scripts')

import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM, worker_init_fn
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

    # Build KG symptom prompts (same as prompt_ablation.py)
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

    kg_prompts = [f"tobacco leaf: {kg_symptoms[name]}" for name in class_names]
    kg_lengths = [len(p) for p in kg_prompts]
    print("KG-Symptom prompts:")
    for n, l, p in zip(class_names, kg_lengths, kg_prompts):
        print(f"  {n:25s} len={l:2d}  '{p[:60]}{'...' if len(p)>60 else ''}'")

    # Build EN-Photo base
    base_template = "a photo of a tobacco leaf with {class_name} disease"
    en_photo_base = [base_template.format(class_name=n) for n in class_names]
    en_base_lengths = [len(p) for p in en_photo_base]

    # Determine common target length: max of max(KG_len, EN_base_len)
    max_per_class = [max(kl, el) for kl, el in zip(kg_lengths, en_base_lengths)]
    target_len = max(max_per_class)
    print(f"\nCommon target length: {target_len} chars")

    # Pad all prompts to target_len
    pad_char = ' '
    def pad_to(s, length):
        if len(s) >= length:
            return s[:length]  # For KG prompts that are longer than target, truncate
        return s + pad_char * (length - len(s))

    kg_padded = [pad_to(p, target_len) for p in kg_prompts]
    en_photo_padded = [pad_to(p, target_len) for p in en_photo_base]
    en_symptom_base = [f"tobacco leaf showing {n} symptoms" for n in class_names]
    en_symptom_padded = [pad_to(p, target_len) for p in en_symptom_base]

    # Verify lengths
    assert all(len(p) == target_len for p in kg_padded)
    assert all(len(p) == target_len for p in en_photo_padded)
    assert all(len(p) == target_len for p in en_symptom_padded)

    test_ds = TPDDDataset('test', transform=EVAL_TRANSFORM)
    loader = DataLoader(test_ds, 16, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)

    prompt_sets = {
        "EN-Photo (original)": en_photo_base,
        "EN-Photo (padded)": en_photo_padded,
        "EN-Symptom (original)": en_symptom_base,
        "EN-Symptom (padded)": en_symptom_padded,
        "KG-Symptom (original)": kg_prompts,
        "KG-Symptom (padded)": kg_padded,
    }

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
        avg_len = np.mean([len(p) for p in prompts])
        results.append({"template": set_name, "accuracy": round(acc, 2), "avg_len": round(avg_len, 1)})
        print(f"{set_name:30s}: {acc:.1f}%  (avg {avg_len:.0f} chars)")

    # Key comparison
    kg_orig = results[4]["accuracy"]
    kg_pad = results[5]["accuracy"]
    en_orig = results[0]["accuracy"]
    en_pad = results[1]["accuracy"]
    print(f"\n{'='*60}")
    print(f"KEY FINDING: Length-controlled comparison")
    print(f"  EN-Photo (original, avg {np.mean(en_base_lengths):.0f}ch): {en_orig:.1f}%")
    print(f"  EN-Photo (padded to {target_len}ch)     : {en_pad:.1f}%")
    print(f"  KG-Symptom (original, varying)          : {kg_orig:.1f}%")
    print(f"  KG-Symptom (padded to {target_len}ch)   : {kg_pad:.1f}%")
    print(f"  KG advantage over padded EN-Photo       : {kg_pad - en_pad:+.1f} pp")
    print(f"{'='*60}")

    json.dump(results, open(ROOT + '/data/text_length_ablation.json', 'w'), indent=2)
    print("\nSaved.")


if __name__ == '__main__':
    main()
