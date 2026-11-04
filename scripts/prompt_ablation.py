"""
CLIP Zero-Shot & Prompt Ablation for Tobacco Disease Classification.
Fixed: inverse ImageNet normalization before converting to PIL.
"""
import os
import sys, json, torch, numpy as np
sys.path.insert(0, ROOT + '/scripts')

import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM, worker_init_fn
from PIL import Image
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Inverse ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_pil(img_tensor):
    """Correctly convert ImageNet-normalized tensor back to PIL Image."""
    # img_tensor: (C, H, W), normalized by (x - mean) / std
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN  # inverse normalize
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
                # Fallback: use class name + generic template
                cn = cn_map.get(name, name)
                kg_symptoms[name] = f"{cn} disease symptoms on tobacco leaf"
                kg_conditions[name] = f"conditions favoring {cn} development"

    templates = [
        ("EN-Label","{class_name}"),
        ("CN-Label","{cn_name}"),
        ("EN-Photo","a photo of a tobacco leaf with {class_name} disease"),
        ("CN-Photo","一张患有{cn_name}的烟草叶片照片"),
        ("EN-Symptom","tobacco leaf showing {class_name} symptoms"),
        ("CN-Symptom","烟草叶片表现出{cn_name}的症状"),
        ("EN-Closeup","a close-up photo of {class_name} on tobacco plant"),
        ("CN-Closeup","烟草植株上{cn_name}的特写照片"),
        ("EN-Disease","{class_name} disease on tobacco leaf"),
        ("CN-Disease","烟草叶片上的{cn_name}病害"),
        ("KG-Symptom","tobacco leaf: {kg_symptom}"),
        ("KG-Condition","tobacco leaf disease occurring when: {kg_condition}"),
        ("KG-Combined","tobacco leaf with {kg_symptom}, occurs when {kg_condition}"),
        ("EN-Expert","tobacco plant pathology: {class_name} disease identification"),
    ]

    test_ds = TPDDDataset('test', transform=EVAL_TRANSFORM)
    loader = DataLoader(test_ds, 16, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    results = []

    for tname, template in templates:
        prompts = []
        for name in class_names:
            s = template.replace('{class_name}', name).replace('{cn_name}', cn_map.get(name, name))
            s = s.replace('{kg_symptom}', kg_symptoms.get(name, ''))
            s = s.replace('{kg_condition}', kg_conditions.get(name, ''))
            prompts.append(s)

        tokens = tokenizer(prompts).cuda()
        with torch.no_grad():
            tf = model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)

        correct, total = 0, 0
        per_class = {i: {'c': 0, 't': 0} for i in range(16)}
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
            for p, l in zip(preds.tolist(), labels.tolist()):
                per_class[l]['c'] += (p == l); per_class[l]['t'] += 1

        acc = correct / total * 100
        nonzero = sum(1 for i in range(16) if per_class[i]['c'] > 0)
        scores = [(class_names[i], per_class[i]['c']/max(per_class[i]['t'],1)*100) for i in range(16)]
        scores.sort(key=lambda x: x[1], reverse=True)
        results.append({"template": tname, "accuracy": acc, "nonzero": nonzero,
                         "per_class": {class_names[i]: per_class[i]['c']/max(per_class[i]['t'],1)*100 for i in range(16)}})
        print(f"{tname:25s} {acc:5.1f}% ({nonzero}/16 nonzero)")

    # Write both names. rigorous_analysis.py and zero_shot_clean.py read
    # prompt_ablation.json, but this script only wrote prompt_ablation_fixed.json,
    # so the "fixed" rerun could never reach the consumers -- and the file it
    # writes did not exist in data/ at all, meaning the fixed pipeline had never
    # actually been run. One producer, one canonical filename, plus the alias.
    for _name in ("prompt_ablation.json", "prompt_ablation_fixed.json"):
        json.dump(results, open(f'{ROOT}/data/{_name}', 'w'), indent=2)
    print("Saved data/prompt_ablation.json (+ _fixed alias).")
    print(f"\nSaved {len(results)} templates.")

if __name__ == '__main__':
    main()
