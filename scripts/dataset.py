"""
TPDD Tobacco Leaf Disease Dataset.
Wraps image data with KG entity mapping.
"""

import json, os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

DATA_ROOT = Path(os.environ.get('TPDD_DATA_ROOT', Path(__file__).resolve().parent.parent / 'data' / 'tpdd'))
PROJ_ROOT = Path(__file__).parent.parent
TPD_MAP_PATH = PROJ_ROOT / 'data' / 'tpd_to_kg_map.json'

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.1),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    NORMALIZE,
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    NORMALIZE,
])


class TPDDDataset(Dataset):
    def __init__(self, split, transform=EVAL_TRANSFORM):
        split_dir = DATA_ROOT / split
        with open(split_dir / 'labels.json') as f:
            labels_data = json.load(f)
        self.images = [DATA_ROOT / split / item['image'] for item in labels_data]
        self.class_labels = [item['label'] for item in labels_data]

        with open(TPD_MAP_PATH) as f:
            self.tpd_map = json.load(f)

        self.class_names = sorted(set(self.class_labels))
        self.name_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.label_ids = [self.name_to_idx[l] for l in self.class_labels]

        # The KG map's KEY ORDER is load-bearing in ~9 call sites: they take
        # `list(tpd_map.keys())` as the text-column order and therefore as the
        # label order, and pair column j of a zero-shot logit matrix with label
        # j purely on the strength of this file's insertion order. Nothing
        # checked that, so regenerating or editing the map with a different key
        # order -- or adding/dropping a class -- would desynchronise every
        # zero-shot prediction from its label with no error and no shape change.
        # Assert it once, here, where every one of those scripts already loads.
        kg_keys = list(self.tpd_map.keys())
        assert kg_keys == sorted(kg_keys), (
            "data/tpd_to_kg_map.json keys are not in sorted order; the scripts "
            "that use list(tpd_map.keys()) as the label order would silently "
            "mis-pair every prediction. Sort the file or fix the call sites.")
        assert set(kg_keys) == set(self.class_names), (
            f"KG map covers {len(kg_keys)} classes but this split has "
            f"{len(self.class_names)}: {set(kg_keys) ^ set(self.class_names)}")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        class_name = self.class_labels[idx]
        mapping = self.tpd_map.get(class_name, {})
        kg_entity_name = mapping.get('kg_entity') or ''
        kg_entity_id = mapping.get('kg_id')
        kg_entity_id = -1 if kg_entity_id is None else int(kg_entity_id)

        return {
            'image': img,
            'label': self.label_ids[idx],
            'class_name': class_name,
            'kg_entity': kg_entity_name,
            'kg_entity_id': kg_entity_id,
        }


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)

def get_dataloaders(batch_size=32, seed=42):
    train_ds = TPDDDataset('train', transform=TRAIN_TRANSFORM)
    val_ds = TPDDDataset('validation', transform=EVAL_TRANSFORM)
    test_ds = TPDDDataset('test', transform=EVAL_TRANSFORM)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn, generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn, generator=g)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn, generator=g)

    print(f"TPDD {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    print(f"  Classes: {len(train_ds.class_names)}")
    print(f"  Label distribution (train):")
    for n, i in train_ds.name_to_idx.items():
        cnt = sum(1 for l in train_ds.label_ids if l == i)
        print(f"    {n}: {cnt}")
    print(f"  KG-mapped classes: {sum(1 for v in train_ds.tpd_map.values() if v['kg_id'] >= 0)}/{len(train_ds.tpd_map)}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_ds,
        'val_dataset': val_ds,
        'test_dataset': test_ds,
        'num_classes': len(train_ds.class_names),
    }


if __name__ == '__main__':
    data = get_dataloaders()
    batch = next(iter(data['train_loader']))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch labels: {batch['label'][:5]}")
    print(f"Batch class_names: {batch['class_name'][:5]}")
