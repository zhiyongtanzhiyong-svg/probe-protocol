"""
Reusable model zoo for tobacco disease classification experiments.
Provides standard model architectures, training loops, and evaluation utilities.
"""

import math, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResNetEncoder(nn.Module):
    def __init__(self, feat_dim=512, backbone='resnet18'):
        super().__init__()
        import torchvision.models as M
        b = getattr(M, backbone)(weights='IMAGENET1K_V1')
        in_dim = b.fc.in_features            # 512 (RN18) / 2048 (RN50/101/152)
        self.backbone = nn.Sequential(*list(b.children())[:-1])
        self.fc = nn.Linear(in_dim, feat_dim)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return F.normalize(self.fc(f), dim=-1)


class ViTEncoder(nn.Module):
    """Vision Transformer (torchvision vit_b_16, ImageNet-pretrained) -> normalized 512-dim.
    Feature = class token after the encoder (backbone forward minus the linear head)."""
    def __init__(self, feat_dim=512, backbone='vit_b_16'):
        super().__init__()
        import torchvision.models as M
        self.backbone = getattr(M, backbone)(weights='IMAGENET1K_V1')
        in_dim = self.backbone.heads.head.in_features   # 768 for ViT-B/16
        self.fc = nn.Linear(in_dim, feat_dim)

    def forward(self, x):
        x = self.backbone._process_input(x)
        n = x.shape[0]
        x = torch.cat([self.backbone.class_token.expand(n, -1, -1), x], dim=1)
        x = self.backbone.encoder(x)
        feat = x[:, 0]
        return F.normalize(self.fc(feat), dim=-1)


class KGEncoder(nn.Module):
    def __init__(self, kg_emb_dim=32, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(kg_emb_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class GatedAttention(nn.Module):
    def __init__(self, query_dim=512, key_dim=128, n_ent=151, gate_hidden=256):
        super().__init__()
        self.q = nn.Linear(query_dim, key_dim)
        self.gate = nn.Sequential(
            nn.Linear(query_dim, gate_hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(gate_hidden, n_ent), nn.Sigmoid(),
        )

    def forward(self, img_feat, kg_feats):
        a = F.softmax(
            torch.matmul(self.q(img_feat).unsqueeze(1), kg_feats.transpose(1, 2))
            / math.sqrt(kg_feats.size(-1)), dim=-1).squeeze(1)
        g = self.gate(img_feat)
        gw = a * g
        gw = gw / (gw.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(gw.unsqueeze(1), kg_feats).squeeze(1), gw


class ResNetBaseline(nn.Module):
    """Vision-only baseline."""
    def __init__(self, num_classes, backbone='resnet18'):
        super().__init__()
        self.enc = ViTEncoder(512, backbone) if backbone.startswith('vit') else ResNetEncoder(512, backbone)
        self.clf = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.clf(self.enc(x))


class GatedFusion(nn.Module):
    """Image + KG fusion with gated attention."""
    def __init__(self, num_classes, n_ent=151, kg_emb_dim=32):
        super().__init__()
        self.img_enc = ResNetEncoder(512)
        self.kg_enc = KGEncoder(kg_emb_dim=kg_emb_dim)
        self.fuser = GatedAttention(n_ent=n_ent)
        self.clf = nn.Sequential(
            nn.Linear(512 + 128, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, kg_emb):
        f = self.img_enc(img)
        k = self.kg_enc(kg_emb).unsqueeze(0).expand(img.size(0), -1, -1)
        c, w = self.fuser(f, k)
        return self.clf(torch.cat([f, c], dim=1)), w


class KGTNFusion(nn.Module):
    """KGTN-style prototype-based KG fusion.
    
    KG entity embeddings are projected through a prototype MLP to generate
    class weight vectors, analogous to KGTN's knowledge graph transfer network.
    Classification is done by cosine similarity between image features and
    KG-generated prototypes.
    """
    def __init__(self, num_classes, n_ent=151, kg_emb_dim=32, feat_dim=512):
        super().__init__()
        self.img_enc = ResNetEncoder(feat_dim)
        self.kg_proto = nn.Sequential(
            nn.Linear(kg_emb_dim, 128), nn.ReLU(),
            nn.Linear(128, feat_dim),
        )
        self.prototype_scale = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, img, kg_emb, class_kg_ids=None):
        f = self.img_enc(img)
        all_protos = self.kg_proto(kg_emb)
        if class_kg_ids is not None:
            protos = all_protos[class_kg_ids]
        else:
            protos = all_protos
        f_norm = F.normalize(f, dim=-1)
        p_norm = F.normalize(protos, dim=-1)
        logits = self.prototype_scale * torch.matmul(f_norm, p_norm.t())
        return logits


def train_one_epoch(model, loader, optimizer, criterion, device, is_kg_model=False, kg_emb=None):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for batch in loader:
        img = batch['image'].to(device)
        lbl = batch['label'].to(device)
        optimizer.zero_grad()
        if is_kg_model:
            logits, _ = model(img, kg_emb)
        else:
            logits = model(img)
        loss = criterion(logits, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * img.size(0)
        _, p = logits.max(1)
        correct += (p == lbl).sum().item()
        n += lbl.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_kg_model=False, kg_emb=None):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    all_preds, all_labels = [], []
    for batch in loader:
        img = batch['image'].to(device)
        lbl = batch['label'].to(device)
        if is_kg_model:
            logits, _ = model(img, kg_emb)
        else:
            logits = model(img)
        loss = criterion(logits, lbl)
        total_loss += loss.item() * img.size(0)
        _, p = logits.max(1)
        correct += (p == lbl).sum().item()
        n += lbl.size(0)
        all_preds.extend(p.cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())
    return total_loss / n, correct / n, all_preds, all_labels
