"""UNet3+ with a VGG16 encoder - the network from train_v2_torch.py, for inference.

Identical layer structure to training (so the saved state_dicts load with strict=True), but the
VGG16 is built without downloading ImageNet weights: the checkpoint contains all weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from config import N_LEVELS, DEC_CAT_CH


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def _vgg16_features():
    try:
        return torchvision.models.vgg16(weights=None).features
    except TypeError:  # torchvision < 0.13
        return torchvision.models.vgg16(pretrained=False).features


class UNet3Plus(nn.Module):
    def __init__(self, n_classes, levels=N_LEVELS, cat_ch=DEC_CAT_CH):
        super().__init__()
        vgg = _vgg16_features()
        cuts = [(0, 4), (5, 9), (10, 16), (17, 23), (24, 30)][:levels]      # conv blocks, pooling excluded
        self.enc = nn.ModuleList([nn.Sequential(*[vgg[i] for i in range(a, b)]) for a, b in cuts])
        self.L = levels
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        ch = [64, 128, 256, 512, 512][:levels]
        self.branch, self.fuse = nn.ModuleDict(), nn.ModuleDict()
        for s in range(levels - 2, -1, -1):
            for j in range(s + 1):
                self.branch[f"{s}_e{j}"] = cbr(ch[j], cat_ch)
            for k in range(s + 1, levels - 1):
                self.branch[f"{s}_d{k}"] = cbr(cat_ch * levels, cat_ch)
            self.branch[f"{s}_e{levels - 1}"] = cbr(ch[levels - 1], cat_ch)
            self.fuse[str(s)] = cbr(cat_ch * levels, cat_ch * levels)
        self.head = nn.Conv2d(cat_ch * levels, n_classes, 1)

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats, h = [], x
        for i, blk in enumerate(self.enc):
            h = blk(h if i == 0 else F.max_pool2d(h, 2))
            feats.append(h)
        L, d = self.L, {}
        for s in range(L - 2, -1, -1):
            parts = []
            for j in range(s + 1):
                f = feats[j] if j == s else F.max_pool2d(feats[j], 2 ** (s - j))
                parts.append(self.branch[f"{s}_e{j}"](f))
            for k in range(s + 1, L - 1):
                parts.append(self.branch[f"{s}_d{k}"](
                    F.interpolate(d[k], scale_factor=2 ** (k - s), mode="bilinear", align_corners=False)))
            parts.append(self.branch[f"{s}_e{L - 1}"](
                F.interpolate(feats[L - 1], scale_factor=2 ** (L - 1 - s), mode="bilinear", align_corners=False)))
            d[s] = self.fuse[str(s)](torch.cat(parts, 1))
        return self.head(d[0])   # logits; softmax is applied in inference.py


def build(n_classes):
    return UNet3Plus(n_classes)
