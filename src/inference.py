"""Inference with the PyTorch models from train_v2_torch.py.

Same steps as predict_probs / evaluate_ensemble in training:
  per-channel percentile normalisation (1.5-99 %) -> 448 px tiles shrunk 2x to 224 ->
  softmax averaged over flips (TTA), shifted tilings and every checkpoint (ensemble).
predict() keeps the old interface: (labels H x W uint8, probabilities H x W x NCLASS).
"""
import gc

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (NCLASS, CLASS_RGB, CONTEXT, TILE, N_SHIFTS, TTA, PREDICT_BATCH, SHIFT_SEED, DEVICE,
                    MODEL_WEIGHTS, model_weight_paths)
from model_unet3plus import build

TTA_DIMS = [(), (3,), (2,), (2, 3)]


def percentile_norm(x, lo=1.5, hi=99.0):
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[-1]):
        a, b = np.percentile(x[..., c], [lo, hi])
        out[..., c] = np.clip((x[..., c] - a) / (b - a + 1e-8), 0, 1)
    return out


def _load_state(path):
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13
        state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = {}
    for k, v in state.items():
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        clean[k] = v
    return clean


class ModelService:
    def __init__(self):
        self.models = None
        if DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(DEVICE)
        self.amp = self.device.type == "cuda"

    def load(self):
        paths = model_weight_paths()
        missing = [p for p in paths if not p.is_file()]
        if not paths or missing:
            raise FileNotFoundError(
                f"Model checkpoint not found: {missing[0] if missing else MODEL_WEIGHTS}. "
                "Copy best_seed42.pt (and best_seed43.pt, best_seed44.pt for the ensemble) from "
                "results/v2_torch into the model/ folder, or set MODEL_WEIGHTS."
            )
        models = []
        for p in paths:
            model = build(NCLASS)
            try:
                model.load_state_dict(_load_state(p), strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{p.name} does not match the UNet3+ in model_unet3plus.py. Check N_LEVELS / DEC_CAT_CH "
                    f"in config.py against train_v2_torch.py. Details: {str(exc)[:600]}"
                ) from exc
            models.append(model.to(self.device).eval())
        self.models = models
        print(f"Loaded {len(models)} model(s) on {self.device}: {', '.join(p.name for p in paths)}")
        return models

    @torch.no_grad()
    def _predict_probs(self, model, img):
        """train_v2_torch.predict_probs: shifted tilings + flip TTA, softmax averaged."""
        rng = np.random.default_rng(SHIFT_SEED)
        H, W, _ = img.shape
        acc = np.zeros((H, W, NCLASS), np.float32)
        variants = TTA_DIMS if TTA else [()]
        for s in range(N_SHIFTS):
            dy, dx = (0, 0) if s == 0 else rng.integers(0, TILE, 2)
            pb, pr = (-(H + dy)) % TILE, (-(W + dx)) % TILE
            try:
                pad = np.pad(img, ((dy, pb), (dx, pr), (0, 0)), mode="reflect")
            except ValueError:  # 1-pixel-wide images cannot be reflected
                pad = np.pad(img, ((dy, pb), (dx, pr), (0, 0)), mode="edge")
            ny, nx = pad.shape[0] // TILE, pad.shape[1] // TILE
            tiles = pad.reshape(ny, TILE, nx, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(-1, TILE, TILE, 3)
            probs = np.empty((len(tiles), NCLASS, TILE, TILE), np.float16)
            for k in range(0, len(tiles), PREDICT_BATCH):
                xb = torch.from_numpy(tiles[k:k + PREDICT_BATCH].astype(np.float32)).permute(0, 3, 1, 2).to(self.device)
                if CONTEXT > 1:
                    xb = F.avg_pool2d(xb, CONTEXT)
                psum = 0
                for dims in variants:
                    xt = xb.flip(dims) if dims else xb
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                        lg = model(xt).float()
                    p = lg.softmax(1)
                    psum = psum + (p.flip(dims) if dims else p)
                p = psum / len(variants)
                if CONTEXT > 1:
                    p = F.interpolate(p, scale_factor=CONTEXT, mode="bilinear", align_corners=False)
                probs[k:k + PREDICT_BATCH] = p.cpu().numpy().astype(np.float16)
            full = probs.reshape(ny, nx, NCLASS, TILE, TILE).transpose(0, 3, 1, 4, 2).reshape(ny * TILE, nx * TILE, NCLASS)
            acc += full[dy:dy + H, dx:dx + W].astype(np.float32)
            del pad, tiles, probs, full
        gc.collect()
        return acc / N_SHIFTS

    def predict(self, image: np.ndarray):
        """image: H x W x 3 in [0, 1], channels CARS, SHG, TPEF (preprocessing.load_modalities)."""
        if self.models is None:
            self.load()
        x = percentile_norm(image)
        probabilities = np.zeros(image.shape[:2] + (NCLASS,), np.float32)
        for model in self.models:
            probabilities += self._predict_probs(model, x)
        probabilities /= len(self.models)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        labels = np.argmax(probabilities, axis=-1).astype(np.uint8)
        return labels, probabilities


def colorize_mask(labels: np.ndarray) -> Image.Image:
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_RGB.items():
        rgb[labels == class_id] = color
    return Image.fromarray(rgb, mode="RGB")


def make_overlay(input_rgb: Image.Image, mask_rgb: Image.Image, alpha=0.45):
    if input_rgb.size != mask_rgb.size:
        mask_rgb = mask_rgb.resize(input_rgb.size, Image.Resampling.NEAREST)
    return Image.blend(input_rgb.convert("RGB"), mask_rgb.convert("RGB"), alpha)
