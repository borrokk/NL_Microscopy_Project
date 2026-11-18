from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from config import PATCH, STRIDE, COLORS


def open_grayscale(path: Path) -> np.ndarray:
    # Match the supplied training preprocessing: PIL convert("L"), uint8 / 255.
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        return np.asarray(im, dtype=np.float32) / 255.0


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)


def load_modalities(cars_path: Path, shg_path: Path, tpef_path: Path,
                    reference_size=None) -> np.ndarray:
    paths = [cars_path, shg_path, tpef_path]
    channels = [open_grayscale(p) for p in paths]
    if reference_size is None:
        h, w = channels[0].shape
    else:
        h, w = reference_size
    resized = []
    for channel in channels:
        if channel.shape != (h, w):
            # PIL bilinear resize, matching preprocessing_sample_split.py.
            im = Image.fromarray(np.clip(channel * 255, 0, 255).astype(np.uint8))
            im = im.resize((w, h), Image.Resampling.BILINEAR)
            channel = np.asarray(im, dtype=np.float32) / 255.0
        resized.append(channel)
    return np.stack(resized, axis=-1).astype(np.float32)


def decode_ground_truth(path: Path, expected_shape=None) -> np.ndarray:
    rgb = load_rgb(path)
    if expected_shape and rgb.shape[:2] != tuple(expected_shape):
        raise ValueError(
            f"Ground-truth mask dimensions {rgb.shape[:2]} do not match "
            f"input dimensions {tuple(expected_shape)}. Align/resize the mask explicitly."
        )
    y = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, class_id in COLORS.items():
        y[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = class_id
    if np.any(y == 255):
        unknown = np.unique(rgb[y == 255].reshape(-1, 3), axis=0)[:10].tolist()
        raise ValueError(f"Unknown ground-truth RGB colors (first 10): {unknown}")
    return y


def patch_starts(length: int):
    if length <= PATCH:
        return [0]
    starts = list(range(0, length - PATCH + 1, STRIDE))
    if starts[-1] != length - PATCH:
        starts.append(length - PATCH)
    return starts


def make_patches(image: np.ndarray):
    h, w, _ = image.shape
    pad_h, pad_w = max(0, PATCH - h), max(0, PATCH - w)
    if pad_h or pad_w:
        mode = "reflect" if h > 1 and w > 1 else "edge"
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    h2, w2, _ = image.shape
    coords = [(r, c) for r in patch_starts(h2) for c in patch_starts(w2)]
    patches = np.stack([image[r:r+PATCH, c:c+PATCH, :] for r, c in coords])
    return patches.astype(np.float32), coords, (h, w)
