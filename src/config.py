# Memory-aware configuration for the GPU training script.
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data/paper_dataset/02_Data/Multimodal Images"
MASK_DIR = ROOT / "data/paper_dataset/02_Data/Reduced Masks/Three Class Masks"
OUT = ROOT / "results/paper_unet3plus_memory_safe"

# 224 divisible by 16; lower to 160 or 128 if still OOM.
PATCH = 224
STRIDE = 112
BATCH = 4

EPOCHS = 400
LR = 1e-5
VAL_PATCHES = 50
PATIENCE = 20
SEED = 42
NCLASS = 3
NAMES = ["Background", "Tissue to resect", "Tissue to preserve"]

# These must exactly match the RGB values observed in the mask PNGs.
COLORS = {
    (31, 119, 180): 0,      # Background
    (140, 86, 75): 1,       # Tissue to resect
    (158, 218, 229): 2,     # Tissue to preserve
}
