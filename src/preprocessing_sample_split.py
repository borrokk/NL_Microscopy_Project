# Sample-level preprocessing for NL_Microscopy_CNN
from pathlib import Path
import numpy as np
from PIL import Image

DATA_ROOT = Path("/home/megha/projects/NL_Microscopy_CNN/data/paper_dataset/02_Data")
IMAGE_DIR = DATA_ROOT / "Multimodal Images"
MASK_DIR = DATA_ROOT / "Reduced Masks" / "Three Class Masks"
NUM_CLASSES = 3
COLORS = {(31,119,180):0, (140,86,75):1, (158,218,229):2}
PATCH_SIZE, STRIDE = 224, 112
SKIP_BACKGROUND_ONLY = True

def discover():
    samples=[]
    for mp in sorted(MASK_DIR.glob("*__gt.png")):
        sid=mp.name.replace("__gt.png","")
        if all((IMAGE_DIR/f"{sid}_{m}.png").exists() for m in ("CARS","SHG","TPEF")):
            samples.append({"id":sid})
    print(f"Complete samples: {len(samples)}")
    return samples

def load_pair(sample):
    sid=sample["id"]; mp=MASK_DIR/f"{sid}__gt.png"
    paths=[IMAGE_DIR/f"{sid}_{m}.png" for m in ("CARS","SHG","TPEF")]
    for p in [mp,*paths]:
        if not p.exists(): raise FileNotFoundError(f"Missing {p}")
    mask_img=Image.open(mp).convert("RGB"); size=mask_img.size
    channels=[]
    for p in paths:
        im=Image.open(p).convert("L")
        if im.size!=size: im=im.resize(size,Image.Resampling.BILINEAR)
        channels.append(np.asarray(im,dtype=np.float32)/255.)
    x=np.stack(channels,axis=-1)
    rgb=np.asarray(mask_img,dtype=np.uint8)
    y=np.full(rgb.shape[:2],255,dtype=np.uint8)
    for color,k in COLORS.items(): y[np.all(rgb==np.asarray(color,dtype=np.uint8),axis=-1)]=k
    if np.any(y==255):
        raise ValueError(f"Unknown mask RGB colors in {sid}: {np.unique(rgb[y==255].reshape(-1,3),axis=0)[:20]}")
    return x.astype(np.float32),y.astype(np.int32)

def patches_for_sample(sample):
    x,y=load_pair(sample); h,w=y.shape
    ph,pw=max(0,PATCH_SIZE-h),max(0,PATCH_SIZE-w)
    if ph or pw:
        x=np.pad(x,((0,ph),(0,pw),(0,0)),mode="reflect")
        y=np.pad(y,((0,ph),(0,pw)),mode="constant",constant_values=0)
    h,w=y.shape
    rows=list(range(0,h-PATCH_SIZE+1,STRIDE)); cols=list(range(0,w-PATCH_SIZE+1,STRIDE))
    if not rows: rows=[0]
    if not cols: cols=[0]
    if rows[-1]!=h-PATCH_SIZE: rows.append(h-PATCH_SIZE)
    if cols[-1]!=w-PATCH_SIZE: cols.append(w-PATCH_SIZE)
    recs=[]
    for top in rows:
        for left in cols:
            xp=x[top:top+PATCH_SIZE,left:left+PATCH_SIZE,:]
            yp=y[top:top+PATCH_SIZE,left:left+PATCH_SIZE]
            if SKIP_BACKGROUND_ONLY and np.all(yp==0): continue
            recs.append({"id":sample["id"],"r":top,"c":left,"x":xp.astype(np.float32),"y":yp.astype(np.uint8)})
    return recs
