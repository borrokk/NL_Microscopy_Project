"""
train_v2_torch.py - improved PyTorch pipeline for the 3-class task (background / resect / preserve).

Improvements over train_paper_chessboard_torch.py (all switchable below):
  * fine-tunes the last UNFREEZE_BLOCKS VGG16 blocks (low LR) instead of one
  * wider UNet3+ decoder, 448-px context (downscaled to 224), class-weighted CE + soft Dice
  * model selection on validation F1 (resect + preserve)
  * inference averages PROBABILITIES over shifted tilings and flip test-time augmentation (instead of hard votes)
  * ensemble of several independently trained models (SEEDS) - averaged probabilities

Run:  python train_v2_torch.py            (needs config.py + preprocessing_sample_split.py, torch, torchvision)
Time: roughly (len(SEEDS) x training time) + inference. Use SEEDS = [42] for a quick single-model run.
"""
import gc, json, random, time, csv
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

from config import PATCH, BATCH, NAMES, NCLASS, SEED
from preprocessing_sample_split import discover, load_pair

# ------------------------------------------------------------------ settings
CHESS, CROP = 512, PATCH
CONTEXT = 2                  # 448 px of tissue downscaled to 224 (1 = off)
T = CROP * CONTEXT
assert T <= CHESS
SEEDS = [42, 43, 44]         # one model per seed, averaged at test time
EPOCHS, LR, PATIENCE = 400, 1e-5, 20
BATCH_SIZE, ACCUM_STEPS = BATCH, 4          # 4 x 4 = effective 16
USE_AMP = True
N_VAL, CROPS_PER_PATCH = 10, 4   # 10 of the ~91 train patches for validation (~10%, like the paper's 50 of 498 tiles).
                                 # Was 50, which left only ~41 patches for training.
VAL_CROPS_PER_PATCH = 8          # fixed validation crops per held-out patch (80 crops)
N_LEVELS, DEC_CAT_CH = 4, 128
UNFREEZE_BLOCKS = 2          # last N VGG blocks are fine-tuned (0 = fully frozen like the paper)
LR_ENC = 2e-6
BALANCED_SAMPLING = True     # pick crops around pixels of a chosen class, so each class is seen equally often
CLASS_SAMPLE_P = [0.2, 0.4, 0.4]   # probability a crop is centred on background / resect / preserve
CLASS_WEIGHT_POWER = 0.25    # CE class weights ~ freq^-power (lowered because sampling already balances; 0 = off)
DICE_WEIGHT = 0.5            # soft Dice added to the loss (0 = off)
N_SHIFTS = 6                 # shifted tilings at inference
TTA = True                   # flip test-time augmentation (4 variants)
PREDICT_BATCH = 4

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = USE_AMP and dev.type == "cuda"
OUT = Path(__file__).resolve().parent / "results" / "v2_torch"
OUT.mkdir(parents=True, exist_ok=True)
torch.backends.cudnn.benchmark = True


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ data
def percentile_norm(x, lo=1.5, hi=99.0):
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[-1]):
        a, b = np.percentile(x[..., c], [lo, hi])
        out[..., c] = np.clip((x[..., c] - a) / (b - a + 1e-8), 0, 1)
    return out


def pad_multiple(a, m, value=0):
    ph, pw = (-a.shape[0]) % m, (-a.shape[1]) % m
    pads = ((0, ph), (0, pw)) + ((0, 0),) * (a.ndim - 2)
    return np.pad(a, pads, mode="constant", constant_values=value)


def load_all():
    imgs, masks, valid = [], [], []
    for s in discover():
        x, y = load_pair(s)                                  # CARS, SHG, TPEF in [0,1]; y in {0,1,2}
        imgs.append(pad_multiple(percentile_norm(x).astype(np.float16), CHESS))
        masks.append(pad_multiple(y.astype(np.uint8), CHESS))
        valid.append(pad_multiple(np.ones(y.shape, bool), CHESS, False))
    return imgs, masks, valid


def chessboard(imgs, masks, valid):
    tx, ty, test_maps = [], [], []
    for im, mk, vd in zip(imgs, masks, valid):
        H, W = mk.shape
        tmap = np.zeros((H, W), bool)
        for i in range(H // CHESS):
            for j in range(W // CHESS):
                sl = (slice(i * CHESS, (i + 1) * CHESS), slice(j * CHESS, (j + 1) * CHESS))
                if (i + j) % 2 == 0:
                    if vd[sl].any():
                        tx.append(im[sl]); ty.append(mk[sl])
                else:
                    tmap[sl] = True
        test_maps.append(tmap & vd)
    return np.stack(tx), np.stack(ty), test_maps


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, augment, crops_per_patch, seed, balanced=False):
        self.x, self.y, self.aug = x, y, augment
        self.n = len(x) * crops_per_patch
        self.rng = np.random.default_rng(seed)
        self.fixed = None if augment else [tuple(self.rng.integers(0, CHESS - T + 1, 2)) for _ in range(self.n)]
        self.coords = None
        if balanced:
            # per class: random sample of (patch, row, col) pixel positions; crops are later centred on them
            self.coords = []
            for c in range(NCLASS):
                parts = []
                for i in range(len(y)):
                    yy, xx = np.nonzero(y[i] == c)
                    if len(yy):
                        sel = self.rng.choice(len(yy), min(len(yy), 4000), replace=False)
                        parts.append(np.stack([np.full(len(sel), i), yy[sel], xx[sel]], 1))
                self.coords.append(np.concatenate(parts) if parts else np.zeros((0, 3), np.int64))
            p = np.array(CLASS_SAMPLE_P, np.float64)
            p[[len(a) == 0 for a in self.coords]] = 0
            self.cls_p = p / p.sum()
            print("balanced sampling: candidate pixels per class", [len(a) for a in self.coords],
                  "| crop-centre class probabilities", np.round(self.cls_p, 2))

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        if self.coords is not None:
            c = self.rng.choice(NCLASS, p=self.cls_p)
            i, cy, cx = self.coords[c][self.rng.integers(len(self.coords[c]))]
            j = T // 4                                                     # jitter: pixel is near, not always at, the centre
            oy = int(np.clip(cy - T // 2 + self.rng.integers(-j, j + 1), 0, CHESS - T))
            ox = int(np.clip(cx - T // 2 + self.rng.integers(-j, j + 1), 0, CHESS - T))
        else:
            i = k % len(self.x)
            oy, ox = self.fixed[k] if self.fixed else self.rng.integers(0, CHESS - T + 1, 2)
        xi = self.x[i][oy:oy + T, ox:ox + T].astype(np.float32)
        yi = self.y[i][oy:oy + T, ox:ox + T]
        if self.aug:
            if self.rng.random() < .5: xi, yi = xi[:, ::-1], yi[:, ::-1]
            if self.rng.random() < .5: xi, yi = xi[::-1], yi[::-1]
            r = int(self.rng.integers(0, 4)); xi, yi = np.rot90(xi, r), np.rot90(yi, r)
            xi = np.clip((xi - .5) * self.rng.uniform(.8, 1.2) + .5 + self.rng.uniform(-.1, .1), 0, 1)
        if CONTEXT > 1:
            xi = xi.reshape(CROP, CONTEXT, CROP, CONTEXT, 3).mean((1, 3))
            yi = yi[CONTEXT // 2::CONTEXT, CONTEXT // 2::CONTEXT]
        return (torch.from_numpy(np.ascontiguousarray(xi)).permute(2, 0, 1),
                torch.from_numpy(np.ascontiguousarray(yi).astype(np.int64)))


# ------------------------------------------------------------------ model
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet3Plus(nn.Module):
    def __init__(self, n_classes, levels=N_LEVELS, cat_ch=DEC_CAT_CH):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        cuts = [(0, 4), (5, 9), (10, 16), (17, 23), (24, 30)][:levels]      # conv blocks, pooling excluded
        self.enc = nn.ModuleList([nn.Sequential(*[vgg[i] for i in range(a, b)]) for a, b in cuts])
        self.L = levels
        self.first_train = levels - UNFREEZE_BLOCKS                          # blocks >= this index are fine-tuned
        for i, blk in enumerate(self.enc):
            for p in blk.parameters():
                p.requires_grad = i >= self.first_train
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
            inp = h if i == 0 else F.max_pool2d(h, 2)
            if i >= self.first_train:
                h = blk(inp)
            else:
                with torch.no_grad():                                        # frozen blocks: no activations stored
                    h = blk(inp)
            feats.append(h)
        L, d = self.L, {}
        for s in range(L - 2, -1, -1):
            parts = []
            for j in range(s + 1):
                f = feats[j] if j == s else F.max_pool2d(feats[j], 2 ** (s - j))
                parts.append(self.branch[f"{s}_e{j}"](f))
            for k in range(s + 1, L - 1):
                parts.append(self.branch[f"{s}_d{k}"](F.interpolate(d[k], scale_factor=2 ** (k - s), mode="bilinear", align_corners=False)))
            parts.append(self.branch[f"{s}_e{L - 1}"](F.interpolate(feats[L - 1], scale_factor=2 ** (L - 1 - s), mode="bilinear", align_corners=False)))
            d[s] = self.fuse[str(s)](torch.cat(parts, 1))
        return self.head(d[0])


# ------------------------------------------------------------------ loss + metrics
def make_loss(freq):
    wts = np.asarray(freq, dtype=np.float64) ** -CLASS_WEIGHT_POWER
    wts = wts / wts.mean()
    print("class pixel fractions", np.round(freq, 4), "-> CE class weights", np.round(wts, 2))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(wts, dtype=torch.float32, device=dev))

    def fn(logits, y):
        loss = ce(logits, y)
        if DICE_WEIGHT > 0:
            p = logits.softmax(1)
            oh = F.one_hot(y, NCLASS).permute(0, 3, 1, 2).float()
            inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
            loss = loss + DICE_WEIGHT * (1 - ((2 * inter + 1) / (den + 1)).mean())
        return loss
    return fn


def f1_from_cm(cm):
    tp = np.diag(cm).astype(np.float64); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


def metrics_from(y_true, y_pred):
    cm = np.bincount(y_true.astype(np.int64) * NCLASS + y_pred.astype(np.int64),
                     minlength=NCLASS ** 2).reshape(NCLASS, NCLASS)
    res = {}
    for c, name in enumerate(NAMES):
        TP = cm[c, c]; FP = cm[:, c].sum() - TP; FN = cm[c, :].sum() - TP; TN = cm.sum() - TP - FP - FN
        sens = TP / (TP + FN + 1e-9); spec = TN / (TN + FP + 1e-9); prec = TP / (TP + FP + 1e-9)
        res[name] = dict(specificity=spec, sensitivity=sens, precision=prec,
                         f1=2 * prec * sens / (prec + sens + 1e-9), iou=TP / (TP + FP + FN + 1e-9))
    res["Average"] = {k: float(np.mean([v[k] for v in list(res.values())])) for k in next(iter(res.values()))}
    res["pixel_accuracy"] = float(np.trace(cm) / cm.sum())
    return res, cm


def report(title, res, cm):
    print(f"\n=== {title} ===")
    print(f"{'Class':22s}{'Spec':>8s}{'Sens':>8s}{'Prec':>8s}{'F1':>8s}{'IoU':>8s}")
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:22s}{v['specificity']:8.2f}{v['sensitivity']:8.2f}{v['precision']:8.2f}{v['f1']:8.2f}{v['iou']:8.2f}")
    print(f"pixel accuracy: {res['pixel_accuracy']:.4f}")
    print("paper Table 2 average: spec 0.96 | sens 0.88 | prec 0.88 | F1 0.88")
    print("confusion matrix (rows = true):\n", cm)


# ------------------------------------------------------------------ training
def run_epoch(model, loader, opt, scaler, loss_fn, train):
    model.train(train)
    tot, n = 0.0, 0
    cm = np.zeros((NCLASS, NCLASS), np.int64)
    if train: opt.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
        if train:
            scaler.scale(loss / ACCUM_STEPS).backward()
            if (step + 1) % ACCUM_STEPS == 0 or step + 1 == len(loader):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        tot += loss.item() * len(x); n += len(x)
        pred = logits.argmax(1)
        cm += np.bincount((y * NCLASS + pred).cpu().numpy().ravel(), minlength=NCLASS ** 2).reshape(NCLASS, NCLASS)
    return tot / n, float(np.trace(cm) / cm.sum()), cm


def train_one(seed, tx, ty, ti, vi, loss_fn):
    set_seed(seed)
    tr_ld = torch.utils.data.DataLoader(CropDataset(tx[ti], ty[ti], True, CROPS_PER_PATCH, seed,
                                                    balanced=BALANCED_SAMPLING), batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=0, pin_memory=dev.type == "cuda")
    va_ld = torch.utils.data.DataLoader(CropDataset(tx[vi], ty[vi], False, VAL_CROPS_PER_PATCH, SEED), batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=0)
    model = UNet3Plus(NCLASS).to(dev)
    tot = sum(p.numel() for p in model.parameters()); trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[seed {seed}] params total {tot / 1e6:.1f}M | trainable {trn / 1e6:.1f}M")
    enc_p = [p for blk in model.enc[model.first_train:] for p in blk.parameters()]
    enc_ids = {id(p) for p in enc_p}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    opt = torch.optim.Adam([{"params": rest, "lr": LR}] + ([{"params": enc_p, "lr": LR_ENC}] if enc_p else []))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    path = OUT / f"best_seed{seed}.pt"
    best, bad, t0 = -1e9, 0, time.time()
    with open(OUT / f"training_log_seed{seed}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "loss", "val_loss", "val_f1_resect", "val_f1_preserve"])
        for ep in range(1, EPOCHS + 1):
            l, _, _ = run_epoch(model, tr_ld, opt, scaler, loss_fn, True)
            vl, va, vcm = run_epoch(model, va_ld, opt, scaler, loss_fn, False)
            f1s = f1_from_cm(vcm); score = float(f1s[1:].mean())
            w.writerow([ep, l, vl, f1s[1], f1s[2]]); fh.flush()
            print(f"[seed {seed}] epoch {ep:3d} loss {l:.4f} | val_loss {vl:.4f} val_acc {va:.4f} | "
                  f"val F1 resect {f1s[1]:.3f} preserve {f1s[2]:.3f}")
            if score > best:
                best, bad = score, 0; torch.save(model.state_dict(), path)
            else:
                bad += 1
                if bad >= PATIENCE:
                    print(f"[seed {seed}] early stopping"); break
    print(f"[seed {seed}] done in {(time.time() - t0) / 60:.1f} min, best val F1 (resect+preserve) {best:.3f}")
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return path


# ------------------------------------------------------------------ inference: shifted tilings + flip TTA, probability averaging
TTA_DIMS = [(), (3,), (2,), (2, 3)]


@torch.no_grad()
def predict_probs(model, img):
    model.eval()
    rng = np.random.default_rng(SEED)
    H, W, _ = img.shape
    acc = np.zeros((H, W, NCLASS), np.float32)
    variants = TTA_DIMS if TTA else [()]
    for s in range(N_SHIFTS):
        dy, dx = (0, 0) if s == 0 else rng.integers(0, T, 2)
        pb, pr = (-(H + dy)) % T, (-(W + dx)) % T
        pad = np.pad(img, ((dy, pb), (dx, pr), (0, 0)), mode="reflect")
        ny, nx = pad.shape[0] // T, pad.shape[1] // T
        tiles = pad.reshape(ny, T, nx, T, 3).transpose(0, 2, 1, 3, 4).reshape(-1, T, T, 3)
        probs = np.empty((len(tiles), NCLASS, T, T), np.float16)
        for k in range(0, len(tiles), PREDICT_BATCH):
            xb = torch.from_numpy(tiles[k:k + PREDICT_BATCH].astype(np.float32)).permute(0, 3, 1, 2).to(dev)
            if CONTEXT > 1: xb = F.avg_pool2d(xb, CONTEXT)
            psum = 0
            for dims in variants:
                xt = xb.flip(dims) if dims else xb
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                    lg = model(xt).float()
                p = lg.softmax(1)
                psum = psum + (p.flip(dims) if dims else p)
            p = psum / len(variants)
            if CONTEXT > 1: p = F.interpolate(p, scale_factor=CONTEXT, mode="bilinear", align_corners=False)
            probs[k:k + PREDICT_BATCH] = p.cpu().numpy().astype(np.float16)
        full = probs.reshape(ny, nx, NCLASS, T, T).transpose(0, 3, 1, 4, 2).reshape(ny * T, nx * T, NCLASS)
        acc += full[dy:dy + H, dx:dx + W].astype(np.float32)
        del pad, tiles, probs, full
    gc.collect()
    return acc / N_SHIFTS


def evaluate_ensemble(paths, imgs, masks, test_maps):
    yt = np.concatenate([mk[tm] for mk, tm in zip(masks, test_maps)])
    ens = [np.zeros((int(tm.sum()), NCLASS), np.float32) for tm in test_maps]
    model = UNet3Plus(NCLASS).to(dev)
    for pth in paths:
        model.load_state_dict(torch.load(pth, map_location=dev))
        single = []
        for i, (im, tm) in enumerate(zip(imgs, test_maps)):
            p = predict_probs(model, im)[tm]
            ens[i] += p; single.append(p.argmax(1).astype(np.uint8))
        r, _ = metrics_from(yt, np.concatenate(single))
        print(f"  {pth.name}: avg F1 {r['Average']['f1']:.3f} | resect F1 {r[NAMES[1]]['f1']:.3f} | preserve F1 {r[NAMES[2]]['f1']:.3f}")
        del single; gc.collect()
    return metrics_from(yt, np.concatenate([e.argmax(1).astype(np.uint8) for e in ens]))


def main():
    print(f"device {dev} | batch {BATCH_SIZE} x accum {ACCUM_STEPS} | AMP {AMP}")
    print(f"SETTINGS: CONTEXT={CONTEXT} SEEDS={SEEDS} UNFREEZE_BLOCKS={UNFREEZE_BLOCKS} DEC_CAT_CH={DEC_CAT_CH} "
          f"BALANCED_SAMPLING={BALANCED_SAMPLING} CLASS_WEIGHT_POWER={CLASS_WEIGHT_POWER} DICE_WEIGHT={DICE_WEIGHT} N_SHIFTS={N_SHIFTS} TTA={TTA}")
    set_seed(SEED)
    imgs, masks, valid = load_all()
    tx, ty, test_maps = chessboard(imgs, masks, valid)
    print(f"images: {len(imgs)} | train patches (512): {len(tx)} | test pixels: {sum(int(t.sum()) for t in test_maps):,}")
    perm = np.random.default_rng(SEED).permutation(len(tx))
    vi, ti = perm[:N_VAL], perm[N_VAL:]
    print(f"patches used for training: {len(ti)} | for validation: {len(vi)}")
    loss_fn = make_loss(np.bincount(ty[ti].ravel(), minlength=NCLASS) / ty[ti].size)

    paths = [train_one(s, tx, ty, ti, vi, loss_fn) for s in SEEDS]
    print("\nevaluating on held-out chessboard patches (per model, then ensemble):")
    res, cm = evaluate_ensemble(paths, imgs, masks, test_maps)
    report(f"TEST, ensemble of {len(paths)} model(s)", res, cm)
    json.dump({**res, "confusion_matrix": cm.tolist()}, open(OUT / "test_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
