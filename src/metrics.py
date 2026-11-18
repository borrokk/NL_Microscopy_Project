import numpy as np
from config import NAMES


def evaluate_mask(y_true: np.ndarray, y_pred: np.ndarray):
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Mask shape mismatch: {y_true.shape} vs {y_pred.shape}")
    n = len(NAMES)
    cm = np.zeros((n, n), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < n) & (y_pred >= 0) & (y_pred < n)
    np.add.at(cm, (y_true[valid].astype(int), y_pred[valid].astype(int)), 1)
    total = int(cm.sum())
    if total == 0:
        raise ValueError("No valid pixels available for metric evaluation")
    classes = []
    for i, name in enumerate(NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        div = lambda a, b: float(a / b) if b else None
        precision = div(tp, tp + fp)
        sensitivity = div(tp, tp + fn)
        specificity = div(tn, tn + fp)
        f1 = div(2 * tp, 2 * tp + fp + fn)
        iou = div(tp, tp + fp + fn)
        classes.append({
            "class_id": i, "class_name": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "sensitivity": sensitivity, "specificity": specificity,
            "f1": f1, "dice": f1, "iou": iou,
        })
    def mean_metric(key):
        vals = [x[key] for x in classes if x[key] is not None]
        return float(np.mean(vals)) if vals else None
    supports = cm.sum(axis=1)
    weighted = {}
    for key in ("precision", "sensitivity", "specificity", "f1", "dice", "iou"):
        pairs = [(x[key], int(supports[x["class_id"]])) for x in classes if x[key] is not None]
        denom = sum(w for _, w in pairs)
        weighted[key] = float(sum(v*w for v, w in pairs) / denom) if denom else None
    return {
        "pixel_accuracy": float(np.trace(cm) / total),
        "macro": {k: mean_metric(k) for k in ("precision", "sensitivity", "specificity", "f1", "dice", "iou")},
        "weighted": weighted,
        "per_class": classes,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": NAMES,
        "evaluated_pixels": total,
    }
