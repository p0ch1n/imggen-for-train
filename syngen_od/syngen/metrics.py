"""Detection metrics: COCO-style AP plus operating-point stats (at a fixed confidence threshold).

gt[img]   = (N,5) [cls, x1, y1, x2, y2] pixels
pred[img] = (M,6) [cls, x1, y1, x2, y2, conf] pixels
"""
import numpy as np

from .boxes import area, greedy_match

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
SIZE_BUCKETS = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, float("inf"))}


def _ap_from_matches(scores, tps, npos):
    """scores (D,), tps (T,D) bool -> AP per IoU threshold (T,), COCO 101-point interpolation."""
    if npos == 0:
        return np.full(tps.shape[0], np.nan)
    if len(scores) == 0:
        return np.zeros(tps.shape[0])
    order = np.argsort(-scores, kind="mergesort")
    tps = tps[:, order]
    tp_cum = np.cumsum(tps, axis=1).astype(np.float64)
    fp_cum = np.cumsum(~tps, axis=1).astype(np.float64)
    aps = []
    for t in range(tps.shape[0]):
        rc = tp_cum[t] / npos
        pr = tp_cum[t] / np.maximum(tp_cum[t] + fp_cum[t], np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        inds = np.searchsorted(rc, REC_THRS, side="left")
        q = np.where(inds < len(pr), pr[np.minimum(inds, len(pr) - 1)], 0.0)
        aps.append(q.mean())
    return np.asarray(aps)


def average_precision(gt, pred, num_classes, max_dets=100):
    """Returns dict with per-class AP50, AP50-95 and their means over classes that have GT."""
    images = sorted(set(gt) | set(pred))
    per_class = {}
    for c in range(num_classes):
        scores, tps, npos = [], [], 0
        for img in images:
            g = gt.get(img, np.zeros((0, 5)))
            p = pred.get(img, np.zeros((0, 6)))
            g = g[g[:, 0] == c]
            p = p[p[:, 0] == c]
            npos += len(g)
            if len(p) == 0:
                continue
            p = p[np.argsort(-p[:, 5], kind="mergesort")][:max_dets]
            flags = np.zeros((len(IOU_THRS), len(p)), dtype=bool)
            for t, thr in enumerate(IOU_THRS):
                det_gt, _ = greedy_match(g[:, 1:5], p[:, 1:5], p[:, 5], thr)
                flags[t] = det_gt >= 0
            scores.append(p[:, 5])
            tps.append(flags)
        s = np.concatenate(scores) if scores else np.zeros(0)
        t = np.concatenate(tps, axis=1) if tps else np.zeros((len(IOU_THRS), 0), dtype=bool)
        ap = _ap_from_matches(s, t, npos)
        per_class[c] = {"num_gt": int(npos),
                        "ap50": None if np.isnan(ap[0]) else float(ap[0]),
                        "ap50_95": None if np.isnan(ap[0]) else float(np.mean(ap))}
    valid = [v for v in per_class.values() if v["ap50"] is not None]
    return {
        "map50": float(np.mean([v["ap50"] for v in valid])) if valid else None,
        "map50_95": float(np.mean([v["ap50_95"] for v in valid])) if valid else None,
        "per_class": per_class,
    }


def operating_point(gt, pred, num_classes, conf_thr, iou_thr=0.5):
    """Per-image and aggregated TP/FP/FN at a confidence threshold, plus recall by GT size bucket.
    Per-image counts are kept so that compare.py can run a paired bootstrap."""
    images = sorted(set(gt) | set(pred))
    per_image = {}
    size_hits = {c: {b: [0, 0] for b in SIZE_BUCKETS} for c in range(num_classes)}  # [tp, total]
    for img in images:
        g = gt.get(img, np.zeros((0, 5)))
        p = pred.get(img, np.zeros((0, 6)))
        p = p[p[:, 5] >= conf_thr]
        row = {}
        for c in range(num_classes):
            gc, pc = g[g[:, 0] == c], p[p[:, 0] == c]
            det_gt, gt_hit = greedy_match(gc[:, 1:5], pc[:, 1:5], pc[:, 5], iou_thr)
            tp = int((det_gt >= 0).sum())
            row[c] = [tp, int(len(pc) - tp), int(len(gc) - tp)]  # tp, fp, fn
            areas = area(gc[:, 1:5])
            for b, (lo, hi) in SIZE_BUCKETS.items():
                m = (areas >= lo) & (areas < hi)
                size_hits[c][b][0] += int(gt_hit[m].sum())
                size_hits[c][b][1] += int(m.sum())
        per_image[img] = row
    return summarize_operating_point(per_image, num_classes) | {
        "recall_by_size": {c: {b: (v[0] / v[1] if v[1] else None) for b, v in d.items()}
                           for c, d in size_hits.items()},
        "gt_count_by_size": {c: {b: v[1] for b, v in d.items()} for c, d in size_hits.items()},
        "per_image": per_image,
    }


def summarize_operating_point(per_image, num_classes):
    n_img = max(len(per_image), 1)
    out = {"num_images": len(per_image), "per_class": {}}
    for c in range(num_classes):
        tp = sum(r[c][0] for r in per_image.values())
        fp = sum(r[c][1] for r in per_image.values())
        fn = sum(r[c][2] for r in per_image.values())
        imgs_with_fp = sum(1 for r in per_image.values() if r[c][1] > 0)
        out["per_class"][c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "fp_per_image": fp / n_img,
            "frac_images_with_fp": imgs_with_fp / n_img,
        }
    tot_fp = sum(v["fp"] for v in out["per_class"].values())
    out["fp_per_image_all"] = tot_fp / n_img
    out["frac_images_with_fp_all"] = sum(
        1 for r in per_image.values() if any(r[c][1] > 0 for c in range(num_classes))) / n_img
    return out
