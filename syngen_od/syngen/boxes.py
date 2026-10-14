import numpy as np


def xywhn_to_xyxy(xywhn, w, h):
    a = np.asarray(xywhn, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(a)
    out[:, 0] = (a[:, 0] - a[:, 2] / 2) * w
    out[:, 1] = (a[:, 1] - a[:, 3] / 2) * h
    out[:, 2] = (a[:, 0] + a[:, 2] / 2) * w
    out[:, 3] = (a[:, 1] + a[:, 3] / 2) * h
    return out


def xyxy_to_xywhn(xyxy, w, h):
    a = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    x1 = np.clip(a[:, 0], 0, w)
    y1 = np.clip(a[:, 1], 0, h)
    x2 = np.clip(a[:, 2], 0, w)
    y2 = np.clip(a[:, 3], 0, h)
    return np.stack([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h], axis=1)


def labels_to_pixel(labels, w, h):
    """YOLO labels (N,5|6) -> (N,5|6) [cls, x1,y1,x2,y2, (conf)] in pixels."""
    labels = np.asarray(labels, dtype=np.float64)
    if len(labels) == 0:
        return np.zeros((0, labels.shape[1] if labels.ndim == 2 else 5))
    out = labels.copy()
    out[:, 1:5] = xywhn_to_xyxy(labels[:, 1:5], w, h)
    return out


def area(xyxy):
    a = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    return np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)


def iou_matrix(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def greedy_match(gt_xyxy, det_xyxy, det_conf, iou_thr):
    """COCO-style matching: dets in descending confidence each take the highest-IoU unmatched GT with IoU >= thr.
    Returns (det_matched_gt_idx (-1 = FP), gt_matched bool)."""
    order = np.argsort(-np.asarray(det_conf), kind="mergesort")
    ious = iou_matrix(det_xyxy, gt_xyxy)
    gt_used = np.zeros(len(gt_xyxy), dtype=bool)
    det_gt = np.full(len(det_xyxy), -1, dtype=int)
    for d in order:
        best, best_iou = -1, min(iou_thr, 1 - 1e-10)
        for g in range(len(gt_xyxy)):
            if gt_used[g] or ious[d, g] < best_iou:
                continue
            best, best_iou = g, ious[d, g]
        if best >= 0:
            gt_used[best] = True
            det_gt[d] = best
    return det_gt, gt_used
