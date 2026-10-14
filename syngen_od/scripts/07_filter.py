#!/usr/bin/env python3
"""07 Quality-filter a generated task using teacher detections + model-free structure checks.

Output in the task dir:
  kept.txt            images that passed (used by 12_build_experiments)
  filter_report.csv   one row per image with scores and reject reasons (use it to tune the thresholds)
  filter_summary.json pass rates overall / per reason / per prompt tag (style, confuser, ...)
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.boxes import iou_matrix, labels_to_pixel, xyxy_to_xywhn  # noqa: E402
from syngen.common import base_parser, load  # noqa: E402
from syngen.imgutil import is_degenerate, structure_similarity  # noqa: E402
from syngen.records import read_dets, read_meta, write_json  # noqa: E402
from syngen.yolo_io import img2label_path, list_images, read_labels, write_labels, write_list  # noqa: E402


def mask_boxes(img, boxes):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for b in boxes:
        d.rectangle([float(x) for x in b], fill=(128, 128, 128))
    return img


def verify_boxes(src, edit, gt_px, dets, f):
    """Each GT box must survive the edit: teacher re-detects it, or its local structure is preserved."""
    verifiable, lost, scores = 0, 0, []
    ious = iou_matrix(gt_px[:, 1:5], dets[:, 1:5]) if len(dets) else np.zeros((len(gt_px), 0))
    for i, g in enumerate(gt_px):
        if min(g[3] - g[1], g[4] - g[2]) < f.get("min_box_px", 16):
            continue
        verifiable += 1
        same = (dets[:, 0] == g[0]) & (dets[:, 5] >= f.get("keep_teacher_conf", 0.1)) if len(dets) else []
        if len(dets) and (ious[i][same] >= 0.5).any():
            continue
        s = structure_similarity(src, edit, box=g[1:5])
        scores.append(s)
        if s is None or s < f.get("box_struct_min", 0.4):
            lost += 1
    return verifiable, lost, (min(scores) if scores else None)


def unmatched_dets(gt_px, dets, conf, iou=0.3):
    d = dets[dets[:, 5] >= conf]
    if len(d) == 0:
        return d
    if len(gt_px) == 0:
        return d
    m = iou_matrix(d[:, 1:5], gt_px[:, 1:5])
    same = d[:, 0][:, None] == gt_px[:, 0][None, :]
    return d[~((m >= iou) & same).any(axis=1)]


def check(task_name, f, img_path, img, gt, dets, meta, classes):
    """-> (keep, reasons, info, new_labels_or_None)"""
    w, h = img.size
    gt_px = labels_to_pixel(gt, w, h)
    reasons, info = [], {}
    if is_degenerate(img):
        return False, ["degenerate"], info, None

    if task_name == "neg_t2i":
        bad = dets[dets[:, 5] >= f["phantom_conf"]]
        info["max_det_conf"] = float(dets[:, 5].max()) if len(dets) else 0.0
        if len(bad):
            reasons.append("target_detected:" + ",".join(sorted({classes[int(c)] for c in bad[:, 0]})))
        return not reasons, reasons, info, None

    src = Image.open(meta["source"]).convert("RGB") if meta and meta.get("source") else None

    if task_name == "remove":
        rm = meta["removed_classes"]
        resid = dets[np.isin(dets[:, 0], rm) & (dets[:, 5] >= f["phantom_conf"])] if len(dets) else dets
        info["max_removed_conf"] = float(dets[np.isin(dets[:, 0], rm), 5].max()) if len(dets) and np.isin(
            dets[:, 0], rm).any() else 0.0
        if len(resid):
            reasons.append("removed_class_still_detected")
        g = structure_similarity(mask_boxes(src, meta["removed_boxes"]), mask_boxes(img, meta["removed_boxes"]))
        info["global_struct"] = g
        if g < f["global_struct_min"]:
            reasons.append("scene_changed")
        nv, nl, ms = verify_boxes(src, img, gt_px, dets, f)
        info.update(verifiable=nv, lost=nl, min_box_struct=ms)
        if nl > 0:
            reasons.append("kept_object_lost")
        return not reasons, reasons, info, None

    if task_name == "style":
        g = structure_similarity(src, img)
        info["global_struct"] = g
        if g < f["global_struct_min"]:
            reasons.append("scene_changed")
        nv, nl, ms = verify_boxes(src, img, gt_px, dets, f)
        info.update(verifiable=nv, lost=nl, min_box_struct=ms)
        if nv and nl / nv > f.get("max_lost_frac", 0.0):
            reasons.append("object_lost_or_moved")
        ph = unmatched_dets(gt_px, dets, f.get("phantom_conf", 0.6))
        info["phantoms"] = int(len(ph))
        if len(ph):
            reasons.append("hallucinated_object")
        return not reasons, reasons, info, None

    if task_name == "insert":
        n_exist = meta["num_existing"]
        new_rows = []
        for d in meta["inserted"]:
            cand = dets[(dets[:, 0] == d["cls"]) & (dets[:, 5] >= f["match_conf"])] if len(dets) else dets
            if len(cand) == 0:
                reasons.append("insert_not_detected")
                break
            ious = iou_matrix(np.array(d["box"])[None], cand[:, 1:5])[0]
            j = int(ious.argmax())
            if ious[j] < f["match_iou"]:
                reasons.append("insert_not_detected")
                break
            new_rows.append([d["cls"], *xyxy_to_xywhn(cand[j, 1:5], w, h)[0]])  # tight teacher box
        if reasons:
            return False, reasons, info, None
        return True, [], info, np.vstack([gt[:n_exist].reshape(-1, 5), np.array(new_rows)])

    raise ValueError(task_name)


def main():
    p = base_parser(__doc__)
    p.add_argument("--task", required=True, choices=["neg_t2i", "remove", "style", "insert"])
    args = p.parse_args()
    cfg = load(args)
    task = cfg["tasks"][args.task]
    f = task["filter"]
    out = Path(task["out"])
    meta = read_meta(out)
    dets_all = read_dets(out)
    imgs = list_images(out / "images")
    kept, rows = [], []
    reason_ct = Counter()
    tag_ct = defaultdict(lambda: [0, 0])  # tag -> [kept, total]
    tag_key = {"neg_t2i": "confuser", "style": "style"}.get(args.task)
    missing = 0
    for ip in imgs:
        if ip not in dets_all:
            missing += 1
            continue
        _, _, dets = dets_all[ip]
        m = meta.get(ip)
        if args.task != "neg_t2i" and m is None:
            reason_ct["no_meta"] += 1
            continue
        img = Image.open(ip).convert("RGB")
        gt = read_labels(img2label_path(ip))
        keep, reasons, info, new_labels = check(args.task, f, ip, img, gt, dets, m, cfg["classes"])
        if keep and new_labels is not None:
            write_labels(img2label_path(ip), new_labels)
        if keep:
            kept.append(ip)
        for r in reasons:
            reason_ct[r.split(":")[0]] += 1
        if tag_key and m:
            tag_ct[m.get(tag_key)][0] += keep
            tag_ct[m.get(tag_key)][1] += 1
        rows.append({"image": ip, "keep": int(keep), "reasons": ";".join(reasons),
                     **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()}})
    if missing:
        print(f"WARNING: {missing} images have no teacher detections yet -> run 06_teacher_detect first")
    write_list(out / "kept.txt", kept)
    keys = sorted({k for r in rows for k in r})
    with open(out / "filter_report.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "keep", "reasons"] + [k for k in keys if k not in
                                                                          ("image", "keep", "reasons")])
        wr.writeheader()
        wr.writerows(rows)
    summary = {"task": args.task, "checked": len(rows), "kept": len(kept),
               "pass_rate": len(kept) / max(len(rows), 1), "reject_reasons": dict(reason_ct),
               "pass_rate_by_" + (tag_key or "tag"): {str(k): {"kept": v[0], "total": v[1],
                                                               "rate": round(v[0] / v[1], 3)}
                                                      for k, v in sorted(tag_ct.items(), key=lambda x: str(x[0]))}}
    write_json(out / "filter_summary.json", summary)
    print(f"[{args.task}] kept {len(kept)}/{len(rows)} ({summary['pass_rate']:.1%}); reasons: {dict(reason_ct)}")


if __name__ == "__main__":
    main()
