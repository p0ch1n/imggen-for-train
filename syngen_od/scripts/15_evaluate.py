#!/usr/bin/env python3
"""15 Score predictions from 14 against ground truth -> <results>/<model>/<set>/metrics.json

Metrics: COCO-style mAP50 / mAP50-95 (sets with GT), and at the production threshold eval.conf_thr:
per-class precision / recall / FP-per-image, recall by object size (small <32², medium <96², large), and per-image
TP/FP/FN (used by 16_compare for paired bootstrap confidence intervals).
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.boxes import labels_to_pixel  # noqa: E402
from syngen.common import base_parser, load  # noqa: E402
from syngen.metrics import average_precision, operating_point  # noqa: E402
from syngen.records import write_json  # noqa: E402
from syngen.yolo_io import image_size, img2label_path, list_images, read_labels  # noqa: E402


def load_set(sdir, pred_dir):
    imgs = list_images(sdir)
    dup = [k for k, v in Counter(Path(x).stem for x in imgs).items() if v > 1]
    if dup:
        sys.exit(f"{sdir}: duplicate file stems {dup[:5]} (predictions are keyed by stem); rename them")
    gt, pred = {}, {}
    for ip in imgs:
        w, h = image_size(ip)
        k = Path(ip).stem
        try:
            lab = read_labels(img2label_path(ip))
        except ValueError:  # e.g. deploy_bg given as a dir without the images/labels layout
            lab = read_labels("")
        gt[k] = labels_to_pixel(lab, w, h)
        pred[k] = labels_to_pixel(read_labels(Path(pred_dir) / f"{k}.txt", with_conf=True), w, h)
    return gt, pred


def main():
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="*")
    args = p.parse_args()
    cfg = load(args)
    ev = cfg["eval"]
    nc = len(cfg["classes"])
    res_root = Path(cfg["paths"]["results"])
    for mdir in sorted(x for x in res_root.iterdir() if x.is_dir()):
        if args.models and mdir.name not in args.models:
            continue
        for sname, sdir in ev["sets"].items():
            pred_dir = mdir / sname / "pred"
            if not pred_dir.exists():
                continue
            gt, pred = load_set(sdir, pred_dir)
            has_gt = any(len(v) for v in gt.values())
            m = {"model": mdir.name, "set": sname, "num_images": len(gt), "classes": cfg["classes"],
                 "conf_thr": ev["conf_thr"]}
            if has_gt:
                m["ap"] = average_precision(gt, pred, nc)
            m["op"] = operating_point(gt, pred, nc, ev["conf_thr"], ev.get("iou_thr", 0.5))
            write_json(mdir / sname / "metrics.json", m)
            ap = f"mAP50-95={m['ap']['map50_95']:.4f} mAP50={m['ap']['map50']:.4f} " if has_gt else ""
            print(f"[{mdir.name}/{sname}] {ap}FP/img@{ev['conf_thr']}={m['op']['fp_per_image_all']:.4f}")


if __name__ == "__main__":
    main()
