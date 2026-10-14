#!/usr/bin/env python3
"""Export an experiment arm to the COCO layout RF-DETR trains on:
  <out>/train/_annotations.coco.json + images, <out>/valid/..., <out>/test/...   (images are symlinked)

  python tools/export_rfdetr_coco.py --config C --arm A1 --out /data/syngen_od/rfdetr/A1
Then (isolated env):
  from rfdetr import RFDETRMedium
  RFDETRMedium(pretrain_weights=".../rf-detr-medium.pth").train(dataset_dir=".../rfdetr/A1", epochs=20,
                                                                batch_size=8, grad_accum_steps=2, lr=1e-4,
                                                                output_dir=".../runs_rfdetr/A1")
Category ids start at 0 in project class order. Before trusting evaluation, check that the fine-tuned model's
class_id -> name mapping is what you expect (14_predict --backend rfdetr, with eval.predict.id_to_name if needed).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.boxes import labels_to_pixel  # noqa: E402
from syngen.common import base_parser, load  # noqa: E402
from syngen.yolo_io import image_size, img2label_path, list_images, read_labels  # noqa: E402


def export(paths, classes, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    aid = 0
    for i, ip in enumerate(paths):
        w, h = image_size(ip)
        name = f"{i:07d}_{Path(ip).name}"  # unique even if sources share file names
        dst = out_dir / name
        if not dst.exists():
            os.symlink(os.path.realpath(ip), dst)
        images.append({"id": i, "file_name": name, "width": w, "height": h})
        for r in labels_to_pixel(read_labels(img2label_path(ip)), w, h):
            x1, y1, x2, y2 = (float(v) for v in r[1:5])
            anns.append({"id": aid, "image_id": i, "category_id": int(r[0]), "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            aid += 1
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(classes)]}
    with open(out_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"{out_dir}: {len(images)} images, {len(anns)} boxes")


def main():
    p = base_parser(__doc__)
    p.add_argument("--arm", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = load(args)
    arm_dir = Path(cfg["paths"]["experiments"]) / args.arm
    out = Path(args.out)
    export(list_images(arm_dir / "train.txt"), cfg["classes"], out / "train")
    val = list_images(cfg["experiments"]["val"])
    export(val, cfg["classes"], out / "valid")
    export(val, cfg["classes"], out / "test")


if __name__ == "__main__":
    main()
