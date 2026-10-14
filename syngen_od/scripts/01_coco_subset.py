#!/usr/bin/env python3
"""01 Convert COCO instances json into a YOLO subset containing only the project classes.

Output: <coco_subset>/images/{train,val}, <coco_subset>/labels/{train,val}
"""
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.common import base_parser, load  # noqa: E402
from syngen.yolo_io import write_labels  # noqa: E402

import numpy as np  # noqa: E402


def convert(ann_path, img_dir, out_root, split, classes, max_images, bg_frac, min_box_px, link_mode, seed):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    name_to_idx = {n: i for i, n in enumerate(classes)}
    cat_map = {c["id"]: name_to_idx[c["name"]] for c in coco["categories"] if c["name"] in name_to_idx}
    missing = set(classes) - {c["name"] for c in coco["categories"]}
    if missing:
        print(f"WARNING: classes not in COCO, they will have no labels: {sorted(missing)}")
    imgs = {im["id"]: im for im in coco["images"]}
    anns = defaultdict(list)
    has_crowd_target = set()
    for a in coco["annotations"]:
        if a["category_id"] not in cat_map:
            continue
        if a.get("iscrowd", 0):
            has_crowd_target.add(a["image_id"])
            continue
        anns[a["image_id"]].append(a)

    pos_ids = sorted(anns)
    # background candidates: no target class at all (crowd-only images are excluded: they *do* contain targets)
    bg_ids = sorted(set(imgs) - set(anns) - has_crowd_target)
    rng = random.Random(seed)
    rng.shuffle(pos_ids)
    rng.shuffle(bg_ids)
    pos_ids = pos_ids[:max_images] if max_images else pos_ids
    bg_ids = bg_ids[: int(len(pos_ids) * bg_frac)]

    img_out = Path(out_root) / "images" / split
    lbl_out = Path(out_root) / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n_box = 0
    for iid in pos_ids + bg_ids:
        im = imgs[iid]
        w, h = im["width"], im["height"]
        rows = []
        for a in anns.get(iid, []):
            x, y, bw, bh = a["bbox"]
            if bw < min_box_px or bh < min_box_px:
                continue
            rows.append([cat_map[a["category_id"]], (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h])
        src = Path(img_dir) / im["file_name"]
        dst = img_out / im["file_name"]
        if link_mode == "download" and not src.exists():  # fetch only the sampled images from coco_url
            import urllib.request
            src.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(im["coco_url"], src)
        if not dst.exists():
            if link_mode in ("symlink", "download"):
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
        write_labels(lbl_out / (Path(im["file_name"]).stem + ".txt"), np.asarray(rows).reshape(-1, 5))
        n_box += len(rows)
    print(f"[{split}] {len(pos_ids)} images with targets + {len(bg_ids)} background, {n_box} boxes -> {img_out}")


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg = load(args)
    c, P = cfg["coco_subset"], cfg["paths"]
    convert(P["coco_train_ann"], P["coco_train_images"], P["coco_subset"], "train", cfg["classes"],
            c["train_max_images"], c["background_frac"], c["min_box_px"], c["link_mode"], cfg["seed"])
    convert(P["coco_val_ann"], P["coco_val_images"], P["coco_subset"], "val", cfg["classes"],
            c["val_max_images"], c["background_frac"], c["min_box_px"], c["link_mode"], cfg["seed"])


if __name__ == "__main__":
    main()
