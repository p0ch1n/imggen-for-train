#!/usr/bin/env python3
"""Create a tiny synthetic COCO + deployment dataset so the whole pipeline can be smoke-tested without real data.

  python tools/make_fake_data.py --root /tmp/smoke
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

COCO_CATS = [(1, "person"), (2, "bicycle"), (3, "car"), (4, "motorcycle"), (6, "bus"), (8, "truck"), (18, "dog")]
SHAPE = {"person": (0.35, 1.0), "car": (1.0, 0.55), "bicycle": (1.0, 0.7), "motorcycle": (1.0, 0.7),
         "bus": (1.0, 0.5), "truck": (1.0, 0.6), "dog": (1.0, 0.7)}


def scene(rng, w, h, n_obj, names, bumps=True):
    a = np.clip(np.array([rng.randint(50, 180) for _ in range(3)]) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 20, (h, w, 3)),
                0, 255).astype(np.uint8)
    img = Image.fromarray(a).filter(ImageFilter.GaussianBlur(1.5))
    d = ImageDraw.Draw(img)
    if bumps:
        for _ in range(rng.randint(0, 4)):
            cx, cy, r = rng.randrange(w), rng.randrange(h), rng.randint(4, 20)
            d.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=(90, 70, 40))
    objs = []
    for _ in range(n_obj):
        name = rng.choice(names)
        sw, sh = SHAPE[name]
        s = rng.uniform(20, min(w, h) * 0.6)
        bw, bh = s * sw, s * sh
        x, y = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        d.rectangle([x, y, x + bw, y + bh], fill=tuple(rng.randint(0, 255) for _ in range(3)),
                    outline=(0, 0, 0), width=2)
        objs.append((name, [x, y, bw, bh]))
    return img, objs


def make_coco(root, split, n, rng):
    img_dir = root / "coco" / f"{split}2017"
    img_dir.mkdir(parents=True, exist_ok=True)
    names = [n for _, n in COCO_CATS]
    cid = {n: i for i, n in COCO_CATS}
    images, anns = [], []
    for i in range(n):
        w, h = rng.choice([(320, 240), (240, 320), (360, 360)])
        k = 0 if rng.random() < 0.15 else rng.randint(1, 4)
        img, objs = scene(rng, w, h, k, names)
        fn = f"{i:012d}.jpg"
        img.save(img_dir / fn, quality=90)
        images.append({"id": i, "file_name": fn, "width": w, "height": h})
        for name, bb in objs:
            anns.append({"id": len(anns), "image_id": i, "category_id": cid[name], "bbox": bb,
                         "area": bb[2] * bb[3], "iscrowd": 1 if rng.random() < 0.02 else 0})
    ann_dir = root / "coco" / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_dir / f"instances_{split}2017.json", "w") as f:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": i, "name": n} for i, n in COCO_CATS]}, f)


def make_yolo_set(root, n, rng, classes, with_labels, max_obj):
    (root / "images").mkdir(parents=True, exist_ok=True)
    if with_labels:
        (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        w, h = 320, 240
        img, objs = scene(rng, w, h, rng.randint(0, max_obj) if max_obj else 0, ["person", "car"])
        img.save(root / "images" / f"frame_{i:05d}.jpg", quality=80)
        if with_labels:
            lines = [f"{classes.index(nm)} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                     for nm, (x, y, bw, bh) in objs]
            (root / "labels" / f"frame_{i:05d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-val", type=int, default=30)
    args = p.parse_args()
    root = Path(args.root)
    rng = random.Random(0)
    classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
    make_coco(root, "train", args.n_train, rng)
    make_coco(root, "val", args.n_val, rng)
    make_yolo_set(root / "data" / "deploy_eval", 40, rng, classes, True, 3)
    make_yolo_set(root / "data" / "deploy_bg", 30, rng, classes, False, 0)
    # unlabeled deployment frames: hidden labels next to them feed the mock "oracle" teacher
    make_yolo_set(root / "data" / "deploy_unlabeled", 60, rng, classes, True, 3)
    print(f"fake data under {root}")


if __name__ == "__main__":
    main()
