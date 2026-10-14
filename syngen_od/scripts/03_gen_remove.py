#!/usr/bin/env python3
"""03 Paired negatives: Qwen-Image-Edit removes the target class (default: person) from COCO images.

Labels = source labels minus the removed classes. 07_filter checks that the removed objects are really gone and
that everything else is unchanged.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.boxes import labels_to_pixel  # noqa: E402
from syngen.common import Progress, add_shard_args, base_parser, load, shard  # noqa: E402
from syngen.config import load_prompts  # noqa: E402
from syngen.generators import build_generator, item_seed  # noqa: E402
from syngen.imgutil import is_degenerate  # noqa: E402
from syngen.records import append_jsonl  # noqa: E402
from syngen.yolo_io import image_size, img2label_path, list_images, read_labels, write_labels  # noqa: E402


def select_sources(cfg, task):
    classes = cfg["classes"]
    rm = [classes.index(c) for c in task["remove_classes"]]
    cands = []
    for ip in list_images(Path(cfg["paths"]["coco_subset"]) / "images" / "train"):
        lab = read_labels(img2label_path(ip))
        is_rm = np.isin(lab[:, 0], rm)
        n = int(is_rm.sum())
        if n == 0 or n > task["max_instances"]:
            continue
        areas = lab[is_rm, 3] * lab[is_rm, 4]
        if areas.min() < task["min_area_frac"] or areas.max() > task["max_area_frac"]:
            continue
        cands.append(ip)
    random.Random(cfg["seed"]).shuffle(cands)
    print(f"{len(cands)} candidate source images")
    return cands, rm


def main():
    p = base_parser(__doc__)
    add_shard_args(p)
    args = p.parse_args()
    cfg = load(args)
    task = cfg["tasks"]["remove"]
    pr = load_prompts(cfg)["remove"]
    out = Path(task["out"])
    (out / "images").mkdir(parents=True, exist_ok=True)

    cands, rm = select_sources(cfg, task)
    sources = []
    for ip in cands:  # short-side check reads headers only, stop as soon as we have enough
        if min(image_size(ip)) >= task["min_short_side"]:
            sources.append(ip)
        if len(sources) >= task["num_images"]:
            break
    jobs = shard([(f"rm_{Path(ip).stem}", ip) for ip in sources], args)
    todo = [j for j in jobs if not (out / "images" / f"{j[0]}.jpg").exists()]
    print(f"{len(jobs)} jobs in this shard, {len(todo)} to generate")
    if not todo:
        return
    gen = build_generator("edit", task.get("generator") or cfg["generation"]["edit"])
    prog = Progress(len(todo), "remove")
    for jid, ip in todo:
        src = Image.open(ip).convert("RGB")
        lab = read_labels(img2label_path(ip))
        is_rm = np.isin(lab[:, 0], rm)
        rm_px = labels_to_pixel(lab[is_rm], *src.size)[:, 1:5]
        s = item_seed(cfg["seed"], jid)
        img = gen.edit(src, pr["prompt"], s, negative_prompt=pr.get("negative_prompt"), hint_boxes=rm_px)
        if is_degenerate(img):
            prog.step()
            continue
        op = out / "images" / f"{jid}.jpg"
        img.convert("RGB").save(op, quality=95)
        write_labels(img2label_path(str(op)), lab[~is_rm])
        append_jsonl(out / f"meta.shard{args.shard}.jsonl",
                     {"image": os.path.abspath(op), "task": "remove", "source": ip, "prompt": pr["prompt"], "seed": s,
                      "removed_classes": rm, "removed_boxes": rm_px.round(1).tolist()})
        prog.step()


if __name__ == "__main__":
    main()
