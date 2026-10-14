#!/usr/bin/env python3
"""05 (optional) Insert objects into deployment backgrounds with mask inpainting (e.g. FLUX.1-Fill-dev).

Backgrounds must come with trustworthy labels (pseudo-labelled frames or verified-empty deploy_bg frames), because
unlabelled real objects would become false "background". The inpainted region is a crop around the mask so small,
distant objects (the typical misses) get enough resolution. Inserted boxes are provisional: 07_filter replaces each
with the teacher's tight box, or rejects the image.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.boxes import iou_matrix, labels_to_pixel, xyxy_to_xywhn  # noqa: E402
from syngen.common import Progress, add_shard_args, base_parser, load, shard  # noqa: E402
from syngen.config import load_prompts  # noqa: E402
from syngen.generators import build_generator, item_seed  # noqa: E402
from syngen.records import append_jsonl  # noqa: E402
from syngen.yolo_io import img2label_path, list_images, read_labels, write_labels  # noqa: E402


def sample_boxes(rng, w, h, existing, task, classes):
    """Random non-overlapping boxes, bottom-half biased (objects stand on the ground in most deployment views)."""
    out = []
    n = rng.randint(*task["objects_per_image"])
    names = list(task["object_height_px"])
    for _ in range(n * 20):
        if len(out) >= n:
            break
        name = names[rng.randrange(len(names))]
        oh = rng.uniform(*task["object_height_px"][name])
        ow = oh / rng.uniform(*task["aspect_hw"][name])
        if ow >= w * 0.9 or oh >= h * 0.9:
            continue
        x1 = rng.uniform(0, w - ow)
        y2 = rng.uniform(max(oh, h * 0.35), h)
        box = np.array([x1, y2 - oh, x1 + ow, y2])
        others = np.array([b for _, b in out] + list(existing)).reshape(-1, 4)
        if len(others) and iou_matrix(box[None], others).max() > 0:
            continue
        out.append((classes.index(name), box))
    return out


def crop_window(box, w, h, scale, min_crop):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = max(min_crop, scale * max(box[2] - box[0], box[3] - box[1]))
    s = min(s, w, h)
    x1 = int(np.clip(cx - s / 2, 0, w - s))
    y1 = int(np.clip(cy - s / 2, 0, h - s))
    return x1, y1, x1 + int(s), y1 + int(s)


def main():
    p = base_parser(__doc__)
    add_shard_args(p)
    args = p.parse_args()
    cfg = load(args)
    task = cfg["tasks"]["insert"]
    classes = cfg["classes"]
    obj_prompts = load_prompts(cfg)["insert"]["objects"]
    out = Path(task["out"])
    (out / "images").mkdir(parents=True, exist_ok=True)

    bgs = list_images(task["backgrounds"])
    rng0 = random.Random(cfg["seed"] + 2)
    jobs = [(f"ins_{i:06d}", bgs[rng0.randrange(len(bgs))]) for i in range(task["num_images"])]
    jobs = shard(jobs, args)
    todo = [j for j in jobs if not (out / "images" / f"{j[0]}.jpg").exists()]
    print(f"{len(jobs)} jobs in this shard, {len(todo)} to generate ({len(bgs)} backgrounds)")
    if not todo:
        return
    gen = build_generator("inpaint", cfg["generation"]["inpaint"])
    size = task["inpaint_size"]
    prog = Progress(len(todo), "insert")
    for jid, bg in todo:
        rng = random.Random(item_seed(cfg["seed"], jid))
        img = Image.open(bg).convert("RGB")
        w, h = img.size
        lab = read_labels(img2label_path(bg))
        existing = labels_to_pixel(lab, w, h)[:, 1:5]
        inserted = []
        for k, (cls, box) in enumerate(sample_boxes(rng, w, h, existing, task, classes)):
            cx1, cy1, cx2, cy2 = crop_window(box, w, h, task["context_scale"], task["min_crop"])
            crop = img.crop((cx1, cy1, cx2, cy2))
            f = size / crop.width
            mask = Image.new("L", (size, size), 0)
            ImageDraw.Draw(mask).rectangle([(box[0] - cx1) * f, (box[1] - cy1) * f,
                                            (box[2] - cx1) * f, (box[3] - cy1) * f], fill=255)
            prompt = rng.choice(obj_prompts[classes[cls]])
            res = gen.inpaint(crop.resize((size, size), Image.LANCZOS), mask, prompt, item_seed(cfg["seed"], f"{jid}_{k}"))
            img.paste(res.resize(crop.size, Image.LANCZOS), (cx1, cy1))
            inserted.append({"cls": cls, "box": box.round(1).tolist(), "prompt": prompt})
        if not inserted:
            prog.step()
            continue
        op = out / "images" / f"{jid}.jpg"
        img.save(op, quality=95)
        new = np.array([[d["cls"], *xyxy_to_xywhn(np.array(d["box"]), w, h)[0]] for d in inserted])
        write_labels(img2label_path(str(op)), np.vstack([lab.reshape(-1, 5), new]))
        append_jsonl(out / f"meta.shard{args.shard}.jsonl",
                     {"image": os.path.abspath(op), "task": "insert", "source": bg, "inserted": inserted,
                      "num_existing": int(len(lab))})
        prog.step()


if __name__ == "__main__":
    main()
