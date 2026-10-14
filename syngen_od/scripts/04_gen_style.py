#!/usr/bin/env python3
"""04 Label-preserving style transfer with Qwen-Image-Edit (night / IR / rain / fog / CCTV ...).

Boxes are copied from the source image; 07_filter rejects edits that moved, dropped or invented objects.
Each source image gets one style (round-robin over styles) so the source set is not over-represented.
"""
import os
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.common import Progress, add_shard_args, base_parser, load, shard  # noqa: E402
from syngen.config import load_prompts  # noqa: E402
from syngen.generators import build_generator, item_seed  # noqa: E402
from syngen.imgutil import is_degenerate  # noqa: E402
from syngen.records import append_jsonl  # noqa: E402
from syngen.yolo_io import image_size, img2label_path, list_images, read_labels, write_labels  # noqa: E402


def main():
    p = base_parser(__doc__)
    add_shard_args(p)
    p.add_argument("--styles", nargs="*", help="only these style keys (default: all in prompts.yaml)")
    args = p.parse_args()
    cfg = load(args)
    task = cfg["tasks"]["style"]
    sp = load_prompts(cfg)["style"]
    styles = {k: v for k, v in sp["styles"].items() if not args.styles or k in args.styles}
    out = Path(task["out"])
    (out / "images").mkdir(parents=True, exist_ok=True)

    imgs = list_images(Path(cfg["paths"]["coco_subset"]) / "images" / "train")
    random.Random(cfg["seed"] + 1).shuffle(imgs)
    sources = []
    for ip in imgs:
        if len(read_labels(img2label_path(ip))) == 0:  # need objects: the point is to keep them under new styles
            continue
        if min(image_size(ip)) < task["min_short_side"]:
            continue
        sources.append(ip)
        if len(sources) >= task["num_images"]:
            break
    keys = list(styles)
    jobs = shard([(f"st_{keys[i % len(keys)]}_{Path(ip).stem}", ip, keys[i % len(keys)])
                  for i, ip in enumerate(sources)], args)
    todo = [j for j in jobs if not (out / "images" / f"{j[0]}.jpg").exists()]
    print(f"{len(jobs)} jobs in this shard, {len(todo)} to generate; styles={keys}")
    if not todo:
        return
    gen = build_generator("edit", task.get("generator") or cfg["generation"]["edit"])
    prog = Progress(len(todo), "style")
    for jid, ip, key in todo:
        src = Image.open(ip).convert("RGB")
        prompt = styles[key] + sp.get("suffix", "")
        s = item_seed(cfg["seed"], jid)
        img = gen.edit(src, prompt, s, negative_prompt=sp.get("negative_prompt"))
        if is_degenerate(img):
            prog.step()
            continue
        op = out / "images" / f"{jid}.jpg"
        img.convert("RGB").save(op, quality=95)
        write_labels(img2label_path(str(op)), read_labels(img2label_path(ip)))
        append_jsonl(out / f"meta.shard{args.shard}.jsonl",
                     {"image": os.path.abspath(op), "task": "style", "style": key, "source": ip,
                      "prompt": prompt, "seed": s})
        prog.step()


if __name__ == "__main__":
    main()
