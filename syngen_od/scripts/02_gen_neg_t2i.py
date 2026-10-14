#!/usr/bin/env python3
"""02 Hard negatives via text-to-image (Qwen-Image): scenes with look-alike objects and no target classes.

Writes <out>/images/*.jpg, empty <out>/labels/*.txt, <out>/meta.jsonl. Resumable; supports --shard/--num-shards.
"""
import itertools
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.common import Progress, add_shard_args, base_parser, load, shard  # noqa: E402
from syngen.config import load_prompts  # noqa: E402
from syngen.generators import build_generator, item_seed  # noqa: E402
from syngen.imgutil import is_degenerate  # noqa: E402
from syngen.records import append_jsonl  # noqa: E402


def build_jobs(np_cfg, n, sizes, seed):
    combos = list(itertools.product(np_cfg["views"], np_cfg["scenes"], np_cfg["confusers"], np_cfg["conditions"]))
    rng = random.Random(seed)
    rng.shuffle(combos)
    jobs = []
    for i in range(n):  # cycle through shuffled combos so coverage is even
        view, scene, conf, cond = combos[i % len(combos)]
        w, h = sizes[rng.randrange(len(sizes))]
        prompt = f"{view}, {scene}, with {conf}, {cond}. {np_cfg['suffix']}"
        jobs.append({"id": f"neg_{i:06d}", "prompt": prompt, "w": w, "h": h,
                     "tags": {"view": view, "scene": scene, "confuser": conf, "condition": cond}})
    return jobs


def main():
    p = base_parser(__doc__)
    add_shard_args(p)
    args = p.parse_args()
    cfg = load(args)
    task = cfg["tasks"]["neg_t2i"]
    np_cfg = load_prompts(cfg)["negatives"]
    out = Path(task["out"])
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    jobs = shard(build_jobs(np_cfg, task["num_images"], task["sizes"], cfg["seed"]), args)
    todo = [j for j in jobs if not (out / "images" / f"{j['id']}.jpg").exists()]
    print(f"{len(jobs)} jobs in this shard, {len(todo)} to generate")
    if not todo:
        return
    gen = build_generator("t2i", task.get("generator") or cfg["generation"]["t2i"])
    prog = Progress(len(todo), "neg_t2i")
    for j in todo:
        s = item_seed(cfg["seed"], j["id"])
        img = gen.generate(j["prompt"], np_cfg.get("negative_prompt"), j["w"], j["h"], s).convert("RGB")
        if is_degenerate(img):
            print(f"skip degenerate output {j['id']}")
            prog.step()
            continue
        ip = out / "images" / f"{j['id']}.jpg"
        img.save(ip, quality=95)
        (out / "labels" / f"{j['id']}.txt").write_text("")  # background image: empty label file
        append_jsonl(out / f"meta.shard{args.shard}.jsonl",
                     {"image": os.path.abspath(ip), "task": "neg_t2i", "prompt": j["prompt"], "seed": s, **j["tags"]})
        prog.step()


if __name__ == "__main__":
    main()
