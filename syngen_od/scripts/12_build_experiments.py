#!/usr/bin/env python3
"""12 Build one Ultralytics dataset per experiment arm (no copying: train.txt lists absolute image paths).

experiments/<arm>/train.txt, experiments/<arm>/data.yaml, experiments/<arm>/composition.json
Every arm shares the same base_train list so that the only difference between arms is the added data.
"""
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.common import base_parser, load  # noqa: E402
from syngen.records import write_json  # noqa: E402
from syngen.yolo_io import img2label_path, list_images, write_list  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--arms", nargs="*", help="subset of arms (default: all)")
    args = p.parse_args()
    cfg = load(args)
    ex = cfg["experiments"]
    root = Path(cfg["paths"]["experiments"])
    base = list_images(ex["base_train"])
    val = ex["val"]
    for arm, sources in ex["arms"].items():
        if args.arms and arm not in args.arms:
            continue
        files, comp = list(base), {"base_train": len(base)}
        for i, s in enumerate(sources or []):
            if not Path(s["list"]).exists():
                sys.exit(f"[{arm}] missing source list {s['list']} (run the generation/filter steps first)")
            imgs = list_images(s["list"])
            random.Random(cfg["seed"] + i).shuffle(imgs)
            imgs = imgs[: s.get("max", len(imgs))]
            missing = [x for x in imgs if not Path(img2label_path(x)).exists()]
            if missing:
                sys.exit(f"[{arm}] {len(missing)} images in {s['list']} have no label file, e.g. {missing[0]}")
            if s.get("max") and len(imgs) < s["max"]:
                print(f"[{arm}] note: {s['list']} has only {len(imgs)} images (< max {s['max']})")
            files += imgs * int(s.get("repeat", 1))
            comp[s["list"]] = len(imgs) * int(s.get("repeat", 1))
        d = root / arm
        d.mkdir(parents=True, exist_ok=True)
        write_list(d / "train.txt", files)
        data = {"path": str(d), "train": str(d / "train.txt"), "val": str(val),
                "names": {i: n for i, n in enumerate(cfg["classes"])}}
        with open(d / "data.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        comp["total"] = len(files)
        comp["added_frac"] = round(1 - len(base) / max(len(files), 1), 4)
        write_json(d / "composition.json", comp)
        print(f"[{arm}] {len(files)} train images ({comp['added_frac']:.1%} added) -> {d / 'data.yaml'}")


if __name__ == "__main__":
    main()
