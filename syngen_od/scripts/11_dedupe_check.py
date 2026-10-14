#!/usr/bin/env python3
"""11 Leakage check: near-duplicate images (dHash) between training sources and evaluation sets.

Consecutive frames from a fixed camera are almost identical; if such frames are in both pseudo-label training data
and deploy_eval, A4 looks far better than it really is. Run before training; fix by splitting per camera / per day.

  python scripts/11_dedupe_check.py --config C [--max-dist 6] [--remove]
--remove drops offending training images from the corresponding kept.txt lists.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.common import base_parser, load  # noqa: E402
from syngen.imgutil import dhash, hamming_matrix  # noqa: E402
from syngen.records import write_json  # noqa: E402
from syngen.yolo_io import list_images, write_list  # noqa: E402


def hashes(paths):
    return np.stack([dhash(p) for p in paths]) if paths else np.zeros((0, 8), dtype=np.uint8)


def main():
    p = base_parser(__doc__)
    p.add_argument("--max-dist", type=int, default=6, help="hamming distance (of 64 bits) counted as duplicate")
    p.add_argument("--remove", action="store_true")
    args = p.parse_args()
    cfg = load(args)
    eval_sets = {k: v for k, v in cfg["eval"]["sets"].items() if k != "coco_val" and Path(v).exists()}
    eval_paths = {k: list_images(v) for k, v in eval_sets.items()}
    eval_h = {k: hashes(v) for k, v in eval_paths.items()}
    sources = sorted({s["list"] for arm in cfg["experiments"]["arms"].values() for s in arm})
    report = {}
    for src in sources:
        if not Path(src).exists():
            continue
        paths = list_images(src)
        h = hashes(paths)
        dup = np.zeros(len(paths), dtype=bool)
        for k, eh in eval_h.items():
            for i in range(0, len(h), 2048):  # chunk to bound memory
                if len(eh):
                    d = hamming_matrix(h[i:i + 2048], eh)
                    dup[i:i + 2048] |= (d <= args.max_dist).any(axis=1)
        report[src] = {"images": len(paths), "near_duplicates_of_eval": int(dup.sum()),
                       "examples": [paths[i] for i in np.flatnonzero(dup)[:10]]}
        flag = "  <-- LEAK" if dup.any() else ""
        print(f"{src}: {int(dup.sum())}/{len(paths)} near-duplicates of eval images{flag}")
        if args.remove and dup.any():
            write_list(src, [x for x, d in zip(paths, dup) if not d])
            print(f"  removed from {src}")
    # within-eval duplicates inflate the effective sample size used by the bootstrap
    for k, eh in eval_h.items():
        if len(eh) > 1:
            d = hamming_matrix(eh, eh)
            np.fill_diagonal(d, 99)
            report[f"eval:{k}:internal_near_duplicates"] = int((d <= args.max_dist).any(axis=1).sum())
    write_json(Path(cfg["paths"]["results"]) / "dedupe_report.json", report)


if __name__ == "__main__":
    main()
