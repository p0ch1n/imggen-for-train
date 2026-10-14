#!/usr/bin/env python3
"""10 Pseudo-labels for unlabelled deployment frames from teacher detections (arm A4; also insert backgrounds).

Prerequisite: 06_teacher_detect.py --images <deploy_unlabeled> --out-dir <paths.pseudo>
Output: <pseudo>/images (symlinks), <pseudo>/labels, <pseudo>/kept.txt, <pseudo>/pseudo_summary.json
IMPORTANT: frames must not come from the same cameras/time window as deploy_eval (check with 11_dedupe_check).
"""
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.boxes import xyxy_to_xywhn  # noqa: E402
from syngen.common import base_parser, load  # noqa: E402
from syngen.records import read_dets, write_json  # noqa: E402
from syngen.yolo_io import img2label_path, write_labels, write_list  # noqa: E402


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg = load(args)
    pc = cfg["pseudo_label"]
    out = Path(cfg["paths"]["pseudo"])
    dets_all = read_dets(out)
    if not dets_all:
        sys.exit(f"no teacher_dets*.jsonl in {out}; run 06_teacher_detect.py --images ... --out-dir {out} first")
    rng = random.Random(cfg["seed"])
    items = sorted(dets_all.items())
    rng.shuffle(items)
    pos, empty, n_amb, n_box = [], [], 0, 0
    for ip, (w, h, d) in items:
        hi = d[d[:, 5] >= pc["conf_high"]]
        amb = d[(d[:, 5] >= pc["conf_low"]) & (d[:, 5] < pc["conf_high"])]
        if pc["drop_ambiguous"] and len(amb):
            n_amb += 1
            continue
        (pos if len(hi) else empty).append((ip, w, h, hi))
    n_pos = min(len(pos), pc["max_images"])
    n_empty = min(len(empty), int(n_pos * pc["max_empty_frac"] / max(1e-6, 1 - pc["max_empty_frac"])))
    chosen = pos[:n_pos] + empty[:n_empty]
    kept = []
    (out / "images").mkdir(parents=True, exist_ok=True)
    for i, (ip, w, h, hi) in enumerate(chosen):
        # unique name: different cameras often reuse the same frame file names
        dst = out / "images" / f"{i:06d}_{Path(ip).name}"
        if not dst.exists():
            os.symlink(ip, dst)
        lab = np.column_stack([hi[:, 0], xyxy_to_xywhn(hi[:, 1:5], w, h)]) if len(hi) else np.zeros((0, 5))
        write_labels(img2label_path(str(dst)), lab)
        n_box += len(hi)
        kept.append(os.path.abspath(dst))
    write_list(out / "kept.txt", kept)
    s = {"teacher_images": len(items), "dropped_ambiguous": n_amb, "with_labels": n_pos, "empty": n_empty,
         "boxes": n_box}
    write_json(out / "pseudo_summary.json", s)
    print(s)


if __name__ == "__main__":
    main()
