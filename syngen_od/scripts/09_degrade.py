#!/usr/bin/env python3
"""09 Camera-like degradation (downscale / blur / noise / JPEG) on the kept images of a task, in place.

Generated images are too clean compared to real camera frames; without this the model can learn "clean = synthetic".
Run after 07/08. A marker file prevents degrading the same task twice. Labels are unaffected (size unchanged).
"""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.common import Progress, base_parser, load  # noqa: E402
from syngen.imgutil import degrade, stable_seed  # noqa: E402
from syngen.yolo_io import list_images  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--force", action="store_true", help="run even if already degraded")
    args = p.parse_args()
    cfg = load(args)
    task_dir = Path(cfg["tasks"][args.task]["out"])
    marker = task_dir / ".degraded"
    if marker.exists() and not args.force:
        print(f"{task_dir} already degraded ({marker}); use --force to apply again")
        return
    imgs = list_images(task_dir / "kept.txt")
    prog = Progress(len(imgs), "degrade", every=200)
    for ip in imgs:
        img = Image.open(ip).convert("RGB")
        degrade(img, cfg["degrade"], stable_seed(cfg["seed"], Path(ip).name)).save(ip, quality=95)
        prog.step()
    marker.write_text(f"{len(imgs)} images\n")


if __name__ == "__main__":
    main()
