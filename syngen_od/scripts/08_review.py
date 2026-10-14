#!/usr/bin/env python3
"""08 Manual spot-check: contact sheets of a random sample of kept images (boxes drawn), then apply rejections.

  # 1) make sheets: <task>/review/sheet_000.png ... + index.csv
  python scripts/08_review.py --config C --task style --frac 0.05
  # 2) write the bad cell numbers (one per line, or comma-separated) into <task>/review/rejects.txt
  # 3) apply: removes them from kept.txt (original kept as kept.before_review.txt) and reports the reject rate
  python scripts/08_review.py --config C --task style --apply
For the side-by-side comparison with the source image (remove/style), use --pairs.
"""
import csv
import random
import re
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.boxes import labels_to_pixel  # noqa: E402
from syngen.common import base_parser, load  # noqa: E402
from syngen.records import read_meta  # noqa: E402
from syngen.yolo_io import img2label_path, list_images, read_labels, write_list  # noqa: E402

COLORS = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 200, 0), (200, 64, 255), (0, 220, 220)]


def draw(path, classes, tile):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    for r in labels_to_pixel(read_labels(img2label_path(path)), w, h):
        c = COLORS[int(r[0]) % len(COLORS)]
        d.rectangle(r[1:5].tolist(), outline=c, width=max(2, w // 300))
        d.text((r[1] + 2, r[2] + 1), classes[int(r[0])], fill=c)
    img.thumbnail((tile, tile))
    return img


def make_sheets(paths, meta, classes, out_dir, tile, cols, rows, pairs):
    out_dir.mkdir(parents=True, exist_ok=True)
    per = cols * rows
    index = []
    for s in range(0, len(paths), per):
        chunk = paths[s:s + per]
        cw = tile * (2 if pairs else 1)
        sheet = Image.new("RGB", (cols * cw, rows * (tile + 18)), (30, 30, 30))
        d = ImageDraw.Draw(sheet)
        for k, ip in enumerate(chunk):
            n = s + k
            x, y = (k % cols) * cw, (k // cols) * (tile + 18)
            im = draw(ip, classes, tile)
            sheet.paste(im, (x, y + 18))
            if pairs and meta.get(ip, {}).get("source"):
                src = Image.open(meta[ip]["source"]).convert("RGB")
                src.thumbnail((tile, tile))
                sheet.paste(src, (x + tile, y + 18))
            d.text((x + 4, y + 3), f"#{n}  {Path(ip).name[:40]}", fill=(255, 255, 255))
            index.append((n, ip))
        sheet.save(out_dir / f"sheet_{s // per:03d}.png")
    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([("idx", "image")] + index)
    print(f"{len(paths)} images -> {out_dir}/sheet_*.png; put bad #numbers into {out_dir}/rejects.txt")


def apply(task_dir):
    rev = task_dir / "review"
    idx = {int(r["idx"]): r["image"] for r in csv.DictReader(open(rev / "index.csv", encoding="utf-8"))}
    rej_file = rev / "rejects.txt"
    nums = {int(x) for x in re.findall(r"\d+", rej_file.read_text())} if rej_file.exists() else set()
    bad = {idx[n] for n in nums if n in idx}
    kept_file = task_dir / "kept.txt"
    backup = task_dir / "kept.before_review.txt"
    if not backup.exists():
        shutil.copy(kept_file, backup)
    kept = [x for x in list_images(backup) if x not in bad]
    write_list(kept_file, kept)
    rate = len(bad) / max(len(idx), 1)
    print(f"rejected {len(bad)}/{len(idx)} reviewed ({rate:.1%}); kept.txt now {len(kept)} images")
    if rate > 0.10:
        print("WARNING: >10% of the auto-kept sample is bad -> tighten 07 thresholds / fix prompts rather than "
              "relying on manual review")


def main():
    p = base_parser(__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--frac", type=float, default=0.05)
    p.add_argument("--max", type=int, default=400)
    p.add_argument("--tile", type=int, default=384)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--pairs", action="store_true", help="show source image next to each edit")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    cfg = load(args)
    task_dir = Path(cfg["tasks"][args.task]["out"])
    if args.apply:
        return apply(task_dir)
    kept = list_images(task_dir / "kept.txt")
    n = min(args.max, max(1, int(len(kept) * args.frac)))
    sample = sorted(random.Random(cfg["seed"]).sample(kept, min(n, len(kept))))
    make_sheets(sample, read_meta(task_dir), cfg["classes"], task_dir / "review", args.tile, args.cols, args.rows,
                args.pairs)


if __name__ == "__main__":
    main()
