#!/usr/bin/env python3
"""06 Run the teacher detector over a generated task (or any image dir/list) -> teacher_dets.shard<k>.jsonl.

  python scripts/06_teacher_detect.py --config C --task style
  python scripts/06_teacher_detect.py --config C --images /path/deploy_unlabeled/images --out-dir /path/pseudo
Resumable: images already present in the output are skipped.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.common import Progress, add_shard_args, base_parser, load, shard  # noqa: E402
from syngen.detectors import build_detector  # noqa: E402
from syngen.records import append_jsonl, det_record, read_dets  # noqa: E402
from syngen.yolo_io import list_images  # noqa: E402


def main():
    p = base_parser(__doc__)
    add_shard_args(p)
    p.add_argument("--task", help="task name under config `tasks` (uses <out>/images)")
    p.add_argument("--images", help="image dir or .txt list (instead of --task)")
    p.add_argument("--out-dir", help="where teacher_dets*.jsonl go (default: task out dir)")
    p.add_argument("--chunk", type=int, default=64, help="images per detector call / flush")
    args = p.parse_args()
    cfg = load(args)
    if args.task:
        src = Path(cfg["tasks"][args.task]["out"]) / "images"
        out_dir = Path(args.out_dir or cfg["tasks"][args.task]["out"])
    else:
        if not (args.images and args.out_dir):
            p.error("give --task, or --images together with --out-dir")
        src, out_dir = args.images, Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set(read_dets(out_dir))
    imgs = [x for x in shard(list_images(src), args) if x not in done]
    print(f"{len(imgs)} images to run teacher on ({len(done)} already done)")
    if not imgs:
        return
    det = build_detector(cfg["teacher"], cfg["classes"])
    out_file = out_dir / f"teacher_dets.shard{args.shard}.jsonl"
    prog = Progress(len(imgs), "teacher", every=100)
    for i in range(0, len(imgs), args.chunk):
        batch = imgs[i:i + args.chunk]
        for path, (w, h, dets) in zip(batch, det.detect(batch)):
            append_jsonl(out_file, det_record(os.path.abspath(path), w, h, dets))
            prog.step()


if __name__ == "__main__":
    main()
