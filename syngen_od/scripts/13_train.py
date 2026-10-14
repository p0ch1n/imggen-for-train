#!/usr/bin/env python3
"""13 Fine-tune the production YOLO checkpoint once per arm with an identical recipe (Ultralytics).

  python scripts/13_train.py --config C                 # all arms in train.arms, sequentially
  python scripts/13_train.py --config C --arms A1 --device 1   # one arm per GPU in parallel
Outputs: <runs>/<arm>/weights/best.pt, last.pt
For RF-DETR use tools/export_rfdetr_coco.py and train with the rfdetr package (see README).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.common import base_parser, load  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--arms", nargs="*")
    p.add_argument("--device", help="override train.args.device")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs="*", default=[], help="extra key=value overrides for model.train()")
    args = p.parse_args()
    cfg = load(args)
    tr = cfg["train"]
    arms = args.arms or tr["arms"]
    targs = dict(tr["args"])
    if args.device is not None:
        targs["device"] = args.device
    for kv in args.extra:
        k, v = kv.split("=", 1)
        try:
            v = __import__("ast").literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        targs[k] = v
    for arm in arms:
        data = Path(cfg["paths"]["experiments"]) / arm / "data.yaml"
        if not data.exists():
            sys.exit(f"{data} not found; run 12_build_experiments.py first")
        kw = dict(data=str(data), project=cfg["paths"]["runs"], name=arm, exist_ok=True, **targs)
        print(f"[{arm}] train from {tr['weights']} with {kw}")
        if args.dry_run:
            continue
        from ultralytics import YOLO
        YOLO(tr["weights"]).train(**kw)


if __name__ == "__main__":
    main()
