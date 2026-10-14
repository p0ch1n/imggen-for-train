#!/usr/bin/env python3
"""14 Run a model over every eval set and save predictions as YOLO txt with confidence.

  python scripts/14_predict.py --config C                       # prod + every arm with runs/<arm>/weights/best.pt
  python scripts/14_predict.py --config C --models prod A1
  python scripts/14_predict.py --config C --name rfdetr_ft --backend rfdetr --weights /path/ckpt.pth
Output: <results>/<model>/<set>/pred/*.txt   (cls cx cy w h conf, normalised)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.boxes import xyxy_to_xywhn  # noqa: E402
from syngen.common import Progress, base_parser, load  # noqa: E402
from syngen.detectors import build_detector  # noqa: E402
from syngen.yolo_io import list_images, write_labels  # noqa: E402


def discover_models(cfg):
    models = dict(cfg["eval"].get("models") or {})
    runs = Path(cfg["paths"]["runs"])
    for arm in cfg["experiments"]["arms"]:
        w = runs / arm / "weights" / "best.pt"
        if w.exists():
            models.setdefault(arm, str(w))
    return models


def main():
    p = base_parser(__doc__)
    p.add_argument("--models", nargs="*", help="names from eval.models / arms (default: all found)")
    p.add_argument("--name", help="ad-hoc model name (with --weights)")
    p.add_argument("--weights")
    p.add_argument("--backend", help="override eval.predict.backend (ultralytics | rfdetr | mock)")
    p.add_argument("--sets", nargs="*")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    cfg = load(args)
    models = {args.name: args.weights} if args.name else discover_models(cfg)
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}
    sets = {k: v for k, v in cfg["eval"]["sets"].items() if not args.sets or k in args.sets}
    for name, weights in models.items():
        pcfg = dict(cfg["eval"]["predict"], weights=weights)
        if args.backend:
            pcfg["backend"] = args.backend
        det = None
        for sname, sdir in sets.items():
            if not Path(sdir).exists():
                print(f"skip set {sname}: {sdir} not found")
                continue
            out = Path(cfg["paths"]["results"]) / name / sname / "pred"
            if out.exists() and not args.overwrite and any(out.iterdir()):
                print(f"[{name}/{sname}] predictions exist, skip (--overwrite to redo)")
                continue
            det = det or build_detector(pcfg, cfg["classes"])
            imgs = list_images(sdir)
            prog = Progress(len(imgs), f"{name}/{sname}", every=200)
            for ip, (w, h, d) in zip(imgs, det.detect(imgs)):
                rows = np.column_stack([d[:, 0], xyxy_to_xywhn(d[:, 1:5], w, h), d[:, 5]]) if len(d) \
                    else np.zeros((0, 6))
                write_labels(out / (Path(ip).stem + ".txt"), rows, with_conf=True)
                prog.step()


if __name__ == "__main__":
    main()
