#!/usr/bin/env python3
"""00 Pre-flight check for the isolated environment: packages, GPU/VRAM, model files, data paths, disk space.

  python scripts/00_check_env.py --config configs/pipeline.yaml
"""
import importlib
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.common import base_parser, load  # noqa: E402

OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "
status = {"fail": 0}


def report(level, msg):
    if level == FAIL:
        status["fail"] += 1
    print(f"[{level}] {msg}")


def check_pkg(name, needed_for):
    try:
        m = importlib.import_module(name)
        report(OK, f"{name} {getattr(m, '__version__', '')}")
        return m
    except Exception as e:  # noqa: BLE001
        report(WARN, f"{name} not importable ({e.__class__.__name__}) -- needed for {needed_for}")


def main():
    p = base_parser(__doc__)
    p.add_argument("--skip-models", action="store_true")
    args = p.parse_args()
    cfg = load(args)
    print("== packages")
    for name, why in [("numpy", "everything"), ("PIL", "everything"), ("yaml", "everything")]:
        if not check_pkg(name, why):
            report(FAIL, f"{name} is required")
    gen_backends = {cfg["generation"][k].get("backend") for k in ("t2i", "edit", "inpaint")}
    torch = None
    if "diffusers" in gen_backends:
        torch = check_pkg("torch", "generation")
        check_pkg("diffusers", "generation (needs a version with QwenImage pipelines)")
        check_pkg("transformers", "generation text encoders")
        check_pkg("accelerate", "cpu offload")
        try:
            import diffusers
            for cls in ("QwenImagePipeline", "QwenImageEditPipeline", "QwenImageEditPlusPipeline", "FluxFillPipeline"):
                report(OK if hasattr(diffusers, cls) else WARN, f"diffusers.{cls}")
        except ImportError:
            pass
    tb = cfg["teacher"]["backend"]
    if tb == "rfdetr":
        check_pkg("rfdetr", "teacher")
    elif tb in ("ultralytics", "grounding_dino"):
        check_pkg("ultralytics" if tb == "ultralytics" else "transformers", "teacher")
    check_pkg("ultralytics", "training/eval (13-15)")

    print("== GPU")
    try:
        import torch as _t
        if _t.cuda.is_available():
            for i in range(_t.cuda.device_count()):
                pr = _t.cuda.get_device_properties(i)
                gb = pr.total_memory / 1024 ** 3
                lvl = OK if gb >= 40 else WARN
                report(lvl, f"cuda:{i} {pr.name} {gb:.0f} GB" + ("" if gb >= 40 else
                       " (<40GB: set generation.*.offload: model for Qwen-Image(-Edit))"))
        else:
            report(WARN, "no CUDA device visible (generation/training will not work)")
    except ImportError:
        report(WARN, "torch not installed")

    print("== offline settings")
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        report(OK if os.environ.get(k) == "1" else WARN, f"{k}={os.environ.get(k)} (set to 1 in an isolated env)")

    if not args.skip_models:
        print("== model files")
        for k in ("t2i", "edit", "inpaint"):
            g = cfg["generation"][k]
            if g.get("backend") == "diffusers":
                mi = Path(g["model_path"]) / "model_index.json"
                report(OK if mi.exists() else (FAIL if k != "inpaint" else WARN), f"{k}: {mi}")
                if g.get("lora_path"):
                    report(OK if Path(g["lora_path"]).exists() else FAIL, f"{k} lora: {g['lora_path']}")
        t = cfg["teacher"]
        tp = t.get("weights") or t.get("model_path")
        if tp and t["backend"] != "mock":
            report(OK if Path(tp).exists() else FAIL, f"teacher: {tp}")
        w = cfg["train"]["weights"]
        report(OK if Path(w).exists() else FAIL, f"train start weights: {w}")

    print("== data")
    P = cfg["paths"]
    for k in ("coco_train_ann", "coco_val_ann", "coco_train_images", "coco_val_images"):
        report(OK if Path(P[k]).exists() else FAIL, f"{k}: {P[k]}")
    for k in ("deploy_eval", "deploy_bg"):
        d = Path(P[k])
        n = sum(1 for _ in d.glob("*")) if d.exists() else 0
        lvl = OK if n >= 100 else (WARN if n else FAIL)
        report(lvl, f"{k}: {d} ({n} files){'  <- needs >=200 real frames for a meaningful study' if n < 200 else ''}")
        if k == "deploy_eval" and d.exists():
            lbl = d.parent / "labels"
            report(OK if lbl.exists() else FAIL, f"deploy_eval labels: {lbl}")
    d = Path(P["deploy_unlabeled"])
    report(OK if d.exists() else WARN, f"deploy_unlabeled (A4/insert only): {d}")

    print("== disk")
    root = Path(cfg["root"])
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free / 1024 ** 3
    report(OK if free > 100 else WARN, f"free space at {root}: {free:.0f} GB")

    print(f"\n{'ALL REQUIRED CHECKS PASSED' if status['fail'] == 0 else str(status['fail']) + ' FAILED CHECK(S)'}")
    sys.exit(1 if status["fail"] else 0)


if __name__ == "__main__":
    main()
