"""YOLO-format dataset I/O, following Ultralytics path conventions."""
import os
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def img2label_path(img_path):
    """Same rule as ultralytics.data.utils.img2label_paths: last /images/ -> /labels/, ext -> .txt."""
    p = str(img_path)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa not in p:
        raise ValueError(f"image path must contain '{sa}': {p}")
    return sb.join(p.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"


def list_images(src):
    """src: directory (recursive), .txt list file, or iterable of paths. Returns sorted absolute paths."""
    if isinstance(src, (list, tuple)):
        out = []
        for s in src:
            out.extend(list_images(s))
        return out
    p = Path(src)
    if p.is_dir():
        # abspath, not resolve(): images may be symlinks and the label lookup needs the link's own path
        return sorted(os.path.abspath(x) for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS)
    if p.suffix == ".txt":
        return [x if os.path.isabs(x) else os.path.abspath(p.parent / x) for x in read_list(p)]
    raise FileNotFoundError(f"not a directory or .txt list: {src}")


def read_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def write_list(path, items):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(f"{x}\n")


def read_labels(path, with_conf=False):
    """Returns (N,5) [cls,cx,cy,w,h] or (N,6) [cls,cx,cy,w,h,conf]; missing/empty file -> empty array."""
    ncol = 6 if with_conf else 5
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, ncol), dtype=np.float64)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.split()
            if not parts:
                continue
            if with_conf:
                if len(parts) < 6:
                    raise ValueError(f"{path}: expected 6 columns with confidence, got: {ln.strip()}")
                rows.append([float(x) for x in parts[:6]])
            else:
                rows.append([float(x) for x in parts[:5]])
    return np.asarray(rows, dtype=np.float64).reshape(-1, ncol)


def write_labels(path, arr, with_conf=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in np.asarray(arr).reshape(-1, 6 if with_conf else 5):
            line = f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}"
            if with_conf:
                line += f" {r[5]:.5f}"
            f.write(line + "\n")


def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (w, h)


def dataset_dirs(root, split=None):
    """<root>/images[/split], <root>/labels[/split]"""
    root = Path(root)
    img = root / "images" / split if split else root / "images"
    lbl = root / "labels" / split if split else root / "labels"
    return img, lbl
