"""JSONL records: generation metadata and detector outputs."""
import json
import os
from pathlib import Path

import numpy as np


def append_jsonl(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:  # tolerate a truncated last line after a crash
                    pass
    return out


def read_meta(task_dir):
    """All <task_dir>/meta*.jsonl (one per shard) -> {image_path: record}; last record wins."""
    out = {}
    for p in sorted(Path(task_dir).glob("meta*.jsonl")):
        for r in read_jsonl(p):
            out[r["image"]] = r
    return out


def det_record(image, w, h, dets):
    return {"image": image, "w": int(w), "h": int(h),
            "dets": [[int(d[0]), *(round(float(x), 2) for x in d[1:5]), round(float(d[5]), 5)] for d in dets]}


def read_dets(path_or_dir, prefix="teacher_dets"):
    """A dets jsonl file, or a directory holding <prefix>*.jsonl shards
    -> {image: (w, h, (K,6) array [cls,x1,y1,x2,y2,conf])}"""
    p = Path(path_or_dir)
    files = sorted(p.glob(f"{prefix}*.jsonl")) if p.is_dir() else [p]
    out = {}
    for f in files:
        for r in read_jsonl(f):
            out[r["image"]] = (r["w"], r["h"], np.asarray(r["dets"], dtype=np.float64).reshape(-1, 6))
    return out


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_np_default)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _np_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
