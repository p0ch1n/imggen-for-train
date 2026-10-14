import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.boxes import greedy_match, iou_matrix, xywhn_to_xyxy, xyxy_to_xywhn  # noqa: E402
from syngen.config import load_config  # noqa: E402
from syngen.imgutil import degrade, structure_similarity  # noqa: E402
from syngen.metrics import average_precision, operating_point  # noqa: E402
from syngen.yolo_io import img2label_path, read_labels, write_labels  # noqa: E402


def test_img2label_path():
    assert img2label_path(f"{os.sep}a{os.sep}images{os.sep}images{os.sep}x.y.jpg") == \
        f"{os.sep}a{os.sep}images{os.sep}labels{os.sep}x.y.txt"


def test_box_roundtrip():
    xyxy = np.array([[10, 20, 50, 100.0]])
    np.testing.assert_allclose(xywhn_to_xyxy(xyxy_to_xywhn(xyxy, 200, 150), 200, 150), xyxy)
    assert iou_matrix(xyxy, xyxy)[0, 0] == pytest.approx(1.0)


def test_greedy_match_prefers_high_conf():
    gt = np.array([[0, 0, 10, 10.0]])
    dets = np.array([[0, 0, 10, 10.0], [1, 1, 10, 10.0]])
    det_gt, hit = greedy_match(gt, dets, np.array([0.3, 0.9]), 0.5)
    assert list(det_gt) == [-1, 0] and hit.all()


def test_labels_io(tmp_path):
    p = tmp_path / "labels" / "a.txt"
    write_labels(p, np.array([[1, 0.5, 0.5, 0.2, 0.3, 0.9]]), with_conf=True)
    np.testing.assert_allclose(read_labels(p, with_conf=True), [[1, 0.5, 0.5, 0.2, 0.3, 0.9]])
    assert read_labels(tmp_path / "missing.txt").shape == (0, 5)


def _random_case(rng, n_img=40, nc=3):
    gt, pred = {}, {}
    for i in range(n_img):
        g = []
        for _ in range(rng.integers(0, 6)):
            x, y = rng.uniform(0, 500, 2)
            w, h = rng.uniform(5, 150, 2)
            g.append([rng.integers(0, nc), x, y, x + w, y + h])
        g = np.array(g).reshape(-1, 5)
        p = []
        for r in g:  # noisy true positives
            if rng.random() < 0.8:
                b = r[1:5] + rng.normal(0, 6, 4)
                b = [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]) + 1, max(b[1], b[3]) + 1]  # well-formed
                p.append([r[0], *b, rng.random()])
        for _ in range(rng.integers(0, 4)):  # false positives
            x, y = rng.uniform(0, 500, 2)
            p.append([rng.integers(0, nc), x, y, x + 40, y + 60, rng.random() * 0.7])
        gt[f"im{i}"], pred[f"im{i}"] = g, np.array(p).reshape(-1, 6)
    return gt, pred


def test_ap_matches_pycocotools():
    pycoco = pytest.importorskip("pycocotools.cocoeval")
    from pycocotools.coco import COCO
    rng = np.random.default_rng(0)
    nc = 3
    gt, pred = _random_case(rng, nc=nc)
    images, anns, dets = [], [], []
    for i, k in enumerate(sorted(gt)):
        images.append({"id": i, "width": 700, "height": 700})
        for r in gt[k]:
            w, h = r[3] - r[1], r[4] - r[2]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(r[0]) + 1,
                         "bbox": [r[1], r[2], w, h], "area": w * h, "iscrowd": 0})
        for r in pred[k]:
            dets.append({"image_id": i, "category_id": int(r[0]) + 1,
                         "bbox": [r[1], r[2], r[3] - r[1], r[4] - r[2]], "score": float(r[5])})
    cg = COCO()
    cg.dataset = {"images": images, "annotations": anns, "categories": [{"id": c + 1} for c in range(nc)]}
    cg.createIndex()
    cd = cg.loadRes(dets)
    E = pycoco.COCOeval(cg, cd, "bbox")
    E.evaluate()
    E.accumulate()
    E.summarize()
    ours = average_precision(gt, pred, nc)
    assert ours["map50_95"] == pytest.approx(E.stats[0], abs=1e-6)
    assert ours["map50"] == pytest.approx(E.stats[1], abs=1e-6)


def test_operating_point_counts():
    gt = {"a": np.array([[0, 0, 0, 10, 10.0]]), "b": np.zeros((0, 5))}
    pred = {"a": np.array([[0, 0, 0, 10, 10, 0.9], [0, 50, 50, 60, 60, 0.9], [0, 0, 0, 10, 10, 0.1]]),
            "b": np.array([[0, 5, 5, 20, 20, 0.5]])}
    op = operating_point(gt, pred, 1, conf_thr=0.25)
    c = op["per_class"][0]
    assert (c["tp"], c["fp"], c["fn"]) == (1, 2, 0)
    assert op["fp_per_image_all"] == 1.0
    assert op["recall_by_size"][0]["small"] == 1.0


def _scene(objects, size=(320, 240), fill=(200, 60, 60)):
    rng = np.random.default_rng(1)
    img = Image.fromarray(np.clip(rng.normal(120, 15, (size[1], size[0], 3)), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for b in objects:
        d.rectangle(b, fill=fill, outline=(0, 0, 0), width=3)
    return img


def test_structure_similarity_style_vs_moved():
    box = (100, 60, 160, 180)
    src = _scene([box])
    # style change: darker, tinted, noisy -> structure kept
    a = np.asarray(src, dtype=np.float32) * 0.35 + np.array([0, 0, 25])
    a += np.random.default_rng(2).normal(0, 3, a.shape)
    styled = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    moved = _scene([(200, 60, 260, 180)])
    gone = _scene([])
    s_style = structure_similarity(src, styled, box=box)
    s_moved = structure_similarity(src, moved, box=box)
    s_gone = structure_similarity(src, gone, box=box)
    assert s_style > 0.6
    assert s_moved < 0.3 and s_gone < 0.3
    assert structure_similarity(src, degrade(src, {}, 0)) > 0.5


def test_config_interpolation(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("root: /r\nclasses: [a, b]\npaths: {x: '${root}/x'}\nt: {out: '${paths.x}/t', cls: '${classes}'}\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\nroot: /other\n")
    c = load_config(child)
    assert c["t"]["out"] == "/other/x/t"
    assert c["t"]["cls"] == ["a", "b"]
