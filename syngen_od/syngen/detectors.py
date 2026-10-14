"""Detector backends shared by teacher filtering, pseudo-labelling and evaluation-time prediction.

Every backend returns, per image, (w, h, dets) with dets = (K,6) [cls, x1, y1, x2, y2, conf] in pixels, where
cls is the index into the *project* class list (config `classes`). Native classes are mapped by name; classes
not in the project list are dropped.
"""
import os

import numpy as np
from PIL import Image

from .boxes import labels_to_pixel
from .imgutil import stable_seed
from .yolo_io import img2label_path, read_labels


def _name_map(native_names, project_classes, overrides=None):
    """native_names: {native_id: name}. overrides: {project_name: native_name_or_id}. -> {native_id: project_idx}"""
    overrides = overrides or {}
    by_name = {str(n).lower(): i for i, n in native_names.items()}
    out = {}
    for pidx, pname in enumerate(project_classes):
        key = overrides.get(pname, pname)
        if isinstance(key, int):
            out[key] = pidx
        elif str(key).lower() in by_name:
            out[by_name[str(key).lower()]] = pidx
    return out


def _batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


class UltralyticsDetector:
    """Any Ultralytics model: YOLOv8-12, RT-DETR, ... (also your current production YOLOv12 weights)."""

    def __init__(self, cfg, classes):
        from ultralytics import YOLO
        self.model = YOLO(cfg["weights"])
        self.cfg = cfg
        self.map = _name_map(self.model.names, classes, cfg.get("class_map"))
        missing = [c for c in classes if c not in [classes[v] for v in self.map.values()]]
        if missing:
            print(f"[ultralytics] WARNING: model has no class for: {missing}")

    def detect(self, paths):
        c = self.cfg
        for batch in _batched(paths, c.get("batch", 16)):
            results = self.model.predict(batch, conf=c.get("conf", 0.001), iou=c.get("iou", 0.7),
                                         imgsz=c.get("imgsz", 640), device=c.get("device"), half=c.get("half", False),
                                         max_det=c.get("max_det", 300), verbose=False, augment=c.get("tta", False))
            for r in results:
                h, w = r.orig_shape
                b = r.boxes
                xyxy = b.xyxy.cpu().numpy()
                conf = b.conf.cpu().numpy()
                cls = b.cls.cpu().numpy().astype(int)
                keep = np.array([k in self.map for k in cls], dtype=bool)
                dets = np.column_stack([[self.map[k] for k in cls[keep]], xyxy[keep], conf[keep]]) \
                    if keep.any() else np.zeros((0, 6))
                yield w, h, dets.reshape(-1, 6)


class RFDETRDetector:
    """roboflow `rfdetr` package. variant: RFDETRNano/Small/Medium/Base/Large (or other class in rfdetr)."""

    def __init__(self, cfg, classes):
        import rfdetr
        cls = getattr(rfdetr, cfg.get("variant", "RFDETRLarge"))
        kw = {"pretrain_weights": cfg["weights"]} if cfg.get("weights") else {}
        if cfg.get("resolution"):
            kw["resolution"] = cfg["resolution"]
        self.model = cls(**kw)
        if cfg.get("optimize", False):
            self.model.optimize_for_inference()
        self.cfg = cfg
        names = cfg.get("id_to_name")  # explicit override, e.g. for fine-tuned checkpoints
        if names is None:
            cn = getattr(self.model, "class_names", None)
            if isinstance(cn, dict):
                names = cn
            elif isinstance(cn, (list, tuple)):
                names = {i + int(cfg.get("id_offset", 0)): n for i, n in enumerate(cn)}
            else:
                from rfdetr.util.coco_classes import COCO_CLASSES
                names = COCO_CLASSES
        self.map = _name_map({int(k): v for k, v in names.items()}, classes, cfg.get("class_map"))

    def detect(self, paths):
        thr = self.cfg.get("conf", 0.01)
        for p in paths:
            img = Image.open(p).convert("RGB")
            d = self.model.predict(img, threshold=thr)
            cls = np.asarray(d.class_id).astype(int)
            keep = np.array([k in self.map for k in cls], dtype=bool)
            if keep.any():
                dets = np.column_stack([[self.map[k] for k in cls[keep]], np.asarray(d.xyxy)[keep],
                                        np.asarray(d.confidence)[keep]])
            else:
                dets = np.zeros((0, 6))
            yield img.width, img.height, dets.reshape(-1, 6)


class GroundingDinoDetector:
    """Open-vocabulary teacher via transformers (IDEA-Research/grounding-dino-base or -tiny, local dir)."""

    def __init__(self, cfg, classes):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")
        self.processor = AutoProcessor.from_pretrained(cfg["model_path"], local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg["model_path"], local_files_only=True).to(self.device).eval()
        # prompts: {project_class: "text phrase"}; default phrase = class name
        prompts = cfg.get("prompts") or {}
        self.phrases = [(i, prompts.get(c, c)) for i, c in enumerate(classes) if c in prompts or not prompts]
        self.text = ". ".join(p for _, p in self.phrases) + "."

    def _label_to_cls(self, label):
        label = str(label).lower()
        for i, p in self.phrases:
            if p.lower() in label or label in p.lower():
                return i
        return None

    def detect(self, paths):
        c = self.cfg
        for p in paths:
            img = Image.open(p).convert("RGB")
            inputs = self.processor(images=img, text=self.text, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            kw = dict(text_threshold=c.get("text_threshold", 0.25), target_sizes=[img.size[::-1]])
            try:
                res = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=c.get("conf", 0.25), **kw)[0]
            except TypeError:  # older transformers
                res = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=c.get("conf", 0.25), **kw)[0]
            labels = res.get("text_labels", res.get("labels"))
            rows = []
            for box, score, lab in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels):
                k = self._label_to_cls(lab)
                if k is not None:
                    rows.append([k, *box.tolist(), float(score)])
            yield img.width, img.height, np.asarray(rows, dtype=np.float64).reshape(-1, 6)


class MockDetector:
    """Oracle-with-noise: reads the YOLO label next to each image (if any), jitters boxes, randomly drops some and
    adds phantom boxes. Only for smoke-testing the pipeline."""

    def __init__(self, cfg, classes):
        self.cfg = cfg
        self.nc = len(classes)

    def detect(self, paths):
        c = self.cfg
        for p in paths:
            rng = np.random.default_rng(stable_seed(c.get("seed", 0), os.path.basename(p)))
            with Image.open(p) as im:
                w, h = im.size
            try:
                lab = read_labels(img2label_path(p))
            except ValueError:
                lab = np.zeros((0, 5))
            # the mock edit/insert backends mark "lost" objects as uniform patches; nothing to model here
            g = labels_to_pixel(lab, w, h)
            rows = []
            for r in g:
                if rng.random() < c.get("drop_prob", 0.05):
                    continue
                j = rng.normal(0, c.get("jitter_px", 2.0), 4)
                rows.append([r[0], *(r[1:5] + j), rng.uniform(0.5, 0.98)])
            if rng.random() < c.get("phantom_prob", 0.1):
                x, y = rng.uniform(0, w * 0.8), rng.uniform(0, h * 0.8)
                rows.append([rng.integers(0, self.nc), x, y, x + w * 0.1, y + h * 0.2, rng.uniform(0.2, 0.9)])
            for _ in range(c.get("low_conf_noise", 3)):  # low-score clutter like a real detector at conf=0.001
                x, y = rng.uniform(0, w * 0.9), rng.uniform(0, h * 0.9)
                rows.append([rng.integers(0, self.nc), x, y, x + w * 0.05, y + h * 0.08, rng.uniform(0.001, 0.1)])
            yield w, h, np.asarray(rows, dtype=np.float64).reshape(-1, 6)


_REGISTRY = {"ultralytics": UltralyticsDetector, "rfdetr": RFDETRDetector,
             "grounding_dino": GroundingDinoDetector, "mock": MockDetector}


def build_detector(cfg, classes):
    return _REGISTRY[cfg.get("backend", "ultralytics")](cfg, classes)
