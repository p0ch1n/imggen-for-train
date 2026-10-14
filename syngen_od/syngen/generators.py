"""Image generation backends.

diffusers backends load from a *local* directory (offline env); mock backends let the whole pipeline run on CPU
without any model so scripts can be smoke-tested.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .imgutil import stable_seed

LIGHTNING_SCHEDULER = {  # recommended by lightx2v/Qwen-Image-Lightning
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def _snap(x, m=16):
    return max(m, int(round(x / m)) * m)


class _DiffusersBase:
    def __init__(self, cfg):
        import torch
        from diffusers import DiffusionPipeline

        self.torch = torch
        self.cfg = cfg
        dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
        if cfg.get("num_threads"):
            torch.set_num_threads(int(cfg["num_threads"]))
        kw = {"variant": cfg["variant"]} if cfg.get("variant") else {}  # e.g. fp16 weights, upcast to dtype
        self.pipe = self._load(DiffusionPipeline, cfg["model_path"], torch_dtype=dtype, local_files_only=True, **kw)
        if cfg.get("lora_path"):
            self.pipe.load_lora_weights(cfg["lora_path"], weight_name=cfg.get("lora_weight_name"))
            if cfg.get("lora_fuse", True):
                self.pipe.fuse_lora()
        if cfg.get("lightning_scheduler"):
            from diffusers import FlowMatchEulerDiscreteScheduler
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
        offload = cfg.get("offload", "none")
        if offload == "model":
            self.pipe.enable_model_cpu_offload()
        elif offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(cfg.get("device", "cuda"))
        self.pipe.set_progress_bar_config(disable=True)
        self.class_name = type(self.pipe).__name__

    def _load(self, default_cls, path, **kw):
        return default_cls.from_pretrained(path, **kw)

    def _fit(self, image):
        """Optionally shrink the input (max_side) before editing; snapped to a multiple of 8."""
        m = self.cfg.get("max_side")
        if not m or max(image.size) <= m:
            return image
        f = m / max(image.size)
        return image.resize((max(64, int(image.width * f) // 8 * 8), max(64, int(image.height * f) // 8 * 8)),
                            Image.LANCZOS)

    def _gen(self, seed):
        return self.torch.Generator(device="cpu").manual_seed(int(seed))

    def _common(self):
        c = self.cfg
        kw = {"num_inference_steps": c.get("steps", 50)}
        if "true_cfg_scale" in c:
            kw["true_cfg_scale"] = c["true_cfg_scale"]
        if "guidance_scale" in c:
            kw["guidance_scale"] = c["guidance_scale"]
        kw.update(c.get("extra_kwargs", {}))
        return kw


class DiffusersT2I(_DiffusersBase):
    def generate(self, prompt, negative_prompt, width, height, seed):
        out = self.pipe(prompt=prompt, negative_prompt=negative_prompt or " ", width=_snap(width),
                        height=_snap(height), generator=self._gen(seed), **self._common())
        return out.images[0]


class DiffusersEdit(_DiffusersBase):
    """Qwen-Image-Edit (QwenImageEditPipeline) or Qwen-Image-Edit-2509/2511 (QwenImageEditPlusPipeline)."""

    def edit(self, image, prompt, seed, negative_prompt=" ", hint_boxes=None):
        small = self._fit(image)
        img_arg = [small] if "Plus" in self.class_name else small
        out = self.pipe(image=img_arg, prompt=prompt, negative_prompt=negative_prompt or " ",
                        generator=self._gen(seed), **self._common())
        res = out.images[0]
        # The pipeline resizes to ~1MP; bring it back so normalised YOLO boxes stay pixel-aligned.
        return res.resize(image.size, Image.LANCZOS) if res.size != image.size else res


class DiffusersInpaintRemove(_DiffusersBase):
    """Object removal as an `edit` backend: inpaint the hinted boxes (dilated) with a background prompt.
    Works with dedicated inpainting checkpoints or, via AutoPipelineForInpainting, plain/turbo SD checkpoints."""

    def _load(self, default_cls, path, **kw):
        from diffusers import AutoPipelineForInpainting
        return AutoPipelineForInpainting.from_pretrained(path, **kw)

    def edit(self, image, prompt, seed, negative_prompt=" ", hint_boxes=None):
        from PIL import ImageDraw as _D
        small = self._fit(image)
        f = small.width / image.width
        mask = Image.new("L", small.size, 0)
        d = _D.Draw(mask)
        grow = self.cfg.get("mask_dilate", 0.15)
        for x1, y1, x2, y2 in (hint_boxes if hint_boxes is not None else []):
            px, py = (x2 - x1) * grow + 4, (y2 - y1) * grow + 4
            d.rectangle([(x1 - px) * f, (y1 - py) * f, (x2 + px) * f, (y2 + py) * f], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(self.cfg.get("mask_blur", 4)))
        kw = self._common()
        if "strength" in self.cfg:
            kw["strength"] = self.cfg["strength"]
        out = self.pipe(prompt=self.cfg.get("fill_prompt", prompt), image=small, mask_image=mask,
                        width=small.width, height=small.height, generator=self._gen(seed), **kw)
        res = out.images[0]
        if res.size != small.size:
            res = res.resize(small.size, Image.LANCZOS)
        # paste back only the masked area so pixels outside the mask stay exactly the original
        res = Image.composite(res, small, mask)
        return res.resize(image.size, Image.LANCZOS) if res.size != image.size else res


class DiffusersInpaint(_DiffusersBase):
    """Mask-based inpainting, e.g. FLUX.1-Fill-dev (FluxFillPipeline). Works on a crop around the mask."""

    def inpaint(self, image, mask, prompt, seed):
        w, h = image.size
        out = self.pipe(prompt=prompt, image=image, mask_image=mask, width=_snap(w), height=_snap(h),
                        generator=self._gen(seed), **self._common())
        res = out.images[0]
        return res.resize(image.size, Image.LANCZOS) if res.size != image.size else res


# ---------------------------------------------------------------- mock backends (CPU, for smoke tests)

class MockT2I:
    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self, prompt, negative_prompt, width, height, seed):
        rng = np.random.default_rng(seed)
        base = rng.integers(40, 200, 3)
        a = np.clip(base + rng.normal(0, 25, (height, width, 3)), 0, 255).astype(np.uint8)
        img = Image.fromarray(a).filter(ImageFilter.GaussianBlur(2))
        d = ImageDraw.Draw(img)
        for _ in range(rng.integers(3, 10)):  # "bumps / rocks"
            cx, cy = rng.integers(0, width), rng.integers(0, height)
            r = rng.integers(5, max(6, width // 10))
            d.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=tuple(int(x) for x in rng.integers(0, 255, 3)))
        return img


class MockEdit:
    """Style edits -> colour/brightness change; remove edits -> blur the hinted boxes away."""

    def __init__(self, cfg):
        self.cfg = cfg

    def edit(self, image, prompt, seed, negative_prompt=" ", hint_boxes=None):
        rng = np.random.default_rng(seed)
        img = image.convert("RGB")
        if hint_boxes is not None and "remove" in prompt.lower():
            blurred = img.filter(ImageFilter.GaussianBlur(25))
            for x1, y1, x2, y2 in hint_boxes:
                box = tuple(int(v) for v in (x1, y1, x2, y2))
                img.paste(blurred.crop(box), box[:2])
            return img
        a = np.asarray(img, dtype=np.float32)
        a = a * rng.uniform(0.3, 0.8) + rng.normal(0, 4, a.shape) + rng.uniform(-10, 10, 3)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


class MockInpaint:
    def __init__(self, cfg):
        self.cfg = cfg

    def inpaint(self, image, mask, prompt, seed):
        rng = np.random.default_rng(seed)
        img = image.convert("RGB").copy()
        bbox = mask.getbbox()
        if bbox:
            x1, y1, x2, y2 = bbox
            mx, my = (x2 - x1) // 6, (y2 - y1) // 10  # object slightly smaller than mask, like a real model
            ImageDraw.Draw(img).rectangle([x1 + mx, y1 + my, x2 - mx, y2 - my],
                                          fill=tuple(int(x) for x in rng.integers(0, 255, 3)))
        return img


_REGISTRY = {
    ("t2i", "diffusers"): DiffusersT2I, ("t2i", "mock"): MockT2I,
    ("edit", "diffusers"): DiffusersEdit, ("edit", "mock"): MockEdit,
    ("edit", "diffusers_inpaint"): DiffusersInpaintRemove,
    ("inpaint", "diffusers"): DiffusersInpaint, ("inpaint", "mock"): MockInpaint,
}


def build_generator(kind, cfg):
    backend = cfg.get("backend", "diffusers")
    try:
        cls = _REGISTRY[(kind, backend)]
    except KeyError:
        raise ValueError(f"unknown {kind} backend: {backend}")
    return cls(cfg)


def item_seed(base_seed, key):
    return stable_seed(base_seed, key) % (2 ** 31)
