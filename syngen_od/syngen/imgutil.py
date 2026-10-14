"""Model-free image checks and transforms: structure similarity, camera-like degradation, dHash."""
import datetime
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def stable_seed(*parts):
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def _gray(img, size):
    g = img.convert("L").resize(size, Image.BILINEAR)
    a = np.asarray(g, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-6)


def _grad_mag(a):
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    return np.hypot(gx, gy)


def _corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def structure_similarity(img_a, img_b, box=None, size=64, pad=0.1):
    """Correlation of contrast-normalised gradient magnitude between two images (optionally inside a box).
    Robust to brightness/colour/weather style changes, low when the object moved, vanished or was redrawn.
    box: (x1,y1,x2,y2) in img_a pixels; img_b is resized to img_a's size first."""
    if img_b.size != img_a.size:
        img_b = img_b.resize(img_a.size, Image.BILINEAR)
    if box is not None:
        w, h = img_a.size
        x1, y1, x2, y2 = box
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        crop = (max(0, int(x1 - px)), max(0, int(y1 - py)), min(w, int(x2 + px + 0.999)), min(h, int(y2 + py + 0.999)))
        if crop[2] - crop[0] < 4 or crop[3] - crop[1] < 4:
            return None
        img_a, img_b = img_a.crop(crop), img_b.crop(crop)
        ratio = (crop[2] - crop[0]) / (crop[3] - crop[1])
        sz = (size, max(8, int(size / ratio))) if ratio >= 1 else (max(8, int(size * ratio)), size)
    else:
        w, h = img_a.size
        sz = (256, max(8, int(256 * h / w)))
    return _corr(_grad_mag(_gray(img_a, sz)), _grad_mag(_gray(img_b, sz)))


def barrel_distort(img, k):
    """Fisheye-like barrel distortion (nearest-neighbour remap), as seen on wide-angle dome CCTV lenses.
    Division-model mapping (r_src = r_out / (1 + k*r_out^2)): sampled radius is always <= output radius,
    so it never samples outside the source image (no clipping/repeat-edge artifacts). k typically 0.1-0.4."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    nx, ny = (xx - cx) / cx, (yy - cy) / cy
    r2 = nx * nx + ny * ny
    f = 1 / (1 + k * r2)
    sx = np.clip(cx + nx * f * cx, 0, w - 1).astype(np.int32)
    sy = np.clip(cy + ny * f * cy, 0, h - 1).astype(np.int32)
    return Image.fromarray(a[sy, sx])


def vignette(img, strength):
    """Darken the corners, as cheap wide-angle lenses do. strength in ~[0,1]."""
    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    mask = np.clip(1 - strength * np.clip(r - 0.3, 0, None) ** 2, 0, 1)
    return Image.fromarray(np.clip(a * mask[..., None], 0, 255).astype(np.uint8))


_OSD_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def osd_overlay(img, rng, ir=False):
    """Burn in a fake NVR-style camera-id + timestamp, bottom-right -- a near-universal real-CCTV
    tell that a clean generated/COCO image otherwise never has. Deterministic text, not drawn by a
    generative model (which can't reliably render small legible text in a fixed position)."""
    img = img.copy()
    w, h = img.size
    fs = max(10, int(h * 0.028))
    try:
        font = ImageFont.truetype(_OSD_FONT_PATH, fs)
    except Exception:
        font = ImageFont.load_default()
    cam = f"CAM-{rng.integers(1, 64):02d}"
    t0 = datetime.datetime(2025, 1, 1) + datetime.timedelta(seconds=int(rng.integers(0, 3 * 365 * 24 * 3600)))
    text = f"{cam}  {t0:%Y-%m-%d %H:%M:%S}"
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(2, fs // 4)
    x, y = w - tw - 12 - pad, h - th - 12 - pad
    d.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad * 2], fill=(0, 0, 0))
    colour = (160, 160, 160) if ir else (235, 235, 235)
    d.text((x, y), text, fill=colour, font=font)
    return img


def degrade(img, cfg, seed):
    """Camera-like degradation: down/up-scale, blur, sensor noise, JPEG, + optional fisheye/vignette/OSD.
    cfg values are [min,max] ranges. The optional knobs (p_fisheye/p_vignette/p_osd) default to 0, so
    existing configs that don't set them behave exactly as before."""
    import io
    rng = np.random.default_rng(seed)
    w, h = img.size
    img = img.convert("RGB")
    if rng.random() < cfg.get("p_rescale", 0.7):
        s = rng.uniform(*cfg.get("rescale", [0.4, 1.0]))
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if rng.random() < cfg.get("p_blur", 0.3):
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(*cfg.get("blur_radius", [0.3, 1.2]))))
    if rng.random() < cfg.get("p_noise", 0.5):
        a = np.asarray(img, dtype=np.float32)
        a = a + rng.normal(0, rng.uniform(*cfg.get("noise_sigma", [2, 8])), a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < cfg.get("p_fisheye", 0.0):
        img = barrel_distort(img, rng.uniform(*cfg.get("fisheye_k", [0.1, 0.4])))
    if rng.random() < cfg.get("p_vignette", 0.0):
        img = vignette(img, rng.uniform(*cfg.get("vignette_strength", [0.3, 0.7])))
    if rng.random() < cfg.get("p_jpeg", 0.8):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(rng.uniform(*cfg.get("jpeg_quality", [35, 90]))))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    if rng.random() < cfg.get("p_osd", 0.0):
        img = osd_overlay(img, rng, ir=cfg.get("osd_ir", False))
    return img


def dhash(path, size=8):
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def hamming_matrix(a, b):
    """a (N,B) uint8 packed bits, b (M,B) -> (N,M) hamming distances."""
    x = np.bitwise_xor(a[:, None, :], b[None, :, :])
    return np.unpackbits(x, axis=2).sum(axis=2)


def is_degenerate(img, min_std=4.0):
    """Blank / near-uniform outputs (NSFW-filtered black frames, failed generations)."""
    return float(np.asarray(img.convert("L"), dtype=np.float32).std()) < min_std
