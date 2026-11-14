import io
import zlib

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from . import config as C


def rng_for(key, ctype, s):
    return np.random.default_rng(zlib.crc32(f"{key}|{ctype}|{s}".encode()))


def _apply(pil, ctype, v, rng):
    if ctype == "gaussian_blur":
        return pil.filter(ImageFilter.GaussianBlur(v))
    if ctype in ("brightness", "darkening"):
        return ImageEnhance.Brightness(pil).enhance(v)
    if ctype == "contrast":
        return ImageEnhance.Contrast(pil).enhance(v)
    if ctype == "gaussian_noise":
        a = np.asarray(pil).astype(np.float32) + rng.normal(0, v, (pil.size[1], pil.size[0], 3)).astype(np.float32)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if ctype == "jpeg":
        b = io.BytesIO()
        pil.save(b, "JPEG", quality=int(v))
        b.seek(0)
        return Image.open(b).convert("RGB")
    raise ValueError(ctype)


def _p(ctype, s):
    return C.PARAMS[ctype][s - 1]


def _compound(pil, ctype, s, rng):
    if ctype == "compound_blur_dark_jpeg":
        for t in ["darkening", "gaussian_blur", "jpeg"]:
            pil = _apply(pil, t, _p(t, s), rng)
        return pil
    if ctype == "compound_noise_contrast_jpeg":
        for t in ["contrast", "gaussian_noise", "jpeg"]:
            pil = _apply(pil, t, _p(t, s), rng)
        return pil
    if ctype == "compound_field":
        lv = lambda: int(rng.integers(1, s + 1))
        if rng.random() < 0.7:
            t = "brightness" if rng.random() < 0.5 else "darkening"
            pil = _apply(pil, t, _p(t, lv()), rng)
        if rng.random() < 0.5:
            pil = _apply(pil, "contrast", _p("contrast", lv()), rng)
        if rng.random() < 0.7:
            pil = _apply(pil, "gaussian_blur", _p("gaussian_blur", lv()), rng)
        if rng.random() < 0.5:
            pil = _apply(pil, "gaussian_noise", _p("gaussian_noise", lv()), rng)
        return _apply(pil, "jpeg", _p("jpeg", lv()), rng)
    raise ValueError(ctype)


def corrupt(img, ctype, s, key):
    if ctype in (None, "clean") or s == 0:
        return img
    rng = rng_for(key, ctype, s)
    pil = Image.fromarray(np.asarray(img))
    out = _compound(pil, ctype, s, rng) if ctype.startswith("compound") else _apply(pil, ctype, _p(ctype, s), rng)
    return np.asarray(out, dtype=np.uint8)


def corrupt_native(img, ctype, s, key):
    rng = rng_for(key, "native_" + ctype, s)
    return np.asarray(_apply(Image.fromarray(img), ctype, C.NATIVE_PARAMS[ctype][s - 1], rng), dtype=np.uint8)


def conditions(include_compound=True):
    out = [("clean", 0)]
    for c in C.CORRUPTIONS:
        for s in C.SEVERITIES:
            out.append((c, s))
    if include_compound:
        for c in C.COMPOUNDS:
            for s in C.SEVERITIES:
                out.append((c, s))
    return out
