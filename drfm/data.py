import os
import glob
import json
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config as C
from .corruptions import corrupt_native

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(path).convert("L"))
    return img


def to_square(img, size=C.SHORT, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    s = size / min(h, w)
    nw, nh = max(size, int(round(w * s))), max(size, int(round(h * s)))
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    return np.ascontiguousarray(r[y0:y0 + size, x0:x0 + size])


def center_crop(arr, size=C.IMG):
    h, w = arr.shape[-3], arr.shape[-2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return arr[..., y0:y0 + size, x0:x0 + size, :]


def quality_metrics(gray):
    g = gray.astype(np.float64)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    ten = float(np.mean(gx ** 2 + gy ** 2))
    bren = float(np.mean((g[:, 2:] - g[:, :-2]) ** 2))
    mask = g > 10
    fov = float(mask.mean())
    vals = g[mask] if mask.any() else g.ravel()
    return {"lap_var": lap, "tenengrad": ten, "brenner": bren, "fov": fov,
            "brightness": float(vals.mean() / 255.0), "rms_contrast": float(vals.std() / 255.0)}


def v1_gray(path):
    return cv2.resize(read_gray(path), (256, 256))


def load_aptos_frame():
    root = C.find_aptos()
    df = pd.read_csv(os.path.join(root, "train.csv"))
    df["path"] = df["id_code"].apply(lambda c: os.path.join(root, "train_images", c + ".png"))
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    if C.QUICK and len(df) > C.QUICK_N:
        df, _ = train_test_split(df, train_size=C.QUICK_N, stratify=df["diagnosis"], random_state=0)
        df = df.sort_index().reset_index(drop=True)
    return df


def _strat(y):
    return y if y.value_counts().min() >= 2 else None


def stratified_split(frame, seed=C.SPLIT_SEED):
    tr, tmp = train_test_split(frame, test_size=0.30, stratify=_strat(frame["diagnosis"]), random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=_strat(tmp["diagnosis"]), random_state=seed)
    return list(tr.index), list(va.index), list(te.index)


def _aptos_worker(args):
    path, native = args
    q = quality_metrics(v1_gray(path))
    rgb = read_rgb(path)
    q["native_h"], q["native_w"] = int(rgb.shape[0]), int(rgb.shape[1])
    sq = to_square(rgb)
    nat = None
    if native:
        nat = {}
        for c in C.NATIVE_CORRUPTIONS:
            for s in C.SEVERITIES:
                nat[(c, s)] = to_square(corrupt_native(rgb, c, s, os.path.basename(path)))
    return q, sq, nat


def build_aptos_cache():
    C.ensure_dirs()
    meta_path = os.path.join(C.CACHE, "aptos_meta.csv")
    if os.path.exists(meta_path) and os.path.exists(os.path.join(C.CACHE, "splits.json")):
        print("APTOS cache exists")
        return
    df = load_aptos_frame()
    full_tr, full_va, full_te = stratified_split(df)
    test_set = set(full_te)
    n = len(df)
    arr = np.lib.format.open_memmap(os.path.join(C.CACHE, "aptos_256.npy"), mode="w+", dtype=np.uint8, shape=(n, C.SHORT, C.SHORT, 3))
    nat_arrs = {}
    for c in C.NATIVE_CORRUPTIONS:
        for s in C.SEVERITIES:
            nat_arrs[(c, s)] = np.lib.format.open_memmap(os.path.join(C.CACHE, f"native_{c}_{s}.npy"), mode="w+", dtype=np.uint8, shape=(len(full_te), C.SHORT, C.SHORT, 3))
    te_pos = {i: k for k, i in enumerate(full_te)}
    rows = []
    jobs = [(p, i in test_set) for i, p in enumerate(df["path"])]
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        for i, (q, sq, nat) in enumerate(pool.imap(_aptos_worker, jobs, chunksize=4)):
            arr[i] = sq
            if nat is not None:
                for key, v in nat.items():
                    nat_arrs[key][te_pos[i]] = v
            rows.append(q)
            if (i + 1) % 250 == 0:
                print(f"  cached {i + 1}/{n}", flush=True)
    arr.flush()
    for v in nat_arrs.values():
        v.flush()
    meta = pd.concat([df[["id_code", "diagnosis"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    thr = float(meta["lap_var"].median())
    meta["quality"] = np.where(meta["lap_var"] >= thr, "high", "low")
    q33, q67 = meta["lap_var"].quantile([1 / 3, 2 / 3]).tolist()
    meta["tertile"] = np.where(meta["lap_var"] <= q33, "low", np.where(meta["lap_var"] >= q67, "high", "mid"))
    meta.to_csv(meta_path, index=False)
    ints = lambda parts: [[int(i) for i in p] for p in parts]
    splits = {"full": ints([full_tr, full_va, full_te]), "threshold": thr, "tertile_thresholds": [float(q33), float(q67)]}
    for q in ["high", "low"]:
        splits[q] = ints(stratified_split(meta[meta["quality"] == q]))
        splits["t" + q] = ints(stratified_split(meta[meta["tertile"] == q]))
    with open(os.path.join(C.CACHE, "splits.json"), "w") as f:
        json.dump(splits, f)
    print("Sharpness median threshold =", round(thr, 1))
    print(meta["quality"].value_counts().to_string())
    for k in ["full", "high", "low", "thigh", "tlow"]:
        print(k, [len(p) for p in splits[k]])


def load_aptos():
    meta = pd.read_csv(os.path.join(C.CACHE, "aptos_meta.csv"))
    arr = np.load(os.path.join(C.CACHE, "aptos_256.npy"), mmap_mode="r")
    with open(os.path.join(C.CACHE, "splits.json")) as f:
        splits = json.load(f)
    return meta, arr, splits


def load_native(c, s):
    return np.load(os.path.join(C.CACHE, f"native_{c}_{s}.npy"), mmap_mode="r")


def _images_under(root):
    out = {}
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if p.lower().endswith(IMG_EXT) and os.path.isfile(p):
            out.setdefault(os.path.splitext(os.path.basename(p))[0].lower(), p)
    return out


def _split_of(p):
    s = p.lower()
    return "test" if "test" in s else ("train" if "train" in s else "")


def find_idrid_grading():
    csvs = [p for p in glob.glob(os.path.join(C.INPUT, "**", "*.csv"), recursive=True) if "grading" in p.lower() and "label" in p.lower()]
    rows = []
    for p in csvs:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        d.columns = [c.strip() for c in d.columns]
        if "Image name" in d.columns and "Retinopathy grade" in d.columns:
            d = d[["Image name", "Retinopathy grade"]].dropna()
            d.columns = ["image", "grade"]
            d["image"] = d["image"].astype(str).str.strip()
            d["split"] = _split_of(os.path.basename(p))
            d["root"] = os.path.dirname(os.path.dirname(p))
            rows.append(d)
    if not rows:
        return None
    d = pd.concat(rows).drop_duplicates(["split", "image"])
    imgs = {}
    for r in d["root"].unique():
        for p in glob.glob(os.path.join(r, "**", "*"), recursive=True):
            if p.lower().endswith(IMG_EXT) and os.path.isfile(p):
                rel = os.path.relpath(p, r)
                imgs.setdefault((_split_of(rel), os.path.splitext(os.path.basename(p))[0].lower()), p)
    anyname = {}
    for (sp, n), p in imgs.items():
        anyname.setdefault(n, p)
    d["path"] = [imgs.get((s, i.lower())) or (anyname.get(i.lower()) if not s else None) for s, i in zip(d["split"], d["image"])]
    d = d.dropna(subset=["path"])
    d["grade"] = d["grade"].astype(int)
    d["image"] = d["split"] + "_" + d["image"]
    return d[["image", "grade", "path"]].reset_index(drop=True) if len(d) else None


def find_messidor():
    aptos = C.find_aptos()
    for p in glob.glob(os.path.join(C.INPUT, "**", "*.csv"), recursive=True):
        if os.path.abspath(p).startswith(os.path.abspath(aptos)):
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        cols = {c.lower().strip(): c for c in d.columns}
        if not any("adjudicated" in c for c in cols):
            continue
        idc = next((cols[c] for c in ["image_id", "id_code", "image", "image name"] if c in cols), None)
        gc = next((cols[c] for c in ["adjudicated_dr_grade", "diagnosis", "dr_grade"] if c in cols), None)
        if idc is None or gc is None:
            continue
        if "adjudicated_gradable" in cols:
            d = d[d[cols["adjudicated_gradable"]] == 1]
        d = d[[idc, gc]].dropna()
        d.columns = ["image", "grade"]
        root = os.path.dirname(p)
        while root != os.path.dirname(root) and os.path.abspath(os.path.dirname(root)) != os.path.abspath(C.INPUT):
            root = os.path.dirname(root)
        imgs = _images_under(root)
        d["path"] = d["image"].astype(str).map(lambda s: imgs.get(os.path.splitext(os.path.basename(s))[0].lower()))
        d = d.dropna(subset=["path"])
        if len(d):
            d["grade"] = d["grade"].astype(int)
            return d.reset_index(drop=True)
    return None


def find_idrid_seg():
    masks = {}
    for t in ["MA", "HE", "EX", "SE"]:
        for p in glob.glob(os.path.join(C.INPUT, "**", f"*_{t}.tif*"), recursive=True):
            stem = os.path.basename(p).rsplit("_", 1)[0]
            masks.setdefault(stem, {})[t] = p
    if not masks:
        return None
    imgs = {}
    roots = {os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p)))) for v in masks.values() for p in v.values()}
    for r in roots:
        imgs.update(_images_under(r))
    rows = []
    for stem, m in sorted(masks.items()):
        p = imgs.get(stem.lower())
        if p is not None and not p.lower().endswith((".tif", ".tiff")):
            rows.append({"image": stem, "path": p, **{t: m.get(t) for t in ["MA", "HE", "EX", "SE"]}})
    return pd.DataFrame(rows) if rows else None


def _ext_worker(path):
    return to_square(read_rgb(path))


def _mask_worker(row):
    out = []
    for t in ["MA", "HE", "EX", "SE"]:
        p = row.get(t)
        if isinstance(p, str):
            m = np.array(Image.open(p).convert("L")) > 0
            out.append(to_square(m.astype(np.float32), interp=cv2.INTER_AREA) > 0)
        else:
            out.append(None)
    out = [o if o is not None else np.zeros((C.SHORT, C.SHORT), bool) for o in out]
    return np.stack(out).astype(np.uint8)


def build_external_cache():
    C.ensure_dirs()
    found = {"idrid_grading": find_idrid_grading(), "messidor2": find_messidor()}
    for name, d in found.items():
        if d is None:
            print(f"{name}: not attached, skipped")
            continue
        if C.QUICK:
            d = d.head(60)
        npy = os.path.join(C.CACHE, f"{name}_256.npy")
        if not os.path.exists(npy):
            with Pool(max(1, os.cpu_count() or 1)) as pool:
                arr = np.stack(pool.map(_ext_worker, list(d["path"]), chunksize=4))
            np.save(npy, arr)
            d.to_csv(os.path.join(C.CACHE, f"{name}_meta.csv"), index=False)
        print(f"{name}: {len(d)} images, grades {d['grade'].value_counts().sort_index().to_dict()}")
    seg = find_idrid_seg()
    if seg is None:
        print("idrid_seg: not attached, skipped")
        return
    npy = os.path.join(C.CACHE, "idrid_seg_256.npy")
    if not os.path.exists(npy):
        with Pool(max(1, os.cpu_count() or 1)) as pool:
            arr = np.stack(pool.map(_ext_worker, list(seg["path"]), chunksize=2))
            masks = np.stack(pool.map(_mask_worker, seg.to_dict("records"), chunksize=2))
        np.save(npy, arr)
        np.save(os.path.join(C.CACHE, "idrid_seg_masks_256.npy"), masks)
        seg.to_csv(os.path.join(C.CACHE, "idrid_seg_meta.csv"), index=False)
    print(f"idrid_seg: {len(seg)} images with lesion masks")


def load_external(name):
    npy = os.path.join(C.CACHE, f"{name}_256.npy")
    if not os.path.exists(npy):
        return None, None
    return pd.read_csv(os.path.join(C.CACHE, f"{name}_meta.csv")), np.load(npy, mmap_mode="r")
