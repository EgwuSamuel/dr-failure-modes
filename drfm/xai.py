import os
import json

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from . import config as C
from . import data as D
from .corruptions import corrupt
from .engine import DEV, build_model, ckpt_path, prep


class RefLogOdds:
    def __call__(self, out):
        return torch.logsumexp(out[..., C.REFERABLE:], -1) - torch.logsumexp(out[..., :C.REFERABLE], -1)


def _layer(model, arch):
    if arch.startswith("vit"):
        return model.blocks[-1].norm1
    if arch.startswith("resnet"):
        return model.layer4[-1]
    return model.conv_head


def _reshape(model):
    k = getattr(model, "num_prefix_tokens", 1)

    def f(t):
        n = t.size(1) - k
        h = int(round(n ** 0.5))
        return t[:, k:, :].reshape(t.size(0), h, h, t.size(2)).permute(0, 3, 1, 2)
    return f


def load_cam(arch, seed, lr):
    from pytorch_grad_cam import GradCAM
    m = build_model(arch).float()
    sd = torch.load(ckpt_path(arch, seed, lr), map_location=DEV)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    m.eval()
    rt = _reshape(m) if arch.startswith("vit") else None
    return m, GradCAM(model=m, target_layers=[_layer(m, arch)], reshape_transform=rt)


def cams(cam, imgs, targets, bs=32):
    out = []
    for i in range(0, len(imgs), bs):
        x = prep(torch.from_numpy(np.ascontiguousarray(imgs[i:i + bs])).permute(0, 3, 1, 2))
        out.append(cam(input_tensor=x, targets=targets[i:i + bs]).astype(np.float32))
    return np.concatenate(out)


def _crop(a):
    return np.ascontiguousarray(D.center_crop(a))


def _conds():
    out = [("clean", 0)]
    for c in C.CORRUPTIONS:
        for s in C.XAI_SEVERITIES:
            out.append((c, s))
    return out


def run_xai_job(job):
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    out_dir = os.path.join(C.RUNS, "xai", job["arch"], f"seed{job['seed']}")
    if os.path.exists(os.path.join(out_dir, "done.json")):
        return
    os.makedirs(out_dir, exist_ok=True)
    meta, arr, splits = D.load_aptos()
    te = splits["full"][2]
    tm = meta.iloc[te].reset_index(drop=True)
    model, cam = load_cam(job["arch"], job["seed"], job["lr"])
    clean = np.stack([_crop(arr[i]) for i in te])
    fov = (clean.mean(-1) / 255.0 > 0.10).astype(np.float32)
    samp = list(tm.sample(n=min(6, len(tm)), random_state=0).index)
    rows, keep = [], {}
    for c, s in _conds():
        imgs = clean if c == "clean" else np.stack([_crop(corrupt(arr[i], c, s, k)) for i, k in zip(te, tm["id_code"])])
        g = cams(cam, imgs, [ClassifierOutputTarget(int(t)) for t in tm["diagnosis"]])
        e = (g * fov).sum((1, 2)) / (g.sum((1, 2)) + 1e-9)
        for j in range(len(tm)):
            rows.append({"dataset": "aptos", "image": tm["id_code"][j], "grade": int(tm["diagnosis"][j]),
                         "corruption": c, "severity": s, "energy_on_retina": float(e[j])})
        if job["seed"] == 0:
            keep[f"aptos/{c}/{s}"] = g[samp].astype(np.float16)
    if job["seed"] == 0:
        keep["aptos/ids"] = np.array(tm["id_code"][samp].tolist())
        keep["aptos/grades"] = tm["diagnosis"].values[samp]
    sm, sa = D.load_external("idrid_seg")
    if sm is not None:
        masks = np.load(os.path.join(C.CACHE, "idrid_seg_masks_256.npy"), mmap_mode="r")
        sclean = np.stack([_crop(a) for a in sa])
        sfov = sclean.mean(-1) / 255.0 > 0.10
        les = np.stack([_crop(m.transpose(1, 2, 0)).transpose(2, 0, 1) for m in masks]).astype(bool)
        union = les.any(1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        near = np.stack([cv2.dilate(u.astype(np.uint8), kern) > 0 for u in union])
        for c, s in _conds():
            imgs = sclean if c == "clean" else np.stack([_crop(corrupt(a, c, s, k)) for a, k in zip(sa, sm["image"])])
            g = cams(cam, imgs, [RefLogOdds() for _ in range(len(imgs))])
            for j in range(len(sm)):
                f, u, gj = sfov[j], union[j] & sfov[j], g[j]
                tot = float((gj * f).sum()) + 1e-9
                area = u.sum() / max(1, f.sum())
                frac = float((gj * u).sum()) / tot
                gm = np.where(f, gj, -1)
                yx = np.unravel_index(int(gm.argmax()), gm.shape)
                try:
                    pa = float(roc_auc_score(u[f], gj[f])) if 0 < u[f].sum() < f.sum() else float("nan")
                except Exception:
                    pa = float("nan")
                r = {"dataset": "idrid_seg", "image": sm["image"][j], "grade": -1, "corruption": c, "severity": s,
                     "energy_on_retina": float((gj * f).sum() / (gj.sum() + 1e-9)), "lesion_fraction": frac,
                     "lesion_area": float(area), "lesion_lift": frac / area if area > 0 else float("nan"),
                     "pointing_hit": float(near[j][yx]), "pixel_auc": pa}
                for t, name in enumerate(["MA", "HE", "EX", "SE"]):
                    lt = les[j, t] & f
                    a_t = lt.sum() / max(1, f.sum())
                    r[f"lift_{name}"] = float((gj * lt).sum() / tot / a_t) if a_t > 0 else float("nan")
                rows.append(r)
            if job["seed"] == 0:
                keep[f"idrid/{c}/{s}"] = g[:4].astype(np.float16)
        if job["seed"] == 0:
            keep["idrid/ids"] = np.array(sm["image"][:4].tolist())
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "xai.csv"), index=False)
    if keep:
        np.savez_compressed(os.path.join(out_dir, "cams.npz"), **keep)
    with open(os.path.join(out_dir, "done.json"), "w") as f:
        json.dump(job, f)
    del model, cam
    if DEV == "cuda":
        torch.cuda.empty_cache()
