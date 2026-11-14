import os
import json
import math
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from . import config as C
from . import data as D
from .corruptions import corrupt, conditions
from .metrics import softmax, from_conf

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED = os.environ.get("DRFM_PRETRAINED", "1") == "1"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class ArrDS(Dataset):
    def __init__(self, arr, idx, labels, keys, train=False, ctype="clean", sev=0):
        self.arr, self.idx, self.labels, self.keys = arr, list(idx), list(labels), list(keys)
        self.train, self.ctype, self.sev = train, ctype, sev
        self.aug = v2.Compose([v2.RandomResizedCrop(C.IMG, scale=(0.8, 1.0), antialias=True), v2.RandomHorizontalFlip()])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        img = np.asarray(self.arr[self.idx[k]])
        if self.ctype != "clean":
            img = corrupt(img, self.ctype, self.sev, self.keys[k])
        if self.train:
            x = self.aug(torch.from_numpy(np.array(img)).permute(2, 0, 1))
        else:
            x = torch.from_numpy(np.array(D.center_crop(img))).permute(2, 0, 1)
        return x, int(self.labels[k])


def _winit(wid):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, batch, shuffle, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=C.WORKERS, pin_memory=DEV == "cuda",
                      drop_last=False, worker_init_fn=_winit, generator=g, persistent_workers=False)


_MEAN = torch.tensor(C.MEAN).view(1, 3, 1, 1) * 255
_STD = torch.tensor(C.STD).view(1, 3, 1, 1) * 255


def prep(x):
    x = x.to(DEV, non_blocking=True).float()
    return ((x - _MEAN.to(DEV)) / _STD.to(DEV)).contiguous(memory_format=torch.channels_last)


def build_model(arch):
    import timm
    m = timm.create_model(arch, pretrained=PRETRAINED, num_classes=C.NUM_CLASSES)
    return m.to(DEV).to(memory_format=torch.channels_last)


@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    out, ys = [], []
    for x, y in loader:
        xx = prep(x)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            z = model(xx).float()
        bad = ~torch.isfinite(z).all(1)
        if bad.any():
            z[bad] = model(xx[bad]).float()
        out.append(z.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(out), np.concatenate(ys)


def _qwk(y, p):
    o = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES))
    np.add.at(o, (y, p), 1)
    return float(from_conf(o)["qwk"])


def run_dir(job):
    return os.path.join(C.RUNS, job["protocol"], job["arch"], job["domain"], f"lr{job['lr']:g}", f"seed{job['seed']}")


def ckpt_path(arch, seed, lr):
    return os.path.join(C.CKPT, f"{arch}_lr{lr:g}_seed{seed}.pt")


def _sets(meta, splits, domain):
    tr, va, te = splits[domain]
    return tr, va, te


def train(job, meta, arr, splits):
    p = C.PROTOCOLS[job["protocol"]]
    set_seed(job["seed"])
    tr, va, _ = _sets(meta, splits, job["domain"])
    y = meta["diagnosis"].values
    keys = meta["id_code"].values
    model = build_model(job["arch"])
    counts = np.bincount(y[tr], minlength=C.NUM_CLASSES).astype(np.float64)
    w = counts.sum() / (counts + 1e-6)
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=job["lr"], weight_decay=p["weight_decay"])
    tl = make_loader(ArrDS(arr, tr, y[tr], keys[tr], train=True), p["batch"], True, job["seed"])
    vl = make_loader(ArrDS(arr, va, y[va], keys[va]), C.EVAL_BATCH, False)
    spe = len(tl)
    total = p["epochs"] * spe
    wu = p["warmup"] * spe

    def factor(step):
        if p["warmup"] == 0:
            return 0.5 * (1 + math.cos(math.pi * (step // spe) / p["epochs"]))
        if step < wu:
            return (step + 1) / wu
        return 0.5 * (1 + math.cos(math.pi * (step - wu) / max(1, total - wu)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    best, best_ep, best_state, hist, bad = -2.0, -1, None, [], 0
    for ep in range(p["epochs"]):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, t in tl:
            t = t.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                loss = crit(model(prep(x)), t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        z, yv = predict_logits(model, vl)
        nonfinite = int((~np.isfinite(z).all(1)).sum())
        q = _qwk(yv, z.argmax(1)) if nonfinite == 0 else -1.0
        vloss = float(-np.mean(np.log(softmax(z)[np.arange(len(yv)), yv] + 1e-12))) if nonfinite == 0 else float("nan")
        hist.append({"epoch": ep + 1, "val_nonfinite": nonfinite, "train_loss": tot / max(1, nb), "val_loss": vloss, "val_qwk": q,
                     "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        print(f"  [{job['id']}] epoch {ep + 1}/{p['epochs']} val_QWK={q:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if q > best:
            best, best_ep, bad = q, ep + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if p["patience"] and bad >= p["patience"]:
                break
    model.load_state_dict(best_state)
    return model, hist, best, best_ep


def _eval_set(model, arr, idx, labels, keys, ctype="clean", sev=0):
    z, y = predict_logits(model, make_loader(ArrDS(arr, idx, labels, keys, ctype=ctype, sev=sev), C.EVAL_BATCH, False))
    return z


def evaluate(model, job, meta, arr, splits):
    y = meta["diagnosis"].values
    keys = meta["id_code"].values
    res = {}
    dom = job["domain"]
    if dom == "full":
        _, va, te = splits["full"]
        res["y/test"] = y[te]
        res["y/val"] = y[va]
        for c, s in conditions(True):
            res[f"test/{c}/{s}"] = _eval_set(model, arr, te, y[te], keys[te], c, s)
            if job["protocol"] == "main" or c == "clean":
                res[f"val/{c}/{s}"] = _eval_set(model, arr, va, y[va], keys[va], c, s)
        pos = np.arange(len(te))
        for c in C.NATIVE_CORRUPTIONS:
            for s in C.SEVERITIES:
                res[f"native/{c}/{s}"] = _eval_set(model, D.load_native(c, s), pos, y[te], keys[te])
        if job["protocol"] == "main":
            for name in ["idrid_grading", "messidor2"]:
                em, ea = D.load_external(name)
                if em is None:
                    continue
                ey, ek, ei = em["grade"].values, em["image"].astype(str).values, np.arange(len(em))
                res[f"y/{name}"] = ey
                res[f"{name}/clean/0"] = _eval_set(model, ea, ei, ey, ek)
                for c in C.CORRUPTIONS:
                    for s in C.EXTERNAL_SEVERITIES:
                        res[f"{name}/{c}/{s}"] = _eval_set(model, ea, ei, ey, ek, c, s)
    else:
        pair = ["high", "low"] if dom in ("high", "low") else ["thigh", "tlow"]
        _, va, _ = splits[dom]
        res["y/val"] = y[va]
        res["val/clean/0"] = _eval_set(model, arr, va, y[va], keys[va])
        for d in pair:
            te = splits[d][2]
            res[f"y/{d}_test"] = y[te]
            res[f"{d}_test/clean/0"] = _eval_set(model, arr, te, y[te], keys[te])
    return res


def run_train_job(job):
    out = run_dir(job)
    if os.path.exists(os.path.join(out, "done.json")):
        return
    os.makedirs(out, exist_ok=True)
    meta, arr, splits = D.load_aptos()
    t0 = time.time()
    model, hist, best, best_ep = train(job, meta, arr, splits)
    t1 = time.time()
    if job["protocol"] == "main" and job["domain"] == "full":
        os.makedirs(C.CKPT, exist_ok=True)
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                   ckpt_path(job["arch"], job["seed"], job["lr"]))
    res = evaluate(model, job, meta, arr, splits)
    np.savez_compressed(os.path.join(out, "logits.npz"), **res)
    import pandas as pd
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    with open(os.path.join(out, "done.json"), "w") as f:
        json.dump({**job, "best_val_qwk": best, "best_epoch": best_ep, "epochs_run": len(hist),
                   "train_seconds": t1 - t0, "eval_seconds": time.time() - t1}, f)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
