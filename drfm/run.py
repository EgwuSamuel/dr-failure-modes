import os
import sys
import json
import glob
import time
import shutil
import argparse
import traceback
import subprocess

from . import config as C

COST = {"resnet50": 3, "vit_small_patch16_224": 2, "efficientnet_b0": 2}
DOMCOST = {"full": 4, "high": 2, "low": 2, "thigh": 1, "tlow": 1}


def _job(kind, protocol, arch, domain, lr, seed):
    j = {"kind": kind, "protocol": protocol, "arch": arch, "domain": domain, "lr": lr, "seed": seed}
    j["id"] = f"{kind}-{protocol}-{arch}-{domain}-lr{lr:g}-s{seed}"
    return j


def lr_choice():
    p = os.path.join(C.OUT, "lr_selection.json")
    if not os.path.exists(p):
        raise RuntimeError("Run the grid phase first")
    with open(p) as f:
        return json.load(f)


def jobs(phase):
    out = []
    if phase == "grid":
        for a in C.ARCHS:
            for lr in C.LR_GRID:
                out.append(_job("train", "main", a, "full", lr, C.SEEDS[0]))
    elif phase == "main":
        lr = lr_choice()
        for a in C.ARCHS:
            for s in C.SEEDS:
                for d in ["full", "high", "low"]:
                    out.append(_job("train", "main", a, d, lr[a], s))
                    out.append(_job("train", "v1", a, d, C.PROTOCOLS["v1"]["lr"], s))
            for s in C.TERTILE_SEEDS:
                for d in ["thigh", "tlow"]:
                    out.append(_job("train", "main", a, d, lr[a], s))
        out.sort(key=lambda j: -(COST.get(j["arch"], 2) * DOMCOST[j["domain"]] * (3 if j["protocol"] == "main" else 1)))
    elif phase == "xai":
        lr = lr_choice()
        for a in C.ARCHS:
            for s in C.SEEDS:
                out.append(_job("xai", "main", a, "full", lr[a], s))
    return out


def _done(j):
    if j["kind"] == "xai":
        return os.path.exists(os.path.join(C.RUNS, "xai", j["arch"], f"seed{j['seed']}", "done.json"))
    from .engine import run_dir
    return os.path.exists(os.path.join(run_dir(j), "done.json"))


def worker(phase):
    from .engine import run_train_job
    from .xai import run_xai_job
    for j in jobs(phase):
        if _done(j):
            continue
        claim = os.path.join(C.CLAIMS, j["id"])
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        print(f"== {j['id']}", flush=True)
        try:
            (run_xai_job if j["kind"] == "xai" else run_train_job)(j)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
            with open(os.path.join(C.LOGS, "failed.txt"), "a") as f:
                f.write(j["id"] + "\n")


def launch(phase):
    C.ensure_dirs()
    shutil.rmtree(C.CLAIMS, ignore_errors=True)
    os.makedirs(C.CLAIMS, exist_ok=True)
    js = jobs(phase)
    try:
        import torch
        ngpu = max(1, torch.cuda.device_count())
    except Exception:
        ngpu = 1
    print(f"phase={phase} jobs={len(js)} already_done={sum(_done(j) for j in js)} gpus={ngpu}", flush=True)
    procs = []
    for g in range(ngpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        log = open(os.path.join(C.LOGS, f"{phase}_gpu{g}.log"), "a")
        procs.append((subprocess.Popen([sys.executable, "-m", "drfm.run", "worker", "--phase", phase], env=env,
                                       stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), log))
        time.sleep(5)
    t0 = time.time()
    while any(p.poll() is None for p, _ in procs):
        time.sleep(60)
        print(f"  [{(time.time() - t0) / 60:.0f} min] done {sum(_done(j) for j in js)}/{len(js)}", flush=True)
    for _, log in procs:
        log.close()
    missing = [j["id"] for j in js if not _done(j)]
    print(f"phase {phase} finished: {len(js) - len(missing)}/{len(js)} done", flush=True)
    for m in missing:
        print("  NOT DONE:", m)
    if missing:
        for g in range(ngpu):
            lp = os.path.join(C.LOGS, f"{phase}_gpu{g}.log")
            print(f"--- tail {lp}")
            with open(lp) as f:
                print("".join(f.readlines()[-40:]))
    if phase == "grid":
        select_lr()


def select_lr():
    import pandas as pd
    rows = []
    for j in jobs("grid"):
        from .engine import run_dir
        p = os.path.join(run_dir(j), "done.json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            rows.append({"arch": j["arch"], "lr": j["lr"], "best_val_qwk": d["best_val_qwk"], "best_epoch": d["best_epoch"]})
    df = pd.DataFrame(rows)
    os.makedirs(C.TABLES, exist_ok=True)
    df.to_csv(os.path.join(C.TABLES, "table_lr_selection.csv"), index=False)
    choice = {a: float(g.sort_values(["best_val_qwk", "lr"], ascending=[False, True]).iloc[0]["lr"]) for a, g in df.groupby("arch")}
    with open(os.path.join(C.OUT, "lr_selection.json"), "w") as f:
        json.dump(choice, f, indent=1)
    print(df.to_string(index=False))
    print("selected:", choice)


def prefetch():
    import timm
    for a in C.ARCHS:
        timm.create_model(a, pretrained=True, num_classes=C.NUM_CLASSES)
        print("weights ready:", a)


def pack():
    base = os.path.join(C.WORK, "drfm_results")
    shutil.make_archive(base, "zip", C.OUT)
    print("Saved", base + ".zip", f"({os.path.getsize(base + '.zip') / 1e6:.1f} MB)")


def restore(src):
    for sub in ["outputs", "checkpoints"]:
        s = os.path.join(src, sub)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(C.WORK, sub), dirs_exist_ok=True)
            print("restored", sub)
    for z in glob.glob(os.path.join(src, "*.zip")):
        shutil.unpack_archive(z, C.OUT)
        print("restored", z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache", "prefetch", "launch", "worker", "select", "analyze", "pack", "restore", "all"])
    ap.add_argument("--phase", default="main")
    ap.add_argument("--src", default="")
    a = ap.parse_args()
    C.ensure_dirs()
    if a.stage == "cache":
        from . import data as D
        D.build_aptos_cache()
        D.build_external_cache()
    elif a.stage == "prefetch":
        prefetch()
    elif a.stage == "launch":
        launch(a.phase)
    elif a.stage == "worker":
        worker(a.phase)
    elif a.stage == "select":
        select_lr()
    elif a.stage == "analyze":
        from .analyze import run_all
        run_all()
    elif a.stage == "pack":
        pack()
    elif a.stage == "restore":
        restore(a.src)
    elif a.stage == "all":
        from . import data as D
        D.build_aptos_cache()
        D.build_external_cache()
        for ph in ["grid", "main", "xai"]:
            launch(ph)
        from .analyze import run_all
        run_all()
        pack()


if __name__ == "__main__":
    main()
