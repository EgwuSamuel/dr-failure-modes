import os
import glob
import json
import zlib
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy import stats as sst
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import data as D
from . import metrics as M
from .corruptions import corrupt, conditions

MAIN_METRICS = ["accuracy", "qwk", "macro_f1", "referable_sens", "referable_spec", "ref_auc", "ece", "msge"]
DELTA_METRICS = ["accuracy", "qwk", "referable_sens", "referable_spec", "ref_auc", "ece", "msge"]
COLORS = {"gaussian_blur": "#1f77b4", "brightness": "#ff7f0e", "darkening": "#8c564b", "contrast": "#2ca02c",
          "gaussian_noise": "#9467bd", "jpeg": "#d62728", "compound_blur_dark_jpeg": "#17becf",
          "compound_noise_contrast_jpeg": "#bcbd22", "compound_field": "#7f7f7f"}
NICE = {"gaussian_blur": "Gaussian blur", "brightness": "Brightness increase", "darkening": "Underexposure",
        "contrast": "Contrast reduction", "gaussian_noise": "Gaussian noise", "jpeg": "JPEG compression",
        "compound_blur_dark_jpeg": "Blur + underexposure + JPEG", "compound_noise_contrast_jpeg": "Noise + low contrast + JPEG",
        "compound_field": "Random field profile", "clean": "Clean"}
LR = {}
_W = {}
NONFINITE = {}


def T(name):
    return os.path.join(C.TABLES, name)


def F(name):
    return os.path.join(C.FIGS, name)


def save_fig(fig, name):
    fig.savefig(F(name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(F(name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def lr_of(protocol, arch):
    return LR[arch] if protocol == "main" else C.PROTOCOLS["v1"]["lr"]


def runs(protocol, arch, domain, seeds=None):
    out = {}
    for s in seeds or C.SEEDS:
        p = os.path.join(C.RUNS, protocol, arch, domain, f"lr{lr_of(protocol, arch):g}", f"seed{s}", "logits.npz")
        if os.path.exists(p):
            with np.load(p) as z:
                r = {k: z[k] for k in z.files}
            bad = {k: int((~np.isfinite(v).all(-1)).sum()) for k, v in r.items() if v.dtype.kind == "f"}
            bad = {k: v for k, v in bad.items() if v}
            if bad:
                NONFINITE[p] = {"protocol": protocol, "arch": arch, "domain": domain, "seed": s,
                                "n_keys": len(bad), "n_rows": sum(bad.values()), "keys": ";".join(sorted(bad)[:20])}
                continue
            out[s] = r
    return out


def probs(rs, key, t=None):
    return np.stack([M.softmax(r[key], 1.0 if t is None else t[s]) for s, r in rs.items()])


def W(name, n):
    if name not in _W:
        _W[name] = M.boot_weights(n, seed=zlib.crc32(name.encode()))
    return _W[name]


def S(m, arr):
    return M.summarize(arr, basic=m.startswith("ece"))


def fmt(d, nd=3):
    return f"{d['mean']:.{nd}f} ({d['lo']:.{nd}f}-{d['hi']:.{nd}f})"


def temps(rs):
    return {s: M.fit_temperature(r["val/clean/0"], r["y/val"]) for s, r in rs.items()}


def quality_tables(meta):
    cols = ["lap_var", "tenengrad", "brenner", "brightness", "rms_contrast", "fov", "native_w", "native_h"]
    rows = []
    for split, a, b in [("median", "high", "low"), ("tertile", "high", "low")]:
        col = "quality" if split == "median" else "tertile"
        A, B = meta[meta[col] == a], meta[meta[col] == b]
        for c in cols:
            p = sst.mannwhitneyu(A[c], B[c]).pvalue
            rows.append({"split": split, "feature": c, "high_median": A[c].median(), "high_q1": A[c].quantile(.25),
                         "high_q3": A[c].quantile(.75), "low_median": B[c].median(), "low_q1": B[c].quantile(.25),
                         "low_q3": B[c].quantile(.75), "mannwhitney_p": p,
                         "cliffs_delta": 2 * sst.mannwhitneyu(A[c], B[c]).statistic / (len(A) * len(B)) - 1})
    pd.DataFrame(rows).to_csv(T("table_quality_domains.csv"), index=False)
    g = []
    for split, col in [("median", "quality"), ("tertile", "tertile")]:
        ct = pd.crosstab(meta[meta[col].isin(["high", "low"])][col], meta[meta[col].isin(["high", "low"])]["diagnosis"])
        chi = sst.chi2_contingency(ct.values)
        for q in ["high", "low"]:
            r = {"split": split, "domain": q, "n": int(ct.loc[q].sum()), "chi2": chi[0], "chi2_p": chi[1]}
            for k in range(C.NUM_CLASSES):
                r[f"grade_{k}_n"] = int(ct.loc[q].get(k, 0))
                r[f"grade_{k}_pct"] = 100 * ct.loc[q].get(k, 0) / ct.loc[q].sum()
            g.append(r)
    pd.DataFrame(g).to_csv(T("table_quality_grades.csv"), index=False)
    cor = []
    for c in ["tenengrad", "brenner", "brightness", "rms_contrast", "fov", "native_w", "diagnosis"]:
        r = sst.spearmanr(meta["lap_var"], meta[c])
        cor.append({"feature": c, "spearman_rho": r.statistic if hasattr(r, "statistic") else r[0], "p": r.pvalue})
    pd.DataFrame(cor).to_csv(T("table_quality_correlations.csv"), index=False)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    for q, col in [("high", "#1f77b4"), ("low", "#d62728")]:
        s = meta[meta["quality"] == q]
        ax[0].hist(np.log10(s["lap_var"] + 1), bins=40, alpha=.6, color=col, label=q + " quality")
        ax[3].bar(np.arange(5) + (-.2 if q == "high" else .2), np.bincount(s["diagnosis"], minlength=5) / len(s), .4, color=col, label=q)
    ax[0].set_xlabel("log10 variance of Laplacian")
    ax[0].legend(frameon=False)
    for k, c in [(1, "brightness"), (2, "rms_contrast")]:
        ax[k].boxplot([meta[meta["quality"] == q][c] for q in ["high", "low"]])
        ax[k].set_xticks([1, 2], ["high", "low"])
        ax[k].set_title(c.replace("_", " "))
    ax[0].set_title("Sharpness")
    ax[3].set_title("Grade distribution")
    ax[3].set_xlabel("Grade")
    ax[3].legend(frameon=False)
    save_fig(fig, "fig_quality_domains")


def _sev_worker(args):
    img, c, s, key = args
    out = D.center_crop(corrupt(img, c, s, key)) if c != "clean" else D.center_crop(img)
    return D.quality_metrics(cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_RGB2GRAY))


def severity_mapping(meta, arr, splits):
    te = splits["full"][2]
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        ref = pd.DataFrame(pool.map(_sev_worker, [(np.asarray(arr[i]), "clean", 0, k) for i, k in zip(range(len(meta)), meta["id_code"])], chunksize=32))
        rows = []
        for c, s in conditions(True):
            if c == "clean":
                continue
            q = pd.DataFrame(pool.map(_sev_worker, [(np.asarray(arr[i]), c, s, meta["id_code"][i]) for i in te], chunksize=16))
            r = {"corruption": c, "severity": s, "parameter": C.PARAMS[c][s - 1] if c in C.PARAMS else ""}
            for f in ["lap_var", "brightness", "rms_contrast"]:
                med = q[f].median()
                lo, hi = ref[f].quantile([.01, .99])
                r[f"{f}_median"] = med
                r[f"{f}_clean_percentile"] = 100 * (ref[f] < med).mean()
                r[f"{f}_frac_outside_clean_1_99"] = float(((q[f] < lo) | (q[f] > hi)).mean())
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(T("table_severity_mapping.csv"), index=False)
    ref.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_csv(T("table_clean_quality_reference.csv"))


def benchmark():
    rows, tests, pts = [], [], []
    for protocol in ["main", "v1"]:
        stt = {}
        for a in C.ARCHS:
            rs = runs(protocol, a, "full")
            if not rs:
                continue
            y = next(iter(rs.values()))["y/test"]
            w = W("test", len(y))
            P = probs(rs, "test/clean/0")
            st = M.stats(P, y, w)
            tt = temps(rs)
            st["ece_ts"] = M.stats(probs(rs, "test/clean/0", tt), y, w)["ece"]
            stt[a] = st
            for m in MAIN_METRICS + ["ece_ts"] + [f"recall_{k}" for k in range(5)]:
                rows.append({"protocol": protocol, "arch": a, "metric": m, **S(m, st[m])})
            am = [M.point_metrics(P[i], y)["auc_macro"] for i in range(len(P))]
            rows.append({"protocol": protocol, "arch": a, "metric": "auc_macro", "mean": float(np.mean(am)),
                         "sd": float(np.std(am, ddof=1)) if len(am) > 1 else np.nan, "lo": np.nan, "hi": np.nan})
            rows.append({"protocol": protocol, "arch": a, "metric": "temperature", "mean": float(np.mean(list(tt.values()))),
                         "sd": float(np.std(list(tt.values()), ddof=1)) if len(tt) > 1 else np.nan, "lo": np.nan, "hi": np.nan})
            if protocol == "main":
                cm = sum(np.bincount(y * 5 + p.argmax(-1), minlength=25).reshape(5, 5) for p in P)
                pts.append((a, cm))
        archs = list(stt)
        for i in range(len(archs)):
            for j in range(i + 1, len(archs)):
                for m in ["qwk", "accuracy", "ece", "referable_spec", "ref_auc"]:
                    d = M.summarize_diff(stt[archs[i]][m], stt[archs[j]][m])
                    tests.append({"protocol": protocol, "a": archs[i], "b": archs[j], "metric": m, **d,
                                  "welch_p": M.welch(stt[archs[i]][m][:, 0], stt[archs[j]][m][:, 0])})
    df = pd.DataFrame(rows)
    df.to_csv(T("table_benchmark.csv"), index=False)
    if tests:
        tdf = pd.DataFrame(tests)
        tdf["p_holm"] = np.nan
        for _, g in tdf.groupby("protocol"):
            tdf.loc[g.index, "p_holm"] = M.holm(g["p"].values)
        tdf.to_csv(T("table_benchmark_tests.csv"), index=False)
    if pts:
        fig, axes = plt.subplots(1, len(pts), figsize=(4.6 * len(pts), 4.2))
        axes = np.atleast_1d(axes)
        for ax, (a, cm) in zip(axes, pts):
            cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(C.label(a))
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=8)
        save_fig(fig, "fig_confusion")
    return df


def crosscond(meta, splits):
    rows, gaps, gtests, budget = [], [], [], []
    for prefix, doms, protocols in [("", ["high", "low"], ["main", "v1"]), ("t", ["thigh", "tlow"], ["main"])]:
        for protocol in protocols:
            gs = {}
            for a in C.ARCHS:
                for d in doms:
                    rs = runs(protocol, a, d, C.TERTILE_SEEDS if prefix else None)
                    if not rs:
                        continue
                    other = [x for x in doms if x != d][0]
                    st = {}
                    for td in [d, other]:
                        y = next(iter(rs.values()))[f"y/{td}_test"]
                        st[td] = M.stats(probs(rs, f"{td}_test/clean/0"), y, W(td + "_test", len(y)))
                        for m in ["accuracy", "qwk", "referable_sens", "referable_spec", "ref_auc", "msge", "ece"]:
                            rows.append({"split": "tertile" if prefix else "median", "protocol": protocol, "arch": a,
                                         "train": d, "test": td, "setting": "in_domain" if td == d else "cross_domain",
                                         "metric": m, **S(m, st[td][m])})
                    gap = np.nanmean(st[d]["qwk"], 0) - np.nanmean(st[other]["qwk"], 0)
                    per_seed = st[d]["qwk"][:, 0] - st[other]["qwk"][:, 0]
                    gs[(a, d)] = (gap, per_seed)
                    lo, hi = np.nanpercentile(gap[1:], [2.5, 97.5])
                    gaps.append({"split": "tertile" if prefix else "median", "protocol": protocol, "arch": a, "train": d,
                                 "in_domain_qwk": float(np.nanmean(st[d]["qwk"][:, 0])),
                                 "cross_domain_qwk": float(np.nanmean(st[other]["qwk"][:, 0])),
                                 "gap": float(gap[0]), "gap_lo": lo, "gap_hi": hi, "gap_p": M.boot_p(gap),
                                 "gap_sd_seeds": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan, "n_seeds": len(rs)})
            for d in doms:
                ar = [a for a in C.ARCHS if (a, d) in gs]
                for i in range(len(ar)):
                    for j in range(i + 1, len(ar)):
                        ga, gb = gs[(ar[i], d)], gs[(ar[j], d)]
                        dd = ga[0] - gb[0]
                        lo, hi = np.nanpercentile(dd[1:], [2.5, 97.5])
                        gtests.append({"split": "tertile" if prefix else "median", "protocol": protocol, "train": d,
                                       "a": ar[i], "b": ar[j], "gap_a": float(ga[0][0]), "gap_b": float(gb[0][0]),
                                       "diff": float(dd[0]), "lo": lo, "hi": hi, "p": M.boot_p(dd), "welch_p": M.welch(ga[1], gb[1])})
    cdf = pd.DataFrame(rows)
    cdf.to_csv(T("table_crosscondition.csv"), index=False)
    gdf = pd.DataFrame(gaps)
    gdf.to_csv(T("table_generalisation_gap.csv"), index=False)
    if gtests:
        t = pd.DataFrame(gtests)
        t["p_holm"] = np.nan
        for _, g in t.groupby(["split", "protocol"]):
            t.loc[g.index, "p_holm"] = M.holm(g["p"].values)
        t.to_csv(T("table_gap_tests.csv"), index=False)
    for a in C.ARCHS:
        for d in ["full", "high", "low"]:
            rm, rv = runs("main", a, d), runs("v1", a, d)
            if not rm or not rv:
                continue
            key = "test/clean/0" if d == "full" else f"{d}_test/clean/0"
            y = next(iter(rm.values()))["y/test" if d == "full" else f"y/{d}_test"]
            w = W("test" if d == "full" else d + "_test", len(y))
            sm, sv = M.stats(probs(rm, key), y, w), M.stats(probs(rv, key), y, w)
            for m in ["qwk", "accuracy", "ece", "msge"]:
                budget.append({"arch": a, "domain": d, "metric": m, "main": M.summarize(sm[m])["mean"],
                               "v1": M.summarize(sv[m])["mean"], **{f"diff_{k}": v for k, v in M.summarize_diff(sm[m], sv[m]).items()}})
    pd.DataFrame(budget).to_csv(T("table_training_budget.csv"), index=False)
    grade_matched(splits)
    return gdf


def grade_matched(splits):
    rng = np.random.default_rng(7)
    rows = []
    for a in C.ARCHS:
        for d in ["high", "low"]:
            rs = runs("main", a, d)
            if not rs:
                continue
            other = "low" if d == "high" else "high"
            yi, yc = next(iter(rs.values()))[f"y/{d}_test"], next(iter(rs.values()))[f"y/{other}_test"]
            pi, pc = probs(rs, f"{d}_test/clean/0").argmax(-1), probs(rs, f"{other}_test/clean/0").argmax(-1)
            vals = []
            for _ in range(C.GRADE_MATCH_REPS):
                ii, cc = [], []
                for g in range(C.NUM_CLASSES):
                    a1, a2 = np.flatnonzero(yi == g), np.flatnonzero(yc == g)
                    k = min(len(a1), len(a2))
                    if k:
                        ii += list(rng.choice(a1, k, replace=False))
                        cc += list(rng.choice(a2, k, replace=False))
                ii, cc = np.array(ii), np.array(cc)
                qi = np.mean([M.from_conf(np.bincount(yi[ii] * 5 + p[ii], minlength=25).reshape(5, 5))["qwk"] for p in pi])
                qc = np.mean([M.from_conf(np.bincount(yc[cc] * 5 + p[cc], minlength=25).reshape(5, 5))["qwk"] for p in pc])
                vals.append((qi, qc, qi - qc, len(ii)))
            v = np.array(vals)
            rows.append({"arch": a, "train": d, "in_domain_qwk": v[:, 0].mean(), "cross_domain_qwk": v[:, 1].mean(),
                         "gap": v[:, 2].mean(), "gap_lo": np.percentile(v[:, 2], 2.5), "gap_hi": np.percentile(v[:, 2], 97.5),
                         "n_per_domain": int(v[0, 3])})
    pd.DataFrame(rows).to_csv(T("table_grade_matched.csv"), index=False)


def taxonomy(dd):
    a, s, p, g = dd["ref_auc"], dd["referable_sens"], dd["referable_spec"], dd["msge"]
    loss = a["hi"] < 0 and a["mean"] <= -C.AUC_MARGIN
    shift = (abs(p["mean"]) >= C.SHIFT_MARGIN and (p["hi"] < 0 or p["lo"] > 0)) or (abs(s["mean"]) >= C.SHIFT_MARGIN and (s["hi"] < 0 or s["lo"] > 0))
    if loss and shift:
        base = "mixed"
    elif loss:
        base = "discriminability_loss"
    elif shift:
        base = "operating_point_shift"
    else:
        return "robust"
    direction = "over" if g["lo"] > 0 else ("under" if g["hi"] < 0 else "neutral")
    return f"{base}_{direction}"


def robustness():
    rows, tax, ece_rows, shift_rows, curve = [], [], [], [], {}
    for protocol in ["main", "v1"]:
        for a in C.ARCHS:
            rs = runs(protocol, a, "full")
            if not rs:
                continue
            r0 = next(iter(rs.values()))
            y = r0["y/test"]
            w = W("test", len(y))
            P0 = probs(rs, "test/clean/0")
            s0 = M.stats(P0, y, w)
            tt = temps(rs) if protocol == "main" else None
            pred0 = P0.argmax(-1)
            keys = [("working", c, s, f"test/{c}/{s}") for c, s in conditions(True)]
            keys += [("native", c, s, f"native/{c}/{s}") for c in C.NATIVE_CORRUPTIONS for s in C.SEVERITIES]
            delong_rows = []
            for proto2, c, s, k in keys:
                if k not in r0:
                    continue
                P = probs(rs, k)
                st = M.stats(P, y, w)
                sh = M.shift_stats(P.argmax(-1), pred0, w)
                base = {"protocol": protocol, "arch": a, "corruption_protocol": proto2, "corruption": c, "severity": s}
                dd = {}
                for m in MAIN_METRICS:
                    rows.append({**base, "metric": m, **S(m, st[m])})
                for m in DELTA_METRICS:
                    dd[m] = M.summarize_diff(st[m], s0[m])
                    rows.append({**base, "metric": "delta_" + m, "mean": dd[m]["mean"], "sd": np.nan, "lo": dd[m]["lo"],
                                 "hi": dd[m]["hi"], "p": dd[m]["p"]})
                for m in ["pred_shift", "up_rate", "down_rate"]:
                    rows.append({**base, "metric": m, **M.summarize(sh[m])})
                if tt is not None and proto2 == "working":
                    e2 = M.stats(probs(rs, k, tt), y, w)["ece"]
                    ece_rows.append({**base, "ece": S("ece", st["ece"])["mean"], **{f"ece_ts_{q}": v for q, v in S("ece", e2).items()}})
                if c != "clean":
                    ps = [M.delong(y >= C.REFERABLE, P0[i][:, 2:].sum(-1), P[i][:, 2:].sum(-1))[2] for i in range(len(P))]
                    delong_rows.append((len(tax), ps))
                    tax.append({**base, "label": taxonomy(dd), **{f"d_{m}": dd[m]["mean"] for m in ["ref_auc", "referable_sens", "referable_spec", "msge", "qwk"]},
                                **{f"d_{m}_lo": dd[m]["lo"] for m in ["ref_auc", "referable_sens", "referable_spec", "msge"]},
                                **{f"d_{m}_hi": dd[m]["hi"] for m in ["ref_auc", "referable_sens", "referable_spec", "msge"]},
                                "delong_p_median": float(np.median(ps))})
                if protocol == "main" and proto2 == "working":
                    curve[(a, c, s)] = {m: S(m, st[m]) for m in MAIN_METRICS}
            if delong_rows:
                mat = np.array([p for _, p in delong_rows])
                adj = np.column_stack([M.holm(mat[:, j]) for j in range(mat.shape[1])])
                for (ti, _), row in zip(delong_rows, adj):
                    tax[ti]["delong_seeds_sig_holm"] = int((row < C.ALPHA).sum())
                    tax[ti]["n_seeds"] = len(row)
    df = pd.DataFrame(rows)
    df.to_csv(T("table_robustness.csv"), index=False)
    pd.DataFrame(tax).to_csv(T("table_taxonomy.csv"), index=False)
    pd.DataFrame(ece_rows).to_csv(T("table_calibration.csv"), index=False)
    wide = []
    sel = df[(df.corruption_protocol == "working") & ((df.severity == max(C.SEVERITIES)) | (df.corruption == "clean"))]
    for (protocol, a, c), g in sel.groupby(["protocol", "arch", "corruption"], sort=False):
        r = {"protocol": protocol, "arch": a, "corruption": c}
        for m in ["accuracy", "qwk", "referable_sens", "referable_spec", "ref_auc", "ece", "msge"]:
            x = g[g.metric == m].iloc[0]
            r[m] = x["mean"]
            r[m + "_ci"] = f"{x['lo']:.3f}-{x['hi']:.3f}"
        wide.append(r)
    pd.DataFrame(wide).to_csv(T("table_robustness_sev5.csv"), index=False)
    burden = []
    for x in wide:
        g = sel[(sel.protocol == x["protocol"]) & (sel.arch == x["arch"]) & (sel.corruption == x["corruption"])]
        sp, se = g[g.metric == "referable_spec"].iloc[0], g[g.metric == "referable_sens"].iloc[0]
        burden.append({"protocol": x["protocol"], "arch": x["arch"], "corruption": x["corruption"],
                       "false_pos_per_1000": (1 - sp["mean"]) * C.PREVALENCE_NONREF, "fp_lo": (1 - sp["hi"]) * C.PREVALENCE_NONREF,
                       "fp_hi": (1 - sp["lo"]) * C.PREVALENCE_NONREF, "missed_per_1000": (1 - se["mean"]) * C.PREVALENCE_REF,
                       "missed_lo": (1 - se["hi"]) * C.PREVALENCE_REF, "missed_hi": (1 - se["lo"]) * C.PREVALENCE_REF})
    pd.DataFrame(burden).to_csv(T("table_clinical_burden.csv"), index=False)
    if curve:
        fig_robustness(curve)
    return df


def fig_robustness(curve):
    archs = [a for a in C.ARCHS if any(k[0] == a for k in curve)]
    ms = [("qwk", "QWK"), ("referable_sens", "Referable sensitivity"), ("referable_spec", "Referable specificity"),
          ("ref_auc", "Referable AUC"), ("msge", "MSGE"), ("ece", "ECE")]
    for group, cs in [("single", C.CORRUPTIONS), ("compound", C.COMPOUNDS)]:
        fig, axes = plt.subplots(len(archs), len(ms), figsize=(3.3 * len(ms), 2.8 * len(archs)), squeeze=False)
        for i, a in enumerate(archs):
            for j, (m, mn) in enumerate(ms):
                ax = axes[i, j]
                for c in cs:
                    xs = [s for s in C.SEVERITIES if (a, c, s) in curve]
                    if not xs:
                        continue
                    v = [curve[(a, c, s)][m] for s in xs]
                    ax.plot(xs, [q["mean"] for q in v], marker="o", ms=3, color=COLORS[c], label=NICE[c])
                    ax.fill_between(xs, [q["lo"] for q in v], [q["hi"] for q in v], color=COLORS[c], alpha=.15, lw=0)
                ax.axhline(curve[(a, "clean", 0)][m]["mean"], ls="--", color="grey", lw=1)
                if i == 0:
                    ax.set_title(mn)
                if j == 0:
                    ax.set_ylabel(C.label(a))
                if i == len(archs) - 1:
                    ax.set_xlabel("Severity")
                ax.grid(alpha=.25)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(cs), frameon=False, bbox_to_anchor=(.5, -.03))
        fig.tight_layout(rect=[0, .04, 1, 1])
        save_fig(fig, f"fig_robustness_{group}")


def recalibration():
    rows = []
    for a in C.ARCHS:
        rs = runs("main", a, "full")
        if not rs:
            continue
        for c, s in conditions(True):
            if c == "clean":
                continue
            per = []
            for sd, r in rs.items():
                if f"val/{c}/{s}" not in r:
                    continue
                yv, yt = r["y/val"], r["y/test"]
                pv0, pv, pt = M.softmax(r["val/clean/0"]), M.softmax(r[f"val/{c}/{s}"]), M.softmax(r[f"test/{c}/{s}"])
                target = float((pv0.argmax(1) >= C.REFERABLE)[yv >= C.REFERABLE].mean())
                d_pred = pt.argmax(1) >= C.REFERABLE
                pos = yt >= C.REFERABLE
                _, sa, pa = M.recalibrated(pv0[:, 2:].sum(1), yv, pt[:, 2:].sum(1), yt, target)
                _, sb, pb = M.recalibrated(pv[:, 2:].sum(1), yv, pt[:, 2:].sum(1), yt, target)
                p0t = M.softmax(r["test/clean/0"])[:, 2:].sum(1)
                per.append([d_pred[pos].mean(), (~d_pred[~pos]).mean(), sa, pa, sb, pb, target,
                            M.spec_at_sens(p0t, yt, target), M.spec_at_sens(pt[:, 2:].sum(1), yt, target)])
            if not per:
                continue
            v = np.array(per)
            r = {"arch": a, "corruption": c, "severity": s, "target_sens": v[:, 6].mean(),
                 "clean_spec_at_target": v[:, 7].mean(), "oracle_spec_at_target": v[:, 8].mean(),
                 "oracle_spec_at_target_sd": v[:, 8].std(ddof=1) if len(v) > 1 else np.nan}
            for k, name in enumerate(["default_sens", "default_spec", "cleanthr_sens", "cleanthr_spec", "condthr_sens", "condthr_spec"]):
                r[name] = v[:, k].mean()
                r[name + "_sd"] = v[:, k].std(ddof=1) if len(v) > 1 else np.nan
            rows.append(r)
    pd.DataFrame(rows).to_csv(T("table_recalibration.csv"), index=False)


def roc_and_reliability():
    archs = [a for a in C.ARCHS if runs("main", a, "full", [C.SEEDS[0]])]
    if not archs:
        return
    sel = ["brightness", "darkening", "jpeg", "gaussian_noise"]
    sv = max(C.SEVERITIES)
    grid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(len(archs), len(sel), figsize=(3.4 * len(sel), 3.2 * len(archs)), squeeze=False)
    fig2, axes2 = plt.subplots(1, len(archs), figsize=(4.2 * len(archs), 4), squeeze=False)
    for i, a in enumerate(archs):
        rs = runs("main", a, "full")
        y = next(iter(rs.values()))["y/test"]
        pos = y >= C.REFERABLE

        def mroc(key):
            ts = []
            for r in rs.values():
                sc = M.softmax(r[key])[:, 2:].sum(1)
                o = np.argsort(-sc)
                tp = np.r_[0, np.cumsum(pos[o])] / pos.sum()
                fp = np.r_[0, np.cumsum(~pos[o])] / (~pos).sum()
                ts.append(np.interp(grid, fp, tp))
            return np.mean(ts, 0)
        base = mroc("test/clean/0")
        for j, c in enumerate(sel):
            ax = axes[i, j]
            ax.plot(grid, base, color="grey", ls="--", label="clean")
            ax.plot(grid, mroc(f"test/{c}/{sv}"), color=COLORS[c], label=f"{NICE[c]} (sev {sv})")
            ax.plot([0, 1], [0, 1], color="k", lw=.5)
            ax.set_xlabel("1 - specificity")
            if j == 0:
                ax.set_ylabel(f"{C.label(a)}\nsensitivity")
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax = axes2[0, i]
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
        for c, col in [("clean", "k"), ("brightness", COLORS["brightness"]), ("jpeg", COLORS["jpeg"]), ("gaussian_noise", COLORS["gaussian_noise"])]:
            key = "test/clean/0" if c == "clean" else f"test/{c}/{sv}"
            P = np.concatenate([M.softmax(r[key]) for r in rs.values()])
            yy = np.concatenate([y for _ in rs])
            conf, acc = P.max(1), (P.argmax(1) == yy).astype(float)
            b = np.clip(np.ceil(conf * 10).astype(int) - 1, 0, 9)
            xs = [conf[b == k].mean() for k in range(10) if (b == k).sum() >= 5]
            ys = [acc[b == k].mean() for k in range(10) if (b == k).sum() >= 5]
            ece = np.mean([M.point_metrics(M.softmax(r[key]), y)["ece"] for r in rs.values()])
            ax.plot(xs, ys, marker="o", color=col, label=f"{NICE[c]} (ECE={ece:.3f})")
        ax.set_title(C.label(a))
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig_roc_referable")
    fig2.tight_layout()
    save_fig(fig2, "fig_reliability")


def label_noise():
    rows, byg = [], []
    rng = np.random.default_rng(3)
    sv = max(C.SEVERITIES)
    for a in C.ARCHS:
        rs = runs("main", a, "full")
        if not rs:
            continue
        y = next(iter(rs.values()))["y/test"]
        conds = [("clean", 0)] + [(c, sv) for c in C.CORRUPTIONS + C.COMPOUNDS]
        P = {cs: probs(rs, f"test/{cs[0]}/{cs[1]}") for cs in conds}
        pred = {cs: p.argmax(-1) for cs, p in P.items()}
        for cs in conds:
            for g in range(C.NUM_CLASSES):
                mk = y == g
                if mk.any():
                    v = (pred[cs][:, mk] - g).mean(1)
                    byg.append({"arch": a, "corruption": cs[0], "severity": cs[1], "grade": g, "n": int(mk.sum()),
                                "msge_mean": v.mean(), "msge_sd": v.std(ddof=1) if len(v) > 1 else np.nan})
        for rate in [0.0] + C.LABEL_NOISE_RATES:
            reps = 1 if rate == 0 else C.LABEL_NOISE_REPS
            acc = {cs: [] for cs in conds}
            for _ in range(reps):
                yn = M.adjacent_noise(y, rate, rng) if rate else y
                for cs in conds:
                    e = np.mean([np.abs(M.ece_terms(P[cs][i], yn).sum(0)).sum() / len(yn) for i in range(len(P[cs]))])
                    acc[cs].append(((pred[cs] - yn).mean(), e))
            base = np.array(acc[("clean", 0)])
            for cs in conds[1:]:
                v = np.array(acc[cs])
                rows.append({"arch": a, "corruption": cs[0], "severity": cs[1], "noise_rate": rate,
                             "msge_clean": base[:, 0].mean(), "msge_corrupt": v[:, 0].mean(),
                             "delta_msge": (v[:, 0] - base[:, 0]).mean(), "delta_msge_sd": (v[:, 0] - base[:, 0]).std(),
                             "ece_clean": base[:, 1].mean(), "ece_corrupt": v[:, 1].mean(), "delta_ece": (v[:, 1] - base[:, 1]).mean()})
    pd.DataFrame(rows).to_csv(T("table_label_noise.csv"), index=False)
    pd.DataFrame(byg).to_csv(T("table_msge_by_grade.csv"), index=False)


def external():
    rows = []
    for name in ["idrid_grading", "messidor2"]:
        for a in C.ARCHS:
            rs = runs("main", a, "full")
            rs = {s: r for s, r in rs.items() if f"{name}/clean/0" in r}
            if not rs:
                continue
            y = next(iter(rs.values()))[f"y/{name}"]
            w = W(name, len(y))
            s0 = M.stats(probs(rs, f"{name}/clean/0"), y, w)
            keys = [("clean", 0)] + [(c, s) for c in C.CORRUPTIONS for s in C.EXTERNAL_SEVERITIES]
            for c, s in keys:
                st = M.stats(probs(rs, f"{name}/{c}/{s}"), y, w)
                for m in MAIN_METRICS:
                    r = {"dataset": name, "arch": a, "corruption": c, "severity": s, "metric": m, **S(m, st[m])}
                    if c != "clean":
                        dd = M.summarize_diff(st[m], s0[m])
                        r.update({"delta": dd["mean"], "delta_lo": dd["lo"], "delta_hi": dd["hi"], "delta_p": dd["p"]})
                    rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(T("table_external.csv"), index=False)


def training_summary():
    rows, hist = [], []
    for p in glob.glob(os.path.join(C.RUNS, "*", "*", "*", "*", "*", "done.json")):
        with open(p) as f:
            d = json.load(f)
        rows.append(d)
        h = pd.read_csv(os.path.join(os.path.dirname(p), "history.csv"))
        for k, v in d.items():
            if k in ("protocol", "arch", "domain", "lr", "seed"):
                h[k] = v
        hist.append(h)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(T("table_runs.csv"), index=False)
    g = df.groupby(["protocol", "arch", "domain", "lr"]).agg(n=("seed", "count"), best_epoch_mean=("best_epoch", "mean"),
                                                            best_epoch_sd=("best_epoch", "std"), epochs_run_mean=("epochs_run", "mean"),
                                                            best_val_qwk_mean=("best_val_qwk", "mean"), best_val_qwk_sd=("best_val_qwk", "std"),
                                                            train_minutes=("train_seconds", lambda x: x.mean() / 60)).reset_index()
    g.to_csv(T("table_training_summary.csv"), index=False)
    H = pd.concat(hist)
    doms = ["full", "high", "low"]
    fig, axes = plt.subplots(len(doms), len(C.ARCHS), figsize=(4 * len(C.ARCHS), 2.8 * len(doms)), squeeze=False)
    for i, d in enumerate(doms):
        for j, a in enumerate(C.ARCHS):
            ax = axes[i, j]
            for protocol, col in [("main", "#1f77b4"), ("v1", "#d62728")]:
                lr = lr_of(protocol, a) if (protocol != "main" or a in LR) else None
                h = H[(H.protocol == protocol) & (H.arch == a) & (H.domain == d)]
                if lr is not None:
                    h = h[np.isclose(h.lr, lr)]
                if not len(h):
                    continue
                q = h.groupby("epoch")["val_qwk"].agg(["mean", "std", "count"]).reset_index()
                ax.plot(q["epoch"], q["mean"], color=col, label=f"{protocol} protocol")
                ax.fill_between(q["epoch"], q["mean"] - q["std"].fillna(0), q["mean"] + q["std"].fillna(0), color=col, alpha=.15, lw=0)
            if i == 0:
                ax.set_title(C.label(a))
            if j == 0:
                ax.set_ylabel(f"{d} training\nvalidation QWK")
            if i == len(doms) - 1:
                ax.set_xlabel("Epoch")
            ax.grid(alpha=.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig_learning_curves")


def xai_tables():
    fs = glob.glob(os.path.join(C.RUNS, "xai", "*", "seed*", "xai.csv"))
    if not fs:
        return
    df = pd.concat([pd.read_csv(f).assign(arch=f.split(os.sep)[-3], seed=int(f.split(os.sep)[-2][4:])) for f in fs])
    ap = df[df.dataset == "aptos"].copy()
    ap["grade_group"] = ap["grade"].map({0: "0", 1: "1", 2: "2", 3: "3-4", 4: "3-4"})
    rows = []
    for gg in ["all", "0", "1", "2", "3-4"]:
        sub = ap if gg == "all" else ap[ap.grade_group == gg]
        for (a, c, s), g in sub.groupby(["arch", "corruption", "severity"]):
            per = g.groupby("seed")["energy_on_retina"].mean()
            img = g.groupby("image")["energy_on_retina"].mean()
            rows.append({"arch": a, "corruption": c, "severity": s, "grade_group": gg, "n_images": img.size,
                         "mean": per.mean(), "sd_seeds": per.std(ddof=1) if len(per) > 1 else np.nan,
                         "p05_images": img.quantile(.05), "frac_below_0.9": (img < .9).mean()})
    pd.DataFrame(rows).to_csv(T("table_xai_energy.csv"), index=False)
    idr = df[df.dataset == "idrid_seg"]
    if len(idr):
        rows = []
        for (a, c, s), g in idr.groupby(["arch", "corruption", "severity"]):
            r = {"arch": a, "corruption": c, "severity": s, "n_images": g["image"].nunique()}
            for m in ["lesion_lift", "lesion_fraction", "pointing_hit", "pixel_auc", "energy_on_retina", "lift_MA", "lift_HE", "lift_EX", "lift_SE"]:
                per = g.groupby("seed")[m].mean()
                r[m] = per.mean()
                r[m + "_sd"] = per.std(ddof=1) if len(per) > 1 else np.nan
            img = g.groupby("image")["lesion_lift"].mean().dropna()
            r["lift_gt1_wilcoxon_p"] = float(sst.wilcoxon(img - 1).pvalue) if len(img) > 5 else np.nan
            rows.append(r)
        pd.DataFrame(rows).to_csv(T("table_xai_lesion.csv"), index=False)


def xai_figs(meta, arr, splits):
    archs = [a for a in C.ARCHS if os.path.exists(os.path.join(C.RUNS, "xai", a, "seed0", "cams.npz"))]
    if not archs:
        return
    z = {a: np.load(os.path.join(C.RUNS, "xai", a, "seed0", "cams.npz")) for a in archs}
    ids = list(z[archs[0]]["aptos/ids"])
    idx = {k: i for i, k in enumerate(meta["id_code"])}
    imgs = [D.center_crop(np.asarray(arr[idx[k]])) for k in ids]
    grades = list(z[archs[0]]["aptos/grades"])
    fig, axes = plt.subplots(1 + len(archs), len(ids), figsize=(2.6 * len(ids), 2.7 * (1 + len(archs))), squeeze=False)
    for j, im in enumerate(imgs):
        axes[0, j].imshow(im)
        axes[0, j].set_title(f"grade {grades[j]}", fontsize=9)
        for i, a in enumerate(archs):
            axes[i + 1, j].imshow(im)
            axes[i + 1, j].imshow(z[a]["aptos/clean/0"][j].astype(np.float32), cmap="jet", alpha=.45)
            if j == 0:
                axes[i + 1, j].set_ylabel(C.label(a))
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save_fig(fig, "fig_xai_examples")
    sv = max(C.XAI_SEVERITIES)
    j = next((k for k, g in enumerate(grades) if g >= C.REFERABLE), 0)
    conds = [("clean", 0)] + [(c, sv) for c in C.CORRUPTIONS]
    fig, axes = plt.subplots(1 + len(archs), len(conds), figsize=(2.4 * len(conds), 2.5 * (1 + len(archs))), squeeze=False)
    raw = np.asarray(arr[idx[ids[j]]])
    for k, (c, s) in enumerate(conds):
        im = D.center_crop(raw if c == "clean" else corrupt(raw, c, s, ids[j]))
        axes[0, k].imshow(im)
        axes[0, k].set_title(NICE[c] + ("" if c == "clean" else f" ({s})"), fontsize=8)
        for i, a in enumerate(archs):
            axes[i + 1, k].imshow(im)
            axes[i + 1, k].imshow(z[a][f"aptos/{c}/{s}"][j].astype(np.float32), cmap="jet", alpha=.45)
            if k == 0:
                axes[i + 1, k].set_ylabel(C.label(a))
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save_fig(fig, "fig_xai_corrupted")
    sm, sa = D.load_external("idrid_seg")
    if sm is not None and "idrid/ids" in z[archs[0]].files:
        masks = np.load(os.path.join(C.CACHE, "idrid_seg_masks_256.npy"), mmap_mode="r")
        n = len(z[archs[0]]["idrid/ids"])
        fig, axes = plt.subplots(n, 1 + len(archs), figsize=(2.8 * (1 + len(archs)), 2.8 * n), squeeze=False)
        for r in range(n):
            im = D.center_crop(np.asarray(sa[r]))
            u = D.center_crop(np.asarray(masks[r]).transpose(1, 2, 0)).any(-1)
            axes[r, 0].imshow(im)
            axes[r, 0].contour(u, levels=[.5], colors="w", linewidths=.6)
            for i, a in enumerate(archs):
                axes[r, i + 1].imshow(im)
                axes[r, i + 1].imshow(z[a]["idrid/clean/0"][r].astype(np.float32), cmap="jet", alpha=.45)
                axes[r, i + 1].contour(u, levels=[.5], colors="w", linewidths=.6)
                if r == 0:
                    axes[r, i + 1].set_title(C.label(a))
        axes[0, 0].set_title("IDRiD lesions")
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        save_fig(fig, "fig_xai_idrid")


def severity_examples(meta, arr, splits):
    te = splits["full"][2]
    i = next((t for t in te if meta["diagnosis"][t] >= C.REFERABLE), te[0])
    raw, key = np.asarray(arr[i]), meta["id_code"][i]
    cs = C.CORRUPTIONS + C.COMPOUNDS
    fig, axes = plt.subplots(len(cs), len(C.SEVERITIES), figsize=(1.9 * len(C.SEVERITIES), 1.9 * len(cs)), squeeze=False)
    for r, c in enumerate(cs):
        for k, s in enumerate(C.SEVERITIES):
            axes[r, k].imshow(D.center_crop(corrupt(raw, c, s, key)))
            axes[r, k].set_xticks([])
            axes[r, k].set_yticks([])
            if r == 0:
                axes[r, k].set_title(f"severity {s}", fontsize=8)
            if k == 0:
                axes[r, k].set_ylabel(NICE[c], fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig_severity_examples")


def latex():
    def tex(df, name, cols, header):
        lines = ["\\begin{tabular}{" + "l" * 2 + "r" * (len(cols) - 2) + "}", "\\hline", " & ".join(header) + " \\\\", "\\hline"]
        for _, r in df.iterrows():
            lines.append(" & ".join(str(r[c]) for c in cols) + " \\\\")
        lines += ["\\hline", "\\end{tabular}"]
        with open(T(name), "w") as f:
            f.write("\n".join(lines))
    p = T("table_benchmark.csv")
    if os.path.exists(p):
        b = pd.read_csv(p)
        b = b[b.protocol == "main"]
        rows = []
        for m in ["accuracy", "qwk", "macro_f1", "referable_sens", "referable_spec", "ref_auc", "auc_macro", "ece", "ece_ts", "msge"]:
            r = {"metric": m.replace("_", " ")}
            for a in C.ARCHS:
                x = b[(b.arch == a) & (b.metric == m)]
                r[a] = "" if not len(x) else (f"{x.iloc[0]['mean']:.3f} $\\pm$ {x.iloc[0]['sd']:.3f}" if np.isnan(x.iloc[0]["lo"]) else
                                             f"{x.iloc[0]['mean']:+.3f} ({x.iloc[0]['lo']:+.3f}, {x.iloc[0]['hi']:+.3f})" if m == "msge" else
                                             f"{x.iloc[0]['mean']:.3f} ({x.iloc[0]['lo']:.3f}--{x.iloc[0]['hi']:.3f})")
            rows.append(r)
        tex(pd.DataFrame(rows), "table_benchmark.tex", ["metric"] + C.ARCHS, ["Metric"] + [C.label(a) for a in C.ARCHS])
    p = T("table_generalisation_gap.csv")
    if os.path.exists(p):
        g = pd.read_csv(p)
        g = g[(g.split == "median")]
        g["arch"] = g["arch"].map(C.label)
        for c in ["in_domain_qwk", "cross_domain_qwk"]:
            g[c] = g[c].map(lambda v: f"{v:.3f}")
        g["gap_s"] = g.apply(lambda r: f"{r['gap']:+.3f} ({r['gap_lo']:+.3f}, {r['gap_hi']:+.3f})", axis=1)
        tex(g, "table_generalisation_gap.tex", ["protocol", "arch", "train", "in_domain_qwk", "cross_domain_qwk", "gap_s"],
            ["Protocol", "Model", "Train", "In-domain QWK", "Cross-domain QWK", "Gap (95\\% CI)"])
    p = T("table_robustness_sev5.csv")
    if os.path.exists(p):
        r = pd.read_csv(p)
        r = r[r.protocol == "main"].copy()
        r["arch"] = r["arch"].map(C.label)
        r["corruption"] = r["corruption"].map(NICE)
        for c in ["qwk", "referable_sens", "referable_spec", "ref_auc", "ece"]:
            r[c] = r[c].map(lambda v: f"{v:.3f}")
        r["msge"] = r["msge"].map(lambda v: f"{v:+.3f}")
        tex(r, "table_robustness_sev5.tex", ["arch", "corruption", "qwk", "referable_sens", "referable_spec", "ref_auc", "ece", "msge"],
            ["Model", "Condition", "QWK", "Ref. sens.", "Ref. spec.", "Ref. AUC", "ECE", "MSGE"])


def run_all():
    C.ensure_dirs()
    p = os.path.join(C.OUT, "lr_selection.json")
    if os.path.exists(p):
        with open(p) as f:
            LR.update(json.load(f))
    cache = {}

    def L():
        if not cache:
            cache["v"] = D.load_aptos()
        return cache["v"]
    steps = [("quality", lambda: quality_tables(L()[0])), ("severity mapping", lambda: severity_mapping(*L())),
             ("severity examples", lambda: severity_examples(*L())), ("training summary", training_summary),
             ("benchmark", benchmark), ("cross-condition", lambda: crosscond(L()[0], L()[2])), ("robustness", robustness),
             ("recalibration", recalibration), ("roc/reliability", roc_and_reliability), ("label noise", label_noise),
             ("external", external), ("xai tables", xai_tables), ("xai figures", lambda: xai_figs(*L())), ("latex", latex)]
    only = os.environ.get("DRFM_STEPS")
    if only:
        keep = set(x.strip() for x in only.split(","))
        steps = [x for x in steps if x[0] in keep]
    for name, fn in steps:
        print("analyze:", name, flush=True)
        try:
            fn()
        except Exception:
            import traceback
            traceback.print_exc()
    pd.DataFrame(list(NONFINITE.values()), columns=["protocol", "arch", "domain", "seed", "n_keys", "n_rows", "keys"]).to_csv(
        T("table_nonfinite_runs.csv"), index=False)
    if only:
        return
    meta, arr, splits = L()
    info = {"lr_selection": LR, "sharpness_threshold": splits["threshold"], "tertile_thresholds": splits["tertile_thresholds"],
            "split_sizes": {k: [len(x) for x in splits[k]] for k in ["full", "high", "low", "thigh", "tlow"]},
            "quality_counts": meta["quality"].value_counts().to_dict(), "quick": C.QUICK, "seeds": C.SEEDS, "n_boot": C.N_BOOT}
    with open(os.path.join(C.OUT, "run_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    for t in sorted(glob.glob(os.path.join(C.TABLES, "*.csv"))):
        print("=" * 70)
        print(os.path.basename(t))
        print("=" * 70)
        d = pd.read_csv(t)
        print(d.head(60).to_string(index=False))
