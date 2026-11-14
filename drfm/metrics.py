import numpy as np
from scipy import stats as st
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

from . import config as C

K = C.NUM_CLASSES
_I, _J = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
_W2 = (_I - _J) ** 2 / (K - 1) ** 2


def softmax(z, t=1.0):
    z = np.asarray(z, dtype=np.float64) / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def boot_weights(n, reps=None, seed=0):
    reps = C.N_BOOT if reps is None else reps
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (reps, n))
    w = np.stack([np.bincount(r, minlength=n) for r in idx]).astype(np.float32)
    return np.vstack([np.ones((1, n), np.float32), w])


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def from_conf(o):
    o = o.astype(np.float64)
    n = o.sum((-1, -2))
    row, col = o.sum(-1), o.sum(-2)
    diag = np.diagonal(o, axis1=-2, axis2=-1)
    e = row[..., :, None] * col[..., None, :] / np.where(n > 0, n, 1)[..., None, None]
    num, den = (_W2 * o).sum((-1, -2)), (_W2 * e).sum((-1, -2))
    rec, prec = _div(diag, row), _div(diag, col)
    rec0, prec0 = np.nan_to_num(rec), np.nan_to_num(prec)
    f1 = np.nan_to_num(_div(2 * rec0 * prec0, rec0 + prec0))
    out = {
        "accuracy": _div(diag.sum(-1), n),
        "qwk": 1 - _div(num, den),
        "macro_f1": f1.mean(-1),
        "referable_sens": _div(o[..., C.REFERABLE:, C.REFERABLE:].sum((-1, -2)), o[..., C.REFERABLE:, :].sum((-1, -2))),
        "referable_spec": _div(o[..., :C.REFERABLE, :C.REFERABLE].sum((-1, -2)), o[..., :C.REFERABLE, :].sum((-1, -2))),
        "msge": _div(((_J - _I) * o).sum((-1, -2)), n),
    }
    for k in range(K):
        out[f"recall_{k}"] = rec[..., k]
    return out


def weighted_auc(score, pos, w):
    order = np.argsort(score, kind="mergesort")
    s, p = score[order], pos[order].astype(np.float32)
    ww = w[:, order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    wp = np.add.reduceat(ww * p, starts, axis=1)
    wn = np.add.reduceat(ww * (1 - p), starts, axis=1)
    below = np.cumsum(wn, axis=1) - wn
    num = (wp * (below + 0.5 * wn)).sum(1)
    return _div(num, wp.sum(1) * wn.sum(1))


def ece_terms(probs, y, bins=10):
    conf = probs.max(-1)
    correct = (probs.argmax(-1) == y).astype(np.float64)
    b = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    g = np.zeros(conf.shape + (bins,))
    np.put_along_axis(g, b[..., None], (correct - conf)[..., None], axis=-1)
    return g


def stats(probs, y, w):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 2:
        probs = probs[None]
    y = np.asarray(y).astype(int)
    pred = probs.argmax(-1)
    cell = y[None] * K + pred
    m = np.zeros(cell.shape + (K * K,), np.float32)
    np.put_along_axis(m, cell[..., None], 1.0, axis=-1)
    o = np.einsum("rn,snk->srk", w, m).reshape(probs.shape[0], w.shape[0], K, K)
    out = from_conf(o)
    n = w.sum(1)[None]
    out["ece"] = np.abs(np.einsum("rn,snb->srb", w, ece_terms(probs, y).astype(np.float32))).sum(-1) / n
    pos = y >= C.REFERABLE
    ref = probs[..., C.REFERABLE:].sum(-1)
    out["ref_auc"] = np.stack([weighted_auc(ref[s], pos, w) for s in range(probs.shape[0])])
    out["mean_pred"] = (w @ pred.T.astype(np.float32)).T / n
    return out


def shift_stats(pred, pred_clean, w):
    d = (pred - pred_clean).astype(np.float32)
    n = w.sum(1)[None]
    return {"pred_shift": (w @ d.T).T / n, "up_rate": (w @ (d > 0).T.astype(np.float32)).T / n,
            "down_rate": (w @ (d < 0).T.astype(np.float32)).T / n}


def point_metrics(probs, y):
    out = {k: float(v[0, 0]) for k, v in stats(probs, y, np.ones((1, len(y)), np.float32)).items()}
    try:
        present = [c for c in range(K) if (y == c).sum() > 0]
        yoh = np.eye(K)[y][:, present]
        out["auc_macro"] = float(roc_auc_score(yoh, probs[:, present], average="macro", multi_class="ovr"))
    except Exception:
        out["auc_macro"] = float("nan")
    return out


def summarize(arr, basic=False):
    arr = np.asarray(arr, dtype=np.float64)
    m = np.nanmean(arr, 0)
    pt, rep = m[0], m[1:]
    lo, hi = np.nanpercentile(rep, [2.5, 97.5]) if np.isfinite(rep).any() else (np.nan, np.nan)
    if basic:
        bias = float(np.nanmean(rep)) - pt
        lo, hi = max(0.0, lo - bias), hi - bias
    sd = float(np.nanstd(arr[:, 0], ddof=1)) if arr.shape[0] > 1 else float("nan")
    return {"mean": float(pt), "sd": sd, "lo": float(lo), "hi": float(hi)}


def boot_p(d):
    d = np.asarray(d, dtype=np.float64)[1:]
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def summarize_diff(a, b):
    d = np.nanmean(np.asarray(a, np.float64), 0) - np.nanmean(np.asarray(b, np.float64), 0)
    lo, hi = np.nanpercentile(d[1:], [2.5, 97.5])
    return {"mean": float(d[0]), "lo": float(lo), "hi": float(hi), "p": boot_p(d)}


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(st.ttest_ind(a, b, equal_var=False).pvalue)


def holm(p):
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out


def fit_temperature(logits, y):
    logits, y = np.asarray(logits, np.float64), np.asarray(y).astype(int)

    def nll(lt):
        p = softmax(logits, np.exp(lt))
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def _midrank(x):
    return st.rankdata(x)


def delong(y, s1, s2):
    y = np.asarray(y).astype(bool)
    preds = np.vstack([s1, s2]).astype(np.float64)
    if not np.isfinite(preds).all():
        return float("nan"), float("nan"), float("nan")
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")
    tx = np.stack([_midrank(r) for r in pos])
    ty = np.stack([_midrank(r) for r in neg])
    tz = np.stack([_midrank(r) for r in np.hstack([pos, neg])])
    auc = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(auc[0]), float(auc[1]), 1.0
    z = (auc[0] - auc[1]) / np.sqrt(var)
    return float(auc[0]), float(auc[1]), float(2 * st.norm.sf(abs(z)))


def recalibrated(ref_val, yv, ref_test, yt, target_sens):
    pv = yv >= C.REFERABLE
    cand = np.unique(ref_val[pv])[::-1]
    thr = cand[-1] if len(cand) else 0.5
    for t in cand:
        if (ref_val[pv] >= t).mean() >= target_sens:
            thr = t
            break
    pt = yt >= C.REFERABLE
    pr = ref_test >= thr
    return float(thr), float(pr[pt].mean()), float((~pr[~pt]).mean())


def spec_at_sens(score, y, target):
    pos = y >= C.REFERABLE
    sp = np.sort(score[pos])[::-1]
    k = int(np.ceil(target * len(sp))) - 1
    thr = sp[min(max(k, 0), len(sp) - 1)]
    return float((score[~pos] < thr).mean())


def adjacent_noise(y, rate, rng):
    y = y.copy()
    flip = rng.random(len(y)) < rate
    step = np.where(rng.random(len(y)) < 0.5, -1, 1)
    step = np.where(y == 0, 1, np.where(y == K - 1, -1, step))
    y[flip] = y[flip] + step[flip]
    return y
