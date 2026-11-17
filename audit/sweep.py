# -*- coding: utf-8 -*-
"""Threshold sweep, false-alarm rate, bootstrap intervals and false-positive examples.

This is the script used for Section V-D of the paper. It expects the layout that
the accompanying Kaggle notebook builds, namely

    /kaggle/working/Cell_Pair/{images,instances}/<name>.png
    /kaggle/working/Cell_Pair/val.txt
    /kaggle/working/configs/fv_<cell>_<seed>.py
    /kaggle/working/work_dir/fv_<cell>_<seed>/iter_1000.pth

Edit the WORK/ROOT constants below when running outside Kaggle. Outputs:

  * per-cell F1 / TPR / FAR at the default threshold with 95% bootstrap intervals
  * the best F1 reachable under a 5% and 10% false-alarm budget
  * sweep.json with the F1(tau) and FAR(tau) curves
  * fp_examples.png with the authentic images flagged most strongly by E4/seed42
"""


import glob
import json
import re
import sys

import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, "/kaggle/working/code")
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

WORK = "/kaggle/working"
ROOT = f"{WORK}/Cell_Pair"
SIZE, BATCH = 518, 8
THR = 0.3
NBINS = 2001
TAU = np.arange(0.0, 1.001, 0.01)
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def letterbox(img):
    h, w = img.shape[:2]
    s = min(SIZE / h, SIZE / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    return ((rgb - MEAN) / STD).transpose(2, 0, 1), (top, top + nh, left, left + nw)


def letterbox_mask(m):
    h, w = m.shape[:2]
    s = min(SIZE / h, SIZE / w)
    return cv2.resize((m > 0).astype(np.uint8),
                      (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_NEAREST)


def read_names(path):
    return [ln.strip() for ln in open(path) if ln.strip()]


@torch.no_grad()
def predict(model, names, cache):
    out = {}
    for i in range(0, len(names), BATCH):
        chunk = names[i:i + BATCH]
        xs = torch.from_numpy(np.stack([cache[n][0] for n in chunk])).cuda()
        feats = model.extract_feat(xs)
        pr = torch.sigmoid(model.decode_head(feats))
        pr = F.interpolate(pr, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        pr = pr.squeeze(1).cpu().numpy()
        for j, n in enumerate(chunk):
            t0, t1, l0, l1 = cache[n][1]
            out[n] = pr[j][t0:t1, l0:l1]
    return out


def hist_of(prob, mask=None):
    idx = np.clip((prob * (NBINS - 1)).astype(np.int32), 0, NBINS - 1).ravel()
    if mask is None:
        return None, np.bincount(idx, minlength=NBINS).astype(np.float64), 0, int(idx.size)
    gt = mask.ravel().astype(bool)
    return (np.bincount(idx[gt], minlength=NBINS).astype(np.float64),
            np.bincount(idx[~gt], minlength=NBINS).astype(np.float64),
            int(gt.sum()), int((~gt).sum()))


def curve_from_hist(hg, hb, n_gt, n_bg, taus):
    ks = np.clip((taus * (NBINS - 1)).astype(np.int32), 0, NBINS - 1)
    tail_gt = np.cumsum(hg[::-1])[::-1]
    tail_bg = np.cumsum(hb[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = tail_gt[ks]
        fp = tail_bg[ks]
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0.0)
        rec = np.where(n_gt > 0, tp / max(n_gt, 1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0)
    return f1, prec, rec


def load_model(cfgp, ckpt):
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(cfgp)
    model = MODELS.build(cfg.model)
    st = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(st.get("state_dict", st), strict=False)
    return model.cuda().eval()


def bootstrap_ci(values, B=1000, seed=0):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, size=v.size, replace=True).mean() for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    val = read_names(f"{ROOT}/val.txt")
    forged = [n for n in val if not n.endswith("_authentic")]
    auth = [n for n in val if n.endswith("_authentic")]
    print(f"val forged={len(forged)} authentic={len(auth)}", flush=True)
    assert forged and auth, "val.txt 为空或路径错误"

    cache, masks = {}, {}
    for n in forged + auth:
        img = cv2.imread(f"{ROOT}/images/{n}.png")
        if img is None:
            raise SystemExit(f"图片缺失: {ROOT}/images/{n}.png")
        cache[n] = letterbox(img)
    for n in forged:
        inst = cv2.imread(f"{ROOT}/instances/{n}.png", cv2.IMREAD_UNCHANGED)
        if inst is None:
            raise SystemExit(f"实例图缺失: {ROOT}/instances/{n}.png")
        masks[n] = letterbox_mask((inst > 0).astype(np.uint8))
    print("prepared", len(cache), "images", flush=True)

    per_cell = {}
    example_probs = {}

    for cfgp in sorted(glob.glob(f"{WORK}/configs/fv_E*_*.py")):
        m = re.search(r"fv_(E\d)_(\d+)\.py$", cfgp)
        if not m:
            continue
        cell, seed = m.group(1), m.group(2)
        ck = sorted(glob.glob(f"{WORK}/work_dir/fv_{cell}_{seed}/*.pth"),
                    key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))
        if not ck:
            print("skip", cell, seed, flush=True)
            continue
        model = load_model(cfgp, ck[-1])

        fg = predict(model, forged, cache)
        au = predict(model, auth, cache)

        hg = np.zeros(NBINS)
        hb = np.zeros(NBINS)
        n_gt = n_bg = 0
        f1_images, tpr_images = [], []
        for n, p in fg.items():
            g = masks[n]
            h_g, h_b, a, b = hist_of(p, g)
            hg += h_g
            hb += h_b
            n_gt += a
            n_bg += b
            tp = int(((p > THR) & (g > 0)).sum())
            fp = int(((p > THR) & (g == 0)).sum())
            fn = int(((p <= THR) & (g > 0)).sum())
            f1_images.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
            tpr_images.append(float((p > THR).any()))

        far_flags, far_pix = [], []
        for n, p in au.items():
            far_flags.append(float((p > THR).any()))
            far_pix.append(float((p > THR).mean()))
            if cell == "E4" and seed == "42":
                example_probs[n] = (p, float((p > THR).mean()))

        f1c, prc, rec = curve_from_hist(hg, hb, n_gt, n_bg, TAU)
        maxp = np.array([p.max() for p in au.values()])
        far_any = np.array([(maxp > t).mean() for t in TAU])
        far_pix_curve = np.array([np.mean([(p > t).mean() for p in au.values()])
                                  for t in TAU])

        rec_ = dict(
            f1=float(np.mean(f1_images)),
            tpr_img=float(np.mean(tpr_images)),
            far_any=float(np.mean(far_flags)),
            far_pix=float(np.mean(far_pix)),
            f1_ci=bootstrap_ci(f1_images),
            far_ci=bootstrap_ci(far_flags),
            tau=TAU.tolist(),
            f1_curve=f1c.tolist(),
            far_curve=far_any.tolist(),
            farpix_curve=far_pix_curve.tolist(),
        )
        per_cell.setdefault(cell, []).append(rec_)
        print(f"{cell} seed{seed}: F1={rec_['f1']:.4f} TPR={rec_['tpr_img']:.3f} "
              f"FAR={rec_['far_any']:.4f}", flush=True)

    summary = {}
    for cell, runs in per_cell.items():
        taus = np.array(runs[0]["tau"])
        f1c = np.mean([r["f1_curve"] for r in runs], axis=0)
        far = np.mean([r["far_curve"] for r in runs], axis=0)
        picks = {}
        for target in (0.05, 0.10):
            ok = np.where(far <= target)[0]
            k = int(ok[0]) if ok.size else None
            picks[f"far<={int(target * 100)}"] = (
                None if k is None else
                dict(tau=float(taus[k]), f1=float(f1c[k]), far=float(far[k])))
        summary[cell] = dict(
            f1_at_03=float(np.mean([r["f1"] for r in runs])),
            tpr_at_03=float(np.mean([r["tpr_img"] for r in runs])),
            far_at_03=float(np.mean([r["far_any"] for r in runs])),
            f1_ci=[float(np.mean([r["f1_ci"][0] for r in runs])),
                   float(np.mean([r["f1_ci"][1] for r in runs]))],
            far_ci=[float(np.mean([r["far_ci"][0] for r in runs])),
                    float(np.mean([r["far_ci"][1] for r in runs]))],
            tau=taus.tolist(), f1_curve=f1c.tolist(), far_curve=far.tolist(),
            operating_points=picks,
        )

    print()
    print(f"{'cell':>5} {'F1@0.3':>8} {'95%CI':>16} {'TPR@0.3':>8} "
          f"{'FAR@0.3':>8} {'95%CI':>16} {'FAR<=5%':>18} {'FAR<=10%':>18}")
    for cell, s in summary.items():
        def fmt(p, key):
            return "  n/a  " if not p else f"{p['f1']:.3f}@tau={p['tau']:.2f}"
        print(f"{cell:>5} {s['f1_at_03']:8.4f} "
              f"[{s['f1_ci'][0]:.3f},{s['f1_ci'][1]:.3f}] "
              f"{s['tpr_at_03']:8.3f} {s['far_at_03']:8.4f} "
              f"[{s['far_ci'][0]:.3f},{s['far_ci'][1]:.3f}] "
              f"{fmt(s['operating_points']['far<=5'], 'F1'):>18} "
              f"{fmt(s['operating_points']['far<=10'], 'F1'):>18}", flush=True)

    with open(f"{WORK}/sweep.json", "w") as f:
        json.dump(summary, f)
    print("\nwrote /kaggle/working/sweep.json", flush=True)

    if example_probs:
        top = sorted(example_probs.items(), key=lambda kv: -kv[1][1])[:6]
        print("\ntop false positives:", [(n, round(v[1], 3)) for n, v in top], flush=True)
        tiles = []
        for n, (p, ratio) in top:
            img = cv2.imread(f"{ROOT}/images/{n}.png")
            h, w = img.shape[:2]
            s = min(SIZE / h, SIZE / w)
            nh, nw = int(round(h * s)), int(round(w * s))
            pv = cv2.resize(p, (nw, nh))
            base = cv2.resize(img, (nw, nh))
            m = (pv > THR).astype(np.uint8)
            overlay = base.copy()
            overlay[m > 0] = (0.4 * overlay[m > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            heat = cv2.applyColorMap(((pv / max(pv.max(), 1e-6)) * 255).astype(np.uint8),
                                     cv2.COLORMAP_JET)
            tile = np.hstack([base, overlay, heat])
            cv2.putText(tile, f"{n}  FPpix={ratio:.3f}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        hmin = min(t.shape[0] for t in tiles)
        wmin = min(t.shape[1] for t in tiles)
        rows = [np.hstack([cv2.resize(t, (wmin, hmin)) for t in tiles[i:i + 3]])
                for i in (0, 3)]
        cv2.imwrite(f"{WORK}/fp_examples.png", np.vstack(rows))
        print("wrote /kaggle/working/fp_examples.png", flush=True)


if __name__ == "__main__":
    main()
