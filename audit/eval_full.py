# -*- coding: utf-8 -*-
"""Full-validation evaluation of the twelve ablation runs.

Evaluates every checkpoint on the full validation split (550 forged and 475
authentic images), with the letterbox preprocessing used during training:

    resize with the aspect ratio preserved -> centre pad to 518 x 518

For each run it reports
  * per-image pixel F1, precision and recall on the forged images (550);
  * pixel-level and image-level false-positive rates on authentic images (475);
  * pixel F1 stratified by the fraction of positive pixels in the ground truth;
and at the end the cell means, the three factorial effects and the per-seed
values. These are the numbers in Tables III-IV and Fig. 2 of the paper.

Usage:
    python audit/eval_full.py --root <DATA_ROOT> --splits splits/ \
        --configs configs/generated/ --work-dir runs/ [--threshold 0.3]
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F


SIZE = 518
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
BUCKETS = [(0.0, 0.005, "tiny"), (0.005, 0.02, "small"), (0.02, 1.01, "large")]


def load_names(path: Path):
    return [line.strip() for line in path.read_text(encoding="utf-8").split() if line.strip()]


def letterbox(img: np.ndarray):
    """Resize preserving the aspect ratio, then centre pad to SIZE x SIZE."""
    h, w = img.shape[:2]
    scale = min(SIZE / h, SIZE / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)
    return x, (top, top + nh, left, left + nw)


def letterbox_mask(mask: np.ndarray):
    h, w = mask.shape[:2]
    scale = min(SIZE / h, SIZE / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize((mask > 0).astype(np.uint8), (nw, nh),
                      interpolation=cv2.INTER_NEAREST)


def f1_of(pred: np.ndarray, gt: np.ndarray, thr: float):
    p, g = pred > thr, gt.astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def prepare(root: Path, forged, authentic):
    cache = {}
    for name in forged + authentic:
        img = cv2.imread(str(root / "train_images" / "forged" / f"{name}.png"))
        if img is None:
            img = cv2.imread(str(root / "train_images" / "authentic" / f"{name}.png"))
        if img is None:
            continue
        cache[name] = letterbox(img)

    masks = {}
    for name in forged:
        npy = root / "train_masks" / f"{name}.npy"
        if not npy.exists():
            continue
        arr = np.load(npy)
        masks[name] = letterbox_mask((arr > 0).any(axis=0) if arr.ndim == 3 else arr)
    return cache, masks


@torch.no_grad()
def predict(model, names, cache, batch=8):
    out = {}
    for i in range(0, len(names), batch):
        chunk = [n for n in names[i:i + batch] if n in cache]
        if not chunk:
            continue
        xs = torch.from_numpy(np.stack([cache[n][0] for n in chunk])).cuda()
        feats = model.extract_feat(xs)
        prob = torch.sigmoid(model.decode_head(feats))
        prob = F.interpolate(prob, size=(SIZE, SIZE), mode="bilinear",
                             align_corners=False)
        prob = prob.squeeze(1).cpu().numpy()
        for j, name in enumerate(chunk):
            t0, t1, l0, l1 = cache[name][1]
            out[name] = prob[j][t0:t1, l0:l1]
    return out


def load_model(cfg_path: Path, ckpt: Path):
    from mmengine.config import Config
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(cfg_path))
    model = MODELS.build(cfg.model)
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--splits", default=Path("splits"), type=Path)
    ap.add_argument("--configs", default=Path("configs/generated"), type=Path)
    ap.add_argument("--work-dir", default=Path("runs"), type=Path)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    forged = [n for n in load_names(args.splits / "val.txt")
              if (args.root / "train_images" / "forged" / f"{n}.png").exists()]
    authentic = [n for n in load_names(args.splits / "val.txt")
                 if (args.root / "train_images" / "authentic" / f"{n}.png").exists()]
    print(f"forged: {len(forged)} | authentic: {len(authentic)}")

    cache, masks = prepare(args.root, forged, authentic)
    print(f"prepared: {len(cache)} images, {len(masks)} masks")

    results = {}
    for cfg_path in sorted(glob.glob(str(args.configs / "fv_E*_*.py"))):
        m = re.search(r"fv_(E\d)_(\d+)\.py$", cfg_path)
        if not m:
            continue
        cell, seed = m.group(1), m.group(2)
        ckpt_dir = args.work_dir / f"fv_{cell}_{seed}"
        ckpts = sorted(ckpt_dir.glob("*.pth"),
                       key=lambda p: int(p.stem.split("_")[-1]))
        if not ckpts:
            print(f"skip {cell}/{seed}: no checkpoint in {ckpt_dir}")
            continue
        model = load_model(Path(cfg_path), ckpts[-1])

        fg = predict(model, forged, cache, args.batch)
        au = predict(model, authentic, cache, args.batch)

        f1s, per, bkt = [], [], {b[2]: [] for b in BUCKETS}
        for name, prob in fg.items():
            gt = masks.get(name)
            if gt is None:
                continue
            f1, precision, recall = f1_of(prob, gt, args.threshold)
            f1s.append(f1)
            per.append((f1, precision, recall))
            ratio = float(gt.mean())
            for lo, hi, label in BUCKETS:
                if lo <= ratio < hi:
                    bkt[label].append(f1)
                    break

        fp_pix = [float((p > args.threshold).mean()) for p in au.values()]
        fp_img = [float((p > args.threshold).any()) for p in au.values()]

        results[(cell, seed)] = dict(
            f1=float(np.mean([p[0] for p in per])),
            precision=float(np.mean([p[1] for p in per])),
            recall=float(np.mean([p[2] for p in per])),
            bkt={k: (float(np.mean(v)) if v else float("nan")) for k, v in bkt.items()},
            fp_pix=float(np.mean(fp_pix)),
            fp_img=float(np.mean(fp_img)),
            n=len(per),
        )
        r = results[(cell, seed)]
        print(f"{cell} seed{seed}: F1={r['f1']:.4f} P={r['precision']:.4f} "
              f"R={r['recall']:.4f} FP_img={r['fp_img']:.4f} n={r['n']}")

    print()
    print(f"{'cell':>5} {'F1':>8} {'std':>7} {'P':>7} {'R':>7} "
          f"{'FP_img':>7} {'tiny':>7} {'small':>7} {'large':>7}")
    means = {}
    for cell in ("E1", "E2", "E3", "E4"):
        rs = [results[(cell, s)] for s in ("42", "2024", "7") if (cell, s) in results]
        if not rs:
            continue
        f1s = [r["f1"] for r in rs]
        means[cell] = float(np.mean(f1s))
        print(f"{cell:>5} {np.mean(f1s):8.4f} {np.std(f1s):7.4f} "
              f"{np.mean([r['precision'] for r in rs]):7.4f} "
              f"{np.mean([r['recall'] for r in rs]):7.4f} "
              f"{np.mean([r['fp_img'] for r in rs]):7.4f} "
              f"{np.mean([r['bkt']['tiny'] for r in rs]):7.4f} "
              f"{np.mean([r['bkt']['small'] for r in rs]):7.4f} "
              f"{np.mean([r['bkt']['large'] for r in rs]):7.4f}")

    if len(means) == 4:
        delta_d = ((means["E2"] + means["E4"]) - (means["E1"] + means["E3"])) / 2
        delta_m = ((means["E3"] + means["E4"]) - (means["E1"] + means["E2"])) / 2
        delta_dm = means["E4"] - means["E3"] - means["E2"] + means["E1"]
        print()
        print(f"Delta_D  = {delta_d:+.4f}")
        print(f"Delta_M  = {delta_m:+.4f}")
        print(f"Delta_DM = {delta_dm:+.4f}")

    print()
    print("per-seed:")
    for (cell, seed), r in sorted(results.items()):
        print(f"  {cell} seed{seed}: F1={r['f1']:.4f} "
              f"FP_img={r['fp_img']:.4f} n={r['n']}")


if __name__ == "__main__":
    sys.exit(main())
