#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Near-duplicate audit of the ReCoD.ai LUC split.

The audit has two stages, matching Section 2.2 of the accompanying study:

1. Perceptual hashing. Every image is reduced to an 8x8 difference hash
   (dHash); images whose hashes differ by at most ``--max-hamming`` bits are
   grouped into clusters (union-find).
2. Verification. Every candidate pair that spans the train/validation split is
   verified on 64x64 grey-scale thumbnails with the normalised correlation;
   only pairs above ``--corr-thr`` are kept.

Outputs a JSON file with the clusters, the verified cross-split pairs, the
affected validation identifiers and the training identifiers that should be
removed for the de-overlapped design (Section 2.8 of the main text).

Usage:
    python audit/near_duplicate_audit.py \
        --data-root /path/to/recodai-luc-scientific-image-forgery-detection \
        --train-list splits/train.txt --val-list splits/val.txt \
        --out data/near_duplicate_audit_v2.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

HASH_SIZE = 8
THUMB = 64


def dhash(path: Path, size: int = HASH_SIZE) -> np.uint64:
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return np.uint64(value)


def thumb(path: Path, size: int = THUMB) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float64).ravel()
    a -= a.mean()
    norm = np.linalg.norm(a)
    return a / norm if norm else a


def normalised_correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def classify(paths, split_of):
    """Return a list of (name, kind, split, path) entries."""
    entries = []
    for path in paths:
        name = f"{path.stem}#{path.parent.name}"
        entries.append((name, path.parent.name, split_of[path.stem], path))
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--train-list", type=Path, default=Path("splits/train.txt"))
    ap.add_argument("--val-list", type=Path, default=Path("splits/val.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/near_duplicate_audit_v2.json"))
    ap.add_argument("--max-hamming", type=int, default=2)
    ap.add_argument("--corr-thr", type=float, default=0.95)
    args = ap.parse_args()

    root = args.data_root
    forged_dir = root / "train_images" / "forged"
    auth_dir = root / "train_images" / "authentic"

    train_ids = {line.strip() for line in args.train_list.read_text().splitlines() if line.strip()}
    val_ids = {line.strip() for line in args.val_list.read_text().splitlines() if line.strip()}
    split_of = {}
    for i in train_ids:
        split_of[i] = "train"
    for i in val_ids:
        split_of[i] = "val"

    forged = sorted(p for p in forged_dir.glob("*.png") if p.stem in split_of)
    auth = sorted(p for p in auth_dir.glob("*.png") if p.stem in split_of)
    entries = classify(forged, split_of) + classify(auth, split_of)
    print(f"images: {len(entries)} (forged {len(forged)}, authentic {len(auth)})")

    hashes = np.array([dhash(e[3]) for e in entries], dtype=np.uint64)
    n = len(entries)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Chunked pairwise Hamming distance (n^2/2 comparisons).
    chunk = 512
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        xor = hashes[start:stop, None] ^ hashes[None, :]
        close = np.bitwise_count(xor) <= np.uint64(args.max_hamming)
        for i in range(stop - start):
            for j in np.nonzero(close[i])[0]:
                j = int(j)
                if j > start + i:
                    union(start + i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    multi = {k: v for k, v in clusters.items() if len(v) > 1}
    cross = {k: v for k, v in multi.items() if len({entries[i][2] for i in v}) > 1}
    print(f"clusters with >= 2 images: {len(multi)}")
    print(f"clusters spanning train/val: {len(cross)}")

    # Verify candidate cross-split pairs on thumbnails.
    cache = {}
    def get_thumb(i):
        if i not in cache:
            cache[i] = thumb(entries[i][3])
        return cache[i]

    pairs = []
    for members in cross.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if entries[i][2] == entries[j][2]:
                    continue
                corr = normalised_correlation(get_thumb(i), get_thumb(j))
                if corr >= args.corr_thr:
                    pairs.append(dict(a=entries[i][0], b=entries[j][0],
                                      split_a=entries[i][2], split_b=entries[j][2],
                                      corr=round(corr, 4)))

    affected_val = sorted({e[0].split("#")[0] for pair in pairs
                           for e, split in (((pair["a"], pair["split_a"]),
                                             (pair["b"], pair["split_b"]))) if split == "val"})
    deleak_train = sorted({e[0].split("#")[0] for pair in pairs
                           for e, split in (((pair["a"], pair["split_a"]),
                                             (pair["b"], pair["split_b"]))) if split == "train"})

    payload = dict(
        images=n,
        max_hamming=args.max_hamming,
        corr_thr=args.corr_thr,
        clusters=len(multi),
        cross_split_clusters=len(cross),
        verified_pairs=len(pairs),
        affected_val_ids=affected_val,
        affected_val_count=len(affected_val),
        val_ids_total=len(val_ids),
        pairs=pairs,
        deleak_train_ids=deleak_train,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"verified cross-split pairs: {len(pairs)}")
    print(f"affected validation ids: {len(affected_val)} / {len(val_ids)}")
    print(f"train ids to remove for the de-overlapped design: {len(deleak_train)}")
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
