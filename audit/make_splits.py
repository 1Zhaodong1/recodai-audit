# -*- coding: utf-8 -*-
"""Deterministic reconstruction of the 80/20 split used in the paper.

The audited work does not release its private test split, so the reproduction
uses a random split over *identifiers*: the forged image and its authentic
counterpart always fall on the same side, which prevents pair leakage.

Split rule (matches the paper): sort all identifiers, shuffle with
``random.Random(seed)``, take the first 20 % as validation. Identifiers with a
forged image and identifiers with only an authentic image are both included in
the shuffle.

Usage:
    python audit/make_splits.py --root <DATA_ROOT> --out splits/
    python audit/make_splits.py --root <DATA_ROOT> --out splits/ --check
"""

import argparse
import random
from pathlib import Path


def collect_ids(root: Path):
    forged_dir = root / "train_images" / "forged"
    auth_dir = root / "train_images" / "authentic"
    forged = sorted(p.stem for p in forged_dir.glob("*.png"))
    authentic = sorted(p.stem for p in auth_dir.glob("*.png"))
    if not forged:
        raise SystemExit(f"no forged images under {forged_dir}")
    return forged, authentic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="dataset root with train_images/ and train_masks/")
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--check", action="store_true",
                    help="compare against the split files in this repository")
    args = ap.parse_args()

    forged, authentic = collect_ids(args.root)
    ids = sorted(set(forged) | set(authentic))
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * args.val_frac)
    val_ids = set(ids[:n_val])
    train_ids = [i for i in ids if i not in val_ids]
    val_ids_sorted = [i for i in ids if i in val_ids]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (args.out / "val.txt").write_text("\n".join(val_ids_sorted) + "\n", encoding="utf-8")

    print(f"identifiers      : {len(ids)}")
    print(f"forged images    : {len(forged)}")
    print(f"authentic images : {len(authentic)}")
    print(f"train ids        : {len(train_ids)}")
    print(f"val ids          : {len(val_ids_sorted)}")
    print(f"val forged       : {sum(1 for i in forged if i in val_ids)}")
    print(f"val authentic    : {sum(1 for i in authentic if i in val_ids)}")
    print(f"written          : {args.out}/train.txt, {args.out}/val.txt")

    if args.check:
        ok = True
        for name in ("train.txt", "val.txt"):
            ref = Path(__file__).resolve().parent.parent / "splits" / name
            if not ref.exists():
                print(f"[check] reference {ref} missing, skipped")
                continue
            same = ref.read_text(encoding="utf-8").split() == \
                (args.out / name).read_text(encoding="utf-8").split()
            print(f"[check] {name}: {'match' if same else 'MISMATCH'}")
            ok = ok and same
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
