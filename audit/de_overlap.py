#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the de-overlapped training list used in Section 2.8.

The near-duplicate audit (audit/near_duplicate_audit.py) writes a JSON file with
the training identifiers that share near-duplicate content with a validation
image. This script removes exactly those identifiers from the training list and
writes the de-overlapped list; the validation list is unchanged.

Usage:
    python audit/de_overlap.py \
        --train-list splits/train.txt \
        --exclusions data/exclusions_train_ids.txt \
        --out splits/train_deleaked.txt
"""

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-list", type=Path, default=Path("splits/train.txt"))
    ap.add_argument("--exclusions", type=Path, default=Path("data/exclusions_train_ids.txt"))
    ap.add_argument("--out", type=Path, default=Path("splits/train_deleaked.txt"))
    args = ap.parse_args()

    train = [line.strip() for line in args.train_list.read_text().splitlines() if line.strip()]
    excluded = {line.strip() for line in args.exclusions.read_text().splitlines() if line.strip()}
    kept = [i for i in train if i not in excluded]

    args.out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"train ids: {len(train)}")
    print(f"excluded (near-duplicate with a validation image): {len(excluded)}")
    print(f"kept: {len(kept)}")
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
