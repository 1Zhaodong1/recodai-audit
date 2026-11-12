# -*- coding: utf-8 -*-
"""Derive the twelve factorial-ablation configurations from a base config.

The audited repository ships one training config for the mixed setting. The
factorial design varies exactly two keys:

    Dice entry     : dict(type='DiceLoss', use_sigmoid=True, loss_weight=2.0)
                     dict(type='BinaryDiceLoss', loss_weight=2.0)
    real-mask mix  : p_real=0.0   (synthetic supervision only)
                     p_real=0.5   (mixed supervision)

plus three seeds. Everything else is inherited.

Usage:
    python configs/make_configs.py --base <AUDITED_REPO>/configs/mix.py \
        --out configs/generated --iters 1000

The base config is expected to contain the two strings above and the line
`work_dir = '/kaggle/working/work_dir/mix'`; adjust the constants below if the
audited file differs.
"""

import argparse
from pathlib import Path

BROKEN = "dict(type='DiceLoss', use_sigmoid=True, loss_weight=2.0)"
FIXED = "dict(type='BinaryDiceLoss', loss_weight=2.0)"
WORK_DIR_LINE = "work_dir = '/kaggle/working/work_dir/mix'"

CELLS = {
    "E1": (BROKEN, "p_real=0.0"),
    "E2": (FIXED, "p_real=0.0"),
    "E3": (BROKEN, "p_real=0.5"),
    "E4": (FIXED, "p_real=0.5"),
}
SEEDS = (42, 2024, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("configs/generated"))
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=3000)
    args = ap.parse_args()

    base = args.base.read_text(encoding="utf-8")
    missing = [s for s in (BROKEN, "p_real=0.5", WORK_DIR_LINE) if s not in base]
    if missing:
        raise SystemExit("base config does not contain the expected lines:\n  "
                         + "\n  ".join(missing))

    args.out.mkdir(parents=True, exist_ok=True)
    for cell, (dice, p_real) in CELLS.items():
        for seed in SEEDS:
            cfg = base.replace(FIXED, dice)
            if p_real == "p_real=0.0":
                cfg = cfg.replace("p_real=0.5", "p_real=0.0")
            cfg = cfg.replace(
                WORK_DIR_LINE,
                f"work_dir = '{args.out.resolve().parent}/runs/fv_{cell}_{seed}'")
            cfg += (
                f"\nrandomness = dict(seed={seed})\n"
                f"train_cfg = dict(type='IterBasedTrainLoop', max_iters={args.iters},"
                f" val_interval=100000)\n"
            )
            path = args.out / f"fv_{cell}_{seed}.py"
            path.write_text(cfg, encoding="utf-8")
            print("written", path)

    print()
    print(f"{len(CELLS) * len(SEEDS)} configs written to {args.out}")
    print(f"train iterations: {args.iters}, schedule horizon: {args.horizon}")


if __name__ == "__main__":
    main()
