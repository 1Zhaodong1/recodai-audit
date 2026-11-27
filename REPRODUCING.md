# Reproducing the audit

This file maps every table and figure of the manuscript to the script and data
file that produce it. The release is self-contained for the CPU checks; the
training runs require a GPU and the ReCoD.ai LUC competition data.

## Quick start (CPU, no dataset)

```bash
docker build -t recodai-audit .
docker run --rm recodai-audit                      # unit test (Table 1)
docker run --rm recodai-audit python audit/probe_loss_channels.py   # C=1 vs C=2 probe
docker run --rm recodai-audit python audit/scan_configs.py --help   # static scan (L1)
```

## Training and evaluation (GPU, Kaggle)

1. Mount the competition data and the audited code (see `data/README.md`).
2. `python configs/make_configs.py --base <AUDITED_REPO>/configs/mix.py --out configs/generated`
3. `python <AUDITED_REPO>/tools/train.py configs/generated/it_E1_42.py` (20 runs: E1–E4 × seeds 42, 2024, 7, 1, 2).
4. `python audit/eval_full.py --root <DATA_ROOT> --splits splits/ --configs configs/generated --threshold 0.3`

## Data files released with the paper

| File | Content |
| --- | --- |
| `main_5seed.json` | per-image F1 (550 forged) and flags (475 authentic), 20 runs |
| `deleak_5seed.json` | same, de-overlapped training set |
| `budget_5seed.json` | 6 runs × 1k/2k/3k iterations, F1/FAR |
| `sweep_5seed.json` | threshold sweep for the 1k checkpoints |
| `strata.json` | per-image F1 and stratum, 20 runs (stratified panel) |

## Table / figure map

See `results/manifest.json`.

## Environment

- Python 3.10, PyTorch 2.0.1 (+cu118 for training on T4), mmengine 0.8.3,
  mmpretrain 1.0.0, OpenCV headless 4.11.
- CPU checks need only torch + numpy.

## Expected outputs

- `audit/unit_test.py`: three rows with configured loss = 0 and gradient = 0,
  corrected loss in [0, 2].
- `audit/probe_loss_channels.py`: C=2 non-zero loss/gradient; C=1 Dice loss = 0
  and gradient norm = 0.
- `audit/scan_configs.py`: JSON with the scanned entries and the number of
  unread keys.
