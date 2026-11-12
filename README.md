# Implementation audit of a two-stage scientific forgery detector

This repository contains the audit protocol, the unit tests, the split files and
the evaluation script for the paper

> **Revisiting Scientific Image Forgery Detection: An Implementation Audit and
> Reproducibility Study of a Two-Stage Framework**

The audited pipeline is described in an unpublished manuscript released together
with its training code; this repository does **not** redistribute that code. It
contains everything needed to (i) verify the three reported defects and (ii)
re-measure the numbers reported in the paper.

---

## 1. What can be reproduced, and how long it takes

| Step | Script | Time | Output |
| --- | --- | --- | --- |
| Audit of the Dice term | `audit/unit_test.py` | < 1 min (CPU) | Table I of the paper |
| Split reconstruction | `audit/make_splits.py` | < 1 min | `splits/train.txt`, `splits/val.txt` |
| Factorial ablation | `tools/train.py` + `configs/` | ~2.5 h on one T4 | 12 checkpoints |
| Full validation | `audit/eval_full.py` | ~10 min | Tables III–IV and Fig. 2 |

The unit test is the part that matters for the audit claim: it runs on CPU and
does not need the dataset, a GPU, or any training.

## 2. Environment

Python 3.10, PyTorch 2.0.1 (CUDA 11.8), MMCV 2.0.1, MMEngine 0.8.3,
MMPretrain 1.0.0, mmsegmentation (the fork that accompanies the audited work).

```bash
pip install -r requirements.txt
python audit/unit_test.py                 # expect the paper's Table I values
python audit/make_splits.py --root <DATA_ROOT> --out splits/
```

Set `DATA_ROOT` to the directory containing `train_images/{forged,authentic}`
and `train_masks/`. The dataset is the Recod.ai *Scientific Image Forgery
Detection* Kaggle competition data; see `data/README.md`.

## 3. Dataset layout expected by the scripts

```
<DATA_ROOT>/
├── train_images/forged/<id>.png
├── train_images/authentic/<id>.png            # same <id> as its forged counterpart
└── train_masks/<id>.npy                       # (C,H,W) uint8, C = instances
```

## 4. Reproducing the ablation

The audited code ships a single training configuration for the mixed setting.
`configs/make_configs.py` derives the twelve configurations of the factorial
design from it:

```bash
python configs/make_configs.py \
    --base <AUDITED_REPO>/configs/mix.py \
    --out configs/generated \
    --iters 1000

for cfg in configs/generated/fv_E*_*.py; do
    python <AUDITED_REPO>/tools/train.py "$cfg"
done
```

Each cell differs only in two keys: the Dice entry of the decode head
(`DiceLoss` with `use_sigmoid=True` versus `BinaryDiceLoss`) and the real-mask
mixing probability of the tampering augmentation (`p_real=0.0` versus `0.5`).
Seeds are 42, 2024 and 7. All other settings are inherited from the base file.

`audit/binary_dice_loss.py` contains the corrected loss used for the "active"
cells; it is registered as `BinaryDiceLoss` and must be importable by the
training code, e.g. by adding `audit/` to `PYTHONPATH`.

## 5. Reproducing the evaluation

```bash
python audit/eval_full.py \
    --root <DATA_ROOT> \
    --splits splits/ \
    --configs configs/generated/ \
    --work-dir runs/ \
    --threshold 0.3
```

The script evaluates all twelve runs on the **full** validation split
(550 forged and 475 authentic images, letterbox preprocessing to 518×518,
batched inference) and prints:

- per-image pixel F1, precision and recall on the forged images;
- pixel-level and image-level false-positive rates on the authentic images;
- pixel F1 stratified by the fraction of positive pixels in the ground truth
  (tiny < 0.5 %, small 0.5–2 %, large > 2 %);
- per-seed values and the three factorial effect estimates.

## 6. Expected numbers

Cell means (three seeds, full validation split):

| Cell | Dice term | Supervision | Pixel F1 | FP (authentic) |
| --- | --- | --- | --- | --- |
| E1 | inactive | synthetic | 0.154 ± 0.012 | 0.051 |
| E2 | active | synthetic | 0.385 ± 0.029 | 0.284 |
| E3 | inactive | mixed | 0.235 ± 0.023 | 0.218 |
| E4 | active | mixed | 0.456 ± 0.007 | 0.606 |

Effect estimates: Δ_D = +0.226, Δ_M = +0.076, Δ_DM = −0.011.

Small deviations are expected across GPU models and library versions; the seed
spread we observe (up to ±0.029 within a cell) is larger than typical hardware
noise.

## 7. Scope and caveats

- Only the segmentation stage is re-trained and re-measured. The instance-pair
  verification stage is audited from the code but not retrained.
- The operating threshold is fixed a priori at τ = 0.3 and is not tuned.
- The audited work's private test split is not released; the split used here is
  a random 80/20 split over identifiers, regenerated deterministically by
  `audit/make_splits.py` with seed 42.
- Scores are therefore not directly comparable with the values reported in the
  audited work; the paper compares orderings and effect magnitudes only.

## 8. Files

```
audit/unit_test.py        five checks on the Dice term (Table I)
audit/binary_dice_loss.py corrected binary Dice loss
audit/make_splits.py      deterministic split reconstruction
audit/eval_full.py        full-validation evaluation
audit/protocol.md         the five-step audit protocol
configs/make_configs.py   derive the twelve ablation configs
configs/README.md         what differs between the twelve runs
splits/                   train.txt / val.txt identifier lists
data/README.md            dataset download and layout
```

## 9. Citation

```bibtex
@unpublished{huo2026audit,
  author = {Huo, Zhaodong},
  title  = {Revisiting Scientific Image Forgery Detection: An Implementation
            Audit and Reproducibility Study of a Two-Stage Framework},
  note   = {Unpublished manuscript},
  year   = {2026}
}
```

## 10. License

Code and text in this repository are released under the MIT License; see
`LICENSE`. The dataset is governed by the terms of the Kaggle competition and is
not redistributed here.

