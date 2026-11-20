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
| Factorial ablation | `tools/train.py` + `configs/` | ~4 h on one T4 | 20 checkpoints |
| Full validation | `audit/eval_full.py` | ~10 min | Tables III–IV and Fig. 2 |
| Threshold sweep | `audit/sweep.py` | ~15 min | Table V and Fig. 4 |

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
`configs/make_configs.py` derives the twenty configurations of the factorial
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
Seeds are 42, 2024, 7, 1 and 2. All other settings are inherited from the base file.

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

The script evaluates all twenty runs on the **full** validation split
(550 forged and 475 authentic images, letterbox preprocessing to 518×518,
batched inference) and prints:

- per-image pixel F1, precision and recall on the forged images;
- pixel-level and image-level false-positive rates on the authentic images;
- pixel F1 stratified by the fraction of positive pixels in the ground truth
  (tiny < 0.5 %, small 0.5–2 %, large > 2 %);
- per-seed values and the five-seed factorial effect estimates.

## 6. Expected numbers

Cell means (five seeds, full validation split, 20 runs):

| Cell | Dice term | Supervision | Pixel F1 | FP (authentic) |
| --- | --- | --- | --- | --- |
| E1 | inactive | synthetic | 0.153 ± 0.032 | 4.3% |
| E2 | active | synthetic | 0.395 ± 0.017 | 27.1% |
| E3 | inactive | mixed | 0.181 ± 0.092 | 14.1% |
| E4 | active | mixed | 0.457 ± 0.020 | 65.9% |

Effect estimates: Δ_D = +0.259, Δ_M = +0.046, Δ_DM = +0.035.

The per-seed values behind these means were not retained as per-image files for
the first run set; the stratified analysis in the paper uses an independent
five-seed replication whose per-image F1 is released in the paper's
supplementary material (Table S6).

Threshold behaviour of the two extreme cells (Table V of the paper):

| Cell | F1 at τ=0.3 | FAR at τ=0.3 | TPR at τ=0.3 | best F1 at FAR ≤ 5% | best F1 at FAR ≤ 10% |
| --- | --- | --- | --- | --- | --- |
| E1 | 0.151 | 4.1% | 34.1% | 0.411 (τ=0.25) | 0.447 (τ=0.13) |
| E4 | 0.461 | 60.0% | 94.2% | 0.207 (τ=0.99) | 0.325 (τ=0.97) |

Under a fixed false-alarm budget the released configuration is therefore the
better detector, even though its unconstrained pixel F1 is much lower.

Small deviations are expected across GPU models and library versions; the seed
spread we observe (up to ±0.029 within a cell) is larger than typical hardware
noise.

## 6b. Systematic scan of the release

`audit/scan_configs.py` applies Step 1 of the audit protocol mechanically: it
parses every `dict(type=..., ...)` loss entry in a release and compares the
configured keys with the `__init__` signature of the registered class. A key that
is absent from the signature but absorbed by a catch-all argument is reported as
unread. Nothing is imported or executed.

```bash
python audit/scan_configs.py <AUDITED_REPO>
```

Result on the audited release (`audit/scan_results.json`): 223 loss entries, of
which 29 are in the 20 author-written configurations and 194 in the vendored
upstream configurations. Four unread keys are reported, all in the author-written
files: `use_sigmoid=False` passed to `DiceLoss` for three-class heads, where the
class default coincides with the intended behaviour. The submission configuration
passes the same key with `use_sigmoid=True` to a single-channel head, which is the
fatal case analysed in the paper: five of the 30 author-written loss entries carry
the unread key.

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
audit/sweep.py            threshold sweep, false-alarm budget, bootstrap intervals
audit/scan_configs.py     Step-1 scan of every loss entry against its constructor
audit/near_duplicate_audit.py  dHash + correlation audit of the split
audit/de_overlap.py       build the de-overlapped training list
data/exclusions_train_ids.txt train ids removed in the de-overlapped design
data/affected_val_ids.txt validation ids with a verified near-duplicate
data/near_duplicate_pairs.csv verified cross-split pairs
audit/scan_results.json   output of that scan on the audited release
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

