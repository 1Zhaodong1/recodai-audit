# Dataset

The experiments use the Recod.ai *Scientific Image Forgery Detection* Kaggle
competition data:

<https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection>

The competition rules apply to the data; this repository redistributes only
identifier lists, not images.

## Composition

| Split | Forged | Authentic | Total |
| --- | --- | --- | --- |
| train (competition) | 2,751 | 2,377 | 5,128 |
| our training split (80 % of identifiers) | 2,201 | 1,902 | 4,103 |
| our validation split (20 % of identifiers) | 550 | 475 | 1,025 |

Instance counts per forged image: 1 instance (1,869), 2 (701), 3 (174),
4 (5), 5 (2).

## Expected layout after downloading

```
<DATA_ROOT>/
├── train_images/
│   ├── forged/<id>.png
│   └── authentic/<id>.png
└── train_masks/
    └── <id>.npy        # (C, H, W) uint8; C = number of instances
```

The same `<id>` names the forged image and its authentic counterpart. The
authentic images have no mask file (the forgery mask is empty by definition).

## Masks

`train_masks/<id>.npy` stores a multi-channel array, one channel per annotated
instance. The binary ground truth used in this repository is the union:

```python
mask = (np.load(path) > 0).any(axis=0)
```

Authentic images are treated as having an all-zero mask and are used only for
the false-positive measurement.

## Modality labels

The release does not provide modality labels (fluorescence microscopy,
scanning-electron micrographs, specimen photography), so no per-modality
breakdown is possible.
