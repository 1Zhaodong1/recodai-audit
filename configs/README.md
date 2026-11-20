# Ablation configurations

`make_configs.py` derives the twenty runs of the factorial design from the single
training configuration released with the audited work. Only the following
entries differ between runs; everything else is inherited from the base file.

| Cell | Dice entry in the decode head | Real-mask mixing |
| --- | --- | --- |
| E1 | `DiceLoss(use_sigmoid=True, loss_weight=2.0)` | `p_real=0.0` |
| E2 | `BinaryDiceLoss(loss_weight=2.0)` | `p_real=0.0` |
| E3 | `DiceLoss(use_sigmoid=True, loss_weight=2.0)` | `p_real=0.5` |
| E4 | `BinaryDiceLoss(loss_weight=2.0)` | `p_real=0.5` |

Seeds: 42, 2024, 7. Training horizon: 1,000 iterations per cell (schedule
horizon 3,000, polynomial decay, power 0.9, η_min 1e-6).

## Notes

* `use_sigmoid=True` is *not* honoured by the audited `DiceLoss`: the key is
  absorbed by a catch-all argument and the forward pass applies softmax
  unconditionally. This is Defect 1; see `../audit/unit_test.py`.
* `BinaryDiceLoss` lives in `../audit/binary_dice_loss.py`. Add `audit/` to
  `PYTHONPATH` so that the training code can import it, or copy the file next to
  the audited losses package.
* The base configuration is the mixed setting (`p_real=0.5`), so the
  `p_real=0.0` substitution only affects the two synthetic-only cells; the
  generator replaces the *first* occurrence, which is the one in the training
  pipeline.
* `work_dir` is rewritten to `<repo>/runs/fv_<cell>_<seed>`.
