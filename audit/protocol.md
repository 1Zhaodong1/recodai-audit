# Audit protocol

Five independent steps, each producing evidence that does not depend on the
others. This is the protocol described in Section IV-A of the paper. It is
released so that the same checks can be applied to other forensic pipelines.

## Step 1 — configuration versus constructor signature

For every loss and post-processing component, compare the keys used in the
training config against the constructor signature of the registered class.

```bash
python - <<'PY'
import inspect
from mmseg.models.losses.dice_loss import DiceLoss
print(inspect.signature(DiceLoss.__init__))
PY
# then grep the audited config for the loss entry
```

A key that is absent from the signature but absorbed by a catch-all
(`**kwargs` / `**kwards`) is accepted at construction time and never used. That
is the shape of Defect 1.

## Step 2 — numerical probe of every configured loss

Build the loss with the configured arguments, feed a random prediction and three
target patterns (all background, half positive, sparse positive), and check that

- the value is non-zero,
- the gradient with respect to the prediction is non-zero,
- the value changes between target patterns.

```bash
python audit/unit_test.py
```

A configured term that is exactly zero on all three patterns with an exactly
zero gradient cannot influence training, regardless of what the logs say.

## Step 3 — supervision provenance

Instrument the data pipeline and assert that the mask used in the loss is the
released annotation, not a replacement created by an augmentation.

The audited pipeline composes a synthetic copy-paste pair and overwrites the
ground-truth mask; the branch that would have used the real mask was commented
out. The check is a logging assertion on the mask identity per iteration.

## Step 4 — submission-path verification

Start from the inference entry point of the released submission (notebook or
script) and follow the call chain. For each stage described in the paper, record
which module is actually instantiated, or that none is.

```bash
grep -rniE "pair|contrast|cosine" <submission-notebook-or-script>
```

A description stage with no corresponding call is a documentation defect
(Defect 3).

## Step 5 — controlled reproduction

Re-train the reproduced pipeline in a factorial design over the two independent
defects (Dice term active/inactive, real masks off/on), three seeds per cell,
with everything else fixed. Report main effects and the interaction; do not
compare absolute scores across different splits

```bash
python configs/make_configs.py --base <AUDITED_REPO>/configs/mix.py --out configs/generated
python audit/eval_full.py --root <DATA_ROOT> --configs configs/generated --work-dir runs/
```

## What the protocol cannot establish

- It cannot tell whether a defect was deliberate or accidental.
- It cannot establish prevalence: one release is one data point.
- It only covers the stages that are re-trained. Here the instance-pair
  verification stage is audited in Step 4 but not re-trained.
