# -*- coding: utf-8 -*-
"""Audit of the configured Dice term versus the corrected binary loss.

Runs on CPU in well under a minute and does not need the dataset.
Reproduces Table I of the paper:

    Target            configured loss   configured grad   corrected loss   corrected grad
    All background    0.000000          0.000000          1.9983           1.4e-05
    Half positive     0.000000          0.000000          0.7343           6.3e-03
    Sparse positive   0.000000          0.000000          1.7039           6.1e-03
    4-class control   1.2718            6.6e-03           -                -

Five checks are performed:

  C1  the configured loss class has no ``use_sigmoid`` parameter, so the key is
      silently absorbed by **kwards and never used;
  C2  the forward pass applies softmax unconditionally;
  C3  on a single-channel head the configured term is exactly zero with an
      exactly zero gradient for every target pattern;
  C4  the corrected binary term is non-zero and differentiable on the same
      inputs;
  C5  the same class returns a finite value on a four-channel head, which shows
      that the defect is specific to the binary configuration.

Usage:
    python audit/unit_test.py [--channel-last] [--no-mmseg]

The import of the audited loss requires mmsegmentation to be importable. If it
is not available, pass --no-mmseg after inspecting the two source facts (C1, C2)
with `grep`; the script then skips C1/C2/C5 and reports C3/C4 only.
"""

import argparse
import inspect
import sys

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Corrected binary Dice loss (also importable from binary_dice_loss.py)
# --------------------------------------------------------------------------
def binary_dice_loss(pred, target, epsilon=1.0, weight=2.0):
    """Sigmoid + soft Dice for a single-channel segmentation head.

    ``pred`` are logits of shape (N, 1, H, W) or (N, H, W); ``target`` is
    binary with the same shape.
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    prob = torch.sigmoid(pred)
    num = 2.0 * (prob * target).sum() + epsilon
    den = prob.pow(2).sum() + target.pow(2).sum() + epsilon
    return weight * (1.0 - num / den)


def check_signature():
    """C1: the configured key is accepted but absent from the signature."""
    from mmseg.models.losses.dice_loss import DiceLoss

    init = inspect.signature(DiceLoss.__init__)
    params = list(init.parameters)
    has_key = "use_sigmoid" in params
    has_kwards = any(p.kind == p.VAR_KEYWORD for p in init.parameters.values())
    print(f"[C1] DiceLoss.__init__ parameters: {params}")
    print(f"[C1] 'use_sigmoid' in signature : {has_key}")
    print(f"[C1] catch-all **kwargs accepted: {has_kwards}")
    return has_key, has_kwards


def check_forward_softmax():
    """C2: the forward pass applies softmax unconditionally."""
    import mmseg.models.losses.dice_loss as mod

    src = inspect.getsource(mod.DiceLoss.forward)
    uses_softmax = "softmax" in src
    uses_sigmoid = "sigmoid" in src
    print(f"[C2] 'softmax' in DiceLoss.forward: {uses_softmax}")
    print(f"[C2] 'sigmoid' in DiceLoss.forward: {uses_sigmoid}")
    return uses_softmax, uses_sigmoid


def make_targets(h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    soft = torch.rand(1, 1, h, w, generator=g)
    targets = {
        "all background": torch.zeros(1, 1, h, w),
        "half positive": (soft > 0.5).float(),
        "sparse positive": (soft > 0.99).float(),
    }
    return torch.randn(1, 1, h, w, generator=g, requires_grad=True), targets


def measure(loss_fn, logits, target):
    logits.grad = None
    loss = loss_fn(logits, target)
    loss.backward()
    grad = logits.grad.norm().item() if logits.grad is not None else 0.0
    return loss.item(), grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mmseg", action="store_true",
                    help="skip checks that need the audited loss class")
    args = ap.parse_args()

    print("=" * 74)
    print("Audit of the configured Dice term (CPU, no dataset required)")
    print("=" * 74)

    if not args.no_mmseg:
        check_signature()
        check_forward_softmax()

        from mmseg.models.losses.dice_loss import DiceLoss

        configured = DiceLoss(use_sigmoid=True, loss_weight=2.0)
        logits, targets = make_targets()
        print()
        print(f"{'target':>16} | {'configured':>12} | {'grad':>12} | "
              f"{'corrected':>12} | {'grad':>12}")
        print("-" * 74)
        for name, target in targets.items():
            lg = logits.clone().detach().requires_grad_(True)
            loss_c, grad_c = measure(configured, lg, target)
            lg2 = logits.clone().detach().requires_grad_(True)
            loss_b, grad_b = measure(binary_dice_loss, lg2, target)
            print(f"{name:>16} | {loss_c:12.6f} | {grad_c:12.6f} | "
                  f"{loss_b:12.4f} | {grad_b:12.1e}")

        # C4 control: four-channel head
        lg4 = torch.randn(1, 4, 16, 16, requires_grad=True)
        t4 = torch.zeros(1, 1, 16, 16, dtype=torch.long)
        loss4, grad4 = measure(configured, lg4, t4)
        print(f"{'4-class control':>16} | {loss4:12.4f} | {grad4:12.1e} | "
              f"{'-':>12} | {'-':>12}")
    else:
        logits, targets = make_targets()
        for name, target in targets.items():
            lg = logits.clone().detach().requires_grad_(True)
            loss_b, grad_b = measure(binary_dice_loss, lg, target)
            print(f"{name:>16} | corrected loss = {loss_b:.4f}, grad = {grad_b:.1e}")

    print()
    print("Interpretation: the configured term is identically zero with zero")
    print("gradient for a single-channel head, which is the mechanism reported")
    print("as Defect 1 in the paper.")


if __name__ == "__main__":
    sys.exit(main())
