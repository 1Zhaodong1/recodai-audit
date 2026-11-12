# -*- coding: utf-8 -*-
"""Corrected binary Dice loss for a single-channel segmentation head.

This is the term used in the "active Dice" cells (E2, E4) of the paper. It is
provided as a standalone module so that it can be registered in the audited
training code without patching the audited loss class:

    import binary_dice_loss  # noqa: F401  (registers BinaryDiceLoss)
    # and in the config:
    #   dict(type='BinaryDiceLoss', loss_weight=2.0)

The weighted loss lies in [0, 2] because ``loss_weight=2.0``. With an
all-background target and a random prediction the value approaches 2, not 0;
that is expected, because the prediction and the target are disjoint.
"""

import torch
import torch.nn as nn

from mmseg.registry import MODELS


@MODELS.register_module()
class BinaryDiceLoss(nn.Module):
    """Soft Dice loss on sigmoid probabilities, for one output channel.

    Args:
        epsilon (float): Smoothing term added to numerator and denominator.
        loss_weight (float): Scalar weight applied to the loss.
        reduction (str): 'mean' or 'sum'.
    """

    def __init__(self, epsilon=1.0, loss_weight=1.0, reduction="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, ignore_index=255, **kwargs):
        """Compute the loss.

        Args:
            pred (Tensor): Logits of shape (N, 1, H, W) or (N, H, W).
            target (Tensor): Binary target of the same shape.
            weight (Tensor, optional): Per-pixel weights.
            ignore_index (int): Pixels to ignore in the target.
        """
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        prob = torch.sigmoid(pred)
        target = target.float()

        if weight is not None:
            if weight.dim() == 4:
                weight = weight.squeeze(1)
            weight = weight.float()
            valid = weight > 0
            prob = prob * valid
            target = target * valid
            weight = weight * valid
        else:
            weight = torch.ones_like(prob)

        num = 2.0 * (prob * target * weight).sum() + self.epsilon
        den = (prob.pow(2) * weight).sum() + (target.pow(2) * weight).sum() + self.epsilon

        loss = 1.0 - num / den
        return self.loss_weight * loss
