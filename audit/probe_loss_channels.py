"""Reproduce the single-channel Dice degeneration with a plain PyTorch probe.

Usage: python audit/probe_loss_channels.py
Prints loss and gradient norm for DiceLoss and BinaryDiceLoss at C=2 and C=1.
"""
import torch
import torch.nn.functional as F


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1, exponent=2, ignore_label=-1):
        super().__init__()
        self.smooth, self.exponent, self.ignore_index = smooth, exponent, ignore_label

    def forward(self, score, target):
        num_classes = score.size(1)
        score = F.softmax(score, dim=1)
        target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes)
        target = target.permute(0, 3, 1, 2).float()
        valid = (target.sum(dim=1, keepdim=True) >= 0).float()
        num = (score * target * valid).sum() * 2 + self.smooth
        den = (score.pow(self.exponent) * valid).sum() + (target.pow(self.exponent) * valid).sum() + self.smooth
        return (1 - num / den).mean()


def run(cls, C):
    torch.manual_seed(0)
    x = torch.randn(1, C, 8, 8, requires_grad=True)
    t = torch.zeros(1, 8, 8, dtype=torch.long); t[:, 2:5, 2:6] = 1
    try:
        loss = cls()(x, t); loss.backward()
        return dict(loss=round(float(loss.detach()), 6), grad_norm=round(float(x.grad.norm()), 6))
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    for C in (2, 1):
        print(f"C={C}: DiceLoss {run(DiceLoss, C)}")
