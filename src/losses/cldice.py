"""Soft-clDice: a differentiable topology-preserving loss for tubular
structures (Shit et al., "clDice -- A Novel Topology-Preserving Loss
Function for Tubular Structure Segmentation", CVPR 2021).

Standard reference implementation (soft skeletonization via iterative
soft erosion/dilation using min/max pooling), adapted for our artery/vein
channels specifically. Motivated directly by real leaderboard data: our
first submission's COR/INF (vessel connectivity) badly trailed the #1 team
(COR ~0.1-0.3 vs their ~0.78; INF ~0.7-0.9 vs their ~0.22) despite our raw
pixel-DSC being *higher* than theirs -- i.e. the gap is connectivity, not
pixel-overlap precision, which is exactly what clDice targets and plain
BCE/Dice do not.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """img: BxCxHxW soft mask in [0,1]. Returns a soft skeleton of the same shape."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iterations):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(pred: torch.Tensor, target: torch.Tensor, iterations: int = 10, smooth: float = 1.0) -> torch.Tensor:
    """pred, target: BxCxHxW soft masks in [0,1] (sigmoid probabilities /
    binary GT). Returns a scalar loss (1 - clDice), lower is better."""
    skel_pred = soft_skeletonize(pred, iterations)
    skel_true = soft_skeletonize(target, iterations)

    t_prec = (torch.sum(skel_pred * target) + smooth) / (torch.sum(skel_pred) + smooth)
    t_sens = (torch.sum(skel_true * pred) + smooth) / (torch.sum(skel_true) + smooth)

    cl_dice = 2.0 * (t_prec * t_sens) / (t_prec + t_sens)
    return 1.0 - cl_dice


class ArteryVeinClDiceLoss(nn.Module):
    """Applies soft-clDice separately to the artery (channel 0) and vein
    (channel 2) probability maps, weighted-averaged. Expects `pred_logits` in
    the [artery, vessel, vein] channel order used throughout this project
    (see src/biomarkers/labels.py).

    artery_weight/vein_weight let the two channels be weighted asymmetrically.
    Real leaderboard data (2026-07-18) showed the symmetric 0.5/0.5 version
    leaves Task2's vein topology far worse than artery (COR 0.35 vs 0.61,
    INF 0.65 vs 0.38) despite both getting equal loss weight -- so this
    defaults to still-symmetric 1.0/1.0 and callers pass an explicit
    vein-favoring ratio where the data shows it's needed."""

    def __init__(self, iterations: int = 10, smooth: float = 1.0, artery_weight: float = 1.0, vein_weight: float = 1.0):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth
        self.artery_weight = artery_weight
        self.vein_weight = vein_weight

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred_logits)
        roi = torch.round(mask[:, :1])  # single-channel ROI, broadcastable

        artery_loss = soft_cldice(pred[:, 0:1] * roi, target[:, 0:1] * roi, self.iterations, self.smooth)
        vein_loss = soft_cldice(pred[:, 2:3] * roi, target[:, 2:3] * roi, self.iterations, self.smooth)
        total_weight = self.artery_weight + self.vein_weight
        return (self.artery_weight * artery_loss + self.vein_weight * vein_loss) / total_weight
