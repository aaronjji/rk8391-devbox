"""Pixel-level DSC / Sensitivity / Specificity / Accuracy for one AV channel."""
import numpy as np


def dsc(gt: np.ndarray, pred: np.ndarray) -> float:
    """gt, pred: binary (bool or 0/1) masks, same shape."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    intersection = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def confusion_counts(gt: np.ndarray, pred: np.ndarray, roi: np.ndarray | None = None) -> tuple[int, int, int, int]:
    """Returns (TP, TN, FP, FN), optionally restricted to an ROI mask."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if roi is not None:
        roi = roi.astype(bool)
        gt = gt & roi
        pred = pred & roi
        valid = roi
    else:
        valid = np.ones_like(gt, dtype=bool)

    tp = np.logical_and(gt, pred).sum()
    tn = np.logical_and(~gt & valid, ~pred & valid).sum()
    fp = np.logical_and(~gt & valid, pred).sum()
    fn = np.logical_and(gt, ~pred & valid).sum()
    return int(tp), int(tn), int(fp), int(fn)


def sensitivity(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def specificity(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def channel_metrics(gt: np.ndarray, pred: np.ndarray, roi: np.ndarray | None = None) -> dict:
    """DSC + Sen/Spec/Acc for a single binary channel (artery or vein)."""
    d = dsc(gt if roi is None else (gt & roi.astype(bool)), pred if roi is None else (pred & roi.astype(bool)))
    tp, tn, fp, fn = confusion_counts(gt, pred, roi)
    return {
        "DSC": d,
        "Sen": sensitivity(tp, fn),
        "Spec": specificity(tn, fp),
        "Acc": accuracy(tp, tn, fp, fn),
    }
