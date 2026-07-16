"""AV label channel-convention conversion.

Two different RGB conventions are in play and must not be confused:

- Raw GT label files (`training/av/g_*.png`): R=artery-only, G=intersection-
  of-vessels-only, B=vein-only (per the baseline README figure).
- Submission / model-prediction format (what Task 1/2 predictions must use,
  and what get_biomarker.py's extract_av_masks expects): R=artery (full,
  including crossings), G=all-vessel (full), B=vein (full).

`gt_to_prediction_format` reconstructs the second convention from the first,
so the same downstream biomarker code works on both GT and predictions.
"""
import numpy as np


def gt_to_prediction_format(raw_gt: np.ndarray) -> np.ndarray:
    """raw_gt: HxWx3(or4) uint8, R=artery-only, G=intersection-only, B=vein-only.

    Returns HxWx3 uint8 in prediction format: R=artery-full, G=vessel-full, B=vein-full.
    """
    r = raw_gt[..., 0] > 0
    g = raw_gt[..., 1] > 0
    b = raw_gt[..., 2] > 0

    artery_full = r | g
    vein_full = b | g
    vessel_full = r | g | b

    out = np.zeros(raw_gt.shape[:2] + (3,), dtype=np.uint8)
    out[..., 0] = artery_full.astype(np.uint8) * 255
    out[..., 1] = vessel_full.astype(np.uint8) * 255
    out[..., 2] = vein_full.astype(np.uint8) * 255
    return out


def extract_av_masks(av_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """av_img: prediction-format HxWx3 (R=artery, G=vessel, B=vein).

    Returns (artery_mask, vein_mask), each HxW uint8 0/255.
    """
    r = av_img[..., 0] > 0
    g = av_img[..., 1] > 0
    b = av_img[..., 2] > 0

    artery_mask = (g & ~b).astype(np.uint8) * 255
    vein_mask = (g & ~r).astype(np.uint8) * 255
    return artery_mask, vein_mask
