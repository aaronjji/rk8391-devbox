"""Revised Knudtson-Hubbard CRAE/CRVE (Knudtson et al. 2003).

Ported from external/cmrrwnet/get_biomarker.py (MIT licensed baseline).
"""
import math

import cv2
import numpy as np
from skimage.morphology import medial_axis


def get_top_n_vessels_in_c(vessel_mask: np.ndarray, c_mask: np.ndarray, top_n: int = 6) -> list[float]:
    """Max caliber (2x medial-axis distance) of the top-N largest vessel
    connected components within Zone C, sorted by component area."""
    vessel_in_c = cv2.bitwise_and(vessel_mask, vessel_mask, mask=c_mask)
    _, bin_mask = cv2.threshold(vessel_in_c, 127, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8, cv2.CV_32S)

    vessels = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        vessels.append((-area, x, y, w, h))

    vessels_sorted = sorted(vessels)[:top_n]
    diameters = []
    for area_neg, x, y, w, h in vessels_sorted:
        mask_roi = np.zeros_like(vessel_mask)
        mask_roi[y : y + h, x : x + w] = 255
        vessel_roi = cv2.bitwise_and(vessel_mask, mask_roi)
        skeleton, dist = medial_axis(vessel_roi, return_distance=True)
        vessel_diameters = dist[skeleton] * 2
        diameters.append(vessel_diameters.max() if vessel_diameters.size else 0.0)

    if len(diameters) < top_n:
        diameters += [0.0] * (top_n - len(diameters))
    return diameters


def calculate_crae_crve_revised(vessel_calibers: list[float], is_artery: bool = True) -> float:
    """Iterative largest+smallest pairing per the revised Knudtson formula.

    p=0.88 for arterioles, p=0.95 for venules.
    """
    coeff = 0.88 if is_artery else 0.95
    values = sorted(vessel_calibers, reverse=True)

    while len(values) > 1:
        values = sorted(values, reverse=True)
        next_values = []
        i, j = 0, len(values) - 1
        while i < j:
            w1, w2 = values[i], values[j]
            next_values.append(coeff * math.sqrt(w1**2 + w2**2))
            i += 1
            j -= 1
        values = next_values

    return values[0]
