"""Zone-C vascular density. Ported from external/cmrrwnet/get_biomarker.py."""
import cv2
import numpy as np


def calculate_density_in_c(vessel_mask: np.ndarray, c_mask: np.ndarray) -> float:
    _, vessel_bin = cv2.threshold(vessel_mask, 127, 1, cv2.THRESH_BINARY)
    _, c_bin = cv2.threshold(c_mask, 127, 1, cv2.THRESH_BINARY)

    vessel_in_c = vessel_bin * c_bin
    c_pixels = np.sum(c_bin)
    if c_pixels == 0:
        return 0.0
    return float(np.sum(vessel_in_c) / c_pixels)
