"""Optic-disc-relative SIVA zone geometry (Cheung et al. 2010).

Ported from external/cmrrwnet/get_biomarker.py (MIT licensed baseline).
"""
import cv2
import numpy as np


def get_od_max_circle(od_mask: np.ndarray) -> tuple[tuple[int, int], float]:
    """Center and diameter of the optic disc's minimum enclosing circle.

    Args:
        od_mask: binary (0/255) optic disc mask, uint8.

    Returns:
        (cx, cy), dd (disc diameter, pixels). ((0, 0), 0.0) if no contour found.
    """
    contours, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0), 0.0
    max_contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(max_contour)
    return (int(cx), int(cy)), 2 * radius


def generate_annular_masks(
    shape: tuple[int, int], od_center: tuple[int, int], dd: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SIVA Zone A/B/C annular masks relative to the optic disc.

    Zone A: disc margin to 1 DD from OD center.
    Zone B: 1 DD to 1.5 DD.
    Zone C: 1.5 DD to 2.5 DD.
    """
    h, w = shape
    cx, cy = od_center
    a_mask = np.zeros((h, w), dtype=np.uint8)
    b_mask = np.zeros((h, w), dtype=np.uint8)
    c_mask = np.zeros((h, w), dtype=np.uint8)

    od_radius = dd / 2
    a_outer_radius = od_radius + 0.5 * dd
    b_outer_radius = od_radius + 1.0 * dd
    c_outer_radius = od_radius + 2.0 * dd

    cv2.circle(a_mask, (cx, cy), int(a_outer_radius), 255, -1)
    cv2.circle(a_mask, (cx, cy), int(od_radius), 0, -1)

    cv2.circle(b_mask, (cx, cy), int(b_outer_radius), 255, -1)
    cv2.circle(b_mask, (cx, cy), int(a_outer_radius), 0, -1)

    cv2.circle(c_mask, (cx, cy), int(c_outer_radius), 255, -1)
    cv2.circle(c_mask, (cx, cy), int(b_outer_radius), 0, -1)

    return a_mask, b_mask, c_mask
