"""Whole-image box-counting fractal dimension of a skeletonized vessel mask.

Ported from external/cmrrwnet/get_biomarker.py. NOTE: unlike density (Zone-C
scoped), fractal dimension is computed on the whole-image skeleton, not
restricted to Zone C -- confirmed by reading the baseline's reference
implementation.
"""
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize


def calculate_fractal_dimension_skeleton(binary_img: np.ndarray) -> float:
    if binary_img.max() == 0:
        return 0.0

    _, binary = cv2.threshold(binary_img, 127, 1, cv2.THRESH_BINARY)
    skeleton = skeletonize(binary).astype(np.uint8)

    rows, cols = skeleton.shape
    max_box_size = min(rows, cols) // 2

    box_sizes, box_counts = [], []
    for box_size in range(1, max_box_size + 1):
        count = 0
        for i in range(0, rows, box_size):
            for j in range(0, cols, box_size):
                i_end = min(i + box_size, rows)
                j_end = min(j + box_size, cols)
                if np.sum(skeleton[i:i_end, j:j_end]) > 0:
                    count += 1
        if count > 0:
            box_sizes.append(math.log(1.0 / box_size))
            box_counts.append(math.log(count))

    if len(box_sizes) < 2:
        return 0.0

    coeffs = np.polyfit(box_sizes, box_counts, 1)
    return float(coeffs[0])
