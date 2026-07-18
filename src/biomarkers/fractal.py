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
    skeleton = skeletonize(binary).astype(bool)

    rows, cols = skeleton.shape
    max_box_size = min(rows, cols) // 2

    box_sizes, box_counts = [], []
    for box_size in range(1, max_box_size + 1):
        # Vectorized box-counting: pad with False (never flips occupancy of
        # real pixels) so every block reshapes evenly, then reduce with .any()
        # instead of a pure-Python double loop -- ~25s/image -> sub-second,
        # same box-occupancy result since padding only touches out-of-bounds cells.
        pad_r = (-rows) % box_size
        pad_c = (-cols) % box_size
        padded = np.pad(skeleton, ((0, pad_r), (0, pad_c)), mode="constant", constant_values=False)
        new_rows, new_cols = padded.shape
        reshaped = padded.reshape(new_rows // box_size, box_size, new_cols // box_size, box_size)
        count = int(reshaped.any(axis=(1, 3)).sum())
        if count > 0:
            box_sizes.append(math.log(1.0 / box_size))
            box_counts.append(math.log(count))

    if len(box_sizes) < 2:
        return 0.0

    coeffs = np.polyfit(box_sizes, box_counts, 1)
    return float(coeffs[0])
