"""End-to-end vascular biomarker computation from an AV mask + optic disc mask."""
import numpy as np

from .density import calculate_density_in_c
from .fractal import calculate_fractal_dimension_skeleton
from .knudtson import calculate_crae_crve_revised, get_top_n_vessels_in_c
from .labels import extract_av_masks
from .siva_zones import generate_annular_masks, get_od_max_circle


def compute_biomarkers(av_img: np.ndarray, od_mask: np.ndarray) -> dict:
    """av_img: prediction-format HxWx3 uint8 (R=artery, G=vessel, B=vein).
    od_mask: binary (0/255) uint8 optic disc mask, same HxW as av_img.

    Returns dict with CRAE, CRVE, AVR, artery_density, vein_density,
    artery_fractal_dimension, vein_fractal_dimension. Raises ValueError if no
    OD contour is found.
    """
    artery_mask, vein_mask = extract_av_masks(av_img)

    od_center, dd = get_od_max_circle(od_mask)
    if dd == 0:
        raise ValueError("No optic disc contour found in od_mask")

    _, _, c_mask = generate_annular_masks(av_img.shape[:2], od_center, dd)

    top6_artery = get_top_n_vessels_in_c(artery_mask, c_mask, top_n=6)
    top6_vein = get_top_n_vessels_in_c(vein_mask, c_mask, top_n=6)

    crae = calculate_crae_crve_revised(top6_artery, is_artery=True)
    crve = calculate_crae_crve_revised(top6_vein, is_artery=False)
    avr = crae / crve if crve > 0 and crae > 0 else float("inf")

    return {
        "CRAE": crae,
        "CRVE": crve,
        "AVR": avr,
        "artery_density": calculate_density_in_c(artery_mask, c_mask),
        "vein_density": calculate_density_in_c(vein_mask, c_mask),
        "artery_fractal_dimension": calculate_fractal_dimension_skeleton(artery_mask),
        "vein_fractal_dimension": calculate_fractal_dimension_skeleton(vein_mask),
    }
