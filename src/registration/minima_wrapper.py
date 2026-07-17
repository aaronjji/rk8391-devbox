"""Batch CFP<->FFA registration using MINIMA (sp_lg / SuperPoint+LightGlue).

Verified empirically (2026-07-17) on g_001: classical ORB matching is
unusable across the CFP/FFA modality gap (std of match displacement ~256px,
essentially noise), while MINIMA's sp_lg gives 443 matches with std ~8px and
a visually near-perfect vessel-tree overlay after warping. FFA_A and FFA_AV
are each matched independently against CFP (not against each other) since
that's the coordinate frame both Task 2 training and the biomarker pipeline
need them in.

Usage:
    python src/registration/minima_wrapper.py \
        --data-root data/raw/GAVE2_preliminary --split training \
        --out-root data/registered
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MINIMA_ROOT = _REPO_ROOT / "external" / "minima"


def _load_matcher():
    """MINIMA's internal imports assume cwd == its own repo root, so we
    chdir there for the duration of loading (and restore afterward)."""
    import os

    prev_cwd = os.getcwd()
    os.chdir(_MINIMA_ROOT)
    sys.path.insert(0, str(_MINIMA_ROOT))
    try:
        from load_model import load_model

        args = argparse.Namespace(ckpt=str(_MINIMA_ROOT / "weights" / "minima_lightglue.pth"))
        matcher = load_model("sp_lg", args)
    finally:
        os.chdir(prev_cwd)
    return matcher


def register_pair(
    matcher, cfp_path: str, moving_path: str, min_inliers: int = 15, ransac_thresh: float = 3.0
):
    """Returns (warped_moving_bgr, H, n_inliers, n_matches). H is the
    homography mapping moving-image coords -> CFP coords. Falls back to
    identity (H=None) if too few inliers are found."""
    res = matcher(cfp_path, moving_path)
    mkpts0, mkpts1 = res["mkpts0"], res["mkpts1"]  # 0=cfp, 1=moving

    cfp = cv2.imread(cfp_path)
    moving = cv2.imread(moving_path)
    h, w = cfp.shape[:2]

    if len(mkpts0) < 4:
        return moving, None, 0, len(mkpts0)

    H, inlier_mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, ransac_thresh)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    if H is None or n_inliers < min_inliers:
        return moving, None, n_inliers, len(mkpts0)

    warped = cv2.warpPerspective(moving, H, (w, h))
    return warped, H, n_inliers, len(mkpts0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/raw/GAVE2_preliminary")
    p.add_argument("--split", type=str, required=True, choices=["training", "validation"])
    p.add_argument("--out-root", type=str, default="data/registered")
    p.add_argument("--save-overlays", type=int, default=10, help="Save visual overlays for the first N cases (spot-check)")
    args = p.parse_args()

    data_root = Path(args.data_root) / args.split
    out_root = Path(args.out_root) / args.split
    (out_root / "FFA_A").mkdir(parents=True, exist_ok=True)
    (out_root / "FFA_AV").mkdir(parents=True, exist_ok=True)
    (out_root / "homographies").mkdir(parents=True, exist_ok=True)
    overlay_dir = out_root / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MINIMA sp_lg matcher...")
    matcher = _load_matcher()

    image_paths = sorted((data_root / "images").glob("*.png"))
    log_rows = []
    for i, img_path in enumerate(image_paths):
        name = img_path.stem
        cfp_path = str(img_path)

        for phase, subdir in [("FFA_A", "FFA_A"), ("FFA_AV", "FFA_AV")]:
            moving_path = str(data_root / subdir / f"{name}.png")
            if not Path(moving_path).exists():
                print(f"  {name} [{phase}]: SKIP (file not found)")
                continue

            warped, H, n_inliers, n_matches = register_pair(matcher, cfp_path, moving_path)
            status = "ok" if H is not None else "fallback_identity"
            print(f"  {name} [{phase}]: {status} inliers={n_inliers}/{n_matches}")
            log_rows.append((name, phase, status, n_inliers, n_matches))

            cv2.imwrite(str(out_root / subdir / f"{name}.png"), warped)
            if H is not None:
                np.save(out_root / "homographies" / f"{name}_{phase}.npy", H)

            if i < args.save_overlays and phase == "FFA_A":
                cfp = cv2.imread(cfp_path)
                overlay = cv2.addWeighted(cfp, 0.5, warped, 0.5, 0)
                cv2.imwrite(str(overlay_dir / f"{name}_overlay.png"), overlay)

    log_path = out_root / "registration_log.csv"
    with open(log_path, "w") as f:
        f.write("case,phase,status,n_inliers,n_matches\n")
        for row in log_rows:
            f.write(",".join(str(x) for x in row) + "\n")

    n_fallback = sum(1 for r in log_rows if r[2] == "fallback_identity")
    print(f"\nDone. {len(log_rows)} pairs processed, {n_fallback} fell back to identity (low-confidence match).")
    print(f"Log: {log_path}")
    print(f"Spot-check overlays in: {overlay_dir}")


if __name__ == "__main__":
    main()
