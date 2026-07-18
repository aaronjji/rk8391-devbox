"""Multi-checkpoint ensemble inference: averages sigmoid probability maps
across several fold checkpoints (of the same task) before saving, rather than
ensembling at the mask/biomarker level. Reuses predict_task1.py's sliding-window
tiling so ensembled predictions are still full-resolution.

Usage (Task 2, 5-fold ensemble):
    python src/predict_ensemble.py --task task2 \
        --checkpoints runs/task2/fold0/best.pth runs/task2/fold1/best.pth \
                      runs/task2/fold2/best.pth runs/task2/fold3/best.pth runs/task2/fold4/best.pth \
        --images-dir data/raw/GAVE2_preliminary/validation/images \
        --masks-dir data/raw/GAVE2_preliminary/validation/masks \
        --ffa-dir data/registered/validation \
        --out-dir predictions/task2/validation
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.rrwnet import build_model  # noqa: E402
from predict_task1 import sliding_window_predict  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="task1", choices=["task1", "task2"])
    p.add_argument("--checkpoints", type=str, nargs="+", required=True, help="One or more checkpoint paths -- probabilities are averaged across all of them")
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--masks-dir", type=str, default=None)
    p.add_argument("--ffa-dir", type=str, default=None, help="Task2 only: dir with registered FFA_A/FFA_AV subfolders")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--tile", type=int, default=384)
    p.add_argument("--stride", type=int, default=288)
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    p.add_argument("--quantize-levels", type=int, default=32)
    args = p.parse_args()

    if args.task == "task2" and not args.ffa_dir:
        raise SystemExit("--ffa-dir is required for --task task2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for ckpt_path in args.checkpoints:
        model = build_model(args.task, base_ch=args.base_ch, iterations=args.iterations, pretrained=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    print(f"Loaded {len(models)} checkpoints for ensembling: {args.checkpoints}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    ffa_dir = Path(args.ffa_dir) if args.ffa_dir else None

    image_paths = sorted(images_dir.glob("*.png"))
    print(f"Running {args.task} ensemble inference ({len(models)}-way) on {len(image_paths)} images -> {out_dir}")
    for img_path in image_paths:
        image = np.array(Image.open(img_path).convert("RGB"))

        ffa = None
        if ffa_dir is not None:
            ffa_a = np.array(Image.open(ffa_dir / "FFA_A" / img_path.name).convert("L"))
            ffa_av = np.array(Image.open(ffa_dir / "FFA_AV" / img_path.name).convert("L"))
            ffa = np.stack([ffa_a, ffa_av], axis=2)

        probs_sum = None
        for model in models:
            probs = sliding_window_predict(model, image, device, ffa=ffa, tile=args.tile, stride=args.stride, amp_dtype=args.amp_dtype)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        probs = probs_sum / len(models)

        if masks_dir is not None:
            roi = np.array(Image.open(masks_dir / img_path.name).convert("L"))
            probs[roi == 0] = 0.0

        out_img = (probs * 255).astype(np.uint8)
        if args.quantize_levels > 0:
            n = args.quantize_levels - 1
            out_img = (np.round(out_img.astype(np.float32) / 255 * n) / n * 255).astype(np.uint8)
        Image.fromarray(out_img, mode="RGB").save(out_dir / img_path.name, compress_level=9)
        print(f"  {img_path.name}")


if __name__ == "__main__":
    main()
