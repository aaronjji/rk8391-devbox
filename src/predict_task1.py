"""Full-resolution sliding-window inference for a trained Task 1 or Task 2 checkpoint.

Training happens on 384x384 native-resolution patches (GPU VRAM constraint);
inference tiles the full 1536x1024 image with overlap and averages logits in
overlap regions, so predictions are produced at full resolution as the
submission format requires -- no downscaling.

Usage (Task 1):
    python src/predict_task1.py --task task1 --checkpoint runs/task1/fold0/latest.pth \
        --images-dir data/raw/GAVE2_preliminary/validation/images \
        --masks-dir data/raw/GAVE2_preliminary/validation/masks \
        --out-dir predictions/task1/validation

Usage (Task 2, needs registered FFA -- see src/registration/minima_wrapper.py):
    python src/predict_task1.py --task task2 --checkpoint runs/task2/fold0/latest.pth \
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


def sliding_window_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    ffa: np.ndarray | None = None,
    tile: int = 384,
    stride: int = 288,
    amp_dtype: str = "bf16",
) -> np.ndarray:
    """image: HxWx3 uint8. ffa (Task2 only): HxWx2 uint8 (FFA_A, FFA_AV channels,
    already registered to the CFP coordinate frame). Returns HxWx3 float32
    sigmoid probabilities (artery, vessel, vein), tiled with overlap-averaging."""
    h, w = image.shape[:2]
    accum = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)

    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if not ys or ys[-1] + tile < h:
        ys.append(max(h - tile, 0))
    if not xs or xs[-1] + tile < w:
        xs.append(max(w - tile, 0))

    n_channels = 5 if ffa is not None else 3

    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                y1, x1 = min(y + tile, h), min(x + tile, w)
                y0, x0 = y1 - tile if y1 - tile >= 0 else 0, x1 - tile if x1 - tile >= 0 else 0
                patch = image[y0:y1, x0:x1]
                ph, pw = patch.shape[:2]
                if ph < tile or pw < tile:
                    padded = np.zeros((tile, tile, 3), dtype=patch.dtype)
                    padded[:ph, :pw] = patch
                    patch = padded

                patch_full = patch.astype(np.float32)
                if ffa is not None:
                    ffa_patch = ffa[y0:y1, x0:x1]
                    if ffa_patch.shape[0] < tile or ffa_patch.shape[1] < tile:
                        ffa_padded = np.zeros((tile, tile, 2), dtype=ffa_patch.dtype)
                        ffa_padded[: ffa_patch.shape[0], : ffa_patch.shape[1]] = ffa_patch
                        ffa_patch = ffa_padded
                    patch_full = np.concatenate([patch_full, ffa_patch.astype(np.float32)], axis=2)

                tensor = torch.from_numpy(patch_full / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                amp_torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}[amp_dtype]
                with torch.amp.autocast("cuda", dtype=amp_torch_dtype, enabled=amp_dtype != "none" and device.type == "cuda"):
                    predictions = model(tensor)
                probs = torch.sigmoid(predictions[-1][:, :3]).float().cpu().numpy()[0]  # 3xHxW
                probs = np.transpose(probs, (1, 2, 0))[:ph, :pw]  # HxWx3

                accum[y0:y1, x0:x1] += probs
                weight[y0:y1, x0:x1] += 1.0

    weight = np.maximum(weight, 1e-6)
    return accum / weight


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="task1", choices=["task1", "task2"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--masks-dir", type=str, default=None)
    p.add_argument("--ffa-dir", type=str, default=None, help="Task2 only: dir with registered FFA_A/FFA_AV subfolders (see minima_wrapper.py)")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--tile", type=int, default=384, help="Match training patch size (384 verified stable on the T550)")
    p.add_argument("--stride", type=int, default=288)
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    p.add_argument(
        "--quantize-levels", type=int, default=32,
        help="Quantize output PNGs to this many distinct levels to keep submission zips under AI Studio's "
             "100MB limit (32 gave ~55%% smaller files with negligible precision loss in testing, since "
             "every disclosed Task1/Task2 metric is threshold-based). 0 disables (full 8-bit precision).",
    )
    args = p.parse_args()

    if args.task == "task2" and not args.ffa_dir:
        raise SystemExit("--ffa-dir is required for --task task2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.task, base_ch=args.base_ch, iterations=args.iterations, pretrained=False).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    ffa_dir = Path(args.ffa_dir) if args.ffa_dir else None

    image_paths = sorted(images_dir.glob("*.png"))
    print(f"Running {args.task} inference on {len(image_paths)} images -> {out_dir}")
    for img_path in image_paths:
        image = np.array(Image.open(img_path).convert("RGB"))

        ffa = None
        if ffa_dir is not None:
            ffa_a = np.array(Image.open(ffa_dir / "FFA_A" / img_path.name).convert("L"))
            ffa_av = np.array(Image.open(ffa_dir / "FFA_AV" / img_path.name).convert("L"))
            ffa = np.stack([ffa_a, ffa_av], axis=2)

        probs = sliding_window_predict(model, image, device, ffa=ffa, tile=args.tile, stride=args.stride, amp_dtype=args.amp_dtype)

        if masks_dir is not None:
            roi = np.array(Image.open(masks_dir / img_path.name).convert("L"))
            probs[roi == 0] = 0.0

        out_img = (probs * 255).astype(np.uint8)
        if args.quantize_levels > 0:
            # Quantize to fewer distinct levels -- dramatically improves PNG
            # compression (32 levels: ~55% smaller in testing) with minimal
            # precision loss. Worth doing since Task1+Task2 combined submissions
            # exceeded AI Studio's 100MB upload limit (137MB observed for 100
            # full-res probability maps) -- and every disclosed Task1/Task2
            # metric (DSC/Sen/Spec/Acc/COR/INF) is threshold-based anyway, so
            # coarser-than-8-bit precision costs effectively nothing.
            n = args.quantize_levels - 1
            out_img = (np.round(out_img.astype(np.float32) / 255 * n) / n * 255).astype(np.uint8)
        Image.fromarray(out_img, mode="RGB").save(out_dir / img_path.name, compress_level=9)
        print(f"  {img_path.name}")


if __name__ == "__main__":
    main()
