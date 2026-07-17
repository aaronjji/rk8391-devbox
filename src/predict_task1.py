"""Full-resolution sliding-window inference for a trained Task 1 checkpoint.

Training happens on 512x512 native-resolution patches (GPU VRAM constraint);
inference tiles the full 1536x1024 image with overlap and averages logits in
overlap regions, so predictions are produced at full resolution as the
submission format requires -- no downscaling.

Usage:
    python src/predict_task1.py --checkpoint runs/task1/fold0/latest.pth \
        --images-dir data/raw/GAVE2_preliminary/validation/images \
        --masks-dir data/raw/GAVE2_preliminary/validation/masks \
        --out-dir predictions/task1/validation
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
    tile: int = 384,
    stride: int = 288,
    amp_dtype: str = "bf16",
) -> np.ndarray:
    """image: HxWx3 uint8. Returns HxWx3 float32 sigmoid probabilities
    (artery, vessel, vein), tiled with overlap-averaging."""
    h, w = image.shape[:2]
    accum = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)

    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if not ys or ys[-1] + tile < h:
        ys.append(max(h - tile, 0))
    if not xs or xs[-1] + tile < w:
        xs.append(max(w - tile, 0))

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

                tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
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
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--masks-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--tile", type=int, default=384, help="Match training patch size (384 verified stable on the T550)")
    p.add_argument("--stride", type=int, default=288)
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("task1", base_ch=args.base_ch, iterations=args.iterations, pretrained=False).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else None

    image_paths = sorted(images_dir.glob("*.png"))
    print(f"Running inference on {len(image_paths)} images -> {out_dir}")
    for img_path in image_paths:
        image = np.array(Image.open(img_path).convert("RGB"))
        probs = sliding_window_predict(model, image, device, tile=args.tile, stride=args.stride, amp_dtype=args.amp_dtype)

        if masks_dir is not None:
            roi = np.array(Image.open(masks_dir / img_path.name).convert("L"))
            probs[roi == 0] = 0.0

        out_img = (probs * 255).astype(np.uint8)
        Image.fromarray(out_img, mode="RGB").save(out_dir / img_path.name)
        print(f"  {img_path.name}")


if __name__ == "__main__":
    main()
