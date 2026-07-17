"""Task 1 (CFP-only) and Task 2 (CFP+FFA) AV segmentation datasets.

Native-resolution random-crop patches (see plan doc: whole-image downscaling
was rejected as it loses thin-vessel detail relevant to the COR/INF topology
metric; native 512x512 patches fit the local 4GB GPU instead).
"""
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biomarkers.labels import gt_to_prediction_format  # noqa: E402


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def _read_gray(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("L"))


def _min_vessel_content(label_full: np.ndarray, min_frac: float = 0.002) -> bool:
    vessel = label_full[..., 1] > 0  # G channel = all-vessel (full)
    return vessel.mean() >= min_frac


class GaveAVDataset(Dataset):
    """AV segmentation dataset. use_ffa=False -> Task1 (3ch CFP), True -> Task2 (5ch CFP+FFA_A+FFA_AV)."""

    def __init__(
        self,
        data_root: str,
        split: str,
        case_ids: list[int],
        patch_size: int = 512,
        use_ffa: bool = False,
        train: bool = True,
        max_crop_attempts: int = 10,
        seed: int = 0,
        ffa_root: str | None = None,
    ):
        """ffa_root: directory containing registered FFA_A/FFA_AV images
        (see src/registration/minima_wrapper.py -- CFP/FFA pairs are not
        pre-registered, verified empirically). Defaults to `data_root` if
        not given, i.e. assumes FFA is already registered there."""
        self.root = Path(data_root) / split
        self.ffa_root = Path(ffa_root) / split if ffa_root else self.root
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.use_ffa = use_ffa
        self.train = train
        self.max_crop_attempts = max_crop_attempts
        self.rng = np.random.default_rng(seed)

        geo = [
            A.RandomCrop(height=patch_size, width=patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), shear=(-5, 5), p=0.5, border_mode=0),
        ]
        photo = [
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8, p=0.3),
            A.CLAHE(p=0.3),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.3),
        ]
        additional_targets = {"label": "image", "roi": "mask"}
        if use_ffa:
            additional_targets["ffa_a"] = "image"
            additional_targets["ffa_av"] = "image"

        if train:
            self.transform = A.Compose(geo + photo, additional_targets=additional_targets)
        else:
            self.transform = A.Compose(
                [A.RandomCrop(height=patch_size, width=patch_size)], additional_targets=additional_targets
            )

    def __len__(self):
        return len(self.case_ids)

    def _case_name(self, i: int) -> str:
        return f"g_{self.case_ids[i]:03d}"

    def __getitem__(self, idx):
        name = self._case_name(idx)
        image = _read_rgb(self.root / "images" / f"{name}.png")
        raw_gt = _read_rgb(self.root / "av" / f"{name}.png")
        label_full = gt_to_prediction_format(raw_gt)  # HxWx3 uint8, R=artery,G=vessel,B=vein
        roi = _read_gray(self.root / "masks" / f"{name}.png")

        kwargs = {"image": image, "label": label_full, "roi": roi}
        if self.use_ffa:
            kwargs["ffa_a"] = _read_gray(self.ffa_root / "FFA_A" / f"{name}.png")[..., None].repeat(3, axis=2)
            kwargs["ffa_av"] = _read_gray(self.ffa_root / "FFA_AV" / f"{name}.png")[..., None].repeat(3, axis=2)

        for attempt in range(self.max_crop_attempts):
            out = self.transform(**kwargs)
            if not self.train or _min_vessel_content(out["label"]):
                break

        img_t = torch.from_numpy(out["image"].astype(np.float32) / 255.0).permute(2, 0, 1)
        if self.use_ffa:
            a_t = torch.from_numpy(out["ffa_a"][..., :1].astype(np.float32) / 255.0).permute(2, 0, 1)
            av_t = torch.from_numpy(out["ffa_av"][..., :1].astype(np.float32) / 255.0).permute(2, 0, 1)
            img_t = torch.cat([img_t, a_t, av_t], dim=0)

        label_t = torch.from_numpy(out["label"].astype(np.float32) / 255.0).permute(2, 0, 1)
        roi_t = torch.from_numpy((out["roi"] > 0).astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)

        return {"image": img_t, "label": label_t, "roi": roi_t, "case_id": self.case_ids[idx]}
