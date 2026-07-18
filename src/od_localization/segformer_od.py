"""Optic disc localization via pamixsun/segformer_for_optic_disc_cup_segmentation,
a SegFormer pretrained on REFUGE. Replaces the earlier heuristic (brightest-blob)
detector -- validated empirically (2026-07-18): class 0=background, class 1=disc,
class 2=cup (nested inside disc); use class>=1 as the full optic disc region
(cup is anatomically part of the disc, needed for the SIVA zone's disc-diameter
measurement).
"""
import numpy as np
import torch
from torch import nn

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is None:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        _processor = AutoImageProcessor.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
        _model = SegformerForSemanticSegmentation.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
        _model.eval()
    return _processor, _model


def find_od_segformer(image_rgb: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """image_rgb: HxWx3 uint8 RGB. Returns a binary (0/255) uint8 OD mask,
    same HxW, covering disc+cup (class >= 1)."""
    processor, model = _load()
    if device is not None:
        model.to(device)

    inputs = processor(image_rgb, return_tensors="pt")
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu()

    upsampled = nn.functional.interpolate(
        logits, size=image_rgb.shape[:2], mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1)[0].numpy().astype(np.uint8)
    return (pred >= 1).astype(np.uint8) * 255
