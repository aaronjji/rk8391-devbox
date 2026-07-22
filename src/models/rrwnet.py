"""RRWNet / CMRRWNet model + pretrained-weight loading.

Imports the architecture directly from the vendored external/rrwnet and
external/cmrrwnet submodules rather than re-implementing it (both MIT
licensed). `second_u` (the recursive refinement module) has tied weights
across iterations, so `iterations` is just a runtime loop count -- it doesn't
affect the state_dict shape, and pretrained weights load regardless of the
iteration count used at train/inference time.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "external" / "rrwnet"))
sys.path.insert(0, str(_REPO_ROOT / "external" / "cmrrwnet"))

from model import RRWNet  # noqa: E402
import importlib.util as _ilu  # noqa: E402

_cmrrwnet_spec = _ilu.spec_from_file_location(
    "cmrrwnet_model", _REPO_ROOT / "external" / "cmrrwnet" / "model.py"
)
_cmrrwnet_model = _ilu.module_from_spec(_cmrrwnet_spec)
_cmrrwnet_spec.loader.exec_module(_cmrrwnet_model)
CMRRWNet = _cmrrwnet_model.CMRRWNet


def load_hrf_pretrained(model: RRWNet, strict: bool = False) -> RRWNet:
    """Warm-start a 3-channel-input RRWNet from j-morano/rrwnet-hrf weights."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights_path = hf_hub_download("j-morano/rrwnet-hrf", "model.safetensors")
    state_dict = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    # ConvBlock registers conv1/conv2 both directly and via nn.Sequential
    # (self.conv_block), so state_dict() lists the same tensor under two
    # names -- "missing" entries under *.conv_block.{0,2}.* are harmless
    # aliases of the already-loaded *.conv1/*.conv2, not unloaded weights.
    real_missing = [k for k in missing if ".conv_block." not in k]
    if real_missing or unexpected:
        print(f"[load_hrf_pretrained] missing={real_missing} unexpected={unexpected}")
    return model


def transfer_task1_to_task2(task2_model: CMRRWNet, task1_state_dict: dict, strict_shapes: bool = True) -> CMRRWNet:
    """Warm-starts CMRRWNet's shared components from a trained RRWNet (Task1)
    checkpoint.

    RRWNet.first_u (UNetModule) and CMRRWNet.first_u (NewUNetModule) share
    identical decoder layer names/shapes (upconv1-4, conv6-9, outconv) --
    those transfer directly. The RGB encoder differs only in naming
    (RRWNet's generic conv1..conv5 vs CMRRWNet's conv1_rgb..conv5_rgb, same
    shapes) -- remapped below. CMRRWNet's FFA-branch encoders (conv1_a..5_a)
    and SE/channel-attention layers have no Task1 counterpart and are left
    at their random init. `second_u` (recursive refinement) is the exact
    same architecture in both models and transfers 1:1 by name.
    """
    task2_sd = task2_model.state_dict()
    new_sd = dict(task2_sd)
    transferred, skipped = [], []

    rgb_encoder_map = {f"first_u.conv{i}.": f"first_u.conv{i}_rgb." for i in range(1, 6)}
    decoder_prefixes = ["first_u.upconv1.", "first_u.upconv2.", "first_u.upconv3.", "first_u.upconv4.",
                         "first_u.conv6.", "first_u.conv7.", "first_u.conv8.", "first_u.conv9.", "first_u.outconv."]

    for src_key, src_val in task1_state_dict.items():
        dst_key = src_key
        for src_prefix, dst_prefix in rgb_encoder_map.items():
            if src_key.startswith(src_prefix):
                dst_key = src_key.replace(src_prefix, dst_prefix, 1)
                break
        # second_u.* and the decoder_prefixes above keep the same key name in both models.

        if dst_key not in new_sd:
            skipped.append((src_key, dst_key, "no matching key"))
            continue
        if new_sd[dst_key].shape != src_val.shape:
            skipped.append((src_key, dst_key, f"shape mismatch {new_sd[dst_key].shape} vs {src_val.shape}"))
            if strict_shapes:
                continue
        new_sd[dst_key] = src_val
        transferred.append(dst_key)

    task2_model.load_state_dict(new_sd, strict=True)
    print(f"[transfer_task1_to_task2] transferred {len(transferred)} tensors, skipped {len(skipped)}")
    if skipped[:5]:
        print(f"  sample skipped: {skipped[:5]}")
    return task2_model


def build_model(task: str, base_ch: int = 64, iterations: int = 5, pretrained: bool = True):
    """task: 'task1' (3ch CFP), 'task2' (5ch CFP+FFA, SE-gated additive fusion),
    or 'task2_xattn' (5ch CFP+FFA, cross-attention fusion -- see
    models/cmrrwnet_xattn.py).
    """
    if task == "task1":
        model = RRWNet(input_ch=3, output_ch=3, base_ch=base_ch, iterations=iterations)
        if pretrained:
            model = load_hrf_pretrained(model)
        return model
    elif task == "task2":
        model = CMRRWNet(input_ch=5, output_ch=3, base_ch=base_ch, iterations=iterations)
        # No automatic pretrained warm-start here -- CMRRWNet's encoder
        # differs structurally from RRWNet's (3-branch vs 1). Use
        # transfer_task1_to_task2() explicitly with a trained Task1
        # checkpoint (see train_task2.py --warm-start-task1).
        return model
    elif task == "task2_xattn":
        from .cmrrwnet_xattn import CMRRWNetXAttn
        return CMRRWNetXAttn(input_ch=5, output_ch=3, base_ch=base_ch, iterations=iterations)
    else:
        raise ValueError(f"Unknown task: {task}")
