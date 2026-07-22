"""Cross-attention fusion variant of CMRRWNet's first_u encoder/fusion stage.

The vendored external/cmrrwnet's NewUNetModule fuses the RGB/FFA-a/FFA-av
branches with plain elementwise addition followed by a channel-only SE-style
gate (see Asy_fusion1-4 in external/cmrrwnet/model.py) -- there is no spatial
interaction between modalities, so a vessel that's visible in FFA but faint in
CFP (or vice versa) can't pull attention toward its own location in the other
branch. This subclasses the vendored classes (external/ stays untouched, per
the project's vendoring convention) and swaps that fusion mechanism for real
spatial cross-attention between RGB and the combined FFA (a+av) branches.

K/V are spatially downsampled before the attention matmul -- full O(HW x HW)
attention at fusion1's resolution (e.g. 160x160 for a 320px patch) is too
expensive to justify here; downsampling K/V to a fixed grid bounds the cost
to O(HW x downsampled) while queries still run at full resolution, so nothing
loses spatial precision on the query side.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .rrwnet import _cmrrwnet_model  # noqa

NewUNetModule = _cmrrwnet_model.NewUNetModule
CMRRWNet = _cmrrwnet_model.CMRRWNet
ConvBlock = _cmrrwnet_model.ConvBlock


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between an RGB feature map and a
    combined FFA (a+av) feature map at one encoder scale, replacing
    Asy_fusion's add + channel-attention.
    """

    def __init__(self, channels: int, num_heads: int = 4, kv_downsample: int = 4):
        super().__init__()
        assert channels % num_heads == 0, f"channels={channels} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.kv_downsample = kv_downsample

        self.q_rgb = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_ffa = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_ffa = nn.Conv2d(channels, channels, 1, bias=False)

        self.q_ffa = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_rgb = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_rgb = nn.Conv2d(channels, channels, 1, bias=False)

        self.out_proj = nn.Conv2d(channels * 2, channels, 1)
        self.norm = nn.GroupNorm(min(32, channels), channels)

    def _attend(self, q, k, v):
        b, c, h, w = q.shape
        if self.kv_downsample > 1:
            k = F.avg_pool2d(k, self.kv_downsample, ceil_mode=True)
            v = F.avg_pool2d(v, self.kv_downsample, ceil_mode=True)
        kh, kw = k.shape[-2:]

        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(-1, -2)          # B,heads,HW,hd
        k = k.reshape(b, self.num_heads, self.head_dim, kh * kw)                       # B,heads,hd,KHW
        v = v.reshape(b, self.num_heads, self.head_dim, kh * kw).transpose(-1, -2)     # B,heads,KHW,hd

        attn = torch.softmax((q @ k) * self.scale, dim=-1)
        out = attn @ v                                                                  # B,heads,HW,hd
        return out.transpose(-1, -2).reshape(b, self.num_heads * self.head_dim, h, w)

    def forward(self, fea_rgb, fea_ffa):
        rgb_attends_ffa = self._attend(self.q_rgb(fea_rgb), self.k_ffa(fea_ffa), self.v_ffa(fea_ffa))
        ffa_attends_rgb = self._attend(self.q_ffa(fea_ffa), self.k_rgb(fea_rgb), self.v_rgb(fea_rgb))
        fused = self.out_proj(torch.cat([rgb_attends_ffa, ffa_attends_rgb], dim=1))
        return self.norm(fused + fea_rgb + fea_ffa)


class NewUNetModuleXAttn(NewUNetModule):
    """Same encoder/decoder as NewUNetModule; replaces Asy_fusion1-4 (additive
    + channel-SE) with CrossAttentionFusion at each of the 4 skip-connection
    scales. fea_a and fea_av are combined into one "ffa" feature via a 1x1
    conv over their concatenation before cross-attending against RGB --
    keeping them as two separate cross-attention branches would double the
    already-nontrivial attention cost for no clear benefit, since both are
    single-channel modalities processed by the same weight-shared encoder.
    """

    def __init__(self, input_ch, output_ch, base_ch, kv_downsamples=(8, 4, 2, 1)):
        super().__init__(input_ch, output_ch, base_ch)
        del self.ca1, self.ca2, self.ca3, self.ca4  # replaced by cross-attention below

        dims = [2 * base_ch, 4 * base_ch, 8 * base_ch, 16 * base_ch]
        self.ffa_combine1 = nn.Conv2d(2 * dims[0], dims[0], 1)
        self.ffa_combine2 = nn.Conv2d(2 * dims[1], dims[1], 1)
        self.ffa_combine3 = nn.Conv2d(2 * dims[2], dims[2], 1)
        self.ffa_combine4 = nn.Conv2d(2 * dims[3], dims[3], 1)

        heads = [4, 4, 8, 8]
        self.xattn1 = CrossAttentionFusion(dims[0], num_heads=heads[0], kv_downsample=kv_downsamples[0])
        self.xattn2 = CrossAttentionFusion(dims[1], num_heads=heads[1], kv_downsample=kv_downsamples[1])
        self.xattn3 = CrossAttentionFusion(dims[2], num_heads=heads[2], kv_downsample=kv_downsamples[2])
        self.xattn4 = CrossAttentionFusion(dims[3], num_heads=heads[3], kv_downsample=kv_downsamples[3])

    def Asy_fusion1(self, fea_rgb, fea_a, fea_av):
        fea_rgb = self.conv2_rgb(F.max_pool2d(fea_rgb, 2, 2))
        fea_ffa = self.ffa_combine1(torch.cat([fea_a, fea_av], dim=1))
        return self.xattn1(fea_rgb, fea_ffa)

    def Asy_fusion2(self, fea_rgb, fea_a, fea_av):
        fea_rgb = self.conv3_rgb(F.max_pool2d(fea_rgb, 2, 2))
        fea_ffa = self.ffa_combine2(torch.cat([fea_a, fea_av], dim=1))
        return self.xattn2(fea_rgb, fea_ffa)

    def Asy_fusion3(self, fea_rgb, fea_a, fea_av):
        fea_rgb = self.conv4_rgb(F.max_pool2d(fea_rgb, 2, 2))
        fea_ffa = self.ffa_combine3(torch.cat([fea_a, fea_av], dim=1))
        return self.xattn3(fea_rgb, fea_ffa)

    def Asy_fusion4(self, fea_rgb, fea_a, fea_av):
        fea_rgb = self.conv5_rgb(F.max_pool2d(fea_rgb, 2, 2))
        fea_ffa = self.ffa_combine4(torch.cat([fea_a, fea_av], dim=1))
        return self.xattn4(fea_rgb, fea_ffa)


class CMRRWNetXAttn(CMRRWNet):
    def __init__(self, input_ch=5, output_ch=3, base_ch=64, iterations=5):
        super().__init__(input_ch, output_ch, base_ch, iterations)
        self.first_u = NewUNetModuleXAttn(input_ch, output_ch, base_ch)
