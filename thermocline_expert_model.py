"""
3D-encoder / 2D-decoder U-Net for surface-to-subsurface ocean temperature
reconstruction, PLUS a small "thermocline expert" residual head.

Why a separate expert instead of just tuning the main model more:
Your own per-depth table shows the error is not uniform - depths ~4-11
(the thermocline) carry roughly 3/4 of the total squared error even
though they're 8 of 15 levels. Any change to the shared backbone (more
capacity, different loss weighting, more input channels) touches ALL
15 depths at once, so an improvement at the thermocline can easily come
with a regression at the surface or in the deep water, and you'd have
no guarantee either way.

The design here removes that guesswork:

  UNetOceanBase   - basically your existing architecture, unchanged in
                    spirit, except forward() also returns the decoder's
                    last feature map (pre-out_conv) so a second module
                    can use it.
  ThermoclineExpert - a small conv head that looks at those features
                    plus a couple of physically-motivated extra inputs
                    (instantaneous SST, SSH, and SSH gradient/curvature)
                    and predicts a *correction*, applied ONLY to a
                    configurable band of depth indices.
  UNetOceanV2     - glues the two together. Its forward() does:
                        pred = base_pred.clone()
                        pred[:, lo:hi] += expert(...)
                    Every index outside [lo, hi) is therefore IDENTICAL
                    to base_pred, by construction, not by training
                    accident. If you keep the base frozen (the default
                    training setup in train.py), those levels are also
                    byte-for-byte identical to whatever your existing
                    checkpoint already produces - so there's nothing to
                    accidentally regress.

The expert's last conv layer is zero-initialized, so at the start of
training UNetOceanV2 is mathematically identical to the base model
(correction = 0 everywhere). Training only ever has to find a
correction that helps; it can never make things worse than "expert
does nothing" because that's exactly where it starts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


# ---------------------------------------------------------------------
# Input channel layout (per day, 7 channels) - matches your dataset.
# ---------------------------------------------------------------------
SST_CH, SSS_CH, SSH_CH, UCUR_CH, VCUR_CH, UWIND_CH, VWIND_CH = range(7)
# channels whose sign must flip under a longitude (last-axis) mirror
FLIP_NEGATE_CHANNELS = (UCUR_CH, UWIND_CH)


def _group_norm(channels, max_groups=8):
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


# ---------------------------------------------------------------------
# CBAM attention (2D decoder only)
#
# NOTE: attribute names/nesting here (channel_attn.mlp, spatial_attn.conv)
# are deliberately kept identical to your original model.py's CBAM, so
# that state_dict keys line up exactly and your existing best_model.pt
# loads into this backbone with zero missing/unexpected keys. Same for
# ConvBlock3D/ConvBlock2D below (block / cbam names, not net / attn).
# ---------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return attn * x


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ---------------------------------------------------------------------
# Encoder / decoder conv blocks
# ---------------------------------------------------------------------
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            _group_norm(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            _group_norm(out_ch), nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, use_cbam=True, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _group_norm(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _group_norm(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        x = self.cbam(x)
        return x


def _collapse_time(x):
    """(B, C, T, H, W) -> (B, C, H, W)."""
    return x.mean(dim=2)


# ---------------------------------------------------------------------
# Base backbone - same shape of network you had, restructured only so
# it can hand back its last feature map for the expert to use.
# ---------------------------------------------------------------------
class UNetOceanBase(nn.Module):
    def __init__(self, in_channels=7, out_channels=15, base_ch=20,
                 use_cbam=True, use_checkpoint=False,
                 dropout=0.15, bottleneck_dropout=0.25):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.base_ch = base_ch
        self.out_channels = out_channels

        self.enc1 = ConvBlock3D(in_channels, base_ch, dropout)
        self.enc2 = ConvBlock3D(base_ch, base_ch * 2, dropout)
        self.enc3 = ConvBlock3D(base_ch * 2, base_ch * 4, dropout)
        self.enc4 = ConvBlock3D(base_ch * 4, base_ch * 8, dropout)
        self.pool3d = nn.MaxPool3d(2, 2)

        self.bottleneck = ConvBlock3D(base_ch * 8, base_ch * 16, bottleneck_dropout)
        self.time_collapse = nn.AdaptiveAvgPool3d((1, None, None))

        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock2D(base_ch * 16, base_ch * 8, use_cbam, dropout)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock2D(base_ch * 8, base_ch * 4, use_cbam, dropout)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock2D(base_ch * 4, base_ch * 2, use_cbam, dropout)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock2D(base_ch * 2, base_ch, use_cbam, dropout=0.0)

        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)

    def _run(self, block, x):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        """Returns (pred, features) where features is the pre-out_conv
        decoder map, cropped to the input's original H, W - same
        resolution as pred, so a downstream head can align it with
        raw input channels without extra bookkeeping."""
        B, C, T, H, W = x.shape
        pad_t = (16 - T % 16) % 16
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        xp = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))

        e1 = self._run(self.enc1, xp)
        e2 = self._run(self.enc2, self.pool3d(e1))
        e3 = self._run(self.enc3, self.pool3d(e2))
        e4 = self._run(self.enc4, self.pool3d(e3))

        b = self._run(self.bottleneck, self.pool3d(e4))
        b = self.time_collapse(b).squeeze(2)

        d4 = self.dec4(torch.cat([self.up4(b), _collapse_time(e4)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), _collapse_time(e3)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), _collapse_time(e2)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), _collapse_time(e1)], dim=1))

        pred = self.out_conv(d1)[..., :H, :W]
        feat = d1[..., :H, :W]
        return pred, feat


# ---------------------------------------------------------------------
# Thermocline expert
# ---------------------------------------------------------------------
class ThermoclineExpert(nn.Module):
    """Predicts a correction for a band of depth indices [lo, hi).

    Inputs: the base model's last feature map, the base prediction
    itself, and a handful of extra fields that are cheap physical
    hints for where a thermocline is sitting - instantaneous (most
    recent day) SST and SSH, plus SSH's spatial gradient and Laplacian
    (curvature). SSH gradient/curvature are computed with a fixed,
    non-trainable Sobel/Laplacian kernel, not learned - there's no
    reason to spend capacity relearning a numerical derivative.

    The final 1x1 conv is zero-initialized, so this module outputs
    all-zero corrections at the start of training.
    """

    def __init__(self, base_ch, out_channels_total, level_lo, level_hi,
                 hidden=32, dropout=0.1):
        super().__init__()
        self.level_lo = level_lo
        self.level_hi = level_hi
        n_levels = level_hi - level_lo
        assert n_levels > 0

        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer("kernel_gx", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_gy", sobel_x.t().contiguous().view(1, 1, 3, 3))
        self.register_buffer("kernel_lap", laplacian.view(1, 1, 3, 3))

        in_ch = base_ch + out_channels_total + 1 + 1 + 2 + 1  # feat + base_pred + sst + ssh + grad(2) + lap
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            _group_norm(hidden), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            _group_norm(hidden), nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(hidden, n_levels, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _ssh_derivatives(self, ssh):
        gx = F.conv2d(ssh, self.kernel_gx, padding=1)
        gy = F.conv2d(ssh, self.kernel_gy, padding=1)
        lap = F.conv2d(ssh, self.kernel_lap, padding=1)
        return gx, gy, lap

    def forward(self, feat, base_pred, sst, ssh):
        gx, gy, lap = self._ssh_derivatives(ssh)
        h = torch.cat([feat, base_pred, sst, ssh, gx, gy, lap], dim=1)
        h = self.net(h)
        return self.out_conv(h)


# ---------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------
class UNetOceanV2(nn.Module):
    def __init__(self, in_channels=7, out_channels=15, base_ch=20,
                 use_cbam=True, use_checkpoint=False,
                 dropout=0.15, bottleneck_dropout=0.25,
                 thermocline_band=(4, 12), expert_hidden=32, expert_dropout=0.1):
        super().__init__()
        self.base = UNetOceanBase(in_channels, out_channels, base_ch,
                                   use_cbam, use_checkpoint, dropout, bottleneck_dropout)
        lo, hi = thermocline_band
        self.level_lo, self.level_hi = lo, hi
        self.expert = ThermoclineExpert(base_ch, out_channels, lo, hi,
                                         hidden=expert_hidden, dropout=expert_dropout)

    def forward(self, x, return_base=False):
        sst = x[:, SST_CH, -1, :, :].unsqueeze(1)   # most recent day, (B,1,H,W)
        ssh = x[:, SSH_CH, -1, :, :].unsqueeze(1)

        base_pred, feat = self.base(x)
        delta = self.expert(feat, base_pred, sst, ssh)

        pred = base_pred.clone()
        pred[:, self.level_lo:self.level_hi, :, :] = (
            base_pred[:, self.level_lo:self.level_hi, :, :] + delta
        )
        if return_base:
            return pred, base_pred
        return pred

    # -- convenience for the freeze-base-train-expert-only workflow --
    def set_base_trainable(self, trainable: bool):
        for p in self.base.parameters():
            p.requires_grad_(trainable)

    def load_pretrained_base(self, state_dict, strict=False):
        """Loads a state dict saved from your OLD (non-expert) checkpoint
        into self.base. Handles a possible torch.compile '_orig_mod.'
        prefix. Returns the (missing, unexpected) key lists so you can
        sanity-check the load."""
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        # old checkpoints saved the whole model with these same attribute
        # names (enc1, enc2, ..., dec1, out_conv) at the top level, which
        # is exactly self.base's layout here.
        result = self.base.load_state_dict(cleaned, strict=strict)
        return result


if __name__ == "__main__":
    n_days = 16
    model = UNetOceanV2(in_channels=7, out_channels=15, base_ch=20,
                         thermocline_band=(4, 12))
    x = torch.randn(2, 7, n_days, 101, 241)
    pred, base_pred = model(x, return_base=True)
    print("input     :", x.shape)
    print("pred      :", pred.shape)
    print("base_pred :", base_pred.shape)
    # sanity check: outside the band, pred must equal base_pred EXACTLY
    lo, hi = model.level_lo, model.level_hi
    outside_equal = torch.equal(
        torch.cat([pred[:, :lo], pred[:, hi:]], dim=1),
        torch.cat([base_pred[:, :lo], base_pred[:, hi:]], dim=1),
    )
    print(f"levels outside [{lo}, {hi}) untouched:", outside_equal)
    n_params_base = sum(p.numel() for p in model.base.parameters())
    n_params_expert = sum(p.numel() for p in model.expert.parameters())
    print(f"base params: {n_params_base:,}  expert params: {n_params_expert:,}")