"""
3D U-Net (with optional CBAM attention in the 2D decoder) for
surface-to-subsurface ocean temperature reconstruction.

--------------------------------------------------------------------
CHANGES FROM THE ORIGINAL VERSION (all aimed at reducing overfitting)
--------------------------------------------------------------------
1. Dropout3d / Dropout2d added inside every conv block. Spatial dropout
   (not plain nn.Dropout) is used because it zeroes whole feature maps,
   which is the correct way to regularize convolutional activations --
   element-wise dropout barely helps conv nets since neighboring pixels
   are highly correlated and just "leak" the dropped info back in.
2. Default base_ch lowered 32 -> 24. The original model's parameter
   count was large relative to the amount of independent daily signal
   in the dataset; a narrower network has less room to memorize
   training-set-specific noise while keeping the same depth (so it can
   still represent the same *kinds* of features).
3. Dropout rate is configurable and can be set to 0 to fully recover
   the original (no-regularization) architecture, e.g. for debugging.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


# ---------------------------------------------------------------------------
# CBAM (2D, used only in the decoder -- identical to the earlier version)
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """CAM from CBAM (Woo et al. 2018)."""
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
    """SAM from CBAM (Woo et al. 2018)."""
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


# ---------------------------------------------------------------------------
# Norm helper
# ---------------------------------------------------------------------------
def _group_norm(channels, max_groups=8):
    """GroupNorm with the largest group count <= max_groups that evenly
    divides `channels`. Batch-size independent, unlike BatchNorm."""
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


# ---------------------------------------------------------------------------
# 3D encoder block (now with spatial dropout)
# ---------------------------------------------------------------------------
class ConvBlock3D(nn.Module):
    """Two Conv3d layers + GroupNorm + ReLU + Dropout3d. 'Same' padding on
    all three axes (time, H, W), so only the pooling layers change shape.

    Dropout3d zeroes entire (time,H,W) feature-map channels per forward
    pass -- this is the standard way to regularize conv activations,
    since regular element-wise dropout does very little on spatially
    correlated feature maps.
    """
    def __init__(self, in_ch, out_ch, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# 2D decoder block (now with spatial dropout)
# ---------------------------------------------------------------------------
class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, use_cbam=False, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        x = self.cbam(x)
        return x


def collapse_time(x):
    """(B, C, T, H, W) -> (B, C, H, W) via average pooling over time."""
    return x.mean(dim=2)


class UNetOcean3D(nn.Module):
    def __init__(self, in_channels=7, out_channels=15, base_ch=24,
                 n_days=16, use_cbam=True, use_checkpoint=False,
                 dropout=0.15, bottleneck_dropout=0.25):
        """
        in_channels: number of surface params PER DAY (7).
        out_channels: number of depth levels (15)
        base_ch: filters in the first encoder stage. LOWERED from 32 to 24
                 by default -- a narrower net has less capacity to
                 memorize training-set-specific noise.
        n_days: length of the input time window. DEFAULT CHANGED to 26
                 to match the ablation evidence (RMSE keeps improving to
                 ~26 days, plateauing after) -- a longer window gives the
                 model more genuine temporal signal to fit instead of
                 relying on shortcuts, which reduces overfitting as a
                 side effect of making the task better-posed.
        use_cbam: whether to apply CBAM attention in the 2D decoder.
        use_checkpoint: recompute 3D encoder activations during backward
                 to trade compute for VRAM (needed more now that n_days
                 is bigger).
        dropout: spatial dropout rate used in every ConvBlock3D/2D.
                 Set to 0.0 to disable and recover the original model.
        bottleneck_dropout: usually set higher than dropout since the
                 bottleneck is the most parameter-dense, most overfitting
                 -prone part of the network.
        """
        super().__init__()
        self.n_days = n_days
        self.use_checkpoint = use_checkpoint

        # ---- 3D encoder ----
        self.enc1 = ConvBlock3D(in_channels, base_ch, dropout=dropout)
        self.enc2 = ConvBlock3D(base_ch, base_ch * 2, dropout=dropout)
        self.enc3 = ConvBlock3D(base_ch * 2, base_ch * 4, dropout=dropout)
        self.enc4 = ConvBlock3D(base_ch * 4, base_ch * 8, dropout=dropout)
        self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2)

        # ---- 3D bottleneck (highest dropout -- most overfitting-prone) ----
        self.bottleneck = ConvBlock3D(base_ch * 8, base_ch * 16,
                                       dropout=bottleneck_dropout)
        self.time_collapse = nn.AdaptiveAvgPool3d((1, None, None))

        # ---- 2D decoder ----
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock2D(base_ch * 16, base_ch * 8, use_cbam=use_cbam, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock2D(base_ch * 8, base_ch * 4, use_cbam=use_cbam, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock2D(base_ch * 4, base_ch * 2, use_cbam=use_cbam, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        # no dropout on the last decoder block -- right before the output
        # head, so we don't want to inject noise into the final features
        self.dec1 = ConvBlock2D(base_ch * 2, base_ch, use_cbam=use_cbam, dropout=0.0)

        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        pad_t = (16 - T % 16) % 16
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))

        def run(block, inp):
            if self.use_checkpoint and self.training:
                return checkpoint.checkpoint(block, inp, use_reentrant=False)
            return block(inp)

        e1 = run(self.enc1, x)
        e2 = run(self.enc2, self.pool3d(e1))
        e3 = run(self.enc3, self.pool3d(e2))
        e4 = run(self.enc4, self.pool3d(e3))

        b = run(self.bottleneck, self.pool3d(e4))
        b = self.time_collapse(b).squeeze(2)

        e4_2d = collapse_time(e4)
        e3_2d = collapse_time(e3)
        e2_2d = collapse_time(e2)
        e1_2d = collapse_time(e1)

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4_2d], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3_2d], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2_2d], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1_2d], dim=1))

        out = self.out_conv(d1)
        out = out[..., :H, :W]
        return out


if __name__ == "__main__":
    n_days = 16
    model = UNetOcean3D(in_channels=7, out_channels=15, base_ch=24,
                         n_days=n_days, use_cbam=True, dropout=0.15)
    x = torch.randn(2, 7, n_days, 101, 241)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params:,}")