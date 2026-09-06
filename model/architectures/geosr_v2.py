"""
GeoSRv2 — trained super-resolution model for 10 m -> 5 m Sentinel-2 imagery.

This module is an EXACT reproduction of the architecture shipped with the trained
checkpoint ``GeoSR_v2_epoch34_best.pt`` (scale factor 2, 817,092 parameters). It is
intentionally minimal and faithful: no BatchNorm, no Dropout, no attention, no extra
residual blocks, and **no final clamp** (the residual is bounded instead).

Forward graph (matches the model card exactly):

    head:        Conv2d(4, 64, 3, padding=1)
    body:        8 x ResidualBlock(64)        # conv-relu-conv + identity residual
    body_conv:   Conv2d(64, 64, 3, padding=1)
    feat = head(x)
    body_out = body(feat)
    feat = feat + body_conv(body_out)        # feature connection
    upsample_conv:  Conv2d(64, 256, 3, padding=1)
    PixelShuffle(2)
    ReLU
    upsample_out:   Conv2d(64, 64, 3, padding=1)
    ReLU
    tail:        Conv2d(64, 4, 3, padding=1)     # -> raw residual [B,4,2H,2W]
    residual_scale = register_buffer([0.015, 0.020, 0.020, 0.050]).view(1,4,1,1)
    bounded_residual = tanh(raw_residual) * residual_scale
    output = bicubic(x, scale_factor=2, align_corners=False) + bounded_residual

The residual_scale buffer is a non-learned constant and is therefore EXPECTED to be
absent from the checkpoint state_dict; the checkpoint loader treats it as the only
legitimately-missing key.

DO NOT modify channels, blocks, activation, PixelShuffle factor, or residual scales.
The checkpoint is the source of truth; this file exists only to reproduce it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# (B02, B03, B04, B08)
RESIDUAL_SCALE = torch.tensor([0.015, 0.020, 0.020, 0.050], dtype=torch.float32).view(1, 4, 1, 1)

BANDED_RESIDUAL_KEY = "residual_scale"  # registered buffer; allowed missing in ckpt


class ResidualBlock(nn.Module):
    """Conv2d(64,64,3,p=1) -> ReLU -> Conv2d(64,64,3,p=1), output = input + block(input).

    The convs are wrapped in ``self.block`` (a Sequential) so the parameter names
    match the trained checkpoint exactly (``body.<i>.block.0.*`` / ``block.2.*``,
    with ReLU at index 1 having no parameters). No BatchNorm, no Dropout.
    """

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GeoSRv2(nn.Module):
    """Residual-learning super-resolution model: 10 m -> 5 m (2x), 4 bands.

    Parameters
    ----------
    num_channels : int
        Input/output spectral bands (4 for Sentinel-2 B02/B03/B04/B08). Must stay 4.
    channels : int
        Internal feature width (64).
    num_resblocks : int
        Residual blocks in the body (8).
    """

    def __init__(self, num_channels: int = 4, channels: int = 64, num_resblocks: int = 8) -> None:
        super().__init__()
        if num_channels != 4:
            raise ValueError(f"GeoSRv2 expects 4 bands (B02/B03/B04/B08); got {num_channels}")
        if channels != 64:
            raise ValueError(f"GeoSRv2 feature width is fixed at 64; got {channels}")
        if num_resblocks != 8:
            raise ValueError(f"GeoSRv2 body is fixed at 8 residual blocks; got {num_resblocks}")

        self.num_channels = num_channels
        self.scale_factor = 2  # 10 m -> 5 m only

        self.head = nn.Conv2d(num_channels, channels, kernel_size=3, padding=1)

        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_resblocks)])
        self.body_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # 64 -> 256 channels, then PixelShuffle(2) collapses to 64 channels at 2x spatial.
        self.upsample_conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

        self.upsample_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.tail = nn.Conv2d(channels, num_channels, kernel_size=3, padding=1)

        # Non-learned residual bounding, registered as a buffer so it is part of the
        # state-dict contract but is NOT optimized. It is expected to be the only
        # legitimately-missing key when loading a checkpoint that predates registration.
        self.register_buffer(
            BANDED_RESIDUAL_KEY,
            RESIDUAL_SCALE.clone(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, H, W] reflectance in [0, 1.5]
        feat = self.head(x)                       # [B, 64, H, W]
        body_out = self.body(feat)                # [B, 64, H, W]
        feat = feat + self.body_conv(body_out)    # feature connection

        up = self.upsample_conv(feat)             # [B, 256, H, W]
        up = self.pixel_shuffle(up)               # [B, 64, 2H, 2W]
        up = self.relu(up)
        up = self.upsample_out(up)                # [B, 64, 2H, 2W]
        up = self.relu(up)

        raw_residual = self.tail(up)              # [B, 4, 2H, 2W]
        bounded_residual = torch.tanh(raw_residual) * self.residual_scale

        bicubic = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False
        )
        # NO final clamp (by design). Output is a bounded residual off the bicubic base.
        return bicubic + bounded_residual


def build_geosr_v2(num_channels: int = 4, channels: int = 64, num_resblocks: int = 8) -> GeoSRv2:
    """Factory mirroring the trained checkpoint defaults."""
    return GeoSRv2(num_channels=num_channels, channels=channels, num_resblocks=num_resblocks)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_geosr_v2_checkpoint(model: GeoSRv2, checkpoint_path) -> dict:
    """Strictly load a GeoSRv2 checkpoint.

    Policy (per integration contract):
      * ``residual_scale`` (registered buffer) is the ONLY allowed missing key.
      * ANY missing learned parameter (Conv weight/bias) raises RuntimeError —
        the model never silently falls back to random weights.
      * Unexpected keys are returned and surfaced in the log/report for inspection;
        an empty unexpected set is the expected/normal case.
      * Asserts parameter count == 817092 and logs checkpoint path/epoch/architecture.

    Returns a dict with diagnostic metadata.
    """
    logger = logging.getLogger(__name__)
    ckpt_path = Path(checkpoint_path)
    logger.info(f"[geosr_v2] checkpoint path: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    sd = ckpt.get("model_state_dict", ckpt)
    res = model.load_state_dict(sd, strict=False)
    missing = sorted(res.missing_keys)
    unexpected = sorted(res.unexpected_keys)

    allowed_missing = {BANDED_RESIDUAL_KEY}
    learned_missing = [k for k in missing if k not in allowed_missing]

    cfg = ckpt.get("config", {}) or {}
    logger.info(
        f"[geosr_v2] epoch={ckpt.get('epoch')} model_name={cfg.get('model_name') if isinstance(cfg, dict) else 'n/a'} "
        f"params={count_parameters(model)} missing_keys={missing} unexpected_keys={unexpected}"
    )

    if learned_missing:
        # Fail loudly: do NOT silently proceed with random/uninitialized weights.
        raise RuntimeError(
            "GeoSRv2 checkpoint is missing learned parameters (would silently fall back to "
            f"random weights). Missing learned keys: {learned_missing}"
        )
    if count_parameters(model) != 817092:
        raise RuntimeError(
            f"GeoSRv2 parameter count mismatch: expected 817092, got {count_parameters(model)}. "
            "Architecture does not match the trained checkpoint."
        )

    return {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "architecture": "GeoSRv2",
        "parameter_count": count_parameters(model),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


if __name__ == "__main__":
    m = build_geosr_v2()
    lr = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        out = m(lr)
    params = count_parameters(m)
    print("input", tuple(lr.shape), "-> output", tuple(out.shape), "params", params)
    assert out.shape == (1, 4, 128, 128), out.shape
    assert params == 817092, params
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    print("GeoSRv2 self-test OK")
