"""Phase 7 — GeoSR torch model integration for the backend pipeline.

This module makes the trained `model/` package usable from the backend without
taking a hard dependency on torch: if torch (or the model package) is not
importable, callers fall back to the bicubic baseline. The integration is
opt-in via the `GEOSSR_CHECKPOINT` setting.

Two responsibilities:
  1. ``build_model_fn`` — wraps the trained CNN as the ``model_fn`` callable
     expected by ``app.processing.pipeline.process_satellite_image``. It receives
     normalized reflectance batches [B, C, H, W] (DN/10000, range [0, 1.5] —
     matching the model's training normalization) and returns super-resolved
     reflectance batches [B, C, H*s, W*s].
  2. ``run_validation`` — post-processing step that produces the Phase 7
     artifacts: a per-pixel confidence/uncertainty GeoTIFF, a uint8 preview PNG,
     and a unified validation report JSON. When no HR reference is available it
     writes the report with PSNR/SSIM/SAM = null and the explicit message
     "Reference-based quantitative validation unavailable for this scene."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

from app.core.logging import logger


def _ensure_repo_root() -> None:
    """Make the repository root importable so `model/` is discoverable."""
    from app.core.config import settings
    root = str(settings.repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)


GEOSSR_AVAILABLE = False
_torch = None
_build_advanced = None
_build_baseline = None


def _try_import() -> bool:
    global GEOSSR_AVAILABLE, _torch, _build_advanced, _build_baseline
    if GEOSSR_AVAILABLE:
        return True
    try:
        _ensure_repo_root()
        import torch  # noqa: F401
        from model.architectures import build_advanced, build_baseline
        _torch = torch
        _build_advanced = build_advanced
        _build_baseline = build_baseline
        GEOSSR_AVAILABLE = True
    except Exception as e:  # torch or model package unavailable
        logger.warning(f"GeoSR torch backend unavailable ({e}); using baseline.")
        GEOSSR_AVAILABLE = False
    return GEOSSR_AVAILABLE


def describe_checkpoint(checkpoint_path) -> dict:
    """Inspect a checkpoint header without instantiating the model.

    Lets the orchestrator reconcile the job's requested ``scale_factor`` with the
    checkpoint's native scale (GeoSRv2 is trained for 10 m -> 5 m, i.e. scale 2
    only). Returns ``{"is_geosr_v2", "native_scale", "model_name", "epoch"}``.
    Any read failure is non-fatal and reports ``is_geosr_v2=False``.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:  # pragma: no cover - best-effort
        return {"is_geosr_v2": False, "native_scale": None,
                "model_name": None, "epoch": None, "error": str(e)}
    ch_cfg = ckpt.get("config", {}) or {}
    arch = ch_cfg.get("architecture", "")
    model_name = ch_cfg.get("model_name") or arch or "advanced"

    def _norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    return {
        "is_geosr_v2": ("geosrv2" in _norm(str(model_name)))
                       or ("geosrv2" in _norm(str(checkpoint_path))),
        "native_scale": ch_cfg.get("scale_factor"),
        "model_name": str(model_name),
        "epoch": ckpt.get("epoch"),
    }


def _normalize_dn_to_reflectance(dn: np.ndarray, scale: float = 10000.0) -> np.ndarray:
    return np.clip(dn.astype(np.float32) / scale, 0.0, 1.5)


def build_model_fn(checkpoint_path: Optional[str], scale_factor: int = 4) -> Optional[callable]:
    """Build a numpy→numpy SR model_fn, or None if unavailable."""
    if not checkpoint_path:
        return None
    if not _try_import():
        logger.warning("GEOSSR_CHECKPOINT set but torch/model unavailable; using bicubic baseline.")
        return None
    import torch
    from model.architectures import build_advanced, build_baseline
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ch_cfg = ckpt.get("config", {}) if ckpt else {}
    # The trained checkpoint stores its descriptor under `architecture` (not
    # `model_name`); read both so the v2 path is detected regardless of the key
    # the producer used.
    model_name = ch_cfg.get("model_name") or ch_cfg.get("architecture", "advanced")
    base_ch = ch_cfg.get("base_channels", 32)

    def _norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    # Detect GeoSRv2 from the config descriptor AND the checkpoint filename, so
    # it is recognised even if the config omits either field. GeoSRv2 is 10 m ->
    # 5 m (scale 2) only.
    is_geosr_v2 = ("geosrv2" in _norm(str(model_name))) or (
        "geosrv2" in _norm(str(checkpoint_path))
    )
    if is_geosr_v2:
        if scale_factor != 2:
            raise ValueError(
                f"GeoSRv2 was trained for 10m -> 5m (scale factor 2); "
                f"scale_factor={scale_factor} is not supported by this checkpoint."
            )
        from model.architectures.geosr_v2 import build_geosr_v2, load_geosr_v2_checkpoint
        model = build_geosr_v2(num_channels=4)
        info = load_geosr_v2_checkpoint(model, checkpoint_path)
        logger.info(
            f"[geosr_v2] loaded {info['checkpoint']} epoch={info['epoch']} "
            f"params={info['parameter_count']} missing={info['missing_keys']} "
            f"unexpected={info['unexpected_keys']}"
        )
    elif model_name == "baseline":
        model = _build_baseline(num_channels=4, base_channels=base_ch,
                                num_resblocks=ch_cfg.get("num_resblocks", 4), scale_factor=scale_factor)
    else:
        model = _build_advanced(num_channels=4, base_channels=base_ch,
                                num_groups=ch_cfg.get("num_groups", 2),
                                blocks_per_group=ch_cfg.get("blocks_per_group", 2),
                                reduction=ch_cfg.get("reduction", 4), scale_factor=scale_factor)
    if ckpt and "model_state_dict" in ckpt and not is_geosr_v2:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    logger.info(f"Loaded GeoSR torch model '{model_name}' from {checkpoint_path}")

    def model_fn(batch: np.ndarray) -> np.ndarray:
        # batch: [B, C, H, W] reflectance in [0, 1.5]
        x = torch.as_tensor(np.ascontiguousarray(batch), dtype=torch.float32)
        with torch.no_grad():
            out = model(x).float()
        return out.cpu().numpy()

    return model_fn


def _read_band_descriptions(path: Path) -> List[str]:
    with rasterio.open(path) as src:
        return [str(d) if d else f"Band_{i+1}" for i, d in enumerate(src.descriptions)]


def run_validation(
    input_path: Path,
    output_geotiff: Path,
    scale_factor: int,
    out_dir: Path,
) -> Tuple[Optional[Path], Optional[Path], Dict]:
    """Compute confidence map + validation report for a completed SR job.

    Reads the input LR GeoTIFF (DN) and the super-resolved GeoTIFF (DN) written
    by the pipeline, normalises both to reflectance, and produces:
      * confidence_<stem>.png  — uint8 preview heatmap
      * confidence_<stem>.tif  — float32 GeoTIFF (0..1), same CRS/transform as output
      * validation_report_<stem>.json  — Phase 7 unified report

    Returns (confidence_png, confidence_tif, report_dict). Any failure is logged
    and returns (None, None, {}); it never raises (Phase-7 best-effort).
    """
    if not _try_import():
        return None, None, {}
    try:
        from model.evaluation.uncertainty import self_consistency, confidence_to_uint8
        from model.evaluation.validation import build_validation_report
        from model.datasets.splits import BAND_NAMES_10M
        from PIL import Image

        with rasterio.open(input_path) as src:
            lr_dn = src.read()[:4]  # [C, H, W]
            in_crs = src.crs
            in_transform = src.transform
            in_gsd = float(abs(src.transform.a))
        with rasterio.open(output_geotiff) as src:
            sr_dn = src.read()[:4]  # [C, H*s, W*s]
            out_crs = src.crs
            out_transform = src.transform

        lr_refl = _normalize_dn_to_reflectance(lr_dn)
        sr_refl = _normalize_dn_to_reflectance(sr_dn)
        # align band count
        c = min(lr_refl.shape[0], sr_refl.shape[0], 4)
        lr_refl = lr_refl[:c]; sr_refl = sr_refl[:c]
        scale = float(scale_factor)
        data_range = 1.5

        # Confidence is derived from self-consistency (LR↔SR round-trip), which
        # needs no ground-truth HR and no extra forward passes. The CLI
        # (model/inference.py) additionally supports an input-noise ensemble
        # when a live model handle is in scope; the worker here does not carry
        # one, so self-consistency is the appropriate backend choice.
        unc = self_consistency(sr_refl, lr_refl, scale_factor=scale_factor)
        confidence = unc["confidence_map"]  # [H*s, W*s] float32

        # HR reference: not available for real uploads (no ground truth).
        hr_refl = None
        report = build_validation_report(
            pred_refl=sr_refl, lr_refl=lr_refl, hr_refl=hr_refl,
            scale_factor=scale, data_range=data_range,
            band_names=list(BAND_NAMES_10M[:c]),
            uncertainty_summary=unc["summary"],
        )
        report["input_gsd_meters"] = in_gsd
        report["output_gsd_meters"] = in_gsd / scale

        # --- write confidence GeoTIFF (float32) + uint8 preview PNG ---
        out_dir.mkdir(parents=True, exist_ok=True)
        conf_tif = out_dir / f"confidence_{output_geotiff.stem}.tif"
        cprof = {
            "driver": "GTiff", "height": confidence.shape[0], "width": confidence.shape[1],
            "count": 1, "dtype": "float32", "crs": out_crs or in_crs,
            "transform": out_transform or new_out_transform(in_transform, scale),
            "compress": "lzw", "nodata": None,
        }
        with rasterio.open(conf_tif, "w", **cprof) as dst:
            dst.write(confidence.astype("float32"), 1)
            dst.set_band_description(1, "SR confidence (self-consistency, 0-1)")

        conf_png = out_dir / f"confidence_{output_geotiff.stem}.png"
        Image.fromarray(confidence_to_uint8(confidence), mode="L").save(conf_png)

        report_path = out_dir / f"validation_report_{output_geotiff.stem}.json"
        report_path.write_text(json.dumps(report, indent=2))
        logger.info(f"[geosr] confidence -> {conf_tif}; report -> {report_path} "
                    f"score={unc['summary']['confidence_score']:.1f}%")
        return conf_png, conf_tif, report
    except Exception as e:
        logger.exception(f"[geosr] validation failed (non-fatal): {e}")
        return None, None, {}


def new_out_transform(transform, scale: int):
    from rasterio.transform import Affine
    return Affine(transform.a / scale, transform.b / scale, transform.c,
                  transform.d / scale, transform.e / scale, transform.f)
