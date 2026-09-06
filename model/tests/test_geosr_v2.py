"""
Tests for the GeoSRv2 architecture and integration.

Architecture-level tests (TEST 1, 2, 4, 5) run against a freshly instantiated model
and therefore validate the architecture WITHOUT the trained checkpoint. A parameter
count of exactly 817,092 is only achievable if the architecture matches the checkpoint
specification, so TEST 2 is a strong correctness gate.

Checkpoint-dependent tests (TEST 3, 6, 7, 8, 9) are skipped when
``model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt`` is absent — they never fabricate
results or fall back silently.
"""
import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable so `model.*` resolves when running from any cwd.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from model.architectures.geosr_v2 import (  # noqa: E402
    GeoSRv2,
    RESIDUAL_SCALE,
    build_geosr_v2,
    count_parameters,
    load_geosr_v2_checkpoint,
)

CHECKPOINT = Path(_REPO) / "model" / "checkpoints" / "geosr_v2" / "GeoSR_v2_epoch34_best.pt"
SAMPLE_TIF = Path(_REPO) / "data" / "sample_sentinel2_10m.tif"


# ---------------------------------------------------------------------------
# TEST 1 — Architecture: correct 2x output shape
# ---------------------------------------------------------------------------
def test_geosr_v2_output_shape():
    model = build_geosr_v2()
    x = torch.randn(1, 4, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 256, 256)


# ---------------------------------------------------------------------------
# TEST 2 — Parameter count == 817,092 (proves architecture matches checkpoint)
# ---------------------------------------------------------------------------
def test_geosr_v2_parameter_count():
    model = build_geosr_v2()
    n = count_parameters(model)
    assert n == 817092, f"expected 817092 params, got {n}"


# ---------------------------------------------------------------------------
# TEST 4 — Forward pass: no NaN / no Inf
# ---------------------------------------------------------------------------
def test_geosr_v2_forward_no_nan_inf():
    model = build_geosr_v2()
    model.eval()
    x = torch.randn(2, 4, 64, 64) * 0.5  # reflectance-like range
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 4, 128, 128)
    assert not torch.isnan(out).any(), "output contains NaN"
    assert not torch.isinf(out).any(), "output contains Inf"


# ---------------------------------------------------------------------------
# TEST 5 — Residual bound: |output - bicubic| <= residual_scale per band
# ---------------------------------------------------------------------------
def test_geosr_v2_residual_bound():
    model = build_geosr_v2()
    model.eval()
    x = torch.randn(1, 4, 64, 64) * 0.3
    with torch.no_grad():
        out = model(x)
    bicubic = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
    diff = (out - bicubic).abs()  # [1,4,128,128]
    scales = RESIDUAL_SCALE.view(4)  # [0.015, 0.020, 0.020, 0.050]
    for b in range(4):
        max_diff = diff[0, b].max().item()
        assert max_diff <= scales[b].item() + 1e-6, (
            f"band {b} residual {max_diff} exceeds scale {scales[b].item()}"
        )


# ---------------------------------------------------------------------------
# TEST 3 — Checkpoint loading (only expected missing key: residual_scale)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CHECKPOINT.exists(),
                    reason="GeoSR_v2 checkpoint not present in repo; set model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt")
def test_geosr_v2_checkpoint_loading():
    model = build_geosr_v2()
    info = load_geosr_v2_checkpoint(model, str(CHECKPOINT))
    assert info["parameter_count"] == 817092
    allowed = {"residual_scale"}
    learned_missing = [k for k in info["missing_keys"] if k not in allowed]
    assert learned_missing == [], f"unexpected missing learned keys: {learned_missing}"
    assert info["unexpected_keys"] == [], f"unexpected keys: {info['unexpected_keys']}"
    assert info["epoch"] == 34


# ---------------------------------------------------------------------------
# TEST 6-9 — GeoTIFF inference with the real checkpoint (skipped if absent)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CHECKPOINT.exists(),
                    reason="GeoSR_v2 checkpoint not present in repo")
@pytest.mark.skipif(not SAMPLE_TIF.exists(),
                    reason="sample GeoTIFF not present")
def test_geosr_v2_geotiff_inference():
    from model.inference import run_inference

    class Args:
        pass
    args = Args()
    args.input = str(SAMPLE_TIF)
    args.output = str(_REPO / "model" / "outputs" / "geosr_v2_sr_out.tif")
    args.checkpoint = str(CHECKPOINT)
    args.scale = 2  # GeoSRv2 is 10 m -> 5 m only
    args.tile_size = 512
    args.reference = None
    args.metrics = None
    args.report = str(_REPO / "model" / "outputs" / "geosr_v2_report.json")
    args.uncertainty_method = "self_consistency"
    args.ensemble_n = 5
    args.noise_std = 0.01

    with __import__("rasterio").open(args.input) as src:
        args.gsd = float(abs(src.transform.a))

    import rasterio
    result = run_inference(args)
    assert result["scale_factor"] == 2
    assert result["output_gsd_meters"] == pytest.approx(5.0, abs=1e-6)

    # TEST 7, 8, 9 — GeoTIFF metadata
    with rasterio.open(args.output) as out, rasterio.open(args.input) as inp:
        # TEST 9 — dimensions doubled
        assert out.height == inp.height * 2
        assert out.width == inp.width * 2
        assert out.count == 4
        # TEST 7 — CRS preserved
        assert out.crs == inp.crs
        # TEST 8 — resolution halved, origin preserved
        assert abs(out.transform.a - inp.transform.a / 2) < 1e-6
        assert abs(out.transform.e - inp.transform.e / 2) < 1e-6
        assert abs(out.transform.c - inp.transform.c) < 1e-6
        assert abs(out.transform.f - inp.transform.f) < 1e-6

    # Reference unavailable -> metrics null, not fabricated
    import json
    with open(args.report) as f:
        report = json.load(f)
    assert report["reference_available"] is False or report.get("psnr") is None
