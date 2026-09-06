"""Phase 11 — GeoSRv2 demo data contract tests.

These tests lock the invariant that the `/demo` experience is backed by the real
GeoSRv2 10 m -> 5 m pipeline (not the legacy Advanced GeoSR-ESRGAN 10 m -> 2.5 m
precompute). They verify the checkpoint identity, the real-input/real-output
artifacts, and that no fabricated metrics are exposed.
"""
from pathlib import Path

import json
import pytest
import rasterio
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]  # SparkX root
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CHECKPOINT = REPO / "model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt"
INPUT_TIF = REPO / "data/sample_sentinel2_10m.tif"
OUTPUT_TIF = REPO / "data/demo/geosr_v2/sr_output.tif"
REPORT_JSON = REPO / "data/demo/geosr_v2/validation_report.json"


@pytest.fixture(scope="module")
def demo_scene():
    return {
        "input": INPUT_TIF,
        "output": OUTPUT_TIF,
        "report": REPORT_JSON,
    }


# TEST 15: checkpoint identity + strict load (only residual_scale may be missing)
def test_geosr_v2_checkpoint_identity():
    assert CHECKPOINT.exists(), f"GeoSRv2 checkpoint missing: {CHECKPOINT}"
    from model.architectures.geosr_v2 import build_geosr_v2, load_geosr_v2_checkpoint

    model = build_geosr_v2(num_channels=4)
    info = load_geosr_v2_checkpoint(model, CHECKPOINT)
    assert info["epoch"] == 34
    assert info["parameter_count"] == 817092
    assert info["missing_keys"] == ["residual_scale"]
    assert info["unexpected_keys"] == []


# TEST 1: demo uses GeoSRv2 (checkpoint is the v2 epoch-34 file)
def test_demo_checkpoint_is_geosr_v2():
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    arch = (cfg.get("architecture") or "").lower()
    assert "geosrv2" in arch.replace("_", "").replace("-", "")


# TEST 2: input is 10 m Sentinel-2 4-band
def test_demo_input_is_10m(demo_scene):
    with rasterio.open(demo_scene["input"]) as src:
        assert src.count == 4
        assert src.crs.to_epsg() == 32643
        assert round(float(abs(src.transform.a)), 1) == 10.0


# TEST 3 / TEST 8: output is 5 m, 4 bands, EPSG:32643, finite
def test_demo_output_is_5m_geosr_v2(demo_scene):
    assert demo_scene["output"].exists()
    with rasterio.open(demo_scene["output"]) as src:
        assert src.count == 4                  # TEST 8: 4 bands
        assert src.crs.to_epsg() == 32643       # TEST 8: CRS
        assert round(float(abs(src.transform.a)), 1) == 5.0   # TEST 3 & TEST 8: 5 m
        data = src.read()
        assert np.isfinite(data).all()          # TEST 8: no NaN/Inf
        assert data.min() > 0                   # valid DN (nodata is 0)
    # scale factor 2 from input 10m -> output 5m
    with rasterio.open(demo_scene["input"]) as li, rasterio.open(demo_scene["output"]) as lo:
        assert round(float(abs(lo.transform.a)) / float(abs(li.transform.a)), 3) == 0.5  # TEST 4


# TEST 4: scale factor 2x
def test_demo_scale_is_2x(demo_scene):
    rep = json.loads(REPORT_JSON.read_text())
    assert rep.get("scale_factor") == 2


# TEST 5 / TEST 6: NOT 2.5 m and NOT 4x
def test_demo_does_not_claim_2_5m_or_4x(demo_scene):
    with rasterio.open(demo_scene["output"]) as src:
        gsd = round(float(abs(src.transform.a)), 3)
    assert gsd == 5.0
    assert gsd != 2.5
    rep = json.loads(REPORT_JSON.read_text())
    assert rep.get("scale_factor") != 4


# TEST 7: real scene does not expose fabricated PSNR/SSIM/SAM
def test_demo_no_fabricated_metrics(demo_scene):
    rep = json.loads(REPORT_JSON.read_text())
    assert rep.get("reference_available") is False
    for k in ("psnr", "ssim", "sam"):
        assert rep.get(k) is None


# TEST 8: output metadata — bands/CRS/5m (combined with finite check)
def test_demo_output_metadata(demo_scene):
    with rasterio.open(demo_scene["output"]) as src:
        assert src.count == 4
        assert src.crs.to_epsg() == 32643
        assert round(float(abs(src.transform.a)), 1) == 5.0
        assert src.dtypes[0] in ("uint16", "float32")
