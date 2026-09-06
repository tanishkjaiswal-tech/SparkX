from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.core.config import Settings
from app.api.dependencies import get_app_settings

router = APIRouter()


@router.get(
    "/report",
    summary="GeoSRv2 demo validation report (JSON)",
    description="Serves the Phase 7 unified validation report for the GeoSRv2 demo scene "
                "(real Sentinel-2 10 m -> 5 m). Reference-based image-quality metrics are "
                "null for this real scene (no 5 m ground truth).",
)
def get_demo_report(settings: Settings = Depends(get_app_settings)):
    report_path = settings.repo_root / "data" / "demo" / "geosr_v2" / "validation_report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            description="GeoSRv2 demo validation report not found. "
                        "Run: python model/inference.py --input data/sample_sentinel2_10m.tif "
                        "--checkpoint model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt "
                        "--output data/demo/geosr_v2/sr_output.tif --scale 2 "
                        "--report data/demo/geosr_v2/validation_report.json",
        )
    import json
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return JSONResponse(content=report)
