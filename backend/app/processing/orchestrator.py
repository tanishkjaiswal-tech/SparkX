"""Phase 8 — GeoSR pipeline orchestration service.

Provides an explicit 8-stage pipeline abstraction over the existing geospatial
processing stack (`app.processing.pipeline.process_satellite_image`) and the
Phase 7 torch validation layer (`app.processing.geosr_backend`).

Design goals (Phase 8):
  * An explicit, ordered ``PipelineStage`` enum (INGEST .. EXPORT) with a stable
    integer ``stage`` index and a baseline ``progress`` value. Each stage emits a
    status update through a caller-supplied ``StatusCallback`` so the
    ``JobManager`` can publish progress to the frontend in real time.
  * The trained GeoSR model (when a checkpoint is configured) is loaded exactly
    once per run and reused across every tile batch — never per-tile and never
    per-job re-import.
  * Stages 1-6 (INGEST -> VALIDATE) are driven by ``process_satellite_image``'s
    internal progress callback, which is remapped into the 8-stage coordinate
    system. Stages 7 (UNCERTAINTY) and 8 (EXPORT) are executed by this service
    after the raster is reconstructed.
  * The service never hard-depends on torch: when no checkpoint is configured it
    transparently falls back to the deterministic bicubic baseline, preserving
    the demo-metrics contract used by the test suite.

The service is intentionally framework-agnostic: it is driven by ``run(job_id)``
semantics where ``job_id`` is supplied by the caller (the ``JobManager``) which
owns persistence of the ``JobRecord``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Tuple, Union

import numpy as np

from app.core.logging import logger
from app.processing.geosr_backend import build_model_fn, describe_checkpoint, run_validation
from app.processing.pipeline import (
    BaselineDeterministicUpsampler,
    BASELINE_MODEL_LABEL,
    ModelInferenceFn,
    PipelineConfig,
    PipelineResult,
)
from app.processing.pipeline import process_satellite_image
from app.schemas.job import ProcessJobRequest, SpectralPoint, ValidationMetrics
from app.utils.geo_utils import (
    generate_preview_png,
    generate_uncertainty_png,
)


Number = Union[int, float]


class PipelineStage(Enum):
    """Ordered execution stages for the GeoSR super-resolution pipeline.

    Each member carries:
      * ``stage``  — stable 1-based ordinal index.
      * ``progress`` — baseline progress percentage (1-100) reported when the
        stage *completes*. Smooth in-flight progress is interpolated around this.
      * ``description`` — human label surfaced to the frontend.
    """

    INGEST = (1, 12, "Data ingestion & GeoTIFF validation")
    PREPROCESS = (2, 22, "Preprocessing & reflectance normalization")
    TILE = (3, 28, "Window tiling")
    INFERENCE = (4, 62, "Model inference")
    RECONSTRUCT = (5, 76, "Tile reconstruction & denormalization")
    VALIDATE = (6, 90, "Geospatial alignment validation")
    UNCERTAINTY = (7, 95, "Confidence & uncertainty estimation")
    EXPORT = (8, 100, "Export & preview generation")

    def __init__(self, stage: int, progress: int, description: str):
        self._stage = stage
        self._progress = progress
        self.description = description

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def progress(self) -> int:
        return self._progress

    @classmethod
    def by_index(cls, index: int) -> "PipelineStage":
        for member in cls:
            if member.stage == index:
                return member
        raise ValueError(f"No PipelineStage with index {index}")


# Type alias for the callback the JobManager registers to publish progress.
StatusCallback = Callable[[int, str, Optional[PipelineStage]], None]


@dataclass
class PipelineRunResult:
    """Outcome of a single ``GeoSRPipeline.run`` execution."""

    pipeline_result: PipelineResult
    metrics: ValidationMetrics
    spectral_points: List[SpectralPoint]
    is_real_sr: bool
    model_name: str
    output_geotiff_path: Path
    preview_paths: dict = field(default_factory=dict)
    uncertainty_preview_path: Optional[Path] = None
    validation_report_path: Optional[Path] = None


# Segments map ``process_satellite_image``'s 0-100 progress into the
# (stage, global_progress) coordinate system of ``PipelineStage``. Each segment
# is monotonic non-decreasing so job progress never regresses.
_PROGRESS_SEGMENTS: List[Tuple[float, float, "PipelineStage", int, int]] = [
    (0.0, 15.0, PipelineStage.INGEST, 5, 12),
    (15.0, 26.0, PipelineStage.PREPROCESS, 12, 20),
    (26.0, 36.0, PipelineStage.PREPROCESS, 20, 24),
    (36.0, 46.0, PipelineStage.TILE, 24, 28),
    (46.0, 82.0, PipelineStage.INFERENCE, 28, 62),
    (82.0, 88.0, PipelineStage.RECONSTRUCT, 62, 76),
    (88.0, 100.0, PipelineStage.VALIDATE, 76, 90),
]


def _translate_progress(pct: int) -> Tuple[PipelineStage, int]:
    """Translate a ``process_satellite_image`` progress value (0-100) into the
    8-stage coordinate system used by ``PipelineStage``.

    Returns the (stage, progress) pair. Progress is clamped to [0, 100] and
    monotonic non-decreasing over the input domain.
    """
    pct = max(0, min(100, pct))
    for p_lo, p_hi, stage, prog_lo, prog_hi in _PROGRESS_SEGMENTS:
        if pct < p_hi:
            span = p_hi - p_lo
            frac = (pct - p_lo) / span if span > 0 else 0.0
            prog = prog_lo + frac * (prog_hi - prog_lo)
            return stage, max(1, min(100, int(round(prog))))
    return PipelineStage.VALIDATE, 90


class GeoSRPipeline:
    """Orchestrates the 8-stage super-resolution pipeline for a single job.

    The torch model (when a checkpoint path is supplied) is resolved to a
    ``model_fn`` callable exactly once during construction and is then reused
    for every tile batch inside ``process_satellite_image``.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        scale_factor: int = 4,
        params: Optional[ProcessJobRequest] = None,
        status_cb: Optional[StatusCallback] = None,
    ):
        self.scale_factor = scale_factor
        self.params = params or ProcessJobRequest(scale_factor=scale_factor)
        self.status_cb = status_cb

        # GeoSRv2 is trained for 10 m -> 5 m (scale 2) only. Coerce the job's scale
        # to the checkpoint's native scale before building the model so the tiling,
        # reconstruction, and validation stages all agree on the output resolution.
        if checkpoint_path:
            desc = describe_checkpoint(checkpoint_path)
            if desc.get("is_geosr_v2") and desc.get("native_scale") == 2 and scale_factor != 2:
                logger.warning(
                    f"GeoSRv2 checkpoint is trained for 10m->5m (scale 2); coercing "
                    f"requested scale_factor={scale_factor} -> 2. Output GSD will be "
                    f"5 m (the checkpoint's only supported scale)."
                )
                scale_factor = 2
                self.scale_factor = scale_factor
                self.params.scale_factor = scale_factor

        model_fn = build_model_fn(checkpoint_path, scale_factor=scale_factor)
        self.model_fn: Optional[ModelInferenceFn] = model_fn
        self.is_real_sr = model_fn is not None
        if self.is_real_sr:
            self.model_name = "GeoSR torch model"
        else:
            self.model_name = BASELINE_MODEL_LABEL
        logger.info(
            f"GeoSRPipeline initialized: real_sr={self.is_real_sr} "
            f"model={self.model_name}"
        )

    def _emit(self, progress: int, stage: Optional[PipelineStage] = None,
              desc: Optional[str] = None) -> None:
        """Push a status update to the registered callback, if any."""
        if self.status_cb is None:
            return
        label = desc if desc is not None else (stage.description if stage else "")
        self.status_cb(progress, label, stage)

    def _progress_bridge(self, pct: int, msg: str) -> None:
        """Adapter translating ``process_satellite_image`` progress (0-100) into
        the 8-stage coordinate system and forwarding via ``_emit``."""
        try:
            stage, progress = _translate_progress(pct)
        except Exception:
            stage, progress = PipelineStage.INFERENCE, max(1, min(100, pct))
        logger.info(f"[stage={stage.name} {progress}%] {msg}")
        self._emit(progress, stage=stage, desc=msg)

    def run(
        self,
        job_id: str,
        input_path: Union[str, Path],
        output_geotiff: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> PipelineRunResult:
        """Execute the full 8-stage pipeline for ``job_id``.

        ``job_id`` is supplied by the caller (JobManager) for log correlation and
        output-path derivation; this service does not persist its own job state.
        """
        in_file = Path(input_path)
        out_file = Path(output_geotiff)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        config = PipelineConfig(
            patch_size=self.params.tile_size,
            overlap=int(self.params.tile_size * (self.params.overlap_percent / 100.0)),
            scale_factor=self.params.scale_factor,
            target_dtype="uint16",
        )

        self._emit(6, stage=PipelineStage.INGEST,
                   desc=f"Ingesting GeoTIFF '{in_file.name}' for job {job_id}...")

        # Stages INGEST -> VALIDATE are executed inside ``process_satellite_image``,
        # whose fine-grained progress is remapped into the 8-stage system.
        pipeline_result = process_satellite_image(
            input_path=in_file,
            output_path=out_file,
            config=config,
            model_fn=self.model_fn,
            progress_cb=self._progress_bridge,
        )
        # process_satellite_image drives up to VALIDATE (~90%); emit terminal.
        self._emit(90, stage=PipelineStage.VALIDATE,
                   desc="Geospatial alignment validated; entering post-processing.")

        band_combo = self.params.band_combination.value
        low_res_png = out_dir / "preview_low_res.png"
        super_res_png = out_dir / "preview_super_res.png"
        uncertainty_png = out_dir / "preview_uncertainty.png"

        # Stage 8 (previews) prerequisites: render lightweight visualizations so
        # the uncertainty fallback (Stage 7) can sample the super-resolved image.
        generate_preview_png(in_file, low_res_png, band_combo=band_combo)
        generate_preview_png(out_file, super_res_png, band_combo=band_combo)

        # Stage 7 — UNCERTAINTY (real model computes a self-consistency
        # confidence GeoTIFF + report; baseline synthesises a deterministic
        # heatmap so the UI always has a preview).
        real_sr = self.is_real_sr
        if real_sr:
            self._emit(92, stage=PipelineStage.UNCERTAINTY,
                       desc="Computing self-consistency confidence map...")
            conf_png, _conf_tif, report = run_validation(
                in_file, out_file, self.scale_factor, out_dir
            )
            report = report if isinstance(report, dict) else {}
            confidence_png = conf_png or self._fallback_uncertainty(super_res_png, uncertainty_png)
            report_json = out_dir / f"validation_report_{out_file.stem}.json"
            validation_report_path = report_json if report_json.exists() else None
        else:
            self._emit(92, stage=PipelineStage.UNCERTAINTY,
                       desc="Generating fallback uncertainty heatmap (baseline)...")
            confidence_png = self._fallback_uncertainty(super_res_png, uncertainty_png)
            report = {}
            validation_report_path = None

        # Stage 8 — EXPORT (finalize).
        self._emit(96, stage=PipelineStage.EXPORT,
                   desc="Finalizing super-resolved assets...")
        if not confidence_png:
            confidence_png = self._fallback_uncertainty(super_res_png, uncertainty_png)

        self._emit(100, stage=PipelineStage.EXPORT,
                   desc="Super-resolution pipeline complete.")

        metrics, spectral_points = self._build_metrics(
            pipeline_result, report, real_sr, band_combo
        )

        return PipelineRunResult(
            pipeline_result=pipeline_result,
            metrics=metrics,
            spectral_points=spectral_points,
            is_real_sr=real_sr,
            model_name=self.model_name,
            output_geotiff_path=out_file,
            preview_paths={
                "low_res": low_res_png,
                "super_res": super_res_png,
                "uncertainty": confidence_png,
            },
            uncertainty_preview_path=confidence_png,
            validation_report_path=validation_report_path,
        )

    @staticmethod
    def _fallback_uncertainty(super_res_png: Path, uncertainty_png: Path) -> Path:
        try:
            generate_uncertainty_png(super_res_png, uncertainty_png)
        except Exception as exc:  # never let preview generation kill the job
            logger.warning(f"uncertainty preview fallback failed: {exc}")
        return uncertainty_png

    def _build_metrics(
        self,
        pipeline_result: PipelineResult,
        report: dict,
        real_sr: bool,
        band_combo: str,
    ) -> Tuple[ValidationMetrics, List[SpectralPoint]]:
        elapsed_ms = int(pipeline_result.elapsed_seconds * 1000)
        pixels_in = (
            pipeline_result.input_metadata.width * pipeline_result.input_metadata.height
        )
        pixels_out = (
            pipeline_result.output_metadata.width * pipeline_result.output_metadata.height
        )

        if real_sr:
            unc_summary: dict = report.get("uncertainty_summary", {}) or {}
            is_ref = bool(report.get("reference_available", False))
            band_wl = {"B02": 490, "B03": 560, "B04": 665, "B08": 842}
            band_name = {"B02": "Blue", "B03": "Green", "B04": "Red", "B08": "NIR"}
            band_cmp = report.get("spectral_validation", {}).get("band_comparison", []) or []
            spectral_points = [
                SpectralPoint(
                    band=bc["band"],
                    name=f"{band_name.get(bc['band'], bc['band'])} ({bc['band']})",
                    wavelength_nm=band_wl.get(bc["band"], 0),
                    original_reflectance=float(bc["lr_mean_reflectance"]),
                    sr_reflectance=float(bc["sr_mean_reflectance"]),
                    diff_percent=float(bc["diff_percent"]),
                )
                for bc in band_cmp
            ]
            metrics = ValidationMetrics(
                psnr=report.get("psnr"),
                ssim=report.get("ssim"),
                sam=report.get("sam"),
                ergas=report.get("ergas"),
                uiqi=None,
                spatial_correlation=None,
                inference_time_ms=elapsed_ms,
                pixel_count_original=pixels_in,
                pixel_count_super_resolved=pixels_out,
                is_demo=False,
                reference_available=is_ref,
                confidence_score=unc_summary.get("confidence_score"),
            )
        else:
            spectral_points = [
                SpectralPoint(band="B02", name="Blue (490nm)", wavelength_nm=490,
                              original_reflectance=0.142, sr_reflectance=0.141, diff_percent=-0.7),
                SpectralPoint(band="B03", name="Green (560nm)", wavelength_nm=560,
                              original_reflectance=0.168, sr_reflectance=0.169, diff_percent=0.6),
                SpectralPoint(band="B04", name="Red (665nm)", wavelength_nm=665,
                              original_reflectance=0.195, sr_reflectance=0.194, diff_percent=-0.5),
                SpectralPoint(band="B08", name="NIR (842nm)", wavelength_nm=842,
                              original_reflectance=0.312, sr_reflectance=0.315, diff_percent=0.9),
            ]
            scale = self.params.scale_factor
            metrics = ValidationMetrics(
                psnr=35.80 if scale == 2 else 34.42,
                ssim=0.941 if scale == 2 else 0.925,
                sam=1.92 if scale == 2 else 2.18,
                ergas=1.65,
                uiqi=0.952,
                spatial_correlation=0.970,
                inference_time_ms=elapsed_ms,
                pixel_count_original=pixels_in,
                pixel_count_super_resolved=pixels_out,
                is_demo=True,
            )
        return metrics, spectral_points
