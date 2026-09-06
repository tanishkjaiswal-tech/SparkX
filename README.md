# GeoSR — AI-Powered Satellite Super-Resolution Mapping

> **SIH 2026 Problem Statement 26142** — *"Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries"*
>
> GeoSR converts medium-resolution Sentinel-2 imagery (10 m GSD) into high-resolution geospatial products, demonstrated live at **5 m GSD** via GeoSRv2 (×2); a legacy GeoSR-ESRGAN (×4, 10 m→2.5 m) checkpoint is also preserved. Outputs preserve **spatial structure**, **spectral radiometry**, **geospatial integrity (CRS/transform)**, and surface **pixel-wise uncertainty**.

---

## Status at a Glance

| Area | Status | Evidence |
|---|---|---|
| Frontend (Next.js dashboard) | ✅ Implemented | Lint 0 errors · `tsc` clean · `next build` ✓ (8/8 routes prerendered) |
| Backend (FastAPI + geospatial) | ✅ Implemented | 34/34 pytest pass · e2e upload→process→download verified |
| Model (GeoSRv2, 10 m→5 m) | ✅ Trained checkpoint + inference | val PSNR **39.30 dB**, 817,092 params, epoch 34 (runtime-verified) |
| Model (GeoSR-ESRGAN, legacy 10 m→2.5 m, ×4) | ✅ Trained checkpoint (preserved) | val PSNR 27.92 / SSIM 0.690 / SAM 10.60° (2 epochs); legacy precompute, not served by `/demo` |
| SIH demo (`/demo`) | ✅ Implemented | Real GeoSRv2 (10 m→5 m) scene; null ref metrics, confidence ≈ 16.6% (see [Demo](#demo)) |

Legend: ✅ Implemented · 🧪 Experimental · ⌛ Planned

---

## Team SparkX & Responsibilities

- **Ankit Singh Tomar** — Backend & Geospatial Engine Architect
- **Tanishk** — Frontend & Interactive GIS Dashboard Lead
- **Anvay** — Research, Documentation & Project Presentation
- **Ashika, Ashta, Bhavana** — Deep Learning Architecture, SRM Training & Validation Pipelines

---

## Architecture & Project Structure

```
GeoSR/
├── frontend/             # Next.js 16 (React 19, TS, Tailwind, Leaflet, Recharts)
│   └── src/app/          # Routes: / (dashboard), /upload, /processing/[id],
│                         # /results/[id], /explorer, /demo, /demo/[id], /analytics
├── backend/              # FastAPI + Uvicorn geospatial service
│   └── app/
│       ├── api/routes/   # upload, jobs (status/results/metrics/download/preview/report), health
│       ├── processing/   # 8-stage pipeline, tiler, reconstruction, normalization,
│       │                 # preprocessing, raster_reader, geotiff_writer, geosr_backend
│       ├── services/     # job_manager (async, thread-safe, in-memory)
│       ├── schemas/      # pydantic models
│       └── utils/geo_utils.py
├── model/                # PyTorch GeoSR-ESRGAN (archs, datasets, losses, train, eval, inference)
├── data/                 # Sentinel-2 samples + Phase 9 demo scenes (see DATASET.md)
├── docs/                 # Architecture, dataset, model, evaluation, demo docs
└── docker/               # (planned) containerized deployment
```

**Request flow:** a user uploads a GeoTIFF → FastAPI validates it and opens a job → the **8-stage `GeoSRPipeline`** ([docs/architecture.md](./docs/architecture.md)) reads raster I/O via **rasterio**, selects the 4 Sentinel-2 10 m bands (B02/B03/B04/B08), normalizes to TOA reflectance, tiles with overlap, runs the **torch SR model** (loaded once per job, CPU), reassembles with Hann-window feathering, denormalizes, restores NoData, writes a CRS-preserving **GeoTIFF**, and emits a **confidence/uncertainty** map + validation report. The frontend polls `GET /api/jobs/{id}/status` and renders before/after + metrics.

---

## Tech Stack

- **Frontend:** Next.js 16.3 (Turbopack), React 19, TypeScript, Tailwind CSS v4, Lucide, Recharts, Leaflet
- **Backend:** FastAPI 0.109+, Pydantic v2, Uvicorn, rasterio 1.5, numpy, pyproj, affine
- **Model:** PyTorch 2.13 (CPU build used here), numpy 2.x
- **Data:** Sentinel-2 L2A style multispectral GeoTIFF (uint16 DN, ÷10000 = reflectance)

---

## Installation

### Frontend
```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000   # http://127.0.0.1:8000
```
To enable the **real SR model** (GeoSRv2, 10 m→5 m) instead of the deterministic bicubic baseline, point the env var at the trained checkpoint:
```bash
export GEOSR_CHECKPOINT=/path/to/GeoSR_v2_epoch34_best.pt
```
Then restart: the model is loaded once per job and `is_demo` becomes `false` in job metrics. The launch script `start_app.bat` already sets this to `model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt` automatically when the file is present. GeoSRv2 is scale-2 only; the orchestrator coerces a 4× request to 2× (10 m → 5 m) with a warning.

### Model (training / batch inference)
```bash
cd model
pip install -r requirements.txt
python model/training/train.py            # trains GeoSR-ESRGAN (config-driven)
python model/inference.py --input data/sample_sentinel2_10m.tif \
  --checkpoint model/checkpoints/advanced_geosr_baseline_run/checkpoint_best.pt \
  --output model/outputs/sr_out.tif --scale 4
```
> **GeoSRv2 (10 m → 5 m, scale 2)** — a separate, trained residual-learning model
> (`model/architectures/geosr_v2.py`, 817,092 params, val PSNR 39.30 dB). It is
> scale-2 only; `--scale 4` raises a clear error:
> ```bash
> python model/inference.py --input data/sample_sentinel2_10m.tif \
>   --checkpoint model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt \
>   --output model/outputs/geosr_v2_sr.tif --scale 2 \
>   --report model/outputs/geosr_v2_report.json
> ```
> The GeoSRv2 checkpoint is **not** in the repository (gitignored). Supply it locally
> to run real SR; without it, reference-based metrics are `null` (never fabricated).
> See [docs/model.md](./docs/model.md).

---

## Dataset

GeoSR operates on **4-band Sentinel-2 10 m imagery (B02 Blue, B03 Green, B04 Red, B08 NIR)**. Pixel values are uint16 DN scaled by 10000 → TOA reflectance in `[0, 1.5]`.

- **Training data:** synthetic 40 m→10 m paired samples generated by `model/datasets/degradation.py` (controlled bicubic downscale + noise). See [docs/dataset.md](./docs/dataset.md) and [`model/datasets/DATASET.md`](./model/datasets/DATASET.md).
- **Sample:** `data/sample_sentinel2_10m.tif` — 256×256, EPSG:32643, 10 m, uint16.
- **SIH demo scenes:** `data/demo/geosr_v2/` (live `/demo`, GeoSRv2) plus legacy ESGAN precompute in `data/demo/{urban,agriculture,water}/` and `frontend/public/samples/sih/` (preserved, not routed).

---

## Training

[docs/model.md](./docs/model.md) · [docs/evaluation.md](./docs/evaluation.md)

| Setting | Value |
|---|---|
| Architecture | GeoSR-ESRGAN (Residual-in-Residual Dense + ESRGAN-style upscaling) |
| Input bands | 4 (B02, B03, B04, B08) @ 10 m |
| Scale | 4× (10 m → 2.5 m target) |
| Patch size | 64 px |
| Epochs | 2 (seed 42) |
| Optimizer | AdamW, lr 1e-4, weight_decay 1e-5, cosine schedule |
| Loss | GeoSR compound = L1 (1.0) + spectral (0.2) + SSIM (0.1) + edge (0.05) |
| Checkpoint | `model/checkpoints/advanced_geosr_baseline_run/checkpoint_best.pt` |

Final (epoch 1, val split): **PSNR 27.923 dB · SSIM 0.690 · SAM 10.604° · ERGAS 6.912.**

---

## Inference

Two inference entry points:

1. **Batch CLI** — `model/inference.py` (whole-image or 512-px tiled sliding window with Hann blending). Reads DN uint16 GeoTIFF → reflectance → model → uint16 GeoTIFF with **preserved CRS, geotransform, band descriptions**; emits confidence map + validation report.
2. **Backend service** — `backend/app/processing/geosr_backend.py` wraps the checkpoint as a numpy→numpy `model_fn` consumed by the pipeline. **Loaded once per job**; never per-tile. Falls back to the deterministic bicubic baseline when no checkpoint or torch is unavailable (`is_demo=true`).

---

## API

Full spec at `/docs` (Swagger UI) when the backend runs. Key endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/upload` | Validate GeoTIFF + create job |
| `POST` | `/api/jobs/{id}/process` | Start 8-stage SR pipeline (async) |
| `GET`  | `/api/jobs/{id}/status` | Poll progress (0–100, stage, elapsed) |
| `GET`  | `/api/jobs/{id}` | Full job detail |
| `GET`  | `/api/jobs/{id}/results` | Preview + download + report URLs |
| `GET`  | `/api/jobs/{id}/metrics` | PSNR/SSIM/SAM/ERGAS + confidence |
| `GET`  | `/api/jobs/{id}/download` | Stream the super-resolved GeoTIFF |
| `GET`  | `/api/jobs/{id}/preview/{low_res\|super_res\|uncertainty}` | PNG thumbnails |
| `GET`  | `/api/jobs/{id}/validation-report` | Phase 7 unified report JSON |
| `DELETE` | `/api/jobs/{id}` | Cancel a queued/processing job |

Frontend API client: `frontend/src/lib/api.ts` (`fetchJobStatus`, `fetchJobResults`, `fetchValidationReport`).

---

## Demo (Phase 9 SIH Demonstration)

The live **SIH demo** is the `/demo` route. It serves a **single real Sentinel-2 scene**
super-resolved with the **GeoSRv2 (10 m → 5 m, ×2)** checkpoint
(`model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt` — 817,092 params, val PSNR 39.30 dB,
epoch 34). No 5 m ground-truth reference exists for this scene, so image-quality metrics are
reported as **N/A / `null`** (never fabricated): PSNR/SSIM/SAM are `null`, the validation
report states `"Reference-based quantitative validation unavailable for this scene."`, and
pixel-wise confidence is computed via **self-consistency** (round-trip degradation),
mean ≈ **16.6 / 100**.

This is exercised end-to-end by the live backend (`POST /api/upload → /process →
/download`): a 256×256, 4-band, uint16 input at 10 m (EPSG:32643) is produced as a
**512×512, 4-band, uint16 GeoTIFF at 5 m, EPSG:32643 preserved** (output CRS/geotransform
exact). Job metrics carry `is_demo=false` (real torch inference, not the bicubic baseline).
Precompute artifacts live under `data/demo/geosr_v2/` and `frontend/public/samples/geosr_v2/`;
the typed scene data is `frontend/src/data/geosrV2Demo.ts`. **No model runs in the browser**;
the frontend renders committed PNG previews + the scene data module.

Legacy scene ids (`/demo/urban`, `/demo/agriculture`, `/demo/water`) redirect to the picker —
the pre-Phase-9 synthetic 2.5 m GeoSR-ESRGAN precompute they referenced is preserved on disk
for reference but is **not** what `/demo` serves today. See [docs/demo.md](./docs/demo.md).

### Legacy precompute (preserved, not served by `/demo`)

Before Phase 9, `/demo` served synthetic 2.5 m GeoSR-ESRGAN scenes. Those artifacts remain
on disk (`data/demo/{urban,agriculture,water}/`, `frontend/public/samples/sih/`,
`frontend/src/data/sihDemoScenes.ts`) and were scored against a synthetic 2.5 m reference;
their real numbers (ESRGAN ×4, 2 epochs, geosr loss):

| Scene | Model | PSNR (dB) | SSIM | SAM (°) | ERGAS | Confidence | Inf. time |
|---|---|---|---|---|---|---|---|
| Urban | GeoSR-ESRGAN (2 ep, ×4) | 23.64 | 0.430 | 7.53 | 4.76 | 9.1% | 402 ms |
| Agriculture | GeoSR-ESRGAN (2 ep, ×4) | 25.20 | 0.512 | 7.61 | 5.26 | 7.8% | 163 ms |
| Water | GeoSR-ESRGAN (2 ep, ×4) | 28.15 | 0.709 | 15.29 | 7.54 | 6.3% | 196 ms |

```bash
# Regenerate the live /demo precompute (GeoSRv2, scale 2)
export GEOSR_CHECKPOINT=model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt
python model/inference.py --input data/sample_sentinel2_10m.tif \
  --checkpoint "$GEOSR_CHECKPOINT" --output model/outputs/geosr_v2_sr.tif --scale 2 \
  --report model/outputs/geosr_v2_report.json
```

---

## Evaluation

Quality gates and measured values. Reference-based metrics are `null` whenever no HR
reference is available for a scene (live uploads + the real `/demo` scene); confidence is
self-consistency-based. The GeoSRv2 live `/demo` matches the live `/upload` run:

| Metric | Meaning | Live `/upload` (GeoSRv2, ×2) | `/demo` scene (GeoSRv2, ×2, precomputed) |
|---|---|---|---|
| **PSNR** | Reconstruction fidelity (dB) | `null` | `null` |
| **SSIM** | Structural similarity | `null` | `null` |
| **SAM** | Spectral angle (°, ↓ better) | `null` | `null` |
| **ERGAS** | Global relative error (↓) | — | — |
| **Confidence** | Self-consistency (0–100) | 16.6 | 16.6 |
| **Output** | SR GeoTIFF | 512×512 @ 5 m, EPSG:32643 | 512×512 @ 5 m, EPSG:32643 |
| **is_demo** | Real SR vs baseline | `false` | n/a (precomputed) |

Verification script: `backend/tests/`, [docs/evaluation.md](./docs/evaluation.md).

---

## Testing & Verification

```bash
# Backend
python -m pytest -q            # 48 passed (42 backend + 6 model)

# Frontend
npm run lint                   # 0 errors
npx tsc --noEmit               # type-clean
npm run build                  # compiled successfully, 8/8 routes

# End-to-end (GeoSRv2, baseline=false): upload sample -> 512x512 GeoTIFF,
# 5 m resolution, EPSG:32643 preserved, confidence map + validation report returned.
```

---

## Limitations

- **Out-of-distribution scale.** GeoSRv2 is **trained on synthetic 40 m→10 m** pairs and applied to **10 m→5 m** (×2) — an extrapolation beyond the 10 m input signal, so the live demo reports `null` reference metrics with self-consistency confidence ≈ 16.6%. The legacy GeoSR-ESRGAN (×4) likewise extrapolates 10 m→2.5 m. This is the single most important caveat (see [docs/dataset.md](./docs/dataset.md) and [docs/model.md](./docs/model.md)).
- **Synthetic references.** Demo HR references are procedurally generated, not real satellite imagery.
- **4-band only.** Only B02/B03/B04/B08 are processed. Full 12-band Sentinel-2 is ⌛ planned.
- **No HR reference in live uploads.** Quantitative metrics are `null` unless a reference is supplied; confidence is then self-consistency-based.
- **CPU inference.** Tested on CPU builds; GPU serving is ⌛ planned (latency is CPU-bound here).
- **Gitignored binaries.** The trained checkpoint (`*.pt`, `model/checkpoints/*`) and the sample GeoTIFF (`*.tif`) are **not** in the repository. The committed, reproducible artifacts are the live `/demo` scene data (`frontend/src/data/geosrV2Demo.ts`), the web preview PNGs (`frontend/public/samples/geosr_v2/`), and the JSON reports under `data/demo/geosr_v2/`; legacy ESGAN precompute artifacts (`frontend/src/data/sihDemoScenes.ts`, `frontend/public/samples/sih/`, `data/demo/{urban,agriculture,water}/`) are likewise committed. To reproduce real-SR runs, provide the checkpoint + a Sentinel-2 GeoTIFF locally.

---

## Future Work (planned)

- 12-band full-spectrum Sentinel-2 SR (SWIR, visible, vegetation indices).
- GPU-accelerated serving + batch tiling for large scenes (> 256 px).
- Real Copernicus (SciHub / CDSE) raster ingestion + on-the-fly tiling.
- Input-perturbation ensemble uncertainty in the backend (currently only the CLI supports it).
- `GET /api/jobs` list endpoint + persistent (DB-backed) job store.
- GIS Explorer wired to real georeferenced rasters (currently demo presets).
