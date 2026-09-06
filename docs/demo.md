# Demo (Phase 9 SIH Demonstration)

> Status: ✅ **Implemented & live.** The `/demo` route serves a real Sentinel-2 scene
> super-resolved with the **GeoSRv2 (10 m → 5 m, ×2)** checkpoint. No HR reference exists
> for the scene, so reference metrics are `null` / **N/A** and confidence is
> self-consistency-based. 🧪 Scenes are real Sentinel-2 imagery (not procedurally
> generated); the 10 m→5 m task is an extrapolation beyond the 10 m input signal for the
> 40 m→10 m-trained model.

## What the demo is

The `/demo` route is the SIH walk-through demonstration. It serves **one real Sentinel-2
scene** (the committed sample at `data/sample_sentinel2_10m.tif`, 256×256 @ 10 m, EPSG:32643)
super-resolved end-to-end by the **GeoSRv2** checkpoint
(`model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt`, 817,092 params, val PSNR 39.30 dB):

1. Run the trained GeoSRv2 checkpoint (10 m→5 m, scale 2) on the 10 m input.
2. Emit **real** artifacts (no fabrication; reference metrics are `null`, not invented):

```
frontend/public/samples/geosr_v2/
  lr_preview.png        sr_preview.png      confidence_preview.png
frontend/src/data/geosrV2Demo.ts            # typed scene data + N/A metrics
data/demo/geosr_v2/
  sr_output.tif         confidence_sr_output.tif/.png
  validation_report.json                     # reference_available=false, null PSNR/SSIM/SAM
  lr_input.tif
```

All SR is **precomputed offline** by the backend pipeline (`GeoSRPipeline` with the GeoSRv2
checkpoint) and committed as static artifacts; the frontend renders the PNGs and scene data.
`is_demo=false` in job metrics (real torch inference, not the bicubic baseline).

| Scene | Model | Output | Confidence | Reference |
|---|---|---|---|---|
| Sentinel-2 sample | GeoSRv2 (×2, 10 m→5 m) | 512×512 @ 5 m, EPSG:32643, 4-band uint16 | ≈ 16.6% | N/A (null metrics) |

## What the demo is NOT

- It does **not** run torch or the model in the browser. All SR is precomputed.
- It does **not** fabricate metrics. PSNR/SSIM/SAM are reported as `null` (with
  `reference_available=false`) because no 5 m ground truth exists for the real scene.
- It does **not** claim to recover "true" 5 m detail — 10 m→5 m is out-of-training
  distribution for the 40 m→10 m-trained model. The (null) metrics and low confidence are
  an **honest measure of extrapolation**.

## Frontend

- `/demo` — scene picker with the GeoSRv2 demonstration scene.
- `/demo/geosr_v2` — results screen:
  - before/after **swipe** (10 m input / 5 m SR) and confidence overlay toggle,
  - metric cards showing `N/A` for PSNR/SSIM/SAM/SAM and the self-consistency confidence,
  - **How it works** / **Technical info** / **Spectral bands** panels,
  - PNG artifact downloads (SR preview, LR preview, confidence map).
- Legacy scene ids (`/demo/urban`, `/demo/agriculture`, `/demo/water`) redirect to the
  picker; the older synthetic 2.5 m scene pages are no longer routed.

## Regenerating the demo

```bash
export GEOSR_CHECKPOINT=model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt
python model/inference.py --input data/sample_sentinel2_10m.tif \
  --checkpoint "$GEOSR_CHECKPOINT" --output model/outputs/geosr_v2_sr.tif --scale 2 \
  --report model/outputs/geosr_v2_report.json
```
Requires the checkpoint and `data/sample_sentinel2_10m.tif` (both gitignored — see
[model.md](model.md)). Outputs are then copied into `data/demo/geosr_v2/` and
`frontend/public/samples/geosr_v2/` for the frontend.

## Legacy precompute (preserved, not routed)

Before Phase 9, `/demo` served three synthetic 2.5 m GeoSR-ESRGAN scenes. Those artifacts
remain on disk for reference (`data/demo/{urban,agriculture,water}/`,
`frontend/public/samples/sih/`, `frontend/src/data/sihDemoScenes.ts`) but are **not** served
by `/demo`. They were scored against a synthetic 2.5 m reference:

| Scene | Model | PSNR (dB) | SSIM | SAM (°) | ERGAS | Confidence | Inf. time |
|---|---|---|---|---|---|---|---|
| Urban | GeoSR-ESRGAN (2 ep, ×4) | 23.64 | 0.430 | 7.53 | 4.76 | 9.1% | 402 ms |
| Agriculture | GeoSR-ESRGAN (2 ep, ×4) | 25.20 | 0.512 | 7.61 | 5.26 | 7.8% | 163 ms |
| Water | GeoSR-ESRGAN (2 ep, ×4) | 28.15 | 0.709 | 15.29 | 7.54 | 6.3% | 196 ms |

## Relationship to the live uploader

| Feature | Live `/upload` flow | `/demo` scene (GeoSRv2) |
|---|---|---|
| Runs torch | ✅ yes (if checkpoint set) | ✅ once, offline |
| HR reference | ❌ none (real upload) | ❌ none (real scene) |
| Metrics | PSNR/SSIM/SAM = `null` | PSNR/SSIM/SAM = `null` |
| Confidence | Self-consistency ≈ 16.6% | Self-consistency ≈ 16.6% |
| Output | 512×512 @ 5 m, EPSG:32643 | 512×512 @ 5 m, EPSG:32643 |
| `is_demo` | `false` (real SR) | n/a (precomputed) |

## Known gaps (post-Phase 9)

- 🧪 GIS Explorer (`/explorer`) still renders demo `SCENE_PRESETS`, not real uploaded
  rasters — ⌛ planned wiring.
- ⌛ `GET /api/jobs` list endpoint + persistent (DB-backed) job store — ⌛ planned.
