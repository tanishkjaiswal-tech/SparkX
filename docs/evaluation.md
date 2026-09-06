# Evaluation

> Status: ✅ **Validation/uncertainty implemented & exercised.** 🧪 Quantitative metrics require an HR reference (only available for the synthetic demo precompute, not live uploads); uncertainty is self-consistency-based in the backend.

## Metrics

All reference-based metrics are computed in `model/evaluation/metrics.py` on
`[B,C,H,W]` reflectance tensors, averaged across the batch. `data_range = 1.5`.

| Metric | Definition | Lower / Upper |
|---|---|---|
| **PSNR** | `10·log10(data_range² / MSE)` over mean MSE across bands (dB) | ↑ |
| **SSIM** | `1 − SsimLoss` (Structural Similarity, 0–1) | ↑ |
| **SAM** | Mean spectral angle across bands, **degrees** | ↓ |
| **ERGAS** | Wald (2002): `(100/d)·√(mean_b(RMSE_b/µ_b)²)`, `d` = resolution ratio | ↓ |

For live uploads (no HR reference), the report returns these fields as `null` and states:
`Reference-based quantitative validation unavailable for this scene.`

## Uncertainty / confidence

`model/evaluation/uncertainty.py` produces a per-pixel confidence map in `[0, 1]` (1 =
confident) and a numeric summary. Two methods:

1. **Self-consistency (round-trip)** — the LR input is the only evidence the model has.
   The SR output is degraded back to LR with the *same* `degradation.degrade` operator
   used in training; where the round-trip fails to reproduce the input, the model
   invented unsupported structure → low confidence. Used by the **backend**
   (`geosr_backend.run_validation`).
2. **Input-perturbation ensemble** — run the model `n` times with small Gaussian noise on
   the LR tensor, measure per-pixel prediction spread. Used by the **CLI**
   (`run_inference --ensemble`).

### Honest framing

Confidence is a **reliability** indicator, **not** a per-pixel correctness label. It can
be optimistic on smooth regions, and a model can be confidently wrong. Without an HR
ground truth, "correctness" is never measured — only *self-consistency* and *sensitivity*.

## Measured results

### Live backend e2e (GeoSRv2 checkpoint, no HR reference)

A full `POST /api/upload → /process → poll → /download` run on
`data/sample_sentinel2_10m.tif` (256×256 @ 10 m, EPSG:32643). The backend
`GEOSR_CHECKPOINT` is set to `model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt`
(GeoSRv2, scale 2): the orchestrator coerces the job scale to the checkpoint's
native 2 and runs the real torch model (`is_demo=false`).

- Status: `COMPLETED`, `is_demo=false`, elapsed ~1.4 s (sample scene, CPU).
- Output GeoTIFF: **512×512, 4 bands, EPSG:32643, 5 m, uint16** — CRS/geotransform
  preserved (10 m → 5 m, exactly the model's trained scale).
- Validation report: `reference_available=false`, `reference_note` =
  `"Reference-based quantitative validation unavailable for this scene."`,
  PSNR/SSIM/SAM/ERGAS = `null` (no metrics fabricated).
- Self-consistency confidence ≈ **16.6 / 100** (spectral round-trip; B02/B03/B04/B08
  band means within ~1.5–6.4% of the LR input).

### Live `/demo` scene (GeoSRv2, Phase 9)

The **live SIH demo** is `/demo` — a single real Sentinel-2 scene super-resolved with
the GeoSRv2 (10 m→5 m, ×2) checkpoint above. It follows the same path as the live
backend e2e run (the frontend consumes the precomputed GeoTIFF + confidence map + report
committed under `data/demo/geosr_v2/` and `frontend/public/samples/geosr_v2/`). Because
no 5 m HR reference exists, PSNR/SSIM/SAM/ERGAS are `null` and confidence is
self-consistency-based (≈ 16.6%). No simulation runs in the browser.

### Legacy ESGAN precompute (preserved, not routed)

Before Phase 9, `/demo` served synthetic 2.5 m GeoSR-ESRGAN scenes. These artifacts
remain on disk for reference but are **no longer routed** (legacy ids redirect to the
picker). They were scored against a synthetic 2.5 m reference, using the advanced
GeoSR-ESRGAN checkpoint (`model/checkpoints/advanced_geosr_baseline_run/checkpoint_best.pt`,
scale 4, 10 m→2.5 m):

| Scene | Model | PSNR (dB) | SSIM | SAM (°) | ERGAS | Confidence | Inf. time |
|---|---|---|---|---|---|---|---|
| Urban | GeoSR-ESRGAN (2 ep, ×4) | 23.64 | 0.430 | 7.53 | 4.76 | 9.1% | 402 ms |
| Agriculture | GeoSR-ESRGAN (2 ep, ×4) | 25.20 | 0.512 | 7.61 | 5.26 | 7.8% | 163 ms |
| Water | GeoSR-ESRGAN (2 ep, ×4) | 28.15 | 0.709 | 15.29 | 7.54 | 6.3% | 196 ms |

### Training-distribution baseline (advanced checkpoint)

`best_val_psnr = 27.92 dB / SSIM 0.690 / SAM 10.60°` (epoch 1, val split, advanced
GeoSR-ESRGAN, 40 m → 10 m training distribution). The legacy precompute numbers above sit
**below** this because they are the out-of-distribution 10 m → 2.5 m extrapolation regime —
see [dataset.md](dataset.md). GeoSRv2 (scale 2) reports `val_psnr = 39.30 dB`
on its own training/validation split (see [model.md](model.md)).

## Verification

```bash
python -m pytest -q                       # 48 passed (42 backend + 6 model)
python model/inference.py --input data/sample_sentinel2_10m.tif \
  --checkpoint model/checkpoints/advanced_geosr_baseline_run/checkpoint_best.pt \
  --output /tmp/sr.tif --report /tmp/report.json   # produces confidence + report
```
