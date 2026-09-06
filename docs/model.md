# Model

> Status: ✅ **Trained checkpoint + inference implemented** (CPU). 🧪 Synthetic training data; extrapolation at inference. ⌛ GPU serving, 12-band, real-data finetuning planned.

## Architecture — GeoSR-ESRGAN

A residual-in-residual dense generator with an ESRGAN-style upscaling head, trained with
a compound loss. Implemented in `model/architectures/advanced_sr.py`
(`build_advanced` / `build_baseline`).

### Design choices

- **Global residual.** `use_global_residual=true`: the network learns a residual around
  the bicubically-upsampled input, which stabilizes reflectance preservation.
- **4 input/output channels** (B02/B03/B04/B08), multispectral end-to-end.
- **4× upscaling** realized with sub-pixel/shuffle-style upsampling; the operator is
  scale-agnostic per training, so the same weights apply to any 4× task.

## Training (Phase 6)

`model/training/train.py` with the config frozen into the checkpoint
(`checkpoint_best.pt` → `config.json`).

| Field | Value |
|---|---|
| Patch size | 64 |
| Scale | 4× |
| Epochs | 2 (seed 42) |
| Batch size | 8 |
| Optimizer | AdamW · lr 1e-4 · weight_decay 1e-5 · cosine schedule |
| Loss | GeoSR compound = L1(1.0) + Spectral(0.2) + SSIM(0.1) + Edge(0.05) + TV(0.0) |
| Gradient clip | 0.0 (disabled) |
| Mixed precision | disabled (`use_amp=false`) |

### Train log (`model/checkpoints/advanced_geosr_baseline_run/train.log`)

```
[epoch 000] train_loss=0.2654 (0.097/0.358/0.097) val_psnr=27.766 ssim=0.682 sam=10.784 ergas=7.044 BEST
[epoch 001] train_loss=0.2470 (0.089/0.333/0.092) val_psnr=27.923 ssim=0.690 sam=10.604 ergas=6.912 BEST
[done] best_val_psnr=27.9231
```

`best_val_psnr=27.92 dB` is the **40 m→10 m** training-distribution metric. The 10 m→2.5 m
numbers in [ Evaluation ](evaluation.md) and the demo are the held-out extrapolation regime.

## Inference (Phase 7)

- **Batch CLI** — `model/inference.py`: whole-image for small inputs, or a 512 px tiled
  sliding window (`DEFAULT_TILE = 512`) with Hann-window blending for large scenes.
  Writes a CRS/geotransform-preserving uint16 GeoTIFF, a confidence map, and a
  validation report JSON.
- **Backend service** — `backend/app/processing/geosr_backend.py` wraps the checkpoint as
  a numpy→numpy `model_fn` (the contract expected by `process_satellite_image`). Input is
  reflectance `[0, 1.5]`; output is clipped and returned for denormalization.

### Reproducibility

- Seed 42 is fixed for dataset generation and weight init.
- Inference is deterministic (`torch.no_grad`, `model.eval()`, CPU; no dropout at test
  time). Two consecutive runs produce byte-identical SR GeoTIFFs.
- CPU-only in this environment; results are reproducible on any machine with the same
  checkpoint.

## Memory & runtime (CPU)

Measured on the 256×256 sample (single forward pass, no tiling, GeoSRv2 ×2):

| Scene | Inference time | Output size |
|---|---|---|
| Sample (256×256) | ~1.4 s | 512×512 @ 5 m |

The advanced GeoSR-ESRGAN (×4) path produces 1024×1024 @ 2.5 m; the legacy ESGAN precompute
times (163–402 ms) are per-tile figures from that batched, scale-4 path (preserved as
reference; the live `/demo` uses the GeoSRv2 ×2 path above).

## Checkpoint availability

`model/checkpoints/advanced_geosr_baseline_run/checkpoint_best.pt` is present on disk but
**gitignored** (see `.gitignore` → `*.pt`, `model/checkpoints/*`). It must be provided
locally to run real-SR inference.

---

## GeoSRv2 (10 m → 5 m)

> Status: ✅ **Trained checkpoint + inference implemented, runtime-verified** (CPU).
> The checkpoint `model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt` is present locally
> and loads exactly (817,092 params; only `residual_scale` legitimately missing).
> 🧪 Synthetic training data; inference extrapolates 10 m → 5 m (scale factor ≠ 2 is
> rejected/coerced). ⌛ GPU serving, 12-band, real-data finetuning planned.

GeoSRv2 is a trained **residual-learning** super-resolution model for **10 m → 5 m**
Sentinel-2 imagery (scale factor **2**, 4 bands B02/B03/B04/B08). It is implemented
**exactly** to the supplied model card in `model/architectures/geosr_v2.py`:

- `head: Conv2d(4, 64, 3, p=1)` → 8 × `ResidualBlock(64)` (Conv→ReLU→Conv + identity) →
  `body_conv: Conv2d(64,64,3,1)` → feature connection `feat + body_conv(body(feat))` →
  `upsample_conv: Conv2d(64,256,3,1)` → `PixelShuffle(2)` → ReLU →
  `upsample_out: Conv2d(64,64,3,1)` → ReLU → `tail: Conv2d(64,4,3,1)` (raw residual).
- **Residual bounding:** `bounded_residual = tanh(raw_residual) * residual_scale`, where
  `residual_scale = [0.015, 0.020, 0.020, 0.050]` is a **registered buffer** (not learned).
- **No final clamp.** `output = bicubic(x, scale=2, align_corners=False) + bounded_residual`.

### Loader contract (`load_geosr_v2_checkpoint`)

- Loads via `torch.load(..., weights_only=False)` and `model.load_state_dict(strict=False)`.
- **Only `residual_scale`** (the buffer) is an allowed missing key.
- **Any missing learned parameter raises `RuntimeError`** — the model never silently falls
  back to random weights.
- Asserts parameter count == **817,092** (verified by `test_geosr_v2_parameter_count`).

### Verified (architecture, no checkpoint required)

| Check | Result |
|---|---|
| Param count | 817,092 ✅ (proves architecture matches the card) |
| TEST 1 shape `[1,4,128,128]→[1,4,256,256]` | ✅ 2× |
| TEST 4 no NaN/Inf | ✅ |
| TEST 5 `\|output - bicubic\| ≤ residual_scale` (per band) | ✅ |
| Loader fail-loud on missing Conv weight | ✅ raises |
| Loader allows missing `residual_scale` only | ✅ |

### Verified at runtime (with the checkpoint, scale 2)

With `model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt` present and
`GEOSR_CHECKPOINT` pointing at it, the following were exercised end-to-end on
`data/sample_sentinel2_10m.tif` (256×256 @ 10 m, EPSG:32643):

| Check | Result |
|---|---|
| TEST 3 strict load | ✅ epoch=34, params=817,092, missing=`['residual_scale']` only, unexpected=`[]` |
| TEST 6/7/8/9 inference | ✅ `[1,4,256,256]→[1,4,512,512]`, no NaN/Inf, output 5 m GSD, CRS/EPSG:32643 + geotransform preserved, 4-band uint16 |
| Backend e2e (`/upload→/process→/download`) | ✅ COMPLETED, `is_demo=false`, 512×512 @ 5 m, confidence map + validation report produced |
| TEST 10 numerical equivalence | ⛔ Not verifiable — no HR reference (or an independent recompute) was supplied against which to compare the checkpoint's output. Architecture match (817,092) gives high confidence; a reference recompute is required to close this gate. |

### How to run once the checkpoint is supplied

```bash
python model/inference.py \
  --input data/sample_sentinel2_10m.tif \
  --checkpoint model/checkpoints/geosr_v2/GeoSR_v2_epoch34_best.pt \
  --output model/outputs/geosr_v2_sr.tif \
  --scale 2 \
  --reference <optional 5 m HR reference> \
  --report model/outputs/geosr_v2_report.json
```

- **Scale must be 2.** `--scale 4` raises: `GeoSRv2 was trained for 10m -> 5m (scale
  factor 2). Scale factor 4 is not supported by this checkpoint.`
- Output: 4-band GeoTIFF, **5 m** GSD, same CRS/origin, uint16 DN.
- With an HR reference: PSNR/SSIM/SAM/MAE/RMSE are computed. **Without** a reference,
  reference-based metrics are `null` and the report states
  `"Reference-based quantitative validation unavailable for this scene."` — no metrics
  are fabricated.

### Backend integration

`geosr_backend.build_model_fn` detects GeoSRv2 from the checkpoint's `config.architecture`
field **or** the `model_name` field **or** the checkpoint filename (the earlier
implementation keyed only on `config.model_name`, which this checkpoint omits, causing
the trained weights to be silently loaded into the wrong "advanced" architecture —
**fixed**). The `GeoSRPipeline` orchestrator reads the checkpoint's native
`config.scale_factor` (2) and coerces the job's `scale_factor` to it (with a logged
warning if a 4× request is received), so tiling, reconstruction, and validation all
agree on the 2× output. The existing `advanced`/`baseline` (scale 4) paths remain
unchanged and are still used by the legacy ESGAN precompute (`data/demo/{urban,agriculture,water}/`), which is preserved on disk but no longer routed by `/demo`.

### Training metadata (from the checkpoint card, not retrained)

| Setting | Value |
|---|---|
| Epochs | 40 (best @ epoch 34) |
| Optimizer | Adam, lr 1e-4, weight_decay 1e-4 |
| Scale | 2 (10 m → 5 m) |
| Validation PSNR | 39.2962281363351 dB |
| Parameters | 817,092 |

The repository does **not** contain the training code for GeoSRv2 (it is a frozen
checkpoint); only inference/integration is performed here. The model is **not** retrained.
