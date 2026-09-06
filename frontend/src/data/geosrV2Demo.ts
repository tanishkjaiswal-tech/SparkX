// GeoSRv2 demonstration data source (Phase 11 integration).
//
// Scientific contract:
//  * Single REAL Sentinel-2 input scene (10 m, 4-band B02/B03/B04/B08, EPSG:32643)
//    super-resolved with the trained GeoSRv2 checkpoint (scale factor 2 -> 5 m).
//  * The 5 m output is an ESTIMATE. There is no corresponding 5 m ground-truth
//    image for this real scene, so reference-based image-quality metrics are null.
//  * Held-out benchmark metrics (38.60 dB vs 36.79 dB bicubic) come from a
//    SEPARATE synthetic 5 m validation benchmark and are NOT claimed for this scene.
//
// The interface mirrors the shape consumed by the existing /demo components so no
// component API has to change; metric fields are nullable here (real scene).
export interface GeoSRv2DemoMetric {
  psnr: number | null;
  ssim: number | null;
  sam: number | null;
  ergas: number | null;
}

export interface GeoSRv2DemoScene {
  id: string;
  category: string;
  title: string;
  location: string;
  gsd_input_meters: number;
  gsd_output_meters: number;
  scale_factor: number;
  bands: string[];
  band_descriptions: string[];
  crs: string;
  scene_note: string;
  reference_note: string;
  model: string;
  params: number;
  metrics: GeoSRv2DemoMetric;
  confidence_score: number | null;
  uncertainty_summary: Record<string, unknown>;
  inference_time_ms: number;
  reference_available: boolean;
  band_comparison: Array<{
    band: string;
    name: string;
    wavelength_nm: number;
    original_reflectance: number;
    sr_reflectance: number;
    diff_percent: number;
  }>;
  web_previews: {
    lr_preview: string;
    sr_preview: string;
    confidence_preview: string;
  };
  artifacts: {
    lr_input: string;
    sr_output: string;
    confidence: string;
    report: string;
  };
}

export interface GeoSRv2Benchmark {
  label: string;
  psnr_db: number;
  bicubic_psnr_db: number;
  improvement_db: number;
  patches_better: number;
  patches_total: number;
  note: string;
}

// Held-out benchmark (synthetic 5 m reference benchmark). NOT the real scene.
export const GEOSR_V2_BENCHMARK: GeoSRv2Benchmark = {
  label: 'Held-out benchmark (synthetic 5 m validation)',
  psnr_db: 38.6028,
  bicubic_psnr_db: 36.7905,
  improvement_db: 1.8122,
  patches_better: 103,
  patches_total: 106,
  note: 'GeoSRv2 vs bicubic on held-out synthetic 10 m->5 m patches (817,092 params, epoch 34). Not comparable to the real Sentinel-2 scene below, which has no 5 m ground truth.',
};

export const GEOSSR_V2_DEMO: GeoSRv2DemoScene = {
  id: 'geosr_v2',
  category: 'Sentinel-2 (real)',
  title: 'Real Sentinel-2 Scene — 10 m → 5 m (GeoSRv2)',
  location: 'data/sample_sentinel2_10m.tif · B02/B03/B04/B08 · EPSG:32643',
  gsd_input_meters: 10,
  gsd_output_meters: 5,
  scale_factor: 2,
  bands: ['B02', 'B03', 'B04', 'B08'],
  band_descriptions: [
    'B02 (Blue, 10 m)',
    'B03 (Green, 10 m)',
    'B04 (Red, 10 m)',
    'B08 (NIR, 10 m)',
  ],
  crs: 'EPSG:32643 (WGS 84 / UTM Zone 43N)',
  scene_note:
    'Real Sentinel-2 L2A-style 4-band sample (256×256 @ 10 m). The 5 m output is the GeoSRv2 estimate; no 5 m ground-truth reference exists for this scene, so PSNR/SSIM/SAM are not reported.',
  reference_note:
    'Reference-based quantitative validation unavailable for this real scene (no corresponding 5 m ground-truth image was supplied).',
  model: 'GeoSRv2 (817,092 params, best epoch 34, scale factor 2)',
  params: 817092,
  metrics: { psnr: null, ssim: null, sam: null, ergas: null },
  confidence_score: 16.6,
  uncertainty_summary: {
    confidence_score: 16.605,
    confidence_mean: 0.166,
    confidence_std: 0.0051,
    max_uncertainty: 0.851,
    mean_uncertainty: 0.834,
    high_uncertainty_pixel_percent: 100.0,
  },
  inference_time_ms: 1437,
  reference_available: false,
  band_comparison: [
    { band: 'B02', name: 'Blue (490 nm)', wavelength_nm: 490, original_reflectance: 0.180, sr_reflectance: 0.168, diff_percent: -6.40 },
    { band: 'B03', name: 'Green (560 nm)', wavelength_nm: 560, original_reflectance: 0.220, sr_reflectance: 0.208, diff_percent: -5.49 },
    { band: 'B04', name: 'Red (665 nm)', wavelength_nm: 665, original_reflectance: 0.200, sr_reflectance: 0.188, diff_percent: -6.01 },
    { band: 'B08', name: 'NIR (842 nm)', wavelength_nm: 842, original_reflectance: 0.495, sr_reflectance: 0.487, diff_percent: -1.56 },
  ],
  web_previews: {
    lr_preview: '/samples/geosr_v2/lr_preview.png',
    sr_preview: '/samples/geosr_v2/sr_preview.png',
    confidence_preview: '/samples/geosr_v2/confidence_preview.png',
  },
  artifacts: {
    lr_input: 'data/demo/geosr_v2/lr_input.tif',
    sr_output: 'data/demo/geosr_v2/sr_output.tif',
    confidence: 'data/demo/geosr_v2/confidence_sr_output.tif',
    report: '/api/demo/report',
  },
};
