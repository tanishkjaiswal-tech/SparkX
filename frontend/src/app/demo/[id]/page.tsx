'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useParams, redirect } from 'next/navigation';
import { GEOSSR_V2_DEMO, GEOSR_V2_BENCHMARK, GeoSRv2DemoScene } from '@/data/geosrV2Demo';
import { ImageComparison } from '@/components/ImageComparison';
import { MetricCard } from '@/components/MetricCard';
import { StatusBadge } from '@/components/StatusBadge';
import { BandCombination } from '@/types/geosr';
import {
  Zap,
  ArrowLeft,
  BarChart3,
  Compass,
  Clock,
  ShieldCheck,
  Download,
  Info,
  Palette,
  Globe,
  Layers,
  FileText,
  ExternalLink,
} from 'lucide-react';

const SCALE_FACTOR = 2;

// Legacy demo scene ids (Advanced GeoSR-ESRGAN, 10 m -> 2.5 m). They are preserved
// on disk but are NOT served by the new GeoSRv2 demo; redirect them to the picker
// instead of fabricating per-scene content.
const LEGACY_DEMO_IDS = new Set(['urban', 'agriculture', 'water']);

export default function SihDemoResultsPage() {
  const params = useParams();
  const sceneId = params?.id as string | undefined;

  // The canonical GeoSRv2 demo scene. Legacy/foreign ids redirect to the picker
  // so existing demo links keep working without inventing data.
  if (!sceneId || sceneId !== GEOSSR_V2_DEMO.id || LEGACY_DEMO_IDS.has(sceneId)) {
    redirect('/demo');
  }

  const scene = GEOSSR_V2_DEMO;

  const [bandCombo, setBandCombo] = useState<BandCombination>('RGB');
  const [openSection, setOpenSection] = useState<'works' | 'tech' | 'spectral'>('works');

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <BackLink />

        <Header scene={scene} />

        <MetricsRow scene={scene} />

        <Disclaimers />

        <section className="mb-8">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 font-mono mb-3 flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            Before / After Super-Resolution
          </h2>
          <p className="text-xs text-slate-400 mb-3 max-w-3xl">
            Left: 10 m Sentinel-2 input (B04/B03/B02 = Red/Green/Blue). Right: GeoSRv2
            2× estimate (5 m). Drag the slider to compare. The confidence overlay shows
            pixel-wise self-consistency (0 = extrapolated detail, 1 = grounded in the 10 m
            input).
          </p>
          <ImageComparison
            lowResImageUrl={scene.web_previews.lr_preview}
            superResImageUrl={scene.web_previews.sr_preview}
            uncertaintyMapUrl={scene.web_previews.confidence_preview}
            title={`${scene.title} — 10 m → 5 m`}
            coordinates={[28.6139, 77.209]}
            scaleFactor={SCALE_FACTOR}
            initialMode="swipe"
            bandCombination={bandCombo}
            onBandCombinationChange={setBandCombo}
            className="h-[620px]"
          />
        </section>

        <Accordion
          openSection={openSection}
          setOpenSection={setOpenSection}
          scene={scene}
        />

        <Downloads scene={scene} />
      </div>
    </div>
  );
}

function BackLink() {
  return (
    <Link
      href="/demo"
      className="inline-flex items-center gap-1.5 text-xs font-mono text-slate-400 hover:text-cyan-300 transition-colors mb-6"
    >
      <ArrowLeft className="w-3.5 h-3.5" />
      Back to GeoSR Demo
    </Link>
  );
}

function Header({ scene }: { scene: GeoSRv2DemoScene }) {
  return (
    <section className="mb-8">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          {scene.title}
        </h1>
        <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-cyan-950/70 text-cyan-300 border border-cyan-800/60">
          {scene.category}
        </span>
        <StatusBadge status="completed" showIcon={true} />
      </div>
      <p className="mt-2 text-sm text-slate-300 max-w-3xl">
        {scene.location}
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs font-mono text-slate-400">
        <span className="flex items-center gap-1">
          <Globe className="w-3.5 h-3.5" /> CRS: {scene.crs}
        </span>
        <span className="flex items-center gap-1">
          <Palette className="w-3.5 h-3.5" /> {scene.bands.join(' · ')}
        </span>
        <span className="flex items-center gap-1">
          <Layers className="w-3.5 h-3.5" /> Scale {scene.scale_factor}× (10 m → {scene.gsd_output_meters} m)
        </span>
      </div>
    </section>
  );
}

function Disclaimers() {
  return (
    <section className="mb-8 p-4 rounded-xl border border-slate-800 bg-slate-900/40 text-xs text-slate-300">
      <p className="flex items-start gap-2">
        <Info className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
        <span>
          GeoSRv2 ({GEOSSR_V2_DEMO.params.toLocaleString()} params, epoch 34) generates a
          higher-resolution 5 m <strong>estimate</strong> from 10 m Sentinel-2 imagery. This
          real scene has no corresponding 5 m ground-truth reference, so PSNR/SSIM/SAM are
          not reported. Confidence is derived from spectral self-consistency (LR↔SR
          round-trip) and is a reliability indicator — not a correctness label.
        </span>
      </p>
    </section>
  );
}

function MetricsRow({ scene }: { scene: GeoSRv2DemoScene }) {
  const { psnr, ssim, sam, ergas } = scene.metrics;
  const na = (v: number | null) => (v != null ? v : 'N/A');
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
      <MetricCard
        label="Peak SNR"
        value={na(psnr) === 'N/A' ? 'N/A' : (psnr as number).toFixed(2)}
        unit={psnr != null ? 'dB' : undefined}
        benchmark="No 5 m reference"
        description="Not reported for this real scene (no HR ground truth)."
        icon={<Zap className="w-5 h-5" />}
        isDemo={false}
      />
      <MetricCard
        label="Structural Similarity"
        value={na(ssim) === 'N/A' ? 'N/A' : (ssim as number).toFixed(3)}
        unit={ssim != null ? '' : undefined}
        benchmark="No 5 m reference"
        description="Not reported for this real scene (no HR ground truth)."
        icon={<ShieldCheck className="w-5 h-5" />}
        isDemo={false}
      />
      <MetricCard
        label="Spectral Angle"
        value={na(sam) === 'N/A' ? 'N/A' : (sam as number).toFixed(2)}
        unit={sam != null ? '°' : undefined}
        benchmark="No 5 m reference"
        description="Not reported for this real scene (no HR ground truth)."
        icon={<Compass className="w-5 h-5" />}
        isDemo={false}
      />
      <MetricCard
        label="Confidence"
        value={scene.confidence_score != null ? scene.confidence_score.toFixed(1) : 'N/A'}
        unit={scene.confidence_score != null ? '/100' : undefined}
        benchmark="Self-consistency"
        description="Pixel-wise LR↔SR round-trip (1 = grounded in the 10 m input)."
        icon={<Info className="w-5 h-5" />}
        isDemo={false}
      />
      <MetricCard
        label="Inference Time"
        value={scene.inference_time_ms.toFixed(1)}
        unit="ms"
        benchmark="< 2 s"
        description="256x256 scene, single forward pass (CPU, GeoSRv2)."
        icon={<Clock className="w-5 h-5" />}
        isDemo={false}
      />
      <MetricCard
        label="ERGAS"
        value={ergas != null ? ergas.toFixed(2) : 'N/A'}
        unit={ergas != null ? '' : undefined}
        benchmark="No 5 m reference"
        description="Not reported for this real scene (no HR ground truth)."
        icon={<BarChart3 className="w-5 h-5" />}
        isDemo={false}
      />
      <MetricCard
        label="Held-out Benchmark PSNR"
        value={GEOSR_V2_BENCHMARK.psnr_db.toFixed(2)}
        unit="dB"
        benchmark={`GeoSRv2 +${(GEOSR_V2_BENCHMARK.psnr_db - GEOSR_V2_BENCHMARK.bicubic_psnr_db).toFixed(2)} vs bicubic · ${Math.round((GEOSR_V2_BENCHMARK.patches_better / GEOSR_V2_BENCHMARK.patches_total) * 100)}% patches`}
        description="Synthetic 5 m validation benchmark — separate from the real scene above."
        icon={<ShieldCheck className="w-5 h-5" />}
        isDemo={false}
      />
    </section>
  );
}

function Accordion({
  openSection,
  setOpenSection,
  scene,
}: {
  openSection: 'works' | 'tech' | 'spectral';
  setOpenSection: (v: 'works' | 'tech' | 'spectral') => void;
  scene: GeoSRv2DemoScene;
}) {
  const items: { id: 'works' | 'tech' | 'spectral'; label: string; icon: React.ReactNode }[] = [
    { id: 'works', label: 'How it works', icon: <Zap className="w-4 h-4" /> },
    { id: 'tech', label: 'Technical info', icon: <FileText className="w-4 h-4" /> },
    { id: 'spectral', label: 'Spectral bands', icon: <Layers className="w-4 h-4" /> },
  ];

  return (
    <section className="mb-12 space-y-3">
      <div className="flex items-center gap-2 border-b border-slate-800">
        {items.map((it) => {
          const active = openSection === it.id;
          return (
            <button
              key={it.id}
              onClick={() => setOpenSection(it.id)}
              className={
                'flex items-center gap-1.5 px-3 py-2 text-sm font-mono transition-colors ' +
                (active
                  ? 'text-cyan-300 border-b-2 border-cyan-500'
                  : 'text-slate-400 hover:text-slate-200')
              }
            >
              {it.icon}
              {it.label}
            </button>
          );
        })}
      </div>

      <div className="p-4 sm:p-6 rounded-2xl bg-slate-900/50 border border-slate-800">
        {openSection === 'works' && <HowItWorks scene={scene} />}
        {openSection === 'tech' && <TechnicalInfo scene={scene} />}
        {openSection === 'spectral' && <SpectralInfo scene={scene} />}
      </div>
    </section>
  );
}

function HowItWorks({ scene }: { scene: GeoSRv2DemoScene }) {
  const steps = [
    'A real Sentinel-2 L2A-style 4-band image (B02/B03/B04/B08) at 10 m resolution is loaded (EPSG:32643).',
    `DN values are scaled by 1/10000 to TOA reflectance in [0, 1.5], matching the model input contract.`,
    `GeoSRv2 (817,092 params, best epoch 34) applies a 2x residual super-resolution step: Y_hat = B(X) + alpha * tanh(R(X)).`,
    `The reflectance output is denormalized back to uint16 DN and written as a 5 m GeoTIFF (CRS/geotransform preserved).`,
    'Pixel-wise confidence is computed via spectral self-consistency (LR↔SR round-trip) — no HR reference exists for this real scene, so PSNR/SSIM/SAM are N/A.',
  ];

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider">
        From 10 m input to 5 m estimate
      </h3>
      <ol className="space-y-2 text-sm text-slate-300 list-decimal list-inside">
        {steps.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ol>
      <p className="text-xs text-slate-400 pt-2">
        <span className="text-slate-300 font-mono">Note:</span> The 5 m output is an
        <strong> estimate</strong>, not ground truth. There is no 5 m reference for this
        real scene, so no reference-based quality metrics are reported. The held-out
        benchmark PSNR ({GEOSR_V2_BENCHMARK.psnr_db.toFixed(2)} dB vs bicubic{' '}
        {GEOSR_V2_BENCHMARK.bicubic_psnr_db.toFixed(2)} dB) is measured on a separate
        synthetic validation set and is not a per-scene number.
      </p>
    </div>
  );
}

function TechnicalInfo({ scene }: { scene: GeoSRv2DemoScene }) {
  const rows: { label: string; value: string }[] = [
    { label: 'Scene ID', value: scene.id },
    { label: 'Category', value: scene.category },
    { label: 'Model', value: scene.model },
    { label: 'Parameters', value: scene.params.toLocaleString() },
    { label: 'Input GSD', value: `${scene.gsd_input_meters} m` },
    { label: 'Output GSD', value: `${scene.gsd_output_meters} m` },
    { label: 'Scale factor', value: `${scene.scale_factor}×` },
    { label: 'Bands', value: scene.bands.join(' / ') },
    { label: 'CRS', value: scene.crs },
    { label: 'Reference available', value: scene.reference_available ? 'Yes' : 'No' },
    { label: 'Inference time', value: `${scene.inference_time_ms.toFixed(1)} ms` },
    { label: 'Confidence score', value: `${scene.confidence_score ?? 'N/A'} / 100` },
  ];

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider">
        Run specification
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2 text-sm">
        {rows.map((r) => (
          <div key={r.label} className="flex items-start justify-between gap-4">
            <span className="text-slate-400">{r.label}</span>
            <span className="text-slate-200 font-mono text-right">{r.value}</span>
          </div>
        ))}
      </div>
      <div className="pt-3 border-t border-slate-800 text-xs text-slate-400">
        {scene.scene_note}
      </div>
    </div>
  );
}

function SpectralInfo({ scene }: { scene: GeoSRv2DemoScene }) {
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider">
        Spectral bands (4-band super-resolution)
      </h3>
      <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
        {scene.band_descriptions.map((b) => (
          <li
            key={b}
            className="flex items-center gap-2 text-slate-300"
          >
            <Palette className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-mono">{b}</span>
          </li>
        ))}
      </ul>
      <p className="text-xs text-slate-400 pt-2">
        {scene.reference_note}
      </p>
    </div>
  );
}

function Downloads({ scene }: { scene: GeoSRv2DemoScene }) {
  const artifacts = [
    {
      label: 'Super-resolved preview (PNG)',
      href: scene.web_previews.sr_preview,
    },
    {
      label: '10 m input preview (PNG)',
      href: scene.web_previews.lr_preview,
    },
    {
      label: 'Confidence map (PNG)',
      href: scene.web_previews.confidence_preview,
    },
  ];

  return (
    <section className="p-5 sm:p-6 rounded-2xl bg-slate-900/40 border border-slate-800">
      <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider mb-4 flex items-center gap-2">
        <Download className="w-4 h-4 text-cyan-400" />
        Download demonstration artifacts
      </h3>
      <p className="text-xs text-slate-400 mb-4">
        Web previews are static PNG files generated from the GeoSRv2 5 m output (B04/B03/B02
        = Red/Green/Blue). The full-resolution GeoTIFFs are stored under{' '}
        <span className="text-slate-300">data/demo/geosr_v2/</span>.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {artifacts.map((a) => (
          <a
            key={a.label}
            href={a.href}
            download
            className="inline-flex items-center justify-between gap-2 px-3 py-2.5 rounded-xl bg-slate-950/80 hover:bg-cyan-950/40 border border-slate-800 hover:border-cyan-500/50 transition-all text-sm"
          >
            <span className="text-slate-200 truncate">{a.label}</span>
            <Download className="w-4 h-4 text-cyan-400 shrink-0" />
          </a>
        ))}
        <a
          href={scene.artifacts.report}
          className="inline-flex items-center justify-between gap-2 px-3 py-2.5 rounded-xl bg-slate-950/80 hover:bg-cyan-950/40 border border-slate-800 hover:border-cyan-500/50 transition-all text-sm"
        >
          <span className="text-slate-200 truncate">Validation report (JSON)</span>
          <FileText className="w-4 h-4 text-cyan-400 shrink-0" />
        </a>
        <a
          href="https://github.com/Kilo-Org/SparkX"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center justify-between gap-2 px-3 py-2.5 rounded-xl bg-slate-950/80 hover:bg-cyan-950/40 border border-slate-800 hover:border-cyan-500/50 transition-all text-sm"
        >
          <span className="text-slate-200 truncate">Source &middot; GeoSR</span>
          <ExternalLink className="w-4 h-4 text-cyan-400 shrink-0" />
        </a>
      </div>
    </section>
  );
}
