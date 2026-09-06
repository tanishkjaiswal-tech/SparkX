'use client';

import React from 'react';
import Link from 'next/link';
import { GEOSSR_V2_DEMO, GEOSR_V2_BENCHMARK, GeoSRv2DemoScene } from '@/data/geosrV2Demo';
import { MetricCard } from '@/components/MetricCard';
import {
  Zap,
  ArrowRight,
  BarChart3,
  Clock,
  Sparkles,
  ShieldCheck,
} from 'lucide-react';

export default function SihDemoPickerPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16">
        <section className="mb-10">
          <div className="flex items-center gap-3 mb-3">
            <Zap className="w-5 h-5 text-cyan-400" />
            <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
              GeoSR Demo
            </h1>
          </div>
          <p className="text-base text-slate-300 max-w-3xl">
            GeoSRv2 super-resolves real 10 m Sentinel-2 imagery (B02/B03/B04/B08) to an
            estimated 5 m output (scale factor 2). The scene below has no corresponding 5 m
            ground-truth reference, so image-quality metrics are <span className="text-slate-400 font-mono">N/A</span> —
            confidence is computed via self-consistency. The held-out benchmark numbers use a
            separate synthetic 5 m validation set. No simulation is performed in the browser.
          </p>
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-12">
          <MetricCard
            label="Model"
            value="GeoSRv2"
            unit={`· ${GEOSSR_V2_DEMO.params.toLocaleString()} params`}
            description={`${GEOSSR_V2_DEMO.model}`}
            icon={<Sparkles className="w-5 h-5" />}
          />
          <MetricCard
            label="Held-out PSNR"
            value={GEOSR_V2_BENCHMARK.psnr_db.toFixed(2)}
            unit="dB"
            benchmark={`GeoSRv2 ${GEOSR_V2_BENCHMARK.psnr_db} / Bicubic ${GEOSR_V2_BENCHMARK.bicubic_psnr_db} (+${GEOSR_V2_BENCHMARK.improvement_db})`}
            description={`${GEOSR_V2_BENCHMARK.patches_better}/${GEOSR_V2_BENCHMARK.patches_total} patches better vs bicubic (synthetic 5 m benchmark).`}
            icon={<ShieldCheck className="w-5 h-5" />}
          />
          <MetricCard
            label="Inference"
            value={(GEOSSR_V2_DEMO.inference_time_ms / 1000).toFixed(2)}
            unit="s"
            benchmark="< 2 s"
            description="256x256 scene, single forward pass (CPU)."
            icon={<Clock className="w-5 h-5" />}
          />
        </section>

        <section>
          <h2 className="text-base font-bold uppercase tracking-wider text-slate-200 font-mono flex items-center gap-2 mb-5">
            <BarChart3 className="w-4 h-4 text-cyan-400" />
            GeoSRv2 Demonstration Scene
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {[GEOSSR_V2_DEMO].map((scene) => (
              <SceneCard key={scene.id} scene={scene} />
            ))}
          </div>
        </section>

        <section className="mt-14 p-5 rounded-2xl border border-slate-800 text-xs text-slate-400">
          <p>
            Input: {GEOSSR_V2_DEMO.location}. GeoSRv2 produces an estimated 5 m GeoTIFF
            (CRS/EPSG:32643 preserved, 4 bands, uint16). The 5 m estimate is not ground
            truth — 10 m→5 m is an extrapolation beyond the 10 m input signal. Review the{' '}
            <Link href="/demo/geosr_v2" className="text-cyan-400 hover:underline">
              demo scene
            </Link>{' '}
            or the{' '}
            <Link
              href="https://github.com/Kilo-Org/SparkX"
              className="text-cyan-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              methodology
            </Link>
            .
          </p>
        </section>
      </div>
    </div>
  );
}

function pluralize(count: number, unit: string) {
  return `${count} ${unit}${count === 1 ? '' : 's'}`;
}

function MetricRow({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-1.5">
      {icon}
      <span className="text-[10px] text-slate-500">{label}</span>
      <span className="text-xs font-mono font-semibold text-white">{value}</span>
    </div>
  );
}

function SceneCard({ scene }: { scene: GeoSRv2DemoScene }) {
  const { psnr, ssim, sam, ergas } = scene.metrics;
  return (
    <Link
      href={`/demo/${scene.id}`}
      className="group flex flex-col rounded-2xl bg-slate-900/60 border border-slate-800 hover:border-cyan-500/50 transition-all overflow-hidden"
    >
      <div className="relative aspect-[4/3] overflow-hidden">
        <img
          src={scene.web_previews.sr_preview}
          alt={`${scene.title} super-resolved preview`}
          className="absolute inset-0 h-full w-full object-cover transition-transform group-hover:scale-105"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent" />
        <div className="absolute top-3 left-3">
          <span className="inline-flex items-center text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-cyan-950/70 text-cyan-300 border border-cyan-800/60">
            {scene.category}
          </span>
        </div>
      </div>
      <div className="p-4 sm:p-5 space-y-3 flex-1 flex flex-col">
        <h3 className="text-lg font-semibold text-white group-hover:text-cyan-200 transition-colors line-clamp-1">
          {scene.title}
        </h3>
        <p className="text-xs text-slate-400 line-clamp-2 flex-1">
          {scene.location}
        </p>
        <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800">
          <MetricRow
            icon={<Sparkles className="w-3.5 h-3.5 text-slate-500" />}
            label="PSNR"
            value={psnr != null ? `${psnr.toFixed(2)} dB` : 'N/A'}
          />
          <MetricRow
            icon={<ShieldCheck className="w-3.5 h-3.5 text-slate-500" />}
            label="SSIM"
            value={ssim != null ? `${ssim.toFixed(3)}` : 'N/A'}
          />
          <MetricRow
            icon={<Zap className="w-3.5 h-3.5 text-slate-500" />}
            label="Scale"
            value={`${scene.scale_factor}× (${scene.gsd_input_meters} m → ${scene.gsd_output_meters} m)`}
          />
          {ergas != null && (
            <MetricRow
              icon={<BarChart3 className="w-3.5 h-3.5 text-slate-500" />}
              label="ERGAS"
              value={`${ergas.toFixed(2)}`}
            />
          )}
        </div>
        <div className="flex items-center justify-between text-xs font-mono text-slate-400 pt-2 border-t border-slate-800/60">
          <span>
            {pluralize(scene.bands.length, 'band')} · {scene.gsd_output_meters} m output
          </span>
          <ArrowRight className="w-3.5 h-3.5 text-cyan-400 group-hover:translate-x-1 transition-transform" />
        </div>
      </div>
    </Link>
  );
}
