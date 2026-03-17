/**
 * Ambient floating stats — subtle overlays, not a control panel.
 */

import type { WorldSnapshot, GenerationResult } from '../../engine';

interface AmbientStatsProps {
  snapshot: WorldSnapshot | null;
  generation: number;
  totalGenerations: number;
  worldIndex: number;
  totalWorlds: number;
  bestScore: number;
  generations: GenerationResult[];
}

function Metric({
  label, value, unit = '', color = '#9ca3af', glow = false,
}: {
  label: string; value: string | number; unit?: string; color?: string; glow?: boolean;
}) {
  return (
    <div className="flex flex-col items-center">
      <span
        className="text-lg font-mono font-bold tabular-nums leading-none"
        style={{
          color,
          textShadow: glow ? `0 0 12px ${color}80` : 'none',
        }}
      >
        {typeof value === 'number' ? value.toLocaleString() : value}
        {unit && <span className="text-xs opacity-60 ml-0.5">{unit}</span>}
      </span>
      <span className="text-[9px] uppercase tracking-widest text-gray-600 mt-0.5">{label}</span>
    </div>
  );
}

export function AmbientStats({
  snapshot, generation, totalGenerations, worldIndex, totalWorlds, bestScore, generations,
}: AmbientStatsProps) {
  const pop = snapshot?.population ?? 0;
  const diversity = snapshot?.diversity ?? 0;
  const signals = snapshot?.signalActivity ?? 0;

  return (
    <div className="flex items-center justify-between w-full px-4 py-3">
      {/* Left cluster */}
      <div className="flex items-center gap-6">
        <Metric label="Generation" value={generation + 1} color="#06b6d4" glow />
        <div className="w-px h-8 bg-gray-800" />
        <Metric label="World" value={`${worldIndex}/${totalWorlds}`} color="#6b7280" />
        <div className="w-px h-8 bg-gray-800" />
        <Metric
          label="Population"
          value={pop}
          color={pop > 50 ? '#10b981' : pop > 10 ? '#f59e0b' : '#ef4444'}
          glow={pop > 50}
        />
      </div>

      {/* Center — tick progress */}
      <div className="flex flex-col items-center gap-1">
        <div className="flex items-center gap-2">
          <span className="text-[9px] uppercase tracking-widest text-gray-600">Tick</span>
          <span className="text-xs font-mono text-gray-400">{snapshot?.tick ?? 0}</span>
        </div>
        {/* Mini score sparkline for last few generations */}
        {generations.length > 0 && (
          <div className="flex items-end gap-0.5 h-4">
            {generations.slice(-12).map((g, i) => {
              const h = (g.bestScore / (bestScore || 1)) * 16;
              return (
                <div
                  key={i}
                  className="w-1 rounded-t"
                  style={{
                    height: `${Math.max(2, h)}px`,
                    background: i === generations.length - 1
                      ? 'rgba(6,182,212,0.8)'
                      : 'rgba(255,255,255,0.08)',
                  }}
                />
              );
            })}
          </div>
        )}
      </div>

      {/* Right cluster */}
      <div className="flex items-center gap-6">
        <Metric label="Diversity" value={diversity.toFixed(2)} color="#8b5cf6" />
        <div className="w-px h-8 bg-gray-800" />
        <Metric label="Signals" value={signals.toFixed(1)} color="#a78bfa" />
        <div className="w-px h-8 bg-gray-800" />
        <Metric
          label="Best Score"
          value={bestScore.toFixed(2)}
          color="#f59e0b"
          glow={bestScore > 3}
        />
      </div>
    </div>
  );
}
