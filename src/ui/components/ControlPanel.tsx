/**
 * Control panel for configuring and running the simulation.
 */

import { useState } from 'react';
import type { MetaConfig } from '../../engine';
import { DEFAULT_META_CONFIG } from '../../engine';

interface ControlPanelProps {
  onStart: (config: MetaConfig) => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  running: boolean;
  paused: boolean;
}

export function ControlPanel({ onStart, onPause, onResume, onStop, running, paused }: ControlPanelProps) {
  const [config, setConfig] = useState<MetaConfig>({ ...DEFAULT_META_CONFIG });

  const updateConfig = (key: keyof MetaConfig, value: number) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg border border-gray-700/50 p-4">
      <h3 className="text-sm font-semibold text-gray-200 mb-3">Controls</h3>

      <div className="space-y-2 mb-4">
        <ConfigSlider label="Seed" value={config.seed} min={0} max={9999}
          onChange={v => updateConfig('seed', v)} disabled={running} />
        <ConfigSlider label="Meta Generations" value={config.metaGenerations} min={1} max={50}
          onChange={v => updateConfig('metaGenerations', v)} disabled={running} />
        <ConfigSlider label="Worlds / Gen" value={config.worldsPerGeneration} min={4} max={32} step={4}
          onChange={v => updateConfig('worldsPerGeneration', v)} disabled={running} />
        <ConfigSlider label="World Steps" value={config.worldSteps} min={100} max={2000} step={100}
          onChange={v => updateConfig('worldSteps', v)} disabled={running} />
        <ConfigSlider label="Grid Size" value={config.gridSize} min={16} max={128} step={16}
          onChange={v => updateConfig('gridSize', v)} disabled={running} />
        <ConfigSlider label="Initial Entities" value={config.initialEntities} min={10} max={200} step={10}
          onChange={v => updateConfig('initialEntities', v)} disabled={running} />
        <ConfigSlider label="Top K Selection" value={config.topK} min={2} max={8}
          onChange={v => updateConfig('topK', v)} disabled={running} />
      </div>

      <div className="flex gap-2">
        {!running ? (
          <button
            onClick={() => onStart(config)}
            className="flex-1 px-3 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium rounded-lg transition-colors"
          >
            Start Evolution
          </button>
        ) : (
          <>
            <button
              onClick={paused ? onResume : onPause}
              className="flex-1 px-3 py-2 bg-amber-600 hover:bg-amber-500 text-white text-sm font-medium rounded-lg transition-colors"
            >
              {paused ? 'Resume' : 'Pause'}
            </button>
            <button
              onClick={onStop}
              className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-500 text-white text-sm font-medium rounded-lg transition-colors"
            >
              Stop
            </button>
          </>
        )}
      </div>
    </div>
  );
}

function ConfigSlider({
  label, value, min, max, step = 1, onChange, disabled,
}: {
  label: string; value: number; min: number; max: number;
  step?: number; onChange: (v: number) => void; disabled: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-xs text-gray-400 w-28 shrink-0">{label}</label>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        disabled={disabled}
        className="flex-1 h-1 accent-cyan-500 disabled:opacity-40"
      />
      <span className="text-xs text-gray-300 font-mono w-10 text-right">{value}</span>
    </div>
  );
}
