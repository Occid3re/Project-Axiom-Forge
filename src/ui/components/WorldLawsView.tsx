/**
 * Displays the current world's laws as a compact visual.
 */

import type { WorldLaws } from '../../engine';

interface WorldLawsViewProps {
  laws: WorldLaws | null;
  title?: string;
}

const RESOURCE_DIST_NAMES = ['Uniform', 'Clustered', 'Gradient'];

export function WorldLawsView({ laws, title = 'World Laws' }: WorldLawsViewProps) {
  if (!laws) {
    return (
      <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg border border-gray-700/50 p-4">
        <h3 className="text-sm font-semibold text-gray-200 mb-2">{title}</h3>
        <p className="text-xs text-gray-500">No world selected</p>
      </div>
    );
  }

  const sections = [
    {
      name: 'Reproduction',
      params: [
        { label: 'Cost', value: laws.reproductionCost, max: 1 },
        { label: 'Offspring Energy', value: laws.offspringEnergy, max: 0.8 },
        { label: 'Mutation Rate', value: laws.mutationRate, max: 0.5 },
        { label: 'Mutation Strength', value: laws.mutationStrength, max: 0.3 },
      ],
    },
    {
      name: 'Energy',
      params: [
        { label: 'Regen Rate', value: laws.resourceRegenRate, max: 0.1 },
        { label: 'Eat Gain', value: laws.eatGain, max: 1 },
        { label: 'Move Cost', value: laws.moveCost, max: 0.1 },
        { label: 'Idle Cost', value: laws.idleCost, max: 0.05 },
        { label: 'Attack Transfer', value: laws.attackTransfer, max: 0.8 },
      ],
    },
    {
      name: 'Communication',
      params: [
        { label: 'Signal Range', value: laws.signalRange, max: 8 },
        { label: 'Channels', value: laws.signalChannels, max: 6 },
        { label: 'Signal Decay', value: laws.signalDecay, max: 1 },
      ],
    },
    {
      name: 'Memory & Perception',
      params: [
        { label: 'Memory Size', value: laws.memorySize, max: 16 },
        { label: 'Persistence', value: laws.memoryPersistence, max: 1 },
        { label: 'Perception Radius', value: laws.maxPerceptionRadius, max: 6 },
      ],
    },
  ];

  return (
    <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg border border-gray-700/50 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-200">{title}</h3>
        <div className="flex gap-2 text-[10px]">
          <span className={`px-1.5 py-0.5 rounded ${laws.sexualReproduction ? 'bg-purple-900/50 text-purple-300' : 'bg-gray-800 text-gray-500'}`}>
            {laws.sexualReproduction ? 'Sexual' : 'Asexual'}
          </span>
          <span className="px-1.5 py-0.5 rounded bg-gray-800 text-gray-400">
            {RESOURCE_DIST_NAMES[laws.resourceDistribution]}
          </span>
        </div>
      </div>

      <div className="space-y-3">
        {sections.map(section => (
          <div key={section.name}>
            <h4 className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">{section.name}</h4>
            <div className="space-y-1">
              {section.params.map(p => {
                const pct = Math.min(100, (p.value / p.max) * 100);
                return (
                  <div key={p.label} className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-400 w-24 shrink-0">{p.label}</span>
                    <div className="flex-1 bg-gray-800 rounded-full h-1">
                      <div
                        className="h-1 rounded-full"
                        style={{
                          width: `${pct}%`,
                          background: 'linear-gradient(90deg, #1e3a5f, #06b6d4)',
                        }}
                      />
                    </div>
                    <span className="text-[10px] text-gray-400 font-mono w-8 text-right">
                      {p.value < 1 ? p.value.toFixed(3) : p.value.toFixed(1)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
