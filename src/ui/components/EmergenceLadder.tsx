/**
 * The Emergence Ladder — tracks observable milestones in the simulation.
 * Each stage maps to a concrete, measurable phenomenon the viewer can see.
 * Stages light up based on real-time score data from the display world.
 */

import type { EmergenceState } from './emergence';

const STAGES = [
  {
    name: 'Survival',
    subtitle: 'Life persists',
    description: 'Entities survive, eat, and reproduce without going extinct. Population stays above zero across ticks.',
    icon: '◉',
    color: '#10b981',
    glowColor: 'rgba(16, 185, 129, 0.4)',
  },
  {
    name: 'Resource Cycling',
    subtitle: 'Shaping the environment',
    description: 'Entities deplete and reshape resource patterns. Resource coverage visibly fluctuates — not uniform.',
    icon: '◈',
    color: '#06b6d4',
    glowColor: 'rgba(6, 182, 212, 0.4)',
  },
  {
    name: 'Glyph Communication',
    subtitle: 'Environmental memory',
    description: 'Glyph deposits and absorbs predict future births (lagged correlation). The environment starts carrying reusable behavioral traces.',
    icon: '◇',
    color: '#8b5cf6',
    glowColor: 'rgba(139, 92, 246, 0.4)',
  },
  {
    name: 'Diversity',
    subtitle: 'Genome divergence',
    description: 'Genomes spread apart through selection (not just random drift). Multiple behavioral strategies coexist.',
    icon: '△',
    color: '#ec4899',
    glowColor: 'rgba(236, 72, 153, 0.4)',
  },
  {
    name: 'Predation',
    subtitle: 'Predator-prey dynamics',
    description: 'Entities attack each other AND the population survives. Curved vibrio predators hunt round prey.',
    icon: '⬡',
    color: '#f59e0b',
    glowColor: 'rgba(245, 158, 11, 0.4)',
  },
  {
    name: 'Cultural Marks',
    subtitle: 'Stigmergic memory',
    description: 'Entities deposit internal state as glyphs and absorb them. Gold marks appear on the grid — knowledge persists beyond individual lifetimes.',
    icon: '⬡',
    color: '#d97706',
    glowColor: 'rgba(217, 119, 6, 0.4)',
  },
  {
    name: 'Kin Selection',
    subtitle: 'Social differentiation',
    description: 'Entities treat kin differently from strangers — cooperating with relatives while attacking outsiders. The seed of tribal behavior.',
    icon: '⬢',
    color: '#0ea5e9',
    glowColor: 'rgba(14, 165, 233, 0.4)',
  },
  {
    name: 'Speciation',
    subtitle: 'Distinct species form',
    description: 'Genome clusters emerge — groups of similar creatures with gaps between them. Different body shapes visible.',
    icon: '✦',
    color: '#f97316',
    glowColor: 'rgba(249, 115, 22, 0.4)',
  },
  {
    name: 'Ecology',
    subtitle: 'Complex ecosystem',
    description: 'Multiple species coexist with predation, glyph communication, cultural marks, kin selection, and resource cycling all active simultaneously.',
    icon: '◎',
    color: '#14b8a6',
    glowColor: 'rgba(20, 184, 166, 0.4)',
  },
  {
    name: 'Meta-Evolution',
    subtitle: 'Physics improving',
    description: 'The laws of physics themselves are evolving to produce richer life. Score trend accelerating across generations.',
    icon: '∞',
    color: '#e11d48',
    glowColor: 'rgba(225, 29, 72, 0.5)',
  },
];

interface EmergenceLadderProps {
  emergence: EmergenceState;
  tick: number;
}

export function EmergenceLadder({ emergence, tick }: EmergenceLadderProps) {
  const pulse = Math.sin(tick * 0.05) * 0.3 + 0.7;

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto py-4 px-2">
      {/* Title */}
      <div className="text-center mb-4 shrink-0">
        <h2 className="text-[10px] uppercase tracking-[0.25em] text-gray-500 font-medium">
          Emergence
        </h2>
      </div>

      {/* Stages — bottom to top */}
      <div className="flex min-h-0 flex-col-reverse gap-0.5 flex-1 justify-start">
        {STAGES.map((stage, i) => {
          const achieved = i <= emergence.stage;
          const current = i === emergence.stage;
          const progress = emergence.progress[i];
          const upcoming = i === emergence.stage + 1;

          return (
            <div
              key={i}
              className="group relative flex items-center gap-2 py-1.5 px-2 rounded-lg transition-all duration-700"
              style={{
                background: achieved
                  ? `linear-gradient(90deg, ${stage.glowColor}, transparent)`
                  : upcoming
                    ? 'rgba(255,255,255,0.02)'
                    : 'transparent',
                opacity: achieved ? 1 : upcoming ? 0.5 : 0.2,
              }}
            >
              {/* Connector line */}
              {i < 9 && (
                <div
                  className="absolute left-[18px] -top-0.5 w-px h-0.5"
                  style={{
                    background: achieved
                      ? stage.color
                      : 'rgba(255,255,255,0.05)',
                  }}
                />
              )}

              {/* Icon */}
              <div
                className="w-6 h-6 flex items-center justify-center text-xs shrink-0 rounded-md transition-all duration-500"
                style={{
                  color: achieved ? stage.color : '#374151',
                  textShadow: current
                    ? `0 0 ${12 * pulse}px ${stage.glowColor}`
                    : 'none',
                  background: achieved
                    ? `${stage.color}15`
                    : 'rgba(255,255,255,0.02)',
                  border: `1px solid ${achieved ? `${stage.color}40` : 'rgba(255,255,255,0.03)'}`,
                  transform: current ? `scale(${0.95 + pulse * 0.1})` : 'scale(1)',
                }}
              >
                {stage.icon}
              </div>

              {/* Text */}
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span
                    className="text-[11px] font-semibold truncate transition-colors duration-500"
                    style={{ color: achieved ? stage.color : '#4b5563' }}
                  >
                    {stage.name}
                  </span>
                  {current && (
                    <span
                      className="text-[8px] px-1 py-0.5 rounded uppercase tracking-wider font-bold"
                      style={{
                        color: stage.color,
                        background: `${stage.color}20`,
                        border: `1px solid ${stage.color}30`,
                      }}
                    >
                      Active
                    </span>
                  )}
                </div>
                <p
                  className="text-[8px] truncate transition-colors duration-500"
                  style={{ color: achieved ? '#9ca3af' : '#1f2937' }}
                >
                  {stage.subtitle}
                </p>

                {/* Progress bar for upcoming */}
                {upcoming && progress > 0 && (
                  <div className="mt-1 w-full bg-gray-800/50 rounded-full h-0.5">
                    <div
                      className="h-0.5 rounded-full transition-all duration-1000"
                      style={{
                        width: `${progress * 100}%`,
                        background: stage.color,
                        boxShadow: `0 0 6px ${stage.glowColor}`,
                      }}
                    />
                  </div>
                )}
              </div>

              {/* Tooltip on hover */}
              <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 hidden group-hover:block z-50 pointer-events-none">
                <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl min-w-[200px]">
                  <p className="text-xs font-medium text-gray-200">{stage.name}</p>
                  <p className="text-[10px] text-gray-400 mt-0.5 leading-relaxed">{stage.description}</p>
                  <div className="mt-1.5 flex items-center gap-2">
                    <div className="flex-1 bg-gray-800 rounded-full h-1">
                      <div
                        className="h-1 rounded-full"
                        style={{ width: `${progress * 100}%`, background: stage.color }}
                      />
                    </div>
                    <span className="text-[9px] font-mono text-gray-500">
                      {(progress * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
