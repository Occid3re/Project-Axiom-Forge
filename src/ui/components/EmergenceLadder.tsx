/**
 * The Emergence Ladder — the soul of the UI.
 * Eight stages from simple replicators to recursive self-improvement.
 * Lights up as the simulation achieves each milestone.
 */

import { useMemo } from 'react';
import type { WorldScores, GenerationResult } from '../../engine';

export interface EmergenceState {
  stage: number;       // 0-7 (index of highest achieved stage)
  progress: number[];  // 0-1 for each stage
}

const STAGES = [
  {
    name: 'Replicators',
    subtitle: 'Self-sustaining life',
    description: 'Populations persist and reproduce reliably',
    icon: '◉',
    color: '#10b981',
    glowColor: 'rgba(16, 185, 129, 0.4)',
  },
  {
    name: 'Communication',
    subtitle: 'Signals carry meaning',
    description: 'Information flows between agents',
    icon: '◈',
    color: '#06b6d4',
    glowColor: 'rgba(6, 182, 212, 0.4)',
  },
  {
    name: 'External Memory',
    subtitle: 'Environment as storage',
    description: 'Trails, structures, encoded signals',
    icon: '◇',
    color: '#8b5cf6',
    glowColor: 'rgba(139, 92, 246, 0.4)',
  },
  {
    name: 'Tool Use',
    subtitle: 'Reusable modification',
    description: 'Agents build lasting environmental changes',
    icon: '⬡',
    color: '#f59e0b',
    glowColor: 'rgba(245, 158, 11, 0.4)',
  },
  {
    name: 'Abstraction',
    subtitle: 'Internal models',
    description: 'Prediction, compression, reasoning seeds',
    icon: '△',
    color: '#ec4899',
    glowColor: 'rgba(236, 72, 153, 0.4)',
  },
  {
    name: 'Civilization',
    subtitle: 'Specialization emerges',
    description: 'Groups form roles — knowledge outlives individuals',
    icon: '⬢',
    color: '#f97316',
    glowColor: 'rgba(249, 115, 22, 0.4)',
  },
  {
    name: 'World Engineering',
    subtitle: 'Optimizing reality',
    description: 'Agents reshape the rules they live under',
    icon: '◎',
    color: '#14b8a6',
    glowColor: 'rgba(20, 184, 166, 0.4)',
  },
  {
    name: 'Recursive Threshold',
    subtitle: 'Self-improving systems',
    description: 'The system improves its own means of improvement',
    icon: '∞',
    color: '#e11d48',
    glowColor: 'rgba(225, 29, 72, 0.5)',
  },
];

export function detectEmergence(
  scores: WorldScores | null,
  generations: GenerationResult[],
): EmergenceState {
  const progress = new Array(8).fill(0);

  if (!scores) return { stage: -1, progress };

  // Stage 0: Replicators — population persists
  progress[0] = Math.min(1, scores.persistence / 0.4);

  // Stage 1: Communication — signals correlate with behavior
  progress[1] = Math.min(1, scores.communication / 0.25);

  // Stage 2: External Memory — environmental structure
  progress[2] = Math.min(1, scores.envStructure / 0.3);

  // Stage 3: Tool Use — env structure + diversity
  progress[3] = Math.min(1, (scores.envStructure * 0.6 + scores.diversity * 0.4) / 0.4);

  // Stage 4: Abstraction — complexity growth
  progress[4] = Math.min(1, scores.complexityGrowth / 0.3);

  // Stage 5: Civilization — everything together
  const civScore = (scores.persistence + scores.diversity + scores.communication + scores.envStructure) / 4;
  progress[5] = Math.min(1, civScore / 0.4);

  // Stage 6: World Engineering — meta-evolution improving over generations
  if (generations.length >= 3) {
    const recent = generations.slice(-3);
    const improving = recent.every((g, i) => i === 0 || g.bestScore >= recent[i - 1].bestScore * 0.95);
    const avgImprovement = generations.length > 1
      ? (generations[generations.length - 1].bestScore - generations[0].bestScore) / generations[0].bestScore
      : 0;
    progress[6] = Math.min(1, (improving ? 0.5 : 0) + Math.max(0, avgImprovement) * 2);
  }

  // Stage 7: Recursive Threshold — acceleration of improvement
  if (generations.length >= 5) {
    const scores2 = generations.map(g => g.bestScore);
    const firstHalf = scores2.slice(0, Math.floor(scores2.length / 2));
    const secondHalf = scores2.slice(Math.floor(scores2.length / 2));
    const firstRate = firstHalf.length > 1 ? (firstHalf[firstHalf.length - 1] - firstHalf[0]) / firstHalf.length : 0;
    const secondRate = secondHalf.length > 1 ? (secondHalf[secondHalf.length - 1] - secondHalf[0]) / secondHalf.length : 0;
    const accelerating = secondRate > firstRate * 1.1;
    progress[7] = Math.min(1, accelerating ? Math.min(1, secondRate / (firstRate + 0.001)) * 0.7 : progress[6] * 0.3);
  }

  // Find highest achieved stage (> 0.6 threshold)
  let stage = -1;
  for (let i = 0; i < 8; i++) {
    if (progress[i] >= 0.6) stage = i;
    else break;
  }

  return { stage, progress };
}

interface EmergenceLadderProps {
  emergence: EmergenceState;
  tick: number;
}

export function EmergenceLadder({ emergence, tick }: EmergenceLadderProps) {
  const pulse = Math.sin(tick * 0.05) * 0.3 + 0.7;

  return (
    <div className="flex flex-col h-full py-6 px-2">
      {/* Title */}
      <div className="text-center mb-6">
        <h2 className="text-[10px] uppercase tracking-[0.25em] text-gray-500 font-medium">
          Emergence
        </h2>
      </div>

      {/* Stages — bottom to top */}
      <div className="flex flex-col-reverse gap-1 flex-1 justify-center">
        {STAGES.map((stage, i) => {
          const achieved = i <= emergence.stage;
          const current = i === emergence.stage;
          const progress = emergence.progress[i];
          const upcoming = i === emergence.stage + 1;

          return (
            <div
              key={i}
              className="group relative flex items-center gap-3 py-2 px-3 rounded-lg transition-all duration-700"
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
              {i < 7 && (
                <div
                  className="absolute left-[22px] -top-1 w-px h-1"
                  style={{
                    background: achieved
                      ? stage.color
                      : 'rgba(255,255,255,0.05)',
                  }}
                />
              )}

              {/* Icon */}
              <div
                className="w-7 h-7 flex items-center justify-center text-sm shrink-0 rounded-md transition-all duration-500"
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
                    className="text-xs font-semibold truncate transition-colors duration-500"
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
                  className="text-[9px] truncate transition-colors duration-500"
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
                <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl min-w-[180px]">
                  <p className="text-xs font-medium text-gray-200">{stage.name}</p>
                  <p className="text-[10px] text-gray-400 mt-0.5">{stage.description}</p>
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
