/**
 * Lineage visualization — shows how world-laws evolved across generations.
 */

import { useMemo } from 'react';
import type { GenerationResult, WorldResult } from '../../engine';

interface LineageTreeProps {
  generations: GenerationResult[];
  bestEverId: string | null;
}

export function LineageTree({ generations, bestEverId }: LineageTreeProps) {
  const maxScore = useMemo(() => {
    let max = 0;
    for (const gen of generations) {
      for (const w of gen.worlds) {
        if (w.scores.total > max) max = w.scores.total;
      }
    }
    return max || 1;
  }, [generations]);

  if (generations.length === 0) {
    return (
      <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg border border-gray-700/50 p-4">
        <h3 className="text-sm font-semibold text-gray-200 mb-2">Lineage</h3>
        <p className="text-xs text-gray-500">No generations yet</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg border border-gray-700/50 p-4">
      <h3 className="text-sm font-semibold text-gray-200 mb-3">World Lineage</h3>

      <div className="overflow-x-auto">
        <div className="flex gap-1 min-w-fit">
          {generations.map((gen) => (
            <div key={gen.generation} className="flex flex-col items-center gap-0.5">
              <span className="text-[10px] text-gray-500 mb-1">G{gen.generation}</span>
              {gen.worlds.map((w) => {
                const intensity = w.scores.total / maxScore;
                const isBest = w.id === bestEverId;
                return (
                  <div
                    key={w.id}
                    title={`ID: ${w.id}\nScore: ${w.scores.total.toFixed(3)}\nPop: ${w.history.finalPopulation}`}
                    className={`w-3 h-3 rounded-sm transition-all cursor-pointer hover:scale-150 ${
                      isBest ? 'ring-1 ring-cyan-400' : ''
                    }`}
                    style={{
                      backgroundColor: `rgba(${Math.floor(40 + intensity * 100)}, ${Math.floor(180 * intensity)}, ${Math.floor(200 * intensity)}, ${0.3 + intensity * 0.7})`,
                    }}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Generation score chart */}
      <div className="mt-4">
        <h4 className="text-xs font-semibold text-gray-300 mb-2">Score Progression</h4>
        <div className="flex items-end gap-1 h-20">
          {generations.map((gen) => {
            const bestHeight = (gen.bestScore / maxScore) * 100;
            const avgHeight = (gen.avgScore / maxScore) * 100;
            return (
              <div key={gen.generation} className="flex-1 flex flex-col items-center gap-0.5 relative h-full justify-end">
                <div
                  className="w-full rounded-t opacity-30"
                  style={{
                    height: `${avgHeight}%`,
                    backgroundColor: '#6366f1',
                  }}
                />
                <div
                  className="w-full rounded-t absolute bottom-0"
                  style={{
                    height: `${bestHeight}%`,
                    background: 'linear-gradient(180deg, #06b6d4, #8b5cf6)',
                    opacity: 0.8,
                  }}
                />
              </div>
            );
          })}
        </div>
        <div className="flex justify-between text-[10px] text-gray-500 mt-1">
          <span>Gen 0</span>
          <span>Gen {generations.length - 1}</span>
        </div>
      </div>
    </div>
  );
}
