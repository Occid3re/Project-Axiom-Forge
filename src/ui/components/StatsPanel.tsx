/**
 * Stats panel showing real-time simulation metrics.
 */

import type { WorldSnapshot, WorldScores } from '../../engine';

interface StatsPanelProps {
  snapshot: WorldSnapshot | null;
  scores: WorldScores | null;
  generation: number;
  worldIndex: number;
  totalWorlds: number;
}

function StatRow({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="flex justify-between items-center py-0.5">
      <span className="text-gray-400 text-xs">{label}</span>
      <span className={`text-sm font-mono font-medium ${color ?? 'text-gray-100'}`}>
        {typeof value === 'number' ? value.toFixed(2) : value}
      </span>
    </div>
  );
}

function ScoreBar({ label, value, max = 1 }: { label: string; value: number; max?: number }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="mb-1.5">
      <div className="flex justify-between text-xs mb-0.5">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-300 font-mono">{value.toFixed(3)}</span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-1.5">
        <div
          className="h-1.5 rounded-full transition-all duration-300"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, #06b6d4, #8b5cf6)`,
          }}
        />
      </div>
    </div>
  );
}

export function StatsPanel({ snapshot, scores, generation, worldIndex, totalWorlds }: StatsPanelProps) {
  return (
    <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg border border-gray-700/50 p-4 w-72">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-200">Simulation</h3>
        <span className="text-xs text-cyan-400 font-mono">
          Gen {generation} — World {worldIndex}/{totalWorlds}
        </span>
      </div>

      {snapshot && (
        <div className="space-y-0.5 mb-4 border-b border-gray-700/50 pb-3">
          <StatRow label="Tick" value={snapshot.tick} color="text-cyan-300" />
          <StatRow label="Population" value={snapshot.population} color="text-emerald-400" />
          <StatRow label="Mean Energy" value={snapshot.meanEnergy} />
          <StatRow label="Diversity" value={snapshot.diversity} />
          <StatRow label="Births" value={snapshot.births} color="text-green-400" />
          <StatRow label="Deaths" value={snapshot.deaths} color="text-red-400" />
          <StatRow label="Attacks" value={snapshot.attacks} color="text-orange-400" />
          <StatRow label="Signals" value={snapshot.signals} color="text-purple-400" />
          <StatRow label="Resource Coverage" value={snapshot.resourceCoverage} />
        </div>
      )}

      {scores && (
        <div>
          <h4 className="text-xs font-semibold text-gray-300 mb-2">World Scores</h4>
          <ScoreBar label="Persistence" value={scores.persistence} />
          <ScoreBar label="Diversity" value={scores.diversity} />
          <ScoreBar label="Complexity" value={scores.complexityGrowth} />
          <ScoreBar label="Communication" value={scores.communication} />
          <ScoreBar label="Environment" value={scores.envStructure} />
          <ScoreBar label="Adaptability" value={scores.adaptability} />
          <ScoreBar label="Speciation" value={scores.speciation ?? 0} />
          <ScoreBar label="Interactions" value={scores.interactions ?? 0} />
          <div className="mt-2 pt-2 border-t border-gray-700/50">
            <div className="flex justify-between items-center">
              <span className="text-gray-300 text-xs font-semibold">Total Score</span>
              <span className="text-lg font-bold font-mono text-cyan-400">
                {scores.total.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      )}

      {!snapshot && !scores && (
        <div className="text-gray-500 text-xs text-center py-4">
          Start simulation to see metrics
        </div>
      )}
    </div>
  );
}
