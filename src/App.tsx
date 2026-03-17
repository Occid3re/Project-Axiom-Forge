/**
 * LostUplink: Axiom Forge
 * Auto-running meta-evolution simulation — view only.
 * No controls. Just emergence.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  World,
  type WorldSnapshot,
  type WorldScores,
  type WorldLaws,
  type MetaConfig,
  type MetaState,
  type GenerationResult,
  runMetaEvolution,
  scoreWorld,
  DEFAULT_META_CONFIG,
} from './engine';
import { WorldView } from './ui/components/WorldView';
import { EmergenceLadder, detectEmergence } from './ui/components/EmergenceLadder';
import { AmbientStats } from './ui/components/AmbientStats';
import { TransmissionLog } from './ui/components/TransmissionLog';
import { PopulationChart } from './ui/components/PopulationChart';

// Continuous auto-run config — tuned for a good viewer experience
const AUTO_CONFIG: MetaConfig = {
  ...DEFAULT_META_CONFIG,
  seed: Math.floor(Math.random() * 99999),
  metaGenerations: 999, // effectively infinite
  worldsPerGeneration: 8,
  worldSteps: 600,
  gridSize: 80,
  initialEntities: 50,
  topK: 3,
  scoreWeights: {
    persistence: 1.0,
    diversity: 1.5,
    complexityGrowth: 1.2,
    communication: 2.0,
    envStructure: 1.0,
    adaptability: 1.2,
  },
};

export default function App() {
  const [world, setWorld] = useState<World | null>(null);
  const [snapshot, setSnapshot] = useState<WorldSnapshot | null>(null);
  const [scores, setScores] = useState<WorldScores | null>(null);
  const [currentLaws, setCurrentLaws] = useState<WorldLaws | null>(null);
  const [generations, setGenerations] = useState<GenerationResult[]>([]);
  const [metaState, setMetaState] = useState<MetaState | null>(null);
  const [snapshots, setSnapshots] = useState<WorldSnapshot[]>([]);
  const [log, setLog] = useState<string[]>([
    'AXIOM FORGE initializing...',
    'Seeding primordial conditions...',
    'Spawning initial world-law variants...',
  ]);
  const [bestScore, setBestScore] = useState(0);
  const [tick, setTick] = useState(0);

  const rafRef = useRef<number>(0);
  const genRef = useRef<ReturnType<typeof runMetaEvolution> | null>(null);
  const replayRef = useRef<boolean>(false);

  const addLog = useCallback((msg: string) => {
    setLog(prev => [...prev.slice(-200), msg]);
  }, []);

  // Auto-replay best world continuously for visualization
  const startReplay = useCallback((laws: WorldLaws, seed: number, label: string) => {
    replayRef.current = true;
    const w = new World(laws, {
      gridSize: AUTO_CONFIG.gridSize,
      steps: AUTO_CONFIG.worldSteps,
      initialEntities: AUTO_CONFIG.initialEntities,
    }, seed);

    const allSnaps: WorldSnapshot[] = [];
    let t = 0;
    setCurrentLaws(laws);
    setSnapshots([]);

    const animate = () => {
      if (!replayRef.current) return;
      if (t >= AUTO_CONFIG.worldSteps) {
        // Replay loops seamlessly
        addLog(`${label} — replaying...`);
        startReplay(laws, seed + 1, label);
        return;
      }
      const s = w.step();
      allSnaps.push(s);
      t++;
      setWorld(w);
      setSnapshot(s);
      setTick(s.tick);
      // Downsample stored snapshots to avoid memory buildup
      if (t % 3 === 0) setSnapshots([...allSnaps.slice(-200)]);
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
  }, [addLog]);

  // Meta-evolution loop — runs continuously in batches per frame
  useEffect(() => {
    const config = { ...AUTO_CONFIG };
    genRef.current = runMetaEvolution(config);

    let bestLaws: WorldLaws | null = null;
    let bestSeed = config.seed;
    let lastReplayedScore = 0;

    const runMeta = () => {
      const gen = genRef.current;
      if (!gen) return;

      // Process multiple worlds per frame tick for speed
      for (let b = 0; b < 3; b++) {
        const result = gen.next();

        if (result.done) {
          // Restart with a new seed — infinite loop
          addLog('— Cycle complete. Restarting evolution —');
          const newSeed = Math.floor(Math.random() * 99999);
          genRef.current = runMetaEvolution({ ...config, seed: newSeed });
          return;
        }

        const state = result.value as MetaState;
        setMetaState(state);
        setGenerations(prev => {
          const updated = [...state.generationResults];
          return updated;
        });

        if (state.bestEver) {
          const s = state.bestEver.scores.total;
          if (s > bestScore) {
            setBestScore(s);
          }

          // Start/update replay with best world when score improves meaningfully
          if (!bestLaws || s > lastReplayedScore * 1.05) {
            bestLaws = state.bestEver.laws;
            lastReplayedScore = s;
            bestSeed = state.bestEver.generation * 7919 + config.seed;
            replayRef.current = false;
            cancelAnimationFrame(rafRef.current);

            const gen_ = state.bestEver.generation;
            const world_ = state.bestEver.id;
            addLog(`Gen ${gen_} | World ${world_} | Score ${s.toFixed(3)} — new best`);
            setTimeout(() => startReplay(bestLaws!, bestSeed, `Gen${gen_}`), 50);
          }
        }

        const latestGen = state.generationResults[state.generationResults.length - 1];
        if (latestGen && state.currentWorld === 1 && state.generationResults.length > 0) {
          addLog(
            `Gen ${latestGen.generation} complete — best ${latestGen.bestScore.toFixed(3)} | avg ${latestGen.avgScore.toFixed(3)} | pop ${state.bestEver?.history.peakPopulation ?? '?'}`
          );
        }
      }

      rafRef.current = requestAnimationFrame(runMeta);
    };

    // Small initial delay so React can render first
    const startTimer = setTimeout(() => {
      rafRef.current = requestAnimationFrame(runMeta);
    }, 300);

    return () => {
      clearTimeout(startTimer);
      cancelAnimationFrame(rafRef.current);
      replayRef.current = false;
    };
  }, []);

  // Update scores from snapshots when simulation ticks
  useEffect(() => {
    if (!currentLaws || snapshots.length < 20) return;
    const sc = scoreWorld(
      {
        snapshots,
        finalPopulation: snapshot?.population ?? 0,
        peakPopulation: Math.max(...snapshots.map(s => s.population), 0),
        disasterCount: 0,
        postDisasterRecoveries: 0,
      },
      currentLaws,
      AUTO_CONFIG.scoreWeights,
    );
    setScores(sc);
  }, [snapshots.length]);

  const emergence = detectEmergence(scores, generations);

  // Canvas size — responsive
  const canvasSize = Math.min(typeof window !== 'undefined' ? window.innerHeight - 180 : 700, 760);

  return (
    <div className="h-screen w-screen bg-[#070809] overflow-hidden flex flex-col select-none">

      {/* Header — minimal */}
      <header className="shrink-0 flex items-center justify-between px-6 py-3 border-b border-white/[0.04]">
        <div className="flex items-center gap-3">
          {/* Logo mark */}
          <div className="relative w-7 h-7">
            <div className="absolute inset-0 rounded-md bg-gradient-to-br from-cyan-500/20 to-purple-600/20 border border-cyan-500/20" />
            <div
              className="absolute inset-0 rounded-md flex items-center justify-center text-[10px] font-black"
              style={{ color: '#06b6d4' }}
            >
              LU
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-bold tracking-wide text-gray-100">LostUplink</span>
            <span className="text-xs text-gray-600 font-light">Axiom Forge</span>
          </div>
        </div>

        {/* Center — current emergence stage */}
        {emergence.stage >= 0 && (
          <div
            className="flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-medium"
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <span className="text-gray-500">Stage reached:</span>
            <span
              className="font-semibold"
              style={{ color: '#06b6d4' }}
            >
              {['Replicators','Communication','External Memory','Tool Use','Abstraction','Civilization','World Engineering','Recursive Threshold'][emergence.stage]}
            </span>
          </div>
        )}

        {/* Right — live indicator */}
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
          </span>
          <span className="text-[10px] text-gray-600 uppercase tracking-widest">Observing</span>
        </div>
      </header>

      {/* Main */}
      <div className="flex-1 flex min-h-0">

        {/* Emergence Ladder — left */}
        <div
          className="w-52 shrink-0 border-r border-white/[0.03]"
          style={{ background: 'rgba(0,0,0,0.3)' }}
        >
          <EmergenceLadder emergence={emergence} tick={tick} />
        </div>

        {/* World canvas — center */}
        <div className="flex-1 flex flex-col min-w-0">

          {/* Canvas area */}
          <div
            className="flex-1 flex items-center justify-center min-h-0 relative"
            style={{ background: 'radial-gradient(ellipse at center, #0d0f14 0%, #070809 70%)' }}
          >
            {/* Corner watermark */}
            <div className="absolute top-3 left-4 text-[9px] text-gray-800 font-mono tracking-widest uppercase pointer-events-none">
              axiom-forge v0.1
            </div>

            {/* Population counter overlay */}
            {snapshot && snapshot.population > 0 && (
              <div className="absolute top-3 right-4 flex flex-col items-end pointer-events-none">
                <span
                  className="text-3xl font-black font-mono tabular-nums leading-none"
                  style={{ color: 'rgba(16,185,129,0.7)', textShadow: '0 0 20px rgba(16,185,129,0.3)' }}
                >
                  {snapshot.population}
                </span>
                <span className="text-[8px] uppercase tracking-widest text-gray-700">entities</span>
              </div>
            )}

            {/* Extinction indicator */}
            {snapshot && snapshot.population === 0 && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="text-center">
                  <div className="text-2xl font-mono text-red-900/60 tracking-[0.5em] uppercase">
                    Extinction
                  </div>
                  <div className="text-[10px] text-gray-700 mt-1 tracking-wider">Awaiting next world...</div>
                </div>
              </div>
            )}

            <WorldView world={world} size={canvasSize} />
          </div>

          {/* Bottom bar — ambient stats + log */}
          <div
            className="shrink-0 border-t border-white/[0.04]"
            style={{ background: 'rgba(0,0,0,0.4)' }}
          >
            <AmbientStats
              snapshot={snapshot}
              generation={metaState?.generation ?? 0}
              totalGenerations={AUTO_CONFIG.metaGenerations}
              worldIndex={metaState?.currentWorld ?? 0}
              totalWorlds={AUTO_CONFIG.worldsPerGeneration}
              bestScore={bestScore}
              generations={generations}
            />
            <TransmissionLog entries={log} />
          </div>
        </div>

        {/* Right panel — charts + score breakdown */}
        <div
          className="w-64 shrink-0 border-l border-white/[0.03] flex flex-col"
          style={{ background: 'rgba(0,0,0,0.3)' }}
        >
          {/* Population timeline */}
          <div className="p-4 border-b border-white/[0.04]">
            <h4 className="text-[9px] uppercase tracking-[0.2em] text-gray-600 mb-2">Timeline</h4>
            <PopulationChart snapshots={snapshots} width={220} height={90} />
          </div>

          {/* Score bars */}
          {scores && (
            <div className="p-4 border-b border-white/[0.04]">
              <h4 className="text-[9px] uppercase tracking-[0.2em] text-gray-600 mb-3">Emergence Scores</h4>
              <div className="space-y-2">
                {([
                  ['Persistence',    scores.persistence,      '#10b981'],
                  ['Diversity',      scores.diversity,        '#8b5cf6'],
                  ['Complexity',     scores.complexityGrowth, '#ec4899'],
                  ['Communication',  scores.communication,    '#06b6d4'],
                  ['Env Structure',  scores.envStructure,     '#f59e0b'],
                  ['Adaptability',   scores.adaptability,     '#f97316'],
                ] as [string, number, string][]).map(([label, val, color]) => (
                  <div key={label}>
                    <div className="flex justify-between text-[9px] mb-0.5">
                      <span className="text-gray-500">{label}</span>
                      <span className="font-mono" style={{ color }}>{val.toFixed(3)}</span>
                    </div>
                    <div className="h-0.5 rounded-full bg-white/[0.04] overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${Math.min(100, val * 100)}%`,
                          background: color,
                          boxShadow: `0 0 6px ${color}60`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-3 pt-3 border-t border-white/[0.04] flex justify-between items-baseline">
                <span className="text-[9px] text-gray-600 uppercase tracking-wider">Total</span>
                <span
                  className="text-xl font-black font-mono"
                  style={{ color: '#f59e0b', textShadow: '0 0 12px rgba(245,158,11,0.4)' }}
                >
                  {scores.total.toFixed(2)}
                </span>
              </div>
            </div>
          )}

          {/* Lineage mini-heatmap */}
          {generations.length > 0 && (
            <div className="p-4 flex-1">
              <h4 className="text-[9px] uppercase tracking-[0.2em] text-gray-600 mb-3">Lineage</h4>
              <div className="flex flex-wrap gap-0.5">
                {generations.flatMap(gen =>
                  gen.worlds.map(w => {
                    const norm = bestScore > 0 ? w.scores.total / bestScore : 0;
                    return (
                      <div
                        key={w.id}
                        title={`Gen ${w.generation} | Score ${w.scores.total.toFixed(2)}`}
                        className="w-2.5 h-2.5 rounded-sm"
                        style={{
                          background: `rgba(6, 182, 212, ${0.08 + norm * 0.92})`,
                          boxShadow: norm > 0.9 ? `0 0 4px rgba(6,182,212,0.6)` : 'none',
                        }}
                      />
                    );
                  })
                )}
              </div>
              {generations.length > 0 && (
                <div className="mt-3 text-[9px] text-gray-700 font-mono">
                  {generations.length} gen · {generations.reduce((a, g) => a + g.worlds.length, 0)} worlds simulated
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
