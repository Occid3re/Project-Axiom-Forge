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
} from './engine';
import { WorldView } from './ui/components/WorldView';
import { StatsPanel } from './ui/components/StatsPanel';
import { ControlPanel } from './ui/components/ControlPanel';
import { LineageTree } from './ui/components/LineageTree';
import { PopulationChart } from './ui/components/PopulationChart';
import { WorldLawsView } from './ui/components/WorldLawsView';

export default function App() {
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [world, setWorld] = useState<World | null>(null);
  const [snapshot, setSnapshot] = useState<WorldSnapshot | null>(null);
  const [scores, setScores] = useState<WorldScores | null>(null);
  const [currentLaws, setCurrentLaws] = useState<WorldLaws | null>(null);
  const [generations, setGenerations] = useState<GenerationResult[]>([]);
  const [metaState, setMetaState] = useState<MetaState | null>(null);
  const [snapshots, setSnapshots] = useState<WorldSnapshot[]>([]);
  const [bestLaws, setBestLaws] = useState<WorldLaws | null>(null);
  const [log, setLog] = useState<string[]>([]);

  const runningRef = useRef(false);
  const pausedRef = useRef(false);
  const animFrameRef = useRef<number>(0);

  const addLog = useCallback((msg: string) => {
    setLog(prev => [...prev.slice(-99), msg]);
  }, []);

  const handleStart = useCallback((config: MetaConfig) => {
    setRunning(true);
    setPaused(false);
    runningRef.current = true;
    pausedRef.current = false;
    setGenerations([]);
    setScores(null);
    setSnapshot(null);
    setSnapshots([]);
    setLog([]);

    addLog(`Starting meta-evolution: ${config.metaGenerations} generations, ${config.worldsPerGeneration} worlds each`);

    const gen = runMetaEvolution(config);

    const runStep = () => {
      if (!runningRef.current) return;
      if (pausedRef.current) {
        animFrameRef.current = requestAnimationFrame(runStep);
        return;
      }

      // Process a batch of simulation steps per frame for speed
      const batchSize = 5;
      for (let b = 0; b < batchSize; b++) {
        const result = gen.next();

        if (result.done) {
          const finalState = result.value as MetaState;
          setMetaState(finalState);
          setGenerations(finalState.generationResults);
          setRunning(false);
          runningRef.current = false;

          if (finalState.bestEver) {
            setBestLaws(finalState.bestEver.laws);
            setScores(finalState.bestEver.scores);
            addLog(`Evolution complete! Best score: ${finalState.bestEver.scores.total.toFixed(3)}`);

            // Run best world for visualization
            replayBestWorld(finalState.bestEver.laws, config);
          }
          return;
        }

        const state = result.value as MetaState;
        setMetaState(state);
        setGenerations([...state.generationResults]);

        if (state.bestEver) {
          setBestLaws(state.bestEver.laws);
        }

        // Log generation completions
        const latestGen = state.generationResults[state.generationResults.length - 1];
        if (latestGen && state.currentWorld === 1) {
          addLog(`Gen ${latestGen.generation}: best=${latestGen.bestScore.toFixed(3)} avg=${latestGen.avgScore.toFixed(3)}`);
        }
      }

      animFrameRef.current = requestAnimationFrame(runStep);
    };

    animFrameRef.current = requestAnimationFrame(runStep);
  }, [addLog]);

  const replayBestWorld = useCallback((laws: WorldLaws, config: MetaConfig) => {
    const replayWorld = new World(laws, {
      gridSize: config.gridSize,
      steps: config.worldSteps,
      initialEntities: config.initialEntities,
    }, config.seed + 99999);

    setWorld(replayWorld);
    setCurrentLaws(laws);
    const allSnaps: WorldSnapshot[] = [];

    let tick = 0;
    const animate = () => {
      if (tick >= config.worldSteps) {
        const finalScores = scoreWorld(
          { snapshots: allSnaps, finalPopulation: replayWorld.entities.count, peakPopulation: 0, disasterCount: 0, postDisasterRecoveries: 0 },
          laws,
          config.scoreWeights
        );
        setScores(finalScores);
        return;
      }

      const snap = replayWorld.step();
      allSnaps.push(snap);
      setSnapshot(snap);
      setSnapshots([...allSnaps]);
      setWorld(replayWorld);
      tick++;

      requestAnimationFrame(animate);
    };

    addLog('Replaying best world...');
    requestAnimationFrame(animate);
  }, [addLog]);

  const handlePause = useCallback(() => {
    setPaused(true);
    pausedRef.current = true;
  }, []);

  const handleResume = useCallback(() => {
    setPaused(false);
    pausedRef.current = false;
  }, []);

  const handleStop = useCallback(() => {
    setRunning(false);
    setPaused(false);
    runningRef.current = false;
    pausedRef.current = false;
    cancelAnimationFrame(animFrameRef.current);
  }, []);

  useEffect(() => {
    return () => cancelAnimationFrame(animFrameRef.current);
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0b0f] text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800/50 bg-gray-900/30 backdrop-blur-sm">
        <div className="max-w-[1800px] mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center">
              <span className="text-sm font-bold">LU</span>
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                LostUplink<span className="text-gray-500 font-normal text-sm ml-1.5">: Axiom Forge</span>
              </h1>
              <p className="text-[10px] text-gray-500">Meta-Evolution Simulation</p>
            </div>
          </div>
          {metaState && (
            <div className="flex items-center gap-4 text-xs text-gray-400">
              <span>Gen <strong className="text-gray-200">{metaState.generation + 1}</strong>/{metaState.totalGenerations}</span>
              <span>World <strong className="text-gray-200">{metaState.currentWorld}</strong>/{metaState.totalWorlds}</span>
              {metaState.bestEver && (
                <span>Best: <strong className="text-cyan-400">{metaState.bestEver.scores.total.toFixed(2)}</strong></span>
              )}
            </div>
          )}
        </div>
      </header>

      {/* Main layout */}
      <main className="max-w-[1800px] mx-auto p-4 grid grid-cols-[280px_1fr_300px] gap-4 h-[calc(100vh-60px)]">
        {/* Left sidebar */}
        <div className="flex flex-col gap-4 overflow-y-auto">
          <ControlPanel
            onStart={handleStart}
            onPause={handlePause}
            onResume={handleResume}
            onStop={handleStop}
            running={running}
            paused={paused}
          />
          <WorldLawsView laws={currentLaws ?? bestLaws} title={currentLaws ? 'Active World Laws' : 'Best World Laws'} />
        </div>

        {/* Center */}
        <div className="flex flex-col gap-4 overflow-hidden">
          {/* World canvas */}
          <div className="flex-1 flex items-center justify-center min-h-0">
            <WorldView world={world} size={Math.min(640, 640)} />
          </div>

          {/* Charts row */}
          <div className="flex gap-4">
            <PopulationChart snapshots={snapshots} width={400} height={120} />
            <LineageTree generations={generations} bestEverId={metaState?.bestEver?.id ?? null} />
          </div>
        </div>

        {/* Right sidebar */}
        <div className="flex flex-col gap-4 overflow-y-auto">
          <StatsPanel
            snapshot={snapshot}
            scores={scores}
            generation={metaState?.generation ?? 0}
            worldIndex={metaState?.currentWorld ?? 0}
            totalWorlds={metaState?.totalWorlds ?? 0}
          />

          {/* Event log */}
          <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg border border-gray-700/50 p-4 flex-1 min-h-0">
            <h3 className="text-sm font-semibold text-gray-200 mb-2">Event Log</h3>
            <div className="overflow-y-auto h-48 text-[11px] font-mono text-gray-400 space-y-0.5">
              {log.map((entry, i) => (
                <div key={i} className="leading-tight">
                  <span className="text-gray-600">[{i.toString().padStart(3, '0')}]</span> {entry}
                </div>
              ))}
              {log.length === 0 && (
                <div className="text-gray-600">Configure and start the simulation...</div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
