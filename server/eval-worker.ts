/**
 * Eval worker — runs one complete eval world per job, on a dedicated OS thread.
 *
 * Protocol:
 *   main → worker:  { jobId, laws, seed, evalSteps?, scoreWeights? }
 *   worker → main:  { jobId, scores, avgTickMs, cpuFactor }
 *
 * Workers are kept alive between jobs — parentPort.on('message') loop.
 * The worker inherits tsx ESM hooks from the parent process (tsx v4).
 */

import { parentPort } from 'worker_threads';
import { World, type WorldHistory } from '../src/engine/world.ts';
import { scoreWorld, type WorldScores } from '../src/engine/scoring.ts';
import type { WorldLaws } from '../src/engine/world-laws.ts';

const DEFAULT_EVAL_STEPS    = 3200;
const GRID_SIZE             = 64;
const INITIAL_ENTITIES      = 50;
const DISPLAY_PREFLIGHT_GRID_SIZE = 256;
const DISPLAY_PREFLIGHT_INITIAL_ENTITIES = 800;
const CPU_TARGET_MS         = 6.0;  // 270-weight NN is expensive; allow rich worlds to run
const CPU_PENALTY_W         = 0.05; // minimal penalty — complex worlds should not be selected against

const DEFAULT_SCORE_WEIGHTS = {
  persistence:           0.5,
  diversity:             1.4,
  complexityGrowth:      1.0,
  communication:         2.8,
  envStructure:          0.6,
  adaptability:          1.0,
  speciation:            2.6,
  interactions:          2.4,
  spatialStructure:      2.0,
  populationDynamics:    2.3,
  stigmergicUse:         3.0,
  socialDifferentiation: 4.5,
};

interface JobMessage {
  jobId: number;
  kind?: 'eval' | 'display-preflight';
  laws: WorldLaws;
  seed: number;
  evalSteps?: number;
  gridSize?: number;
  initialEntities?: number;
  scoreWeights?: Record<string, number>;
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  let sum = 0;
  for (const value of arr) sum += value;
  return sum / arr.length;
}

parentPort!.on('message', ({ jobId, kind, laws, seed, evalSteps, gridSize, initialEntities, scoreWeights }: JobMessage) => {
  const mode = kind ?? 'eval';
  const steps = evalSteps ?? DEFAULT_EVAL_STEPS;
  const weights = scoreWeights ?? DEFAULT_SCORE_WEIGHTS;
  const size = gridSize ?? (mode === 'display-preflight' ? DISPLAY_PREFLIGHT_GRID_SIZE : GRID_SIZE);
  const initial = initialEntities ?? (mode === 'display-preflight' ? DISPLAY_PREFLIGHT_INITIAL_ENTITIES : INITIAL_ENTITIES);

  const world = new World(
    laws,
    { gridSize: size, steps, initialEntities: initial },
    seed,
  );

  const t0      = Date.now();
  let peakPop   = 0;
  const snapshots = [];

  for (let t = 0; t < steps; t++) {
    const snap = world.step();
    snapshots.push(snap);
    if (snap.population > peakPop) peakPop = snap.population;
  }

  const wallMs    = Date.now() - t0;
  const avgTickMs = wallMs / steps;

  const history: WorldHistory = {
    snapshots,
    finalPopulation:       world.entities.count,
    peakPopulation:        peakPop,
    disasterCount:         0,
    postDisasterRecoveries: 0,
  };

  const scores = scoreWorld(history, laws, weights) as WorldScores;

  if (mode === 'display-preflight') {
    const populations = snapshots.map((snap) => snap.population);
    const meanPopulation = mean(populations);
    const aliveTicks = populations.filter((population) => population > 0).length;
    const meanSize = mean(snapshots.map((snap) => snap.meanSize));
    const maxSize = Math.max(...snapshots.map((snap) => snap.maxSize), 0);
    const maxColony = Math.max(...snapshots.map((snap) => snap.largestColony), 0);
    const macroTicks = snapshots.filter((snap) => {
      const largeFraction = snap.population > 0 ? snap.largeOrganisms / snap.population : 0;
      return snap.maxSize > 1.9 || snap.largestColony >= 4 || largeFraction > 0.03;
    }).length;

    parentPort!.postMessage({
      jobId,
      scores,
      avgTickMs,
      aliveTicks,
      finalPopulation: world.entities.count,
      peakPopulation: peakPop,
      meanPopulation,
      meanSize,
      maxSize,
      maxColony,
      macroTicks,
    });
    return;
  }

  // CPU efficiency penalty
  const overload  = Math.max(0, avgTickMs / CPU_TARGET_MS - 1);
  const cpuFactor = 1 / (1 + overload * CPU_PENALTY_W);
  if (cpuFactor < 0.99) scores.total *= cpuFactor;

  parentPort!.postMessage({ jobId, scores, avgTickMs, cpuFactor });
});
