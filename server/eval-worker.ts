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

const DEFAULT_EVAL_STEPS    = 1600;
const GRID_SIZE             = 64;
const INITIAL_ENTITIES      = 45;
const CPU_TARGET_MS         = 0.8;
const CPU_PENALTY_W         = 0.6;

const DEFAULT_SCORE_WEIGHTS = {
  persistence:           0.5,
  diversity:             1.5,
  complexityGrowth:      1.0,
  communication:         2.0,
  envStructure:          0.5,
  adaptability:          1.0,
  speciation:            3.0,
  interactions:          3.5,
  spatialStructure:      1.5,
  populationDynamics:    1.5,
  stigmergicUse:         2.5,
  socialDifferentiation: 3.0,
};

interface JobMessage {
  jobId: number;
  laws: WorldLaws;
  seed: number;
  evalSteps?: number;
  scoreWeights?: Record<string, number>;
}

parentPort!.on('message', ({ jobId, laws, seed, evalSteps, scoreWeights }: JobMessage) => {
  const steps = evalSteps ?? DEFAULT_EVAL_STEPS;
  const weights = scoreWeights ?? DEFAULT_SCORE_WEIGHTS;

  const world = new World(
    laws,
    { gridSize: GRID_SIZE, steps: steps, initialEntities: INITIAL_ENTITIES },
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

  // CPU efficiency penalty
  const overload  = Math.max(0, avgTickMs / CPU_TARGET_MS - 1);
  const cpuFactor = 1 / (1 + overload * CPU_PENALTY_W);
  if (cpuFactor < 0.99) scores.total *= cpuFactor;

  parentPort!.postMessage({ jobId, scores, avgTickMs, cpuFactor });
});
