/**
 * Eval worker — runs one complete eval world per job, on a dedicated OS thread.
 *
 * Protocol:
 *   main → worker:  { jobId: number, laws: WorldLaws, seed: number }
 *   worker → main:  { jobId: number, scores: WorldScores, avgTickMs: number, cpuFactor: number }
 *
 * Workers are kept alive between jobs — parentPort.on('message') loop.
 * The worker inherits tsx ESM hooks from the parent process (tsx v4).
 */

import { parentPort } from 'worker_threads';
import { World, type WorldHistory } from '../src/engine/world.ts';
import { scoreWorld, type WorldScores } from '../src/engine/scoring.ts';
import type { WorldLaws } from '../src/engine/world-laws.ts';

const EVAL_STEPS       = 800;
const GRID_SIZE        = 64;
const INITIAL_ENTITIES = 45;
const CPU_TARGET_MS    = 0.8;
const CPU_PENALTY_W    = 0.6;

const SCORE_WEIGHTS = {
  persistence:      1.0,
  diversity:        1.5,
  complexityGrowth: 1.5,
  communication:    2.5,
  envStructure:     1.0,
  adaptability:     1.8,
};

parentPort!.on('message', ({ jobId, laws, seed }: { jobId: number; laws: WorldLaws; seed: number }) => {
  const world = new World(
    laws,
    { gridSize: GRID_SIZE, steps: EVAL_STEPS, initialEntities: INITIAL_ENTITIES },
    seed,
  );

  const t0      = Date.now();
  let peakPop   = 0;
  const snapshots = [];

  for (let t = 0; t < EVAL_STEPS; t++) {
    const snap = world.step();
    snapshots.push(snap);
    if (snap.population > peakPop) peakPop = snap.population;
  }

  const wallMs    = Date.now() - t0;
  const avgTickMs = wallMs / EVAL_STEPS;

  const history: WorldHistory = {
    snapshots,
    finalPopulation:       world.entities.count,
    peakPopulation:        peakPop,
    disasterCount:         0,
    postDisasterRecoveries: 0,
  };

  const scores = scoreWorld(history, laws, SCORE_WEIGHTS) as WorldScores;

  // CPU efficiency penalty — same formula as the old inline version
  const overload  = Math.max(0, avgTickMs / CPU_TARGET_MS - 1);
  const cpuFactor = 1 / (1 + overload * CPU_PENALTY_W);
  if (cpuFactor < 0.99) scores.total *= cpuFactor;

  parentPort!.postMessage({ jobId, scores, avgTickMs, cpuFactor });
});
