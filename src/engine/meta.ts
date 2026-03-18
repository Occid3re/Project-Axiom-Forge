/**
 * Meta-evolution controller.
 * Evolves world laws across generations by scoring simulated worlds
 * and selecting/mutating the best.
 */

import { World, type WorldConfig, type WorldHistory } from './world';
import { type WorldLaws, randomLaws, mutateLaws, PRNG } from './world-laws';
import { scoreWorld, type WorldScores } from './scoring';

export interface WorldResult {
  id: string;
  parentId: string | null;
  generation: number;
  laws: WorldLaws;
  scores: WorldScores;
  history: WorldHistory;
}

export interface MetaConfig {
  seed: number;
  metaGenerations: number;
  worldsPerGeneration: number;
  worldSteps: number;
  gridSize: number;
  initialEntities: number;
  topK: number;
  scoreWeights: Record<string, number>;
}

export interface GenerationResult {
  generation: number;
  worlds: WorldResult[];
  bestScore: number;
  bestLawsId: string;
  avgScore: number;
}

export interface MetaState {
  generation: number;
  totalGenerations: number;
  currentWorld: number;
  totalWorlds: number;
  generationResults: GenerationResult[];
  running: boolean;
  bestEver: WorldResult | null;
}

export const DEFAULT_META_CONFIG: MetaConfig = {
  seed: 42,
  metaGenerations: 10,
  worldsPerGeneration: 12,
  worldSteps: 500,
  gridSize: 64,
  initialEntities: 40,
  topK: 4,
  scoreWeights: {
    persistence: 1.0,
    diversity: 1.5,
    complexityGrowth: 1.0,
    communication: 2.0,
    envStructure: 1.0,
    adaptability: 1.0,
  },
};

let idCounter = 0;
function newId(): string {
  return (idCounter++).toString(36).padStart(6, '0');
}

export function resetIdCounter(): void {
  idCounter = 0;
}

/**
 * Run the full meta-evolution loop.
 * Yields after each world for progress updates.
 */
export function* runMetaEvolution(config: MetaConfig): Generator<MetaState, MetaState, void> {
  const rng = new PRNG(config.seed);
  resetIdCounter();

  const worldConfig: WorldConfig = {
    gridSize: config.gridSize,
    steps: config.worldSteps,
    initialEntities: config.initialEntities,
  };

  let population: { id: string; parentId: string | null; laws: WorldLaws }[] = [];
  for (let i = 0; i < config.worldsPerGeneration; i++) {
    population.push({ id: newId(), parentId: null, laws: randomLaws(rng) });
  }

  const allGenerationResults: GenerationResult[] = [];
  let bestEver: WorldResult | null = null;

  for (let gen = 0; gen < config.metaGenerations; gen++) {
    const worldResults: WorldResult[] = [];

    for (let w = 0; w < population.length; w++) {
      const { id, parentId, laws } = population[w];
      const worldSeed = rng.int(0, 2147483647);
      const world = new World(laws, worldConfig, worldSeed);
      const history = world.run();
      const scores = scoreWorld(history, laws, config.scoreWeights);

      const result: WorldResult = {
        id,
        parentId,
        generation: gen,
        laws,
        scores,
        history,
      };
      worldResults.push(result);

      if (!bestEver || scores.total > bestEver.scores.total) {
        bestEver = result;
      }

      // Yield progress after each world
      yield {
        generation: gen,
        totalGenerations: config.metaGenerations,
        currentWorld: w + 1,
        totalWorlds: population.length,
        generationResults: allGenerationResults,
        running: true,
        bestEver,
      };
    }

    // Sort by total score
    worldResults.sort((a, b) => b.scores.total - a.scores.total);

    const genResult: GenerationResult = {
      generation: gen,
      worlds: worldResults,
      bestScore: worldResults[0].scores.total,
      bestLawsId: worldResults[0].id,
      avgScore: worldResults.reduce((s, r) => s + r.scores.total, 0) / worldResults.length,
    };
    allGenerationResults.push(genResult);

    // Selection and reproduction
    const survivors = worldResults.slice(0, config.topK);
    const childrenPerSurvivor = Math.floor(config.worldsPerGeneration / config.topK);

    population = [];
    for (const survivor of survivors) {
      // Elitism: keep parent
      population.push({ id: newId(), parentId: survivor.id, laws: survivor.laws });

      // Mutated offspring
      for (let c = 1; c < childrenPerSurvivor; c++) {
        const childLaws = mutateLaws(survivor.laws, rng, 0.1);
        population.push({ id: newId(), parentId: survivor.id, laws: childLaws });
      }
    }

    // Fill remaining slots if needed
    while (population.length < config.worldsPerGeneration) {
      const parentIdx = rng.int(0, survivors.length - 1);
      const childLaws = mutateLaws(survivors[parentIdx].laws, rng, 0.15);
      population.push({ id: newId(), parentId: survivors[parentIdx].id, laws: childLaws });
    }
  }

  const finalState: MetaState = {
    generation: config.metaGenerations - 1,
    totalGenerations: config.metaGenerations,
    currentWorld: config.worldsPerGeneration,
    totalWorlds: config.worldsPerGeneration,
    generationResults: allGenerationResults,
    running: false,
    bestEver,
  };

  return finalState;
}

/**
 * Run a single world with live snapshots for visualization.
 * Yields after each tick.
 */
export function* runSingleWorld(
  laws: WorldLaws,
  config: WorldConfig,
  seed: number
): Generator<World, WorldHistory, void> {
  const world = new World(laws, config, seed);
  const snapshots = [];

  for (let t = 0; t < config.steps; t++) {
    const snap = world.step();
    snapshots.push(snap);
    yield world;
  }

  return {
    snapshots,
    finalPopulation: world.entities.count,
    peakPopulation: Math.max(...snapshots.map(s => s.population)),
    disasterCount: 0,
    postDisasterRecoveries: 0,
  };
}
