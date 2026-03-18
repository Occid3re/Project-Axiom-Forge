/**
 * Server-side simulation.
 * Two decoupled loops:
 *   - Eval loop:    meta-evolution runs on N worker threads in parallel to find best laws
 *   - Display loop: replays best world in slow-motion (30fps) for viewers
 *
 * Worker count adapts to available CPUs automatically:
 *   numWorkers = os.cpus().length  (e.g. 2 CPUs → 2 workers, 4 CPUs → 4 workers)
 * Each generation dispatches all worldsPerGeneration eval worlds in parallel;
 * generation time ≈ ceil(worldsPerGeneration / numWorkers) × singleWorldTime.
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { Worker } from 'worker_threads';
import { cpus } from 'os';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { World, type WorldSnapshot } from '../src/engine/world.ts';
import { type WorldLaws, PRNG, randomLaws, mutateLaws, crossoverLaws, starterLaws } from '../src/engine/world-laws.ts';
import { scoreWorld, type WorldScores } from '../src/engine/scoring.ts';
import { GENOME_LENGTH, NN_HIDDEN, NN_OUTPUTS, NN_W1_SIZE, GLYPH_CHANNELS } from '../src/engine/constants.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);

// ── State persistence ────────────────────────────────────────────────────────

const STATE_PATH    = process.env.STATE_PATH ?? './state.json';
const STATE_VERSION = 8;

interface SavedState {
  version: number;
  generation: number;
  bestScore: number;
  bestLaws: WorldLaws | null;
  lastImprovementGen: number;
  generationSummaries: Array<{ gen: number; best: number; avg: number }>;
  population: WorldLaws[];
}

// ── Config ──────────────────────────────────────────────────────────────────

const EVAL_CONFIG = {
  gridSize:            64,
  worldSteps:        1600,   // doubled: more entity generations per eval
  initialEntities:     45,
  worldsPerGeneration: 12,   // slightly more candidates for exploration
  topK:                 3,   // 3 survivors for more genetic diversity
  mutationStrength:  0.08,
  scoreWeights: {
    persistence:      0.5,   // halved: survival is necessary but shouldn't dominate
    diversity:        1.5,   // increased: genome divergence is key to interesting worlds
    complexityGrowth: 1.0,   // slight decrease
    communication:    2.0,   // keep: lagged correlation is already hard
    envStructure:     0.5,   // halved: too easy to max out
    adaptability:     1.0,   // reduced: stop rewarding static populations
    speciation:       3.0,   // doubled: we WANT visible species
    interactions:     3.5,   // big increase: predator-prey is what makes watching fun
    spatialStructure: 1.5,   // reward clustering/territories, penalize uniform soup
    populationDynamics: 1.5, // reward oscillation, penalize flat population lines
    stigmergicUse: 2.5,     // reward balanced deposit/absorb glyph usage
    socialDifferentiation: 3.0, // reward kin-selective behavior
  },
  // Escalating stagnation tiers
  stagnationMild:           30,   // 25% random, 2× mutation
  stagnationAggressive:    100,   // 50% random, 4× mutation
  stagnationReset:         200,   // full population reset from scratch
  stagnationRandomFraction: 0.25,
  minImprovementRatio:      0.01,
  cpuTargetMs:              0.8,
  cpuPenaltyWeight:         0.6,
};

const DISPLAY_CONFIG = {
  gridSize:        256,
  initialEntities: 180,
  minLifetimeTicks: 240,
};

// ── Worker pool ──────────────────────────────────────────────────────────────

interface WorkerJob {
  resolve: (result: EvalWorkerResult) => void;
  reject:  (err: Error) => void;
}

interface EvalWorkerResult {
  scores:     WorldScores;
  avgTickMs:  number;
  cpuFactor:  number;
}

class WorkerPool {
  readonly size: number;
  private workers: Worker[];
  private pending = new Map<number, WorkerJob>();
  private jobId   = 0;

  constructor(numWorkers: number) {
    this.size    = numWorkers;
    const ext    = __filename.endsWith('.mjs') ? '.mjs' : '.ts';
    const script = resolve(__dirname, `./eval-worker${ext}`);
    this.workers = Array.from({ length: numWorkers }, () => {
      const w = new Worker(script);
      w.on('message', ({ jobId, ...result }: { jobId: number } & EvalWorkerResult) => {
        const job = this.pending.get(jobId);
        if (job) { this.pending.delete(jobId); job.resolve(result); }
      });
      w.on('error', (err) => console.error('[worker] error:', err));
      return w;
    });
  }

  /** Dispatch a job to a specific worker (caller chooses slot for round-robin). */
  submit(workerIdx: number, data: object): Promise<EvalWorkerResult> {
    return new Promise((resolve, reject) => {
      const id = this.jobId++;
      this.pending.set(id, { resolve, reject });
      this.workers[workerIdx % this.workers.length].postMessage({ jobId: id, ...data });
    });
  }

  terminate() { this.workers.forEach(w => w.terminate()); }
}

// ── Binary frame packing ────────────────────────────────────────────────────
//
// Genome is now 80 MLP weights (real-valued floats).
// Visualisation bytes are derived from W2 column means (hidden→action weights),
// sigmoid-mapped to [0, 1]:
//   aggression255: W2 ATTACK column (a=5)  → how strongly this network attacks
//   species255:    W2 SIGNAL column (a=4) + W2 EAT column (a=2) → behaviour fingerprint

export function packFrame(world: World, tick: number): ArrayBuffer {
  const vs = world.getVisualState();
  const { gridW: W, gridH: H, entityCount, signalChannels } = vs;
  const channels   = Math.min(signalChannels, 3);
  // Layout: header(20) + resources(WH) + signals(WH×3) + poison(WH) + glyphs(WH) + entities(N×8)
  const totalBytes = 20 + W * H + W * H * 3 + W * H + W * H + entityCount * 8;

  const buf  = new ArrayBuffer(totalBytes);
  const view = new DataView(buf);
  const u8   = new Uint8Array(buf);

  view.setUint32(0,  0x41584647, true); // magic "AXFG"
  view.setUint32(4,  W,    true);
  view.setUint32(8,  H,    true);
  view.setUint32(12, entityCount, true);
  view.setUint32(16, tick,  true);

  let offset = 20;
  for (let i = 0; i < W * H; i++) u8[offset++] = (vs.resources[i] * 255) | 0;
  for (let i = 0; i < W * H; i++) {
    for (let c = 0; c < 3; c++) {
      u8[offset++] = c < channels
        ? Math.min(255, (vs.signals[i * signalChannels + c] * 255) | 0)
        : 0;
    }
  }
  // Poison grid
  for (let i = 0; i < W * H; i++) u8[offset++] = Math.min(255, (vs.poison[i] * 255) | 0);
  // Glyph grid: encode magnitude of 4-channel glyph vector as uint8
  for (let i = 0; i < W * H; i++) {
    let mag = 0;
    const base = i * GLYPH_CHANNELS;
    for (let c = 0; c < GLYPH_CHANNELS; c++) {
      mag += vs.glyphs[base + c] * vs.glyphs[base + c];
    }
    u8[offset++] = Math.min(255, (Math.sqrt(mag) * 128) | 0);
  }
  // X, Y, Energy, Action arrays
  for (let e = 0; e < entityCount; e++) u8[offset++] = vs.entityX[e] & 0xff;
  for (let e = 0; e < entityCount; e++) u8[offset++] = vs.entityY[e] & 0xff;
  for (let e = 0; e < entityCount; e++) u8[offset++] = Math.min(255, (vs.entityEnergy[e] * 255) | 0);
  for (let e = 0; e < entityCount; e++) u8[offset++] = vs.entityAction[e];

  // Aggression byte: mean of W2 ATTACK column (a=5), sigmoid → hunt tendency
  for (let e = 0; e < entityCount; e++) {
    let attackSum = 0;
    for (let j = 0; j < NN_HIDDEN; j++) {
      attackSum += vs.entityGenomes[e * GENOME_LENGTH + NN_W1_SIZE + j * NN_OUTPUTS + 5];
    }
    u8[offset++] = Math.round(255 / (1 + Math.exp(-attackSum * 0.4)));
  }

  // Species hue byte: W2 SIGNAL column (a=4) + W2 EAT column (a=2) → behaviour fingerprint
  for (let e = 0; e < entityCount; e++) {
    let sigSum = 0, eatSum = 0;
    for (let j = 0; j < NN_HIDDEN; j++) {
      sigSum += vs.entityGenomes[e * GENOME_LENGTH + NN_W1_SIZE + j * NN_OUTPUTS + 4]; // SIGNAL=4
      eatSum += vs.entityGenomes[e * GENOME_LENGTH + NN_W1_SIZE + j * NN_OUTPUTS + 2]; // EAT=2
    }
    const sigTend = 1 / (1 + Math.exp(-sigSum * 0.3));
    const eatTend = 1 / (1 + Math.exp(-eatSum * 0.3));
    u8[offset++] = Math.round((sigTend * 0.6 + eatTend * 0.4) * 255);
  }

  // Genome complexity byte: standard deviation of all genome weights → 0-255
  for (let e = 0; e < entityCount; e++) {
    const gOff = e * GENOME_LENGTH;
    let sum = 0, sum2 = 0;
    for (let g = 0; g < GENOME_LENGTH; g++) {
      const w = vs.entityGenomes[gOff + g];
      sum += w; sum2 += w * w;
    }
    const mean = sum / GENOME_LENGTH;
    const variance = sum2 / GENOME_LENGTH - mean * mean;
    const std = Math.sqrt(Math.max(0, variance));
    u8[offset++] = Math.round(255 / (1 + Math.exp(-(std - 1.2) * 2.5)));
  }

  // Motility byte: W2 MOVE column (a=1) mean, sigmoid → movement drive
  for (let e = 0; e < entityCount; e++) {
    let moveSum = 0;
    for (let j = 0; j < NN_HIDDEN; j++) {
      moveSum += vs.entityGenomes[e * GENOME_LENGTH + NN_W1_SIZE + j * NN_OUTPUTS + 1]; // MOVE=1
    }
    u8[offset++] = Math.round(255 / (1 + Math.exp(-moveSum * 0.3)));
  }

  return buf;
}

// ── Meta broadcast type ─────────────────────────────────────────────────────

export interface MetaBroadcast {
  generation:   number;
  worldIndex:   number;  // completed eval worlds in this generation (0–worldsPerGeneration)
  totalWorlds:  number;
  tick:         number;
  bestLaws:     import('../src/engine/world-laws.ts').WorldLaws | null;
  population:   number;
  scores:       WorldScores | null;
  bestScore:    number;
  generations:  Array<{ gen: number; best: number; avg: number }>;
  logEntry:     string | null;
  gridSize:     number;
  evalSpeed:    number;       // effective eval ticks/sec (across all workers)
  serverMs:     number;       // EMA of display world step time (ms)
  serverPressure: number;     // 0-2: how much the world is punishing creatures for server load
  sampleGenome?: number[];    // 180 MLP weights of the most-energetic display entity
  displaySeed:  number;       // changes every time startDisplayWorld() is called
}

// ── Main controller ─────────────────────────────────────────────────────────

export class SimulationController {
  // Eval state
  private evalRng:     PRNG;
  private population:  WorldLaws[] = [];
  private generation   = 0;
  private evalResults: Array<{ laws: WorldLaws; scores: WorldScores }> = [];
  private generationSummaries: Array<{ gen: number; best: number; avg: number }> = [];
  private completedWorldsThisGen = 0;

  // Display state
  private displayWorld:     World | null    = null;
  private displayTick       = 0;
  private displaySnapshots: WorldSnapshot[] = [];
  private displayScores:    WorldScores | null = null;
  private displaySeed       = 1;

  // Shared
  bestScore = 0;
  bestLaws: WorldLaws | null = null;
  private pendingLog: string | null = null;

  // Stagnation tracking
  private lastImprovementGen = 0;

  // Perf
  private evalSpeedSample = 0;
  private displayStepMs   = 0;

  // Worker pool
  private workerPool: WorkerPool;

  constructor() {
    this.evalRng = new PRNG(Date.now());

    const numWorkers = Math.max(1, cpus().length);
    this.workerPool  = new WorkerPool(numWorkers);

    this.loadState();
    this.log(`Axiom Forge online — ${numWorkers} eval worker(s), ${EVAL_CONFIG.worldsPerGeneration} worlds/gen`);
  }

  // ── State persistence ───────────────────────────────────────────────────

  private loadState() {
    try {
      if (!existsSync(STATE_PATH)) return;
      const s: SavedState = JSON.parse(readFileSync(STATE_PATH, 'utf8'));
      if (s.version !== STATE_VERSION) {
        console.log(`[sim] State version ${s.version} ≠ ${STATE_VERSION} — starting fresh`);
        return;
      }
      this.generation           = s.generation;
      this.bestScore            = s.bestScore;
      this.bestLaws             = s.bestLaws;
      this.lastImprovementGen   = s.lastImprovementGen;
      this.generationSummaries  = s.generationSummaries ?? [];
      this.population           = s.population ?? [];
      console.log(`[sim] Restored: gen ${s.generation}, best ${s.bestScore.toFixed(3)}, pop ${this.population.length}`);
    } catch (e) {
      console.error('[sim] Failed to load state (starting fresh):', e);
    }
  }

  private saveState() {
    try {
      const s: SavedState = {
        version:             STATE_VERSION,
        generation:          this.generation,
        bestScore:           this.bestScore,
        bestLaws:            this.bestLaws,
        lastImprovementGen:  this.lastImprovementGen,
        generationSummaries: this.generationSummaries,
        population:          this.population,
      };
      writeFileSync(STATE_PATH, JSON.stringify(s), 'utf8');
    } catch (e) {
      console.error('[sim] Failed to save state:', e);
    }
  }

  // ── Async eval loop ─────────────────────────────────────────────────────

  /**
   * Start the async eval loop — runs forever, dispatching eval worlds to worker threads.
   * Call once from index.ts; the returned promise never resolves.
   */
  async startEvalLoop(): Promise<never> {
    // Seed initial population if not restored from state
    if (this.population.length === 0) this.seedPopulation();
    // Start display with best known laws immediately
    if (!this.displayWorld) {
      this.startDisplayWorld(this.bestLaws ?? starterLaws());
    }

    while (true) {
      const genStart = Date.now();
      await this.runGeneration();
      const genMs = Date.now() - genStart;
      // Effective ticks/sec across all workers (workers ran in parallel)
      this.evalSpeedSample = Math.round(
        (EVAL_CONFIG.worldsPerGeneration * EVAL_CONFIG.worldSteps) / (genMs / 1000),
      );
    }
  }

  private seedPopulation() {
    const base = starterLaws();
    this.population = [
      base,
      mutateLaws(base, this.evalRng, 0.10),
      mutateLaws(base, this.evalRng, 0.15),
      mutateLaws(base, this.evalRng, 0.20),
      ...Array.from({ length: EVAL_CONFIG.worldsPerGeneration - 4 }, () =>
        randomLaws(this.evalRng),
      ),
    ];
  }

  private async runGeneration(): Promise<void> {
    this.evalResults           = [];
    this.completedWorldsThisGen = 0;

    // Generate all seeds upfront (synchronous, on main thread)
    const seeds = this.population.map(() => this.evalRng.int(0, 0x7fffffff));

    // Dispatch all worlds in parallel — each resolves when its worker is done
    await Promise.all(
      this.population.map(async (laws, i) => {
        const result = await this.workerPool.submit(i, {
          laws, seed: seeds[i],
          evalSteps: EVAL_CONFIG.worldSteps,
          scoreWeights: EVAL_CONFIG.scoreWeights,
        });
        this.completedWorldsThisGen++;

        if (result.cpuFactor < 0.8) {
          this.log(`World ${i + 1} CPU heavy: ×${result.cpuFactor.toFixed(2)} penalty`);
        }

        // Check for new best (JS single-threaded: no race condition)
        if (result.scores.total > this.bestScore) {
          const improvement = this.bestScore > 0
            ? (result.scores.total - this.bestScore) / this.bestScore
            : 1;
          this.bestScore         = result.scores.total;
          this.bestLaws          = laws;
          this.lastImprovementGen = this.generation;
          this.log(`New best: ${result.scores.total.toFixed(3)} · Gen ${this.generation} · World ${i + 1}`);
          this.saveState();
          if (improvement >= EVAL_CONFIG.minImprovementRatio &&
              this.displayTick >= DISPLAY_CONFIG.minLifetimeTicks) {
            this.startDisplayWorld(laws);
          }
        }

        this.evalResults.push({ laws, scores: result.scores });
      }),
    );

    this.finishGeneration();
  }

  private finishGeneration() {
    const sorted = [...this.evalResults].sort((a, b) => b.scores.total - a.scores.total);
    const best   = sorted[0].scores.total;
    const avg    = this.evalResults.reduce((s, r) => s + r.scores.total, 0) / this.evalResults.length;

    this.generationSummaries.push({ gen: this.generation, best, avg });
    if (this.generationSummaries.length > 80) this.generationSummaries.shift();

    this.log(`Gen ${this.generation} · best ${best.toFixed(3)} · avg ${avg.toFixed(3)} · ${this.evalSpeedSample} t/s`);

    const gensSinceImproved = this.generation - this.lastImprovementGen;
    const N = EVAL_CONFIG.worldsPerGeneration;

    // ── Escalating stagnation tiers ──────────────────────────────────────────
    if (gensSinceImproved >= EVAL_CONFIG.stagnationReset) {
      // TIER 3: Full population reset — local optimum is exhausted
      this.log(`Stagnant ${gensSinceImproved} gen — HARD RESET (keeping alltime best)`);
      this.population = [];
      if (this.bestLaws) {
        this.population.push(this.bestLaws);
        this.population.push(mutateLaws(this.bestLaws, this.evalRng, 0.20));
        this.population.push(mutateLaws(this.bestLaws, this.evalRng, 0.30));
      }
      // Add starter variants for known-good seeds
      this.population.push(starterLaws());
      this.population.push(mutateLaws(starterLaws(), this.evalRng, 0.15));
      // Fill rest with fully random exploration
      while (this.population.length < N) {
        this.population.push(randomLaws(this.evalRng));
      }
      this.lastImprovementGen = this.generation; // reset stagnation counter
    } else if (gensSinceImproved >= EVAL_CONFIG.stagnationAggressive) {
      // TIER 2: Aggressive — 50% random, 4× mutation on survivors
      this.log(`Stagnant ${gensSinceImproved} gen — aggressive diversity injection`);
      const survivors = sorted.slice(0, EVAL_CONFIG.topK);
      const randomSlots = Math.ceil(N * 0.5);
      this.population = [];
      for (const s of survivors) {
        this.population.push(s.laws);
        this.population.push(mutateLaws(s.laws, this.evalRng, EVAL_CONFIG.mutationStrength * 4));
      }
      if (survivors.length >= 2) {
        this.population.push(crossoverLaws(survivors[0].laws, survivors[1].laws, this.evalRng));
      }
      while (this.population.length < N - randomSlots) {
        this.population.push(mutateLaws(survivors[0].laws, this.evalRng, EVAL_CONFIG.mutationStrength * 3));
      }
      for (let r = 0; r < randomSlots && this.population.length < N; r++) {
        this.population.push(randomLaws(this.evalRng));
      }
    } else if (gensSinceImproved >= EVAL_CONFIG.stagnationMild) {
      // TIER 1: Mild — 25% random, 2× mutation (original behavior)
      this.log(`Stagnant ${gensSinceImproved} gen — mild diversity injection`);
      const survivors = sorted.slice(0, EVAL_CONFIG.topK);
      const randomSlots = Math.ceil(N * EVAL_CONFIG.stagnationRandomFraction);
      this.population = [];
      for (const s of survivors) {
        this.population.push(s.laws);
      }
      // Crossovers
      if (survivors.length >= 2) {
        this.population.push(crossoverLaws(survivors[0].laws, survivors[1].laws, this.evalRng));
        this.population.push(mutateLaws(
          crossoverLaws(survivors[0].laws, survivors[1].laws, this.evalRng),
          this.evalRng, EVAL_CONFIG.mutationStrength,
        ));
      }
      // Mutated children
      while (this.population.length < N - randomSlots) {
        const parent = survivors[this.population.length % survivors.length].laws;
        this.population.push(mutateLaws(parent, this.evalRng, EVAL_CONFIG.mutationStrength * 2));
      }
      for (let r = 0; r < randomSlots && this.population.length < N; r++) {
        this.population.push(randomLaws(this.evalRng));
      }
    } else {
      // Normal generation — no stagnation
      const survivors = sorted.slice(0, EVAL_CONFIG.topK);
      const childSlots = N - EVAL_CONFIG.topK;
      const childrenPer = Math.floor(childSlots / EVAL_CONFIG.topK);

      this.population = [];
      for (const s of survivors) {
        this.population.push(s.laws); // elitism
        for (let c = 0; c < childrenPer; c++) {
          this.population.push(mutateLaws(s.laws, this.evalRng, EVAL_CONFIG.mutationStrength));
        }
      }
      if (survivors.length >= 2) {
        this.population.push(crossoverLaws(survivors[0].laws, survivors[1].laws, this.evalRng));
        this.population.push(mutateLaws(
          crossoverLaws(survivors[0].laws, survivors[1].laws, this.evalRng),
          this.evalRng, EVAL_CONFIG.mutationStrength,
        ));
      }
      while (this.population.length < N) {
        this.population.push(mutateLaws(survivors[0].laws, this.evalRng, EVAL_CONFIG.mutationStrength * 1.5));
      }
    }

    this.population = this.population.slice(0, N);
    this.generation++;
    this.saveState();
  }

  // ── Display loop (called at 30fps) ─────────────────────────────────────

  startDisplayWorld(laws: WorldLaws) {
    this.displaySeed  = this.evalRng.int(0, 0x7fffffff);
    this.displayWorld = new World(
      laws,
      { gridSize: DISPLAY_CONFIG.gridSize, steps: 999999, initialEntities: DISPLAY_CONFIG.initialEntities },
      this.displaySeed,
    );
    this.displayTick      = 0;
    this.displaySnapshots = [];
    this.displayScores    = null;
  }

  /** Step display world once. Call at ~30fps. Returns broadcast data. */
  displayStep(): { frame: ArrayBuffer; meta: MetaBroadcast } | null {
    if (!this.displayWorld) {
      this.startDisplayWorld(this.bestLaws ?? starterLaws());
    }

    const world = this.displayWorld!;

    // Server pressure: measured from display tick time vs target.
    // At 33ms target (30fps), a 60ms tick → pressure ≈ 0.8.
    // Pressure makes the world harsher: less resources, persistent poison, more disasters.
    const DISPLAY_TARGET_MS = 33;
    const pressure = Math.min(2, Math.max(0, (this.displayStepMs - DISPLAY_TARGET_MS) / DISPLAY_TARGET_MS));
    world.serverPressure = pressure;

    const t0    = Date.now();
    const snap  = world.step();
    const dt    = Date.now() - t0;
    this.displayStepMs = this.displayStepMs * 0.97 + dt * 0.03;
    this.displayTick++;
    this.displaySnapshots.push(snap);
    if (this.displaySnapshots.length > 400) this.displaySnapshots.shift();

    if (this.displayTick % 60 === 0 && this.displaySnapshots.length > 20 && this.bestLaws) {
      this.displayScores = scoreWorld(
        {
          snapshots:             this.displaySnapshots,
          finalPopulation:       world.entities.count,
          peakPopulation:        Math.max(...this.displaySnapshots.map(s => s.population), 0),
          disasterCount:         0,
          postDisasterRecoveries: 0,
        },
        this.bestLaws,
        EVAL_CONFIG.scoreWeights,
      );
    }

    if (this.displayTick > 300 && snap.population === 0) {
      this.log('Display world — extinction. Reseeding with best laws...');
      if (this.bestLaws) this.startDisplayWorld(this.bestLaws);
    }

    if (this.displayTick > 0 && this.displayTick % 9000 === 0 && this.bestLaws) {
      this.log('Display world — periodic refresh with current best laws');
      this.startDisplayWorld(this.bestLaws);
    }

    // Grab genome of the most-energetic entity for neural-net visualisation
    let sampleGenome: number[] | undefined;
    if (world.entities.count > 0) {
      let bestEnergy = -1, bestIdx = 0;
      for (let i = 0; i < world.entities.count; i++) {
        if (world.entities.energy[i] > bestEnergy) {
          bestEnergy = world.entities.energy[i];
          bestIdx = i;
        }
      }
      sampleGenome = Array.from(world.entities.getGenome(bestIdx));
    }

    const frame: ArrayBuffer = packFrame(world, this.displayTick);
    const meta: MetaBroadcast = {
      generation:  this.generation,
      worldIndex:  this.completedWorldsThisGen,
      totalWorlds: EVAL_CONFIG.worldsPerGeneration,
      tick:        this.displayTick,
      bestLaws:    this.bestLaws,
      population:  snap.population,
      scores:      this.displayScores,
      bestScore:   this.bestScore,
      generations: this.generationSummaries,
      logEntry:    this.pendingLog,
      gridSize:    DISPLAY_CONFIG.gridSize,
      evalSpeed:    this.evalSpeedSample,
      serverMs:     this.displayStepMs,
      serverPressure: pressure,
      sampleGenome,
      displaySeed:  this.displaySeed,
    };
    this.pendingLog = null;
    return { frame, meta };
  }

  private log(msg: string) {
    this.pendingLog = msg;
    console.log(`[sim] ${msg}`);
  }
}
