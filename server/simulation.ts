/**
 * Server-side simulation.
 * Two decoupled loops:
 *   - Eval loop:    meta-evolution runs fast (~500 ticks/sec) to find best laws
 *   - Display loop: replays best world in slow-motion (30 ticks/sec) for viewers
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { World, type WorldSnapshot } from '../src/engine/world.ts';
import { type WorldLaws, PRNG, randomLaws, mutateLaws, crossoverLaws, starterLaws } from '../src/engine/world-laws.ts';
import { scoreWorld, type WorldScores } from '../src/engine/scoring.ts';

// ── State persistence ────────────────────────────────────────────────────────

const STATE_PATH = process.env.STATE_PATH ?? './state.json';
const STATE_VERSION = 2;

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
  gridSize: 64,
  worldSteps: 800,
  initialEntities: 45,
  worldsPerGeneration: 10,
  topK: 2,                  // stronger selection — only top 2 survive
  mutationStrength: 0.08,   // tighter mutations — preserve good laws
  scoreWeights: {
    persistence:      1.0,
    diversity:        1.5,
    complexityGrowth: 1.5,
    communication:    2.5,
    envStructure:     1.0,
    adaptability:     1.8,
  },
  // Stagnation escape: inject random diversity after N generations with <5% improvement
  stagnationThreshold: 30,
  stagnationRandomFraction: 0.25, // replace 25% of population with random laws
  minImprovementRatio: 0.01,      // display world updates if >1% better
  // CPU efficiency pressure — worlds that slow the server score worse.
  // avgTickMs = wall-clock ms per world.step() averaged over worldSteps.
  // penalty kicks in above target; each doubling of cost reduces score by ~(weight/(1+weight)).
  cpuTargetMs:     0.8,  // ms/step budget — comfortable for ~100 entities on 64×64
  cpuPenaltyWeight: 0.6, // at 2× target → score × 0.625; at 4× target → score × 0.357
};

const DISPLAY_CONFIG = {
  gridSize: 256,            // larger grid for display
  initialEntities: 180,
  // Minimum display-world lifetime before it can be replaced by a new best.
  // At 30fps, 240 ticks ≈ 8 seconds — enough to see the world, not so long we're stuck with a bad one.
  minLifetimeTicks: 240,
};

// ── Binary frame packing ────────────────────────────────────────────────────

export function packFrame(world: World, tick: number): ArrayBuffer {
  const vs = world.getVisualState();
  const { gridW: W, gridH: H, entityCount, signalChannels } = vs;
  const channels = Math.min(signalChannels, 3);
  const totalBytes = 20 + W * H + W * H * channels + entityCount * 6;

  const buf = new ArrayBuffer(totalBytes);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);

  view.setUint32(0, 0x41584647, true); // magic "AXFG"
  view.setUint32(4, W, true);
  view.setUint32(8, H, true);
  view.setUint32(12, entityCount, true);
  view.setUint32(16, tick, true);

  let offset = 20;
  for (let i = 0; i < W * H; i++) u8[offset++] = (vs.resources[i] * 255) | 0;
  for (let i = 0; i < W * H; i++) {
    for (let c = 0; c < 3; c++) {
      u8[offset++] = c < channels
        ? Math.min(255, (vs.signals[i * signalChannels + c] * 255) | 0)
        : 0;
    }
  }
  // Write entity arrays sequentially to match the decoder in protocol.ts:
  // all X, then all Y, then all Energy, then all Action, then all Aggression.
  for (let e = 0; e < entityCount; e++) u8[offset++] = vs.entityX[e] & 0xff;
  for (let e = 0; e < entityCount; e++) u8[offset++] = vs.entityY[e] & 0xff;
  for (let e = 0; e < entityCount; e++) u8[offset++] = Math.min(255, (vs.entityEnergy[e] * 255) | 0);
  for (let e = 0; e < entityCount; e++) u8[offset++] = vs.entityAction[e];
  // Pack combined "predator role": predatorDrive (gene 15) + aggression (gene 3)
  // Gives full color spectrum: teal herbivores → red predators
  for (let e = 0; e < entityCount; e++) {
    const predatorDrive = vs.entityGenomes[e * 16 + 15];
    const aggression    = vs.entityGenomes[e * 16 + 3];
    u8[offset++] = Math.min(255, ((predatorDrive * 0.65 + aggression * 0.35) * 255) | 0);
  }
  // Species hue: signal-channel gene + cooperation gene + explore gene
  for (let e = 0; e < entityCount; e++) {
    const sigChan = vs.entityGenomes[e * 16 + 6];
    const coop    = vs.entityGenomes[e * 16 + 12];
    const explore = vs.entityGenomes[e * 16 + 13];
    u8[offset++] = Math.min(255, ((sigChan * 0.5 + coop * 0.3 + explore * 0.2) * 255) | 0);
  }
  return buf;
}

// ── Meta broadcast type ─────────────────────────────────────────────────────

export interface MetaBroadcast {
  generation: number;
  worldIndex: number;
  totalWorlds: number;
  tick: number;
  bestLaws: import('../src/engine/world-laws.ts').WorldLaws | null;
  population: number;
  scores: WorldScores | null;
  bestScore: number;
  generations: Array<{ gen: number; best: number; avg: number }>;
  logEntry: string | null;
  gridSize: number;
  evalSpeed: number;      // ticks/sec in eval loop
  serverMs: number;       // EMA of display world step time (ms) — server load indicator
}

// ── Main controller ─────────────────────────────────────────────────────────

export class SimulationController {
  // Eval state
  private evalRng: PRNG;
  private population: WorldLaws[] = [];
  private generation = 0;
  private worldIndex = 0;
  private evalWorld: World | null = null;
  private evalTick = 0;
  private evalSnapshots: WorldSnapshot[] = [];
  private evalResults: Array<{ laws: WorldLaws; scores: WorldScores }> = [];
  private generationSummaries: Array<{ gen: number; best: number; avg: number }> = [];

  // Display state
  private displayWorld: World | null = null;
  private displayTick = 0;
  private displaySnapshots: WorldSnapshot[] = [];
  private displayScores: WorldScores | null = null;
  private displaySeed = 1;

  // Shared
  bestScore = 0;
  bestLaws: WorldLaws | null = null;
  private pendingLog: string | null = null;

  // Stagnation tracking
  private lastImprovementGen = 0;

  // Perf tracking
  private evalTickCount = 0;
  private evalSpeedSample = 0;
  private lastSpeedCheck = Date.now();

  // CPU efficiency tracking
  private evalWorldStartTime = 0;       // wall-clock start of current eval world
  private displayStepMs = 0;            // EMA of display world step time (ms)

  constructor() {
    this.evalRng = new PRNG(Date.now());
    this.loadState();
    this.initGeneration();
    this.log('Axiom Forge online — searching for emergent worlds');
  }

  // ── State persistence ───────────────────────────────────────────────────

  private loadState() {
    try {
      if (!existsSync(STATE_PATH)) return;
      const s: SavedState = JSON.parse(readFileSync(STATE_PATH, 'utf8'));
      if (s.version !== STATE_VERSION) {
        console.log(`[sim] State file version ${s.version} ≠ ${STATE_VERSION} — starting fresh`);
        return;
      }
      this.generation        = s.generation;
      this.bestScore         = s.bestScore;
      this.bestLaws          = s.bestLaws;
      this.lastImprovementGen = s.lastImprovementGen;
      this.generationSummaries = s.generationSummaries ?? [];
      this.population        = s.population ?? [];
      console.log(`[sim] Restored: gen ${s.generation}, best ${s.bestScore.toFixed(3)}, pop ${this.population.length}`);
    } catch (e) {
      console.error('[sim] Failed to load state (starting fresh):', e);
    }
  }

  private saveState() {
    try {
      const s: SavedState = {
        version:              STATE_VERSION,
        generation:           this.generation,
        bestScore:            this.bestScore,
        bestLaws:             this.bestLaws,
        lastImprovementGen:   this.lastImprovementGen,
        generationSummaries:  this.generationSummaries,
        population:           this.population,
      };
      writeFileSync(STATE_PATH, JSON.stringify(s), 'utf8');
    } catch (e) {
      console.error('[sim] Failed to save state:', e);
    }
  }

  // ── Eval loop (called rapidly via setInterval) ──────────────────────────

  /** Run N eval ticks. Returns true if a new best world was found. */
  evalBatch(n: number): boolean {
    let newBest = false;
    for (let i = 0; i < n; i++) {
      if (this.evalStep()) newBest = true;
    }
    // Track speed
    this.evalTickCount += n;
    const now = Date.now();
    if (now - this.lastSpeedCheck >= 2000) {
      this.evalSpeedSample = Math.round(this.evalTickCount / ((now - this.lastSpeedCheck) / 1000));
      this.evalTickCount = 0;
      this.lastSpeedCheck = now;
    }
    return newBest;
  }

  private evalStep(): boolean {
    const world = this.evalWorld!;
    const snap = world.step();
    this.evalTick++;
    this.evalSnapshots.push(snap);
    if (this.evalSnapshots.length > 300) this.evalSnapshots.shift();

    if (this.evalTick >= EVAL_CONFIG.worldSteps) {
      return this.finishEvalWorld();
    }
    return false;
  }

  private finishEvalWorld(): boolean {
    const world = this.evalWorld!;
    const laws = this.population[this.worldIndex];
    const scores = scoreWorld(
      {
        snapshots: this.evalSnapshots,
        finalPopulation: world.entities.count,
        peakPopulation: Math.max(...this.evalSnapshots.map(s => s.population), 0),
        disasterCount: 0,
        postDisasterRecoveries: 0,
      },
      laws,
      EVAL_CONFIG.scoreWeights,
    );

    // ── CPU efficiency penalty ───────────────────────────────────────────────
    // Measure wall-clock time for all 800 steps — gives ~0.1% precision even
    // with 1ms clock resolution. Worlds that slow the server score worse, so
    // evolution selects for computationally lean physics naturally.
    const wallMs    = Date.now() - this.evalWorldStartTime;
    const avgTickMs = wallMs / EVAL_CONFIG.worldSteps;
    const overload  = Math.max(0, avgTickMs / EVAL_CONFIG.cpuTargetMs - 1);
    const cpuFactor = 1 / (1 + overload * EVAL_CONFIG.cpuPenaltyWeight);
    if (cpuFactor < 0.99) {
      scores.total *= cpuFactor;
      // Log when a world is notably expensive
      if (cpuFactor < 0.8) {
        this.log(`World ${this.worldIndex + 1} CPU heavy: ${avgTickMs.toFixed(2)}ms/step → score ×${cpuFactor.toFixed(2)}`);
      }
    }

    this.evalResults.push({ laws, scores });

    let newBest = false;
    if (scores.total > this.bestScore) {
      const improvement = this.bestScore > 0
        ? (scores.total - this.bestScore) / this.bestScore
        : 1;
      this.bestScore = scores.total;
      this.bestLaws = laws;
      this.lastImprovementGen = this.generation;
      this.log(`New best: ${scores.total.toFixed(3)} · Gen ${this.generation} · World ${this.worldIndex + 1}`);
      this.saveState();
      // Only update display world if improvement is meaningful (>5%) AND current world
      // has been running long enough — prevents rapid flicker on early fast improvements.
      if (improvement >= EVAL_CONFIG.minImprovementRatio && this.displayTick >= DISPLAY_CONFIG.minLifetimeTicks) {
        this.startDisplayWorld(laws);
      }
      newBest = true;
    }

    this.worldIndex++;
    if (this.worldIndex >= EVAL_CONFIG.worldsPerGeneration) {
      this.finishGeneration();
    } else {
      this.startEvalWorld();
    }
    return newBest;
  }

  private finishGeneration() {
    const sorted = [...this.evalResults].sort((a, b) => b.scores.total - a.scores.total);
    const best = sorted[0].scores.total;
    const avg = this.evalResults.reduce((s, r) => s + r.scores.total, 0) / this.evalResults.length;

    this.generationSummaries.push({ gen: this.generation, best, avg });
    if (this.generationSummaries.length > 80) this.generationSummaries.shift();

    this.log(`Gen ${this.generation} · best ${best.toFixed(3)} · avg ${avg.toFixed(3)} · eval ${this.evalSpeedSample} t/s`);

    // Stagnation detection — inject random diversity if stuck
    const gensSinceImproved = this.generation - this.lastImprovementGen;
    const stagnating = gensSinceImproved >= EVAL_CONFIG.stagnationThreshold;
    if (stagnating) {
      this.log(`Stagnant for ${gensSinceImproved} gen — injecting diversity`);
    }

    // Evolve: top K survive, rest are mutated children
    const survivors = sorted.slice(0, EVAL_CONFIG.topK);
    const randomSlots = stagnating
      ? Math.ceil(EVAL_CONFIG.worldsPerGeneration * EVAL_CONFIG.stagnationRandomFraction)
      : 0;
    const childSlots = EVAL_CONFIG.worldsPerGeneration - randomSlots;
    const childrenPer = Math.floor(childSlots / EVAL_CONFIG.topK);

    this.population = [];
    for (const s of survivors) {
      this.population.push(s.laws); // elitism
      for (let c = 1; c < childrenPer; c++) {
        const strength = stagnating
          ? EVAL_CONFIG.mutationStrength * 2
          : EVAL_CONFIG.mutationStrength;
        this.population.push(mutateLaws(s.laws, this.evalRng, strength));
      }
    }
    // Crossover between top-2 survivors for added diversity
    if (survivors.length >= 2) {
      this.population.push(crossoverLaws(survivors[0].laws, survivors[1].laws, this.evalRng));
      this.population.push(mutateLaws(
        crossoverLaws(survivors[0].laws, survivors[1].laws, this.evalRng),
        this.evalRng, EVAL_CONFIG.mutationStrength
      ));
    }
    // Fill random slots with entirely new random laws
    for (let r = 0; r < randomSlots; r++) {
      this.population.push(randomLaws(this.evalRng));
    }
    while (this.population.length < EVAL_CONFIG.worldsPerGeneration) {
      this.population.push(mutateLaws(survivors[0].laws, this.evalRng, EVAL_CONFIG.mutationStrength * 1.5));
    }

    this.generation++;
    this.saveState();
    this.initGeneration();
  }

  private initGeneration() {
    this.evalResults = [];
    this.worldIndex = 0;
    if (this.population.length === 0) {
      // Seed gen-0 with starter-law variants + randoms so early display looks good
      const base = starterLaws();
      this.population = [
        base,
        mutateLaws(base, this.evalRng, 0.10),
        mutateLaws(base, this.evalRng, 0.15),
        mutateLaws(base, this.evalRng, 0.20),
        ...Array.from({ length: EVAL_CONFIG.worldsPerGeneration - 4 }, () =>
          randomLaws(this.evalRng)
        ),
      ];
    }
    this.startEvalWorld();
  }

  private startEvalWorld() {
    const seed = this.evalRng.int(0, 0x7fffffff);
    this.evalWorld = new World(
      this.population[this.worldIndex],
      { gridSize: EVAL_CONFIG.gridSize, steps: EVAL_CONFIG.worldSteps, initialEntities: EVAL_CONFIG.initialEntities },
      seed,
    );
    this.evalTick = 0;
    this.evalSnapshots = [];
    this.evalWorldStartTime = Date.now();
  }

  // ── Display loop (called at 30fps) ─────────────────────────────────────

  startDisplayWorld(laws: WorldLaws) {
    this.displaySeed = this.evalRng.int(0, 0x7fffffff);
    this.displayWorld = new World(
      laws,
      { gridSize: DISPLAY_CONFIG.gridSize, steps: DISPLAY_CONFIG.worldSteps, initialEntities: DISPLAY_CONFIG.initialEntities },
      this.displaySeed,
    );
    this.displayTick = 0;
    this.displaySnapshots = [];
    this.displayScores = null;
  }

  /** Step display world once. Call at ~30fps. Returns broadcast data. */
  displayStep(): { frame: ArrayBuffer; meta: MetaBroadcast } | null {
    if (!this.displayWorld) {
      // Start with hand-tuned starter laws so the display is interesting from tick 1.
      // Meta-evolution will replace this as soon as it finds something better.
      if (this.bestLaws) this.startDisplayWorld(this.bestLaws);
      else this.startDisplayWorld(starterLaws());
    }

    const world = this.displayWorld!;
    const t0 = Date.now();
    const snap = world.step();
    const dt = Date.now() - t0;
    // EMA of display step time — smooth out noise from 1ms clock resolution
    this.displayStepMs = this.displayStepMs * 0.97 + dt * 0.03;
    this.displayTick++;
    this.displaySnapshots.push(snap);
    if (this.displaySnapshots.length > 400) this.displaySnapshots.shift();

    // Score display world periodically
    if (this.displayTick % 60 === 0 && this.displaySnapshots.length > 20 && this.bestLaws) {
      this.displayScores = scoreWorld(
        {
          snapshots: this.displaySnapshots,
          finalPopulation: world.entities.count,
          peakPopulation: Math.max(...this.displaySnapshots.map(s => s.population), 0),
          disasterCount: 0,
          postDisasterRecoveries: 0,
        },
        this.bestLaws,
        EVAL_CONFIG.scoreWeights,
      );
    }

    // Restart display world if extinction — always use latest best laws
    if (this.displayTick > 300 && snap.population === 0) {
      this.log('Display world — extinction. Reseeding with best laws...');
      if (this.bestLaws) this.startDisplayWorld(this.bestLaws);
    }

    // Periodically refresh display world with best laws when it has been
    // running a long time (keeps improvement visible over hours/days)
    if (this.displayTick > 0 && this.displayTick % 9000 === 0 && this.bestLaws) {
      this.log('Display world — periodic refresh with current best laws');
      this.startDisplayWorld(this.bestLaws);
    }

    const frame = packFrame(world, this.displayTick);
    const meta: MetaBroadcast = {
      generation: this.generation,
      worldIndex: this.worldIndex + 1,
      totalWorlds: EVAL_CONFIG.worldsPerGeneration,
      tick: this.displayTick,
      bestLaws: this.bestLaws,
      population: snap.population,
      scores: this.displayScores,
      bestScore: this.bestScore,
      generations: this.generationSummaries,
      logEntry: this.pendingLog,
      gridSize: DISPLAY_CONFIG.gridSize,
      evalSpeed: this.evalSpeedSample,
      serverMs: this.displayStepMs,
    };
    this.pendingLog = null;
    return { frame, meta };
  }

  private log(msg: string) {
    this.pendingLog = msg;
    console.log(`[sim] ${msg}`);
  }
}
