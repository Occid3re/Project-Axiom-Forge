/**
 * Server-side simulation loop.
 * Runs the meta-evolution continuously, streams per-tick state to clients.
 * One canonical simulation — all viewers see the same thing.
 */

import { World, type WorldSnapshot } from '../src/engine/world.ts';
import { type WorldLaws, PRNG, randomLaws, mutateLaws } from '../src/engine/world-laws.ts';
import { scoreWorld, type WorldScores } from '../src/engine/scoring.ts';
import type { EventEmitter } from 'events';

// ---- Config ----------------------------------------------------------------

const SIM_CONFIG = {
  gridSize: 64,
  worldSteps: 800,
  initialEntities: 50,
  worldsPerGeneration: 8,
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

// ---- Binary frame packing --------------------------------------------------
// Layout:
//   [0-3]   magic 0x41584647 "AXFG"
//   [4-7]   gridW (uint32)
//   [8-11]  gridH (uint32)
//   [12-15] entityCount (uint32)
//   [16-19] tick (uint32)
//   [20 .. 20+W*H)           resources uint8  (W*H bytes)
//   [20+W*H .. 20+W*H+W*H*3) signals   uint8  (W*H*3 bytes, 3 channels max)
//   [rest]  entities 5 bytes each: x,y,energy,action,aggression

export function packFrame(world: World): ArrayBuffer {
  const vs = world.getVisualState();
  const { gridW: W, gridH: H, entityCount, signalChannels } = vs;
  const channels = Math.min(signalChannels, 3);
  const entBytes = entityCount * 5;
  const totalBytes = 20 + W * H + W * H * channels + entBytes;

  const buf = new ArrayBuffer(totalBytes);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);

  // Header
  view.setUint32(0, 0x41584647, true); // magic
  view.setUint32(4, W, true);
  view.setUint32(8, H, true);
  view.setUint32(12, entityCount, true);
  view.setUint32(16, vs.entityX.length > 0 ? (world as any).tick : 0, true);

  // Resources
  let offset = 20;
  for (let i = 0; i < W * H; i++) {
    u8[offset++] = Math.min(255, (vs.resources[i] * 255) | 0);
  }

  // Signals (3 channels)
  for (let i = 0; i < W * H; i++) {
    for (let c = 0; c < channels; c++) {
      u8[offset++] = Math.min(255, (vs.signals[i * signalChannels + c] * 255) | 0);
    }
    for (let c = channels; c < 3; c++) {
      u8[offset++] = 0;
    }
  }

  // Entities
  for (let e = 0; e < entityCount; e++) {
    u8[offset++] = vs.entityX[e] & 0xff;
    u8[offset++] = vs.entityY[e] & 0xff;
    u8[offset++] = Math.min(255, (vs.entityEnergy[e] * 255) | 0);
    u8[offset++] = vs.entityAction[e];
    u8[offset++] = Math.min(255, (vs.entityGenomes[e * 16 + 3] * 255) | 0); // aggression
  }

  return buf;
}

// ---- Meta state for JSON broadcast ----------------------------------------

export interface MetaBroadcast {
  generation: number;
  worldIndex: number;
  totalWorlds: number;
  tick: number;
  population: number;
  scores: WorldScores | null;
  bestScore: number;
  generations: Array<{ gen: number; best: number; avg: number }>;
  logEntry: string | null;
  gridSize: number;
}

// ---- Simulation controller -------------------------------------------------

export class SimulationController {
  private rng: PRNG;
  private population: WorldLaws[] = [];
  private generation = 0;
  private worldIndex = 0;
  private currentWorld: World | null = null;
  private currentTick = 0;
  private snapshots: WorldSnapshot[] = [];
  private worldResults: Array<{ laws: WorldLaws; scores: WorldScores }> = [];
  private bestScore = 0;
  private generationSummaries: Array<{ gen: number; best: number; avg: number }> = [];
  private latestScores: WorldScores | null = null;
  private pendingLog: string | null = null;
  private emitter: EventEmitter;

  constructor(emitter: EventEmitter) {
    this.emitter = emitter;
    this.rng = new PRNG(Date.now());
    this.initGeneration();
  }

  private initGeneration() {
    if (this.population.length === 0) {
      // First generation: random laws
      this.population = Array.from({ length: SIM_CONFIG.worldsPerGeneration }, () =>
        randomLaws(this.rng)
      );
      this.log(`Axiom Forge initialized — generation 0 beginning`);
    }
    this.worldIndex = 0;
    this.worldResults = [];
    this.startNextWorld();
  }

  private startNextWorld() {
    const laws = this.population[this.worldIndex];
    const seed = this.rng.int(0, 2147483647);
    this.currentWorld = new World(laws, {
      gridSize: SIM_CONFIG.gridSize,
      steps: SIM_CONFIG.worldSteps,
      initialEntities: SIM_CONFIG.initialEntities,
    }, seed);
    this.currentTick = 0;
    this.snapshots = [];
    this.latestScores = null;
    this.log(`Gen ${this.generation} · World ${this.worldIndex + 1}/${SIM_CONFIG.worldsPerGeneration} — simulating`);
  }

  private finishCurrentWorld() {
    const world = this.currentWorld!;
    const laws = this.population[this.worldIndex];
    const scores = scoreWorld(
      {
        snapshots: this.snapshots,
        finalPopulation: world.entities.count,
        peakPopulation: Math.max(...this.snapshots.map(s => s.population), 0),
        disasterCount: 0,
        postDisasterRecoveries: 0,
      },
      laws,
      SIM_CONFIG.scoreWeights,
    );

    this.worldResults.push({ laws, scores });
    this.latestScores = scores;

    if (scores.total > this.bestScore) {
      this.bestScore = scores.total;
      this.log(`New best: ${scores.total.toFixed(3)} — Gen ${this.generation} · World ${this.worldIndex + 1}`);
    } else {
      this.log(`Gen ${this.generation} · World ${this.worldIndex + 1} scored ${scores.total.toFixed(3)}`);
    }

    this.worldIndex++;

    if (this.worldIndex >= SIM_CONFIG.worldsPerGeneration) {
      this.evolveNextGeneration();
    } else {
      this.startNextWorld();
    }
  }

  private evolveNextGeneration() {
    const sorted = [...this.worldResults].sort((a, b) => b.scores.total - a.scores.total);
    const survivors = sorted.slice(0, SIM_CONFIG.topK);

    const best = survivors[0].scores.total;
    const avg = this.worldResults.reduce((s, r) => s + r.scores.total, 0) / this.worldResults.length;

    this.generationSummaries.push({ gen: this.generation, best, avg });
    if (this.generationSummaries.length > 50) this.generationSummaries.shift();

    this.log(`Generation ${this.generation} complete — best ${best.toFixed(3)} · avg ${avg.toFixed(3)}`);

    // Breed next generation
    const childrenPerSurvivor = Math.floor(SIM_CONFIG.worldsPerGeneration / SIM_CONFIG.topK);
    this.population = [];
    for (const survivor of survivors) {
      this.population.push(survivor.laws); // elitism
      for (let c = 1; c < childrenPerSurvivor; c++) {
        this.population.push(mutateLaws(survivor.laws, this.rng, 0.1));
      }
    }
    while (this.population.length < SIM_CONFIG.worldsPerGeneration) {
      this.population.push(mutateLaws(survivors[0].laws, this.rng, 0.15));
    }

    this.generation++;
    this.initGeneration();
  }

  private log(msg: string) {
    this.pendingLog = msg;
    console.log(`[sim] ${msg}`);
  }

  /** Advance one tick. Call from setImmediate loop. */
  tick(): { frame: ArrayBuffer; meta: MetaBroadcast } | null {
    const world = this.currentWorld;
    if (!world) return null;

    const snap = world.step();
    this.currentTick++;
    this.snapshots.push(snap);
    if (this.snapshots.length > 400) this.snapshots.shift();

    // Lazily score every 50 ticks for the broadcast
    if (this.currentTick % 50 === 0 && this.snapshots.length > 20) {
      this.latestScores = scoreWorld(
        {
          snapshots: this.snapshots,
          finalPopulation: world.entities.count,
          peakPopulation: Math.max(...this.snapshots.map(s => s.population), 0),
          disasterCount: 0,
          postDisasterRecoveries: 0,
        },
        this.population[this.worldIndex],
        SIM_CONFIG.scoreWeights,
      );
    }

    const frame = packFrame(world);
    const meta: MetaBroadcast = {
      generation: this.generation,
      worldIndex: this.worldIndex + 1,
      totalWorlds: SIM_CONFIG.worldsPerGeneration,
      tick: this.currentTick,
      population: snap.population,
      scores: this.latestScores,
      bestScore: this.bestScore,
      generations: this.generationSummaries,
      logEntry: this.pendingLog,
      gridSize: SIM_CONFIG.gridSize,
    };
    this.pendingLog = null;

    if (this.currentTick >= SIM_CONFIG.worldSteps) {
      this.finishCurrentWorld();
    }

    return { frame, meta };
  }
}
