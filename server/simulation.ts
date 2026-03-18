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
import { type WorldLaws, PRNG, randomLaws, mutateLaws, crossoverLaws } from '../src/engine/world-laws.ts';
import { scoreWorld, type WorldScores } from '../src/engine/scoring.ts';
import { GENOME_LENGTH, NN_HIDDEN, NN_OUTPUTS, NN_W1_SIZE, GLYPH_CHANNELS } from '../src/engine/constants.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);

// ── State persistence ────────────────────────────────────────────────────────

const STATE_PATH    = process.env.STATE_PATH ?? './state.json';
const STATE_VERSION = 10;

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
  worldSteps:        3200,   // 270-weight NN + directional movement needs more ticks
  initialEntities:     50,   // density = 50/(64²) ≈ 0.012, matches display
  worldsPerGeneration: 12,
  topK:                 3,
  mutationStrength:  0.08,
  scoreWeights: {
    persistence:      0.5,   // survival is necessary but shouldn't dominate
    diversity:        1.5,   // genome divergence is key to interesting worlds
    complexityGrowth: 1.0,
    communication:    2.0,   // lagged correlation is already hard to fake
    envStructure:     0.5,   // variance of resource coverage over time
    adaptability:     1.0,
    speciation:       3.0,   // we WANT visible species clusters
    interactions:     3.5,   // predator-prey makes watching fun
    spatialStructure: 1.5,   // reward territories / clustering
    populationDynamics: 1.5, // reward oscillation, penalize flat lines
    stigmergicUse:    2.5,   // reward balanced deposit/absorb
    socialDifferentiation: 3.0, // reward kin-selective behaviour
  },
  // Escalating stagnation tiers
  stagnationMild:           30,   // 25% random, 2× mutation
  stagnationAggressive:    100,   // 50% random, 4× mutation
  stagnationReset:         150,   // full reset sooner — score plateau is deep
  stagnationRandomFraction: 0.25,
  minImprovementRatio:    0.005,  // count smaller improvements as real progress
  cpuTargetMs:              6.0,  // 270-weight NN + directional scan is expensive; don't penalize rich worlds
  cpuPenaltyWeight:         0.05, // near-zero: complex worlds should NOT be penalized for being interesting
};

const DISPLAY_CONFIG = {
  gridSize:        256,
  initialEntities: 400,  // fix density mismatch: eval is 50/(64²)=0.012; 400/(256²)=0.006 ≈ half
  minLifetimeTicks: 240,
  fieldFrameInterval: 6, // 30fps entities, 5fps field refresh
  fieldDownsample: 2,
  fieldKeyframeInterval: 60, // full field refresh every ~2s for resync
  fieldTileSize: 8,
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
//   aggression255: W2 ATTACK column (a=8)  → how strongly this network attacks
//   species255:    W2 SIGNAL column (a=7) + W2 EAT column (a=5) → behaviour fingerprint

const FIELD_PACKET_HEADER_BYTES = 32;
const FIELD_PACKET_KIND_KEYFRAME = 0;
const FIELD_PACKET_KIND_DELTA = 1;
const FIELD_RESOURCE_DELTA_THRESHOLD = 3;
const FIELD_SIGNAL_DELTA_THRESHOLD = 4;
const FIELD_POISON_DELTA_THRESHOLD = 3;
const FIELD_GLYPH_DELTA_THRESHOLD = 3;
const FIELD_PLANE_RESOURCES = 1;
const FIELD_PLANE_SIGNALS = 2;
const FIELD_PLANE_POISON = 4;
const FIELD_PLANE_GLYPHS = 8;

interface PackedFieldState {
  gridW: number;
  gridH: number;
  step: number;
  outW: number;
  outH: number;
  tick: number;
  resources: Uint8Array;
  signals: Uint8Array;
  poison: Uint8Array;
  glyphs: Uint8Array;
}

interface FieldTile {
  tileX: number;
  tileY: number;
  tileW: number;
  tileH: number;
  planeMask: number;
}

function samplePackedFieldState(world: World, tick: number): PackedFieldState {
  const vs = world.getVisualState();
  const { gridW: W, gridH: H, signalChannels } = vs;
  const channels = Math.min(signalChannels, 3);
  const step = DISPLAY_CONFIG.fieldDownsample;
  const outW = Math.max(1, Math.floor(W / step));
  const outH = Math.max(1, Math.floor(H / step));
  const outCells = outW * outH;
  const resources = new Uint8Array(outCells);
  const signals = new Uint8Array(outCells * 3);
  const poison = new Uint8Array(outCells);
  const glyphs = new Uint8Array(outCells);

  let offset = 0;
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let sum = 0;
      for (let dy = 0; dy < step; dy++) {
        const sy = oy * step + dy;
        for (let dx = 0; dx < step; dx++) {
          const sx = ox * step + dx;
          sum += vs.resources[sy * W + sx];
        }
      }
      resources[offset++] = Math.min(255, ((sum / (step * step)) * 255) | 0);
    }
  }

  offset = 0;
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      for (let c = 0; c < 3; c++) {
        if (c >= channels) {
          signals[offset++] = 0;
          continue;
        }
        let sum = 0;
        for (let dy = 0; dy < step; dy++) {
          const sy = oy * step + dy;
          for (let dx = 0; dx < step; dx++) {
            const sx = ox * step + dx;
            sum += vs.signals[(sy * W + sx) * signalChannels + c];
          }
        }
        signals[offset++] = Math.min(255, ((sum / (step * step)) * 255) | 0);
      }
    }
  }

  offset = 0;
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let sum = 0;
      for (let dy = 0; dy < step; dy++) {
        const sy = oy * step + dy;
        for (let dx = 0; dx < step; dx++) {
          const sx = ox * step + dx;
          sum += vs.poison[sy * W + sx];
        }
      }
      poison[offset++] = Math.min(255, ((sum / (step * step)) * 255) | 0);
    }
  }

  offset = 0;
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let sum = 0;
      for (let dy = 0; dy < step; dy++) {
        const sy = oy * step + dy;
        for (let dx = 0; dx < step; dx++) {
          const sx = ox * step + dx;
          let mag = 0;
          const base = (sy * W + sx) * GLYPH_CHANNELS;
          for (let c = 0; c < GLYPH_CHANNELS; c++) {
            mag += vs.glyphs[base + c] * vs.glyphs[base + c];
          }
          sum += Math.sqrt(mag);
        }
      }
      glyphs[offset++] = Math.min(255, ((sum / (step * step)) * 128) | 0);
    }
  }

  return { gridW: W, gridH: H, step, outW, outH, tick, resources, signals, poison, glyphs };
}

function packFieldKeyframe(state: PackedFieldState): ArrayBuffer {
  const cells = state.outW * state.outH;
  const totalBytes = FIELD_PACKET_HEADER_BYTES + cells + cells * 3 + cells + cells;
  const buf = new ArrayBuffer(totalBytes);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);

  view.setUint32(0, 0x41584646, true);
  view.setUint32(4, state.gridW, true);
  view.setUint32(8, state.gridH, true);
  view.setUint32(12, state.step, true);
  view.setUint32(16, state.tick, true);
  view.setUint32(20, FIELD_PACKET_KIND_KEYFRAME, true);
  view.setUint32(24, 0, true);
  view.setUint32(28, 0, true);

  let offset = FIELD_PACKET_HEADER_BYTES;
  u8.set(state.resources, offset); offset += state.resources.length;
  u8.set(state.signals, offset); offset += state.signals.length;
  u8.set(state.poison, offset); offset += state.poison.length;
  u8.set(state.glyphs, offset);
  return buf;
}

function computeTilePlaneMask(current: PackedFieldState, previous: PackedFieldState, tile: FieldTile): number {
  const tileSize = DISPLAY_CONFIG.fieldTileSize;
  const startX = tile.tileX * tileSize;
  const startY = tile.tileY * tileSize;
  let planeMask = 0;
  for (let dy = 0; dy < tile.tileH; dy++) {
    const row = (startY + dy) * current.outW + startX;
    for (let dx = 0; dx < tile.tileW; dx++) {
      const idx = row + dx;
      if (Math.abs(current.resources[idx] - previous.resources[idx]) >= FIELD_RESOURCE_DELTA_THRESHOLD) planeMask |= FIELD_PLANE_RESOURCES;
      if (Math.abs(current.poison[idx] - previous.poison[idx]) >= FIELD_POISON_DELTA_THRESHOLD) planeMask |= FIELD_PLANE_POISON;
      if (Math.abs(current.glyphs[idx] - previous.glyphs[idx]) >= FIELD_GLYPH_DELTA_THRESHOLD) planeMask |= FIELD_PLANE_GLYPHS;
      const sig = idx * 3;
      if (
        Math.abs(current.signals[sig] - previous.signals[sig]) >= FIELD_SIGNAL_DELTA_THRESHOLD
        || Math.abs(current.signals[sig + 1] - previous.signals[sig + 1]) >= FIELD_SIGNAL_DELTA_THRESHOLD
        || Math.abs(current.signals[sig + 2] - previous.signals[sig + 2]) >= FIELD_SIGNAL_DELTA_THRESHOLD
      ) {
        planeMask |= FIELD_PLANE_SIGNALS;
      }
      if (planeMask === (FIELD_PLANE_RESOURCES | FIELD_PLANE_SIGNALS | FIELD_PLANE_POISON | FIELD_PLANE_GLYPHS)) {
        return planeMask;
      }
    }
  }
  return planeMask;
}

function copyScalarTileToPacket(dest: Uint8Array, offset: number, plane: Uint8Array, outW: number, tile: FieldTile): number {
  const tileSize = DISPLAY_CONFIG.fieldTileSize;
  const startX = tile.tileX * tileSize;
  const startY = tile.tileY * tileSize;
  for (let dy = 0; dy < tile.tileH; dy++) {
    const rowStart = (startY + dy) * outW + startX;
    dest.set(plane.subarray(rowStart, rowStart + tile.tileW), offset);
    offset += tile.tileW;
  }
  return offset;
}

function copySignalTileToPacket(dest: Uint8Array, offset: number, plane: Uint8Array, outW: number, tile: FieldTile): number {
  const tileSize = DISPLAY_CONFIG.fieldTileSize;
  const startX = tile.tileX * tileSize;
  const startY = tile.tileY * tileSize;
  for (let dy = 0; dy < tile.tileH; dy++) {
    const rowStart = ((startY + dy) * outW + startX) * 3;
    dest.set(plane.subarray(rowStart, rowStart + tile.tileW * 3), offset);
    offset += tile.tileW * 3;
  }
  return offset;
}

function packFieldDelta(current: PackedFieldState, previous: PackedFieldState): ArrayBuffer | null {
  if (
    current.gridW !== previous.gridW
    || current.gridH !== previous.gridH
    || current.step !== previous.step
    || current.outW !== previous.outW
    || current.outH !== previous.outH
  ) {
    return null;
  }

  const tileSize = DISPLAY_CONFIG.fieldTileSize;
  const tilesX = Math.ceil(current.outW / tileSize);
  const tilesY = Math.ceil(current.outH / tileSize);
  const changedTiles: FieldTile[] = [];

  for (let tileY = 0; tileY < tilesY; tileY++) {
    const tileH = Math.min(tileSize, current.outH - tileY * tileSize);
    for (let tileX = 0; tileX < tilesX; tileX++) {
      const tileW = Math.min(tileSize, current.outW - tileX * tileSize);
      const planeMask = computeTilePlaneMask(current, previous, { tileX, tileY, tileW, tileH, planeMask: 0 });
      if (planeMask !== 0) changedTiles.push({ tileX, tileY, tileW, tileH, planeMask });
    }
  }

  if (changedTiles.length === 0) return null;

  let totalBytes = FIELD_PACKET_HEADER_BYTES;
  for (const tile of changedTiles) {
    totalBytes += 5;
    if (tile.planeMask & FIELD_PLANE_RESOURCES) totalBytes += tile.tileW * tile.tileH;
    if (tile.planeMask & FIELD_PLANE_SIGNALS) totalBytes += tile.tileW * tile.tileH * 3;
    if (tile.planeMask & FIELD_PLANE_POISON) totalBytes += tile.tileW * tile.tileH;
    if (tile.planeMask & FIELD_PLANE_GLYPHS) totalBytes += tile.tileW * tile.tileH;
  }

  const buf = new ArrayBuffer(totalBytes);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);

  view.setUint32(0, 0x41584646, true);
  view.setUint32(4, current.gridW, true);
  view.setUint32(8, current.gridH, true);
  view.setUint32(12, current.step, true);
  view.setUint32(16, current.tick, true);
  view.setUint32(20, FIELD_PACKET_KIND_DELTA, true);
  view.setUint32(24, tileSize, true);
  view.setUint32(28, changedTiles.length, true);

  let offset = FIELD_PACKET_HEADER_BYTES;
  for (const tile of changedTiles) {
    view.setUint16(offset, tile.tileX, true);
    view.setUint16(offset + 2, tile.tileY, true);
    u8[offset + 4] = tile.planeMask;
    offset += 5;
    if (tile.planeMask & FIELD_PLANE_RESOURCES) offset = copyScalarTileToPacket(u8, offset, current.resources, current.outW, tile);
    if (tile.planeMask & FIELD_PLANE_SIGNALS) offset = copySignalTileToPacket(u8, offset, current.signals, current.outW, tile);
    if (tile.planeMask & FIELD_PLANE_POISON) offset = copyScalarTileToPacket(u8, offset, current.poison, current.outW, tile);
    if (tile.planeMask & FIELD_PLANE_GLYPHS) offset = copyScalarTileToPacket(u8, offset, current.glyphs, current.outW, tile);
  }

  return buf;
}

export function packFieldFrame(world: World, tick: number): ArrayBuffer {
  const vs = world.getVisualState();
  const { gridW: W, gridH: H, signalChannels } = vs;
  const channels   = Math.min(signalChannels, 3);
  const step = DISPLAY_CONFIG.fieldDownsample;
  const outW = Math.max(1, Math.floor(W / step));
  const outH = Math.max(1, Math.floor(H / step));
  const outCells = outW * outH;
  // Layout: header(20) + resources(WH) + signals(WH×3) + poison(WH) + glyphs(WH) + entities(N×8)
  const totalBytes = 20 + outCells + outCells * 3 + outCells + outCells;

  const buf  = new ArrayBuffer(totalBytes);
  const view = new DataView(buf);
  const u8   = new Uint8Array(buf);

  view.setUint32(0,  0x41584646, true); // magic "AXFF"
  view.setUint32(4,  W,    true);
  view.setUint32(8,  H,    true);
  view.setUint32(12, step, true);
  view.setUint32(16, tick,  true);

  let offset = 20;
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let sum = 0;
      for (let dy = 0; dy < step; dy++) {
        const sy = oy * step + dy;
        for (let dx = 0; dx < step; dx++) {
          const sx = ox * step + dx;
          sum += vs.resources[sy * W + sx];
        }
      }
      u8[offset++] = Math.min(255, ((sum / (step * step)) * 255) | 0);
    }
  }
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      for (let c = 0; c < 3; c++) {
        if (c >= channels) {
          u8[offset++] = 0;
          continue;
        }
        let sum = 0;
        for (let dy = 0; dy < step; dy++) {
          const sy = oy * step + dy;
          for (let dx = 0; dx < step; dx++) {
            const sx = ox * step + dx;
            sum += vs.signals[(sy * W + sx) * signalChannels + c];
          }
        }
        u8[offset++] = Math.min(255, ((sum / (step * step)) * 255) | 0);
      }
    }
  }
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let sum = 0;
      for (let dy = 0; dy < step; dy++) {
        const sy = oy * step + dy;
        for (let dx = 0; dx < step; dx++) {
          const sx = ox * step + dx;
          sum += vs.poison[sy * W + sx];
        }
      }
      u8[offset++] = Math.min(255, ((sum / (step * step)) * 255) | 0);
    }
  }
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let sum = 0;
      for (let dy = 0; dy < step; dy++) {
        const sy = oy * step + dy;
        for (let dx = 0; dx < step; dx++) {
          const sx = ox * step + dx;
          let mag = 0;
          const base = (sy * W + sx) * GLYPH_CHANNELS;
          for (let c = 0; c < GLYPH_CHANNELS; c++) {
            mag += vs.glyphs[base + c] * vs.glyphs[base + c];
          }
          sum += Math.sqrt(mag);
        }
      }
      u8[offset++] = Math.min(255, ((sum / (step * step)) * 128) | 0);
    }
  }
  return buf;
}

// ── Meta broadcast type ─────────────────────────────────────────────────────

export function packEntityFrame(world: World, tick: number): ArrayBuffer {
  const vs = world.getVisualState();
  const { gridW: W, gridH: H, entityCount } = vs;
  const totalBytes = 20 + entityCount * 8;

  const buf  = new ArrayBuffer(totalBytes);
  const view = new DataView(buf);
  const u8   = new Uint8Array(buf);

  view.setUint32(0,  0x41584645, true); // magic "AXFE"
  view.setUint32(4,  W,    true);
  view.setUint32(8,  H,    true);
  view.setUint32(12, entityCount, true);
  view.setUint32(16, tick,  true);

  let offset = 20;
  for (let e = 0; e < entityCount; e++) u8[offset++] = vs.entityX[e] & 0xff;
  for (let e = 0; e < entityCount; e++) u8[offset++] = vs.entityY[e] & 0xff;
  for (let e = 0; e < entityCount; e++) u8[offset++] = Math.min(255, (vs.entityEnergy[e] * 255) | 0);
  for (let e = 0; e < entityCount; e++) u8[offset++] = vs.entityAction[e];

  for (let e = 0; e < entityCount; e++) {
    let attackSum = 0;
    for (let j = 0; j < NN_HIDDEN; j++) {
      attackSum += vs.entityGenomes[e * GENOME_LENGTH + NN_W1_SIZE + j * NN_OUTPUTS + 8];
    }
    u8[offset++] = Math.round(255 / (1 + Math.exp(-attackSum * 0.4)));
  }

  for (let e = 0; e < entityCount; e++) {
    let sigSum = 0, eatSum = 0;
    for (let j = 0; j < NN_HIDDEN; j++) {
      sigSum += vs.entityGenomes[e * GENOME_LENGTH + NN_W1_SIZE + j * NN_OUTPUTS + 7];
      eatSum += vs.entityGenomes[e * GENOME_LENGTH + NN_W1_SIZE + j * NN_OUTPUTS + 5];
    }
    const sigTend = 1 / (1 + Math.exp(-sigSum * 0.3));
    const eatTend = 1 / (1 + Math.exp(-eatSum * 0.3));
    u8[offset++] = Math.round((sigTend * 0.6 + eatTend * 0.4) * 255);
  }

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

  for (let e = 0; e < entityCount; e++) {
    let moveSum = 0;
    for (let j = 0; j < NN_HIDDEN; j++) {
      moveSum += vs.entityGenomes[e * GENOME_LENGTH + NN_W1_SIZE + j * NN_OUTPUTS + 1]
               + vs.entityGenomes[e * GENOME_LENGTH + NN_W1_SIZE + j * NN_OUTPUTS + 2]
               + vs.entityGenomes[e * GENOME_LENGTH + NN_W1_SIZE + j * NN_OUTPUTS + 3]
               + vs.entityGenomes[e * GENOME_LENGTH + NN_W1_SIZE + j * NN_OUTPUTS + 4];
    }
    u8[offset++] = Math.round(255 / (1 + Math.exp(-moveSum * 0.075)));
  }

  return buf;
}

export interface MetaBroadcast {
  generation:   number;
  worldIndex:   number;  // completed eval worlds in this generation (0–worldsPerGeneration)
  totalWorlds:  number;
  tick:         number;
  bestLaws:     import('../src/engine/world-laws.ts').WorldLaws | null;
  displayLaws:  import('../src/engine/world-laws.ts').WorldLaws | null;
  population:   number;
  scores:       WorldScores | null;
  bestScore:    number;
  generations:  Array<{ gen: number; best: number; avg: number }>;
  logEntry:     string | null;
  gridSize:     number;
  evalSpeed:    number;       // effective eval ticks/sec (across all workers)
  serverMs:     number;       // EMA of display world step time (ms)
  serverPressure: number;     // 0-2: how much the world is punishing creatures for server load
  sampleGenome?: number[];    // 270 MLP weights of the most-energetic display entity
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
  private lastFieldState: PackedFieldState | null = null;
  private lastFieldKeyframeTick = 0;

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
      this.startDisplayWorld(this.bestLaws ?? this.population[0] ?? randomLaws(this.evalRng));
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
    this.population = Array.from(
      { length: EVAL_CONFIG.worldsPerGeneration },
      () => randomLaws(this.evalRng),
    );
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
      // Fill the rest with fully random exploration.
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
    this.lastFieldState   = null;
    this.lastFieldKeyframeTick = 0;
  }

  private packScheduledFields(world: World): ArrayBuffer | undefined {
    const currentState = samplePackedFieldState(world, this.displayTick);
    const keyframe = packFieldKeyframe(currentState);
    const needsKeyframe = !this.lastFieldState
      || this.displayTick === 1
      || (this.displayTick - this.lastFieldKeyframeTick) >= DISPLAY_CONFIG.fieldKeyframeInterval;

    if (needsKeyframe) {
      this.lastFieldState = currentState;
      this.lastFieldKeyframeTick = this.displayTick;
      return keyframe;
    }

    const delta = packFieldDelta(currentState, this.lastFieldState);
    if (!delta || delta.byteLength >= keyframe.byteLength) {
      this.lastFieldState = currentState;
      this.lastFieldKeyframeTick = this.displayTick;
      return keyframe;
    }

    this.lastFieldState = currentState;
    return delta;
  }

  getBootstrapFrames(): { entities: ArrayBuffer; fields: ArrayBuffer } | null {
    if (!this.displayWorld) {
      this.startDisplayWorld(this.bestLaws ?? this.population[0] ?? randomLaws(this.evalRng));
    }
    const world = this.displayWorld!;
    const baseline = this.lastFieldState ?? samplePackedFieldState(world, this.displayTick);
    return {
      entities: packEntityFrame(world, this.displayTick),
      fields: packFieldKeyframe(baseline),
    };
  }

  /** Step display world once. Call at ~30fps. Returns broadcast data. */
  displayStep(): { entities: ArrayBuffer; fields?: ArrayBuffer; meta: MetaBroadcast } | null {
    if (!this.displayWorld) {
      this.startDisplayWorld(this.bestLaws ?? this.population[0] ?? randomLaws(this.evalRng));
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
        world.laws,
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

    const entities: ArrayBuffer = packEntityFrame(world, this.displayTick);
    const fields = this.displayTick % DISPLAY_CONFIG.fieldFrameInterval === 1
      ? this.packScheduledFields(world)
      : undefined;
    const meta: MetaBroadcast = {
      generation:  this.generation,
      worldIndex:  this.completedWorldsThisGen,
      totalWorlds: EVAL_CONFIG.worldsPerGeneration,
      tick:        this.displayTick,
      bestLaws:    this.bestLaws,
      displayLaws: world.laws,
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
    return { entities, fields, meta };
  }

  private log(msg: string) {
    this.pendingLog = msg;
    console.log(`[sim] ${msg}`);
  }
}
