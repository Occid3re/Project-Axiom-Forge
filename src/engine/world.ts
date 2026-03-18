/**
 * World simulation engine.
 * Runs the grid-based world with entities, resources, signals, and glyphs.
 * All hot-path operations use typed arrays — no object allocation per tick.
 *
 * Decision making: each entity runs an Elman recurrent network each tick:
 *   inputs  = [localResource, energyNorm, glyphStrength, ageNorm,      (4 scalars)
 *              resN, resE, resS, resW,                                  (directional resource)
 *              entN, entE, entS, entW,                                  (directional entity)
 *              glyphN, glyphE, glyphS, glyphW]                         (directional glyph)
 *   h_new   = tanh(W1 · inputs)                                        (10 units)
 *   h_blend = (1-p) * h_new + p * h_prev (p = memoryPersistence; h_prev from entity memory)
 *   logits  = W2 · h_blend                                             (11 action logits)
 *   action  = softmax_sample(logits)
 *
 * Corpse ecology: dead entities return energy to the resource grid (corpseEnergy law).
 * Stigmergic memory: entities can DEPOSIT hidden state into a glyph grid and ABSORB it.
 */

import { GENOME_LENGTH, ActionType, ResourceDist, MAX_MEMORY_SIZE, GLYPH_CHANNELS,
         NN_HIDDEN, NN_OUTPUTS, NN_INPUTS, NN_W1_SIZE, MAX_ENTITIES } from './constants';
import { EntityPool } from './entity-pool';
import { PRNG } from './world-laws';
import type { WorldLaws } from './world-laws';

export interface WorldSnapshot {
  tick: number;
  population: number;
  meanEnergy: number;
  totalEnergy: number;
  diversity: number;
  diversityVariance: number;  // variance of pairwise distances — high = clusters (speciation)
  signalActivity: number;
  resourceCoverage: number;
  births: number;
  deaths: number;
  attacks: number;
  signals: number;
  deposits: number;
  absorbs: number;
  kinContacts: number;
  threatContacts: number;
  kinCooperation: number;
  poisonCoverage: number;   // fraction of cells with poison > 0.1
}

export interface WorldHistory {
  snapshots: WorldSnapshot[];
  finalPopulation: number;
  peakPopulation: number;
  disasterCount: number;
  postDisasterRecoveries: number;
}

export interface WorldConfig {
  gridSize: number;
  steps: number;
  initialEntities: number;
}

export class World {
  readonly laws: WorldLaws;
  readonly config: WorldConfig;
  readonly rng: PRNG;

  readonly gridW: number;
  readonly gridH: number;
  readonly resources: Float32Array;
  readonly resourceCapacity: Float32Array;
  readonly signals: Float32Array; // [H * W * channels]
  readonly poison: Float32Array;  // [H * W] toxin concentration 0–1
  readonly glyphs: Float32Array;  // [H * W * GLYPH_CHANNELS] stigmergic memory
  readonly entityMap: Int16Array; // -1 or entity index at each cell
  readonly entities: EntityPool;

  tick: number = 0;

  // Server pressure: 0 = idle, >0 = server is loaded, world gets harsher
  serverPressure: number = 0;

  // Per-tick counters
  private tickBirths   = 0;
  private tickDeaths   = 0;
  private tickAttacks  = 0;
  private tickSignals  = 0;
  private tickDeposits = 0;
  private tickAbsorbs  = 0;
  private tickKinContacts = 0;
  private tickThreatContacts = 0;
  private tickKinCooperation = 0;

  // Pre-allocated MLP scratch buffers — avoids GC pressure in the hot loop
  private readonly nnInputs  = new Float64Array(NN_INPUTS);
  private readonly nnHidden  = new Float64Array(NN_HIDDEN);
  private readonly nnLogits  = new Float64Array(NN_OUTPUTS);
  // Pre-allocated directional perception buffers (4 cardinal dirs: 0=N,1=E,2=S,3=W)
  private readonly dirRes    = new Float64Array(4);
  private readonly dirEnt    = new Float64Array(4);
  private readonly dirGlyph  = new Float64Array(4);
  private readonly dirCount  = new Int32Array(4);
  private readonly processOrder = new Int32Array(MAX_ENTITIES);

  constructor(laws: WorldLaws, config: WorldConfig, seed: number) {
    this.laws   = laws;
    this.config = config;
    this.rng    = new PRNG(seed);
    this.gridW  = config.gridSize;
    this.gridH  = config.gridSize;

    const cellCount = this.gridW * this.gridH;
    this.resources        = new Float32Array(cellCount);
    this.resourceCapacity = new Float32Array(cellCount);
    this.signals          = new Float32Array(cellCount * laws.signalChannels);
    this.poison           = new Float32Array(cellCount);
    this.glyphs           = new Float32Array(cellCount * GLYPH_CHANNELS);
    this.entityMap        = new Int16Array(cellCount).fill(-1);
    this.entities         = new EntityPool();

    this.initResources();
    this.spawnInitialEntities();
  }

  private initResources(): void {
    const { gridW, gridH, rng, laws } = this;

    for (let y = 0; y < gridH; y++) {
      for (let x = 0; x < gridW; x++) {
        const idx = y * gridW + x;
        let cap = 1.0;

        switch (laws.resourceDistribution) {
          case ResourceDist.UNIFORM:
            cap = 0.5 + rng.random() * laws.terrainVariability * 0.5;
            break;
          case ResourceDist.CLUSTERED: {
            const numClusters = 3 + rng.int(0, 4);
            let val = 0.1;
            for (let c = 0; c < numClusters; c++) {
              const cx = rng.int(0, gridW - 1);
              const cy = rng.int(0, gridH - 1);
              const dx = Math.min(Math.abs(x - cx), gridW - Math.abs(x - cx));
              const dy = Math.min(Math.abs(y - cy), gridH - Math.abs(y - cy));
              const dist = Math.sqrt(dx * dx + dy * dy);
              val += Math.exp(-dist * dist / (gridW * 0.15)) * laws.terrainVariability;
            }
            cap = Math.min(1.0, val);
            break;
          }
          case ResourceDist.GRADIENT:
            cap = (y / gridH) * (0.5 + laws.terrainVariability * 0.5);
            break;
        }

        this.resourceCapacity[idx] = cap;
        this.resources[idx] = cap * (0.3 + rng.random() * 0.7);
      }
    }
  }

  private spawnInitialEntities(): void {
    const { gridW, gridH, rng, entities, entityMap, config } = this;
    for (let e = 0; e < config.initialEntities; e++) {
      const x = rng.int(0, gridW - 1);
      const y = rng.int(0, gridH - 1);
      const idx = entities.spawn(x, y, 0.5, null, -1, rng);
      if (idx >= 0) entityMap[y * gridW + x] = idx;
    }
  }

  step(): WorldSnapshot {
    this.tick++;
    this.tickBirths   = 0;
    this.tickDeaths   = 0;
    this.tickAttacks  = 0;
    this.tickSignals  = 0;
    this.tickDeposits = 0;
    this.tickAbsorbs  = 0;
    this.tickKinContacts = 0;
    this.tickThreatContacts = 0;
    this.tickKinCooperation = 0;

    this.decaySignals();
    this.decayPoison();
    this.decayGlyphs();
    this.regenerateResources();
    this.maybeDisaster();
    this.applyDrift();
    this.processEntities();
    this.removeDeadEntities();

    return this.snapshot();
  }

  run(): WorldHistory {
    const snapshots: WorldSnapshot[] = [];
    let peakPop = 0;

    for (let t = 0; t < this.config.steps; t++) {
      const snap = this.step();
      snapshots.push(snap);
      if (snap.population > peakPop) peakPop = snap.population;
    }

    return {
      snapshots,
      finalPopulation: this.entities.count,
      peakPopulation: peakPop,
      disasterCount: 0,
      postDisasterRecoveries: 0,
    };
  }

  private decaySignals(): void {
    const decay = this.laws.signalDecay;
    const len   = this.signals.length;
    for (let i = 0; i < len; i++) {
      this.signals[i] *= decay;
      if (this.signals[i] < 0.001) this.signals[i] = 0;
    }
  }

  private decayPoison(): void {
    // Base decay: ~0.5% per tick. Server pressure slows decay (poison persists longer).
    const baseDecay = 0.995;
    const decay = baseDecay - this.serverPressure * 0.01; // at pressure=2: 0.975 (much slower decay)
    const len = this.poison.length;
    for (let i = 0; i < len; i++) {
      this.poison[i] *= decay;
      if (this.poison[i] < 0.001) this.poison[i] = 0;
    }
  }

  private decayGlyphs(): void {
    const decay = this.laws.glyphDecay ?? 0.996;
    const len = this.glyphs.length;
    for (let i = 0; i < len; i++) {
      this.glyphs[i] *= decay;
      if (Math.abs(this.glyphs[i]) < 0.001) this.glyphs[i] = 0;
    }
  }

  private regenerateResources(): void {
    // Server pressure slows resource regen
    const pressureMul = 1 / (1 + this.serverPressure * 0.5);
    const rate = this.laws.resourceRegenRate * pressureMul;
    const len  = this.resources.length;
    for (let i = 0; i < len; i++) {
      // Poison reduces resource capacity locally (toxin kills the agar)
      const poisonPenalty = this.poison[i] * 0.5;
      const effectiveCap = this.resourceCapacity[i] * (1 - poisonPenalty);
      this.resources[i] += rate * (effectiveCap - this.resources[i]);
      if (this.resources[i] > this.resourceCapacity[i]) {
        this.resources[i] = this.resourceCapacity[i];
      }
    }
  }

  private maybeDisaster(): void {
    // Server pressure increases disaster probability
    const disasterProb = this.laws.disasterProbability * (1 + this.serverPressure * 3);
    if (this.rng.random() < disasterProb) {
      const cx     = this.rng.int(0, this.gridW - 1);
      const cy     = this.rng.int(0, this.gridH - 1);
      const radius = this.rng.int(5, 15);
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx * dx + dy * dy > radius * radius) continue;
          const nx = ((cx + dx) % this.gridW + this.gridW) % this.gridW;
          const ny = ((cy + dy) % this.gridH + this.gridH) % this.gridH;
          this.resources[ny * this.gridW + nx] *= 0.1;
        }
      }
    }
  }

  private applyDrift(): void {
    const speed = this.laws.driftSpeed ?? 0;
    if (speed <= 0) return;
    const { entities, rng, gridW, gridH, entityMap } = this;

    // Slowly rotating current — direction changes over time
    const angle = this.tick * 0.003;
    const driftDx = Math.cos(angle);
    const driftDy = Math.sin(angle);

    // Each entity has a chance to be pushed by the current
    for (let i = 0; i < entities.count; i++) {
      if (!entities.alive[i]) continue;
      if (rng.random() > speed) continue; // probability-based drift

      const ox = entities.x[i];
      const oy = entities.y[i];
      // Quantize drift direction to grid
      const dx = driftDx > 0.3 ? 1 : driftDx < -0.3 ? -1 : 0;
      const dy = driftDy > 0.3 ? 1 : driftDy < -0.3 ? -1 : 0;
      if (dx === 0 && dy === 0) continue;

      const nx = ((ox + dx) % gridW + gridW) % gridW;
      const ny = ((oy + dy) % gridH + gridH) % gridH;
      const newCell = ny * gridW + nx;
      if (entityMap[newCell] < 0) {
        entityMap[oy * gridW + ox] = -1;
        entities.x[i] = nx;
        entities.y[i] = ny;
        entityMap[newCell] = i;
      }
    }
  }

  /**
   * Compute cosine similarity between two entities' behavioral columns (W2 SIGNAL+EAT).
   * Uses the same 16 weights that determine species hue — entities that look the
   * same to the renderer also recognize each other as kin.
   */
  private genomeSimilarity(i: number, j: number): number {
    const entities = this.entities;
    const base = NN_W1_SIZE;
    let dot = 0, magI = 0, magJ = 0;
    for (let h = 0; h < NN_HIDDEN; h++) {
      const si = entities.genomes[i * GENOME_LENGTH + base + h * NN_OUTPUTS + 7]; // SIGNAL col
      const sj = entities.genomes[j * GENOME_LENGTH + base + h * NN_OUTPUTS + 7];
      const ei = entities.genomes[i * GENOME_LENGTH + base + h * NN_OUTPUTS + 5]; // EAT col
      const ej = entities.genomes[j * GENOME_LENGTH + base + h * NN_OUTPUTS + 5];
      dot  += si * sj + ei * ej;
      magI += si * si + ei * ei;
      magJ += sj * sj + ej * ej;
    }
    const denom = Math.sqrt(magI * magJ);
    if (denom < 1e-6) return 0;
    // Cosine similarity is [-1,1], remap to [0,1]
    return (dot / denom + 1) * 0.5;
  }

  private processEntities(): void {
    const { entities, rng, laws, gridW, gridH } = this;
    const n = entities.count;

    // Shuffled processing order — prevents position-based bias
    const order = this.processOrder;
    for (let i = 0; i < n; i++) order[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = rng.int(0, i);
      const tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    // Carrying-capacity "air" pressure — O(1), applied inline below.
    const MAX_POP = 1024;
    const sustainablePop = Math.max(
      16,
      Math.min(MAX_POP, Math.floor(gridW * gridH * (laws.carryingCapacity ?? 0.10))),
    );
    const overCapacity = n > sustainablePop ? n / sustainablePop - 1 : 0;
    const airPressure = overCapacity > 0
      ? Math.min(0.3, 0.0002 * Math.exp(overCapacity * 9))
      : 0;

    const kinThresh = laws.kinThreshold ?? 0.8;

    for (let oi = 0; oi < n; oi++) {
      const i = order[oi];
      if (!entities.alive[i]) continue;

      // Age, lifespan, and metabolic cost + carrying-capacity air depletion
      entities.age[i]++;
      if (entities.age[i] >= laws.maxAge) { this.killEntity(i); continue; }
      // Server pressure increases base energy costs
      const pressureCost = this.serverPressure * 0.003;
      // Aging: older entities burn more energy (mayfly vs tortoise strategies)
      const ageCost = (laws.agingRate ?? 0) * entities.age[i];
      entities.energy[i] -= laws.idleCost + airPressure + pressureCost + ageCost;
      if (entities.energy[i] <= 0) { this.killEntity(i); continue; }

      // Poison damage — toxin at entity's cell drains energy
      const poisonHere = this.poison[entities.y[i] * gridW + entities.x[i]];
      if (poisonHere > 0.01) {
        entities.energy[i] -= poisonHere * laws.poisonStrength;
        if (entities.energy[i] <= 0) { this.killEntity(i); continue; }
      }

      // Directional perception scan (radius-2): accumulate resource, entity, glyph per quadrant.
      // Quadrant assignment uses dominant axis: |dx|>=|dy| → E/W, else → S/N.
      // Also used for overcrowding / cooperation since it covers the same neighbourhood.
      const crowdThresh = Math.max(3, laws.crowdingThreshold ?? 3); // min 3 — 1 killed social behavior
      const coopBonus   = laws.cooperationBonus ?? 0;
      const cx = entities.x[i], cy = entities.y[i];
      const { dirRes, dirEnt, dirGlyph, dirCount } = this;
      dirRes.fill(0); dirEnt.fill(0); dirGlyph.fill(0); dirCount.fill(0);
      let nCount = 0;
      let kinCount = 0;

      for (let dy0 = -2; dy0 <= 2; dy0++) {
        for (let dx0 = -2; dx0 <= 2; dx0++) {
          if (dx0 === 0 && dy0 === 0) continue;
          const nx0 = ((cx + dx0) % gridW + gridW) % gridW;
          const ny0 = ((cy + dy0) % gridH + gridH) % gridH;
          // Dominant-axis quadrant: 0=N, 1=E, 2=S, 3=W
          const adx = dx0 < 0 ? -dx0 : dx0, ady = dy0 < 0 ? -dy0 : dy0;
          const dir = adx >= ady ? (dx0 > 0 ? 1 : 3) : (dy0 > 0 ? 2 : 0);
          dirCount[dir]++;
          const nIdx = ny0 * gridW + nx0;
          dirRes[dir] += this.resources[nIdx];
          const neighbor = this.entityMap[nIdx];
          if (neighbor >= 0) {
            dirEnt[dir]++;
            nCount++;
            if (this.entities.alive[neighbor] && this.genomeSimilarity(i, neighbor) >= kinThresh) {
              kinCount++;
            }
          }
          // Glyph magnitude at neighbour cell
          const gBase = nIdx * GLYPH_CHANNELS;
          let gMag = 0;
          for (let c = 0; c < GLYPH_CHANNELS; c++) {
            const g = this.glyphs[gBase + c]; gMag += g * g;
          }
          dirGlyph[dir] += Math.sqrt(gMag);
        }
      }
      // Normalise by cell count per quadrant → density-like values in [0,1]
      for (let d = 0; d < 4; d++) {
        if (dirCount[d] > 0) {
          dirRes[d]   /= dirCount[d];
          dirEnt[d]   /= dirCount[d];
          dirGlyph[d]  = Math.min(1, dirGlyph[d] / dirCount[d]);
        }
      }

      // Cooperation is kin-selective: only similar neighbours contribute.
      this.tickKinContacts += kinCount;
      this.tickThreatContacts += Math.max(0, nCount - kinCount);
      if (coopBonus > 0 && kinCount > 0) {
        const coopCount = Math.min(kinCount, crowdThresh);
        entities.energy[i] += coopBonus * coopCount;
        this.tickKinCooperation += coopCount;
        const cap = laws.energyCap ?? 1.5;
        if (entities.energy[i] > cap) entities.energy[i] = cap;
      }
      // Overcrowding penalty
      if (nCount > crowdThresh) {
        entities.energy[i] -= 0.04 * (nCount - crowdThresh);
        if (entities.energy[i] <= 0) { this.killEntity(i); continue; }
      }

      // Decide action from directional perception
      const action = this.decideAction(i, dirRes, dirEnt, dirGlyph);
      entities.action[i] = action;

      switch (action) {
        case ActionType.MOVE_N:    this.executeMove(i,  0, -1); break;
        case ActionType.MOVE_E:    this.executeMove(i,  1,  0); break;
        case ActionType.MOVE_S:    this.executeMove(i,  0,  1); break;
        case ActionType.MOVE_W:    this.executeMove(i, -1,  0); break;
        case ActionType.EAT:       this.executeEat(i);          break;
        case ActionType.REPRODUCE: this.executeReproduce(i);    break;
        case ActionType.SIGNAL:    this.executeSignal(i);       break;
        case ActionType.ATTACK:    this.executeAttack(i, kinThresh); break;
        case ActionType.DEPOSIT:   this.executeDeposit(i);      break;
        case ActionType.ABSORB:    this.executeAbsorb(i);       break;
        // IDLE: do nothing
      }
    }
  }

  /**
   * MLP decision: 16 sensory inputs → 10 hidden (tanh) → 11 action logits → softmax sample.
   * Directional inputs: 4 dirs × {resource, entity density, glyph} from the perception scan.
   * Uses pre-allocated scratch buffers (nnInputs, nnHidden, nnLogits) — zero allocation.
   */
  private decideAction(
    i: number,
    dirRes: Float64Array,   // [N, E, S, W] normalised resource
    dirEnt: Float64Array,   // [N, E, S, W] normalised entity density
    dirGlyph: Float64Array, // [N, E, S, W] normalised glyph magnitude
  ): ActionType {
    const { entities, rng, resources, glyphs, gridW, laws } = this;
    const genome = entities.getGenome(i);

    const ex = entities.x[i];
    const ey = entities.y[i];

    // Glyph at current cell
    const cellBase = (ey * gridW + ex) * GLYPH_CHANNELS;
    let glyphMag = 0;
    for (let c = 0; c < GLYPH_CHANNELS; c++) {
      const g = glyphs[cellBase + c]; glyphMag += g * g;
    }

    // 16 sensory inputs (all normalised to ~[0, 1])
    const inp = this.nnInputs;
    inp[0]  = resources[ey * gridW + ex];                                     // localResource
    inp[1]  = Math.min(1, entities.energy[i] / (laws.energyCap ?? 1.5));      // energyNorm
    inp[2]  = Math.min(1, Math.sqrt(glyphMag));                               // glyphStrength (own cell)
    inp[3]  = Math.min(1, entities.age[i] / 500);                             // ageNorm
    // Directional resource (N, E, S, W)
    inp[4]  = dirRes[0]; inp[5]  = dirRes[1]; inp[6]  = dirRes[2]; inp[7]  = dirRes[3];
    // Directional entity density (N, E, S, W)
    inp[8]  = dirEnt[0]; inp[9]  = dirEnt[1]; inp[10] = dirEnt[2]; inp[11] = dirEnt[3];
    // Directional glyph magnitude (N, E, S, W)
    inp[12] = dirGlyph[0]; inp[13] = dirGlyph[1]; inp[14] = dirGlyph[2]; inp[15] = dirGlyph[3];

    // W1 forward pass: hidden[h] = tanh(Σ_k genome[k * NN_HIDDEN + h] * input[k])
    const h = this.nnHidden;
    for (let j = 0; j < NN_HIDDEN; j++) {
      let sum = 0;
      for (let k = 0; k < NN_INPUTS; k++) {
        sum += genome[k * NN_HIDDEN + j] * inp[k];
      }
      h[j] = Math.tanh(sum);
    }

    // Recurrent memory: blend new hidden state with previous (Elman network).
    const persistence = laws.memoryPersistence;
    if (persistence > 0 && entities.age[i] > 1) {
      const mOff = i * MAX_MEMORY_SIZE;
      for (let j = 0; j < NN_HIDDEN; j++) {
        h[j] = h[j] * (1 - persistence) + entities.memory[mOff + j] * persistence;
      }
      for (let j = 0; j < NN_HIDDEN; j++) {
        entities.memory[mOff + j] = h[j];
      }
    } else if (persistence > 0) {
      // First tick: seed memory with pure reactive hidden state (no blend)
      const mOff = i * MAX_MEMORY_SIZE;
      for (let j = 0; j < NN_HIDDEN; j++) {
        entities.memory[mOff + j] = h[j];
      }
    }

    // W2 forward pass: logits[a] = Σ_h genome[NN_W1_SIZE + h * NN_OUTPUTS + a] * hidden[h]
    const logits = this.nnLogits;
    for (let a = 0; a < NN_OUTPUTS; a++) {
      let sum = 0;
      for (let j = 0; j < NN_HIDDEN; j++) sum += genome[NN_W1_SIZE + j * NN_OUTPUTS + a] * h[j];
      logits[a] = sum;
    }

    // Softmax sampling (numerically stable)
    let maxLogit = logits[0];
    for (let a = 1; a < NN_OUTPUTS; a++) if (logits[a] > maxLogit) maxLogit = logits[a];

    let expSum = 0;
    for (let a = 0; a < NN_OUTPUTS; a++) {
      logits[a] = Math.exp(logits[a] - maxLogit);
      expSum += logits[a];
    }

    let r = rng.random() * expSum;
    for (let a = 0; a < NN_OUTPUTS; a++) {
      r -= logits[a];
      if (r <= 0) return a as ActionType;
    }
    return ActionType.IDLE;
  }

  private executeMove(i: number, dx: number, dy: number): void {
    const { entities, laws, gridW, gridH, entityMap } = this;
    const speed = laws.moveSpeed ?? 1;

    const ox = entities.x[i];
    const oy = entities.y[i];
    let finalX = ox, finalY = oy;
    for (let s = 1; s <= speed; s++) {
      const nx = ((ox + dx * s) % gridW + gridW) % gridW;
      const ny = ((oy + dy * s) % gridH + gridH) % gridH;
      if (entityMap[ny * gridW + nx] >= 0) break;
      finalX = nx;
      finalY = ny;
    }

    if (finalX !== ox || finalY !== oy) {
      entityMap[oy * gridW + ox] = -1;
      entities.x[i]      = finalX;
      entities.y[i]      = finalY;
      entityMap[finalY * gridW + finalX] = i;
      entities.actionDx[i] = dx;
      entities.actionDy[i] = dy;
    }
    entities.energy[i] -= laws.moveCost * speed;
  }

  private executeEat(i: number): void {
    const { entities, laws, resources, gridW } = this;
    const cellIdx   = entities.y[i] * gridW + entities.x[i];
    const cap       = laws.energyCap ?? 1.5;
    const gain      = Math.min(resources[cellIdx], laws.eatGain);
    entities.energy[i]  += gain;
    resources[cellIdx]  -= gain;
    if (entities.energy[i] > cap) entities.energy[i] = cap;
  }

  private executeReproduce(i: number): void {
    const { entities, laws, rng, gridW, gridH, entityMap } = this;

    if (entities.energy[i] < laws.reproductionCost) return;
    if (entities.count >= Math.min(1024, Math.floor(gridW * gridH * 0.07))) return;

    const ox = entities.x[i];
    const oy = entities.y[i];

    const spawnDist = laws.spawnDistance ?? 1;
    for (let attempt = 0; attempt < 8; attempt++) {
      const dx = rng.int(-spawnDist, spawnDist);
      const dy = rng.int(-spawnDist, spawnDist);
      if (dx === 0 && dy === 0) continue;

      const nx   = ((ox + dx) % gridW + gridW) % gridW;
      const ny   = ((oy + dy) % gridH + gridH) % gridH;
      const cell = ny * gridW + nx;

      if (entityMap[cell] < 0) {
        // Sexual reproduction: find a nearby mate
        let mateGenome: Float32Array | null = null;
        if (laws.sexualReproduction) {
          outer: for (let mdy = -2; mdy <= 2; mdy++) {
            for (let mdx = -2; mdx <= 2; mdx++) {
              if (mdx === 0 && mdy === 0) continue;
              const mx   = ((ox + mdx) % gridW + gridW) % gridW;
              const my   = ((oy + mdy) % gridH + gridH) % gridH;
              const mate = this.entityMap[my * gridW + mx];
              if (mate >= 0 && mate !== i && entities.alive[mate]) {
                mateGenome = entities.getGenome(mate);
                break outer;
              }
            }
          }
        }

        // Build child genome: uniform crossover (per weight)
        const parentGenome = entities.getGenome(i);
        const childGenome  = new Float32Array(GENOME_LENGTH);
        for (let g = 0; g < GENOME_LENGTH; g++) {
          childGenome[g] = (mateGenome && rng.random() > 0.5) ? mateGenome[g] : parentGenome[g];
        }

        const childIdx = entities.spawn(nx, ny, laws.offspringEnergy, childGenome, entities.id[i], rng);
        if (childIdx >= 0) {
          entities.mutateGenome(childIdx, laws.mutationRate, laws.mutationStrength, rng);
          entityMap[cell] = childIdx;
          entities.energy[i] -= laws.reproductionCost;
          this.tickBirths++;
        }
        break;
      }
    }
  }

  private executeSignal(i: number): void {
    const { entities, laws, signals, gridW, gridH } = this;
    const genome = entities.getGenome(i);

    const channel = Math.floor(Math.abs(genome[0]) * laws.signalChannels) % laws.signalChannels;

    let sigColSum = 0;
    for (let j = 0; j < NN_HIDDEN; j++) sigColSum += genome[NN_W1_SIZE + j * NN_OUTPUTS + 7];
    const strength = 1 / (1 + Math.exp(-sigColSum * 0.3));

    const range = laws.signalRange;
    const ex    = entities.x[i];
    const ey    = entities.y[i];

    for (let dy = -range; dy <= range; dy++) {
      for (let dx = -range; dx <= range; dx++) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > range) continue;
        const nx          = ((ex + dx) % gridW + gridW) % gridW;
        const ny          = ((ey + dy) % gridH + gridH) % gridH;
        const attenuation = 1 - dist / (range + 1);
        signals[(ny * gridW + nx) * laws.signalChannels + channel] += strength * attenuation;
      }
    }
    const sigCost = laws.signalCost ?? 0;
    if (sigCost > 0) entities.energy[i] -= sigCost;
    this.tickSignals++;
  }

  private executeAttack(i: number, kinThresh: number): void {
    const { entities, laws, gridW, gridH, entityMap } = this;
    const ox = entities.x[i];
    const oy = entities.y[i];

    const atkRange = laws.attackRange ?? 1;
    let minDist = 99, tdx = 0, tdy = 0;
    for (let dy2 = -atkRange; dy2 <= atkRange; dy2++) {
      for (let dx2 = -atkRange; dx2 <= atkRange; dx2++) {
        if (dx2 === 0 && dy2 === 0) continue;
        const nx2 = ((ox + dx2) % gridW + gridW) % gridW;
        const ny2 = ((oy + dy2) % gridH + gridH) % gridH;
        const d   = Math.abs(dx2) + Math.abs(dy2);
        const target = entityMap[ny2 * gridW + nx2];
        if (target < 0 || !entities.alive[target]) continue;
        if (this.genomeSimilarity(i, target) >= kinThresh) continue;
        if (d < minDist) {
          minDist = d; tdx = dx2; tdy = dy2;
        }
      }
    }

    if (minDist === 99 || (tdx === 0 && tdy === 0)) return;

    const nx     = ((ox + tdx) % gridW + gridW) % gridW;
    const ny     = ((oy + tdy) % gridH + gridH) % gridH;
    const cell   = ny * gridW + nx;
    const target = entityMap[cell];

    if (target >= 0 && entities.alive[target]) {
      const stolen = entities.energy[target] * laws.attackTransfer;
      entities.energy[i]      += stolen;
      entities.energy[target] -= stolen;

      if (entities.energy[target] <= 0) {
        entities.energy[i] += 0.45;
        this.killEntity(target);
      }
      entities.energy[i] = Math.min(laws.energyCap ?? 1.5, entities.energy[i]);
      this.tickAttacks++;
    }
  }

  /**
   * DEPOSIT: Write compressed hidden state into the glyph grid.
   * Hidden state (10 floats) → 4 glyph channels (sum pairs).
   * Blend: glyph = glyph × 0.3 + deposit × 0.7 (new info dominates).
   */
  private executeDeposit(i: number): void {
    const { entities, laws, glyphs, gridW } = this;
    const cost = laws.depositCost ?? 0.01;
    entities.energy[i] -= cost;

    const cellBase = (entities.y[i] * gridW + entities.x[i]) * GLYPH_CHANNELS;
    const mOff = i * MAX_MEMORY_SIZE;

    for (let c = 0; c < GLYPH_CHANNELS; c++) {
      // Compress: sum pairs of hidden units
      const h0 = entities.memory[mOff + c * 2] || 0;
      const h1 = (c * 2 + 1 < NN_HIDDEN) ? (entities.memory[mOff + c * 2 + 1] || 0) : 0;
      const deposit = h0 + h1;
      glyphs[cellBase + c] = glyphs[cellBase + c] * 0.3 + deposit * 0.7;
    }
    this.tickDeposits++;
  }

  /**
   * ABSORB: Read glyph at current cell and blend into hidden state.
   * Glyph (4 channels) → expanded to 10 floats (each channel fills 2–3 hidden slots).
   * Blend rate controlled by laws.absorbRate.
   */
  private executeAbsorb(i: number): void {
    const { entities, laws, glyphs, gridW } = this;
    const cost = laws.absorbCost ?? 0.005;
    entities.energy[i] -= cost;

    const rate = laws.absorbRate ?? 0.1;
    if (rate <= 0) return;

    const cellBase = (entities.y[i] * gridW + entities.x[i]) * GLYPH_CHANNELS;
    const mOff = i * MAX_MEMORY_SIZE;

    // Check if there's actually something to absorb
    let glyphMag = 0;
    for (let c = 0; c < GLYPH_CHANNELS; c++) {
      glyphMag += glyphs[cellBase + c] * glyphs[cellBase + c];
    }
    if (glyphMag < 0.001) return;

    // Expand glyph channels back to hidden state: reverse of deposit compression
    for (let c = 0; c < GLYPH_CHANNELS; c++) {
      const g = glyphs[cellBase + c] * 0.5; // Split back across 2 hidden units
      const h0idx = c * 2;
      const h1idx = c * 2 + 1;
      if (h0idx < NN_HIDDEN) {
        entities.memory[mOff + h0idx] = entities.memory[mOff + h0idx] * (1 - rate) + g * rate;
      }
      if (h1idx < NN_HIDDEN) {
        entities.memory[mOff + h1idx] = entities.memory[mOff + h1idx] * (1 - rate) + g * rate;
      }
    }
    // Hidden units 8,9 (last pair from 5th channel slot) — leave unchanged since GLYPH_CHANNELS=4
    this.tickAbsorbs++;
  }

  private killEntity(i: number): void {
    const { entities, entityMap, gridW, resources, resourceCapacity, poison, laws } = this;
    const cellIdx = entities.y[i] * gridW + entities.x[i];

    const corpseRatio = laws.corpseEnergy ?? 0.5;
    const deposit = entities.energy[i] * corpseRatio;
    if (deposit > 0) {
      resources[cellIdx] = Math.min(resourceCapacity[cellIdx], resources[cellIdx] + deposit);
    }

    if (laws.deathToxin > 0) {
      poison[cellIdx] = Math.min(1.0, poison[cellIdx] + laws.deathToxin);
    }

    entityMap[cellIdx] = -1;
    entities.kill(i);
    this.tickDeaths++;
  }

  private removeDeadEntities(): void {
    const { entities, entityMap, gridW } = this;
    entityMap.fill(-1);
    entities.compact();
    for (let i = 0; i < entities.count; i++) {
      entityMap[entities.y[i] * gridW + entities.x[i]] = i;
    }
  }

  private snapshot(): WorldSnapshot {
    const { entities } = this;
    const n = entities.count;

    let totalEnergy = 0;
    for (let i = 0; i < n; i++) totalEnergy += entities.energy[i];

    // Diversity: mean normalised pairwise genome distance
    let diversity  = 0;
    let diversityVariance = 0;
    const sampleSize = Math.min(n, 20);
    let pairs = 0;
    let distSum2 = 0;
    for (let a = 0; a < sampleSize; a++) {
      for (let b = a + 1; b < sampleSize; b++) {
        const ga = entities.getGenome(a);
        const gb = entities.getGenome(b);
        let dist = 0;
        for (let g = 0; g < GENOME_LENGTH; g++) {
          const d = ga[g] - gb[g]; dist += d * d;
        }
        const normDist = Math.sqrt(dist) / Math.sqrt(GENOME_LENGTH);
        diversity += normDist;
        distSum2 += normDist * normDist;
        pairs++;
      }
    }
    if (pairs > 0) {
      diversity /= pairs;
      diversityVariance = distSum2 / pairs - diversity * diversity;
    }

    let signalActivity = 0;
    for (let i = 0; i < this.signals.length; i++) signalActivity += this.signals[i];

    let filledCells = 0;
    for (let i = 0; i < this.resources.length; i++) {
      if (this.resources[i] > 0.1) filledCells++;
    }

    let poisonedCells = 0;
    for (let i = 0; i < this.poison.length; i++) {
      if (this.poison[i] > 0.1) poisonedCells++;
    }

    return {
      tick: this.tick,
      population: n,
      meanEnergy: n > 0 ? totalEnergy / n : 0,
      totalEnergy,
      diversity,
      diversityVariance,
      signalActivity,
      resourceCoverage: filledCells / this.resources.length,
      births:  this.tickBirths,
      deaths:  this.tickDeaths,
      attacks: this.tickAttacks,
      signals: this.tickSignals,
      deposits: this.tickDeposits,
      absorbs:  this.tickAbsorbs,
      kinContacts: this.tickKinContacts,
      threatContacts: this.tickThreatContacts,
      kinCooperation: this.tickKinCooperation,
      poisonCoverage: poisonedCells / this.poison.length,
    };
  }

  /** Get current state for visualisation. */
  getVisualState(): {
    resources: Float32Array;
    signals: Float32Array;
    poison: Float32Array;
    glyphs: Float32Array;
    entityX: Int32Array;
    entityY: Int32Array;
    entityEnergy: Float32Array;
    entityAction: Uint8Array;
    entityGenomes: Float32Array;
    entityCount: number;
    gridW: number;
    gridH: number;
    signalChannels: number;
  } {
    return {
      resources:     this.resources,
      signals:       this.signals,
      poison:        this.poison,
      glyphs:        this.glyphs,
      entityX:       this.entities.x.subarray(0, this.entities.count),
      entityY:       this.entities.y.subarray(0, this.entities.count),
      entityEnergy:  this.entities.energy.subarray(0, this.entities.count),
      entityAction:  this.entities.action.subarray(0, this.entities.count),
      entityGenomes: this.entities.genomes,
      entityCount:   this.entities.count,
      gridW:         this.gridW,
      gridH:         this.gridH,
      signalChannels: this.laws.signalChannels,
    };
  }
}
