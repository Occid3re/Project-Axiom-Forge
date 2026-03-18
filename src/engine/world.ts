/**
 * World simulation engine.
 * Runs the grid-based world with entities, resources, signals.
 * All hot-path operations use typed arrays — no object allocation per tick.
 *
 * Decision making: each entity runs an Elman recurrent network each tick:
 *   inputs  = [localResource, energyNorm, entityDensity, signalStrength]
 *   h_new   = tanh(W1 · inputs)          (8 units, W1 = genome[0..31])
 *   h_blend = (1-p) * h_new + p * h_prev (p = memoryPersistence; h_prev from entity memory)
 *   logits  = W2 · h_blend               (6 action logits, W2 = genome[32..79])
 *   action  = softmax_sample(logits)
 *
 * Corpse ecology: dead entities return half their energy to the resource grid.
 */

import { GENOME_LENGTH, ActionType, ResourceDist, MAX_MEMORY_SIZE,
         NN_HIDDEN, NN_OUTPUTS, NN_W1_SIZE, MAX_ENTITIES } from './constants';
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
  readonly entityMap: Int16Array; // -1 or entity index at each cell
  readonly entities: EntityPool;

  tick: number = 0;

  // Server pressure: 0 = idle, >0 = server is loaded, world gets harsher
  serverPressure: number = 0;

  // Per-tick counters
  private tickBirths  = 0;
  private tickDeaths  = 0;
  private tickAttacks = 0;
  private tickSignals = 0;

  // Pre-allocated MLP scratch buffers — avoids GC pressure in the hot loop
  private readonly nnHidden = new Float64Array(NN_HIDDEN);
  private readonly nnLogits = new Float64Array(NN_OUTPUTS);

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
    this.tickBirths  = 0;
    this.tickDeaths  = 0;
    this.tickAttacks = 0;
    this.tickSignals = 0;

    this.decaySignals();
    this.decayPoison();
    this.regenerateResources();
    this.maybeDisaster();
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

  private processEntities(): void {
    const { entities, rng, laws, gridW, gridH } = this;
    const n = entities.count;

    // Shuffled processing order — prevents position-based bias
    const order = new Int32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = rng.int(0, i);
      const tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    // Carrying-capacity "air" pressure — O(1), applied inline below.
    // Exponential curve keyed to hard cap of 4096.
    // Below 1024: negligible.  At 2048: ~½ idleCost.  Above 3500: fatal within seconds.
    // pressure = min(0.3, 2e-5 × e^(9 × n/4096))
    const MAX_POP     = 4096;
    const ratio       = n / MAX_POP;
    const airPressure = Math.min(0.3, 0.0002 * Math.exp(ratio * 9));

    for (let oi = 0; oi < n; oi++) {
      const i = order[oi];
      if (!entities.alive[i]) continue;

      // Age, lifespan, and metabolic cost + carrying-capacity air depletion
      entities.age[i]++;
      if (entities.age[i] >= laws.maxAge) { this.killEntity(i); continue; }
      // Server pressure increases base energy costs
      const pressureCost = this.serverPressure * 0.003;
      entities.energy[i] -= laws.idleCost + airPressure + pressureCost;
      if (entities.energy[i] <= 0) { this.killEntity(i); continue; }

      // Poison damage — toxin at entity's cell drains energy
      const poisonHere = this.poison[entities.y[i] * gridW + entities.x[i]];
      if (poisonHere > 0.01) {
        entities.energy[i] -= poisonHere * laws.poisonStrength;
        if (entities.energy[i] <= 0) { this.killEntity(i); continue; }
      }

      // Overcrowding death — density-dependent mortality caps population
      {
        const cx = entities.x[i], cy = entities.y[i];
        let nCount = 0;
        for (let dy0 = -1; dy0 <= 1; dy0++) {
          for (let dx0 = -1; dx0 <= 1; dx0++) {
            if (dx0 === 0 && dy0 === 0) continue;
            const nx0 = ((cx + dx0) % gridW + gridW) % gridW;
            const ny0 = ((cy + dy0) % gridH + gridH) % gridH;
            if (this.entityMap[ny0 * gridW + nx0] >= 0) nCount++;
          }
        }
        if (nCount > 2) {
          entities.energy[i] -= 0.04 * (nCount - 2);
          if (entities.energy[i] <= 0) { this.killEntity(i); continue; }
        }
      }

      const action = this.decideAction(i);
      entities.action[i] = action;

      switch (action) {
        case ActionType.MOVE:      this.executeMove(i);      break;
        case ActionType.EAT:       this.executeEat(i);       break;
        case ActionType.REPRODUCE: this.executeReproduce(i); break;
        case ActionType.SIGNAL:    this.executeSignal(i);    break;
        case ActionType.ATTACK:    this.executeAttack(i);    break;
        // IDLE: do nothing
      }
    }
  }

  /**
   * MLP decision: 4 sensory inputs → 8 hidden (tanh) → 6 action logits → softmax sample.
   * Uses pre-allocated scratch buffers (nnHidden, nnLogits) — zero allocation.
   *
   * Genome layout:
   *   W1: genome[k * NN_HIDDEN + h]              k=0..3, h=0..7  (32 weights)
   *   W2: genome[NN_W1_SIZE + h * NN_OUTPUTS + a] h=0..7, a=0..5  (48 weights)
   */
  private decideAction(i: number): ActionType {
    const { entities, rng, resources, signals, gridW, gridH, laws } = this;
    const genome = entities.getGenome(i);

    const ex = entities.x[i];
    const ey = entities.y[i];

    // 4 sensory inputs (all normalised to ~[0, 1])
    const localResource = resources[ey * gridW + ex];
    const energyNorm    = Math.min(1, entities.energy[i] / 1.5);

    let nearbyEntities = 0;
    let nearbySignal   = 0;
    const radius = 2;
    const area   = (2 * radius + 1) * (2 * radius + 1) - 1; // 24

    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx   = ((ex + dx) % gridW + gridW) % gridW;
        const ny   = ((ey + dy) % gridH + gridH) % gridH;
        const nIdx = ny * gridW + nx;
        if (this.entityMap[nIdx] >= 0) nearbyEntities++;
        for (let c = 0; c < laws.signalChannels; c++) {
          nearbySignal += signals[nIdx * laws.signalChannels + c];
        }
      }
    }

    const entityDensity  = nearbyEntities / area;
    const signalStrength = nearbySignal / (area * laws.signalChannels || 1);

    const i0 = localResource;
    const i1 = energyNorm;
    const i2 = entityDensity;
    const i3 = signalStrength;

    // W1 forward pass: hidden[h] = tanh(Σ_k genome[k * 8 + h] * input[k])
    const h = this.nnHidden;
    for (let j = 0; j < NN_HIDDEN; j++) {
      h[j] = Math.tanh(
        genome[      j] * i0 +
        genome[ 8 + j] * i1 +
        genome[16 + j] * i2 +
        genome[24 + j] * i3,
      );
    }

    // Recurrent memory: blend new hidden state with previous (Elman network).
    // memoryPersistence controls how much of the previous hidden state carries over.
    // At 0 → purely reactive; at 1 → frozen hidden state. Evolved by meta-evolution.
    // Skip on first tick (age 1) — newborns have zeroed memory, blending would
    // dampen their first decision and increase infant mortality.
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

    // W2 forward pass: logits[a] = Σ_h genome[32 + h * 6 + a] * hidden[h]
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

  private executeMove(i: number): void {
    const { entities, laws, gridW, gridH, entityMap, rng } = this;

    // Random direction — the MLP learns WHEN to move; where to go emerges from exploration
    let dx = rng.int(-1, 1);
    let dy = rng.int(-1, 1);
    if (dx === 0 && dy === 0) {
      if (rng.random() > 0.5) dx = rng.random() > 0.5 ? 1 : -1;
      else                    dy = rng.random() > 0.5 ? 1 : -1;
    }

    const ox = entities.x[i];
    const oy = entities.y[i];
    const nx = ((ox + dx) % gridW + gridW) % gridW;
    const ny = ((oy + dy) % gridH + gridH) % gridH;

    const newCell = ny * gridW + nx;
    if (entityMap[newCell] < 0) {
      entityMap[oy * gridW + ox] = -1;
      entities.x[i]        = nx;
      entities.y[i]        = ny;
      entityMap[newCell]   = i;
      entities.actionDx[i] = dx;
      entities.actionDy[i] = dy;
    }
    entities.energy[i] -= laws.moveCost;
  }

  private executeEat(i: number): void {
    const { entities, laws, resources, gridW } = this;
    const cellIdx   = entities.y[i] * gridW + entities.x[i];
    const gain      = Math.min(resources[cellIdx], laws.eatGain);
    entities.energy[i]  += gain;
    resources[cellIdx]  -= gain;
    if (entities.energy[i] > 1.5) entities.energy[i] = 1.5;
  }

  private executeReproduce(i: number): void {
    const { entities, laws, rng, gridW, gridH, entityMap } = this;

    if (entities.energy[i] < laws.reproductionCost) return;
    if (entities.count >= Math.min(4096, Math.floor(gridW * gridH * 0.07))) return;

    const ox = entities.x[i];
    const oy = entities.y[i];

    for (let attempt = 0; attempt < 8; attempt++) {
      const dx = rng.int(-1, 1);
      const dy = rng.int(-1, 1);
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

    // Channel: derived from first genome weight (evolves freely with selection)
    const channel = Math.floor(Math.abs(genome[0]) * laws.signalChannels) % laws.signalChannels;

    // Strength: mean of W2 SIGNAL column (a=4), sigmoid-mapped — strong signallers
    // evolve W2 weights that push this output high, making signal strength heritable
    let sigColSum = 0;
    for (let j = 0; j < NN_HIDDEN; j++) sigColSum += genome[NN_W1_SIZE + j * NN_OUTPUTS + 4];
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
    this.tickSignals++;
  }

  private executeAttack(i: number): void {
    const { entities, laws, rng, gridW, gridH, entityMap } = this;
    const ox = entities.x[i];
    const oy = entities.y[i];

    // Always target nearest entity — the MLP learns to use ATTACK strategically
    let minDist = 99, tdx = 0, tdy = 0;
    for (let dy2 = -1; dy2 <= 1; dy2++) {
      for (let dx2 = -1; dx2 <= 1; dx2++) {
        if (dx2 === 0 && dy2 === 0) continue;
        const nx2 = ((ox + dx2) % gridW + gridW) % gridW;
        const ny2 = ((oy + dy2) % gridH + gridH) % gridH;
        const d   = Math.abs(dx2) + Math.abs(dy2);
        if (entityMap[ny2 * gridW + nx2] >= 0 && d < minDist) {
          minDist = d; tdx = dx2; tdy = dy2;
        }
      }
    }

    if (minDist === 99) { tdx = rng.int(-1, 1); tdy = rng.int(-1, 1); }
    if (tdx === 0 && tdy === 0) return;

    const nx     = ((ox + tdx) % gridW + gridW) % gridW;
    const ny     = ((oy + tdy) % gridH + gridH) % gridH;
    const cell   = ny * gridW + nx;
    const target = entityMap[cell];

    if (target >= 0 && entities.alive[target]) {
      const stolen = entities.energy[target] * laws.attackTransfer;
      entities.energy[i]      += stolen;
      entities.energy[target] -= stolen;

      if (entities.energy[target] <= 0) {
        entities.energy[i] += 0.45; // kill bonus — applied before cap
        this.killEntity(target);
      }
      entities.energy[i] = Math.min(1.5, entities.energy[i]);
      this.tickAttacks++;
    }
  }

  private killEntity(i: number): void {
    const { entities, entityMap, gridW, resources, resourceCapacity, poison, laws } = this;
    const cellIdx = entities.y[i] * gridW + entities.x[i];

    // Corpse ecology: dead entities deposit half their energy back into the grid.
    const deposit = entities.energy[i] * 0.5;
    if (deposit > 0) {
      resources[cellIdx] = Math.min(resourceCapacity[cellIdx], resources[cellIdx] + deposit);
    }

    // Death toxin: dying entities release poison into the environment.
    // Creates hazardous dead zones around battlefields and mass die-offs.
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
    // Divided by sqrt(GENOME_LENGTH) so the metric is scale-invariant across genome sizes
    let diversity  = 0;
    let diversityVariance = 0;
    const sampleSize = Math.min(n, 20);
    let pairs = 0;
    let distSum2 = 0;  // sum of squared distances for variance
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

    // Signal activity
    let signalActivity = 0;
    for (let i = 0; i < this.signals.length; i++) signalActivity += this.signals[i];

    // Resource coverage
    let filledCells = 0;
    for (let i = 0; i < this.resources.length; i++) {
      if (this.resources[i] > 0.1) filledCells++;
    }

    // Poison coverage
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
      poisonCoverage: poisonedCells / this.poison.length,
    };
  }

  /** Get current state for visualisation. */
  getVisualState(): {
    resources: Float32Array;
    signals: Float32Array;
    poison: Float32Array;
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
