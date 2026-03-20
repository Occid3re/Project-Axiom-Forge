/**
 * World simulation engine.
 * Runs the grid-based world with entities, resources, signals, and glyphs.
 * All hot-path operations use typed arrays — no object allocation per tick.
 *
 * Decision making: each entity runs an Elman recurrent network each tick:
 *   inputs  = [localResource, energyNorm, glyphStrength, signalStrength, (4 scalars)
 *              resN, resE, resS, resW,                                   (directional resource)
 *              entN, entE, entS, entW,                                   (directional entity)
 *              commN, commE, commS, commW]                               (signal+glyph direction)
 *   h_new   = tanh(W1 · inputs)                                        (10 units)
 *   h_blend = (1-p) * h_new + p * h_prev (p = memoryPersistence; h_prev from entity memory)
 *   logits  = W2 · h_blend                                             (11 action logits)
 *   action  = softmax_sample(logits)
 *
 * Corpse ecology: dead entities return energy to the resource grid (corpseEnergy law).
 * Stigmergic memory: entities can DEPOSIT hidden state into a glyph grid and ABSORB it.
 */

import {
  GENOME_LENGTH,
  ActionType,
  ResourceDist,
  MAX_MEMORY_SIZE,
  GLYPH_CHANNELS,
  NN_HIDDEN,
  NN_HIDDEN_1,
  NN_HIDDEN_2,
  NN_OUTPUTS,
  NN_INPUTS,
  NN_W1_SIZE,
  NN_W3_OFFSET,
  MAX_ENTITIES,
} from './constants';
import { EntityPool } from './entity-pool';
import { PRNG } from './world-laws';
import type { WorldLaws } from './world-laws';

export interface WorldSnapshot {
  tick: number;
  population: number;
  meanEnergy: number;
  totalEnergy: number;
  meanSize: number;
  maxSize: number;
  largeOrganisms: number;
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
  fusedMembers: number;
  largestColony: number;
  colonyBirths: number;
  poisonCoverage: number;        // fraction of cells with poison > 0.1
  harvestEfficiencyRatio: number; // M5: Q4 eat/age ÷ Q1 eat/age. >1.2 = lifetime foraging learning
}

const COLONY_RESERVE_SLOT = NN_HIDDEN;

export interface WorldHistory {
  snapshots: WorldSnapshot[];
  finalPopulation: number;
  peakPopulation: number;
  disasterCount: number;
  postDisasterRecoveries: number;
}

export interface NeuralSample {
  entityId: number;
  action: number;
  age: number;
  energy: number;
  size: number;
  kinNeighbors: number;
  threatNeighbors: number;
  focusScore: number;
  inputs: number[];
  hidden1: number[];
  hidden2: number[];
  probs: number[];
  genome: number[];
}

export interface WorldConfig {
  gridSize: number;
  steps: number;
  initialEntities: number;
  /** Amnesiac mode: hidden state is never read back into the forward pass.
   *  State is still written so the test is fair (no perf difference).
   *  Use to measure how much fitness is actually attributable to recurrent memory. */
  memoryBlind?: boolean;
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
  readonly bodyMap: Int16Array;   // -1 or root entity index occupying this soft body footprint cell
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
  private tickFusedMembers = 0;
  private tickLargestColony = 0;
  private tickColonyBirths = 0;
  // M5 harvest efficiency: track eat-per-age ratio for young vs old entities at death.
  // Q1 = entities in bottom 25% of maxAge at death; Q4 = top 25%.
  // harvestEfficiencyRatio = Q4mean / Q1mean; >1.2 indicates lifetime foraging learning.
  private tickQ1EatSum  = 0;  private tickQ1Count = 0;
  private tickQ4EatSum  = 0;  private tickQ4Count = 0;

  // Pre-allocated MLP scratch buffers — avoids GC pressure in the hot loop
  private readonly nnInputs  = new Float64Array(NN_INPUTS);
  private readonly nnHidden1 = new Float64Array(NN_HIDDEN_1);
  private readonly nnHidden2 = new Float64Array(NN_HIDDEN_2);
  private readonly nnLogits  = new Float64Array(NN_OUTPUTS);
  // Pre-allocated directional perception buffers (4 cardinal dirs: 0=N,1=E,2=S,3=W)
  private readonly dirRes    = new Float64Array(4);
  private readonly dirEnt    = new Float64Array(4);
  private readonly dirGlyph  = new Float64Array(4);
  private readonly dirSignal = new Float64Array(4);
  private readonly dirCount  = new Int32Array(4);
  private readonly processOrder = new Int32Array(MAX_ENTITIES);
  private readonly leaderOrder = new Int32Array(MAX_ENTITIES);
  private readonly colonyParent = new Int32Array(MAX_ENTITIES);
  private readonly colonyRank = new Uint8Array(MAX_ENTITIES);
  private readonly colonyRoot = new Int32Array(MAX_ENTITIES);
  private readonly colonyMembers = new Int32Array(MAX_ENTITIES);
  private readonly colonyEnergy = new Float32Array(MAX_ENTITIES);
  private readonly colonyHidden = new Float32Array(MAX_ENTITIES * NN_HIDDEN);
  private readonly bodyStrength: Float32Array;
  private readonly visualColonyMass = new Uint8Array(MAX_ENTITIES);
  private readonly visualBodyRadius = new Uint8Array(MAX_ENTITIES);

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
    this.bodyMap          = new Int16Array(cellCount).fill(-1);
    this.bodyStrength     = new Float32Array(cellCount);
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
    this.tickFusedMembers = 0;
    this.tickLargestColony = 0;
    this.tickColonyBirths = 0;
    this.tickQ1EatSum = 0; this.tickQ1Count = 0;
    this.tickQ4EatSum = 0; this.tickQ4Count = 0;

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
    const rate      = this.laws.resourceRegenRate * pressureMul;
    const seasonLen = Math.max(1, this.laws.seasonLength ?? 280);
    const amp       = this.laws.seasonAmplitude ?? 0.0;
    const { gridW, gridH, resources, resourceCapacity, poison } = this;
    const len = resources.length;

    for (let i = 0; i < len; i++) {
      // Spatially varying seasonal capacity.
      // Phase offset is a diagonal gradient: NW corner and SE corner are ~half a cycle apart.
      // This rewards spatial-temporal memory (knowing which patch peaks when) rather than
      // mere timing memory (knowing when the global boom occurs).
      const cellX = i % gridW;
      const cellY = (i / gridW) | 0;
      const spatialPhase = (cellX / gridW + cellY / gridH) * 0.5; // [0, 0.5) across grid
      const phase     = ((this.tick / seasonLen) + spatialPhase) % 1.0;
      const seasonMul = 1.0 - amp + amp * (Math.sin(Math.PI * phase) ** 2); // [1-amp, 1.0]

      const poisonPenalty = poison[i] * 0.5;
      const effectiveCap  = resourceCapacity[i] * (1 - poisonPenalty) * seasonMul;
      resources[i] += rate * (effectiveCap - resources[i]);
      if (resources[i] > resourceCapacity[i]) resources[i] = resourceCapacity[i];
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
    const base = NN_W3_OFFSET;
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

  private findColonyRoot(i: number): number {
    let root = i;
    while (this.colonyParent[root] !== root) root = this.colonyParent[root];
    while (this.colonyParent[i] !== i) {
      const next = this.colonyParent[i];
      this.colonyParent[i] = root;
      i = next;
    }
    return root;
  }

  private unionColonies(a: number, b: number): void {
    let rootA = this.findColonyRoot(a);
    let rootB = this.findColonyRoot(b);
    if (rootA === rootB) return;

    const rankA = this.colonyRank[rootA];
    const rankB = this.colonyRank[rootB];
    if (rankA < rankB) {
      const tmp = rootA;
      rootA = rootB;
      rootB = tmp;
    }
    this.colonyParent[rootB] = rootA;
    if (rankA === rankB) this.colonyRank[rootA] = rankA + 1;
  }

  private rebuildColonyTopology(kinThresh: number): void {
    const { entities, entityMap, gridW, gridH } = this;
    const n = entities.count;

    for (let i = 0; i < n; i++) {
      this.colonyParent[i] = i;
      this.colonyRank[i] = 0;
      this.colonyRoot[i] = -1;
      this.colonyMembers[i] = 0;
      this.colonyEnergy[i] = 0;
    }
    this.colonyHidden.fill(0, 0, n * NN_HIDDEN);

    // Orthogonally adjacent kin become a fused multicellular colony.
    for (let i = 0; i < n; i++) {
      if (!entities.alive[i]) continue;
      const x = entities.x[i];
      const y = entities.y[i];

      const east = entityMap[y * gridW + ((x + 1) % gridW)];
      if (east >= 0 && east !== i && entities.alive[east] && this.genomeSimilarity(i, east) >= kinThresh) {
        this.unionColonies(i, east);
      }

      const south = entityMap[((y + 1) % gridH) * gridW + x];
      if (south >= 0 && south !== i && entities.alive[south] && this.genomeSimilarity(i, south) >= kinThresh) {
        this.unionColonies(i, south);
      }
    }

    for (let i = 0; i < n; i++) {
      if (!entities.alive[i]) continue;
      const root = this.findColonyRoot(i);
      this.colonyRoot[i] = root;
      this.colonyMembers[root]++;
      this.colonyEnergy[root] += entities.energy[i];
      const mOff = i * MAX_MEMORY_SIZE;
      const cOff = root * NN_HIDDEN;
      for (let h = 0; h < NN_HIDDEN; h++) {
        this.colonyHidden[cOff + h] += entities.memory[mOff + h];
      }
    }
  }

  private syncColonies(kinThresh: number): void {
    const { entities } = this;
    const n = entities.count;
    this.rebuildColonyTopology(kinThresh);

    this.tickFusedMembers = 0;
    this.tickLargestColony = 0;
    for (let i = 0; i < n; i++) {
      const members = this.colonyMembers[i];
      const memOff = i * MAX_MEMORY_SIZE + COLONY_RESERVE_SLOT;
      if (members <= 1) {
        entities.memory[memOff] *= 0.85;
        continue;
      }
      this.tickFusedMembers += members;
      if (members > this.tickLargestColony) this.tickLargestColony = members;

      const previousReserve = entities.memory[memOff] || 0;
      const surplus = Math.max(0, this.colonyEnergy[i] - members * (this.laws.energyCap ?? 1.5) * 0.56);
      const reserve = previousReserve * 0.93 + surplus * 0.09;
      this.setColonyReserve(i, reserve);
    }

    // Fused colonies equalize energy and partially synchronize recurrent memory.
    for (let i = 0; i < n; i++) {
      if (!entities.alive[i]) continue;
      const root = this.colonyRoot[i];
      const members = root >= 0 ? this.colonyMembers[root] : 1;
      if (members <= 1) continue;

      const meanEnergy = this.colonyEnergy[root] / members;
      entities.energy[i] += (meanEnergy - entities.energy[i]) * 0.18;
      const localCap = this.getEffectiveEnergyCap(i);
      if (entities.energy[i] > localCap) entities.energy[i] = localCap;

      if (entities.energy[i] < localCap * 0.35) {
        const reserve = this.getColonyReserve(root);
        if (reserve > 0.02) {
          const refill = Math.min(reserve, (localCap * 0.42 - entities.energy[i]) * 0.55);
          if (refill > 0) {
            entities.energy[i] += refill;
            this.setColonyReserve(root, reserve - refill);
          }
        }
      }

      const memoryBlend = Math.min(0.26, 0.08 + (members - 1) * 0.04);
      const invMembers = 1 / members;
      const mOff = i * MAX_MEMORY_SIZE;
      const cOff = root * NN_HIDDEN;
      for (let h = 0; h < NN_HIDDEN; h++) {
        const shared = this.colonyHidden[cOff + h] * invMembers;
        entities.memory[mOff + h] = entities.memory[mOff + h] * (1 - memoryBlend) + shared * memoryBlend;
      }
    }
  }

  private prepareVisualState(): void {
    const { entities } = this;
    const n = entities.count;
    this.rebuildColonyTopology(this.laws.kinThreshold ?? 0.8);
    this.rebuildBodyMap();
    for (let i = 0; i < n; i++) {
      if (!entities.alive[i]) {
        this.visualColonyMass[i] = 0;
        this.visualBodyRadius[i] = 0;
        continue;
      }
      const root = this.getBodyRoot(i);
      const members = Math.max(1, this.colonyMembers[root] || 1);
      this.visualColonyMass[i] = Math.min(255, members);
      this.visualBodyRadius[i] = Math.min(255, this.getBodyRadius(root));
    }
  }

  private getColonyMemberCount(i: number): number {
    const root = this.colonyRoot[i];
    return root >= 0 && this.colonyMembers[root] > 0 ? this.colonyMembers[root] : 1;
  }

  private getEffectiveEnergyCap(i: number): number {
    const baseCap = this.laws.energyCap ?? 1.5;
    const sizeBonus = Math.min(1.2, Math.max(0, this.entities.size[i] - 1) * 0.55);
    const colonyBonus = Math.min(0.35, Math.max(0, this.getColonyMemberCount(i) - 1) * 0.05);
    return baseCap * (1 + sizeBonus + colonyBonus);
  }

  private getEffectivePerceptionRadius(i: number): number {
    const base = Math.max(1, Math.min(4, this.laws.maxPerceptionRadius ?? 2));
    const sizeBonus = Math.min(2, Math.floor(Math.max(0, this.entities.size[i] - 1.45)));
    const colonyBonus = this.getColonyMemberCount(i) >= (this.laws.fusionThreshold ?? 6) ? 1 : 0;
    return Math.min(6, base + sizeBonus + colonyBonus);
  }

  private getEffectiveAttackRange(i: number): number {
    const base = this.laws.attackRange ?? 1;
    const sizeBonus = Math.min(2, Math.floor(Math.max(0, this.entities.size[i] - 1.55)));
    return Math.min(4, base + sizeBonus);
  }

  private getEffectiveReproductionCost(i: number): number {
    const base = this.laws.reproductionCost ?? 0.6;
    const sizePenalty = Math.max(0, this.entities.size[i] - 1) * 0.75;
    const colonyPenalty = Math.min(0.45, Math.max(0, this.getColonyMemberCount(i) - 1) * 0.06);
    return base * (1 + sizePenalty + colonyPenalty);
  }

  private getBodyRoot(i: number): number {
    const root = this.colonyRoot[i];
    return root >= 0 && this.entities.alive[root] ? root : i;
  }

  private isSameBodyAlliance(i: number, owner: number): boolean {
    if (owner < 0 || !this.entities.alive[i] || !this.entities.alive[owner]) return false;
    return this.getBodyRoot(i) === this.getBodyRoot(owner);
  }

  private getBodyRadius(i: number): number {
    const root = this.getBodyRoot(i);
    if (root !== i && this.colonyMembers[root] > 1) return 0;

    const size = this.entities.size[root];
    const members = Math.max(1, this.colonyMembers[root] || 1);
    const fusedBias = members > 1 ? 0.35 + Math.min(1.1, (members - 1) * 0.24) : 0;
    const sizeBias = Math.max(0, size - 1.35) * 1.45;
    return Math.min(3, Math.max(0, Math.floor(sizeBias + fusedBias)));
  }

  private rebuildBodyMap(): void {
    const { entities, bodyMap, bodyStrength, gridW, gridH } = this;
    bodyMap.fill(-1);
    bodyStrength.fill(0);

    for (let i = 0; i < entities.count; i++) {
      if (!entities.alive[i]) continue;
      if (this.getBodyRoot(i) !== i) continue;

      const radius = this.getBodyRadius(i);
      if (radius <= 0) continue;

      const ox = entities.x[i];
      const oy = entities.y[i];
      const strengthBase = entities.size[i] + Math.min(1.4, Math.max(0, this.colonyMembers[i] - 1) * 0.18);

      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const dist2 = dx * dx + dy * dy;
          if (dist2 > radius * radius + 1) continue;

          const nx = ((ox + dx) % gridW + gridW) % gridW;
          const ny = ((oy + dy) % gridH + gridH) % gridH;
          const cell = ny * gridW + nx;
          const dist = Math.sqrt(dist2);
          const strength = strengthBase + Math.max(0, 1 - dist / (radius + 0.35)) * 0.45;
          if (strength >= bodyStrength[cell]) {
            bodyStrength[cell] = strength;
            bodyMap[cell] = i;
          }
        }
      }
    }
  }

  private footprintAllowsEntity(i: number, cx: number, cy: number): boolean {
    const { bodyMap, gridW, gridH } = this;
    const radius = this.getBodyRadius(i);
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy > radius * radius + 1) continue;
        const nx = ((cx + dx) % gridW + gridW) % gridW;
        const ny = ((cy + dy) % gridH + gridH) % gridH;
        const owner = bodyMap[ny * gridW + nx];
        if (owner >= 0 && !this.isSameBodyAlliance(i, owner)) return false;
      }
    }
    if (radius <= 0) {
      const owner = bodyMap[cy * gridW + cx];
      if (owner >= 0 && !this.isSameBodyAlliance(i, owner)) return false;
    }
    return true;
  }

  private getColonyReserve(root: number): number {
    if (root < 0 || !this.entities.alive[root]) return 0;
    return this.entities.memory[root * MAX_MEMORY_SIZE + COLONY_RESERVE_SLOT] || 0;
  }

  private setColonyReserve(root: number, value: number): void {
    if (root < 0 || !this.entities.alive[root]) return;
    const capBase = this.laws.energyCap ?? 1.5;
    const members = Math.max(1, this.colonyMembers[root]);
    const reserveCap = capBase * (0.85 + members * 0.55);
    this.entities.memory[root * MAX_MEMORY_SIZE + COLONY_RESERVE_SLOT] = Math.max(0, Math.min(reserveCap, value));
  }

  private alignWithColonyCommand(i: number, localAction: ActionType, kinCount: number, threatCount: number): ActionType {
    const root = this.colonyRoot[i];
    if (root < 0 || root === i || !this.entities.alive[root]) return localAction;
    const members = this.colonyMembers[root];
    if (members <= 1) return localAction;

    const rootAction = this.entities.action[root] as ActionType;
    const reserveNorm = this.getColonyReserve(root) / Math.max(0.001, (this.laws.energyCap ?? 1.5) * (0.75 + members * 0.4));
    const coherence = Math.min(0.78, 0.18 + (members - 1) * 0.07 + reserveNorm * 0.18);

    if (rootAction >= ActionType.MOVE_N && rootAction <= ActionType.MOVE_W) {
      if (
        localAction === ActionType.IDLE ||
        localAction === ActionType.EAT ||
        (localAction >= ActionType.MOVE_N && localAction <= ActionType.MOVE_W)
      ) {
        if (this.rng.random() < coherence) return rootAction;
      }
    }

    if (rootAction === ActionType.ATTACK && threatCount > 0 && this.rng.random() < coherence * 0.8) {
      return ActionType.ATTACK;
    }
    if (rootAction === ActionType.REPRODUCE && members >= (this.laws.fusionThreshold ?? 6) && this.rng.random() < coherence * 0.35) {
      return ActionType.REPRODUCE;
    }
    if (
      (rootAction === ActionType.SIGNAL || rootAction === ActionType.DEPOSIT || rootAction === ActionType.ABSORB)
      && kinCount > 0
      && this.rng.random() < coherence * 0.55
    ) {
      return rootAction;
    }
    return localAction;
  }

  /**
   * Multicellular fusion — large colonies collapse into a single mega-entity.
   * The root entity absorbs all members: gaining their energy, growing larger,
   * and blending their genomes into a genetic chimera.
   * fusionThreshold: min colony size to be eligible.
   * fusionRate:      probability per tick × members that an eligible colony fuses.
   */
  private executeFusion(): void {
    const { entities, entityMap, gridW, laws } = this;
    const n = entities.count;
    const fusThresh = laws.fusionThreshold ?? 4;
    const fusRate   = laws.fusionRate ?? 0.0;
    if (fusRate <= 0) return;

    const cap     = laws.energyCap ?? 1.5;
    const sizeMax = laws.cellSizeMax ?? 2.8;
    const fusionBudget = n >= 720 ? 2 : 1;
    let fusionEvents = 0;

    for (let root = 0; root < n; root++) {
      if (fusionEvents >= fusionBudget) break;
      const members = this.colonyMembers[root];
      if (members < fusThresh) continue;
      if (!entities.alive[root]) continue;
      const meanEnergy = members > 0 ? this.colonyEnergy[root] / members : 0;
      const minAge = 120 + members * 8;
      const energyGate = cap * (0.46 + Math.min(0.16, members * 0.01));
      const sizeGate = sizeMax * 0.72;
      if (entities.age[root] < minAge) continue;
      if (meanEnergy < energyGate) continue;
      if (entities.size[root] < sizeGate && members < fusThresh + 2) continue;

      // Probability scales with maturity and colony surplus, but stays tightly bounded.
      const fusionChance = Math.min(
        0.02,
        fusRate
          * Math.min(3, members / Math.max(1, fusThresh))
          * Math.min(1.35, meanEnergy / Math.max(0.001, cap * 0.75)),
      );
      if (this.rng.random() >= fusionChance) continue;

      // Absorb ONE random non-root member per event (gradual fusion, not mass extinction)
      const rOff = root * GENOME_LENGTH;
      let victim = -1;
      let seen = 0;
      for (let i = 0; i < n; i++) {
        if (i === root || !entities.alive[i]) continue;
        if (this.colonyRoot[i] !== root) continue;
        seen++;
        if (this.rng.random() < 1 / seen) victim = i;
      }

      if (victim >= 0) {
        const rootMemOff = root * MAX_MEMORY_SIZE;
        const victimMemOff = victim * MAX_MEMORY_SIZE;
        const victimEnergy = entities.energy[victim];

        // Energy: absorb most, but burn some as coordination heat.
        entities.energy[root] += victimEnergy * 0.78;

        // Size: fusion can push beyond the solo ceiling, but only modestly.
        const fusedSizeCap = Math.min(sizeMax * 1.55, sizeMax + 1.2);
        entities.size[root] = Math.min(fusedSizeCap, entities.size[root] + Math.max(0.12, entities.size[victim] * 0.42));

        // Genome: blend 10% toward absorbed member
        const iOff = victim * GENOME_LENGTH;
        for (let g = 0; g < GENOME_LENGTH; g++) {
          entities.genomes[rOff + g] = entities.genomes[rOff + g] * 0.90 + entities.genomes[iOff + g] * 0.10;
        }

        for (let h = 0; h < NN_HIDDEN; h++) {
          entities.memory[rootMemOff + h] = entities.memory[rootMemOff + h] * 0.88 + entities.memory[victimMemOff + h] * 0.12;
        }

        const fusionCap = this.getEffectiveEnergyCap(root) * 1.15;
        entities.energy[root] = Math.min(fusionCap, entities.energy[root]);
        entities.energy[root] = Math.max(0.08, entities.energy[root] - 0.04 * members);
        entities.breedCooldown[root] = Math.max(entities.breedCooldown[root], 180);
        this.setColonyReserve(root, this.getColonyReserve(root) + victimEnergy * 0.12);

        // Remove from world cleanly — no corpse energy (absorbed, not dead)
        entityMap[entities.y[victim] * gridW + entities.x[victim]] = -1;
        entities.kill(victim);
        this.colonyRoot[victim] = -1;
        this.colonyMembers[root] = Math.max(1, this.colonyMembers[root] - 1);
        this.colonyEnergy[root] = Math.max(0, this.colonyEnergy[root] - victimEnergy);
        this.tickColonyBirths++; // reuse metric as "fusion events" counter
        fusionEvents++;
      }
    }
  }

  private colonyMoveMaintainsContact(i: number, nx: number, ny: number): boolean {
    const { entities, entityMap, gridW, gridH } = this;
    const root = this.colonyRoot[i];
    if (root < 0 || this.colonyMembers[root] <= 1) return true;

    const dirs = [
      [0, -1],
      [1, 0],
      [0, 1],
      [-1, 0],
    ] as const;
    for (const [dx, dy] of dirs) {
      const tx = ((nx + dx) % gridW + gridW) % gridW;
      const ty = ((ny + dy) % gridH + gridH) % gridH;
      const neighbor = entityMap[ty * gridW + tx];
      if (neighbor < 0 || neighbor === i || !entities.alive[neighbor]) continue;
      if (this.colonyRoot[neighbor] === root) return true;
    }
    return false;
  }

  private getLocalColonySupport(i: number): number {
    const { entities, entityMap, gridW, gridH } = this;
    let support = Math.max(0, entities.energy[i] - (this.laws.energyCap ?? 1.5) * 0.30);
    const root = this.colonyRoot[i];
    if (root < 0 || this.colonyMembers[root] <= 1) return support;

    const ox = entities.x[i];
    const oy = entities.y[i];
    const dirs = [
      [0, -1],
      [1, 0],
      [0, 1],
      [-1, 0],
    ] as const;
    for (const [dx, dy] of dirs) {
      const nx = ((ox + dx) % gridW + gridW) % gridW;
      const ny = ((oy + dy) % gridH + gridH) % gridH;
      const donor = entityMap[ny * gridW + nx];
      if (donor < 0 || donor === i || !entities.alive[donor]) continue;
      if (this.colonyRoot[donor] !== root) continue;
      support += Math.max(0, entities.energy[donor] - 0.12) * 0.5;
    }
    return support;
  }

  private fundReproductionFromColony(i: number, cost: number): boolean {
    const { entities, entityMap, gridW, gridH } = this;
    let remaining = cost;
    let usedDonor = false;

    const parentSpend = Math.min(Math.max(0, entities.energy[i] - 0.12), remaining);
    entities.energy[i] -= parentSpend;
    remaining -= parentSpend;

    const root = this.colonyRoot[i];
    if (remaining > 0 && root >= 0 && this.colonyMembers[root] > 1) {
      const ox = entities.x[i];
      const oy = entities.y[i];
      const dirs = [
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, 0],
      ] as const;
      for (const [dx, dy] of dirs) {
        const nx = ((ox + dx) % gridW + gridW) % gridW;
        const ny = ((oy + dy) % gridH + gridH) % gridH;
        const donor = entityMap[ny * gridW + nx];
        if (donor < 0 || donor === i || !entities.alive[donor]) continue;
        if (this.colonyRoot[donor] !== root) continue;
        const donorSpend = Math.min(Math.max(0, entities.energy[donor] - 0.12) * 0.5, remaining);
        entities.energy[donor] -= donorSpend;
        if (donorSpend > 0) usedDonor = true;
        remaining -= donorSpend;
        if (remaining <= 1e-6) break;
      }
    }

    if (remaining > 0 && root >= 0 && this.colonyMembers[root] > 1) {
      const reserve = this.getColonyReserve(root);
      const reserveFloor = (this.laws.energyCap ?? 1.5) * 0.18;
      const reserveSpend = Math.min(Math.max(0, reserve - reserveFloor), remaining);
      if (reserveSpend > 0) {
        this.setColonyReserve(root, reserve - reserveSpend);
        remaining -= reserveSpend;
        usedDonor = true;
      }
    }

    if (remaining <= 1e-6 && usedDonor) this.tickColonyBirths++;
    return remaining <= 1e-6;
  }

  private imprintColonyMemory(parentIdx: number, childIdx: number): void {
    const { entities } = this;
    const root = this.colonyRoot[parentIdx];
    const childOff = childIdx * MAX_MEMORY_SIZE;
    if (root >= 0 && this.colonyMembers[root] > 1) {
      const cOff = root * NN_HIDDEN;
      const invMembers = 1 / this.colonyMembers[root];
      for (let h = 0; h < NN_HIDDEN; h++) {
        entities.memory[childOff + h] = this.colonyHidden[cOff + h] * invMembers * 0.35;
      }
      return;
    }

    const parentOff = parentIdx * MAX_MEMORY_SIZE;
    for (let h = 0; h < NN_HIDDEN; h++) {
      entities.memory[childOff + h] = entities.memory[parentOff + h] * 0.2;
    }
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

    // Carrying capacity should matter on both eval and display worlds.
    // Map the evolvable carryingCapacity law onto the actual entity hard cap
    // instead of raw grid occupancy, otherwise it becomes almost dead on 256x256.
    const entityHardCap = Math.min(1024, Math.floor(gridW * gridH * 0.07));
    const carryMin = 0.05;
    const carryMax = 0.24;
    const carryNorm = Math.max(
      0,
      Math.min(1, ((laws.carryingCapacity ?? 0.10) - carryMin) / (carryMax - carryMin)),
    );
    const sustainablePop = Math.max(
      16,
      Math.min(entityHardCap, Math.floor(16 + carryNorm * (entityHardCap - 16))),
    );
    const overCapacity = n > sustainablePop ? n / sustainablePop - 1 : 0;
    const airPressure = overCapacity > 0
      ? Math.min(0.3, 0.0002 * Math.exp(overCapacity * 9))
      : 0;

    const kinThresh = laws.kinThreshold ?? 0.8;
    this.syncColonies(kinThresh);
    this.executeFusion();
    this.rebuildBodyMap();

    const leaderOrder = this.leaderOrder;
    let leaderWrite = 0;
    for (let oi = 0; oi < n; oi++) {
      const i = order[oi];
      if (!entities.alive[i]) continue;
      if (this.colonyMembers[i] > 1 && this.colonyRoot[i] === i) leaderOrder[leaderWrite++] = i;
    }
    for (let oi = 0; oi < n; oi++) {
      const i = order[oi];
      if (!entities.alive[i]) continue;
      if (!(this.colonyMembers[i] > 1 && this.colonyRoot[i] === i)) leaderOrder[leaderWrite++] = i;
    }

    for (let oi = 0; oi < leaderWrite; oi++) {
      const i = leaderOrder[oi];
      if (!entities.alive[i]) continue;

      // Age, lifespan, and metabolic cost + carrying-capacity air depletion
      entities.age[i]++;
      if (entities.breedCooldown[i] > 0) entities.breedCooldown[i]--;
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

      // Directional perception scan: accumulate resource, entity, glyph per quadrant.
      // Quadrant assignment uses dominant axis: |dx|>=|dy| → E/W, else → S/N.
      // Also used for overcrowding / cooperation since it covers the same neighbourhood.
      const crowdThresh = Math.max(3, laws.crowdingThreshold ?? 3); // min 3 — 1 killed social behavior
      const coopBonus   = laws.cooperationBonus ?? 0;
      const perceptionRadius = this.getEffectivePerceptionRadius(i);
      const cx = entities.x[i], cy = entities.y[i];
      const { dirRes, dirEnt, dirGlyph, dirSignal, dirCount } = this;
      dirRes.fill(0); dirEnt.fill(0); dirGlyph.fill(0); dirSignal.fill(0); dirCount.fill(0);
      let nCount = 0;
      let kinCount = 0;
      let localSignal = 0;
      const localSignalBase = (cy * gridW + cx) * laws.signalChannels;
      for (let c = 0; c < laws.signalChannels; c++) {
        localSignal += this.signals[localSignalBase + c];
      }
      localSignal = laws.signalChannels > 0 ? Math.min(1, localSignal / laws.signalChannels) : 0;

      for (let dy0 = -perceptionRadius; dy0 <= perceptionRadius; dy0++) {
        for (let dx0 = -perceptionRadius; dx0 <= perceptionRadius; dx0++) {
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
          let signal = 0;
          const sBase = nIdx * laws.signalChannels;
          for (let c = 0; c < laws.signalChannels; c++) {
            signal += this.signals[sBase + c];
          }
          dirSignal[dir] += laws.signalChannels > 0 ? signal / laws.signalChannels : 0;
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
          dirSignal[d] = Math.min(1, dirSignal[d] / dirCount[d]);
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
        const localCap = this.getEffectiveEnergyCap(i);
        if (entities.energy[i] > localCap) entities.energy[i] = localCap;
      }
      // Overcrowding penalty
      if (nCount > crowdThresh) {
        const crowdPenalty = 0.04 * (nCount - crowdThresh) * (0.85 + entities.size[i] * 0.18);
        entities.energy[i] -= crowdPenalty;
        if (entities.energy[i] <= 0) { this.killEntity(i); continue; }
      }

      // Energy-rich kin clusters can swell into larger multicellular blobs.
      const effectiveCap = this.getEffectiveEnergyCap(i);
      const energyNorm = Math.min(1.4, entities.energy[i] / effectiveCap);
      const kinSupport = Math.min(1, kinCount / Math.max(1, crowdThresh));
      const hostileCrowding = Math.min(1, Math.max(0, (nCount - kinCount) / Math.max(1, crowdThresh)));
      const colonyMembers = this.getColonyMemberCount(i);
      const colonyMass = Math.min(1, Math.max(0, colonyMembers - 1) / 4);
      const sizeCeiling = Math.min(
        (this.laws.cellSizeMax ?? 2.8) * 1.55,
        (this.laws.cellSizeMax ?? 2.8) + colonyMass * 1.15,
      );
      const ageLift = Math.min(0.20, entities.age[i] / Math.max(240, laws.maxAge) * 0.30);
      const targetSize = Math.max(
        0.74,
        Math.min(
          sizeCeiling,
          0.80
            + Math.max(0, energyNorm - 0.22) * this.laws.growthEfficiency * 0.95
            + this.resources[cy * gridW + cx] * 0.16
            + kinSupport * 0.24
            + colonyMass * 0.85
            + ageLift
            - hostileCrowding * 0.30,
        ),
      );
      entities.size[i] += (targetSize - entities.size[i]) * 0.10;
      const excessSize = Math.max(0, entities.size[i] - 1);
      const sizeUpkeep =
        excessSize * (this.laws.sizeMaintenance * 1.45 + colonyMass * 0.0010)
        + excessSize * excessSize * 0.0018
        + hostileCrowding * excessSize * 0.0040;
      entities.energy[i] -= sizeUpkeep;
      if (entities.energy[i] <= 0) { this.killEntity(i); continue; }

      // Decide action from directional perception
      const chosenAction = this.decideAction(i, localSignal, dirRes, dirEnt, dirGlyph, dirSignal);
      const action = this.alignWithColonyCommand(i, chosenAction, kinCount, Math.max(0, nCount - kinCount));
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
   * Directional inputs: 4 dirs × {resource, entity density, communication} from the perception scan.
   * Uses pre-allocated scratch buffers (nnInputs, nnHidden, nnLogits) — zero allocation.
   */
  private decideAction(
    i: number,
    localSignal: number,
    dirRes: Float64Array,   // [N, E, S, W] normalised resource
    dirEnt: Float64Array,   // [N, E, S, W] normalised entity density
    dirGlyph: Float64Array, // [N, E, S, W] normalised glyph magnitude
    dirSignal: Float64Array,// [N, E, S, W] normalised signal magnitude
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
    inp[3]  = localSignal;                                                    // signalStrength (own cell)
    // Directional resource (N, E, S, W)
    inp[4]  = dirRes[0]; inp[5]  = dirRes[1]; inp[6]  = dirRes[2]; inp[7]  = dirRes[3];
    // Directional entity density (N, E, S, W)
    inp[8]  = dirEnt[0]; inp[9]  = dirEnt[1]; inp[10] = dirEnt[2]; inp[11] = dirEnt[3];
    // Directional communication (fast signal + persistent glyph), N/E/S/W
    inp[12] = Math.min(1, dirSignal[0] * 0.9 + dirGlyph[0] * 0.7);
    inp[13] = Math.min(1, dirSignal[1] * 0.9 + dirGlyph[1] * 0.7);
    inp[14] = Math.min(1, dirSignal[2] * 0.9 + dirGlyph[2] * 0.7);
    inp[15] = Math.min(1, dirSignal[3] * 0.9 + dirGlyph[3] * 0.7);

    // Selective perceptual noise — directional resource (4–7) and entity density (8–11) only.
    // Own-state inputs (0–3) and communication inputs (12–15) are left clean:
    //   own state is always known precisely; signals are explicit transmissions whose
    //   reliability is what makes communication worth evolving.
    // Forcing uncertainty on spatial sensing creates a fitness premium for using the
    // recurrent hidden state as a compensating prior rather than reacting to raw inputs.
    const noise = this.laws.perceptNoise ?? 0;
    if (noise > 0) {
      for (let k = 4; k <= 11; k++) {
        inp[k] = Math.max(0, Math.min(1, inp[k] + rng.normal(0, noise)));
      }
    }

    // W1 forward pass: hidden[h] = tanh(Σ_k genome[k * NN_HIDDEN + h] * input[k])
    const h1 = this.nnHidden1;
    for (let j = 0; j < NN_HIDDEN_1; j++) {
      let sum = 0;
      for (let k = 0; k < NN_INPUTS; k++) {
        sum += genome[k * NN_HIDDEN_1 + j] * inp[k];
      }
      h1[j] = Math.tanh(sum);
    }

    const h2 = this.nnHidden2;
    for (let j = 0; j < NN_HIDDEN_2; j++) {
      let sum = 0;
      for (let h = 0; h < NN_HIDDEN_1; h++) {
        sum += genome[NN_W1_SIZE + h * NN_HIDDEN_2 + j] * h1[h];
      }
      h2[j] = Math.tanh(sum);
    }

    // Recurrent memory: blend some hidden units with previous state, but always
    // mirror the current hidden state back into memory so stigmergic deposit/absorb
    // can operate on meaningful live state even in reactive worlds.
    const persistence = Math.max(0, Math.min(0.92, laws.memoryPersistence));
    const persistedUnits = Math.max(0, Math.min(NN_HIDDEN, laws.memorySize ?? NN_HIDDEN));
    const mOff = i * MAX_MEMORY_SIZE;
    // memoryBlind=true: skip reading stored state back — amnesiac delta test.
    // State is still written below so the comparison is compute-fair.
    if (!this.config.memoryBlind && persistedUnits > 0 && persistence > 0 && entities.age[i] > 1) {
      for (let j = 0; j < persistedUnits; j++) {
        h2[j] = h2[j] * (1 - persistence) + entities.memory[mOff + j] * persistence;
      }
    }
    for (let j = 0; j < persistedUnits; j++) {
      entities.memory[mOff + j] = h2[j];
    }
    for (let j = persistedUnits; j < NN_HIDDEN; j++) {
      entities.memory[mOff + j] = 0;
    }

    // W2 forward pass: logits[a] = Σ_h genome[NN_W1_SIZE + h * NN_OUTPUTS + a] * hidden[h]
    const logits = this.nnLogits;
    for (let a = 0; a < NN_OUTPUTS; a++) {
      let sum = 0;
      for (let j = 0; j < NN_HIDDEN_2; j++) {
        sum += genome[NN_W3_OFFSET + j * NN_OUTPUTS + a] * h2[j];
      }
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
    const { entities, laws, gridW, gridH, entityMap, bodyMap } = this;
    const speed = laws.moveSpeed ?? 1;

    const ox = entities.x[i];
    const oy = entities.y[i];
    let finalX = ox, finalY = oy;
    for (let s = 1; s <= speed; s++) {
      const nx = ((ox + dx * s) % gridW + gridW) % gridW;
      const ny = ((oy + dy * s) % gridH + gridH) % gridH;
      if (entityMap[ny * gridW + nx] >= 0) break;
      const footprintOwner = bodyMap[ny * gridW + nx];
      if (footprintOwner >= 0 && !this.isSameBodyAlliance(i, footprintOwner)) break;
      finalX = nx;
      finalY = ny;
    }

    const sizeFactor = 0.72 + Math.pow(Math.max(0.75, entities.size[i]), 1.2) * 0.34;
    const moveCost = laws.moveCost * speed * sizeFactor;

    if (finalX !== ox || finalY !== oy) {
      if (!this.colonyMoveMaintainsContact(i, finalX, finalY)) {
        entities.energy[i] -= moveCost * 0.35;
        return;
      }
      if (!this.footprintAllowsEntity(i, finalX, finalY)) {
        entities.energy[i] -= moveCost * 0.35;
        return;
      }
      entityMap[oy * gridW + ox] = -1;
      entities.x[i]      = finalX;
      entities.y[i]      = finalY;
      entityMap[finalY * gridW + finalX] = i;
      entities.actionDx[i] = dx;
      entities.actionDy[i] = dy;
    }
    entities.energy[i] -= moveCost;
  }

  private executeEat(i: number): void {
    const { entities, laws, resources, gridW, gridH, bodyMap } = this;
    const ox = entities.x[i];
    const oy = entities.y[i];
    const bodyRadius = this.getBodyRadius(i);
    let cellIdx = oy * gridW + ox;
    let bestResource = resources[cellIdx];
    if (bodyRadius > 0) {
      for (let dy = -bodyRadius; dy <= bodyRadius; dy++) {
        for (let dx = -bodyRadius; dx <= bodyRadius; dx++) {
          if (dx * dx + dy * dy > bodyRadius * bodyRadius + 1) continue;
          const nx = ((ox + dx) % gridW + gridW) % gridW;
          const ny = ((oy + dy) % gridH + gridH) % gridH;
          const candidate = ny * gridW + nx;
          const owner = bodyMap[candidate];
          if (owner >= 0 && !this.isSameBodyAlliance(i, owner)) continue;
          if (resources[candidate] > bestResource) {
            bestResource = resources[candidate];
            cellIdx = candidate;
          }
        }
      }
    }
    const cap       = this.getEffectiveEnergyCap(i);
    const harvestMul =
      1
      + Math.min(0.28, Math.max(0, entities.size[i] - 1) * 0.18)
      + Math.min(0.12, bodyRadius * 0.04);
    const gain      = Math.min(resources[cellIdx], laws.eatGain * harvestMul);
    entities.energy[i]  += gain;
    resources[cellIdx]  -= gain;
    if (entities.energy[i] > cap) entities.energy[i] = cap;
    entities.eatCount[i]++;
  }

  private executeReproduce(i: number): void {
    const { entities, laws, rng, gridW, gridH, entityMap, bodyMap } = this;
    const reproductionCost = this.getEffectiveReproductionCost(i);

    if (entities.age[i] < 150) return;
    if (entities.breedCooldown[i] > 0) return;
    if (this.getLocalColonySupport(i) < reproductionCost) return;
    const entityHardCap = Math.min(1024, Math.floor(gridW * gridH * 0.07));
    if (entities.count >= entityHardCap) return;

    const ox = entities.x[i];
    const oy = entities.y[i];

    const bodyRadius = this.getBodyRadius(i);
    const spawnDist = Math.max(laws.spawnDistance ?? 1, bodyRadius + 1);
    for (let attempt = 0; attempt < 8; attempt++) {
      const dx = rng.int(-spawnDist, spawnDist);
      const dy = rng.int(-spawnDist, spawnDist);
      if (dx === 0 && dy === 0) continue;
      if (Math.max(Math.abs(dx), Math.abs(dy)) <= bodyRadius) continue;

      const nx   = ((ox + dx) % gridW + gridW) % gridW;
      const ny   = ((oy + dy) % gridH + gridH) % gridH;
      const cell = ny * gridW + nx;

      if (entityMap[cell] < 0 && bodyMap[cell] < 0) {
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
          if (!this.fundReproductionFromColony(i, reproductionCost)) {
            entityMap[cell] = -1;
            entities.kill(childIdx);
            return;
          }
          this.imprintColonyMemory(i, childIdx);
          entities.size[i] = Math.max(0.78, entities.size[i] * 0.9);
          entities.breedCooldown[i] = Math.max(250, Math.round(140 + Math.max(0, entities.size[i] - 1) * 110));
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
    for (let j = 0; j < NN_HIDDEN; j++) sigColSum += genome[NN_W3_OFFSET + j * NN_OUTPUTS + 7];
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
    const { entities, laws, gridW, gridH, entityMap, bodyMap } = this;
    const ox = entities.x[i];
    const oy = entities.y[i];

    const atkRange = this.getEffectiveAttackRange(i) + this.getBodyRadius(i);
    let minDist = 99;
    let bestTarget = -1;
    for (let dy2 = -atkRange; dy2 <= atkRange; dy2++) {
      for (let dx2 = -atkRange; dx2 <= atkRange; dx2++) {
        if (dx2 === 0 && dy2 === 0) continue;
        const nx2 = ((ox + dx2) % gridW + gridW) % gridW;
        const ny2 = ((oy + dy2) % gridH + gridH) % gridH;
        const d   = Math.abs(dx2) + Math.abs(dy2);
        let target = -1;
        const footprintOwner = bodyMap[ny2 * gridW + nx2];
        if (
          footprintOwner >= 0
          && entities.alive[footprintOwner]
          && !this.isSameBodyAlliance(i, footprintOwner)
          && this.genomeSimilarity(i, footprintOwner) < kinThresh
        ) {
          target = footprintOwner;
        } else {
          const centerTarget = entityMap[ny2 * gridW + nx2];
          if (
            centerTarget >= 0
            && entities.alive[centerTarget]
            && this.genomeSimilarity(i, centerTarget) < kinThresh
          ) {
            target = centerTarget;
          }
        }
        if (target < 0) continue;
        if (d < minDist) {
          minDist = d;
          bestTarget = target;
        }
      }
    }

    const target = bestTarget;
    if (target >= 0 && entities.alive[target]) {
      const attackMul = 1 + Math.min(0.45, Math.max(0, entities.size[i] - 1) * 0.22);
      const targetResist =
        1
        + Math.min(0.35, Math.max(0, entities.size[target] - 1) * 0.18)
        + Math.min(0.28, Math.max(0, this.getColonyMemberCount(target) - 1) * 0.05);
      const stolen = entities.energy[target] * laws.attackTransfer * attackMul / targetResist;
      entities.energy[i]      += stolen;
      entities.energy[target] -= stolen;

      if (entities.energy[target] <= 0) {
        entities.energy[i] += 0.45;
        this.killEntity(target);
      }
      entities.energy[i] = Math.min(this.getEffectiveEnergyCap(i), entities.energy[i]);
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
      const start = c * 3;
      let sum = 0;
      let count = 0;
      for (let j = 0; j < 3; j++) {
        const idx = start + j;
        if (idx >= NN_HIDDEN) break;
        sum += entities.memory[mOff + idx] || 0;
        count++;
      }
      const deposit = count > 0 ? sum / count : 0;
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
      const g = glyphs[cellBase + c];
      const start = c * 3;
      for (let j = 0; j < 3; j++) {
        const idx = start + j;
        if (idx >= NN_HIDDEN) break;
        entities.memory[mOff + idx] = entities.memory[mOff + idx] * (1 - rate) + g * rate;
      }
    }
    // Hidden units 8,9 (last pair from 5th channel slot) — leave unchanged since GLYPH_CHANNELS=4
    this.tickAbsorbs++;
  }

  private senseEntityForNetwork(i: number): {
    localSignal: number;
    dirRes: Float64Array;
    dirEnt: Float64Array;
    dirGlyph: Float64Array;
    dirSignal: Float64Array;
    kinNeighbors: number;
    threatNeighbors: number;
  } {
    const { entities, gridW, gridH, laws } = this;
    const kinThresh = laws.kinThreshold ?? 0.8;
    const perceptionRadius = this.getEffectivePerceptionRadius(i);
    const cx = entities.x[i];
    const cy = entities.y[i];
    const dirRes = new Float64Array(4);
    const dirEnt = new Float64Array(4);
    const dirGlyph = new Float64Array(4);
    const dirSignal = new Float64Array(4);
    const dirCount = new Int32Array(4);
    let kinNeighbors = 0;
    let threatNeighbors = 0;
    let localSignal = 0;
    const localSignalBase = (cy * gridW + cx) * laws.signalChannels;
    for (let c = 0; c < laws.signalChannels; c++) {
      localSignal += this.signals[localSignalBase + c];
    }
    localSignal = laws.signalChannels > 0 ? Math.min(1, localSignal / laws.signalChannels) : 0;

    for (let dy0 = -perceptionRadius; dy0 <= perceptionRadius; dy0++) {
      for (let dx0 = -perceptionRadius; dx0 <= perceptionRadius; dx0++) {
        if (dx0 === 0 && dy0 === 0) continue;
        const nx0 = ((cx + dx0) % gridW + gridW) % gridW;
        const ny0 = ((cy + dy0) % gridH + gridH) % gridH;
        const adx = dx0 < 0 ? -dx0 : dx0;
        const ady = dy0 < 0 ? -dy0 : dy0;
        const dir = adx >= ady ? (dx0 > 0 ? 1 : 3) : (dy0 > 0 ? 2 : 0);
        dirCount[dir]++;
        const nIdx = ny0 * gridW + nx0;
        dirRes[dir] += this.resources[nIdx];

        const neighbor = this.entityMap[nIdx];
        if (neighbor >= 0 && this.entities.alive[neighbor]) {
          dirEnt[dir]++;
          if (this.genomeSimilarity(i, neighbor) >= kinThresh) kinNeighbors++;
          else threatNeighbors++;
        }

        let signal = 0;
        const sBase = nIdx * laws.signalChannels;
        for (let c = 0; c < laws.signalChannels; c++) {
          signal += this.signals[sBase + c];
        }
        dirSignal[dir] += laws.signalChannels > 0 ? signal / laws.signalChannels : 0;

        const gBase = nIdx * GLYPH_CHANNELS;
        let gMag = 0;
        for (let c = 0; c < GLYPH_CHANNELS; c++) {
          const g = this.glyphs[gBase + c];
          gMag += g * g;
        }
        dirGlyph[dir] += Math.sqrt(gMag);
      }
    }

    for (let d = 0; d < 4; d++) {
      if (dirCount[d] > 0) {
        dirRes[d] /= dirCount[d];
        dirEnt[d] /= dirCount[d];
        dirSignal[d] = Math.min(1, dirSignal[d] / dirCount[d]);
        dirGlyph[d] = Math.min(1, dirGlyph[d] / dirCount[d]);
      }
    }

    return { localSignal, dirRes, dirEnt, dirGlyph, dirSignal, kinNeighbors, threatNeighbors };
  }

  private buildNeuralSample(i: number): NeuralSample {
    const { entities, resources, glyphs, gridW, laws } = this;
    const genome = entities.getGenome(i);
    const ex = entities.x[i];
    const ey = entities.y[i];
    const { localSignal, dirRes, dirEnt, dirGlyph, dirSignal, kinNeighbors, threatNeighbors } = this.senseEntityForNetwork(i);

    const cellBase = (ey * gridW + ex) * GLYPH_CHANNELS;
    let glyphMag = 0;
    for (let c = 0; c < GLYPH_CHANNELS; c++) {
      const g = glyphs[cellBase + c];
      glyphMag += g * g;
    }

    const inputs = new Array<number>(NN_INPUTS).fill(0);
    inputs[0] = resources[ey * gridW + ex];
    inputs[1] = Math.min(1, entities.energy[i] / (laws.energyCap ?? 1.5));
    inputs[2] = Math.min(1, Math.sqrt(glyphMag));
    inputs[3] = localSignal;
    inputs[4] = dirRes[0]; inputs[5] = dirRes[1]; inputs[6] = dirRes[2]; inputs[7] = dirRes[3];
    inputs[8] = dirEnt[0]; inputs[9] = dirEnt[1]; inputs[10] = dirEnt[2]; inputs[11] = dirEnt[3];
    inputs[12] = Math.min(1, dirSignal[0] * 0.9 + dirGlyph[0] * 0.7);
    inputs[13] = Math.min(1, dirSignal[1] * 0.9 + dirGlyph[1] * 0.7);
    inputs[14] = Math.min(1, dirSignal[2] * 0.9 + dirGlyph[2] * 0.7);
    inputs[15] = Math.min(1, dirSignal[3] * 0.9 + dirGlyph[3] * 0.7);

    const hidden1 = new Array<number>(NN_HIDDEN_1).fill(0);
    for (let j = 0; j < NN_HIDDEN_1; j++) {
      let sum = 0;
      for (let k = 0; k < NN_INPUTS; k++) {
        sum += genome[k * NN_HIDDEN_1 + j] * inputs[k];
      }
      hidden1[j] = Math.tanh(sum);
    }

    const hidden2 = new Array<number>(NN_HIDDEN_2).fill(0);
    for (let j = 0; j < NN_HIDDEN_2; j++) {
      let sum = 0;
      for (let h = 0; h < NN_HIDDEN_1; h++) {
        sum += genome[NN_W1_SIZE + h * NN_HIDDEN_2 + j] * hidden1[h];
      }
      hidden2[j] = Math.tanh(sum);
    }

    const persistence = Math.max(0, Math.min(0.92, laws.memoryPersistence));
    const persistedUnits = Math.max(0, Math.min(NN_HIDDEN, laws.memorySize ?? NN_HIDDEN));
    const mOff = i * MAX_MEMORY_SIZE;
    if (persistedUnits > 0 && persistence > 0 && entities.age[i] > 1) {
      for (let j = 0; j < persistedUnits; j++) {
        hidden2[j] = hidden2[j] * (1 - persistence) + entities.memory[mOff + j] * persistence;
      }
    }

    const logits = new Array<number>(NN_OUTPUTS).fill(0);
    for (let a = 0; a < NN_OUTPUTS; a++) {
      let sum = 0;
      for (let j = 0; j < NN_HIDDEN_2; j++) {
        sum += genome[NN_W3_OFFSET + j * NN_OUTPUTS + a] * hidden2[j];
      }
      logits[a] = sum;
    }
    let maxLogit = logits[0];
    for (let a = 1; a < NN_OUTPUTS; a++) {
      if (logits[a] > maxLogit) maxLogit = logits[a];
    }
    let expSum = 0;
    const probs = new Array<number>(NN_OUTPUTS).fill(0);
    for (let a = 0; a < NN_OUTPUTS; a++) {
      probs[a] = Math.exp(logits[a] - maxLogit);
      expSum += probs[a];
    }
    for (let a = 0; a < NN_OUTPUTS; a++) probs[a] /= expSum || 1;

    const action = entities.action[i];
    const ageScore = Math.min(1, entities.age[i] / 450);
    const activeBonus = (
      action === ActionType.ATTACK ||
      action === ActionType.SIGNAL ||
      action === ActionType.DEPOSIT ||
      action === ActionType.ABSORB
    ) ? 0.08 : action !== ActionType.IDLE && action !== ActionType.REPRODUCE ? 0.04 : 0;
    const socialScore = Math.min(1, (kinNeighbors + threatNeighbors) / 6) * 0.08;
    const focusScore =
      Math.min(1, entities.energy[i] / (laws.energyCap ?? 1.5)) * 0.46 +
      Math.min(1, entities.size[i] / 1.8) * 0.24 +
      ageScore * 0.14 +
      Math.max(...probs) * 0.08 +
      activeBonus +
      socialScore;

    return {
      entityId: entities.id[i],
      action,
      age: entities.age[i],
      energy: entities.energy[i],
      size: entities.size[i],
      kinNeighbors,
      threatNeighbors,
      focusScore,
      inputs,
      hidden1,
      hidden2,
      probs,
      genome: Array.from(genome),
    };
  }

  private scoreNeuralFocusCandidate(i: number): number {
    const { entities, laws } = this;
    const action = entities.action[i];
    const activeBonus = (
      action === ActionType.ATTACK ||
      action === ActionType.SIGNAL ||
      action === ActionType.DEPOSIT ||
      action === ActionType.ABSORB
    ) ? 0.10 : action !== ActionType.IDLE && action !== ActionType.REPRODUCE ? 0.05 : 0;
    return (
      Math.min(1, entities.energy[i] / (laws.energyCap ?? 1.5)) * 0.52 +
      Math.min(1, entities.size[i] / 1.8) * 0.24 +
      Math.min(1, entities.age[i] / 450) * 0.18 +
      activeBonus
    );
  }

  getNeuralSampleById(entityId: number): NeuralSample | null {
    const { entities } = this;
    for (let i = 0; i < entities.count; i++) {
      if (entities.alive[i] && entities.id[i] === entityId) {
        return this.buildNeuralSample(i);
      }
    }
    return null;
  }

  getTopNeuralSample(): NeuralSample | null {
    const { entities } = this;
    let bestIdx = -1;
    let bestScore = -1;
    for (let i = 0; i < entities.count; i++) {
      if (!entities.alive[i]) continue;
      const score = this.scoreNeuralFocusCandidate(i);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    return bestIdx >= 0 ? this.buildNeuralSample(bestIdx) : null;
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

    // M5: record harvest efficiency (eats/age) bucketed by age quartile.
    // Only count entities that lived long enough to have meaningful data (age > 20).
    const age = entities.age[i];
    if (age > 20) {
      const efficiency = entities.eatCount[i] / age;
      const ageQ = age / (this.laws.maxAge || 500); // normalised 0–1 lifetime
      if (ageQ < 0.25) {
        this.tickQ1EatSum += efficiency; this.tickQ1Count++;
      } else if (ageQ >= 0.75) {
        this.tickQ4EatSum += efficiency; this.tickQ4Count++;
      }
    }

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
    let totalSize = 0;
    let maxSize = 0;
    let largeOrganisms = 0;
    for (let i = 0; i < n; i++) totalEnergy += entities.energy[i];
    for (let i = 0; i < n; i++) {
      const size = entities.size[i];
      totalSize += size;
      if (size > maxSize) maxSize = size;
      if (size >= 1.8) largeOrganisms++;
    }

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
      meanSize: n > 0 ? totalSize / n : 0,
      maxSize,
      largeOrganisms,
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
      fusedMembers: this.tickFusedMembers,
      largestColony: this.tickLargestColony,
      colonyBirths: this.tickColonyBirths,
      poisonCoverage: poisonedCells / this.poison.length,
      harvestEfficiencyRatio: (this.tickQ1Count > 0 && this.tickQ4Count > 0)
        ? (this.tickQ4EatSum / this.tickQ4Count) / (this.tickQ1EatSum / this.tickQ1Count)
        : 1.0,
    };
  }

  /** Get current state for visualisation. */
  getVisualState(): {
    resources: Float32Array;
    signals: Float32Array;
    poison: Float32Array;
    glyphs: Float32Array;
    bodyStrength: Float32Array;
    entityX: Int32Array;
    entityY: Int32Array;
    entityEnergy: Float32Array;
    entitySize: Float32Array;
    entityColonyMass: Uint8Array;
    entityBodyRadius: Uint8Array;
    entityAction: Uint8Array;
    entityGenomes: Float32Array;
    entityCount: number;
    gridW: number;
    gridH: number;
    signalChannels: number;
  } {
    this.prepareVisualState();
    return {
      resources:     this.resources,
      signals:       this.signals,
      poison:        this.poison,
      glyphs:        this.glyphs,
      bodyStrength:  this.bodyStrength,
      entityX:       this.entities.x.subarray(0, this.entities.count),
      entityY:       this.entities.y.subarray(0, this.entities.count),
      entityEnergy:  this.entities.energy.subarray(0, this.entities.count),
      entitySize:    this.entities.size.subarray(0, this.entities.count),
      entityColonyMass: this.visualColonyMass.subarray(0, this.entities.count),
      entityBodyRadius: this.visualBodyRadius.subarray(0, this.entities.count),
      entityAction:  this.entities.action.subarray(0, this.entities.count),
      entityGenomes: this.entities.genomes,
      entityCount:   this.entities.count,
      gridW:         this.gridW,
      gridH:         this.gridH,
      signalChannels: this.laws.signalChannels,
    };
  }
}
