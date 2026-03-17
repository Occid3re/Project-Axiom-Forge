/**
 * World simulation engine.
 * Runs the grid-based world with entities, resources, signals.
 * All hot-path operations use typed arrays — no object allocation per tick.
 */

import { GENOME_LENGTH, Gene, ActionType, ResourceDist, MAX_MEMORY_SIZE } from './constants';
import { EntityPool } from './entity-pool';
import { PRNG } from './world-laws';
import type { WorldLaws } from './world-laws';

export interface WorldSnapshot {
  tick: number;
  population: number;
  meanEnergy: number;
  totalEnergy: number;
  diversity: number;
  signalActivity: number;
  resourceCoverage: number;
  births: number;
  deaths: number;
  attacks: number;
  signals: number;
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
  readonly entityMap: Int16Array; // -1 or entity index at each cell
  readonly entities: EntityPool;

  tick: number = 0;

  // Per-tick counters
  private tickBirths = 0;
  private tickDeaths = 0;
  private tickAttacks = 0;
  private tickSignals = 0;

  constructor(laws: WorldLaws, config: WorldConfig, seed: number) {
    this.laws = laws;
    this.config = config;
    this.rng = new PRNG(seed);
    this.gridW = config.gridSize;
    this.gridH = config.gridSize;

    const cellCount = this.gridW * this.gridH;
    this.resources = new Float32Array(cellCount);
    this.resourceCapacity = new Float32Array(cellCount);
    this.signals = new Float32Array(cellCount * laws.signalChannels);
    this.entityMap = new Int16Array(cellCount).fill(-1);
    this.entities = new EntityPool();

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
            // Several resource hotspots
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
      if (idx >= 0) {
        entityMap[y * gridW + x] = idx;
      }
    }
  }

  step(): WorldSnapshot {
    this.tick++;
    this.tickBirths = 0;
    this.tickDeaths = 0;
    this.tickAttacks = 0;
    this.tickSignals = 0;

    this.decaySignals();
    this.regenerateResources();
    this.maybeDisaster();
    this.processEntities();
    this.removeDeadEntities();

    return this.snapshot();
  }

  run(): WorldHistory {
    const snapshots: WorldSnapshot[] = [];
    let peakPop = 0;
    let disasterCount = 0;
    let postDisasterRecoveries = 0;
    let lastDisasterTick = -100;
    let popAtDisaster = 0;

    for (let t = 0; t < this.config.steps; t++) {
      const snap = this.step();
      snapshots.push(snap);

      if (snap.population > peakPop) peakPop = snap.population;

      // Track disasters (detected by sudden resource drop, handled in step)
      if (this.tick === lastDisasterTick + 50 && snap.population > popAtDisaster * 0.5) {
        postDisasterRecoveries++;
      }
    }

    return {
      snapshots,
      finalPopulation: this.entities.count,
      peakPopulation: peakPop,
      disasterCount,
      postDisasterRecoveries,
    };
  }

  private decaySignals(): void {
    const decay = this.laws.signalDecay;
    const len = this.signals.length;
    for (let i = 0; i < len; i++) {
      this.signals[i] *= decay;
      if (this.signals[i] < 0.001) this.signals[i] = 0;
    }
  }

  private regenerateResources(): void {
    const rate = this.laws.resourceRegenRate;
    const len = this.resources.length;
    for (let i = 0; i < len; i++) {
      this.resources[i] += rate * (this.resourceCapacity[i] - this.resources[i]);
      if (this.resources[i] > this.resourceCapacity[i]) {
        this.resources[i] = this.resourceCapacity[i];
      }
    }
  }

  private maybeDisaster(): void {
    if (this.rng.random() < this.laws.disasterProbability) {
      // Wipe resources in a random region
      const cx = this.rng.int(0, this.gridW - 1);
      const cy = this.rng.int(0, this.gridH - 1);
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

    // Shuffled processing order
    const order = new Int32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = rng.int(0, i);
      const tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    for (let oi = 0; oi < n; oi++) {
      const i = order[oi];
      if (!entities.alive[i]) continue;

      // Age and metabolic cost
      entities.age[i]++;
      entities.energy[i] -= laws.idleCost;
      if (entities.energy[i] <= 0) {
        this.killEntity(i);
        continue;
      }

      // Overcrowding death — density-dependent mortality caps population
      // > 4 occupied Moore-neighbours = suffocating; energy drains fast
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
          // Fixed overcrowding cost — independent of idleCost so it always bites
          entities.energy[i] -= 0.04 * (nCount - 2);
          if (entities.energy[i] <= 0) { this.killEntity(i); continue; }
        }
      }

      // Decide action
      const action = this.decideAction(i);
      entities.action[i] = action;

      switch (action) {
        case ActionType.MOVE:
          this.executeMove(i);
          break;
        case ActionType.EAT:
          this.executeEat(i);
          break;
        case ActionType.REPRODUCE:
          this.executeReproduce(i);
          break;
        case ActionType.SIGNAL:
          this.executeSignal(i);
          break;
        case ActionType.ATTACK:
          this.executeAttack(i);
          break;
        // IDLE: do nothing
      }

      // Update memory
      this.updateMemory(i);
    }
  }

  private decideAction(i: number): ActionType {
    const { entities, rng, laws, resources, signals, gridW, gridH } = this;
    const g = entities.getGenome(i);
    const mem = entities.getMemory(i, laws.memorySize);

    const ex = entities.x[i];
    const ey = entities.y[i];
    const cellIdx = ey * gridW + ex;

    // Local perception
    const localResource = resources[cellIdx];
    let nearbyEntities = 0;
    let nearbySignal = 0;
    const radius = Math.max(1, Math.min(
      Math.floor(g[Gene.PERCEPTION_RANGE] * laws.maxPerceptionRadius) + 1,
      laws.maxPerceptionRadius
    ));
    const area = (2 * radius + 1) * (2 * radius + 1) - 1;

    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = ((ex + dx) % gridW + gridW) % gridW;
        const ny = ((ey + dy) % gridH + gridH) % gridH;
        const nIdx = ny * gridW + nx;
        if (this.entityMap[nIdx] >= 0) nearbyEntities++;
        for (let c = 0; c < laws.signalChannels; c++) {
          nearbySignal += signals[nIdx * laws.signalChannels + c];
        }
      }
    }

    const entityDensity = nearbyEntities / area;
    const signalStrength = nearbySignal / (area * laws.signalChannels || 1);
    const memInfluence = mem.length > 0 ? mem[0] * g[Gene.MEMORY_READ_WEIGHT] : 0;
    const noise = rng.normal(0, g[Gene.MOVE_RANDOMNESS] * 0.5);

    // Score each action
    const scores = new Float32Array(6);

    // IDLE
    scores[ActionType.IDLE] = g[Gene.ENERGY_CONSERVATISM] * (1 - entities.energy[i]) * 0.5;

    // MOVE
    scores[ActionType.MOVE] = g[Gene.EXPLORE_EXPLOIT] * 0.5 +
      (1 - localResource) * g[Gene.EXPLORE_EXPLOIT] + noise;

    // EAT
    scores[ActionType.EAT] = g[Gene.EAT_PRIORITY] * localResource * 3.0 +
      (1 - entities.energy[i]) * g[Gene.ENERGY_CONSERVATISM];

    // REPRODUCE
    const energyRatio = entities.energy[i] / Math.max(laws.reproductionCost, 0.01);
    scores[ActionType.REPRODUCE] = Math.max(0, energyRatio - g[Gene.REPRO_THRESHOLD]) * 2.0;

    // SIGNAL
    scores[ActionType.SIGNAL] = g[Gene.SIGNAL_STRENGTH] * 0.3 +
      signalStrength * g[Gene.SIGNAL_RESPONSIVENESS] * 0.3;

    // ATTACK
    scores[ActionType.ATTACK] = g[Gene.AGGRESSION] * entityDensity * 2.0;

    // Predator drive (ADAPTATION_RATE gene): specialize in hunting rather than eating plants
    const predatorDrive = g[Gene.ADAPTATION_RATE];
    if (predatorDrive > 0.4 && nearbyEntities > 0) {
      scores[ActionType.ATTACK]    += predatorDrive * entityDensity * 6.0;
      scores[ActionType.EAT]       *= 1.0 - predatorDrive * 0.7; // predators ignore resources
      scores[ActionType.REPRODUCE] *= 1.0 - predatorDrive * 0.4; // reproduce less while hunting
    }

    // Overcrowding suppresses reproduction — no point reproducing into a full grid
    if (entityDensity > 0.4) {
      scores[ActionType.REPRODUCE] *= Math.max(0, 1.0 - entityDensity * 1.4);
    }

    // Cooperation: if nearby entities share similar genome, reduce aggression and boost signaling
    if (g[Gene.COOPERATION] > 0.3 && nearbyEntities > 0) {
      // Sample one nearby entity for genome similarity
      let neighborGenomeMatch = 0;
      for (let dy2 = -1; dy2 <= 1; dy2++) {
        for (let dx2 = -1; dx2 <= 1; dx2++) {
          if (dx2 === 0 && dy2 === 0) continue;
          const nx2 = ((ex + dx2) % gridW + gridW) % gridW;
          const ny2 = ((ey + dy2) % gridH + gridH) % gridH;
          const nb = this.entityMap[ny2 * gridW + nx2];
          if (nb >= 0 && entities.alive[nb]) {
            const ng = entities.getGenome(nb);
            let dist = 0;
            for (let gi = 0; gi < 4; gi++) { // compare first 4 genes only for speed
              const d = g[gi] - ng[gi];
              dist += d * d;
            }
            neighborGenomeMatch = Math.max(neighborGenomeMatch, 1 - Math.sqrt(dist) / 2);
            break;
          }
        }
        if (neighborGenomeMatch > 0) break;
      }
      const cooperationBonus = g[Gene.COOPERATION] * neighborGenomeMatch;
      scores[ActionType.ATTACK]   -= cooperationBonus * 2.0;  // kin selection: don't attack relatives
      scores[ActionType.SIGNAL]   += cooperationBonus * 1.5;  // kin signaling
      scores[ActionType.REPRODUCE]+= cooperationBonus * 0.5;  // cooperative reproduction pressure
    }

    // Add memory influence
    for (let a = 0; a < 6; a++) scores[a] += memInfluence * 0.1;

    // Softmax selection
    let maxScore = scores[0];
    for (let a = 1; a < 6; a++) if (scores[a] > maxScore) maxScore = scores[a];

    let expSum = 0;
    for (let a = 0; a < 6; a++) {
      scores[a] = Math.exp((scores[a] - maxScore) * 5.0);
      expSum += scores[a];
    }

    let r = rng.random() * expSum;
    for (let a = 0; a < 6; a++) {
      r -= scores[a];
      if (r <= 0) return a as ActionType;
    }
    return ActionType.IDLE;
  }

  private executeMove(i: number): void {
    const { entities, laws, gridW, gridH, entityMap, rng } = this;
    const g = entities.getGenome(i);

    let dx = Math.sign(g[Gene.MOVE_BIAS_X] - 0.5 + rng.normal(0, g[Gene.MOVE_RANDOMNESS] * 0.5));
    let dy = Math.sign(g[Gene.MOVE_BIAS_Y] - 0.5 + rng.normal(0, g[Gene.MOVE_RANDOMNESS] * 0.5));
    if (dx === 0 && dy === 0) dx = rng.int(-1, 1);

    // Predators chase nearest entity within radius 2
    const predatorDrive = g[Gene.ADAPTATION_RATE];
    if (predatorDrive > 0.5) {
      const ox = entities.x[i], oy = entities.y[i];
      let minDist = 99, preyDx = 0, preyDy = 0;
      for (let dy2 = -2; dy2 <= 2; dy2++) {
        for (let dx2 = -2; dx2 <= 2; dx2++) {
          if (dx2 === 0 && dy2 === 0) continue;
          const nx2 = ((ox + dx2) % gridW + gridW) % gridW;
          const ny2 = ((oy + dy2) % gridH + gridH) % gridH;
          const d = Math.abs(dx2) + Math.abs(dy2);
          if (entityMap[ny2 * gridW + nx2] >= 0 && d < minDist) {
            minDist = d; preyDx = dx2; preyDy = dy2;
          }
        }
      }
      if (minDist < 99 && rng.random() < predatorDrive) {
        dx = Math.sign(preyDx) || (rng.random() > 0.5 ? 1 : -1);
        dy = Math.sign(preyDy) || (rng.random() > 0.5 ? 1 : -1);
      }
    }

    dx = Math.max(-1, Math.min(1, Math.round(dx)));
    dy = Math.max(-1, Math.min(1, Math.round(dy)));

    const ox = entities.x[i];
    const oy = entities.y[i];
    const nx = ((ox + dx) % gridW + gridW) % gridW;
    const ny = ((oy + dy) % gridH + gridH) % gridH;

    const newCell = ny * gridW + nx;
    if (this.entityMap[newCell] < 0) {
      entityMap[oy * gridW + ox] = -1;
      entities.x[i] = nx;
      entities.y[i] = ny;
      entityMap[newCell] = i;
      entities.actionDx[i] = dx;
      entities.actionDy[i] = dy;
    }

    entities.energy[i] -= laws.moveCost;
  }

  private executeEat(i: number): void {
    const { entities, laws, resources, gridW } = this;
    const cellIdx = entities.y[i] * gridW + entities.x[i];
    const available = resources[cellIdx];
    const gain = Math.min(available, laws.eatGain);
    entities.energy[i] += gain;
    resources[cellIdx] -= gain;
    if (entities.energy[i] > 1.5) entities.energy[i] = 1.5;
  }

  private executeReproduce(i: number): void {
    const { entities, laws, rng, gridW, gridH, entityMap } = this;

    if (entities.energy[i] < laws.reproductionCost) return;
    // Hard population cap — 7% of grid cells prevents exponential explosion
    if (entities.count >= Math.floor(gridW * gridH * 0.07)) return;

    // Find empty adjacent cell
    const ox = entities.x[i];
    const oy = entities.y[i];
    let placed = false;

    for (let attempt = 0; attempt < 8; attempt++) {
      const dx = rng.int(-1, 1);
      const dy = rng.int(-1, 1);
      if (dx === 0 && dy === 0) continue;

      const nx = ((ox + dx) % gridW + gridW) % gridW;
      const ny = ((oy + dy) % gridH + gridH) % gridH;
      const cell = ny * gridW + nx;

      if (entityMap[cell] < 0) {
        // Sexual reproduction: find a nearby mate with different genome
        let mateGenome: Float32Array | null = null;
        if (laws.sexualReproduction) {
          // Scan for nearby entities within radius 2
          outer: for (let mdy = -2; mdy <= 2; mdy++) {
            for (let mdx = -2; mdx <= 2; mdx++) {
              if (mdx === 0 && mdy === 0) continue;
              const mx = ((ox + mdx) % gridW + gridW) % gridW;
              const my = ((oy + mdy) % gridH + gridH) % gridH;
              const mCell = my * gridW + mx;
              const mate = this.entityMap[mCell];
              if (mate >= 0 && mate !== i && entities.alive[mate]) {
                mateGenome = entities.getGenome(mate);
                break outer;
              }
            }
          }
        }

        // Build child genome: crossover if mate found, else clone
        const parentGenome = entities.getGenome(i);
        const childGenome = new Float32Array(GENOME_LENGTH);
        for (let g = 0; g < GENOME_LENGTH; g++) {
          childGenome[g] = (mateGenome && rng.random() > 0.5) ? mateGenome[g] : parentGenome[g];
        }

        const childIdx = entities.spawn(nx, ny, laws.offspringEnergy, childGenome, entities.id[i], rng);
        if (childIdx >= 0) {
          entities.mutateGenome(childIdx, laws.mutationRate, laws.mutationStrength, rng);
          entityMap[cell] = childIdx;
          entities.energy[i] -= laws.reproductionCost;
          this.tickBirths++;
          placed = true;
        }
        break;
      }
    }
  }

  private executeSignal(i: number): void {
    const { entities, laws, signals, gridW, gridH } = this;
    const g = entities.getGenome(i);
    const channel = Math.floor(g[Gene.SIGNAL_CHANNEL] * laws.signalChannels) % laws.signalChannels;
    const strength = g[Gene.SIGNAL_STRENGTH];
    const range = laws.signalRange;
    const ex = entities.x[i];
    const ey = entities.y[i];

    for (let dy = -range; dy <= range; dy++) {
      for (let dx = -range; dx <= range; dx++) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > range) continue;
        const nx = ((ex + dx) % gridW + gridW) % gridW;
        const ny = ((ey + dy) % gridH + gridH) % gridH;
        const attenuation = 1 - dist / (range + 1);
        signals[(ny * gridW + nx) * laws.signalChannels + channel] += strength * attenuation;
      }
    }
    this.tickSignals++;
  }

  private executeAttack(i: number): void {
    const { entities, laws, rng, gridW, gridH, entityMap } = this;
    const g = entities.getGenome(i);
    const predatorDrive = g[Gene.ADAPTATION_RATE];

    // Predators attack the nearest entity they see; others attack randomly
    let tdx: number, tdy: number;
    if (predatorDrive > 0.5) {
      const ox = entities.x[i], oy = entities.y[i];
      let minDist = 99, bestDx = 0, bestDy = 0;
      for (let dy2 = -1; dy2 <= 1; dy2++) {
        for (let dx2 = -1; dx2 <= 1; dx2++) {
          if (dx2 === 0 && dy2 === 0) continue;
          const nx2 = ((ox + dx2) % gridW + gridW) % gridW;
          const ny2 = ((oy + dy2) % gridH + gridH) % gridH;
          const d = Math.abs(dx2) + Math.abs(dy2);
          if (entityMap[ny2 * gridW + nx2] >= 0 && d < minDist) {
            minDist = d; bestDx = dx2; bestDy = dy2;
          }
        }
      }
      tdx = minDist < 99 ? bestDx : rng.int(-1, 1);
      tdy = minDist < 99 ? bestDy : rng.int(-1, 1);
    } else {
      tdx = rng.int(-1, 1);
      tdy = rng.int(-1, 1);
    }
    if (tdx === 0 && tdy === 0) return;

    const nx = ((entities.x[i] + tdx) % gridW + gridW) % gridW;
    const ny = ((entities.y[i] + tdy) % gridH + gridH) % gridH;
    const cell = ny * gridW + nx;
    const target = entityMap[cell];

    if (target >= 0 && entities.alive[target]) {
      const stolen = entities.energy[target] * laws.attackTransfer;
      entities.energy[i] += stolen;
      entities.energy[target] -= stolen;

      if (entities.energy[target] <= 0) {
        // Predators absorb kill bonus — makes hunting a viable energy strategy
        if (predatorDrive > 0.4) {
          entities.energy[i] += predatorDrive * 0.45;
        }
        this.killEntity(target);
      }
      if (entities.energy[i] > 1.5) entities.energy[i] = 1.5;
      this.tickAttacks++;
    }
  }

  private updateMemory(i: number): void {
    const { entities, laws, resources, signals, gridW } = this;
    const g = entities.getGenome(i);
    const mem = entities.getMemory(i, laws.memorySize);
    const writeRate = g[Gene.MEMORY_WRITE_RATE];
    const cellIdx = entities.y[i] * gridW + entities.x[i];

    // Decay
    for (let m = 0; m < laws.memorySize; m++) {
      mem[m] *= laws.memoryPersistence;
    }

    // Write new perceptions
    if (laws.memorySize >= 1) mem[0] += writeRate * resources[cellIdx];
    if (laws.memorySize >= 2) mem[1] += writeRate * entities.energy[i];
    if (laws.memorySize >= 3) {
      let sig = 0;
      for (let c = 0; c < laws.signalChannels; c++) {
        sig += signals[cellIdx * laws.signalChannels + c];
      }
      mem[2] += writeRate * sig;
    }
    if (laws.memorySize >= 4) mem[3] += writeRate * (entities.action[i] / 5.0);

    // Clamp
    for (let m = 0; m < laws.memorySize; m++) {
      if (mem[m] > 1) mem[m] = 1;
      if (mem[m] < -1) mem[m] = -1;
    }
  }

  private killEntity(i: number): void {
    const { entities, entityMap, gridW } = this;
    entityMap[entities.y[i] * gridW + entities.x[i]] = -1;
    entities.kill(i);
    this.tickDeaths++;
  }

  private removeDeadEntities(): void {
    // Rebuild entity map after compaction
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

    // Diversity: sample pairwise genome distances
    let diversity = 0;
    const sampleSize = Math.min(n, 20);
    let pairs = 0;
    for (let a = 0; a < sampleSize; a++) {
      for (let b = a + 1; b < sampleSize; b++) {
        const ga = entities.getGenome(a);
        const gb = entities.getGenome(b);
        let dist = 0;
        for (let g = 0; g < GENOME_LENGTH; g++) {
          const d = ga[g] - gb[g];
          dist += d * d;
        }
        diversity += Math.sqrt(dist);
        pairs++;
      }
    }
    if (pairs > 0) diversity /= pairs;

    // Signal activity
    let signalActivity = 0;
    for (let i = 0; i < this.signals.length; i++) signalActivity += this.signals[i];

    // Resource coverage
    let filledCells = 0;
    for (let i = 0; i < this.resources.length; i++) {
      if (this.resources[i] > 0.1) filledCells++;
    }

    return {
      tick: this.tick,
      population: n,
      meanEnergy: n > 0 ? totalEnergy / n : 0,
      totalEnergy,
      diversity,
      signalActivity,
      resourceCoverage: filledCells / this.resources.length,
      births: this.tickBirths,
      deaths: this.tickDeaths,
      attacks: this.tickAttacks,
      signals: this.tickSignals,
    };
  }

  /** Get current state for visualization. */
  getVisualState(): {
    resources: Float32Array;
    signals: Float32Array;
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
      resources: this.resources,
      signals: this.signals,
      entityX: this.entities.x.subarray(0, this.entities.count),
      entityY: this.entities.y.subarray(0, this.entities.count),
      entityEnergy: this.entities.energy.subarray(0, this.entities.count),
      entityAction: this.entities.action.subarray(0, this.entities.count),
      entityGenomes: this.entities.genomes,
      entityCount: this.entities.count,
      gridW: this.gridW,
      gridH: this.gridH,
      signalChannels: this.laws.signalChannels,
    };
  }
}
