/**
 * World laws — the evolvable "physics" of each simulated world.
 * Plain object with typed fields. Cheap to clone and mutate.
 */

import { ResourceDist } from './constants';

export interface WorldLaws {
  // Reproduction
  reproductionCost: number;
  offspringEnergy: number;
  mutationRate: number;
  mutationStrength: number;
  sexualReproduction: boolean;

  // Energy
  resourceRegenRate: number;
  eatGain: number;
  moveCost: number;
  idleCost: number;
  attackTransfer: number;

  // Communication
  signalRange: number;
  signalChannels: number;
  signalDecay: number;

  // Memory
  memorySize: number;
  memoryPersistence: number;

  // Environment
  resourceDistribution: ResourceDist;
  disasterProbability: number;
  terrainVariability: number;

  // Perception
  maxPerceptionRadius: number;

  // Lifespan — entities die after this many ticks regardless of energy
  maxAge: number;

  // Carrying capacity — fraction of grid cells sustainably occupied.
  // Above this threshold a shared "air" depletion is applied per entity per tick,
  // proportional to how far over capacity the population is.
  carryingCapacity: number;

  // Poison — toxin deposited by dying entities, damages living ones.
  poisonStrength: number;   // 0.0–0.3: damage per tick at full concentration
  deathToxin: number;       // 0.0–0.8: poison deposited when an entity dies

  // Movement & combat
  moveSpeed: number;        // 1–3: cells per MOVE action (fast movers burn more energy)
  attackRange: number;      // 1–3: search radius for attack targets (ranged vs melee)

  // Reproduction & offspring
  spawnDistance: number;     // 1–4: how far offspring appear (colonial vs dispersal)

  // Social dynamics
  cooperationBonus: number; // 0.0–0.15: energy bonus per friendly neighbor per tick (herding)
  crowdingThreshold: number;// 1–6: neighbors before overcrowding penalty kicks in

  // Entity limits
  energyCap: number;        // 0.5–3.0: max energy per entity (tanky vs fragile)
  signalCost: number;       // 0.0–0.05: energy cost per signal action

  // Corpse ecology
  corpseEnergy: number;     // 0.1–1.0: fraction of energy returned to grid on death

  // Aging
  agingRate: number;        // 0.0–0.01: extra energy drain per tick, proportional to age

  // Environment
  driftSpeed: number;       // 0.0–0.4: strength of environmental current pushing entities

  // Stigmergic memory (glyph grid)
  glyphDecay: number;       // 0.990–0.999: per-tick glyph persistence (half-life 693–6931 ticks)
  depositCost: number;      // 0.0–0.03: energy cost per DEPOSIT action
  absorbCost: number;       // 0.0–0.02: energy cost per ABSORB action
  absorbRate: number;       // 0.0–0.3: how much glyph overwrites hidden state on ABSORB

  // Social perception
  kinThreshold: number;     // 0.6–0.95: species similarity cutoff for kin recognition
}

interface FloatRange {
  min: number;
  max: number;
}

const FLOAT_RANGES: Record<string, FloatRange> = {
  reproductionCost: { min: 0.1, max: 1.0 },
  offspringEnergy: { min: 0.05, max: 0.8 },
  mutationRate: { min: 0.01, max: 0.28 },
  mutationStrength: { min: 0.01, max: 0.18 },
  resourceRegenRate: { min: 0.001, max: 0.1 },
  eatGain: { min: 0.1, max: 1.0 },
  moveCost: { min: 0.001, max: 0.1 },
  idleCost: { min: 0.001, max: 0.05 },
  attackTransfer: { min: 0.0, max: 0.65 },
  signalDecay: { min: 0.1, max: 0.99 },
  memoryPersistence: { min: 0.16, max: 0.82 },
  disasterProbability: { min: 0.0, max: 0.015 },
  terrainVariability: { min: 0.0, max: 1.0 },
  carryingCapacity: { min: 0.05, max: 0.24 },
  poisonStrength: { min: 0.0, max: 0.3 },
  deathToxin: { min: 0.0, max: 0.8 },
  cooperationBonus: { min: 0.0, max: 0.02 },
  energyCap: { min: 0.5, max: 3.0 },
  signalCost: { min: 0.0, max: 0.03 },
  corpseEnergy: { min: 0.1, max: 1.0 },
  agingRate: { min: 0.0, max: 0.01 },
  driftSpeed: { min: 0.0, max: 0.14 },
  glyphDecay: { min: 0.990, max: 0.999 },
  depositCost: { min: 0.0, max: 0.02 },
  absorbCost: { min: 0.0, max: 0.012 },
  absorbRate: { min: 0.0, max: 0.3 },
  kinThreshold: { min: 0.72, max: 0.95 },
};

const INT_RANGES: Record<string, { min: number; max: number }> = {
  signalRange: { min: 1, max: 8 },
  signalChannels: { min: 1, max: 6 },
  memorySize: { min: 3, max: 16 },
  maxPerceptionRadius: { min: 1, max: 6 },
  maxAge: { min: 200, max: 800 },
  moveSpeed: { min: 1, max: 3 },
  attackRange: { min: 1, max: 3 },
  spawnDistance: { min: 1, max: 3 },
  crowdingThreshold: { min: 3, max: 8 },
};

// --- Seeded PRNG (xoshiro128**) for deterministic runs ---
export class PRNG {
  private s: Uint32Array;

  constructor(seed: number) {
    // SplitMix32 to initialize state
    this.s = new Uint32Array(4);
    for (let i = 0; i < 4; i++) {
      seed += 0x9e3779b9;
      let z = seed;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      z ^= z >>> 16;
      this.s[i] = z >>> 0;
    }
  }

  next(): number {
    const s = this.s;
    const result = Math.imul(rotl(Math.imul(s[1], 5), 7), 9) >>> 0;
    const t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 11);
    return result;
  }

  /** Returns float in [0, 1) */
  random(): number {
    return (this.next() >>> 0) / 4294967296;
  }

  /** Returns float in [min, max) */
  uniform(min: number, max: number): number {
    return min + this.random() * (max - min);
  }

  /** Returns int in [min, max] inclusive */
  int(min: number, max: number): number {
    return min + (this.next() % (max - min + 1));
  }

  /** Gaussian via Box-Muller */
  normal(mean: number = 0, std: number = 1): number {
    const u1 = this.random() || 1e-10;
    const u2 = this.random();
    return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

// --- Factory functions ---

export function randomLaws(rng: PRNG): WorldLaws {
  return {
    reproductionCost: rng.uniform(FLOAT_RANGES.reproductionCost.min, FLOAT_RANGES.reproductionCost.max),
    offspringEnergy: rng.uniform(FLOAT_RANGES.offspringEnergy.min, FLOAT_RANGES.offspringEnergy.max),
    mutationRate: rng.uniform(FLOAT_RANGES.mutationRate.min, FLOAT_RANGES.mutationRate.max),
    mutationStrength: rng.uniform(FLOAT_RANGES.mutationStrength.min, FLOAT_RANGES.mutationStrength.max),
    sexualReproduction: rng.random() > 0.5,
    resourceRegenRate: rng.uniform(FLOAT_RANGES.resourceRegenRate.min, FLOAT_RANGES.resourceRegenRate.max),
    eatGain: rng.uniform(FLOAT_RANGES.eatGain.min, FLOAT_RANGES.eatGain.max),
    moveCost: rng.uniform(FLOAT_RANGES.moveCost.min, FLOAT_RANGES.moveCost.max),
    idleCost: rng.uniform(FLOAT_RANGES.idleCost.min, FLOAT_RANGES.idleCost.max),
    attackTransfer: rng.uniform(FLOAT_RANGES.attackTransfer.min, FLOAT_RANGES.attackTransfer.max),
    signalRange: rng.int(INT_RANGES.signalRange.min, INT_RANGES.signalRange.max),
    signalChannels: rng.int(INT_RANGES.signalChannels.min, INT_RANGES.signalChannels.max),
    signalDecay: rng.uniform(FLOAT_RANGES.signalDecay.min, FLOAT_RANGES.signalDecay.max),
    memorySize: rng.int(INT_RANGES.memorySize.min, INT_RANGES.memorySize.max),
    memoryPersistence: rng.uniform(FLOAT_RANGES.memoryPersistence.min, FLOAT_RANGES.memoryPersistence.max),
    resourceDistribution: rng.int(0, 2) as ResourceDist,
    disasterProbability: rng.uniform(FLOAT_RANGES.disasterProbability.min, FLOAT_RANGES.disasterProbability.max),
    terrainVariability: rng.uniform(FLOAT_RANGES.terrainVariability.min, FLOAT_RANGES.terrainVariability.max),
    maxPerceptionRadius: rng.int(INT_RANGES.maxPerceptionRadius.min, INT_RANGES.maxPerceptionRadius.max),
    maxAge: rng.int(INT_RANGES.maxAge.min, INT_RANGES.maxAge.max),
    carryingCapacity: rng.uniform(FLOAT_RANGES.carryingCapacity.min, FLOAT_RANGES.carryingCapacity.max),
    poisonStrength: rng.uniform(FLOAT_RANGES.poisonStrength.min, FLOAT_RANGES.poisonStrength.max),
    deathToxin: rng.uniform(FLOAT_RANGES.deathToxin.min, FLOAT_RANGES.deathToxin.max),
    moveSpeed: rng.int(INT_RANGES.moveSpeed.min, INT_RANGES.moveSpeed.max),
    attackRange: rng.int(INT_RANGES.attackRange.min, INT_RANGES.attackRange.max),
    spawnDistance: rng.int(INT_RANGES.spawnDistance.min, INT_RANGES.spawnDistance.max),
    cooperationBonus: rng.uniform(FLOAT_RANGES.cooperationBonus.min, FLOAT_RANGES.cooperationBonus.max),
    crowdingThreshold: rng.int(INT_RANGES.crowdingThreshold.min, INT_RANGES.crowdingThreshold.max),
    energyCap: rng.uniform(FLOAT_RANGES.energyCap.min, FLOAT_RANGES.energyCap.max),
    signalCost: rng.uniform(FLOAT_RANGES.signalCost.min, FLOAT_RANGES.signalCost.max),
    corpseEnergy: rng.uniform(FLOAT_RANGES.corpseEnergy.min, FLOAT_RANGES.corpseEnergy.max),
    agingRate: rng.uniform(FLOAT_RANGES.agingRate.min, FLOAT_RANGES.agingRate.max),
    driftSpeed: rng.uniform(FLOAT_RANGES.driftSpeed.min, FLOAT_RANGES.driftSpeed.max),
    glyphDecay: rng.uniform(FLOAT_RANGES.glyphDecay.min, FLOAT_RANGES.glyphDecay.max),
    depositCost: rng.uniform(FLOAT_RANGES.depositCost.min, FLOAT_RANGES.depositCost.max),
    absorbCost: rng.uniform(FLOAT_RANGES.absorbCost.min, FLOAT_RANGES.absorbCost.max),
    absorbRate: rng.uniform(FLOAT_RANGES.absorbRate.min, FLOAT_RANGES.absorbRate.max),
    kinThreshold: rng.uniform(FLOAT_RANGES.kinThreshold.min, FLOAT_RANGES.kinThreshold.max),
  };
}

export function mutateLaws(laws: WorldLaws, rng: PRNG, strength: number = 0.1): WorldLaws {
  const result = { ...laws };

  for (const [key, range] of Object.entries(FLOAT_RANGES)) {
    const k = key as keyof WorldLaws;
    const val = result[k] as number;
    const noise = rng.normal(0, strength * (range.max - range.min));
    (result as any)[k] = Math.max(range.min, Math.min(range.max, val + noise));
  }

  for (const [key, range] of Object.entries(INT_RANGES)) {
    const k = key as keyof WorldLaws;
    if (rng.random() < 0.2) {
      const delta = rng.random() > 0.5 ? 1 : -1;
      (result as any)[k] = Math.max(range.min, Math.min(range.max, (result[k] as number) + delta));
    }
  }

  if (rng.random() < 0.1) result.sexualReproduction = !result.sexualReproduction;
  if (rng.random() < 0.1) result.resourceDistribution = rng.int(0, 2) as ResourceDist;

  return result;
}

/**
 * Hand-tuned starter laws — produce interesting visible behaviour from tick 1.
 * Used as the initial display world before meta-evolution finds anything good.
 */
export function starterLaws(): WorldLaws {
  return {
    reproductionCost:     0.28,
    offspringEnergy:      0.20,
    mutationRate:         0.06,
    mutationStrength:     0.06,
    sexualReproduction:   true,
    resourceRegenRate:    0.028,
    eatGain:              0.42,
    moveCost:             0.007,
    idleCost:             0.004,
    attackTransfer:       0.50,
    signalRange:          4,
    signalChannels:       3,
    signalDecay:          0.80,
    memorySize:           4,
    memoryPersistence:    0.65,
    resourceDistribution: ResourceDist.CLUSTERED,
    disasterProbability:  0.003,
    terrainVariability:   0.65,
    maxPerceptionRadius:  3,
    maxAge:               300,
    carryingCapacity:     0.10,
    poisonStrength:       0.05,
    deathToxin:           0.25,
    moveSpeed:            1,
    attackRange:          1,
    spawnDistance:         1,
    cooperationBonus:     0.005,
    crowdingThreshold:    3,
    energyCap:            1.5,
    signalCost:           0.01,
    corpseEnergy:         0.50,
    agingRate:            0.002,
    driftSpeed:           0.05,
    glyphDecay:           0.996,
    depositCost:          0.01,
    absorbCost:           0.005,
    absorbRate:           0.1,
    kinThreshold:         0.8,
  };
}

export function crossoverLaws(a: WorldLaws, b: WorldLaws, rng: PRNG): WorldLaws {
  const result = {} as any;
  const keys = Object.keys(a) as (keyof WorldLaws)[];
  for (const key of keys) {
    result[key] = rng.random() > 0.5 ? a[key] : b[key];
  }
  return result as WorldLaws;
}
