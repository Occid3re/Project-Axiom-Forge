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
}

interface FloatRange {
  min: number;
  max: number;
}

const FLOAT_RANGES: Record<string, FloatRange> = {
  reproductionCost: { min: 0.1, max: 1.0 },
  offspringEnergy: { min: 0.05, max: 0.8 },
  mutationRate: { min: 0.01, max: 0.5 },
  mutationStrength: { min: 0.01, max: 0.3 },
  resourceRegenRate: { min: 0.001, max: 0.1 },
  eatGain: { min: 0.1, max: 1.0 },
  moveCost: { min: 0.001, max: 0.1 },
  idleCost: { min: 0.001, max: 0.05 },
  attackTransfer: { min: 0.0, max: 0.8 },
  signalDecay: { min: 0.1, max: 0.99 },
  memoryPersistence: { min: 0.0, max: 1.0 },
  disasterProbability: { min: 0.0, max: 0.05 },
  terrainVariability: { min: 0.0, max: 1.0 },
};

const INT_RANGES: Record<string, { min: number; max: number }> = {
  signalRange: { min: 1, max: 8 },
  signalChannels: { min: 1, max: 6 },
  memorySize: { min: 1, max: 16 },
  maxPerceptionRadius: { min: 1, max: 6 },
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
    reproductionCost: rng.uniform(0.1, 1.0),
    offspringEnergy: rng.uniform(0.05, 0.8),
    mutationRate: rng.uniform(0.01, 0.5),
    mutationStrength: rng.uniform(0.01, 0.3),
    sexualReproduction: rng.random() > 0.5,
    resourceRegenRate: rng.uniform(0.001, 0.1),
    eatGain: rng.uniform(0.1, 1.0),
    moveCost: rng.uniform(0.001, 0.1),
    idleCost: rng.uniform(0.001, 0.05),
    attackTransfer: rng.uniform(0.0, 0.8),
    signalRange: rng.int(1, 8),
    signalChannels: rng.int(1, 6),
    signalDecay: rng.uniform(0.1, 0.99),
    memorySize: rng.int(1, 16),
    memoryPersistence: rng.uniform(0.0, 1.0),
    resourceDistribution: rng.int(0, 2) as ResourceDist,
    disasterProbability: rng.uniform(0.0, 0.05),
    terrainVariability: rng.uniform(0.0, 1.0),
    maxPerceptionRadius: rng.int(1, 6),
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
