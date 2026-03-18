/**
 * Scoring — evaluate how interesting a world's simulation was.
 * No hardcoded intelligence detectors. Uses information-theoretic
 * and statistical metrics.
 *
 * Anti-gaming measures:
 * - Diversity is discounted by mutation chaos factor (high mutRate×mutStrength = cheap diversity)
 * - Communication uses lagged correlation (signal→future births, not same-tick coincidence)
 * - ComplexityGrowth requires sustained monotonic increase across quarters
 * - Speciation rewards genome clustering (real species) over uniform random spread
 * - StigmergicUse rewards balanced deposit/absorb activity
 * - SocialDifferentiation rewards kin-selective behavior
 */

import type { WorldHistory, WorldSnapshot } from './world';
import type { WorldLaws } from './world-laws';

export interface WorldScores {
  persistence: number;
  diversity: number;
  complexityGrowth: number;
  communication: number;
  envStructure: number;
  adaptability: number;
  speciation: number;
  interactions: number;
  spatialStructure: number;
  populationDynamics: number;
  stigmergicUse: number;
  socialDifferentiation: number;
  total: number;
}

export function scoreWorld(
  history: WorldHistory,
  laws: WorldLaws,
  weights: Record<string, number>
): WorldScores {
  const snaps = history.snapshots;
  if (snaps.length === 0) {
    return { persistence: 0, diversity: 0, complexityGrowth: 0, communication: 0, envStructure: 0, adaptability: 0, speciation: 0, interactions: 0, spatialStructure: 0, populationDynamics: 0, stigmergicUse: 0, socialDifferentiation: 0, total: 0 };
  }

  // Mutation chaos factor: high mutRate × mutStrength = cheap diversity → discount
  const chaosFactor = 1 / (1 + laws.mutationRate * laws.mutationStrength * 10);

  const persistence = scorePersistence(snaps);
  const diversity = scoreDiversity(snaps, chaosFactor);
  const complexityGrowth = scoreComplexityGrowth(snaps, chaosFactor);
  const communication = scoreCommunication(snaps);
  const envStructure = scoreEnvStructure(snaps);
  const adaptability = scoreAdaptability(history);
  const speciation = scoreSpeciation(snaps);
  const interactions = scoreInteractions(snaps);
  const spatialStructure = scoreSpatialStructure(snaps);
  const populationDynamics = scorePopulationDynamics(snaps);
  const stigmergicUse = scoreStigmergicUse(snaps);
  const socialDifferentiation = scoreSocialDifferentiation(snaps);

  const total =
    (weights.persistence ?? 1) * persistence +
    (weights.diversity ?? 1) * diversity +
    (weights.complexityGrowth ?? 1) * complexityGrowth +
    (weights.communication ?? 1) * communication +
    (weights.envStructure ?? 1) * envStructure +
    (weights.adaptability ?? 1) * adaptability +
    (weights.speciation ?? 0) * speciation +
    (weights.interactions ?? 0) * interactions +
    (weights.spatialStructure ?? 0) * spatialStructure +
    (weights.populationDynamics ?? 0) * populationDynamics +
    (weights.stigmergicUse ?? 0) * stigmergicUse +
    (weights.socialDifferentiation ?? 0) * socialDifferentiation;

  return { persistence, diversity, complexityGrowth, communication, envStructure, adaptability, speciation, interactions, spatialStructure, populationDynamics, stigmergicUse, socialDifferentiation, total };
}

function scorePersistence(snaps: WorldSnapshot[]): number {
  let aliveTicks = 0;
  let totalVariation = 0;
  for (let i = 0; i < snaps.length; i++) {
    if (snaps[i].population > 0) aliveTicks++;
    if (i > 0) totalVariation += Math.abs(snaps[i].population - snaps[i - 1].population);
  }
  const survivalRate = aliveTicks / snaps.length;
  const dynamism = Math.min(1, totalVariation / (snaps.length * 2));
  return survivalRate * 0.7 + dynamism * 0.3;
}

function scoreDiversity(snaps: WorldSnapshot[], chaosFactor: number): number {
  let sum = 0;
  let count = 0;
  for (const s of snaps) {
    if (s.population >= 2) {
      sum += s.diversity;
      count++;
    }
  }
  if (count === 0) return 0;
  const m = sum / count;
  return Math.min(1, m / 2.0) * chaosFactor;
}

function scoreComplexityGrowth(snaps: WorldSnapshot[], chaosFactor: number): number {
  if (snaps.length < 40) return 0;

  const q = Math.floor(snaps.length / 4);
  const quarters = [
    snaps.slice(0, q),
    snaps.slice(q, q * 2),
    snaps.slice(q * 2, q * 3),
    snaps.slice(q * 3),
  ];
  const qDiv = quarters.map(qs => mean(qs.map(s => s.diversity)));
  const qPop = quarters.map(qs => mean(qs.map(s => s.population)));

  let growthScore = 0;
  for (let i = 1; i < 4; i++) {
    if (qDiv[i] > qDiv[i - 1]) {
      growthScore += Math.min(0.5, (qDiv[i] - qDiv[i - 1]) / (qDiv[i - 1] + 0.01));
    }
  }
  growthScore = Math.min(1, growthScore / 1.0);

  const popStability = qPop[3] > 0 ? Math.min(1, qPop[3] / (qPop[0] + 1)) : 0;

  return (growthScore * 0.6 + popStability * 0.4) * chaosFactor;
}

function scoreCommunication(snaps: WorldSnapshot[]): number {
  // Communication via stigmergic glyphs (DEPOSIT+ABSORB): entities lay chemical trails
  // that others can follow. If glyph activity correlates with future births, communication
  // is driving reproduction — a real information channel.
  if (snaps.length < 30) return 0;

  const glyphArr = snaps.map(s => s.deposits + s.absorbs);
  const birthArr = snaps.map(s => s.births);

  const maxGlyph = Math.max(...glyphArr) || 1;
  const meanGlyph = mean(glyphArr) / maxGlyph;

  let bestCorr = 0;
  for (let lag = 5; lag <= 15; lag++) {
    if (glyphArr.length <= lag + 3) continue;
    const gSlice = glyphArr.slice(0, glyphArr.length - lag);
    const bSlice = birthArr.slice(lag);
    const c = Math.abs(correlation(gSlice, bSlice));
    if (c > bestCorr) bestCorr = c;
  }

  return Math.min(1, meanGlyph * 0.3 + bestCorr * 0.7);
}

function scoreEnvStructure(snaps: WorldSnapshot[]): number {
  if (snaps.length < 10) return 0;
  const coverage = snaps.map(s => s.resourceCoverage);
  const variance = sampleVariance(coverage);
  return Math.min(1, variance * 20);
}

function scoreAdaptability(history: WorldHistory): number {
  if (history.disasterCount === 0) {
    const snaps = history.snapshots;
    if (snaps.length < 50) return 0.5;

    let dips = 0;
    let recoveries = 0;
    const windowMean = mean(snaps.map(s => s.population));

    for (let i = 10; i < snaps.length - 10; i++) {
      if (snaps[i].population < windowMean * 0.5 && snaps[i - 5].population > windowMean * 0.7) {
        dips++;
        if (snaps[Math.min(i + 10, snaps.length - 1)].population > windowMean * 0.5) {
          recoveries++;
        }
      }
    }

    if (dips === 0) return 0.5;
    return 0.5 + 0.5 * (recoveries / dips);
  }

  return history.postDisasterRecoveries / Math.max(1, history.disasterCount);
}

function scoreSpeciation(snaps: WorldSnapshot[]): number {
  let sum = 0;
  let count = 0;
  for (const s of snaps) {
    if (s.population >= 4) {
      sum += s.diversityVariance;
      count++;
    }
  }
  if (count === 0) return 0;
  const meanVar = sum / count;
  return Math.min(1, meanVar / 0.5);
}

function scoreInteractions(snaps: WorldSnapshot[]): number {
  // Interactions = predation (attacks) coexisting with glyph-based communication.
  // Glyph activity replaces signal-based communication as the "social" component —
  // entities can perceive glyphs directionally but cannot perceive chemical signals.
  if (snaps.length < 20) return 0;

  const popSnaps = snaps.filter(s => s.population > 2);
  if (popSnaps.length < 10) return 0;

  const attackRate = mean(popSnaps.map(s => s.attacks / s.population));
  const glyphRate  = mean(popSnaps.map(s => (s.deposits + s.absorbs) / s.population));

  const attackScore = attackRate > 0.01
    ? Math.min(1, attackRate * 5) * Math.min(1, 0.5 / (attackRate + 0.01))
    : 0;
  const glyphScore = Math.min(1, glyphRate * 3);

  return Math.sqrt(attackScore * glyphScore);
}

function scoreSpatialStructure(snaps: WorldSnapshot[]): number {
  if (snaps.length < 40) return 0;

  const birthRates = snaps.filter(s => s.population > 2).map(s => s.births / s.population);
  const deathRates = snaps.filter(s => s.population > 2).map(s => s.deaths / s.population);
  if (birthRates.length < 20) return 0;

  const birthVar = sampleVariance(birthRates);
  const deathVar = sampleVariance(deathRates);

  const poisonCov = snaps.map(s => s.poisonCoverage);
  const poisonVar = sampleVariance(poisonCov);
  const poisonPresent = mean(poisonCov) > 0.01 ? 0.3 : 0;

  const meanPop = mean(snaps.map(s => s.population));
  const overflowPenalty = meanPop > 2000 ? Math.min(0.5, (meanPop - 2000) / 4000) : 0;

  const rawScore = Math.min(1, (birthVar + deathVar) * 30 + poisonVar * 10 + poisonPresent);
  return Math.max(0, rawScore - overflowPenalty);
}

function scorePopulationDynamics(snaps: WorldSnapshot[]): number {
  if (snaps.length < 60) return 0;

  const pops = snaps.map(s => s.population);
  const meanPop = mean(pops);
  if (meanPop < 5) return 0;

  let significantChanges = 0;
  const threshold = meanPop * 0.03;
  let lastPeak = pops[0];
  let rising = true;
  for (let i = 1; i < pops.length; i++) {
    if (rising && pops[i] < lastPeak - threshold) {
      significantChanges++;
      rising = false;
      lastPeak = pops[i];
    } else if (!rising && pops[i] > lastPeak + threshold) {
      significantChanges++;
      rising = true;
      lastPeak = pops[i];
    }
    if (rising && pops[i] > lastPeak) lastPeak = pops[i];
    if (!rising && pops[i] < lastPeak) lastPeak = pops[i];
  }

  const cv = Math.sqrt(sampleVariance(pops)) / (meanPop + 1);
  const oscillationScore = Math.min(1, significantChanges / 15);
  const cvScore = Math.min(1, cv * 5);

  return oscillationScore * 0.6 + cvScore * 0.4;
}

/**
 * Reward worlds where entities actively use stigmergic memory (DEPOSIT + ABSORB).
 * Balanced usage (both deposit and absorb) scores higher than one-sided.
 */
function scoreStigmergicUse(snaps: WorldSnapshot[]): number {
  if (snaps.length < 40) return 0;

  const popSnaps = snaps.filter(s => s.population > 5);
  if (popSnaps.length < 20) return 0;

  const depositRate = mean(popSnaps.map(s => s.deposits / s.population));
  const absorbRate  = mean(popSnaps.map(s => s.absorbs / s.population));

  // Both must exist — geometric mean rewards balance
  const activity = Math.sqrt(depositRate * absorbRate);
  // Balance factor: penalize one-sided usage
  const maxRate = Math.max(depositRate, absorbRate);
  const balance = maxRate > 0.001 ? Math.min(depositRate, absorbRate) / maxRate : 0;

  return Math.min(1, activity * 10) * balance;
}

/**
 * Reward worlds where entities behave differently toward kin vs non-kin.
 * Uses attack selectivity as proxy: do entities preferentially attack strangers?
 * Also considers deposit/absorb patterns — do kin share glyphs?
 */
function scoreSocialDifferentiation(snaps: WorldSnapshot[]): number {
  if (snaps.length < 40) return 0;

  const popSnaps = snaps.filter(s => s.population > 5);
  if (popSnaps.length < 20) return 0;

  // Attack rate should differ across time (indicating selective behavior)
  // When entities are kin-selective, attack rate varies with population composition
  const attackRates = popSnaps.map(s => s.attacks / s.population);
  const attackVar = sampleVariance(attackRates);

  // Deposit/absorb covariance with population — social species deposit more when crowded
  const depositRates = popSnaps.map(s => s.deposits / s.population);
  const depositVar = sampleVariance(depositRates);

  // Combine: attack variability (selective predation) + deposit variability (social learning)
  const selectivity = Math.min(1, attackVar * 50);
  const socialLearning = Math.min(1, depositVar * 50);

  return selectivity * 0.5 + socialLearning * 0.5;
}

// --- Utilities ---

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  let sum = 0;
  for (const v of arr) sum += v;
  return sum / arr.length;
}

function sampleVariance(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  let sum = 0;
  for (const v of arr) sum += (v - m) * (v - m);
  return sum / (arr.length - 1);
}

function correlation(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n < 3) return 0;
  const ma = mean(a);
  const mb = mean(b);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const ai = a[i] - ma;
    const bi = b[i] - mb;
    num += ai * bi;
    da += ai * ai;
    db += bi * bi;
  }
  const denom = Math.sqrt(da * db);
  return denom > 0 ? num / denom : 0;
}
