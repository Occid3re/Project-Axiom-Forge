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
  total: number;
}

export function scoreWorld(
  history: WorldHistory,
  laws: WorldLaws,
  weights: Record<string, number>
): WorldScores {
  const snaps = history.snapshots;
  if (snaps.length === 0) {
    return { persistence: 0, diversity: 0, complexityGrowth: 0, communication: 0, envStructure: 0, adaptability: 0, speciation: 0, interactions: 0, spatialStructure: 0, populationDynamics: 0, total: 0 };
  }

  // Mutation chaos factor: high mutRate × mutStrength = cheap diversity → discount
  // starterLaws (0.06, 0.06) → 0.96x; degenerate (0.477, 0.3) → 0.41x
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
    (weights.populationDynamics ?? 0) * populationDynamics;

  return { persistence, diversity, complexityGrowth, communication, envStructure, adaptability, speciation, interactions, spatialStructure, populationDynamics, total };
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
  // Normalize and apply chaos discount — random drift diversity is cheap
  return Math.min(1, m / 2.0) * chaosFactor;
}

function scoreComplexityGrowth(snaps: WorldSnapshot[], chaosFactor: number): number {
  // Sustained growth across 4 quarters — reward monotonic increase, not just first-vs-last
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

  // Count consecutive quarters with diversity growth, weighted by magnitude
  let growthScore = 0;
  for (let i = 1; i < 4; i++) {
    if (qDiv[i] > qDiv[i - 1]) {
      growthScore += Math.min(0.5, (qDiv[i] - qDiv[i - 1]) / (qDiv[i - 1] + 0.01));
    }
  }
  growthScore = Math.min(1, growthScore / 1.0); // 3 consecutive growths → ~1.0

  // Population stability — late pop should be sustainable
  const popStability = qPop[3] > 0 ? Math.min(1, qPop[3] / (qPop[0] + 1)) : 0;

  return (growthScore * 0.6 + popStability * 0.4) * chaosFactor;
}

function scoreCommunication(snaps: WorldSnapshot[]): number {
  // Lagged correlation: signals at tick T should predict births at T+lag.
  // Test lags 5-15 and take the best. This prevents gaming via coincidental same-tick correlation.
  if (snaps.length < 30) return 0;

  const signalArr = snaps.map(s => s.signalActivity);
  const birthArr = snaps.map(s => s.births);

  // Normalized signal usage
  const maxSignal = Math.max(...signalArr) || 1;
  const meanSignal = mean(signalArr) / maxSignal;

  // Find best lagged correlation across lag=[5..15]
  let bestCorr = 0;
  for (let lag = 5; lag <= 15; lag++) {
    if (signalArr.length <= lag + 3) continue;
    const sigSlice = signalArr.slice(0, signalArr.length - lag);
    const birthSlice = birthArr.slice(lag);
    const c = Math.abs(correlation(sigSlice, birthSlice));
    if (c > bestCorr) bestCorr = c;
  }

  return Math.min(1, meanSignal * 0.3 + bestCorr * 0.7);
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
  // Reward genome clustering: high variance of pairwise distances = distinct species.
  // Random drift produces uniform distance distributions (low variance).
  // Real speciation produces bimodal/multimodal distributions (high variance).
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
  // Normalize: variance of normalized distances typically 0–2.
  // Score 0.5 → full credit (strong clustering).
  return Math.min(1, meanVar / 0.5);
}

function scoreInteractions(snaps: WorldSnapshot[]): number {
  // Reward ecological richness: worlds where entities attack, signal, AND survive.
  // Passive grazer monocultures score low. Predator-prey arms races score high.
  if (snaps.length < 20) return 0;

  const popSnaps = snaps.filter(s => s.population > 2);
  if (popSnaps.length < 10) return 0;

  // Mean attack rate (attacks per entity per tick)
  const attackRate = mean(popSnaps.map(s => s.attacks / s.population));
  // Mean signal rate
  const signalRate = mean(popSnaps.map(s => s.signals / s.population));

  // Attacks should exist but not dominate (arms race, not massacre)
  // Sweet spot: ~0.05–0.3 attacks per entity per tick
  const attackScore = attackRate > 0.01
    ? Math.min(1, attackRate * 5) * Math.min(1, 0.5 / (attackRate + 0.01))
    : 0;
  // Signal usage — any meaningful signaling
  const signalScore = Math.min(1, signalRate * 3);

  // Both present = rich ecology (geometric mean rewards balance)
  return Math.sqrt(attackScore * signalScore);
}

function scoreSpatialStructure(snaps: WorldSnapshot[]): number {
  // Reward worlds where population isn't uniformly distributed.
  // Use variance of births/deaths across time as proxy for spatial hotspots.
  // A world with localized battles (high birth/death variance) is more
  // spatially interesting than a uniform monoculture (constant births/deaths).
  if (snaps.length < 40) return 0;

  const birthRates = snaps.filter(s => s.population > 2).map(s => s.births / s.population);
  const deathRates = snaps.filter(s => s.population > 2).map(s => s.deaths / s.population);
  if (birthRates.length < 20) return 0;

  const birthVar = sampleVariance(birthRates);
  const deathVar = sampleVariance(deathRates);

  // Also reward poison coverage variation (active dead zones forming and clearing)
  const poisonCov = snaps.map(s => s.poisonCoverage);
  const poisonVar = sampleVariance(poisonCov);
  const poisonPresent = mean(poisonCov) > 0.01 ? 0.3 : 0;

  // Penalize very high population (monoculture soup fills the screen)
  const meanPop = mean(snaps.map(s => s.population));
  const overflowPenalty = meanPop > 2000 ? Math.min(0.5, (meanPop - 2000) / 4000) : 0;

  const rawScore = Math.min(1, (birthVar + deathVar) * 30 + poisonVar * 10 + poisonPresent);
  return Math.max(0, rawScore - overflowPenalty);
}

function scorePopulationDynamics(snaps: WorldSnapshot[]): number {
  // Reward oscillating populations (predator-prey cycles, boom-bust).
  // Penalize flat population lines (boring monoculture equilibrium).
  if (snaps.length < 60) return 0;

  const pops = snaps.map(s => s.population);
  const meanPop = mean(pops);
  if (meanPop < 5) return 0;

  // Count direction changes (peaks/troughs) — more = more dynamic
  let directionChanges = 0;
  for (let i = 2; i < pops.length; i++) {
    const prev = pops[i - 1] - pops[i - 2];
    const curr = pops[i] - pops[i - 1];
    if ((prev > 0 && curr < 0) || (prev < 0 && curr > 0)) directionChanges++;
  }
  // Smooth out noise: count significant changes (>2% of mean population)
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

  // Coefficient of variation — high = dynamic, low = flat
  const cv = Math.sqrt(sampleVariance(pops)) / (meanPop + 1);

  // Combine: significant oscillations + variability
  const oscillationScore = Math.min(1, significantChanges / 15);
  const cvScore = Math.min(1, cv * 5);

  return oscillationScore * 0.6 + cvScore * 0.4;
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
