/**
 * Scoring — evaluate how interesting a world's simulation was.
 * No hardcoded intelligence detectors. Uses information-theoretic
 * and statistical metrics.
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
  total: number;
}

export function scoreWorld(
  history: WorldHistory,
  laws: WorldLaws,
  weights: Record<string, number>
): WorldScores {
  const snaps = history.snapshots;
  if (snaps.length === 0) {
    return { persistence: 0, diversity: 0, complexityGrowth: 0, communication: 0, envStructure: 0, adaptability: 0, total: 0 };
  }

  const persistence = scorePersistence(snaps);
  const diversity = scoreDiversity(snaps);
  const complexityGrowth = scoreComplexityGrowth(snaps);
  const communication = scoreCommunication(snaps);
  const envStructure = scoreEnvStructure(snaps);
  const adaptability = scoreAdaptability(history);

  const total =
    (weights.persistence ?? 1) * persistence +
    (weights.diversity ?? 1) * diversity +
    (weights.complexityGrowth ?? 1) * complexityGrowth +
    (weights.communication ?? 1) * communication +
    (weights.envStructure ?? 1) * envStructure +
    (weights.adaptability ?? 1) * adaptability;

  return { persistence, diversity, complexityGrowth, communication, envStructure, adaptability, total };
}

function scorePersistence(snaps: WorldSnapshot[]): number {
  // Fraction of ticks with population > 0, with penalty for
  // instant extinction and stagnant immortality.
  let aliveTicks = 0;
  let totalVariation = 0;
  for (let i = 0; i < snaps.length; i++) {
    if (snaps[i].population > 0) aliveTicks++;
    if (i > 0) totalVariation += Math.abs(snaps[i].population - snaps[i - 1].population);
  }
  const survivalRate = aliveTicks / snaps.length;
  // Reward some population dynamics (not flat)
  const dynamism = Math.min(1, totalVariation / (snaps.length * 2));
  return survivalRate * 0.7 + dynamism * 0.3;
}

function scoreDiversity(snaps: WorldSnapshot[]): number {
  // Average genome diversity over time
  let sum = 0;
  let count = 0;
  for (const s of snaps) {
    if (s.population >= 2) {
      sum += s.diversity;
      count++;
    }
  }
  if (count === 0) return 0;
  const mean = sum / count;
  // Normalize: max possible diversity in [0,1]^16 genome is sqrt(16)=4
  return Math.min(1, mean / 2.0);
}

function scoreComplexityGrowth(snaps: WorldSnapshot[]): number {
  // Measure if diversity is increasing over time (positive slope).
  if (snaps.length < 20) return 0;

  const quarter = Math.floor(snaps.length / 4);
  const firstQuarter = snaps.slice(0, quarter);
  const lastQuarter = snaps.slice(-quarter);

  const earlyDiv = mean(firstQuarter.map(s => s.diversity));
  const lateDiv = mean(lastQuarter.map(s => s.diversity));

  // Also look at population complexity
  const earlyPop = mean(firstQuarter.map(s => s.population));
  const latePop = mean(lastQuarter.map(s => s.population));

  const divGrowth = lateDiv > earlyDiv ? Math.min(1, (lateDiv - earlyDiv) / (earlyDiv + 0.01)) : 0;
  const popStability = latePop > 0 ? Math.min(1, latePop / (earlyPop + 1)) : 0;

  return divGrowth * 0.6 + popStability * 0.4;
}

function scoreCommunication(snaps: WorldSnapshot[]): number {
  // Measure correlation between signal activity and births/cooperation.
  // If signals correlate with population dynamics, communication is meaningful.
  if (snaps.length < 10) return 0;

  const signalArr = snaps.map(s => s.signalActivity);
  const birthArr = snaps.map(s => s.births);

  // Normalized signal usage
  const maxSignal = Math.max(...signalArr) || 1;
  const meanSignal = mean(signalArr) / maxSignal;

  // Signal-birth correlation (simplified)
  const corr = correlation(signalArr, birthArr);

  // Both signal usage and correlation matter
  return Math.min(1, meanSignal * 0.4 + Math.abs(corr) * 0.6);
}

function scoreEnvStructure(snaps: WorldSnapshot[]): number {
  // Measure if entities form spatial structures.
  // Proxy: resource coverage variation over time (entities depleting/managing resources).
  if (snaps.length < 10) return 0;

  const coverage = snaps.map(s => s.resourceCoverage);
  const variance = sampleVariance(coverage);

  // High variance = entities significantly impact environment
  return Math.min(1, variance * 20);
}

function scoreAdaptability(history: WorldHistory): number {
  // Recovery after disasters
  if (history.disasterCount === 0) {
    // No disasters: measure general resilience from population fluctuations
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
  const n = a.length;
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
