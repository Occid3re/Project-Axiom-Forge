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
  seasonalAdaptation: number;    // M2: birth-phase concentration relative to seasonal cycle
  lifetimeLearning: number;      // M5: old entities harvest more efficiently than young ones
  total: number;
}

export function scoreWorld(
  history: WorldHistory,
  laws: WorldLaws,
  weights: Record<string, number>
): WorldScores {
  const snaps = history.snapshots;
  if (snaps.length === 0) {
    return { persistence: 0, diversity: 0, complexityGrowth: 0, communication: 0, envStructure: 0, adaptability: 0, speciation: 0, interactions: 0, spatialStructure: 0, populationDynamics: 0, stigmergicUse: 0, socialDifferentiation: 0, seasonalAdaptation: 0, lifetimeLearning: 0, total: 0 };
  }

  // Mutation chaos factor: high mutRate × mutStrength = cheap diversity → discount
  const chaosFactor = 1 / (1 + laws.mutationRate * laws.mutationStrength * 10);

  const persistence = scorePersistence(snaps);
  const diversity = scoreDiversity(snaps, chaosFactor);
  const complexityGrowth = scoreComplexityGrowth(snaps, chaosFactor);
  const communication = scoreCommunication(snaps);
  const envStructure = scoreEnvStructure(snaps);
  const adaptability = scoreAdaptability(history);
  const speciation = scoreSpeciation(snaps, chaosFactor);
  const interactions = scoreInteractions(snaps);
  const spatialStructure = scoreSpatialStructure(snaps);
  const populationDynamics = scorePopulationDynamics(snaps);
  const stigmergicUse = scoreStigmergicUse(snaps);
  const socialDifferentiation = scoreSocialDifferentiation(snaps);
  const seasonalAdaptation = scoreSeasonalAdaptation(snaps, laws.seasonLength ?? 0);
  const lifetimeLearning   = scoreLifetimeLearning(snaps);

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
    (weights.socialDifferentiation ?? 0) * socialDifferentiation +
    (weights.seasonalAdaptation ?? 0) * seasonalAdaptation +
    (weights.lifetimeLearning  ?? 0) * lifetimeLearning;

  return { persistence, diversity, complexityGrowth, communication, envStructure, adaptability, speciation, interactions, spatialStructure, populationDynamics, stigmergicUse, socialDifferentiation, seasonalAdaptation, lifetimeLearning, total };
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
  // Communication now includes both fast chemical signaling and persistent glyph memory.
  // We reward worlds where communication activity predicts future births instead of
  // only adding visual noise.
  if (snaps.length < 30) return 0;

  const commArr = snaps.map(s => s.signals * 0.9 + s.deposits + s.absorbs);
  const birthArr = snaps.map(s => s.births);

  const popSnaps = snaps.filter(s => s.population > 4);
  if (popSnaps.length < 12) return 0;

  const maxComm = Math.max(...commArr) || 1;
  const meanComm = mean(commArr) / maxComm;
  const signalRate = mean(popSnaps.map(s => s.signals / s.population));
  const glyphRate = mean(popSnaps.map(s => (s.deposits + s.absorbs) / s.population));

  let bestCorr = 0;
  for (let lag = 5; lag <= 15; lag++) {
    if (commArr.length <= lag + 3) continue;
    const gSlice = commArr.slice(0, commArr.length - lag);
    const bSlice = birthArr.slice(lag);
    const c = Math.abs(correlation(gSlice, bSlice));
    if (c > bestCorr) bestCorr = c;
  }

  const signalScore = Math.min(1, signalRate * 5);
  const glyphScore = Math.min(1, glyphRate * 3.5);
  return Math.min(1, meanComm * 0.2 + bestCorr * 0.5 + signalScore * 0.15 + glyphScore * 0.15);
}

function scoreEnvStructure(snaps: WorldSnapshot[]): number {
  if (snaps.length < 10) return 0;
  const coverage = snaps.map(s => s.resourceCoverage);
  const variance = sampleVariance(coverage);
  const deltas: number[] = [];
  for (let i = 1; i < coverage.length; i++) {
    deltas.push(Math.abs(coverage[i] - coverage[i - 1]));
  }
  const volatility = mean(deltas);
  const range = Math.max(...coverage) - Math.min(...coverage);

  // Resource cycling on the large display world tends to show up as subtle but sustained
  // motion in coverage, not huge variance spikes. Combine variance, drift, and range so
  // the score occupies a useful 0..1 range again after the carrying-capacity changes.
  return Math.min(1, variance * 12000 + volatility * 10 + range * 1.2);
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

function scoreSpeciation(snaps: WorldSnapshot[], chaosFactor: number): number {
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
  return Math.min(1, meanVar / 0.5) * chaosFactor;
}

function scoreInteractions(snaps: WorldSnapshot[]): number {
  // Interactions = predation coexisting with both fast and persistent communication.
  if (snaps.length < 20) return 0;

  const popSnaps = snaps.filter(s => s.population > 2);
  if (popSnaps.length < 10) return 0;

  const attackRate = mean(popSnaps.map(s => s.attacks / s.population));
  const signalRate = mean(popSnaps.map(s => s.signals / s.population));
  const glyphRate  = mean(popSnaps.map(s => (s.deposits + s.absorbs) / s.population));

  const attackScore = attackRate > 0.01
    ? Math.min(1, attackRate * 5) * Math.min(1, 0.5 / (attackRate + 0.01))
    : 0;
  const signalScore = Math.min(1, signalRate * 4);
  const glyphScore = Math.min(1, glyphRate * 3);
  const communicationScore = Math.sqrt(Math.max(0, signalScore) * Math.max(0, glyphScore));

  return Math.sqrt(attackScore * communicationScore);
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
  const territoryCov = snaps.map(s => s.territoryCoverage);
  const territoryVar = sampleVariance(territoryCov);
  const territoryBand = bandScore(mean(territoryCov), 0.10, 0.10);
  const sizeMeans = snaps.map(s => s.meanSize);
  const sizeVar = sampleVariance(sizeMeans);
  const maxMacro = Math.max(...snaps.map(s => s.maxSize));
  const largeFraction = mean(snaps.map(s => s.population > 0 ? s.largeOrganisms / s.population : 0));
  const macroRarity = Math.max(0, 1 - Math.abs(largeFraction - 0.10) / 0.10);
  const macroPresence = Math.max(0, Math.min(1, (maxMacro - 1.45) / 1.1));
  const nichePersistence = scoreNichePersistence(snaps);
  const macroLongevity = scoreMacroLongevity(snaps);

  const meanPop = mean(snaps.map(s => s.population));
  const overflowPenalty = meanPop > 2000 ? Math.min(0.5, (meanPop - 2000) / 4000) : 0;

  const rawScore = Math.min(
    1,
    (birthVar + deathVar) * 30
    + poisonVar * 10
    + poisonPresent
    + territoryBand * 0.18
    + territoryVar * 3.5
    + sizeVar * 5
    + macroRarity * 0.25
    + macroPresence * 0.16
    + nichePersistence * 0.30
    + macroLongevity * 0.22,
  );
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
  const presenceScore = Math.min(1, meanPop / 18);

  return (oscillationScore * 0.6 + cvScore * 0.4) * presenceScore;
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
 * Reward worlds where entities behave differently toward kin vs non-kin and
 * actually stabilize into fused multicellular colonies.
 * Uses direct social counters from the world:
 * - kinContacts: nearby kin observed during perception scans
 * - threatContacts: nearby non-kin observed during perception scans
 * - kinCooperation: kin neighbors that actually contributed cooperation energy
 * - attacks: successful attacks, which are non-kin-only by world rules
 * - fusedMembers / largestColony / colonyBirths: actual colony formation signals
 */
function scoreSocialDifferentiation(snaps: WorldSnapshot[]): number {
  if (snaps.length < 40) return 0;

  const popSnaps = snaps.filter(s => s.population > 5 && (s.kinContacts + s.threatContacts) > 0);
  if (popSnaps.length < 20) return 0;

  const totalKinContacts = popSnaps.reduce((sum, s) => sum + s.kinContacts, 0);
  const totalThreatContacts = popSnaps.reduce((sum, s) => sum + s.threatContacts, 0);
  if (totalKinContacts < 50 || totalThreatContacts < 50) return 0;

  const coopRate = totalKinContacts > 0
    ? popSnaps.reduce((sum, s) => sum + s.kinCooperation, 0) / totalKinContacts
    : 0;
  const threatAttackRate = totalThreatContacts > 0
    ? popSnaps.reduce((sum, s) => sum + s.attacks, 0) / totalThreatContacts
    : 0;

  // Reward worlds that expose entities to both kin and non-kin.
  const exposureBalance = Math.min(totalKinContacts, totalThreatContacts) / Math.max(totalKinContacts, totalThreatContacts);

  // Cooperation toward kin should be common; attacks on threats are rarer, so use a softer scale.
  const cooperationScore = Math.min(1, coopRate * 1.5);
  const aggressionScore = Math.min(1, threatAttackRate * 8);

  const colonyFraction = mean(popSnaps.map(s => s.fusedMembers / Math.max(1, s.population)));
  const colonySizeScore = mean(popSnaps.map(s => Math.min(1, Math.max(0, s.largestColony - 1) / 4)));
  const colonyBirthScore = mean(popSnaps.map(s => s.births > 0 ? s.colonyBirths / s.births : 0));
  const macroPresence = mean(popSnaps.map(s => Math.max(0, Math.min(1, (s.maxSize - 1.8) / 1.2))));
  const largeFraction = mean(popSnaps.map(s => s.largeOrganisms / Math.max(1, s.population)));
  const macroBalance = Math.max(0, 1 - Math.abs(largeFraction - 0.09) / 0.09);
  const macroLongevity = scoreMacroLongevity(popSnaps);
  const nichePersistence = scoreNichePersistence(popSnaps);

  // Selective societies do both: support kin and punish outsiders under the same physics.
  const socialCore = Math.sqrt(cooperationScore * aggressionScore) * exposureBalance;
  const colonyCore =
    Math.min(1, colonyFraction * 1.4) * 0.45 +
    colonySizeScore * 0.35 +
    Math.min(1, colonyBirthScore * 2.5) * 0.20;

  return Math.min(
    1,
    socialCore * 0.48 +
    colonyCore * 0.24 +
    macroPresence * 0.08 +
    macroBalance * 0.05 +
    macroLongevity * 0.09 +
    nichePersistence * 0.06,
  );
}

/**
 * M5 — Lifetime learning: do old entities forage more efficiently than young ones?
 *
 * Uses harvestEfficiencyRatio from each snapshot: Q4_eats/age ÷ Q1_eats/age.
 * Baseline (no learning): ratio ≈ 1.0 — young and old are equally efficient.
 * Learning signal: ratio > 1.2 — older entities have learned better foraging routes.
 *
 * Score = clamp01((meanRatio - 1.0) / 0.5) so:
 *   ratio=1.0 → score=0.0 (no learning)
 *   ratio=1.2 → score=0.4 (moderate learning)
 *   ratio=1.5 → score=1.0 (strong learning, pass threshold)
 *
 * Only meaningful after enough deaths have accumulated; use second half of run.
 */
function scoreLifetimeLearning(snaps: WorldSnapshot[]): number {
  if (snaps.length < 40) return 0;
  // Use second half of run — early ticks don't have enough Q4 deaths yet
  const half = Math.floor(snaps.length / 2);
  const lateSnaps = snaps.slice(half).filter(s => s.harvestEfficiencyRatio > 0);
  if (lateSnaps.length < 10) return 0;
  const meanRatio = mean(lateSnaps.map(s => s.harvestEfficiencyRatio));
  return Math.max(0, Math.min(1, (meanRatio - 1.0) / 0.5));
}

/**
 * M2 — Seasonal adaptation: are births concentrated at specific phases of the resource cycle?
 *
 * Method: bin births by season phase into 8 buckets. Compute Shannon entropy of the
 * distribution. A uniform distribution (no adaptation) has entropy = log(8) ≈ 2.08 nats.
 * A perfectly concentrated distribution has entropy = 0.
 * Score = 1 - H / log(8), so 0 = no adaptation, 1 = perfect phase-locking.
 *
 * Pass threshold: score > 0.18 (uniform = 0.0, random noise ≈ 0.02–0.05).
 * If seasonLength ≤ 0 the world has no season; score is 0.
 */
function scoreSeasonalAdaptation(snaps: WorldSnapshot[], seasonLength: number): number {
  if (seasonLength <= 0 || snaps.length < seasonLength) return 0;

  const BINS = 8;
  const counts = new Float64Array(BINS);
  for (const s of snaps) {
    if (s.births <= 0) continue;
    const phase = (s.tick % seasonLength) / seasonLength; // [0, 1)
    const bin   = Math.min(BINS - 1, Math.floor(phase * BINS));
    counts[bin] += s.births;
  }

  const total = counts.reduce((a, b) => a + b, 0);
  if (total < 10) return 0; // not enough births to measure

  let H = 0;
  for (let b = 0; b < BINS; b++) {
    const p = counts[b] / total;
    if (p > 1e-10) H -= p * Math.log(p);
  }

  const maxH = Math.log(BINS); // entropy of uniform distribution
  return Math.max(0, 1 - H / maxH);
}

// --- Utilities ---

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function bandScore(value: number, center: number, radius: number): number {
  return Math.max(0, 1 - Math.abs(value - center) / radius);
}

function scoreNichePersistence(snaps: WorldSnapshot[]): number {
  if (snaps.length < 30) return 0;
  const active = snaps.filter((s) => s.population > 12);
  if (active.length < 18) return 0;

  let microTicks = 0;
  let macroTicks = 0;
  let coexistTicks = 0;
  let structuredTicks = 0;
  let longestMacroRun = 0;
  let macroRun = 0;

  for (const snap of active) {
    const largeFraction = snap.population > 0 ? snap.largeOrganisms / snap.population : 0;
    const macroPresent = snap.maxSize > 1.95 || snap.largestColony >= 4 || largeFraction > 0.035;
    const microPresent = largeFraction < 0.24;
    const structured = snap.largestColony >= 3 || snap.maxSize > 1.7 || snap.meanSize > 1.08;

    if (microPresent) microTicks++;
    if (macroPresent) {
      macroTicks++;
      macroRun++;
      if (macroRun > longestMacroRun) longestMacroRun = macroRun;
    } else {
      macroRun = 0;
    }
    if (structured) structuredTicks++;
    if (microPresent && macroPresent) coexistTicks++;
  }

  const microScore = clamp01(microTicks / (active.length * 0.75));
  const macroScore = clamp01(macroTicks / (active.length * 0.28));
  const coexistence = clamp01(coexistTicks / (active.length * 0.22));
  const continuity = clamp01(longestMacroRun / Math.max(8, active.length * 0.18));
  const structure = clamp01(structuredTicks / (active.length * 0.6));

  return Math.min(1, microScore * 0.20 + macroScore * 0.20 + coexistence * 0.34 + continuity * 0.16 + structure * 0.10);
}

function scoreMacroLongevity(snaps: WorldSnapshot[]): number {
  if (snaps.length < 30) return 0;
  const active = snaps.filter((s) => s.population > 12);
  if (active.length < 18) return 0;

  let macroTicks = 0;
  let longestRun = 0;
  let currentRun = 0;
  let largestBody = 0;
  let totalLargeFraction = 0;

  for (const snap of active) {
    const largeFraction = snap.population > 0 ? snap.largeOrganisms / snap.population : 0;
    const macroPresent = snap.maxSize > 1.95 || snap.largestColony >= 4 || largeFraction > 0.035;

    totalLargeFraction += largeFraction;
    if (snap.maxSize > largestBody) largestBody = snap.maxSize;
    if (macroPresent) {
      macroTicks++;
      currentRun++;
      if (currentRun > longestRun) longestRun = currentRun;
    } else {
      currentRun = 0;
    }
  }

  const presence = clamp01(macroTicks / (active.length * 0.20));
  const continuity = clamp01(longestRun / Math.max(10, active.length * 0.16));
  const sizeScore = clamp01((largestBody - 1.9) / 1.3);
  const balance = bandScore(totalLargeFraction / active.length, 0.08, 0.08);

  return Math.min(1, presence * 0.36 + continuity * 0.34 + sizeScore * 0.20 + balance * 0.10);
}

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
