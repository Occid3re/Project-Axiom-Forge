import type { GenerationResult, WorldScores } from '../../engine';

export interface EmergenceState {
  stage: number;
  progress: number[];
}

const STAGE_THRESHOLDS = {
  survival: 0.3,
  resourceCycling: 0.08,
  glyphCommunication: 0.12,
  diversity: 0.15,
  predation: 0.12,
  culturalMarks: 0.09,
  kinSelection: 0.12,
  speciation: 0.18,
  ecology: 0.12,
} as const;

export function detectEmergence(
  scores: WorldScores | null,
  generations: GenerationResult[],
): EmergenceState {
  const progress = new Array(10).fill(0);

  if (!scores) return { stage: -1, progress };

  progress[0] = Math.min(1, scores.persistence / STAGE_THRESHOLDS.survival);
  progress[1] = Math.min(1, scores.envStructure / STAGE_THRESHOLDS.resourceCycling);
  progress[2] = Math.min(1, scores.communication / STAGE_THRESHOLDS.glyphCommunication);
  progress[3] = Math.min(1, scores.diversity / STAGE_THRESHOLDS.diversity);
  progress[4] = Math.min(1, scores.interactions / STAGE_THRESHOLDS.predation);
  progress[5] = Math.min(1, scores.stigmergicUse / STAGE_THRESHOLDS.culturalMarks);
  progress[6] = Math.min(1, scores.socialDifferentiation / STAGE_THRESHOLDS.kinSelection);
  progress[7] = Math.min(1, scores.speciation / STAGE_THRESHOLDS.speciation);

  const ecoMetrics = [
    scores.persistence,
    scores.diversity,
    scores.communication,
    scores.interactions,
    scores.speciation,
    scores.stigmergicUse,
    scores.socialDifferentiation,
  ].filter(v => v > 0);
  if (ecoMetrics.length >= 5) {
    const geoMean = Math.pow(ecoMetrics.reduce((a, b) => a * b, 1), 1 / ecoMetrics.length);
    progress[8] = Math.min(1, geoMean / STAGE_THRESHOLDS.ecology);
  }

  if (generations.length >= 5) {
    const scores2 = generations.map(g => g.bestScore);
    const firstHalf = scores2.slice(0, Math.floor(scores2.length / 2));
    const secondHalf = scores2.slice(Math.floor(scores2.length / 2));
    const firstRate = firstHalf.length > 1 ? (firstHalf[firstHalf.length - 1] - firstHalf[0]) / firstHalf.length : 0;
    const secondRate = secondHalf.length > 1 ? (secondHalf[secondHalf.length - 1] - secondHalf[0]) / secondHalf.length : 0;
    const accelerating = secondRate > firstRate * 1.1;
    progress[9] = Math.min(1, accelerating ? Math.min(1, secondRate / (firstRate + 0.001)) * 0.7 : progress[8] * 0.3);
  }

  let stage = -1;
  for (let i = 0; i < 10; i++) {
    if (progress[i] >= 0.6) stage = i;
    else break;
  }

  return { stage, progress };
}
