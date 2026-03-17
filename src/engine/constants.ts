/**
 * Core constants for the simulation engine.
 * All entity data lives in typed arrays — no objects in the hot loop.
 */

// --- Genome layout ---
// Each gene is a float32 in [0, 1].
export const GENOME_LENGTH = 16;

export const enum Gene {
  MOVE_BIAS_X = 0,
  MOVE_BIAS_Y = 1,
  MOVE_RANDOMNESS = 2,
  AGGRESSION = 3,
  REPRO_THRESHOLD = 4,
  EAT_PRIORITY = 5,
  SIGNAL_CHANNEL = 6,
  SIGNAL_STRENGTH = 7,
  SIGNAL_RESPONSIVENESS = 8,
  PERCEPTION_RANGE = 9,
  MEMORY_WRITE_RATE = 10,
  MEMORY_READ_WEIGHT = 11,
  COOPERATION = 12,
  EXPLORE_EXPLOIT = 13,
  ENERGY_CONSERVATISM = 14,
  ADAPTATION_RATE = 15,
}

// --- Entity data layout (Struct of Arrays) ---
// Each entity's data is at index [i] across these parallel arrays.
export const MAX_ENTITIES = 4096;
export const MAX_MEMORY_SIZE = 16;

// --- Action types ---
export const enum ActionType {
  IDLE = 0,
  MOVE = 1,
  EAT = 2,
  REPRODUCE = 3,
  SIGNAL = 4,
  ATTACK = 5,
}

// --- Resource distribution types ---
export const enum ResourceDist {
  UNIFORM = 0,
  CLUSTERED = 1,
  GRADIENT = 2,
}
