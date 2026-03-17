/**
 * Core constants for the simulation engine.
 * All entity data lives in typed arrays — no objects in the hot loop.
 */

// --- Neural-network genome layout ---
// Genome = 80 float weights:
//   W1 [NN_INPUTS × NN_HIDDEN = 32]:  genome[input * NN_HIDDEN + hidden]
//   W2 [NN_HIDDEN × NN_OUTPUTS = 48]: genome[NN_W1_SIZE + hidden * NN_OUTPUTS + action]
//
// Forward pass (per entity, per tick):
//   inputs = [localResource, energyNorm, entityDensity, signalStrength]  (all 0–1)
//   hidden = tanh(W1 · inputs)                                            (8 units)
//   logits = W2 · hidden                                                   (6 values)
//   action = softmax_sample(logits)
//
// Weights are real-valued floats — NOT clamped to [0, 1].
// Init: W1 ~ N(0, √(2/NN_INPUTS)),  W2 ~ N(0, √(2/NN_HIDDEN))  (Xavier)
// Mutation: Gaussian noise, soft-clamped at ±6.

export const NN_INPUTS   = 4;
export const NN_HIDDEN   = 8;
export const NN_OUTPUTS  = 6;
export const NN_W1_SIZE  = NN_INPUTS  * NN_HIDDEN;   // 32
export const NN_W2_SIZE  = NN_HIDDEN  * NN_OUTPUTS;  // 48
export const GENOME_LENGTH = NN_W1_SIZE + NN_W2_SIZE; // 80

// --- Entity data layout (Struct of Arrays) ---
export const MAX_ENTITIES    = 4096;
export const MAX_MEMORY_SIZE = 16;

// --- Action types ---
export const enum ActionType {
  IDLE      = 0,
  MOVE      = 1,
  EAT       = 2,
  REPRODUCE = 3,
  SIGNAL    = 4,
  ATTACK    = 5,
}

// --- Resource distribution types ---
export const enum ResourceDist {
  UNIFORM   = 0,
  CLUSTERED = 1,
  GRADIENT  = 2,
}
