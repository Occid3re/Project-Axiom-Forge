/**
 * Core constants for the simulation engine.
 * All entity data lives in typed arrays — no objects in the hot loop.
 */

// --- Neural-network genome layout (Elman recurrent network) ---
// Genome = 180 float weights:
//   W1 [NN_INPUTS × NN_HIDDEN = 100]:  genome[input * NN_HIDDEN + hidden]
//   W2 [NN_HIDDEN × NN_OUTPUTS = 80]:  genome[NN_W1_SIZE + hidden * NN_OUTPUTS + action]
//
// Forward pass (per entity, per tick):
//   inputs  = [localResource, energyNorm, entityDensity, signalStrength,
//              nearestKinEnergy, nearestThreatDist, kinRatio,
//              glyphStrength, glyphAffinity, ageNorm]                     (10 values, all 0–1)
//   h_new   = tanh(W1 · inputs)                                          (10 units)
//   h_blend = (1 - memoryPersistence) * h_new + memoryPersistence * h_prev
//   logits  = W2 · h_blend                                               (8 values)
//   action  = softmax_sample(logits)
//   h_prev  = h_blend                                       (stored in entity memory[0..9])
//
// The hidden-state carry-over makes this an Elman network: entities remember
// past states and can condition decisions on temporal context (threats, resources,
// signals observed in recent ticks). memoryPersistence is an evolvable world law.
//
// Weights are real-valued floats — NOT clamped to [0, 1].
// Init: W1 ~ N(0, √(2/NN_INPUTS)),  W2 ~ N(0, √(2/NN_HIDDEN))  (Xavier)
// Mutation: Gaussian noise, soft-clamped at ±6.

export const NN_INPUTS   = 10;
export const NN_HIDDEN   = 10;
export const NN_OUTPUTS  = 8;
export const NN_W1_SIZE  = NN_INPUTS  * NN_HIDDEN;   // 100
export const NN_W2_SIZE  = NN_HIDDEN  * NN_OUTPUTS;  // 80
export const GENOME_LENGTH = NN_W1_SIZE + NN_W2_SIZE; // 180

// --- Entity data layout (Struct of Arrays) ---
export const MAX_ENTITIES    = 4096;
export const MAX_MEMORY_SIZE = 16;

// --- Glyph (stigmergic memory) channels ---
export const GLYPH_CHANNELS = 4;

// --- Action types ---
export const enum ActionType {
  IDLE      = 0,
  MOVE      = 1,
  EAT       = 2,
  REPRODUCE = 3,
  SIGNAL    = 4,
  ATTACK    = 5,
  DEPOSIT   = 6,
  ABSORB    = 7,
}

// --- Resource distribution types ---
export const enum ResourceDist {
  UNIFORM   = 0,
  CLUSTERED = 1,
  GRADIENT  = 2,
}
