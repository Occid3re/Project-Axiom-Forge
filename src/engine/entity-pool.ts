/**
 * Entity pool — Struct-of-Arrays storage for all entities.
 * Zero allocation in the hot loop. All data in typed arrays.
 */

import { GENOME_LENGTH, MAX_ENTITIES, MAX_MEMORY_SIZE, ActionType, NN_INPUTS, NN_HIDDEN, NN_W1_SIZE } from './constants';
import { PRNG } from './world-laws';

export class EntityPool {
  readonly capacity: number;
  count: number = 0;
  private nextId: number = 0;

  // --- Parallel arrays (SoA layout) ---
  readonly id: Int32Array;
  readonly x: Int32Array;
  readonly y: Int32Array;
  readonly energy: Float32Array;
  readonly size: Float32Array;
  readonly age: Int32Array;
  readonly alive: Uint8Array;
  readonly parentId: Int32Array;
  readonly genomes: Float32Array; // [capacity * GENOME_LENGTH] — MLP weights (real-valued)
  readonly memory: Float32Array;  // [capacity * MAX_MEMORY_SIZE]
  readonly action: Uint8Array;    // last action taken
  readonly actionDx: Int8Array;
  readonly actionDy: Int8Array;

  constructor(capacity: number = MAX_ENTITIES) {
    this.capacity = capacity;
    this.id       = new Int32Array(capacity);
    this.x        = new Int32Array(capacity);
    this.y        = new Int32Array(capacity);
    this.energy   = new Float32Array(capacity);
    this.size     = new Float32Array(capacity);
    this.age      = new Int32Array(capacity);
    this.alive    = new Uint8Array(capacity);
    this.parentId = new Int32Array(capacity);
    this.genomes  = new Float32Array(capacity * GENOME_LENGTH);
    this.memory   = new Float32Array(capacity * MAX_MEMORY_SIZE);
    this.action   = new Uint8Array(capacity);
    this.actionDx = new Int8Array(capacity);
    this.actionDy = new Int8Array(capacity);
  }

  spawn(
    px: number, py: number, energy: number,
    genome: Float32Array | null, parentId: number,
    rng: PRNG,
  ): number {
    if (this.count >= this.capacity) return -1;

    const i   = this.count++;
    const eid = this.nextId++;
    this.id[i]       = eid;
    this.x[i]        = px;
    this.y[i]        = py;
    this.energy[i]   = energy;
    this.size[i]     = 0.8;
    this.age[i]      = 0;
    this.alive[i]    = 1;
    this.parentId[i] = parentId;
    this.action[i]   = ActionType.IDLE;
    this.actionDx[i] = 0;
    this.actionDy[i] = 0;

    const gOffset = i * GENOME_LENGTH;
    if (genome) {
      this.genomes.set(genome, gOffset);
    } else {
      // Xavier (Glorot) normal init — keeps activations well-scaled at birth.
      // W1 (inputs→hidden): fan_in = NN_INPUTS  → std = √(2 / NN_INPUTS)
      // W2 (hidden→output): fan_in = NN_HIDDEN  → std = √(2 / NN_HIDDEN)
      const stdW1 = Math.sqrt(2 / NN_INPUTS);
      const stdW2 = Math.sqrt(2 / NN_HIDDEN);
      for (let g = 0; g < GENOME_LENGTH; g++) {
        this.genomes[gOffset + g] = rng.normal(0, g < NN_W1_SIZE ? stdW1 : stdW2);
      }
    }

    const mOffset = i * MAX_MEMORY_SIZE;
    for (let m = 0; m < MAX_MEMORY_SIZE; m++) {
      this.memory[mOffset + m] = 0;
    }

    return i;
  }

  kill(i: number): void {
    this.alive[i] = 0;
  }

  /** Compact: remove dead entities by copying live ones forward. */
  compact(): void {
    let write = 0;
    for (let read = 0; read < this.count; read++) {
      if (!this.alive[read]) continue;
      if (write !== read) {
        this.id[write]       = this.id[read];
        this.x[write]        = this.x[read];
        this.y[write]        = this.y[read];
        this.energy[write]   = this.energy[read];
        this.size[write]     = this.size[read];
        this.age[write]      = this.age[read];
        this.alive[write]    = 1;
        this.parentId[write] = this.parentId[read];
        this.action[write]   = this.action[read];
        this.actionDx[write] = this.actionDx[read];
        this.actionDy[write] = this.actionDy[read];

        const wg = write * GENOME_LENGTH, rg = read * GENOME_LENGTH;
        this.genomes.copyWithin(wg, rg, rg + GENOME_LENGTH);

        const wm = write * MAX_MEMORY_SIZE, rm = read * MAX_MEMORY_SIZE;
        this.memory.copyWithin(wm, rm, rm + MAX_MEMORY_SIZE);
      }
      write++;
    }
    this.count = write;
  }

  getGenome(i: number): Float32Array {
    const offset = i * GENOME_LENGTH;
    return this.genomes.subarray(offset, offset + GENOME_LENGTH);
  }

  getMemory(i: number, size: number): Float32Array {
    const offset = i * MAX_MEMORY_SIZE;
    return this.memory.subarray(offset, offset + size);
  }

  /**
   * Mutate a genome in-place for an offspring.
   * MLP weights are real-valued floats — add Gaussian noise, soft-clamp at ±6.
   */
  mutateGenome(i: number, rate: number, strength: number, rng: PRNG): void {
    const offset = i * GENOME_LENGTH;
    for (let g = 0; g < GENOME_LENGTH; g++) {
      if (rng.random() < rate) {
        let val = this.genomes[offset + g] + rng.normal(0, strength);
        if (val >  6) val =  6;
        if (val < -6) val = -6;
        this.genomes[offset + g] = val;
      }
    }
  }
}
