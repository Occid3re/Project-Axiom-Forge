/**
 * Binary frame decoder — matches server/simulation.ts packFrame().
 * Decodes the ArrayBuffer received from the server into renderer-ready data.
 */

export interface DecodedFrame {
  gridW: number;
  gridH: number;
  entityCount: number;
  tick: number;
  resources: Uint8Array;   // W*H uint8
  signals: Uint8Array;     // W*H*3 uint8
  poison: Uint8Array;      // W*H uint8 — toxin concentration
  entityX: Uint8Array;
  entityY: Uint8Array;
  entityEnergy: Uint8Array;
  entityAction: Uint8Array;
  entityAggression: Uint8Array;
  entitySpeciesHue: Uint8Array;
  entityComplexity: Uint8Array;  // genome weight std dev → 0-255 (evolution stage)
  entityMotility: Uint8Array;    // W2 MOVE column drive → 0-255
}

const MAGIC = 0x41584647;

export function decodeFrame(buf: ArrayBuffer): DecodedFrame | null {
  if (buf.byteLength < 20) return null;
  const view = new DataView(buf);
  if (view.getUint32(0, true) !== MAGIC) return null;

  const gridW       = view.getUint32(4, true);
  const gridH       = view.getUint32(8, true);
  const entityCount = view.getUint32(12, true);
  const tick        = view.getUint32(16, true);

  const cells = gridW * gridH;
  const u8 = new Uint8Array(buf);

  let offset = 20;
  const resources = u8.slice(offset, offset + cells); offset += cells;
  const signals   = u8.slice(offset, offset + cells * 3); offset += cells * 3;
  const poison    = u8.slice(offset, offset + cells); offset += cells;

  const entityX          = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityY          = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityEnergy     = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityAction     = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityAggression = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entitySpeciesHue = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityComplexity = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityMotility   = u8.slice(offset, offset + entityCount);

  return { gridW, gridH, entityCount, tick, resources, signals, poison, entityX, entityY, entityEnergy, entityAction, entityAggression, entitySpeciesHue, entityComplexity, entityMotility };
}

export interface ServerMeta {
  generation: number;
  worldIndex: number;
  totalWorlds: number;
  tick: number;
  displayTick?: number;
  bestLaws: import('./world-laws').WorldLaws | null;
  population: number;
  scores: {
    persistence: number; diversity: number; complexityGrowth: number;
    communication: number; envStructure: number; adaptability: number;
    speciation: number; interactions: number; total: number;
  } | null;
  bestScore: number;
  generations: Array<{ gen: number; best: number; avg: number }>;
  logEntry: string | null;
  gridSize: number;
  evalSpeed?: number;
  serverMs?: number;       // EMA ms per display step — server load indicator
  serverPressure?: number; // 0-2: environmental harshness from server load
  sampleGenome?: number[]; // 80 MLP weights of the most-energetic display entity
}
