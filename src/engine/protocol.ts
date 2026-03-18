/**
 * Binary packet decoders — match server/simulation.ts packEntityFrame()/packFieldFrame().
 * Entity packets arrive every display tick; field packets arrive less frequently.
 */

export interface DecodedEntityFrame {
  gridW: number;
  gridH: number;
  entityCount: number;
  tick: number;
  entityX: Uint8Array;
  entityY: Uint8Array;
  entityEnergy: Uint8Array;
  entityAction: Uint8Array;
  entityAggression: Uint8Array;
  entitySpeciesHue: Uint8Array;
  entityComplexity: Uint8Array;  // genome weight std dev → 0-255 (evolution stage)
  entityMotility: Uint8Array;    // W2 MOVE column drive → 0-255
}

export interface DecodedFieldFrame {
  gridW: number;
  gridH: number;
  tick: number;
  resources: Uint8Array;   // W*H uint8
  signals: Uint8Array;     // W*H*3 uint8
  poison: Uint8Array;      // W*H uint8 — toxin concentration
  glyphs: Uint8Array;      // W*H uint8 — glyph magnitude (stigmergic memory)
}

const ENTITY_MAGIC = 0x41584645; // "AXFE"
const FIELD_MAGIC  = 0x41584646; // "AXFF"

function expandScalarField(
  source: Uint8Array,
  gridW: number,
  gridH: number,
  step: number,
): Uint8Array {
  const out = new Uint8Array(gridW * gridH);
  const srcW = Math.max(1, Math.floor(gridW / step));
  const srcH = Math.max(1, Math.floor(gridH / step));

  for (let sy = 0; sy < srcH; sy++) {
    for (let sx = 0; sx < srcW; sx++) {
      const value = source[sy * srcW + sx];
      const startY = sy * step;
      const startX = sx * step;
      for (let dy = 0; dy < step && startY + dy < gridH; dy++) {
        const row = (startY + dy) * gridW;
        for (let dx = 0; dx < step && startX + dx < gridW; dx++) {
          out[row + startX + dx] = value;
        }
      }
    }
  }

  return out;
}

function expandSignalField(
  source: Uint8Array,
  gridW: number,
  gridH: number,
  step: number,
): Uint8Array {
  const out = new Uint8Array(gridW * gridH * 3);
  const srcW = Math.max(1, Math.floor(gridW / step));
  const srcH = Math.max(1, Math.floor(gridH / step));

  for (let sy = 0; sy < srcH; sy++) {
    for (let sx = 0; sx < srcW; sx++) {
      const srcBase = (sy * srcW + sx) * 3;
      const startY = sy * step;
      const startX = sx * step;
      for (let dy = 0; dy < step && startY + dy < gridH; dy++) {
        for (let dx = 0; dx < step && startX + dx < gridW; dx++) {
          const outBase = ((startY + dy) * gridW + startX + dx) * 3;
          out[outBase] = source[srcBase];
          out[outBase + 1] = source[srcBase + 1];
          out[outBase + 2] = source[srcBase + 2];
        }
      }
    }
  }

  return out;
}

export function decodeEntityFrame(buf: ArrayBuffer): DecodedEntityFrame | null {
  if (buf.byteLength < 20) return null;
  const view = new DataView(buf);
  if (view.getUint32(0, true) !== ENTITY_MAGIC) return null;

  const gridW       = view.getUint32(4, true);
  const gridH       = view.getUint32(8, true);
  const entityCount = view.getUint32(12, true);
  const tick        = view.getUint32(16, true);
  const u8 = new Uint8Array(buf);

  let offset = 20;
  const entityX          = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityY          = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityEnergy     = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityAction     = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityAggression = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entitySpeciesHue = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityComplexity = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityMotility   = u8.slice(offset, offset + entityCount);

  return { gridW, gridH, entityCount, tick, entityX, entityY, entityEnergy, entityAction, entityAggression, entitySpeciesHue, entityComplexity, entityMotility };
}

export function decodeFieldFrame(buf: ArrayBuffer): DecodedFieldFrame | null {
  if (buf.byteLength < 20) return null;
  const view = new DataView(buf);
  if (view.getUint32(0, true) !== FIELD_MAGIC) return null;

  const gridW = view.getUint32(4, true);
  const gridH = view.getUint32(8, true);
  const step  = Math.max(1, view.getUint32(12, true));
  const tick  = view.getUint32(16, true);
  const fieldW = Math.max(1, Math.floor(gridW / step));
  const fieldH = Math.max(1, Math.floor(gridH / step));
  const cells = fieldW * fieldH;
  const u8 = new Uint8Array(buf);

  let offset = 20;
  const packedResources = u8.slice(offset, offset + cells); offset += cells;
  const packedSignals   = u8.slice(offset, offset + cells * 3); offset += cells * 3;
  const packedPoison    = u8.slice(offset, offset + cells); offset += cells;
  const packedGlyphs    = u8.slice(offset, offset + cells);

  const resources = expandScalarField(packedResources, gridW, gridH, step);
  const signals = expandSignalField(packedSignals, gridW, gridH, step);
  const poison = expandScalarField(packedPoison, gridW, gridH, step);
  const glyphs = expandScalarField(packedGlyphs, gridW, gridH, step);

  return { gridW, gridH, tick, resources, signals, poison, glyphs };
}

export interface ServerMeta {
  generation: number;
  worldIndex: number;
  totalWorlds: number;
  tick: number;
  displayTick?: number;
  bestLaws: import('./world-laws').WorldLaws | null;
  displayLaws?: import('./world-laws').WorldLaws | null;
  population: number;
  scores: {
    persistence: number; diversity: number; complexityGrowth: number;
    communication: number; envStructure: number; adaptability: number;
    speciation: number; interactions: number;
    spatialStructure: number; populationDynamics: number;
    stigmergicUse: number; socialDifferentiation: number;
    total: number;
  } | null;
  bestScore: number;
  generations: Array<{ gen: number; best: number; avg: number }>;
  logEntry: string | null;
  gridSize: number;
  evalSpeed?: number;
  serverMs?: number;       // EMA ms per display step — server load indicator
  serverPressure?: number; // 0-2: environmental harshness from server load
  sampleGenome?: number[]; // 270 MLP weights of the most-energetic display entity
  displaySeed?: number;    // changes every time the display world is restarted
}
