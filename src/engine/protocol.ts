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
  entitySize: Uint8Array;        // simulated growth state → 0-255 maps to ~0..2x size
  entityColonyMass: Uint8Array;  // colony size / fused-body membership
  entityBodyRadius: Uint8Array;  // soft body footprint radius in cells
}

export interface DecodedFieldFrame {
  gridW: number;
  gridH: number;
  tick: number;
  resources: Uint8Array;   // W*H uint8
  signals: Uint8Array;     // W*H*3 uint8
  poison: Uint8Array;      // W*H uint8 — toxin concentration
  glyphs: Uint8Array;      // W*H uint8 — glyph magnitude (stigmergic memory)
  body: Uint8Array;        // W*H uint8 — macro-body occupancy envelope
}

const ENTITY_MAGIC = 0x41584645; // "AXFE"
const FIELD_MAGIC  = 0x41584646; // "AXFF"
const FIELD_HEADER_BYTES = 32;
const FIELD_PACKET_KIND_KEYFRAME = 0;
const FIELD_PACKET_KIND_DELTA = 1;
const FIELD_PLANE_RESOURCES = 1;
const FIELD_PLANE_SIGNALS = 2;
const FIELD_PLANE_POISON = 4;
const FIELD_PLANE_GLYPHS = 8;
const FIELD_PLANE_BODY = 16;

interface PackedFieldState {
  gridW: number;
  gridH: number;
  step: number;
  outW: number;
  outH: number;
  resources: Uint8Array;
  signals: Uint8Array;
  poison: Uint8Array;
  glyphs: Uint8Array;
  body: Uint8Array;
}

let lastPackedFieldState: PackedFieldState | null = null;

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

function applyScalarTile(
  dest: Uint8Array,
  source: Uint8Array,
  offset: number,
  outW: number,
  tileX: number,
  tileY: number,
  tileSize: number,
  tileW: number,
  tileH: number,
): number {
  const startX = tileX * tileSize;
  const startY = tileY * tileSize;
  for (let dy = 0; dy < tileH; dy++) {
    const rowStart = (startY + dy) * outW + startX;
    dest.set(source.subarray(offset, offset + tileW), rowStart);
    offset += tileW;
  }
  return offset;
}

function applySignalTile(
  dest: Uint8Array,
  source: Uint8Array,
  offset: number,
  outW: number,
  tileX: number,
  tileY: number,
  tileSize: number,
  tileW: number,
  tileH: number,
): number {
  const startX = tileX * tileSize;
  const startY = tileY * tileSize;
  for (let dy = 0; dy < tileH; dy++) {
    const rowStart = ((startY + dy) * outW + startX) * 3;
    dest.set(source.subarray(offset, offset + tileW * 3), rowStart);
    offset += tileW * 3;
  }
  return offset;
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
  const entityMotility   = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entitySize       = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityColonyMass = u8.slice(offset, offset + entityCount); offset += entityCount;
  const entityBodyRadius = u8.slice(offset, offset + entityCount);

  return {
    gridW,
    gridH,
    entityCount,
    tick,
    entityX,
    entityY,
    entityEnergy,
    entityAction,
    entityAggression,
    entitySpeciesHue,
    entityComplexity,
    entityMotility,
    entitySize,
    entityColonyMass,
    entityBodyRadius,
  };
}

export function decodeFieldFrame(buf: ArrayBuffer): DecodedFieldFrame | null {
  if (buf.byteLength < FIELD_HEADER_BYTES) return null;
  const view = new DataView(buf);
  if (view.getUint32(0, true) !== FIELD_MAGIC) return null;

  const gridW = view.getUint32(4, true);
  const gridH = view.getUint32(8, true);
  const step  = Math.max(1, view.getUint32(12, true));
  const tick  = view.getUint32(16, true);
  const kind = view.getUint32(20, true);
  const aux0 = view.getUint32(24, true);
  const aux1 = view.getUint32(28, true);
  const fieldW = Math.max(1, Math.floor(gridW / step));
  const fieldH = Math.max(1, Math.floor(gridH / step));
  const cells = fieldW * fieldH;
  const u8 = new Uint8Array(buf);

  if (kind === FIELD_PACKET_KIND_KEYFRAME) {
    let offset = FIELD_HEADER_BYTES;
    const packedResources = u8.slice(offset, offset + cells); offset += cells;
    const packedSignals   = u8.slice(offset, offset + cells * 3); offset += cells * 3;
    const packedPoison    = u8.slice(offset, offset + cells); offset += cells;
    const packedGlyphs    = u8.slice(offset, offset + cells); offset += cells;
    const packedBody      = u8.slice(offset, offset + cells);
    lastPackedFieldState = {
      gridW,
      gridH,
      step,
      outW: fieldW,
      outH: fieldH,
      resources: packedResources,
      signals: packedSignals,
      poison: packedPoison,
      glyphs: packedGlyphs,
      body: packedBody,
    };
  } else if (kind === FIELD_PACKET_KIND_DELTA) {
    const tileSize = Math.max(1, aux0);
    const tileCount = aux1;
    if (
      !lastPackedFieldState
      || lastPackedFieldState.gridW !== gridW
      || lastPackedFieldState.gridH !== gridH
      || lastPackedFieldState.step !== step
      || lastPackedFieldState.outW !== fieldW
      || lastPackedFieldState.outH !== fieldH
    ) {
      return null;
    }

    let offset = FIELD_HEADER_BYTES;
    for (let i = 0; i < tileCount; i++) {
      const tileX = view.getUint16(offset, true);
      const tileY = view.getUint16(offset + 2, true);
      const planeMask = u8[offset + 4];
      offset += 5;
      const tileW = Math.min(tileSize, fieldW - tileX * tileSize);
      const tileH = Math.min(tileSize, fieldH - tileY * tileSize);
      if (planeMask & FIELD_PLANE_RESOURCES) {
        offset = applyScalarTile(lastPackedFieldState.resources, u8, offset, fieldW, tileX, tileY, tileSize, tileW, tileH);
      }
      if (planeMask & FIELD_PLANE_SIGNALS) {
        offset = applySignalTile(lastPackedFieldState.signals, u8, offset, fieldW, tileX, tileY, tileSize, tileW, tileH);
      }
      if (planeMask & FIELD_PLANE_POISON) {
        offset = applyScalarTile(lastPackedFieldState.poison, u8, offset, fieldW, tileX, tileY, tileSize, tileW, tileH);
      }
      if (planeMask & FIELD_PLANE_GLYPHS) {
        offset = applyScalarTile(lastPackedFieldState.glyphs, u8, offset, fieldW, tileX, tileY, tileSize, tileW, tileH);
      }
      if (planeMask & FIELD_PLANE_BODY) {
        offset = applyScalarTile(lastPackedFieldState.body, u8, offset, fieldW, tileX, tileY, tileSize, tileW, tileH);
      }
    }
  } else {
    return null;
  }

  if (!lastPackedFieldState) return null;

  const resources = expandScalarField(lastPackedFieldState.resources, gridW, gridH, step);
  const signals = expandSignalField(lastPackedFieldState.signals, gridW, gridH, step);
  const poison = expandScalarField(lastPackedFieldState.poison, gridW, gridH, step);
  const glyphs = expandScalarField(lastPackedFieldState.glyphs, gridW, gridH, step);
  const body = expandScalarField(lastPackedFieldState.body, gridW, gridH, step);

  return { gridW, gridH, tick, resources, signals, poison, glyphs, body };
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
    seasonalAdaptation: number; lifetimeLearning: number;
    total: number;
  } | null;
  bestScore: number;
  generations: Array<{ gen: number; best: number; avg: number }>;
  logEntry: string | null;
  gridSize: number;
  evalSpeed?: number;
  serverMs?: number;       // EMA ms per display step — server load indicator
  serverPressure?: number; // 0-2: environmental harshness from server load
  sampleNetwork?: {
    entityId: number;
    action: number;
    age: number;
    energy: number;
    size: number;
    kinNeighbors: number;
    threatNeighbors: number;
    lockTicksRemaining: number;
    inputs: number[];
    hidden1: number[];
    hidden2: number[];
    probs: number[];
    genome: number[];
  };
  displaySeed?: number;    // changes every time the display world is restarted
}
