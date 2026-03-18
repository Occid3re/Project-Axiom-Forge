/**
 * WorldView: WebGL canvas driven by requestAnimationFrame.
 * Socket frames stay out of React state; the director camera runs locally.
 */
import { useEffect, useRef, useState } from 'react';
import { WorldRenderer } from '../renderer';
import type { DecodedEntityFrame, DecodedFieldFrame } from '../../engine/protocol';

interface WorldViewProps {
  entityFrameRef: React.RefObject<DecodedEntityFrame | null>;
  fieldFrameRef: React.RefObject<DecodedFieldFrame | null>;
  className?: string;
}

interface DirectorHud {
  title: string;
  subtitle: string;
}

interface SpeciesMemory {
  firstSeenMs: number;
  lastSeenMs: number;
  lastSpotlightMs: number;
}

interface DirectorShot {
  bucket: number;
  panX: number;
  panY: number;
  zoom: number;
  title: string;
  subtitle: string;
}

interface BucketAggregate {
  bucket: number;
  count: number;
  energySum: number;
  complexitySum: number;
  aggressionSum: number;
  sumCosX: number;
  sumSinX: number;
  sumCosY: number;
  sumSinY: number;
}

interface DirectorState {
  speciesMemory: Map<number, SpeciesMemory>;
  shot: DirectorShot | null;
  shotEndsAt: number;
  nextCutAt: number;
  manualUntil: number;
  wideShot: boolean;
}

const ZOOM_MIN = 0.05;
const ZOOM_MAX = 1.0;
const DIRECTOR_SPECIES_BUCKETS = 24;
const DIRECTOR_MANUAL_HOLD_MS = 18_000;
const DIRECTOR_WIDE_SHOT_MS = 5_000;
const DIRECTOR_SPOTLIGHT_MS = 12_000;
const DIRECTOR_IDLE_SWITCH_MS = 4_500;
const SPECIMEN_LIMIT = 28;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function wrap01(value: number) {
  const wrapped = value % 1;
  return wrapped < 0 ? wrapped + 1 : wrapped;
}

function lerpWrapped(from: number, to: number, alpha: number) {
  let delta = to - from;
  if (delta > 0.5) delta -= 1;
  if (delta < -0.5) delta += 1;
  return wrap01(from + delta * alpha);
}

function wrapDistance01(from: number, to: number) {
  const direct = Math.abs(to - from);
  return Math.min(direct, 1 - direct);
}

function signedWrapDelta01(from: number, to: number) {
  let delta = from - to;
  if (delta > 0.5) delta -= 1;
  if (delta < -0.5) delta += 1;
  return delta;
}

export function WorldView({ entityFrameRef, fieldFrameRef, className = '' }: WorldViewProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WorldRenderer | null>(null);

  const viewRef = useRef({ panX: 0.5, panY: 0.5, zoom: 1.0 });
  const pointersRef = useRef(new Map<number, { x: number; y: number }>());
  const pinchDistRef = useRef<number | null>(null);
  const latestEntityFrameRef = useRef<DecodedEntityFrame | null>(null);
  const hudKeyRef = useRef('');
  const directorRef = useRef<DirectorState>({
    speciesMemory: new Map(),
    shot: null,
    shotEndsAt: 0,
    nextCutAt: 0,
    manualUntil: 0,
    wideShot: false,
  });
  const [directorHud, setDirectorHud] = useState<DirectorHud>({
    title: 'Specimen camera warming up',
    subtitle: 'Scanning for a lineage worth isolating',
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      rendererRef.current = new WorldRenderer(canvas);
    } catch (error) {
      console.error('[WorldView] WebGL init failed:', error);
    }
    return () => {
      rendererRef.current?.destroy();
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const observer = new ResizeObserver(entries => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      const dpr = Math.min(devicePixelRatio, 2);
      const size = Math.min(rect.width, rect.height) * dpr;
      rendererRef.current?.resize(Math.round(size), Math.round(size));
    });
    observer.observe(wrap);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let rafId = 0;

    const setHud = (title: string, subtitle: string) => {
      const key = `${title}|${subtitle}`;
      if (hudKeyRef.current === key) return;
      hudKeyRef.current = key;
      setDirectorHud({ title, subtitle });
    };

    const chooseDirectorShot = (frame: DecodedEntityFrame, ms: number): DirectorShot | null => {
      const director = directorRef.current;
      const buckets = new Map<number, BucketAggregate>();
      const total = Math.max(1, frame.entityCount);
      const tau = Math.PI * 2;

      for (let i = 0; i < frame.entityCount; i++) {
        const hue = frame.entitySpeciesHue[i] / 255;
        const bucket = Math.min(
          DIRECTOR_SPECIES_BUCKETS - 1,
          Math.floor(hue * DIRECTOR_SPECIES_BUCKETS),
        );
        let aggregate = buckets.get(bucket);
        if (!aggregate) {
          aggregate = {
            bucket,
            count: 0,
            energySum: 0,
            complexitySum: 0,
            aggressionSum: 0,
            sumCosX: 0,
            sumSinX: 0,
            sumCosY: 0,
            sumSinY: 0,
          };
          buckets.set(bucket, aggregate);
        }

        const xPhase = (frame.entityX[i] / Math.max(1, frame.gridW)) * tau;
        const yPhase = (frame.entityY[i] / Math.max(1, frame.gridH)) * tau;
        aggregate.count += 1;
        aggregate.energySum += frame.entityEnergy[i] / 255;
        aggregate.complexitySum += frame.entityComplexity[i] / 255;
        aggregate.aggressionSum += frame.entityAggression[i] / 255;
        aggregate.sumCosX += Math.cos(xPhase);
        aggregate.sumSinX += Math.sin(xPhase);
        aggregate.sumCosY += Math.cos(yPhase);
        aggregate.sumSinY += Math.sin(yPhase);
      }

      let bestShot: DirectorShot | null = null;
      let bestScore = -Infinity;

      for (const aggregate of buckets.values()) {
        if (aggregate.count < 6 || aggregate.count > 220) continue;

        const avgComplexity = aggregate.complexitySum / aggregate.count;
        const avgEnergy = aggregate.energySum / aggregate.count;
        const avgAggression = aggregate.aggressionSum / aggregate.count;
        const density = aggregate.count / total;
        const rarity = clamp(1 - density * 6, 0, 1);
        const stability = clamp(
          Math.hypot(aggregate.sumCosX, aggregate.sumSinX) / aggregate.count,
          0.18,
          1,
        );

        let memory = director.speciesMemory.get(aggregate.bucket);
        if (!memory) {
          memory = {
            firstSeenMs: ms,
            lastSeenMs: ms,
            lastSpotlightMs: -Infinity,
          };
          director.speciesMemory.set(aggregate.bucket, memory);
        } else {
          memory.lastSeenMs = ms;
        }

        const ageMs = ms - memory.firstSeenMs;
        const freshness = clamp(1 - ageMs / 45_000, 0, 1);
        const revisitPenalty = clamp((ms - memory.lastSpotlightMs) / 35_000, 0.25, 1);
        const score =
          (rarity * 0.35
            + avgComplexity * 0.3
            + avgEnergy * 0.2
            + freshness * 0.15)
          * stability
          * revisitPenalty;

        if (score <= bestScore) continue;

        const centerX = wrap01(Math.atan2(aggregate.sumSinX, aggregate.sumCosX) / tau);
        const centerY = wrap01(Math.atan2(aggregate.sumSinY, aggregate.sumCosY) / tau);
        let title = 'Prepared specimen slide';
        if (ageMs < 20_000) {
          title = 'New lineage on slide';
        } else if (avgAggression > 0.62) {
          title = 'Predator specimen isolate';
        } else if (avgComplexity > 0.62) {
          title = 'Complex lineage isolate';
        }

        bestScore = score;
        bestShot = {
          bucket: aggregate.bucket,
          panX: centerX,
          panY: centerY,
          zoom: 0.94,
          title,
          subtitle: `${Math.min(SPECIMEN_LIMIT, aggregate.count)} specimens from a ${aggregate.count}-body colony`,
        };
      }

      return bestShot;
    };

    const buildSpecimenFrame = (
      frame: DecodedEntityFrame,
      shot: DirectorShot | null,
    ): DecodedEntityFrame => {
      if (!shot || shot.bucket < 0 || frame.entityCount <= SPECIMEN_LIMIT) return frame;

      const selected: Array<{ index: number; score: number }> = [];

      for (let i = 0; i < frame.entityCount; i++) {
        const hue = frame.entitySpeciesHue[i] / 255;
        const bucket = Math.min(
          DIRECTOR_SPECIES_BUCKETS - 1,
          Math.floor(hue * DIRECTOR_SPECIES_BUCKETS),
        );
        if (bucket !== shot.bucket) continue;

        const x = frame.entityX[i] / Math.max(1, frame.gridW);
        const y = frame.entityY[i] / Math.max(1, frame.gridH);
        const dx = wrapDistance01(x, shot.panX);
        const dy = wrapDistance01(y, shot.panY);
        const distSq = dx * dx + dy * dy;
        const energy = frame.entityEnergy[i] / 255;
        const complexity = frame.entityComplexity[i] / 255;
        const score = distSq - energy * 0.03 - complexity * 0.02;

        if (selected.length < SPECIMEN_LIMIT) {
          selected.push({ index: i, score });
          selected.sort((a, b) => a.score - b.score);
        } else if (score < selected[selected.length - 1].score) {
          selected[selected.length - 1] = { index: i, score };
          selected.sort((a, b) => a.score - b.score);
        }
      }

      if (selected.length < 4) return frame;

      const entityCount = selected.length;
      const padding = 0.18;
      let minDx = Infinity;
      let maxDx = -Infinity;
      let minDy = Infinity;
      let maxDy = -Infinity;
      const localPositions = new Array<{ dx: number; dy: number }>(entityCount);

      for (let i = 0; i < entityCount; i++) {
        const source = selected[i].index;
        const x = frame.entityX[source] / Math.max(1, frame.gridW);
        const y = frame.entityY[source] / Math.max(1, frame.gridH);
        const dx = signedWrapDelta01(x, shot.panX);
        const dy = signedWrapDelta01(y, shot.panY);
        localPositions[i] = { dx, dy };
        if (dx < minDx) minDx = dx;
        if (dx > maxDx) maxDx = dx;
        if (dy < minDy) minDy = dy;
        if (dy > maxDy) maxDy = dy;
      }

      const spanX = Math.max(0.06, maxDx - minDx);
      const spanY = Math.max(0.06, maxDy - minDy);
      const centerDx = (minDx + maxDx) * 0.5;
      const centerDy = (minDy + maxDy) * 0.5;
      const scale = (1 - padding * 2) / Math.max(spanX, spanY);
      const entityX = new Uint8Array(entityCount);
      const entityY = new Uint8Array(entityCount);
      const entityEnergy = new Uint8Array(entityCount);
      const entityAction = new Uint8Array(entityCount);
      const entityAggression = new Uint8Array(entityCount);
      const entitySpeciesHue = new Uint8Array(entityCount);
      const entityComplexity = new Uint8Array(entityCount);
      const entityMotility = new Uint8Array(entityCount);

      for (let i = 0; i < entityCount; i++) {
        const source = selected[i].index;
        const local = localPositions[i];
        const specimenX = clamp(0.5 + (local.dx - centerDx) * scale, padding, 1 - padding);
        const specimenY = clamp(0.5 + (local.dy - centerDy) * scale, padding, 1 - padding);
        entityX[i] = Math.round(specimenX * (frame.gridW - 1));
        entityY[i] = Math.round(specimenY * (frame.gridH - 1));
        entityEnergy[i] = frame.entityEnergy[source];
        entityAction[i] = frame.entityAction[source];
        entityAggression[i] = frame.entityAggression[source];
        entitySpeciesHue[i] = frame.entitySpeciesHue[source];
        entityComplexity[i] = frame.entityComplexity[source];
        entityMotility[i] = frame.entityMotility[source];
      }

      return {
        gridW: frame.gridW,
        gridH: frame.gridH,
        entityCount,
        tick: frame.tick,
        entityX,
        entityY,
        entityEnergy,
        entityAction,
        entityAggression,
        entitySpeciesHue,
        entityComplexity,
        entityMotility,
        specimenMode: true,
        renderScale: clamp(2.0 + 0.35 / Math.max(spanX, spanY), 2.2, 3.2),
      } as DecodedEntityFrame;
    };

    const loop = (ms: number) => {
      const renderer = rendererRef.current;
      if (renderer) {
        const director = directorRef.current;
        const fieldFrame = fieldFrameRef.current;
        if (fieldFrame) {
          renderer.updateFieldFrame(fieldFrame);
          (fieldFrameRef as React.MutableRefObject<DecodedFieldFrame | null>).current = null;
        }

        const entityFrame = entityFrameRef.current;
        if (entityFrame) {
          latestEntityFrameRef.current = entityFrame;
          const displayFrame = ms < director.manualUntil
            ? entityFrame
            : buildSpecimenFrame(entityFrame, director.shot);
          renderer.updateEntityFrame(displayFrame);
          (entityFrameRef as React.MutableRefObject<DecodedEntityFrame | null>).current = null;
        }

        const shot = director.shot;

        if (ms < director.manualUntil) {
          const seconds = Math.max(1, Math.ceil((director.manualUntil - ms) / 1000));
          setHud('Manual control', `Auto specimen camera returns in ${seconds}s`);
        } else {
          if (shot && ms >= director.shotEndsAt) {
            director.shot = null;
          }

          if (!director.shot && ms >= director.nextCutAt) {
            if (!director.wideShot) {
              director.shot = {
                bucket: -1,
                panX: 0.5,
                panY: 0.5,
                zoom: 0.82,
                title: 'Dish overview',
                subtitle: 'Background breeding stays live while the camera searches',
              };
              director.shotEndsAt = ms + DIRECTOR_WIDE_SHOT_MS;
              director.nextCutAt = director.shotEndsAt;
              director.wideShot = true;
            } else {
              const frame = latestEntityFrameRef.current;
              const spotlight = frame ? chooseDirectorShot(frame, ms) : null;
              if (spotlight) {
                director.shot = spotlight;
                director.shotEndsAt = ms + DIRECTOR_SPOTLIGHT_MS;
                director.nextCutAt = director.shotEndsAt;
                director.wideShot = false;
                const memory = director.speciesMemory.get(spotlight.bucket);
                if (memory) memory.lastSpotlightMs = ms;
                if (frame) {
                  renderer.updateEntityFrame(buildSpecimenFrame(frame, spotlight));
                }
              } else {
                director.nextCutAt = ms + DIRECTOR_IDLE_SWITCH_MS;
                setHud('Specimen camera scanning', 'Waiting for a distinct colony worth isolating');
              }
            }
          }

          if (director.shot) {
            const current = viewRef.current;
            const targetPanX = director.shot.bucket >= 0 ? 0.5 : director.shot.panX;
            const targetPanY = director.shot.bucket >= 0 ? 0.5 : director.shot.panY;
            current.panX = lerpWrapped(current.panX, targetPanX, 0.024);
            current.panY = lerpWrapped(current.panY, targetPanY, 0.024);
            current.zoom += (director.shot.zoom - current.zoom) * 0.024;
            renderer.setView(current.panX, current.panY, current.zoom);
            setHud(director.shot.title, director.shot.subtitle);
          }
        }

        renderer.render(ms);
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [entityFrameRef, fieldFrameRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const holdManualControl = (title: string) => {
      directorRef.current.manualUntil = performance.now() + DIRECTOR_MANUAL_HOLD_MS;
      directorRef.current.shot = null;
      directorRef.current.nextCutAt = directorRef.current.manualUntil;
      hudKeyRef.current = '';
      setDirectorHud({
        title,
        subtitle: 'Auto specimen camera returns in 18s',
      });
    };

    const applyZoom = (factor: number, canvasNX: number, canvasNY: number) => {
      const view = viewRef.current;
      const newZoom = clamp(view.zoom * factor, ZOOM_MIN, ZOOM_MAX);
      const cx = canvasNX - 0.5;
      const cy = canvasNY - 0.5;
      const worldX = view.panX + cx * view.zoom;
      const worldY = view.panY + cy * view.zoom;
      view.panX = wrap01(worldX - cx * newZoom);
      view.panY = wrap01(worldY - cy * newZoom);
      view.zoom = newZoom;
      rendererRef.current?.setView(view.panX, view.panY, view.zoom);
    };

    const onPointerDown = (event: PointerEvent) => {
      event.preventDefault();
      holdManualControl('Manual control');
      canvas.setPointerCapture(event.pointerId);
      pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (pointersRef.current.size === 2) {
        const points = Array.from(pointersRef.current.values());
        pinchDistRef.current = Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y);
      }
    };

    const onPointerMove = (event: PointerEvent) => {
      const previous = pointersRef.current.get(event.pointerId);
      if (!previous) return;

      const currentPoint = { x: event.clientX, y: event.clientY };
      pointersRef.current.set(event.pointerId, currentPoint);

      if (pointersRef.current.size === 1) {
        const rect = canvas.getBoundingClientRect();
        const dx = (currentPoint.x - previous.x) / rect.width;
        const dy = (currentPoint.y - previous.y) / rect.height;
        const view = viewRef.current;
        view.panX = wrap01(view.panX - dx * view.zoom);
        view.panY = wrap01(view.panY + dy * view.zoom);
        rendererRef.current?.setView(view.panX, view.panY, view.zoom);
      } else if (pointersRef.current.size >= 2) {
        const points = Array.from(pointersRef.current.values());
        const newDist = Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y);
        if (pinchDistRef.current !== null && pinchDistRef.current > 1 && newDist > 1) {
          const factor = pinchDistRef.current / newDist;
          const rect = canvas.getBoundingClientRect();
          const midX = ((points[0].x + points[1].x) / 2 - rect.left) / rect.width;
          const midY = ((points[0].y + points[1].y) / 2 - rect.top) / rect.height;
          applyZoom(factor, midX, midY);
        }
        pinchDistRef.current = newDist;
      }
    };

    const onPointerUp = (event: PointerEvent) => {
      pointersRef.current.delete(event.pointerId);
      if (pointersRef.current.size < 2) pinchDistRef.current = null;
    };

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      holdManualControl('Manual zoom');
      const rect = canvas.getBoundingClientRect();
      const nx = (event.clientX - rect.left) / rect.width;
      const ny = (event.clientY - rect.top) / rect.height;
      const factor = 1 - event.deltaY * 0.0012;
      applyZoom(factor, nx, ny);
    };

    const onDblClick = () => {
      holdManualControl('Manual reset');
      viewRef.current = { panX: 0.5, panY: 0.5, zoom: 1.0 };
      rendererRef.current?.setView(0.5, 0.5, 1.0);
    };

    canvas.addEventListener('pointerdown', onPointerDown, { passive: false });
    canvas.addEventListener('pointermove', onPointerMove, { passive: true });
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointercancel', onPointerUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('dblclick', onDblClick);

    return () => {
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('dblclick', onDblClick);
    };
  }, []);

  return (
    <div
      ref={wrapRef}
      className={`relative flex items-center justify-center ${className}`}
    >
      <canvas
        ref={canvasRef}
        style={{
          aspectRatio: '1 / 1',
          maxWidth: '100%',
          maxHeight: '100%',
          cursor: 'crosshair',
          touchAction: 'none',
        }}
        title="Scroll to zoom · Drag to pan · Double-click to reset"
      />
      <div className="pointer-events-none absolute left-3 top-3 max-w-[240px] rounded-xl border border-white/10 bg-black/45 px-3 py-2 text-white backdrop-blur-md">
        <div className="text-[10px] uppercase tracking-[0.28em] text-cyan-300/80">
          Specimen Camera
        </div>
        <div className="mt-1 text-sm font-medium leading-tight">
          {directorHud.title}
        </div>
        <div className="mt-1 text-xs leading-snug text-white/70">
          {directorHud.subtitle}
        </div>
      </div>
    </div>
  );
}
