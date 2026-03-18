/**
 * WorldView: manual microscope controls with detail scaling on zoom.
 */
import { useEffect, useRef, useState } from 'react';
import { WorldRenderer } from '../renderer';
import type { DecodedEntityFrame, DecodedFieldFrame } from '../../engine/protocol';

interface WorldViewProps {
  entityFrameRef: React.RefObject<DecodedEntityFrame | null>;
  fieldFrameRef: React.RefObject<DecodedFieldFrame | null>;
  className?: string;
}

const ZOOM_MIN = 0.02;
const ZOOM_MAX = 1.0;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function wrap01(value: number) {
  const wrapped = value % 1;
  return wrapped < 0 ? wrapped + 1 : wrapped;
}

function zoomDetailBoost(zoom: number) {
  if (zoom > 0.48) return 1.4;
  if (zoom > 0.18) return 2.2;
  return 2.8;
}

function atlasLabel(boost: number) {
  return boost >= 2.2 ? '4x' : '2x';
}

export function WorldView({ entityFrameRef, fieldFrameRef, className = '' }: WorldViewProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WorldRenderer | null>(null);
  const viewRef = useRef({ panX: 0.5, panY: 0.5, zoom: 1.0 });
  const pointersRef = useRef(new Map<number, { x: number; y: number }>());
  const pinchDistRef = useRef<number | null>(null);
  const [detailBoost, setDetailBoost] = useState(1);

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
    const loop = (ms: number) => {
      const renderer = rendererRef.current;
      if (renderer) {
        const fieldFrame = fieldFrameRef.current;
        if (fieldFrame) {
          renderer.updateFieldFrame(fieldFrame);
          (fieldFrameRef as React.MutableRefObject<DecodedFieldFrame | null>).current = null;
        }

        const entityFrame = entityFrameRef.current;
        if (entityFrame) {
          const boost = zoomDetailBoost(viewRef.current.zoom);
          setDetailBoost(prev => (prev === boost ? prev : boost));
          renderer.updateEntityFrame({
            ...entityFrame,
            renderScale: boost,
          } as DecodedEntityFrame);
          (entityFrameRef as React.MutableRefObject<DecodedEntityFrame | null>).current = null;
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

    const syncDetail = () => {
      const boost = zoomDetailBoost(viewRef.current.zoom);
      setDetailBoost(prev => (prev === boost ? prev : boost));
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
      syncDetail();
    };

    const onPointerDown = (event: PointerEvent) => {
      event.preventDefault();
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
      const rect = canvas.getBoundingClientRect();
      const nx = (event.clientX - rect.left) / rect.width;
      const ny = (event.clientY - rect.top) / rect.height;
      const factor = 1 - event.deltaY * 0.0012;
      applyZoom(factor, nx, ny);
    };

    const onDblClick = () => {
      viewRef.current = { panX: 0.5, panY: 0.5, zoom: 1.0 };
      rendererRef.current?.setView(0.5, 0.5, 1.0);
      syncDetail();
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
        title="Scroll to zoom, drag to pan, double-click to reset"
      />
      <div className="pointer-events-none absolute left-3 top-3 max-w-[240px] rounded-xl border border-white/10 bg-black/45 px-3 py-2 text-white backdrop-blur-md">
        <div className="text-[10px] uppercase tracking-[0.28em] text-cyan-300/80">
          Manual Microscope
        </div>
        <div className="mt-1 text-sm font-medium leading-tight">
          Zoom for more detail
        </div>
        <div className="mt-1 text-xs leading-snug text-white/70">
          Entity atlas {atlasLabel(detailBoost)} · deep zoom increases morphology detail
        </div>
      </div>
    </div>
  );
}
