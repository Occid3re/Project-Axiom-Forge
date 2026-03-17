/**
 * WorldView — WebGL canvas driven by a requestAnimationFrame loop.
 * Accepts a frameRef so socket frames skip React state entirely.
 * Renders at 60fps using whatever data was last received from the server.
 *
 * Interaction (unified Pointer Events — works for mouse and touch):
 *   - 1 finger / mouse drag  → pan (toroidal wrap)
 *   - 2 fingers pinch/spread → zoom in/out
 *   - Scroll wheel           → zoom in/out
 *   - Double-click/tap       → reset to 1:1
 */
import { useEffect, useRef } from 'react';
import { WorldRenderer } from '../renderer';
import type { DecodedFrame } from '../../engine/protocol';

interface WorldViewProps {
  frameRef: React.RefObject<DecodedFrame | null>;
  className?: string;
}

const ZOOM_MIN = 0.1;  // 10× zoom in
const ZOOM_MAX = 1.0;  // full world view — can't zoom out past this

export function WorldView({ frameRef, className = '' }: WorldViewProps) {
  const wrapRef     = useRef<HTMLDivElement>(null);
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WorldRenderer | null>(null);

  // View state — no re-renders needed
  const viewRef      = useRef({ panX: 0.5, panY: 0.5, zoom: 1.0 });
  // Unified pointer tracking: pointerId → current {x,y}
  const pointersRef  = useRef(new Map<number, { x: number; y: number }>());
  // Previous pinch distance — null when not pinching
  const pinchDistRef = useRef<number | null>(null);

  // Init renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      rendererRef.current = new WorldRenderer(canvas);
    } catch (e) {
      console.error('[WorldView] WebGL init failed:', e);
    }
    return () => {
      rendererRef.current?.destroy();
      rendererRef.current = null;
    };
  }, []);

  // Resize observer — always square, centered in container, never stretched
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const ro = new ResizeObserver(entries => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      const dpr  = Math.min(devicePixelRatio, 2);
      const size = Math.min(rect.width, rect.height) * dpr;
      rendererRef.current?.resize(Math.round(size), Math.round(size));
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  // RAF render loop
  useEffect(() => {
    let rafId: number;
    const loop = (ms: number) => {
      const r = rendererRef.current;
      if (r) {
        const f = frameRef.current;
        if (f) {
          r.updateFrame(f);
          (frameRef as React.MutableRefObject<DecodedFrame | null>).current = null;
        }
        r.render(ms);
      }
      rafId = requestAnimationFrame(loop);
    };
    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [frameRef]);

  // Zoom / pan via unified Pointer Events + wheel
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const applyZoom = (factor: number, canvasNX: number, canvasNY: number) => {
      const v = viewRef.current;
      const newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, v.zoom * factor));
      // Keep the world point under the focal position fixed
      const cx = canvasNX - 0.5;
      const cy = canvasNY - 0.5;
      const worldX = v.panX + cx * v.zoom;
      const worldY = v.panY + cy * v.zoom;
      v.panX = worldX - cx * newZoom;
      v.panY = worldY - cy * newZoom;
      v.zoom = newZoom;
      rendererRef.current?.setView(v.panX, v.panY, v.zoom);
    };

    // ── Pointer events (mouse + touch unified) ──────────────────────────────

    const onPointerDown = (e: PointerEvent) => {
      e.preventDefault();
      canvas.setPointerCapture(e.pointerId); // keep getting events if finger slides off
      pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
      // Reset pinch distance when a new finger lands
      if (pointersRef.current.size === 2) {
        const pts = Array.from(pointersRef.current.values());
        pinchDistRef.current = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
      }
    };

    const onPointerMove = (e: PointerEvent) => {
      const prev = pointersRef.current.get(e.pointerId);
      if (!prev) return;
      const cur = { x: e.clientX, y: e.clientY };
      pointersRef.current.set(e.pointerId, cur);

      if (pointersRef.current.size === 1) {
        // Single pointer — pan
        const rect = canvas.getBoundingClientRect();
        const dx = (cur.x - prev.x) / rect.width;
        const dy = (cur.y - prev.y) / rect.height;
        const v = viewRef.current;
        v.panX -= dx * v.zoom;
        v.panY += dy * v.zoom;
        rendererRef.current?.setView(v.panX, v.panY, v.zoom);
      } else if (pointersRef.current.size >= 2) {
        // Two pointers — pinch zoom toward midpoint
        const pts = Array.from(pointersRef.current.values());
        const newDist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
        if (pinchDistRef.current !== null && pinchDistRef.current > 1) {
          const factor = pinchDistRef.current / newDist;
          const rect = canvas.getBoundingClientRect();
          const midX = ((pts[0].x + pts[1].x) / 2 - rect.left) / rect.width;
          const midY = ((pts[0].y + pts[1].y) / 2 - rect.top) / rect.height;
          applyZoom(factor, midX, midY);
        }
        pinchDistRef.current = newDist;
      }
    };

    const onPointerUp = (e: PointerEvent) => {
      pointersRef.current.delete(e.pointerId);
      if (pointersRef.current.size < 2) pinchDistRef.current = null;
    };

    // ── Scroll wheel ────────────────────────────────────────────────────────

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width;
      const ny = (e.clientY - rect.top)  / rect.height;
      const factor = 1 - e.deltaY * 0.0012; // scroll down = zoom in
      applyZoom(factor, nx, ny);
    };

    // ── Double-click / double-tap reset ─────────────────────────────────────

    const onDblClick = () => {
      viewRef.current = { panX: 0.5, panY: 0.5, zoom: 1.0 };
      rendererRef.current?.setView(0.5, 0.5, 1.0);
    };

    canvas.addEventListener('pointerdown',  onPointerDown, { passive: false });
    canvas.addEventListener('pointermove',  onPointerMove, { passive: true });
    canvas.addEventListener('pointerup',    onPointerUp);
    canvas.addEventListener('pointercancel',onPointerUp);
    canvas.addEventListener('wheel',        onWheel,       { passive: false });
    canvas.addEventListener('dblclick',     onDblClick);

    return () => {
      canvas.removeEventListener('pointerdown',  onPointerDown);
      canvas.removeEventListener('pointermove',  onPointerMove);
      canvas.removeEventListener('pointerup',    onPointerUp);
      canvas.removeEventListener('pointercancel',onPointerUp);
      canvas.removeEventListener('wheel',        onWheel);
      canvas.removeEventListener('dblclick',     onDblClick);
    };
  }, []);

  return (
    <div
      ref={wrapRef}
      className={`flex items-center justify-center ${className}`}
    >
      <canvas
        ref={canvasRef}
        style={{
          aspectRatio: '1 / 1',
          maxWidth: '100%',
          maxHeight: '100%',
          cursor: 'crosshair',
          touchAction: 'none',  // hand ALL touch to pointer events, no browser zoom/scroll
        }}
        title="Scroll to zoom · Drag to pan · Double-click to reset"
      />
    </div>
  );
}
