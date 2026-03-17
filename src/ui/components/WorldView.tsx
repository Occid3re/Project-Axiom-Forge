/**
 * WorldView — WebGL canvas driven by a requestAnimationFrame loop.
 * Accepts a frameRef so socket frames skip React state entirely.
 * Renders at 60fps using whatever data was last received from the server.
 *
 * Interaction:
 *   - Scroll wheel / pinch  → zoom (tiled REPEAT world when zoomed out)
 *   - Click + drag          → pan
 *   - Double-click          → reset to 1:1
 */
import { useEffect, useRef } from 'react';
import { WorldRenderer } from '../renderer';
import type { DecodedFrame } from '../../engine/protocol';

interface WorldViewProps {
  frameRef: React.RefObject<DecodedFrame | null>;
  className?: string;
}

const ZOOM_MIN = 0.1;   // 10× zoom in
const ZOOM_MAX = 1.0;   // 1:1 full world view — can't zoom out past this

export function WorldView({ frameRef, className = '' }: WorldViewProps) {
  const wrapRef     = useRef<HTMLDivElement>(null);
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WorldRenderer | null>(null);

  // View state in a ref — no re-renders needed
  const viewRef  = useRef({ panX: 0.5, panY: 0.5, zoom: 1.0 });
  const dragRef  = useRef<{ x: number; y: number } | null>(null);
  // Pinch-to-zoom state
  const pinchRef = useRef<{ dist: number; midX: number; midY: number } | null>(null);

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

  // RAF render loop — 60 fps, fully decoupled from React state updates
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

  // Zoom/pan interaction
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const applyZoom = (factor: number, canvasNX: number, canvasNY: number) => {
      const v = viewRef.current;
      const newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, v.zoom * factor));
      // Keep the world point under the cursor fixed
      const cx = canvasNX - 0.5;
      const cy = canvasNY - 0.5;
      const worldX = v.panX + cx * v.zoom;
      const worldY = v.panY + cy * v.zoom;
      v.panX = worldX - cx * newZoom;
      v.panY = worldY - cy * newZoom;
      v.zoom = newZoom;
      rendererRef.current?.setView(v.panX, v.panY, v.zoom);
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width;
      const ny = (e.clientY - rect.top)  / rect.height;
      // deltaY > 0 → scroll down → zoom in (factor < 1); up → zoom out
      const factor = 1 - e.deltaY * 0.0012;
      applyZoom(factor, nx, ny);
    };

    const onPointerDown = (e: PointerEvent) => {
      dragRef.current = { x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!dragRef.current) return;
      const rect = canvas.getBoundingClientRect();
      const dx = (e.clientX - dragRef.current.x) / rect.width;
      const dy = (e.clientY - dragRef.current.y) / rect.height;
      dragRef.current = { x: e.clientX, y: e.clientY };
      const v = viewRef.current;
      v.panX -= dx * v.zoom;
      v.panY -= dy * v.zoom;
      rendererRef.current?.setView(v.panX, v.panY, v.zoom);
    };

    const onPointerUp = () => { dragRef.current = null; };

    const onDblClick = () => {
      viewRef.current = { panX: 0.5, panY: 0.5, zoom: 1.0 };
      rendererRef.current?.setView(0.5, 0.5, 1.0);
    };

    const onTouchStart = (e: TouchEvent) => {
      e.preventDefault(); // block browser pinch-zoom fighting our handler
      if (e.touches.length === 2) {
        const t0 = e.touches[0], t1 = e.touches[1];
        const dist = Math.hypot(t1.clientX - t0.clientX, t1.clientY - t0.clientY);
        pinchRef.current = {
          dist,
          midX: (t0.clientX + t1.clientX) / 2,
          midY: (t0.clientY + t1.clientY) / 2,
        };
      }
    };

    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2 && pinchRef.current) {
        e.preventDefault();
        const t0 = e.touches[0], t1 = e.touches[1];
        const dist = Math.hypot(t1.clientX - t0.clientX, t1.clientY - t0.clientY);
        const rect = canvas.getBoundingClientRect();
        const midX = (t0.clientX + t1.clientX) / 2;
        const midY = (t0.clientY + t1.clientY) / 2;
        const factor = dist / pinchRef.current.dist; // spread fingers → zoom in
        applyZoom(factor, (midX - rect.left) / rect.width, (midY - rect.top) / rect.height);
        pinchRef.current = { dist, midX, midY };
      }
    };

    const onTouchEnd = () => { pinchRef.current = null; };

    canvas.addEventListener('wheel',       onWheel,       { passive: false });
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup',   onPointerUp);
    canvas.addEventListener('dblclick',    onDblClick);
    canvas.addEventListener('touchstart',  onTouchStart,  { passive: false });
    canvas.addEventListener('touchmove',   onTouchMove,   { passive: false });
    canvas.addEventListener('touchend',    onTouchEnd);

    return () => {
      canvas.removeEventListener('wheel',       onWheel);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup',   onPointerUp);
      canvas.removeEventListener('dblclick',    onDblClick);
      canvas.removeEventListener('touchstart',  onTouchStart);
      canvas.removeEventListener('touchmove',   onTouchMove);
      canvas.removeEventListener('touchend',    onTouchEnd);
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
          touchAction: 'none',   // hand all touch events to our handlers
        }}
        title="Scroll to zoom · Drag to pan · Double-click to reset"
      />
    </div>
  );
}
