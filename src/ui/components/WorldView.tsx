import { useEffect, useRef } from 'react';
import { WorldRenderer } from '../renderer';
import type { DecodedFrame } from '../../engine/protocol';

interface WorldViewProps {
  frame: DecodedFrame | null;
  className?: string;
}

export function WorldView({ frame, className = '' }: WorldViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WorldRenderer | null>(null);

  // Init renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      rendererRef.current = new WorldRenderer(canvas);
    } catch (e) {
      console.error('WebGL init:', e);
    }
    return () => { rendererRef.current?.destroy(); rendererRef.current = null; };
  }, []);

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(entries => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      const size = Math.min(width, height);
      rendererRef.current?.resize(size * devicePixelRatio, size * devicePixelRatio);
    });
    ro.observe(canvas.parentElement!);
    return () => ro.disconnect();
  }, []);

  // Render each new frame
  useEffect(() => {
    if (!frame || !rendererRef.current) return;
    rendererRef.current.updateFromFrame(frame);
  }, [frame]);

  return (
    <canvas
      ref={canvasRef}
      className={`block w-full h-full ${className}`}
      style={{ imageRendering: 'pixelated' }}
    />
  );
}
