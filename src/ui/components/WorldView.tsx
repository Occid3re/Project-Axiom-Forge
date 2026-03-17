import { useEffect, useRef } from 'react';
import { WorldRenderer } from '../renderer';
import type { World } from '../../engine';

interface WorldViewProps {
  world: World | null;
  size: number;
}

export function WorldView({ world, size }: WorldViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WorldRenderer | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      rendererRef.current = new WorldRenderer(canvas);
      rendererRef.current.resize(size, size);
    } catch (e) {
      console.error('WebGL init:', e);
    }
    return () => { rendererRef.current?.destroy(); rendererRef.current = null; };
  }, [size]);

  useEffect(() => {
    if (!world || !rendererRef.current) return;
    rendererRef.current.update(world.getVisualState());
  }, [world]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      style={{ imageRendering: 'pixelated', width: size, height: size }}
      className="block"
    />
  );
}
