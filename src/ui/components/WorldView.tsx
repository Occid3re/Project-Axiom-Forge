/**
 * WebGL canvas component for rendering a single world.
 */

import { useEffect, useRef, useCallback } from 'react';
import { WorldRenderer } from '../renderer';
import type { World } from '../../engine';

interface WorldViewProps {
  world: World | null;
  size?: number;
}

export function WorldView({ world, size = 512 }: WorldViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WorldRenderer | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    try {
      rendererRef.current = new WorldRenderer(canvas);
      rendererRef.current.resize(size, size);
    } catch (e) {
      console.error('WebGL init failed:', e);
    }

    return () => {
      rendererRef.current?.destroy();
      rendererRef.current = null;
    };
  }, [size]);

  useEffect(() => {
    if (!world || !rendererRef.current) return;
    const state = world.getVisualState();
    rendererRef.current.update(state);
  }, [world]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      className="rounded-lg border border-gray-700 shadow-xl"
      style={{ imageRendering: 'pixelated' }}
    />
  );
}
