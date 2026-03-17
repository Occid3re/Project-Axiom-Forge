/**
 * Simple canvas-based population/diversity chart.
 * No chart library dependency — pure Canvas2D for performance.
 */

import { useEffect, useRef } from 'react';
import type { WorldSnapshot } from '../../engine';

interface PopulationChartProps {
  snapshots: WorldSnapshot[];
  width?: number;
  height?: number;
}

export function PopulationChart({ snapshots, width = 320, height = 120 }: PopulationChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || snapshots.length < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = '#0f1115';
    ctx.fillRect(0, 0, width, height);

    const n = snapshots.length;
    const maxPop = Math.max(1, ...snapshots.map(s => s.population));
    const maxDiv = Math.max(0.01, ...snapshots.map(s => s.diversity));

    const xStep = width / (n - 1);

    // Grid lines
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 4; i++) {
      const y = (i / 4) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Population line
    drawLine(ctx, snapshots, s => s.population / maxPop, n, xStep, height, '#10b981', 1.5);

    // Diversity line
    drawLine(ctx, snapshots, s => s.diversity / maxDiv, n, xStep, height, '#8b5cf6', 1);

    // Signal activity (normalized)
    const maxSig = Math.max(0.01, ...snapshots.map(s => s.signalActivity));
    drawLine(ctx, snapshots, s => s.signalActivity / maxSig, n, xStep, height, '#06b6d4', 0.8);

    // Legend
    ctx.font = '9px monospace';
    const legends = [
      { label: 'Population', color: '#10b981' },
      { label: 'Diversity', color: '#8b5cf6' },
      { label: 'Signals', color: '#06b6d4' },
    ];
    let lx = 4;
    for (const { label, color } of legends) {
      ctx.fillStyle = color;
      ctx.fillRect(lx, 4, 8, 8);
      ctx.fillStyle = '#9ca3af';
      ctx.fillText(label, lx + 11, 11);
      lx += ctx.measureText(label).width + 18;
    }
  }, [snapshots, width, height]);

  return (
    <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg border border-gray-700/50 p-3">
      <h4 className="text-xs font-semibold text-gray-300 mb-2">World Timeline</h4>
      <canvas
        ref={canvasRef}
        style={{ width, height }}
        className="rounded"
      />
    </div>
  );
}

function drawLine(
  ctx: CanvasRenderingContext2D,
  snapshots: WorldSnapshot[],
  accessor: (s: WorldSnapshot) => number,
  n: number,
  xStep: number,
  height: number,
  color: string,
  lineWidth: number,
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = i * xStep;
    const y = height - accessor(snapshots[i]) * height;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
