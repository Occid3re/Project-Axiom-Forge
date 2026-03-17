/**
 * Canvas-based population/diversity/signal chart — dark cinematic style.
 */

import { useEffect, useRef } from 'react';
import type { WorldSnapshot } from '../../engine';

interface PopulationChartProps {
  snapshots: WorldSnapshot[];
  width?: number;
  height?: number;
}

export function PopulationChart({ snapshots, width = 220, height = 90 }: PopulationChartProps) {
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
    ctx.fillStyle = 'rgba(0,0,0,0)';
    ctx.fillRect(0, 0, width, height);

    const n = snapshots.length;
    const maxPop = Math.max(1, ...snapshots.map(s => s.population));
    const maxDiv = Math.max(0.01, ...snapshots.map(s => s.diversity));
    const maxSig = Math.max(0.01, ...snapshots.map(s => s.signalActivity));
    const xStep = width / (n - 1);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 0.5;
    for (let i = 1; i < 4; i++) {
      const y = (i / 4) * height;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    // Fill under population
    const popGrad = ctx.createLinearGradient(0, 0, 0, height);
    popGrad.addColorStop(0, 'rgba(16,185,129,0.15)');
    popGrad.addColorStop(1, 'rgba(16,185,129,0)');
    ctx.fillStyle = popGrad;
    ctx.beginPath();
    ctx.moveTo(0, height);
    for (let i = 0; i < n; i++) {
      const x = i * xStep;
      const y = height - (snapshots[i].population / maxPop) * height;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.lineTo((n - 1) * xStep, height);
    ctx.closePath();
    ctx.fill();

    // Population line
    drawLine(ctx, snapshots, s => s.population / maxPop, n, xStep, height, '#10b981', 1.5);
    // Diversity
    drawLine(ctx, snapshots, s => s.diversity / maxDiv, n, xStep, height, 'rgba(139,92,246,0.7)', 1);
    // Signals
    drawLine(ctx, snapshots, s => s.signalActivity / maxSig, n, xStep, height, 'rgba(6,182,212,0.5)', 0.75);
  }, [snapshots, width, height]);

  if (snapshots.length < 2) {
    return (
      <div
        style={{ width, height }}
        className="flex items-center justify-center rounded"
      >
        <span className="text-[9px] text-gray-800 font-mono tracking-wider">AWAITING DATA</span>
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, display: 'block' }}
      className="rounded"
    />
  );
}

function drawLine(
  ctx: CanvasRenderingContext2D,
  snaps: WorldSnapshot[],
  fn: (s: WorldSnapshot) => number,
  n: number,
  xStep: number,
  h: number,
  color: string,
  lw: number,
) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = i * xStep;
    const y = h - fn(snaps[i]) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}
