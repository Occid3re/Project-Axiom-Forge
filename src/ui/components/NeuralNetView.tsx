/**
 * NeuralNetView — animated X-ray visualisation of the current best entity's neural network.
 *
 * Renders the 80-weight MLP:  4 inputs → 8 hidden (tanh) → 6 action logits
 *
 * Aesthetics: medical scanner / circuit board —
 *   • Dark near-black background with slow scan-line sweep
 *   • Connection lines glow cyan (positive weight) or amber (negative weight),
 *     brightness proportional to |weight|
 *   • Particles flow along each connection at speed ∝ |weight|
 *   • Node circles glow based on actual activation for a sample environment state
 *   • Output nodes show softmax probability bars (what this network prefers to do)
 */

import { useEffect, useRef } from 'react';

// ── Network constants (must match src/engine/constants.ts) ──────────────────
const N_IN  = 4;
const N_HID = 8;
const N_OUT = 6;
const W1_SZ = N_IN * N_HID;  // 32 — genome[input * 8 + hidden]
// W2: genome[32 + hidden * 6 + action]

const INPUT_LABELS  = ['Resource', 'Energy', 'Density', 'Signal'];
const OUTPUT_LABELS = ['Idle', 'Move', 'Eat', 'Breed', 'Signal', 'Attack'];
const INPUT_COLORS  = ['#10b981', '#f59e0b', '#a78bfa', '#06b6d4'];
const OUTPUT_COLORS = ['#6b7280', '#06b6d4', '#10b981', '#ec4899', '#8b5cf6', '#ef4444'];

const POS_COLOR = '#00e5ff';  // cyan  — positive weights
const NEG_COLOR = '#ff6600';  // amber — negative weights

// Canonical sample inputs for computing "typical" activations
const SAMPLE = [0.4, 0.5, 0.25, 0.1]; // [resource, energy, density, signal]

// ── Particle state (2 per connection × 80 connections = 160) ────────────────
const TOTAL_CONNS = W1_SZ + N_HID * N_OUT; // 80

function makeParticleState(): Float32Array {
  const t = new Float32Array(TOTAL_CONNS * 2);
  for (let i = 0; i < t.length; i++) t[i] = Math.random();
  return t;
}

// ── Forward pass ─────────────────────────────────────────────────────────────

function forwardPass(g: number[]): { hidden: number[]; probs: number[] } {
  const h = new Array<number>(N_HID);
  for (let j = 0; j < N_HID; j++) {
    let s = 0;
    for (let k = 0; k < N_IN; k++) s += g[k * N_HID + j] * SAMPLE[k];
    h[j] = Math.tanh(s);
  }
  const logits = new Array<number>(N_OUT).fill(0);
  for (let a = 0; a < N_OUT; a++)
    for (let j = 0; j < N_HID; j++) logits[a] += g[W1_SZ + j * N_OUT + a] * h[j];
  const maxL = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxL));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return { hidden: h, probs: exps.map(e => e / sum) };
}

// ── Main render function ──────────────────────────────────────────────────────

function render(
  ctx: CanvasRenderingContext2D,
  W: number, H: number,
  genome: number[],
  pts: Float32Array,
  ms: number,
): void {
  // Background
  ctx.fillStyle = '#02040a';
  ctx.fillRect(0, 0, W, H);

  // Faint dot grid
  ctx.fillStyle = 'rgba(0,229,255,0.025)';
  const gs = Math.round(W * 0.025);
  for (let x = gs; x < W; x += gs)
    for (let y = gs; y < H; y += gs) {
      ctx.beginPath();
      ctx.arc(x, y, 0.8, 0, Math.PI * 2);
      ctx.fill();
    }

  // Slow scan line
  const scanY = (ms * 0.018) % (H + 80) - 40;
  const scan  = ctx.createLinearGradient(0, scanY - 40, 0, scanY + 40);
  scan.addColorStop(0,   'transparent');
  scan.addColorStop(0.5, 'rgba(0,229,255,0.018)');
  scan.addColorStop(1,   'transparent');
  ctx.fillStyle = scan;
  ctx.fillRect(0, Math.max(0, scanY - 40), W, 80);

  // Node layout
  const R      = Math.max(7, Math.min(W, H) * 0.022); // node radius
  const padL   = W  * 0.02 + R * 4;                   // left padding  — room for input labels
  const padR   = R  * 11;                              // right padding — room for output labels + prob bars + %
  const LX     = [padL, (padL + W - padR) / 2, W - padR]; // hidden centred in available span
  const margY  = H  * 0.13;
  const usable = H  - margY * 2;

  const nodeY = (n: number) =>
    n === 1 ? [H / 2] :
    Array.from({ length: n }, (_, i) => margY + (i / (n - 1)) * usable);

  const inY  = nodeY(N_IN);
  const hidY = nodeY(N_HID);
  const outY = nodeY(N_OUT);

  // Compute activations
  let maxW = 0.001;
  for (const w of genome) if (Math.abs(w) > maxW) maxW = Math.abs(w);
  const { hidden: hidAct, probs } = forwardPass(genome);
  const bestAction = probs.indexOf(Math.max(...probs));

  // ── Connections + particles ─────────────────────────────────────────────
  let ci = 0;

  const drawConn = (
    x1: number, y1: number, x2: number, y2: number,
    w: number, pIdx: number,
  ) => {
    const absW  = Math.abs(w);
    const norm  = absW / maxW;
    const color = w >= 0 ? POS_COLOR : NEG_COLOR;
    const speed = norm * 0.013 + 0.003;

    // Glowing line
    ctx.save();
    ctx.globalAlpha = norm * 0.55 + 0.04;
    ctx.shadowBlur  = R * 2.5;
    ctx.shadowColor = color;
    ctx.strokeStyle = color;
    ctx.lineWidth   = norm * 2 + 0.4;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.restore();

    // Advance + draw 2 particles
    for (let p = 0; p < 2; p++) {
      const slot = pIdx * 2 + p;
      const t    = (pts[slot] + speed) % 1;
      pts[slot]  = t;

      const px = x1 + (x2 - x1) * t;
      const py = y1 + (y2 - y1) * t;

      ctx.save();
      ctx.globalAlpha = norm * 0.9 + 0.05;
      ctx.shadowBlur  = R * 5;
      ctx.shadowColor = color;
      ctx.fillStyle   = '#ffffff';
      ctx.beginPath();
      ctx.arc(px, py, R * 0.28, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  };

  // W1 connections
  for (let k = 0; k < N_IN; k++)
    for (let h = 0; h < N_HID; h++, ci++)
      drawConn(LX[0], inY[k], LX[1], hidY[h], genome[k * N_HID + h], ci);

  // W2 connections
  for (let h = 0; h < N_HID; h++)
    for (let a = 0; a < N_OUT; a++, ci++)
      drawConn(LX[1], hidY[h], LX[2], outY[a], genome[W1_SZ + h * N_OUT + a], ci);

  // ── Nodes ───────────────────────────────────────────────────────────────

  const drawNode = (
    x: number, y: number,
    activation: number,   // [-1, 1] or [0, 1]
    baseColor: string,
    label?: string,
    labelLeft?: boolean,
    isWinner?: boolean,
  ) => {
    const act = Math.max(0, Math.min(1, (activation + 1) / 2)); // map to [0,1]

    // Outer glow halo
    ctx.save();
    ctx.globalAlpha = 0.08 + act * 0.35;
    if (isWinner) ctx.globalAlpha += 0.25;
    ctx.shadowBlur  = R * (isWinner ? 10 : 6);
    ctx.shadowColor = baseColor;
    ctx.beginPath();
    ctx.arc(x, y, R * 2, 0, Math.PI * 2);
    ctx.fillStyle = baseColor;
    ctx.fill();
    ctx.restore();

    // Inner filled circle
    ctx.save();
    const grad = ctx.createRadialGradient(x - R * 0.3, y - R * 0.3, R * 0.1, x, y, R);
    grad.addColorStop(0, `rgba(255,255,255,${0.08 + act * 0.55})`);
    grad.addColorStop(1, baseColor + '44');
    ctx.shadowBlur  = R * 3;
    ctx.shadowColor = baseColor;
    ctx.beginPath();
    ctx.arc(x, y, R, 0, Math.PI * 2);
    ctx.fillStyle   = grad;
    ctx.fill();
    ctx.strokeStyle = baseColor + (isWinner ? 'ff' : '80');
    ctx.lineWidth   = isWinner ? 2 : 1;
    ctx.globalAlpha = 0.6 + act * 0.4;
    ctx.stroke();
    ctx.restore();

    // Label
    if (label) {
      ctx.save();
      const fontSize = Math.max(9, R * 0.9);
      ctx.font        = `${fontSize}px monospace`;
      ctx.fillStyle   = baseColor + (act > 0.5 ? 'ee' : '88');
      ctx.globalAlpha = 0.75;
      ctx.textAlign   = labelLeft ? 'right' : 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x + (labelLeft ? -R * 1.7 : R * 1.7), y);
      ctx.restore();
    }
  };

  // Input nodes
  inY.forEach((y, k) =>
    drawNode(LX[0], y, SAMPLE[k] * 2 - 1, INPUT_COLORS[k], INPUT_LABELS[k], true),
  );

  // Hidden nodes — hue cycles through blue→purple
  hidY.forEach((y, h) => {
    const hue   = 190 + h * 18;
    const color = `hsl(${hue},80%,65%)`;
    drawNode(LX[1], y, hidAct[h], color);
  });

  // Output nodes + probability bars
  outY.forEach((y, a) => {
    const isWinner = a === bestAction;
    drawNode(LX[2], y, probs[a] * 2 - 1, OUTPUT_COLORS[a], OUTPUT_LABELS[a], false, isWinner);

    // Probability bar
    const barW   = Math.min(W * 0.06, R * 3.5);
    const barH   = R * 0.55;
    const barX   = LX[2] + R * 1.7 + (OUTPUT_LABELS[a].length * R * 0.55);
    ctx.save();
    ctx.globalAlpha = 0.55;
    ctx.fillStyle   = OUTPUT_COLORS[a] + '1a';
    ctx.fillRect(barX, y - barH / 2, barW, barH);
    ctx.fillStyle   = OUTPUT_COLORS[a];
    ctx.shadowBlur  = isWinner ? 8 : 0;
    ctx.shadowColor = OUTPUT_COLORS[a];
    ctx.fillRect(barX, y - barH / 2, barW * probs[a], barH);
    if (isWinner) {
      ctx.globalAlpha = 0.9;
      ctx.font        = `bold ${Math.max(7, R * 0.7)}px monospace`;
      ctx.textAlign   = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillStyle   = OUTPUT_COLORS[a];
      ctx.fillText(`${(probs[a] * 100).toFixed(0)}%`, barX + barW + R * 0.4, y);
    }
    ctx.restore();
  });

  // ── Layer headers ───────────────────────────────────────────────────────
  const headY  = margY * 0.45;
  const headSz = Math.max(8, R * 0.85);
  ctx.save();
  ctx.font        = `${headSz}px monospace`;
  ctx.textAlign   = 'center';
  ctx.textBaseline = 'middle';
  ctx.globalAlpha = 0.35;

  ctx.fillStyle = INPUT_COLORS[0];
  ctx.fillText('SENSORY INPUT', LX[0], headY);

  ctx.fillStyle = '#7dd3fc';
  ctx.fillText('HIDDEN ×8', LX[1], headY);

  ctx.fillStyle = OUTPUT_COLORS[bestAction];
  ctx.shadowBlur  = 6;
  ctx.shadowColor = OUTPUT_COLORS[bestAction];
  ctx.fillText('ACTION OUTPUT', LX[2], headY);
  ctx.restore();

  // ── Bottom hint ──────────────────────────────────────────────────────────
  ctx.save();
  ctx.font        = `${Math.max(7, R * 0.7)}px monospace`;
  ctx.textAlign   = 'center';
  ctx.fillStyle   = 'rgba(0,229,255,0.18)';
  ctx.globalAlpha = 0.6;
  ctx.fillText(
    `cyan = positive weight  ·  amber = negative weight  ·  sample: resource=40% energy=50%`,
    W / 2, H - margY * 0.45,
  );
  ctx.restore();
}

// ── Component ────────────────────────────────────────────────────────────────

interface Props {
  genome: number[] | null;
}

export function NeuralNetView({ genome }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ptsRef    = useRef<Float32Array>(makeParticleState());
  const prevKey   = useRef<number>(0);

  // RAF loop — also handles canvas sizing so there is no ResizeObserver race
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let rafId: number;

    const loop = (ms: number) => {
      // Keep physical canvas pixels in sync with CSS display size
      const dpr  = Math.min(devicePixelRatio, 2);
      const pxW  = Math.round(canvas.clientWidth  * dpr);
      const pxH  = Math.round(canvas.clientHeight * dpr);
      if (canvas.width !== pxW || canvas.height !== pxH) {
        canvas.width  = pxW;
        canvas.height = pxH;
      }

      const W = canvas.width;
      const H = canvas.height;

      if (genome && genome.length >= 80) {
        const key = genome[0] + genome[40] + genome[79];
        if (key !== prevKey.current) {
          prevKey.current = key;
          ptsRef.current  = makeParticleState();
        }
        render(ctx, W, H, genome, ptsRef.current, ms);
      } else {
        ctx.fillStyle = '#02040a';
        ctx.fillRect(0, 0, W, H);
        if (W > 0 && H > 0) {
          ctx.font         = `${Math.min(W, H) * 0.035}px monospace`;
          ctx.fillStyle    = 'rgba(0,229,255,0.15)';
          ctx.textAlign    = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('Awaiting neural network data…', W / 2, H / 2);
        }
      }
      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [genome]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  );
}
