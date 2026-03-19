import { useEffect, useRef } from 'react';

const N_IN = 16;
const N_HID = 10;
const N_OUT = 11;
const W1_SZ = N_IN * N_HID;
const TOTAL_CONNS = W1_SZ + N_HID * N_OUT;

const INPUT_LABELS = [
  'Resource', 'Energy', 'Glyph', 'Signal',
  'Res↑', 'Res→', 'Res↓', 'Res←',
  'Ent↑', 'Ent→', 'Ent↓', 'Ent←',
  'Comm↑', 'Comm→', 'Comm↓', 'Comm←',
];

const OUTPUT_LABELS = ['Idle', '↑', '→', '↓', '←', 'Eat', 'Breed', 'Signal', 'Attack', 'Deposit', 'Absorb'];

const INPUT_COLORS = [
  '#6ee7b7', '#fbbf24', '#f59e0b', '#93c5fd',
  '#67e8f9', '#38bdf8', '#22d3ee', '#0ea5e9',
  '#c4b5fd', '#a78bfa', '#8b5cf6', '#7c3aed',
  '#fdba74', '#fb923c', '#f97316', '#ea580c',
];

const OUTPUT_COLORS = [
  '#6b7280',
  '#22d3ee', '#38bdf8', '#06b6d4', '#0ea5e9',
  '#10b981', '#ec4899', '#8b5cf6', '#ef4444', '#d97706', '#0ea5e9',
];

const POS_COLOR = '#7dd3fc';
const NEG_COLOR = '#f97316';

interface SampleNetwork {
  entityId: number;
  action: number;
  age: number;
  energy: number;
  size: number;
  kinNeighbors: number;
  threatNeighbors: number;
  lockTicksRemaining: number;
  inputs: number[];
  hidden: number[];
  probs: number[];
  genome: number[];
}

function makeParticleState(): Float32Array {
  const particles = new Float32Array(TOTAL_CONNS * 2);
  for (let i = 0; i < particles.length; i++) particles[i] = Math.random();
  return particles;
}

function withAlpha(color: string, hexAlpha: string): string {
  if (color.startsWith('#')) return color + hexAlpha;
  const a = (parseInt(hexAlpha, 16) / 255).toFixed(2);
  return color.replace(/^hsl\(/, 'hsla(').replace(/\)$/, `, ${a})`);
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function formatAction(action: number): string {
  return OUTPUT_LABELS[action] ?? 'Unknown';
}

function render(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  sample: SampleNetwork,
  particles: Float32Array,
  ms: number,
): void {
  ctx.fillStyle = '#02040a';
  ctx.fillRect(0, 0, W, H);

  ctx.fillStyle = 'rgba(125,211,252,0.02)';
  const gs = Math.max(14, Math.round(W * 0.025));
  for (let x = gs; x < W; x += gs) {
    for (let y = gs; y < H; y += gs) {
      ctx.beginPath();
      ctx.arc(x, y, 0.8, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const scanY = (ms * 0.015) % (H + 120) - 60;
  const scan = ctx.createLinearGradient(0, scanY - 60, 0, scanY + 60);
  scan.addColorStop(0, 'transparent');
  scan.addColorStop(0.5, 'rgba(125,211,252,0.02)');
  scan.addColorStop(1, 'transparent');
  ctx.fillStyle = scan;
  ctx.fillRect(0, Math.max(0, scanY - 60), W, 120);

  const R = Math.max(7, Math.min(W, H) * 0.022);
  const padL = W * 0.02 + R * 4.6;
  const padR = R * 12.5;
  const layerX = [padL, (padL + W - padR) / 2, W - padR];
  const margY = H * 0.13;
  const usable = H - margY * 2;

  const nodeY = (n: number) =>
    n === 1 ? [H / 2] : Array.from({ length: n }, (_, i) => margY + (i / (n - 1)) * usable);

  const inY = nodeY(N_IN);
  const hidY = nodeY(N_HID);
  const outY = nodeY(N_OUT);

  let maxW = 0.001;
  for (const w of sample.genome) {
    const abs = Math.abs(w);
    if (abs > maxW) maxW = abs;
  }

  const bestAction = sample.probs.indexOf(Math.max(...sample.probs));
  let ci = 0;

  const drawConn = (
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    weight: number,
    activity: number,
    particleIdx: number,
  ) => {
    const norm = Math.abs(weight) / maxW;
    const emphasis = clamp01(activity);
    const alpha = 0.025 + norm * 0.12 + emphasis * 0.52;
    const color = weight >= 0 ? POS_COLOR : NEG_COLOR;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.shadowBlur = R * (1.5 + emphasis * 2.2);
    ctx.shadowColor = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 0.25 + norm * 0.5 + emphasis * 1.8;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.restore();

    if (emphasis < 0.14) return;

    const speed = 0.002 + emphasis * 0.018;
    for (let p = 0; p < 2; p++) {
      const slot = particleIdx * 2 + p;
      const t = (particles[slot] + speed) % 1;
      particles[slot] = t;
      const px = x1 + (x2 - x1) * t;
      const py = y1 + (y2 - y1) * t;

      ctx.save();
      ctx.globalAlpha = 0.12 + emphasis * 0.8;
      ctx.shadowBlur = R * 4;
      ctx.shadowColor = color;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(px, py, R * (0.11 + emphasis * 0.16), 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  };

  for (let k = 0; k < N_IN; k++) {
    const inputStrength = sample.inputs[k];
    for (let h = 0; h < N_HID; h++, ci++) {
      const hiddenStrength = clamp01((sample.hidden[h] + 1) * 0.5);
      const weight = sample.genome[k * N_HID + h];
      const activity = inputStrength * hiddenStrength * (Math.abs(weight) / maxW) * 1.35;
      drawConn(layerX[0], inY[k], layerX[1], hidY[h], weight, activity, ci);
    }
  }

  for (let h = 0; h < N_HID; h++) {
    const hiddenStrength = clamp01((sample.hidden[h] + 1) * 0.5);
    for (let a = 0; a < N_OUT; a++, ci++) {
      const weight = sample.genome[W1_SZ + h * N_OUT + a];
      const activity = hiddenStrength * sample.probs[a] * (Math.abs(weight) / maxW) * 1.8;
      drawConn(layerX[1], hidY[h], layerX[2], outY[a], weight, activity, ci);
    }
  }

  const drawNode = (
    x: number,
    y: number,
    normalized: number,
    baseColor: string,
    label?: string,
    labelLeft?: boolean,
    isWinner?: boolean,
  ) => {
    const act = clamp01(normalized);

    ctx.save();
    ctx.globalAlpha = 0.07 + act * 0.28 + (isWinner ? 0.16 : 0);
    ctx.shadowBlur = R * (3 + act * 4 + (isWinner ? 2 : 0));
    ctx.shadowColor = baseColor;
    ctx.beginPath();
    ctx.arc(x, y, R * (1.4 + act * 0.8), 0, Math.PI * 2);
    ctx.fillStyle = baseColor;
    ctx.fill();
    ctx.restore();

    ctx.save();
    const grad = ctx.createRadialGradient(x - R * 0.3, y - R * 0.3, R * 0.1, x, y, R);
    grad.addColorStop(0, `rgba(255,255,255,${0.10 + act * 0.45})`);
    grad.addColorStop(1, withAlpha(baseColor, '55'));
    ctx.beginPath();
    ctx.arc(x, y, R, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = withAlpha(baseColor, isWinner ? 'ff' : '88');
    ctx.lineWidth = isWinner ? 2 : 1;
    ctx.globalAlpha = 0.55 + act * 0.45;
    ctx.stroke();
    ctx.restore();

    if (label) {
      ctx.save();
      ctx.font = `${Math.max(9, R * 0.85)}px monospace`;
      ctx.fillStyle = withAlpha(baseColor, act > 0.45 ? 'f0' : '88');
      ctx.globalAlpha = 0.8;
      ctx.textAlign = labelLeft ? 'right' : 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x + (labelLeft ? -R * 1.8 : R * 1.8), y);
      ctx.restore();
    }
  };

  inY.forEach((y, k) => drawNode(layerX[0], y, sample.inputs[k], INPUT_COLORS[k], INPUT_LABELS[k], true));

  hidY.forEach((y, h) => {
    const hue = 192 + h * 16;
    drawNode(layerX[1], y, clamp01((sample.hidden[h] + 1) * 0.5), `hsl(${hue}, 72%, 66%)`);
  });

  outY.forEach((y, a) => {
    const isWinner = a === bestAction;
    drawNode(layerX[2], y, sample.probs[a], OUTPUT_COLORS[a], OUTPUT_LABELS[a], false, isWinner);

    const barW = Math.min(W * 0.075, R * 4.3);
    const barH = R * 0.55;
    const barX = layerX[2] + R * 1.9 + OUTPUT_LABELS[a].length * R * 0.56;

    ctx.save();
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = withAlpha(OUTPUT_COLORS[a], '22');
    ctx.fillRect(barX, y - barH / 2, barW, barH);
    ctx.fillStyle = OUTPUT_COLORS[a];
    ctx.shadowBlur = isWinner ? 8 : 0;
    ctx.shadowColor = OUTPUT_COLORS[a];
    ctx.fillRect(barX, y - barH / 2, barW * sample.probs[a], barH);
    ctx.font = `${isWinner ? 'bold ' : ''}${Math.max(7, R * 0.7)}px monospace`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = isWinner ? OUTPUT_COLORS[a] : 'rgba(255,255,255,0.4)';
    ctx.globalAlpha = isWinner ? 0.92 : 0.55;
    ctx.fillText(`${(sample.probs[a] * 100).toFixed(0)}%`, barX + barW + R * 0.45, y);
    ctx.restore();
  });

  const headY = margY * 0.45;
  ctx.save();
  ctx.font = `${Math.max(8, R * 0.82)}px monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.globalAlpha = 0.4;
  ctx.fillStyle = '#9ca3af';
  ctx.fillText('LIVE INPUTS', layerX[0], headY);
  ctx.fillStyle = '#7dd3fc';
  ctx.fillText('CURRENT HIDDEN', layerX[1], headY);
  ctx.fillStyle = OUTPUT_COLORS[bestAction];
  ctx.fillText('ACTION PRESSURE', layerX[2], headY);
  ctx.restore();

  const panelX = W - Math.max(190, W * 0.21);
  const panelY = Math.max(18, H * 0.08);
  const panelW = Math.max(170, W * 0.17);
  const panelH = Math.max(120, H * 0.18);
  ctx.save();
  ctx.fillStyle = 'rgba(3, 7, 13, 0.78)';
  ctx.strokeStyle = 'rgba(125,211,252,0.16)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(panelX, panelY, panelW, panelH, 12);
  ctx.fill();
  ctx.stroke();

  ctx.font = `${Math.max(9, R * 0.78)}px monospace`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#7dd3fc';
  ctx.fillText(`Specimen #${sample.entityId}`, panelX + 12, panelY + 10);

  const rows = [
    `action   ${formatAction(sample.action)} (${(sample.probs[bestAction] * 100).toFixed(0)}%)`,
    `energy   ${sample.energy.toFixed(2)}   size ${sample.size.toFixed(2)}`,
    `age      ${sample.age} ticks`,
    `social   kin ${sample.kinNeighbors} / threat ${sample.threatNeighbors}`,
    `lock     ${Math.ceil(sample.lockTicksRemaining / 30)}s`,
  ];
  ctx.fillStyle = 'rgba(230, 238, 245, 0.76)';
  rows.forEach((row, idx) => {
    ctx.fillText(row, panelX + 12, panelY + 34 + idx * 16);
  });
  ctx.restore();

  ctx.save();
  ctx.font = `${Math.max(7, R * 0.7)}px monospace`;
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
  ctx.fillText(
    'weights are stable while the lock holds · brighter paths are the currently active ones',
    W / 2,
    H - margY * 0.45,
  );
  ctx.restore();
}

interface Props {
  sample: SampleNetwork | null;
}

export function NeuralNetView({ sample }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Float32Array>(makeParticleState());
  const prevEntityIdRef = useRef<number>(-1);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    const sync = () => {
      const dpr = Math.min(devicePixelRatio, 2);
      canvas.width = Math.round(wrap.offsetWidth * dpr);
      canvas.height = Math.round(wrap.offsetHeight * dpr);
    };

    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let rafId: number;

    const loop = (ms: number) => {
      const W = canvas.width;
      const H = canvas.height;
      if (W < 4 || H < 4) {
        rafId = requestAnimationFrame(loop);
        return;
      }

      if (sample && sample.genome.length >= 270) {
        if (sample.entityId !== prevEntityIdRef.current) {
          prevEntityIdRef.current = sample.entityId;
          particlesRef.current = makeParticleState();
        }
        render(ctx, W, H, sample, particlesRef.current, ms);
      } else {
        ctx.fillStyle = '#02040a';
        ctx.fillRect(0, 0, W, H);
        ctx.font = `${Math.min(W, H) * 0.035}px monospace`;
        ctx.fillStyle = 'rgba(125,211,252,0.16)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Awaiting locked specimen data…', W / 2, H / 2);
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [sample]);

  return (
    <div ref={wrapRef} style={{ position: 'absolute', inset: 0 }}>
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }} />
    </div>
  );
}
