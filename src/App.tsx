/**
 * LostUplink: Axiom Forge — client
 * Pure viewer. Connects to the server, renders the shared simulation.
 * No local computation. All state comes from the server over WebSocket.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { io, type Socket } from 'socket.io-client';
import { decodeFrame, type DecodedFrame, type ServerMeta } from './engine/protocol';
import { WorldView } from './ui/components/WorldView';
import { EmergenceLadder, detectEmergence, type EmergenceState } from './ui/components/EmergenceLadder';
import { TransmissionLog } from './ui/components/TransmissionLog';
import { PopulationChart } from './ui/components/PopulationChart';

// ---- Types -----------------------------------------------------------------

interface Snapshot { population: number; diversity?: number; signalActivity?: number; }

// ---- App -------------------------------------------------------------------

export default function App() {
  const [frame, setFrame]         = useState<DecodedFrame | null>(null);
  const [meta, setMeta]           = useState<ServerMeta | null>(null);
  const [connected, setConnected] = useState(false);
  const [log, setLog]             = useState<string[]>(['Connecting to Axiom Forge...']);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [emergence, setEmergence] = useState<EmergenceState>({ stage: -1, progress: new Array(8).fill(0) });

  const socketRef = useRef<Socket | null>(null);
  const addLog = useCallback((msg: string) => setLog(p => [...p.slice(-200), msg]), []);

  // Detect emergence from latest scores
  useEffect(() => {
    if (!meta?.scores) return;
    const scores = meta.scores;
    const genResults = (meta.generations ?? []).map(g => ({
      generation: g.gen, worlds: [], bestScore: g.best, bestLawsId: '', avgScore: g.avg,
    }));
    setEmergence(detectEmergence(scores, genResults));
  }, [meta?.scores]);

  // Accumulate snapshots for chart
  useEffect(() => {
    if (!meta) return;
    setSnapshots(p => {
      const next = [...p, { population: meta.population }].slice(-300);
      return next;
    });
  }, [meta?.tick]);

  // Socket connection
  useEffect(() => {
    const socket = io({ transports: ['websocket'], reconnectionDelay: 2000 });
    socketRef.current = socket;

    socket.on('connect',    () => { setConnected(true);  addLog('Connection established — observing simulation'); });
    socket.on('disconnect', () => { setConnected(false); addLog('Connection lost — reconnecting...'); });

    socket.on('frame', (buf: ArrayBuffer) => {
      const decoded = decodeFrame(buf);
      if (decoded) setFrame(decoded);
    });

    socket.on('meta', (m: ServerMeta) => {
      setMeta(m);
      if (m.logEntry) addLog(m.logEntry);
    });

    return () => { socket.disconnect(); };
  }, [addLog]);

  const pop   = meta?.population ?? 0;
  const gen   = meta?.generation ?? 0;
  const wIdx  = meta?.worldIndex ?? 0;
  const wTot  = meta?.totalWorlds ?? 0;
  const best  = meta?.bestScore ?? 0;
  const tick  = meta?.tick ?? 0;
  const scores = meta?.scores ?? null;

  return (
    <div className="h-[100dvh] w-screen bg-[#070809] overflow-hidden flex flex-col select-none">

      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="shrink-0 flex items-center justify-between px-3 sm:px-5 py-2.5 border-b border-white/[0.04]">
        {/* Brand */}
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 sm:w-7 sm:h-7 rounded-md bg-gradient-to-br from-cyan-500/20 to-purple-600/20 border border-cyan-500/20 flex items-center justify-center text-[9px] font-black text-cyan-400">
            LU
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-sm font-bold tracking-wide text-gray-100">LostUplink</span>
            <span className="hidden sm:inline text-xs text-gray-600 font-light">Axiom Forge</span>
          </div>
        </div>

        {/* Stage pill — center, hidden on xs */}
        {emergence.stage >= 0 && (
          <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-medium bg-white/[0.03] border border-white/[0.05]">
            <span className="text-gray-500">Stage:</span>
            <span className="font-semibold text-cyan-400">
              {['Replicators','Communication','External Memory','Tool Use','Abstraction','Civilization','World Engineering','Recursive Threshold'][emergence.stage]}
            </span>
          </div>
        )}

        {/* Status */}
        <div className="flex items-center gap-2">
          <span className="relative flex h-1.5 w-1.5">
            {connected && <span className="animate-ping absolute inset-0 rounded-full bg-emerald-400 opacity-75" />}
            <span className={`relative inline-flex rounded-full h-1.5 w-1.5 ${connected ? 'bg-emerald-500' : 'bg-red-600'}`} />
          </span>
          <span className="text-[9px] text-gray-600 uppercase tracking-widest hidden sm:inline">
            {connected ? 'Live' : 'Offline'}
          </span>
        </div>
      </header>

      {/* ── Body ───────────────────────────────────────────────────── */}
      <div className="flex-1 flex min-h-0 overflow-hidden">

        {/* Emergence Ladder — hidden on mobile, narrow on tablet */}
        <aside className="hidden md:flex w-44 lg:w-52 shrink-0 border-r border-white/[0.03] bg-black/30">
          <EmergenceLadder emergence={emergence} tick={tick} />
        </aside>

        {/* ── Center column ─────────────────────────────────────── */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0">

          {/* World canvas — fills available space */}
          <div className="flex-1 relative bg-[#070809] min-h-0">
            <WorldView frame={frame} className="absolute inset-0" />

            {/* Overlay: population counter */}
            {pop > 0 && (
              <div className="absolute top-2 right-3 text-right pointer-events-none">
                <div className="text-2xl sm:text-3xl font-black font-mono tabular-nums leading-none"
                  style={{ color: 'rgba(16,185,129,0.75)', textShadow: '0 0 16px rgba(16,185,129,0.3)' }}>
                  {pop}
                </div>
                <div className="text-[8px] uppercase tracking-widest text-gray-700">entities</div>
              </div>
            )}

            {/* Extinction banner */}
            {frame && pop === 0 && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="text-center">
                  <div className="text-xl font-mono text-red-900/50 tracking-[0.4em] uppercase">Extinction</div>
                  <div className="text-[9px] text-gray-700 mt-1 tracking-wider">Awaiting next world...</div>
                </div>
              </div>
            )}

            {/* Connection overlay */}
            {!connected && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                <div className="text-center">
                  <div className="text-3xl mb-2 opacity-20">◌</div>
                  <div className="text-sm text-gray-400 font-mono tracking-wider">Connecting...</div>
                </div>
              </div>
            )}

            {/* Watermark */}
            <div className="absolute bottom-2 left-3 text-[8px] text-gray-800 font-mono tracking-widest uppercase pointer-events-none">
              axiom-forge v0.1 · gen {gen} · world {wIdx}/{wTot} · tick {tick}
            </div>
          </div>

          {/* Bottom strip — stats + log */}
          <div className="shrink-0 border-t border-white/[0.04] bg-black/40">
            {/* Ambient stat row */}
            <div className="flex items-center justify-between px-3 sm:px-5 py-2 gap-2 overflow-x-auto">
              <Stat label="Gen"        value={gen + 1}             color="#06b6d4" />
              <div className="w-px h-6 bg-white/[0.05] shrink-0" />
              <Stat label="World"      value={`${wIdx}/${wTot}`}   color="#6b7280" />
              <div className="w-px h-6 bg-white/[0.05] shrink-0" />
              <Stat label="Population" value={pop}                 color={pop > 50 ? '#10b981' : pop > 0 ? '#f59e0b' : '#ef4444'} />
              <div className="w-px h-6 bg-white/[0.05] shrink-0" />
              {scores && <>
                <Stat label="Score"    value={scores.total.toFixed(2)} color="#f59e0b" />
                <div className="w-px h-6 bg-white/[0.05] shrink-0" />
              </>}
              <Stat label="Best Ever"  value={best.toFixed(2)}    color="#8b5cf6" />

              {/* Sparkline */}
              {meta?.generations && meta.generations.length > 1 && (
                <div className="flex items-end gap-0.5 h-5 ml-2 shrink-0">
                  {meta.generations.slice(-16).map((g, i) => (
                    <div key={i} className="w-1 rounded-t"
                      style={{ height: `${Math.max(2, (g.best / (best || 1)) * 20)}px`,
                        background: i === meta.generations.length - 1 ? 'rgba(6,182,212,0.8)' : 'rgba(255,255,255,0.07)' }} />
                  ))}
                </div>
              )}
            </div>

            {/* Log */}
            <TransmissionLog entries={log} />
          </div>
        </div>

        {/* ── Right panel ───────────────────────────────────────── */}
        <aside className="hidden lg:flex w-56 xl:w-64 shrink-0 border-l border-white/[0.03] bg-black/30 flex-col">
          {/* Chart */}
          <div className="p-3 border-b border-white/[0.04]">
            <h4 className="text-[8px] uppercase tracking-[0.2em] text-gray-600 mb-2">Population</h4>
            <PopulationChart snapshots={snapshots as any} width={200} height={80} />
          </div>

          {/* Score bars */}
          {scores && (
            <div className="p-3 border-b border-white/[0.04]">
              <h4 className="text-[8px] uppercase tracking-[0.2em] text-gray-600 mb-2">Emergence Scores</h4>
              {([
                ['Persistence',   scores.persistence,      '#10b981'],
                ['Diversity',     scores.diversity,        '#8b5cf6'],
                ['Complexity',    scores.complexityGrowth, '#ec4899'],
                ['Communication', scores.communication,    '#06b6d4'],
                ['Env Structure', scores.envStructure,     '#f59e0b'],
                ['Adaptability',  scores.adaptability,     '#f97316'],
              ] as [string, number, string][]).map(([label, val, color]) => (
                <div key={label} className="mb-1.5">
                  <div className="flex justify-between text-[8px] mb-0.5">
                    <span className="text-gray-600">{label}</span>
                    <span className="font-mono" style={{ color }}>{val.toFixed(2)}</span>
                  </div>
                  <div className="h-0.5 rounded-full bg-white/[0.04]">
                    <div className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${Math.min(100, val * 100)}%`, background: color, boxShadow: `0 0 4px ${color}60` }} />
                  </div>
                </div>
              ))}
              <div className="mt-2 pt-2 border-t border-white/[0.04] flex justify-between items-baseline">
                <span className="text-[8px] text-gray-600 uppercase tracking-wider">Total</span>
                <span className="text-lg font-black font-mono"
                  style={{ color: '#f59e0b', textShadow: '0 0 10px rgba(245,158,11,0.35)' }}>
                  {scores.total.toFixed(2)}
                </span>
              </div>
            </div>
          )}

          {/* Lineage heatmap */}
          {meta?.generations && meta.generations.length > 0 && (
            <div className="p-3 flex-1">
              <h4 className="text-[8px] uppercase tracking-[0.2em] text-gray-600 mb-2">Lineage</h4>
              <div className="flex flex-wrap gap-0.5">
                {meta.generations.slice(-60).map((g, i) => {
                  const norm = best > 0 ? g.best / best : 0;
                  return (
                    <div key={i} title={`Gen ${g.gen} · ${g.best.toFixed(2)}`}
                      className="w-2.5 h-2.5 rounded-sm"
                      style={{ background: `rgba(6,182,212,${0.06 + norm * 0.94})`,
                        boxShadow: norm > 0.9 ? '0 0 4px rgba(6,182,212,0.5)' : 'none' }} />
                  );
                })}
              </div>
              <div className="mt-2 text-[8px] text-gray-700 font-mono">
                {meta.generations.length} generations observed
              </div>
            </div>
          )}

          {/* Emergence ladder — visible in right panel on tablet only */}
          {emergence.stage >= 0 && (
            <div className="p-3 border-t border-white/[0.04]">
              <h4 className="text-[8px] uppercase tracking-[0.2em] text-gray-600 mb-1">Current Stage</h4>
              <div className="text-xs font-semibold text-cyan-400">
                {['Replicators','Communication','External Memory','Tool Use','Abstraction','Civilization','World Engineering','Recursive Threshold'][emergence.stage]}
              </div>
              <div className="text-[8px] text-gray-600 mt-0.5">Stage {emergence.stage + 1} of 8</div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className="flex flex-col items-center shrink-0">
      <span className="text-sm sm:text-base font-black font-mono tabular-nums leading-none" style={{ color }}>
        {value}
      </span>
      <span className="text-[7px] sm:text-[8px] uppercase tracking-widest text-gray-600 mt-0.5">{label}</span>
    </div>
  );
}
