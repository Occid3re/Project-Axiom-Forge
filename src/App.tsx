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
import { WorldLawsView } from './ui/components/WorldLawsView';
import { NeuralNetView } from './ui/components/NeuralNetView';

// ---- Types -----------------------------------------------------------------

interface Snapshot { population: number; diversity?: number; signalActivity?: number; }

// ---- Epilepsy Warning ------------------------------------------------------

function EpilepsyGate({ onEnter }: { onEnter: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#03070d]">
      <div className="w-full max-w-lg mx-4 rounded-2xl border border-amber-500/30 bg-[#0d1117] p-8 shadow-2xl shadow-amber-900/10">
        {/* Warning icon */}
        <div className="flex justify-center mb-6">
          <div className="w-16 h-16 rounded-full border-2 border-amber-400/60 flex items-center justify-center">
            <svg viewBox="0 0 24 24" className="w-8 h-8 text-amber-400" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
          </div>
        </div>

        <h1 className="text-center text-xl font-bold text-amber-300 tracking-wide mb-2">
          Photosensitivity Warning
        </h1>
        <p className="text-center text-xs text-amber-500/70 uppercase tracking-widest mb-6">
          Epilepsy / Seizure Risk
        </p>

        <div className="space-y-3 mb-8">
          <p className="text-sm text-gray-400 leading-relaxed">
            <span className="text-gray-200 font-medium">LostUplink: Axiom Forge</span> displays a
            rapidly evolving simulation with flashing lights, flickering patterns, and high-contrast
            animated visuals that change continuously.
          </p>
          <p className="text-sm text-gray-400 leading-relaxed">
            A small percentage of people may experience epileptic seizures when exposed to certain
            light patterns or flashing images. If you or anyone in your household has an epilepsy
            diagnosis or has ever experienced seizures, please consult a doctor before viewing.
          </p>
          <p className="text-sm text-gray-500 leading-relaxed">
            Stop watching and seek medical attention immediately if you experience dizziness,
            altered vision, eye or muscle twitching, or loss of awareness.
          </p>
        </div>

        <button
          onClick={onEnter}
          className="w-full py-3.5 rounded-xl bg-amber-500/10 border border-amber-500/40 text-amber-300 text-sm font-semibold tracking-wider hover:bg-amber-500/20 hover:border-amber-400/60 transition-all duration-200 active:scale-[0.98]"
        >
          I understand — Enter the simulation
        </button>
        <p className="text-center text-[10px] text-gray-700 mt-4">
          lostuplink.com &nbsp;·&nbsp;
          <a href="https://buymeacoffee.com/juliussze" target="_blank" rel="noopener noreferrer" className="hover:text-gray-500 transition-colors">☕ Support</a>
          &nbsp;·&nbsp;
          <a href="/impressum.html" className="hover:text-gray-500 transition-colors">Legal</a>
          &nbsp;·&nbsp;
          <a href="/datenschutz.html" className="hover:text-gray-500 transition-colors">Privacy</a>
        </p>
      </div>
    </div>
  );
}

// ---- App -------------------------------------------------------------------

export default function App() {
  const [entered, setEntered]         = useState(false);
  const [viewMode, setViewMode]       = useState<'simulation' | 'network'>('simulation');

  // Frame stored in a ref — bypasses React state so the RAF render loop
  // in WorldView reads it directly without triggering re-renders.
  const frameRef = useRef<DecodedFrame | null>(null);
  const [meta, setMeta]           = useState<ServerMeta | null>(null);
  const [connected, setConnected] = useState(false);
  const [log, setLog]             = useState<string[]>(['Connecting to Axiom Forge...']);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [emergence, setEmergence] = useState<EmergenceState>({ stage: -1, progress: new Array(8).fill(0) });
  const [laws, setLaws] = useState<import('./engine/world-laws').WorldLaws | null>(null);

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
    setSnapshots(p => [...p, {
      population: meta.population,
      diversity: meta.scores?.diversity ?? 0,
      signalActivity: meta.scores?.communication ?? 0,
    }].slice(-300));
  }, [meta]);

  // Socket connection
  useEffect(() => {
    const socket = io({ transports: ['websocket'], reconnectionDelay: 2000 });
    socketRef.current = socket;

    socket.on('connect',    () => { setConnected(true);  addLog('Connection established — observing simulation'); });
    socket.on('disconnect', () => { setConnected(false); addLog('Connection lost — reconnecting...'); });

    socket.on('frame', (buf: ArrayBuffer) => {
      const decoded = decodeFrame(buf);
      if (decoded) frameRef.current = decoded; // goes directly to RAF loop
    });

    socket.on('meta', (m: ServerMeta) => {
      setMeta(m);
      if (m.bestLaws) setLaws(m.bestLaws);
      if (m.logEntry) addLog(m.logEntry);
    });

    return () => { socket.disconnect(); };
  }, [addLog]);

  const sampleGenome = meta?.sampleGenome ?? null;

  const pop      = meta?.population ?? 0;
  const gen      = meta?.generation ?? 0;
  const wIdx     = meta?.worldIndex ?? 0;
  const wTot     = meta?.totalWorlds ?? 0;
  const best     = meta?.bestScore ?? 0;
  const tick     = meta?.tick ?? 0;
  const scores   = meta?.scores ?? null;
  const serverMs = meta?.serverMs ?? 0;
  const serverPressure = meta?.serverPressure ?? 0;
  // Server load color: green < 5ms, yellow 5–15ms, red > 15ms
  const serverColor = serverMs < 5 ? '#10b981' : serverMs < 15 ? '#f59e0b' : '#ef4444';
  // Pressure color: invisible when 0, yellow at 0.5, red at 1+
  const pressureColor = serverPressure < 0.1 ? '#374151' : serverPressure < 0.5 ? '#f59e0b' : '#ef4444';

  if (!entered) return <EpilepsyGate onEnter={() => setEntered(true)} />;

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

        {/* Right cluster: view toggle + status */}
        <div className="flex items-center gap-3">
          {/* Neural net / simulation toggle */}
          <button
            onClick={() => setViewMode(v => v === 'simulation' ? 'network' : 'simulation')}
            title={viewMode === 'simulation' ? 'View neural network X-ray' : 'View simulation'}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[9px] font-semibold uppercase tracking-wider border transition-all duration-200
              ${viewMode === 'network'
                ? 'bg-cyan-500/15 border-cyan-500/50 text-cyan-400 shadow-[0_0_10px_rgba(0,229,255,0.15)]'
                : 'bg-white/[0.03] border-white/[0.08] text-gray-500 hover:text-cyan-500/70 hover:border-cyan-500/20'}`}
          >
            {viewMode === 'network' ? (
              <><span className="text-[11px]">⬡</span> Network</>
            ) : (
              <><span className="text-[11px]">⬡</span> Network</>
            )}
          </button>

          {/* Connection status */}
          <div className="flex items-center gap-2">
            <span className="relative flex h-1.5 w-1.5">
              {connected && <span className="animate-ping absolute inset-0 rounded-full bg-emerald-400 opacity-75" />}
              <span className={`relative inline-flex rounded-full h-1.5 w-1.5 ${connected ? 'bg-emerald-500' : 'bg-red-600'}`} />
            </span>
            <span className="text-[9px] text-gray-600 uppercase tracking-widest hidden sm:inline">
              {connected ? 'Live' : 'Offline'}
            </span>
          </div>
        </div>
      </header>

      {/* ── Body ───────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">

        {/* Canvas area — sidebars overlay only THIS div, not the bottom strip */}
        <div className="flex-1 relative min-h-0">

          {/* Full-bleed canvas — simulation or network X-ray */}
          <div className="absolute inset-0 bg-[#070809]">
            {viewMode === 'simulation' ? (
              <>
                <WorldView frameRef={frameRef} className="w-full h-full" />

                {/* Extinction banner */}
                {meta && pop === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="text-center">
                      <div className="text-xl font-mono text-red-900/50 tracking-[0.4em] uppercase">Extinction</div>
                      <div className="text-[9px] text-gray-700 mt-1 tracking-wider">Awaiting next world...</div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              /* Neural network X-ray view */
              <NeuralNetView genome={sampleGenome} />
            )}

            {/* Connection overlay — shown in both modes */}
            {!connected && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                <div className="text-center">
                  <div className="text-3xl mb-2 opacity-20">◌</div>
                  <div className="text-sm text-gray-400 font-mono tracking-wider">Connecting...</div>
                </div>
              </div>
            )}

            {/* Watermark */}
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[8px] text-gray-800 font-mono tracking-widest uppercase pointer-events-none whitespace-nowrap">
              {viewMode === 'simulation'
                ? `gen ${gen + 1} · world ${wIdx}/${wTot} · tick ${tick}`
                : `neural network · fittest entity · gen ${gen + 1}`}
            </div>
          </div>

          {/* Left sidebar — overlays canvas only, not bottom strip */}
          <aside className={`absolute left-0 top-0 bottom-0 w-44 lg:w-52 z-10
                            bg-black/60 backdrop-blur-md border-r border-white/[0.05]
                            ${viewMode === 'network' ? 'hidden' : 'hidden md:flex'}`}>
            <EmergenceLadder emergence={emergence} tick={tick} />
          </aside>

          {/* Right sidebar — overlays canvas only, fully scrollable */}
          <aside className={`absolute right-0 top-0 bottom-0 w-56 xl:w-64 flex-col z-10
                            bg-black/60 backdrop-blur-md border-l border-white/[0.05] overflow-y-auto
                            ${viewMode === 'network' ? 'hidden' : 'hidden lg:flex'}`}>
            {/* Chart */}
            <div className="p-3 border-b border-white/[0.04] shrink-0">
              <h4 className="text-[8px] uppercase tracking-[0.2em] text-gray-600 mb-2">Population</h4>
              <PopulationChart snapshots={snapshots as any} width={200} height={80} />
            </div>

            {/* Score bars */}
            {scores && (
              <div className="p-3 border-b border-white/[0.04] shrink-0">
                <h4 className="text-[8px] uppercase tracking-[0.2em] text-gray-600 mb-2">Emergence Scores</h4>
                {([
                  ['Persistence',   scores.persistence,      '#10b981'],
                  ['Diversity',     scores.diversity,        '#8b5cf6'],
                  ['Complexity',    scores.complexityGrowth, '#ec4899'],
                  ['Communication', scores.communication,    '#06b6d4'],
                  ['Env Structure', scores.envStructure,     '#f59e0b'],
                  ['Adaptability',  scores.adaptability,     '#f97316'],
                  ['Speciation',    scores.speciation ?? 0,  '#a855f7'],
                  ['Interactions',  scores.interactions ?? 0, '#ef4444'],
                  ['Spatial',       scores.spatialStructure ?? 0, '#14b8a6'],
                  ['Dynamics',      scores.populationDynamics ?? 0, '#e11d48'],
                  ['Stigmergy',     scores.stigmergicUse ?? 0, '#d97706'],
                  ['Social',        scores.socialDifferentiation ?? 0, '#0ea5e9'],
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

            {/* World Laws — scrollable, no fixed height needed */}
            {laws && (
              <div className="p-3 border-b border-white/[0.04] shrink-0">
                <WorldLawsView laws={laws} title="Evolved Physics" />
              </div>
            )}

            {/* Emergence stage */}
            {emergence.stage >= 0 && (
              <div className="p-3 shrink-0">
                <h4 className="text-[8px] uppercase tracking-[0.2em] text-gray-600 mb-1">Current Stage</h4>
                <div className="text-xs font-semibold text-cyan-400">
                  {['Replicators','Communication','External Memory','Tool Use','Abstraction','Civilization','World Engineering','Recursive Threshold'][emergence.stage]}
                </div>
                <div className="text-[8px] text-gray-600 mt-0.5">Stage {emergence.stage + 1} of 8</div>
              </div>
            )}
          </aside>
        </div>

        {/* Bottom strip — sibling of canvas area, never covered by sidebars */}
        <div className="shrink-0 border-t border-white/[0.04] bg-black/80 backdrop-blur-sm">

          {/* Stat row */}
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
            <Stat label="Best"       value={best.toFixed(2)}    color="#8b5cf6" />
            <div className="w-px h-6 bg-white/[0.05] shrink-0" />
            <Stat label="Server"     value={`${serverMs.toFixed(1)}ms`} color={serverColor} />
            {serverPressure > 0.05 && <>
              <div className="w-px h-6 bg-white/[0.05] shrink-0" />
              <Stat label="Pressure"  value={`${(serverPressure * 100).toFixed(0)}%`} color={pressureColor} />
            </>}

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

          <TransmissionLog entries={log} />

          {/* Mobile panel — scores + physics, shown only when sidebars are hidden */}
          {(scores || laws) && (
            <div className="lg:hidden border-t border-white/[0.04]">

              {/* Emergence scores — horizontal scroll */}
              {scores && (
                <div className="flex gap-3 px-3 py-2 overflow-x-auto">
                  {([
                    ['Persist',  scores.persistence,      '#10b981'],
                    ['Diverse',  scores.diversity,        '#8b5cf6'],
                    ['Complex',  scores.complexityGrowth, '#ec4899'],
                    ['Signal',   scores.communication,    '#06b6d4'],
                    ['Environ',  scores.envStructure,     '#f59e0b'],
                    ['Adapt',    scores.adaptability,     '#f97316'],
                    ['Species',  scores.speciation ?? 0,  '#a855f7'],
                    ['Interact', scores.interactions ?? 0, '#ef4444'],
                    ['Spatial',  scores.spatialStructure ?? 0, '#14b8a6'],
                    ['Dynamic',  scores.populationDynamics ?? 0, '#e11d48'],
                    ['Stigm',    scores.stigmergicUse ?? 0, '#d97706'],
                    ['Social',   scores.socialDifferentiation ?? 0, '#0ea5e9'],
                  ] as [string, number, string][]).map(([label, val, color]) => (
                    <div key={label} className="flex flex-col items-center shrink-0 min-w-[36px]">
                      <div className="w-6 h-6 rounded-full flex items-center justify-center mb-0.5"
                        style={{ background: `${color}18`, border: `1px solid ${color}40` }}>
                        <span className="text-[8px] font-bold font-mono" style={{ color }}>{(val * 10).toFixed(0)}</span>
                      </div>
                      <span className="text-[7px] text-gray-600 text-center leading-tight">{label}</span>
                    </div>
                  ))}
                  <div className="w-px bg-white/[0.05] mx-1 shrink-0" />
                  <div className="flex flex-col items-center shrink-0 justify-center">
                    <span className="text-base font-black font-mono" style={{ color: '#f59e0b' }}>{scores.total.toFixed(1)}</span>
                    <span className="text-[7px] text-gray-600">total</span>
                  </div>
                </div>
              )}

              {/* Key physics — 2-row grid */}
              {laws && (
                <div className="grid grid-cols-3 gap-x-3 gap-y-1 px-3 pb-2 border-t border-white/[0.04] pt-2">
                  {([
                    ['Eat Gain',   laws.eatGain,            1.0],
                    ['Move Cost',  laws.moveCost,            0.1],
                    ['Attack',     laws.attackTransfer,      0.8],
                    ['Regen',      laws.resourceRegenRate,   0.1],
                    ['Repro Cost', laws.reproductionCost,    1.0],
                    ['Mutation',   laws.mutationRate,        0.5],
                  ] as [string, number, number][]).map(([label, val, max]) => (
                    <div key={label} className="flex flex-col">
                      <div className="flex justify-between">
                        <span className="text-[7px] text-gray-600">{label}</span>
                        <span className="text-[7px] font-mono text-gray-500">{val.toFixed(3)}</span>
                      </div>
                      <div className="h-0.5 bg-white/[0.04] rounded-full mt-0.5">
                        <div className="h-full rounded-full bg-cyan-600/60" style={{ width: `${Math.min(100, (val / max) * 100)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Legal footer */}
      <footer className="shrink-0 flex items-center justify-center gap-4 py-2 border-t border-white/[0.04]">
        <a href="https://buymeacoffee.com/juliussze" target="_blank" rel="noopener noreferrer"
           className="text-[9px] text-amber-700 hover:text-amber-500 transition-colors font-medium">
          ☕ Buy me a coffee
        </a>
        <span className="text-gray-800 text-[9px]">·</span>
        <a href="/impressum.html" className="text-[9px] text-gray-700 hover:text-gray-500 transition-colors">Legal Notice</a>
        <span className="text-gray-800 text-[9px]">·</span>
        <a href="/impressum.de.html" className="text-[9px] text-gray-700 hover:text-gray-500 transition-colors">Impressum</a>
        <span className="text-gray-800 text-[9px]">·</span>
        <a href="/datenschutz.html" className="text-[9px] text-gray-700 hover:text-gray-500 transition-colors">Privacy</a>
        <span className="text-gray-800 text-[9px]">·</span>
        <a href="/datenschutz.de.html" className="text-[9px] text-gray-700 hover:text-gray-500 transition-colors">Datenschutz</a>
      </footer>
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
