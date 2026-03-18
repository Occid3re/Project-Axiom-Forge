# LostUplink: Axiom Forge — Agent Context

## What This Is

A **live meta-evolution simulation** deployed at https://lostuplink.com.

The server runs a two-level evolutionary system 24/7:
- **Inner evolution**: entities with 80-weight Elman recurrent network genomes eat, reproduce, signal, and attack inside a physics world — all behaviour is emergent from the neural network weights + temporal memory
- **Outer (meta) evolution**: the *laws of physics* themselves evolve across generations to find universes where complex life spontaneously emerges

All visitors see the **same shared simulation** via Socket.IO binary frames. No user input — purely read-only observation. The client renders with WebGL (4-pass bloom pipeline) and includes an X-ray neural-network visualizer.

---

## Build & Deploy

```bash
# Local dev (client only — no server sim)
npm run dev              # Vite on :5173

# Build everything and deploy to lostuplink-prod VPS
node build-server.mjs    # bundles server/ + src/engine/ → server/dist/server.mjs
bash deploy.sh           # builds client + uploads via tar-over-ssh + restarts PM2
```

**Deploy host alias**: `lostuplink-prod` (defined in `~/.ssh/config`).
**Remote path**: `/opt/axiom-forge/`
**Process manager**: PM2 (`axiom-forge` app, `server/ecosystem.config.cjs`)
**Port**: 3001 (proxied by nginx → `https://lostuplink.com`)
**State file**: `/opt/axiom-forge/state.json` — persists across deploys (outside server/ dir)

The server bundle is built with esbuild (`build-server.mjs`) — no TypeScript runtime on the VPS. The client is built with Vite + React.

---

## Architecture

```
Browser (Socket.IO client)
  └─ App.tsx             React shell, socket wiring, full-bleed layout, view toggle
  └─ WorldView.tsx        60fps RAF loop + pointer/wheel zoom-pan → WorldRenderer
  └─ NeuralNetView.tsx    Canvas X-ray: animated MLP graph of fittest display entity
  └─ WorldRenderer        4-pass WebGL: scene→H-blur→V-blur→composite
  └─ EmergenceLadder      8-stage emergence detection from scores (left overlay)
  └─ WorldLawsView        Evolved physics parameter bars (right overlay, wired)
  └─ PopulationChart      Population over time (right overlay)

Node.js server (server/index.ts)
  ├─ sim.startEvalLoop()  async — dispatches eval worlds to N worker threads in parallel
  └─ setInterval 33ms     displayStep() → Socket.IO binary broadcast
```

### Eval loop (async + workers, replaces old setInterval)

- `startEvalLoop()` runs forever as a background async loop
- Each generation dispatches all 10 eval worlds **in parallel** across N workers
  (`N = os.cpus().length` — auto-adapts to VPS size)
- Workers run `eval-worker.ts`: one 800-step eval world per job, return scores + timing
- Main thread awaits `Promise.all(...)`, then runs selection + breeding
- Display loop (`setInterval(33ms)`) is independent and unaffected
- `evalSpeed` broadcast now reports effective ticks/sec across all workers combined

**CRITICAL**: Do not add blocking synchronous code between `displayStep()` calls or it will drop frames. The display setInterval and async eval loop coexist safely because JS is single-threaded — they interleave via the event loop.

### Frame wire format (`packFrame` → `decodeFrame`)
```
Bytes 0-3:   magic 0x41584647 ("AXFG")
Bytes 4-7:   gridW (uint32 LE)
Bytes 8-11:  gridH (uint32 LE)
Bytes 12-15: entityCount (uint32 LE)
Bytes 16-19: tick (uint32 LE)
Bytes 20...: W*H resource bytes (uint8, 0–255)
           + W*H*3 signal bytes (3 channels interleaved per cell)
           + entityCount X values        (all sequential)
           + entityCount Y values        (all sequential)
           + entityCount Energy values   (uint8, 0–255)
           + entityCount Action values   (0–5)
           + entityCount Aggression      (W2 ATTACK column mean, sigmoid → 0–255)
           + entityCount SpeciesHue      (W2 SIGNAL + W2 EAT column blend → 0–255)
           + entityCount Complexity      (genome weight std dev, sigmoid → 0–255)
           + entityCount Motility        (W2 MOVE column mean, sigmoid → 0–255)
```
**CRITICAL**: entity arrays are written **sequentially** (all X, then all Y, etc.) not interleaved. The decoder in `protocol.ts` expects this layout. Total: 8 entity arrays per frame.

**Aggression / SpeciesHue / Complexity / Motility derivation from MLP weights** (genome = 80 floats, W2 starts at index 32):
```ts
// ATTACK column (a=5): how strongly this network hunts
attackSum = sum_h(genome[32 + h*6 + 5]) for h=0..7
aggression255 = sigmoid(attackSum * 0.4) * 255

// SIGNAL column (a=4) + EAT column (a=2): behaviour fingerprint
sigTend = sigmoid(sum_h(genome[32 + h*6 + 4]) * 0.3)
eatTend = sigmoid(sum_h(genome[32 + h*6 + 2]) * 0.3)
species255 = (sigTend * 0.6 + eatTend * 0.4) * 255

// Complexity: genome weight standard deviation → evolution stage indicator
std = sqrt(variance(all 80 genome weights))
complexity255 = sigmoid((std - 1.2) * 2.5) * 255

// Motility: MOVE column (a=1) drive
moveSum = sum_h(genome[32 + h*6 + 1])
motility255 = sigmoid(moveSum * 0.3) * 255
```

---

## File Map

```
server/
  index.ts              Express + Socket.IO; async eval loop + display setInterval
  simulation.ts         SimulationController, WorkerPool, EVAL_CONFIG, packFrame
  eval-worker.ts        Worker thread script: runs one 800-step eval world per job
  ecosystem.config.cjs  PM2 config — sets STATE_PATH=/opt/axiom-forge/state.json
  package.json          Only express + socket.io (runtime deps)

src/engine/
  constants.ts          GENOME_LENGTH=80, NN_INPUTS/HIDDEN/OUTPUTS, ActionType, ResourceDist
  entity-pool.ts        SoA typed arrays; Xavier init; Gaussian mutation (no [0,1] clamp)
  world.ts              World.step() — hot loop; decideAction() = MLP forward pass
  world-laws.ts         WorldLaws interface, PRNG (xoshiro128**), randomLaws/mutateLaws/starterLaws
  scoring.ts            scoreWorld() — 6 metrics; diversity normalised by √GENOME_LENGTH
  protocol.ts           decodeFrame(), DecodedFrame, ServerMeta (incl. sampleGenome)
  meta.ts               GenerationResult type used by EmergenceLadder

src/ui/
  renderer.ts           WorldRenderer — 4-pass WebGL bloom, u_pan/u_zoom viewport
  components/
    WorldView.tsx        Canvas + RAF loop + unified pointer zoom-pan
    NeuralNetView.tsx    Canvas X-ray: glowing MLP graph, particle flow, activation
    EmergenceLadder.tsx  8-stage emergence detection + left overlay sidebar
    PopulationChart.tsx  Population over time chart
    TransmissionLog.tsx  Server log message feed
    WorldLawsView.tsx    Evolved physics parameter bars (wired to meta.bestLaws)

src/
  App.tsx               Root: epilepsy gate, socket, layout, view-mode toggle (sim ↔ network)
  main.tsx              React entry

public/
  impressum.html / .de.html    Legal Notice (EN + DE)
  datenschutz.html / .de.html  Privacy Notice (EN + DE)

ops/nginx/lostuplink.conf      nginx: HTTPS, WebSocket, static serving
build-server.mjs               esbuild bundler script
deploy.sh                      Full build + SSH tar upload + PM2 restart
```

---

## Simulation Mechanics

### WorldLaws (evolvable physics, ~20 parameters)
| Parameter | Range | Effect |
|---|---|---|
| reproductionCost | 0.1–1.0 | Energy needed to reproduce |
| offspringEnergy | 0.05–0.8 | Child's starting energy |
| mutationRate | 0.01–0.5 | Per-weight mutation probability |
| mutationStrength | 0.01–0.3 | Gaussian noise std per weight |
| sexualReproduction | bool | Offspring get 50/50 crossover with nearby mate |
| resourceRegenRate | 0.001–0.1 | How fast resources grow back |
| eatGain | 0.1–1.0 | Energy gained per EAT action |
| moveCost | 0.001–0.1 | Energy cost per MOVE |
| idleCost | 0.001–0.05 | Metabolic cost per tick |
| signalRange | 1–8 | How far signals propagate |
| signalChannels | 1–6 | Independent signal channels |
| signalDecay | 0.1–0.99 | How fast signals fade |
| memorySize | 1–16 | Entity memory slots (first 8 used for recurrent hidden state) |
| memoryPersistence | 0–1 | Hidden-state carry-over: 0 = reactive, 1 = frozen memory |
| disasterProbability | 0–0.05 | Chance of resource wipe per tick |
| maxPerceptionRadius | 1–6 | Sensing radius (used for density input; not gene-gated) |
| terrainVariability | 0–1 | Spatial variation in resource capacity |
| attackTransfer | 0–0.8 | Fraction of victim energy transferred to attacker |

### Entity Genome — Elman Recurrent Network (GENOME_LENGTH = 80)

Genome = 80 real-valued float weights (NOT clamped to [0,1]):
```
W1[0..31]:  4 inputs × 8 hidden  — genome[input * 8 + hidden]
W2[32..79]: 8 hidden × 6 outputs — genome[32 + hidden * 6 + action]
```

**Forward pass** (every tick, every entity, zero allocation):
```ts
inputs  = [localResource, energyNorm, entityDensity, signalStrength]  // all 0–1
h_new   = tanh(Σ_k W1[k*8+h] * inputs[k])                            // 8 units
h_blend = (1-p) * h_new + p * h_prev                                  // p = memoryPersistence
logits  = Σ_h W2[32 + h*6 + a] * h_blend[h]                           // 6 values
action  = softmax_sample(logits)
h_prev  = h_blend                                                      // stored in memory[0..7]
```
Previous hidden state stored in entity memory slots 0–7. This makes each entity an Elman
recurrent network — it can condition decisions on temporal context (recent threats, resource
changes, signals). `memoryPersistence` is an evolvable world law: at 0 = purely reactive
(feedforward MLP), at ~0.5 = half memory/half new input, at 1 = frozen hidden state.

**Initialization**: Xavier normal — W1 ~ N(0, √(2/4)), W2 ~ N(0, √(2/8))
**Mutation**: Gaussian noise `w += N(0, mutationStrength)`, soft-clamped ±6
**Crossover**: per-weight uniform crossover with nearby mate (if sexualReproduction)

**Signal channel/strength** (derived from genome, not named genes):
```ts
channel  = floor(|genome[0]| * signalChannels) % signalChannels
strength = sigmoid(mean(W2[:, 4]) * 0.3)  // SIGNAL column mean
```

### Entity Actions
- **IDLE** (0) — do nothing
- **MOVE** (1) — random direction (emergent directionality via MLP)
- **EAT** (2) — consume resource at current cell
- **REPRODUCE** (3) — spawn child (+crossover if sexualReproduction)
- **SIGNAL** (4) — emit on derived channel, strength from W2 SIGNAL column
- **ATTACK** (5) — target nearest entity; always gives kill bonus (0.45 energy)

### Corpse Ecology
Dead entities deposit **50% of their remaining energy** back into the resource grid at their cell.
This creates emergent scavenging dynamics: battlefields become resource hotspots, mass die-offs
trigger resource booms, and death feeds life. Entities that evolve to follow predators or
gravitate toward high-mortality areas gain a foraging advantage.

### Population Pressure & Species Turnover
Three layered mechanisms force competitive exclusion and prevent species accumulation:

1. **Max age (`laws.maxAge`, 200–800 ticks)** — entities die of old age regardless of energy.
   Forces generational turnover; early species can't persist indefinitely. Evolved by meta-evolution.

2. **Exponential air pressure (hard cap: 4096 entities)**
   Computed **once per tick (O(1))**:
   ```ts
   const MAX_POP     = 4096;
   const ratio       = n / MAX_POP;
   const airPressure = Math.min(0.3, 0.0002 * Math.exp(ratio * 9));
   ```
   Applied inline in the existing entity loop — no extra passes, zero allocation.
   - **Exponential curve**: noticeable at 1024, painful at 2048, fatal above 2500
   - n=1024 (25%): ~0.002/tick (~½ idleCost). n=2048 (50%): ~0.018 (~4.5× idleCost). n=2500 (61%): ~0.055 (~14× idleCost). n=3000+: capped at 0.3
   - Reproduction hard-capped at `min(4096, gridW * gridH * 0.07)` so even the large display world (256×256) can't exceed 4096
   - Capped at 0.3/tick so entities always have a tick or two to act before dying

3. **Local overcrowding** — each entity with >2 neighbors loses `0.04 × (neighbors − 2)` energy/tick.

### Scoring (7 metrics → `scores.total`)

Anti-gaming measures prevent degenerate "mutation soup" strategies from scoring well.

| Metric | Weight | What it rewards | Anti-gaming |
|---|---|---|---|
| persistence | 1.0 | Fraction of ticks alive + population dynamism | — |
| diversity | 1.0 | Mean pairwise genome distance (√GENOME_LENGTH normalized) | ×chaosFactor: `1/(1 + mutRate×mutStrength×10)` |
| complexityGrowth | 1.5 | Sustained diversity growth across 4 quarters + pop stability | ×chaosFactor + requires monotonic multi-quarter increase |
| communication | 2.0 | Lagged signal→birth correlation (lag 5–15 ticks) | Temporal lag prevents same-tick coincidence gaming |
| envStructure | 1.0 | Variance in resource coverage | — |
| adaptability | 1.8 | Recovery from population crashes | — |
| speciation | 1.5 | Variance of pairwise genome distances (high = clusters = real species) | Random drift → low variance; real species → high variance |

**Chaos factor** prevents high-mutation strategies from getting free diversity points:
- starterLaws (mutRate=0.06, mutStrength=0.06): chaosFactor = 0.96× (negligible penalty)
- Degenerate (mutRate=0.477, mutStrength=0.3): chaosFactor = 0.41× (harsh discount)

**Max theoretical score**: ~10.8 (but chaos discount and lagged correlation make >8.0 genuinely hard)

### CPU Efficiency Penalty
After scoring each eval world (computed in eval-worker.ts):
```
avgTickMs = wallClock(1600 steps) / 1600
overload  = max(0, avgTickMs / 0.8ms - 1)
cpuFactor = 1 / (1 + overload * 0.6)
score    *= cpuFactor
```
Worlds that slow the server get penalized. Evolution selects for lean physics.

### Meta-Evolution Config (EVAL_CONFIG)
- `gridSize`: 64×64 for fast evaluation
- `worldSteps`: 1600 ticks per candidate (doubled from 800 — more entity generations per eval)
- `worldsPerGeneration`: 12 candidates per generation (run in parallel on workers)
- `topK`: 3 survivors (broader genetic diversity in population)
- `mutationStrength`: 0.08 (for WorldLaws, not entity genomes)
- Gen-0 seeded with 4 `starterLaws()` variants + 8 random

### Escalating Stagnation

When meta-evolution finds no improvement for N generations, increasingly aggressive measures fire:

| Stagnation (gens) | Tier | Action |
|---|---|---|
| 30 | Mild | 25% random injection, 2× mutation on survivors |
| 100 | Aggressive | 50% random injection, 4× mutation on survivors |
| 300 | Hard Reset | Keep alltime best + starter variants, reseed rest fully random. Resets stagnation counter. |

The hard reset at 300 gens forces the meta-evolution to re-explore the law space from scratch
while preserving the best discovery so far. This prevents permanent trapping in local optima.

### Display Config (DISPLAY_CONFIG)
- `gridSize`: 256×256 (large, spatially rich)
- `initialEntities`: 180
- `minLifetimeTicks`: 240 (~8s) before a new best can replace current display
- Periodic refresh every 9000 ticks (~5 min)
- Each frame the most-energetic entity's genome is sent as `sampleGenome` in meta

### State Persistence
On every new best score and every generation end, saves to `STATE_PATH`:
```json
{ "version": 3, "generation": N, "bestScore": X, "bestLaws": {...},
  "lastImprovementGen": N, "generationSummaries": [...], "population": [...] }
```
Loaded on startup (version check). Survives redeploys because the file is outside `server/`.
**STATE_VERSION=3** — incremented when scoring semantics change (old scores are incomparable).

---

## Rendering Pipeline

### WorldRenderer (4-pass, adaptive quality — bacteria microscope aesthetic)
1. **Scene pass** → FBO_scene: phase-contrast microscopy shader — agar substrate, squared signal fluorescence, entity presence halos, film grain, circular eyepiece vignette
2. **H-blur** FBO_scene → FBO_blurA (half resolution): 9-tap Gaussian
3. **V-blur** FBO_blurA → FBO_blurB (half resolution): 9-tap Gaussian
4. **Composite** FBO_scene + FBO_blurB → canvas: screen blend + barrel distortion + chromatic aberration + gamma + `u_fade`

**Adaptive quality**: EMA of frame delta. If avg > 22ms, skip blur passes (blit path).

**Cached texture buffers**: CPU-side Uint8Arrays (`_resBuf`, `_entBuf`, `_sigBuf`, `_trailBuf`) are allocated once per grid size change via `ensureTexBufs(n)` instead of per-frame. Eliminates ~1MB/frame GC pressure.

### Evolution-Driven Bacteria Morphology
Each entity is splatted onto a W×H RGBA texture with shape driven by genome traits:
- **Complexity** (genome weight std dev, 0–1): controls aspect ratio (1.0 → 2.2), membrane ruffling amplitude, organelle visibility
- **Motility** (W2 MOVE column, 0–1): drives flagella growth — wavy polar extensions visible at high motility × complexity
- **SpeciesHue**: determines rod orientation angle (species-dependent)
- Low-complexity entities appear as round coccus shapes; high-complexity as elongated rods with organelles, ruffled membranes, and flagella
- RGBA: R=presence intensity, G=speciesHue, B=role(attack+signal blend), A=presence mask

### Phase-Contrast Microscopy Shader
- **Agar substrate**: warm off-white with procedural noise texture
- **Resource visualization**: amber-green glow through substrate
- **Signal fluorescence**: squared values (`sig.r * sig.r`) for soft falloff — prevents "lightning" flash when few entities present. Multipliers: R=0.5, G=0.45, B=0.40
- **Presence halos**: bright membrane edge via `presence * (1 - presence) * 4.0` — phase-contrast ring effect
- **Film grain**: procedural `hash(uv + time)` noise
- **Eyepiece vignette**: `smoothstep(1.05, 0.60, vigR)` circular fade

### Species Color Palette (derived from MLP W2 columns)
```
role axis:    herbivore (low W2 ATTACK mean) ──── hunter (high W2 ATTACK mean)
              teal/lime                             orange/violet
species axis: low W2 SIGNAL/EAT ────────────────── high W2 SIGNAL/EAT
```

### Zoom / Pan Viewport
Scene shader has `u_pan` (vec2) and `u_zoom` (float) uniforms:
```glsl
vec2 wuv = u_pan + (v_uv - 0.5) * u_zoom;
```
Data textures use `REPEAT` wrap (toroidal panning).
FBO textures use `CLAMP_TO_EDGE` (arbitrary canvas size — NOT power-of-2).
`ZOOM_MAX = 1.0` (can't zoom out past full world view).
WorldView uses **unified Pointer Events** for pan (1 pointer) and pinch zoom (2 pointers).

### NeuralNetView X-ray
Canvas 2D component showing the MLP of the most-energetic display entity:
- **Background**: near-black (`#02040a`) + slow scan-line sweep + faint dot grid
- **Connections**: colored lines with shadow glow — cyan = positive weight, amber = negative
  - Alpha proportional to `|weight| / maxWeight`
  - 2 animated particles per connection, speed ∝ |weight|
- **Nodes**: glowing circles with radial gradient; brightness = actual activation for canonical inputs
  - Inputs: colored by type (green/amber/purple/cyan)
  - Hidden: hue cycles blue→purple across the 8 units (`hsl()` colors — use `withAlpha()` helper, NOT `color+'hex'` concatenation which only works for `#rrggbb`)
  - Outputs: each action has its color; winner gets extra glow + probability bar
- **Layout**: asymmetric padding — `padL = W*0.02+R*4` (room for input labels), `padR = R*11` (room for output labels + bars + %)
- **Canvas sizing**: wrapper div uses `position:absolute;inset:0`; canvas has explicit `width:100%;height:100%` CSS — critical to prevent the canvas attribute width from leaking into CSS display size (which would cause DPR×overflow and clip the right half on retina screens)
- Toggle button in header switches between simulation and network view; sidebars hide when network mode is active

### Frame → Render path (zero React state)
```
Socket.IO 'frame' event  → decodeFrame(buf) → frameRef.current = decoded
requestAnimationFrame loop → renderer.updateFrame(f) → renderer.render(ms)
```
```
Socket.IO 'meta' event   → setMeta(m) → sampleGenome → NeuralNetView (when active)
```

---

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Header: brand · emergence pill · [⬡ Network toggle] · live  │
├─────────────────────────────────────────────────────────────┤
│ Canvas area (flex-1, relative)                              │
│  ┌──────────┐  WorldView OR NeuralNetView  ┌─────────────┐  │
│  │Emergence │                              │ Population  │  │
│  │Ladder    │                              │ Scores      │  │
│  │(overlay) │                              │ WorldLaws   │  │
│  │hidden<md │                              │(overlay)    │  │
│  │          │                              │hidden<lg    │  │
│  └──────────┘                              └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│ Bottom strip (shrink-0, never covered):                     │
│   Gen · World · Pop · Score · Best · Server Xms             │
│   TransmissionLog                                           │
│   Mobile panel (lg:hidden): scores + physics grid           │
├─────────────────────────────────────────────────────────────┤
│ Footer: Buy Me a Coffee · Legal · Privacy                   │
└─────────────────────────────────────────────────────────────┘
```

Sidebars are `absolute inset-y-0` overlays on the canvas div only.
NeuralNetView fills the same canvas area as WorldView; **sidebars are hidden** when network mode is active (so the visualizer gets full width).

---

## Legal Pages (Austrian law)
Four static HTML files in `public/` served by Express. Contact: Julius Szemelliker, Hainfelder Strasse 19, 3040 Neulengbach, Austria.

---

## Improvement Roadmap

### Priority 1 — Simulation depth

**A. Environmental engineering / niche construction**
Add a `DEPOSIT` action (ActionType = 6): entity deposits energy into the grid,
permanently raising resource capacity at that cell. Entities that build "nests"
support larger colonies. Implements Stage 3 (External Memory) and Stage 4 (Tool Use).

**B. Hebbian memory**
Memory slots 0–7 are now used for Elman recurrent hidden state. Next step: add memory as
explicit MLP inputs (expand to 8 inputs: 4 sensory + 4 memory), growing genome to
8×8+8×6=112 weights. Add Hebbian correlation update for richer internal state beyond
simple hidden-state persistence.

**C. Directed semantic signaling**
Have signals carry sender behaviour fingerprint. Receivers that detect matching
fingerprints cooperate. Seeds semantic communication — signals with meaningful content.

### Priority 2 — Infrastructure

**A. Island model meta-evolution**
Run 3 independent populations of 4 worlds each. Every 10 generations, migrate the
best laws between islands. Prevents premature convergence.

**B. Progressive evaluation (beam search)**
First pass: 200 ticks, keep top 50%. Second pass: 800 ticks, final scoring.
Eliminates bad laws 4× faster with same compute.

### Priority 3 — Visualization

**A. Event annotations on canvas**
When new best / extinction / stagnation escape fires, show a floating toast overlay.

**B. Ambient camera drift**
Slow UV drift: `wuv += vec2(sin(t*0.03)*0.02, cos(t*0.04)*0.02)` — microscope lens feel.

**C. OG/Twitter card image**
Static screenshot for social sharing previews. Add `<meta og:image>` to `index.html`.

### Priority 4 — Path to actual intelligence

| Stage | Status | What Would Unlock It |
|---|---|---|
| 0 Replicators | ✅ | — |
| 1 Communication | ✅ signals exist | Semantic signal content |
| 2 External Memory | Partial | DEPOSIT action + memory as MLP input |
| 3 Tool Use | Not yet | Deposit + signal reading = constructed niches |
| 4 Abstraction | ✅ Elman recurrent + MLP genome | Hebbian memory as additional inputs |
| 5 Civilization | Not yet | Species specialization + directed signal exchange |
| 6 World Engineering | Meta-evolving | Entities modifying own mutation rate |
| 7 Recursive Threshold | Not yet | Signal patterns encoding offspring behaviour |

---

## Known Issues / Gotchas

- **WebGL1 REPEAT restriction**: Only power-of-2 textures can use `REPEAT` wrap.
  Data textures are 256×256 (✓ PoT) → REPEAT OK. FBO textures are canvas-sized → `CLAMP_TO_EDGE`.
  Using `replace_all` on CLAMP_TO_EDGE accidentally broke FBOs once — black screen.

- **Entity limit**: MAX_ENTITIES=4096. Hard reproduction cap at 7% grid cells means the
  entity limit is never reached in practice.

- **Elman memory slots**: Memory slots 0–7 store hidden state for the recurrent network.
  `memoryPersistence` world law controls carry-over blend. Newborns (age ≤ 1) get pure reactive
  state seeded into memory on first tick to avoid dampening.

- **Signal saturation**: signals clamped to uint8 at pack time. Very high signal values
  saturate. Not currently a problem.

- **State file versioning**: STATE_VERSION=3. Increment if SavedState shape or scoring semantics change.

- **tsx worker inheritance**: Workers spawned with `new Worker(path)` inherit the tsx ESM
  loader from the parent process (tsx v4.7+). No `execArgv` needed.

- **Genome not in saved state**: Entity genomes evolve fresh each eval world. Only
  WorldLaws are persisted. This is by design — laws seed entity evolution, not vice versa.

---

## Performance Notes

- Eval: workers run in parallel. 2-vCPU VPS → 2 workers → ~2× throughput vs single-threaded.
  Each 1600-step world ≈ 1–4s depending on entity count. ~10–20 worlds/minute effective.
- Display: 256×256 + ~200 entities at 30fps. ~10ms/step. Well within 33ms budget.
- Frame size: ~328KB per frame × 30fps ≈ **9.8MB/s per viewer**.
- `sampleGenome` in meta: 80 floats as JSON ≈ 800 bytes per broadcast. Negligible.
- NeuralNetView: 80 connections × 2 particles = 160 particles in a Canvas2D RAF loop.
  Zero allocation per frame (pre-allocated Float32Array for particles). ~0.5ms on any device.
- `serverMs` broadcast: EMA of display world step time, shown in UI as health indicator.

---

## Diagnostic Log: Overnight Run (2026-03-18)

### Observation
After running overnight (~11K generations), meta-evolution plateaued at best score **8.185/9.3** (88% of theoretical max). Stagnation counter reached **288 generations** with identical diversity injection firing every gen — no improvement.

### Best Laws (Degenerate Optimum)
```
reproductionCost: 0.1    (minimum — breed as fast as possible)
mutationRate:     0.477  (near maximum 0.5 — maximum random drift)
mutationStrength: 0.3    (maximum — large mutations)
resourceRegenRate: 0.001 (minimum — scarce resources)
eatGain:          0.989  (near maximum — grab everything)
sexualReproduction: false (faster reproduction)
memoryPersistence: 0.528 (moderate recurrence)
```

### Root Causes Identified
1. **Diversity was free**: High mutRate × mutStrength produced genome distance through random drift, not selection. The scoring rewarded this as real diversity.
2. **Communication was cheap**: Same-tick signal-birth correlation captured coincidental correlations, not causal communication.
3. **ComplexityGrowth rewarded chaos**: With high mutation, diversity always increases from first to last quarter.
4. **800 ticks too short**: Only 2–4 entity generations per eval — not enough for multi-generational evolutionary dynamics.
5. **Stagnation escape never escalated**: Same mild injection (25% random, 2× mutation) fired every generation for 288 gens without escalation.
6. **Score ceiling nearly saturated**: 88% of max left almost no gradient for further improvement.

### Fixes Applied
1. **Chaos factor discount** — `1/(1 + mutRate×mutStrength×10)` applied to diversity and complexityGrowth scores. Degenerate laws get 0.41× discount.
2. **Lagged communication** — Correlation computed with lag 5–15 ticks (signal → future births), not same-tick.
3. **Multi-quarter complexity** — Requires sustained monotonic growth across all 4 quarters, not just first-vs-last jump.
4. **Speciation metric (new)** — Variance of pairwise genome distances. High variance = genome clusters = real species. Random drift = uniform distances = low score.
5. **1600 eval steps** — Doubled from 800 for deeper multi-generational dynamics.
6. **Escalating stagnation** — 3 tiers: mild (30 gen), aggressive (100 gen), hard reset (300 gen).
7. **Larger population** — 12 candidates (was 10), 3 survivors (was 2) for broader search.
8. **STATE_VERSION bumped to 3** — Forces clean restart with new scoring semantics.
