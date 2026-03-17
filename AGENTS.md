# LostUplink: Axiom Forge — Agent Context

## What This Is

A **live meta-evolution simulation** deployed at https://lostuplink.com.

The server runs a two-level evolutionary system 24/7:
- **Inner evolution**: entities with 16-gene genomes eat, reproduce, signal, attack, and cooperate inside a physics world
- **Outer (meta) evolution**: the *laws of physics* themselves evolve across generations to find universes where complex life spontaneously emerges

All visitors see the **same shared simulation** via Socket.IO binary frames. No user input — purely read-only observation. The client renders with WebGL (4-pass bloom pipeline).

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
  └─ App.tsx             React shell, socket wiring, full-bleed layout
  └─ WorldView.tsx        60fps RAF loop + pointer/wheel zoom-pan → WorldRenderer
  └─ WorldRenderer        4-pass WebGL: scene→H-blur→V-blur→composite
  └─ EmergenceLadder      8-stage emergence detection from scores (left overlay)
  └─ WorldLawsView        Evolved physics parameter bars (right overlay, wired)
  └─ PopulationChart      Population over time (right overlay)

Node.js server (server/index.ts)
  ├─ setInterval 10ms    evalBatch(5) — ~500 t/s meta-evolution
  └─ setInterval 33ms    displayStep() → Socket.IO binary broadcast
```

### Two-loop timing (critical — do not break)
- Eval: `evalBatch(5)` every **10ms** → ~500 t/s.
  **Must be small batches** or the 33ms display timer gets starved.
- Display: `displayStep()` every **33ms** → exactly one world tick per frame.

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
           + entityCount Aggression      (predatorDrive*0.65 + aggression*0.35)
           + entityCount SpeciesHue      (genes 6+12+13 combined → 0–255)
```
**CRITICAL**: entity arrays are written **sequentially** (all X, then all Y, etc.) not interleaved. The decoder in `protocol.ts` expects this layout.

---

## File Map

```
server/
  index.ts              Express + Socket.IO, two setInterval loops
  simulation.ts         SimulationController, EVAL_CONFIG, DISPLAY_CONFIG, packFrame
  ecosystem.config.cjs  PM2 config — sets STATE_PATH=/opt/axiom-forge/state.json
  package.json          Only express + socket.io (runtime deps)

src/engine/
  constants.ts          GENOME_LENGTH=16, Gene enum, ActionType enum, ResourceDist
  entity-pool.ts        SoA typed arrays for all entities (MAX_ENTITIES=4096)
  world.ts              World.step() — the hot loop (signals, resources, entities)
  world-laws.ts         WorldLaws interface, PRNG (xoshiro128**), randomLaws/mutateLaws/crossoverLaws/starterLaws
  scoring.ts            scoreWorld() — 6 metrics
  protocol.ts           decodeFrame(), DecodedFrame, ServerMeta
  meta.ts               GenerationResult type used by EmergenceLadder

src/ui/
  renderer.ts           WorldRenderer — 4-pass WebGL bloom, cell ring rendering, u_pan/u_zoom viewport
  components/
    WorldView.tsx        Canvas + RAF loop + unified pointer zoom-pan (no React state)
    EmergenceLadder.tsx  8-stage emergence detection + left overlay sidebar
    PopulationChart.tsx  Population over time chart
    TransmissionLog.tsx  Server log message feed
    WorldLawsView.tsx    Evolved physics parameter bars (wired to meta.bestLaws)
    AmbientStats.tsx     Background stat display
    LineageTree.tsx       (scaffolded, not wired)

src/
  App.tsx               Root: epilepsy gate, socket wiring, full-bleed overlay layout
  main.tsx              React entry

public/
  impressum.html / .de.html    Legal Notice (EN + DE) — Austrian law
  datenschutz.html / .de.html  Privacy Notice (EN + DE) — Austrian law

ops/nginx/lostuplink.conf      nginx: HTTPS, WebSocket upgrade, static serving
build-server.mjs               esbuild script
deploy.sh                      Full build + SSH tar upload + PM2 restart
```

---

## Simulation Mechanics

### WorldLaws (evolvable physics, ~20 parameters)
| Parameter | Range | Effect |
|---|---|---|
| reproductionCost | 0.1–1.0 | Energy needed to reproduce |
| offspringEnergy | 0.05–0.8 | Child's starting energy |
| mutationRate | 0.01–0.5 | Per-gene mutation probability |
| mutationStrength | 0.01–0.3 | Per-gene mutation magnitude |
| sexualReproduction | bool | Offspring get 50/50 crossover with nearby mate |
| resourceRegenRate | 0.001–0.1 | How fast resources grow back |
| eatGain | 0.1–1.0 | Energy gained per EAT action |
| moveCost | 0.001–0.1 | Energy cost per MOVE |
| idleCost | 0.001–0.05 | Metabolic cost per tick |
| signalRange | 1–8 | How far signals propagate |
| signalChannels | 1–6 | Independent signal channels |
| signalDecay | 0.1–0.99 | How fast signals fade |
| memorySize | 1–16 | Entity memory slots |
| memoryPersistence | 0–1 | How fast memory decays |
| disasterProbability | 0–0.05 | Chance of resource wipe per tick |
| maxPerceptionRadius | 1–6 | How far entities see |
| terrainVariability | 0–1 | Spatial variation in resource capacity |
| attackTransfer | 0–0.8 | Fraction of victim energy transferred to attacker |

### Entity Genome (16 genes, each float32 in [0,1])
| Index | Gene | Effect |
|---|---|---|
| 0 | MOVE_BIAS_X | Directional drift in X |
| 1 | MOVE_BIAS_Y | Directional drift in Y |
| 2 | MOVE_RANDOMNESS | Noise on movement direction |
| 3 | AGGRESSION | Weight toward ATTACK action |
| 4 | REPRO_THRESHOLD | Energy ratio needed before reproducing |
| 5 | EAT_PRIORITY | Weight toward EAT action |
| 6 | SIGNAL_CHANNEL | Which channel to use (0–signalChannels) |
| 7 | SIGNAL_STRENGTH | Signal loudness + weight toward SIGNAL |
| 8 | SIGNAL_RESPONSIVENESS | Influence of nearby signals on action |
| 9 | PERCEPTION_RANGE | Sensing radius |
| 10 | MEMORY_WRITE_RATE | How fast entity writes to memory |
| 11 | MEMORY_READ_WEIGHT | How much memory influences action |
| 12 | COOPERATION | Kin-selection: reduce attack near similar genomes |
| 13 | EXPLORE_EXPLOIT | Roam vs. stay at good spots |
| 14 | ENERGY_CONSERVATISM | Prefer IDLE when energy is low |
| 15 | ADAPTATION_RATE | **Predator drive**: directed movement toward prey, attack bonus |

### Entity Actions
- **IDLE** (0) — do nothing
- **MOVE** (1) — step in genome-biased direction; predators chase nearest entity
- **EAT** (2) — consume resource at current cell
- **REPRODUCE** (3) — spawn child (+ mate crossover if sexualReproduction)
- **SIGNAL** (4) — emit signal on chosen channel
- **ATTACK** (5) — steal energy from adjacent entity; kill bonus for predators

### Overcrowding & Population Cap
- Each entity with >2 neighbors loses `0.04 * (neighborCount - 2)` energy per tick
- Hard reproduction cap: entities cannot reproduce if population ≥ 7% of grid cells
- Together these prevent population explosions that would stall the simulation

### Predator-Prey Dynamics
Gene 15 (ADAPTATION_RATE) acts as **predator drive**:
- High predator drive: directed movement toward nearest entity (not random), boosted ATTACK score, kill bonus = `predatorDrive * 0.45`
- Renderer encodes role as `predatorDrive * 0.65 + aggression * 0.35` → color axis teal→red

### Scoring (6 metrics → `scores.total`)
| Metric | Weight | What it rewards |
|---|---|---|
| persistence | 1.0 | Fraction of ticks with living population |
| diversity | 1.5 | Mean pairwise genome distance |
| complexityGrowth | 1.5 | Diversity increasing + population stability |
| communication | 2.5 | Signal-birth correlation (meaningful signals) |
| envStructure | 1.0 | Variance in resource coverage |
| adaptability | 1.8 | Recovery from population crashes |

### CPU Efficiency Penalty (new)
After scoring each eval world, the raw score is multiplied by a CPU factor:
```
avgTickMs = wallClock(800 steps) / 800
overload  = max(0, avgTickMs / 0.8ms - 1)
cpuFactor = 1 / (1 + overload * 0.6)
score    *= cpuFactor
```
Worlds that slow the server get penalized. Evolution selects for lean physics.
- 2× over target (1.6ms/step) → score × 0.625
- 4× over target (3.2ms/step) → score × 0.357

### Meta-Evolution Config (EVAL_CONFIG)
- `gridSize`: 64×64 for fast evaluation
- `worldSteps`: 800 ticks per candidate
- `worldsPerGeneration`: 10 candidates per generation
- `topK`: 2 survivors (strong selection)
- `mutationStrength`: 0.08
- `cpuTargetMs`: 0.8ms/step, `cpuPenaltyWeight`: 0.6
- Stagnation: after 30 generations <1% improvement → inject 25% random + 2× mutation
- Gen-0 seeded with 4 `starterLaws()` variants + 6 random (ensures interesting early display)

### Display Config (DISPLAY_CONFIG)
- `gridSize`: 256×256 (large, spatially rich)
- `initialEntities`: 180
- `minLifetimeTicks`: 240 (~8s) before a new best can replace the current display
- Fade-in on world reset: `fadeValue` 0→1 over ~60 frames via `u_fade` uniform
- Periodic refresh every 9000 ticks (~5 min)
- `starterLaws()` used for initial display before meta-evolution finds anything good

### State Persistence
On every new best score and every generation end, saves to `STATE_PATH`:
```json
{ "version": 2, "generation": N, "bestScore": X, "bestLaws": {...},
  "lastImprovementGen": N, "generationSummaries": [...], "population": [...] }
```
Loaded on startup (version check). Survives redeploys because the file is outside `server/`.

---

## Rendering Pipeline

### WorldRenderer (4-pass, adaptive quality)
1. **Scene pass** → FBO_scene: resources=amber-green glow, signals=3-channel fluorescence (red/teal/magenta), entities=cell ring pattern, trails=green residue
2. **H-blur** FBO_scene → FBO_blurA (half resolution): 9-tap Gaussian
3. **V-blur** FBO_blurA → FBO_blurB (half resolution): 9-tap Gaussian
4. **Composite** FBO_scene + FBO_blurB → canvas: screen blend + chromatic aberration + gamma lift + `u_fade`

**Adaptive quality**: EMA of frame delta. If avg > 22ms, skip blur passes (blit path). Re-enables automatically.

### Cell Ring Rendering (CPU-side entity texture)
Each entity is splatted onto a W×H RGBA texture per frame:
- `rr < 0.7`: bright nucleus (ringVal = 0.95)
- `rr < cellR * 0.60`: dark cytoplasm (ringVal = 0.0)
- `rr ≤ cellR`: membrane ring — sin bump peaking at outer 40% (ringVal = 0.95 * sin(π*t))
- `rr > cellR`: outer glow, fades to 0 within 1.2 grid cells

RGBA channels: R=ring intensity, G=speciesHue, B=role(predator+action), A=presence

### Species Color Palette (4-color 2D space)
```
role axis:    herbivore ─────────────────────── predator
              teal/lime                         orange/violet
species axis: type A ──────────────────────────── type B
```
`cellColor = mix(mix(c00,c01,speciesH), mix(c10,c11,speciesH), role)`
Final: `color += cellColor * ringIntensity * 1.5`

### Zoom / Pan Viewport
Scene shader has `u_pan` (vec2) and `u_zoom` (float) uniforms:
```glsl
vec2 wuv = u_pan + (v_uv - 0.5) * u_zoom;
```
Data textures use `REPEAT` wrap (world is toroidal — seamless pan).
FBO textures use `CLAMP_TO_EDGE` (must be power-of-2 for REPEAT; canvas size is arbitrary).
`ZOOM_MAX = 1.0` (can't zoom out past full world view).
WorldView uses **unified Pointer Events** for pan (1 pointer) and pinch zoom (2 pointers).

### Frame → Render path (zero React state)
```
Socket.IO 'frame' event
  → decodeFrame(buf)
  → frameRef.current = decoded    (ref, no re-render)

requestAnimationFrame loop (60fps)
  → read frameRef.current
  → renderer.updateFrame(f)       (upload to GPU textures)
  → frameRef.current = null
  → renderer.render(ms)           (4 render passes)
```

---

## UI Layout

```
┌─────────────────────────────────────────────────────┐
│ Header: brand · emergence stage pill · live status  │
├─────────────────────────────────────────────────────┤
│ Canvas area (flex-1, relative)                      │
│  ┌──────────┐ full-bleed WorldView ┌─────────────┐  │
│  │Emergence │                     │ Population  │  │
│  │Ladder    │                     │ Scores      │  │
│  │(overlay) │                     │ WorldLaws   │  │
│  │hidden<md │                     │(overlay)    │  │
│  │          │                     │hidden<lg    │  │
│  └──────────┘                     └─────────────┘  │
├─────────────────────────────────────────────────────┤
│ Bottom strip (shrink-0, sibling of canvas — never   │
│ covered by overlays):                               │
│   Gen · World · Pop · Score · Best · Server Xms     │
│   TransmissionLog                                   │
│   Mobile panel (lg:hidden): scores + physics grid   │
├─────────────────────────────────────────────────────┤
│ Footer: Buy Me a Coffee · Legal · Privacy           │
└─────────────────────────────────────────────────────┘
```

Sidebars are `absolute inset-y-0` over the canvas div only. Bottom strip is a flex sibling below the canvas div — never overlapped.

---

## Legal Pages (Austrian law)
Four static HTML files in `public/` served by Express:
- `/impressum.html` — Legal Notice (EN)
- `/impressum.de.html` — Impressum (DE)
- `/datenschutz.html` — Privacy Notice (EN)
- `/datenschutz.de.html` — Datenschutz (DE)

Contact: Julius Szemelliker, Hainfelder Strasse 19, 3040 Neulengbach, Austria.

---

## Improvement Roadmap

### Priority 1 — Simulation depth

**A. Neural network genome**
Replace the 16 fixed-role genes with a genome encoding a tiny MLP:
`[4 inputs] → [8 hidden, tanh] → [6 action logits]`
Inputs: localResource, entityDensity, signalStrength, energy. Genome = 80 weights.
Allows emergent specialization without hand-coded decision functions.

**B. Environmental engineering / niche construction**
Add a `DEPOSIT` action (ActionType = 6): entity deposits energy into the grid,
permanently raising resource capacity at that cell. Entities that build "nests"
support larger colonies. Implements Stage 3 (External Memory) and Stage 4 (Tool Use).
Requires a `resourceDeposit: Float32Array` in World that persists across regeneration.

**C. Hebbian memory**
Current memory is just floats written and read directly. Add correlation-based update:
`memory[i] += rate * input[j] * memory[j]`
Lets entities learn causal structure within their lifetime → unlocks Stage 4 (Abstraction).

**D. Directed / semantic signaling**
Have signals carry sender's REPRO_THRESHOLD gene value. If receivers detect matching
threshold, they cooperate. Seeds semantic communication — signals with meaningful content.

### Priority 2 — Infrastructure

**A. Multi-core worker threads**
Eval loop is CPU-bound. Spawn N workers (one per core) each running independent eval
worlds. Main thread collects results and runs display. Could give 4–8× eval throughput.
Node.js `worker_threads` + `SharedArrayBuffer` for results.

**B. Island model meta-evolution**
Run 3 independent populations of 4 worlds each. Every 10 generations, migrate the
best laws between islands. Prevents premature convergence, wider law-space exploration.

**C. Progressive evaluation (beam search)**
First pass: 200 ticks, keep top 50%. Second pass: 800 ticks, final scoring.
Eliminates bad laws 4× faster with same compute.

### Priority 3 — Visualization

**A. Population chart with species breakdown**
Show species count as stacked colored bands under the population line.

**B. Event annotations on canvas**
When new best / extinction / stagnation escape fires, show a floating toast overlay
on the canvas (currently only in the transmission log).

**C. Ambient camera drift**
Slow UV drift in scene shader: `wuv += vec2(sin(t*0.03)*0.02, cos(t*0.04)*0.02)`
Makes the view feel like a microscope lens rather than a locked camera.

**D. OG/Twitter card image**
Static screenshot for social sharing previews. Add `<meta og:image>` to `index.html`.

### Priority 4 — Path to actual intelligence

| Stage | Status | What Would Unlock It |
|---|---|---|
| 0 Replicators | ✅ | — |
| 1 Communication | ✅ signals exist | Semantic signal content |
| 2 External Memory | Partial | DEPOSIT action (persistent environmental writing) |
| 3 Tool Use | Not yet | Deposit + signal reading = constructed niches |
| 4 Abstraction | Not yet | NN genome + Hebbian memory |
| 5 Civilization | Not yet | Species specialization + directed signal exchange |
| 6 World Engineering | Meta-evolving | Entities modifying own mutation rate |
| 7 Recursive Threshold | Not yet | Signal patterns encoding offspring behavior → cultural evolution |

---

## Known Issues / Gotchas

- **WebGL1 REPEAT restriction**: Only power-of-2 textures can use `REPEAT` wrap.
  Data textures are 256×256 (✓ power-of-2) → REPEAT OK for toroidal panning.
  FBO textures are canvas-sized (arbitrary) → must use `CLAMP_TO_EDGE`.
  Using `replace_all` on CLAMP_TO_EDGE changes will break FBOs (black screen).

- **Entity limit**: MAX_ENTITIES=4096. Hard reproduction cap at 7% of grid cells
  (~4500 for 256×256) means the entity limit is never reached in practice.

- **Memory size mismatch**: `updateMemory` only writes up to 4 slots regardless of
  `memorySize` WorldLaw. Slots 4–15 are allocated but never written.

- **sexualReproduction mate scan**: radius-2 scan = O(25) per reproduction. Fine
  at current densities.

- **Signal saturation**: signals clamped to uint8 at pack time. Very high signal
  values (>1.0) saturate. Not currently a problem but worth noting.

- **State file versioning**: STATE_VERSION=2. If you change the SavedState shape,
  increment this constant or old saves will be silently ignored.

---

## Performance Notes

- Eval: ~500 t/s comfortable on 1-vCPU VPS. 64×64 + ~100 entities ≈ 0.2ms/step.
  CPU efficiency penalty kicks in above 0.8ms/step.
- Display: 256×256 + ~200 entities at 30fps. ~10ms/step measured. Well within 33ms budget.
- Frame size: 20 + 256×256 + 256×256×3 + N×6 ≈ 328KB per frame × 30fps ≈ **9.8MB/s per viewer**.
  Plan for ~10 simultaneous viewers max on a standard VPS uplink.
- WebGL: 4-pass at canvas resolution. Adaptive quality skips blur on mobile (<45fps).
- `serverMs` broadcast: EMA of display world step time, shown in UI stat bar as health indicator.
