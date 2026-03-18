# LostUplink: Axiom Forge — Agent Context

## What This Is

A **live meta-evolution simulation** deployed at https://lostuplink.com.

The server runs a two-level evolutionary system 24/7:
- **Inner evolution**: entities with 180-weight Elman recurrent network genomes eat, reproduce, signal, attack, deposit, and absorb inside a physics world — all behaviour is emergent from the neural network weights + temporal memory + stigmergic glyph grid
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
  └─ EmergenceLadder      10-stage emergence detection from scores (left overlay)
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
           + W*H poison bytes (uint8, 0–255)
           + W*H glyph bytes (uint8, magnitude of 4-channel glyph vector)
           + entityCount X values        (all sequential)
           + entityCount Y values        (all sequential)
           + entityCount Energy values   (uint8, 0–255)
           + entityCount Action values   (0–7)
           + entityCount Aggression      (W2 ATTACK column mean, sigmoid → 0–255)
           + entityCount SpeciesHue      (W2 SIGNAL + W2 EAT column blend → 0–255)
           + entityCount Complexity      (genome weight std dev, sigmoid → 0–255)
           + entityCount Motility        (W2 MOVE column mean, sigmoid → 0–255)
```
**CRITICAL**: entity arrays are written **sequentially** (all X, then all Y, etc.) not interleaved. The decoder in `protocol.ts` expects this layout. Total: 8 entity arrays per frame.

**Aggression / SpeciesHue / Complexity / Motility derivation from MLP weights** (genome = 180 floats, W2 starts at index 100):
```ts
// ATTACK column (a=5): how strongly this network hunts
attackSum = sum_h(genome[100 + h*8 + 5]) for h=0..9
aggression255 = sigmoid(attackSum * 0.4) * 255

// SIGNAL column (a=4) + EAT column (a=2): behaviour fingerprint
sigTend = sigmoid(sum_h(genome[100 + h*8 + 4]) * 0.3)
eatTend = sigmoid(sum_h(genome[100 + h*8 + 2]) * 0.3)
species255 = (sigTend * 0.6 + eatTend * 0.4) * 255

// Complexity: genome weight standard deviation → evolution stage indicator
std = sqrt(variance(all 180 genome weights))
complexity255 = sigmoid((std - 1.2) * 2.5) * 255

// Motility: MOVE column (a=1) drive
moveSum = sum_h(genome[100 + h*8 + 1])
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
  constants.ts          GENOME_LENGTH=180, NN_INPUTS=10/HIDDEN=10/OUTPUTS=8, ActionType (8 actions), GLYPH_CHANNELS=4
  entity-pool.ts        SoA typed arrays; Xavier init; Gaussian mutation (no [0,1] clamp)
  world.ts              World.step() — hot loop; decideAction() = MLP forward pass; glyph grid; kin recognition
  world-laws.ts         WorldLaws interface (38 params), PRNG (xoshiro128**), randomLaws/mutateLaws/starterLaws
  scoring.ts            scoreWorld() — 12 metrics; diversity normalised by √GENOME_LENGTH
  protocol.ts           decodeFrame(), DecodedFrame (incl. glyphs), ServerMeta (incl. sampleGenome)
  meta.ts               GenerationResult type used by EmergenceLadder

src/ui/
  renderer.ts           WorldRenderer — 4-pass WebGL bloom, u_pan/u_zoom viewport
  components/
    WorldView.tsx        Canvas + RAF loop + unified pointer zoom-pan
    NeuralNetView.tsx    Canvas X-ray: glowing MLP graph, particle flow, activation
    EmergenceLadder.tsx  10-stage emergence detection + left overlay sidebar
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

### Entity Genome — Elman Recurrent Network (GENOME_LENGTH = 180)

Genome = 180 real-valued float weights (NOT clamped to [0,1]):
```
W1[0..99]:    10 inputs × 10 hidden — genome[input * 10 + hidden]
W2[100..179]: 10 hidden × 8 outputs — genome[100 + hidden * 8 + action]
```

**10 sensory inputs** (all normalised to ~[0, 1]):
```ts
inputs = [
  localResource,     // resource at own cell
  energyNorm,        // own energy / energyCap
  entityDensity,     // neighbor count / 24 (radius-2)
  signalStrength,    // mean signal in radius-2
  nearestKinEnergy,  // energy of closest same-species neighbor (0 if none)
  nearestThreatDist, // 1/distance to closest non-kin (0 if none)
  kinRatio,          // same-species / total neighbors (0.5 if alone)
  glyphStrength,     // magnitude of glyph vector at current cell
  glyphAffinity,     // dot(own hidden state, local glyph) — cultural similarity
  ageNorm,           // age / 500 capped at 1.0
]
```

**Forward pass** (every tick, every entity, zero allocation):
```ts
h_new   = tanh(Σ_k W1[k*10+h] * inputs[k])                              // 10 units
h_blend = (1-p) * h_new + p * h_prev                                      // p = memoryPersistence
logits  = Σ_h W2[100 + h*8 + a] * h_blend[h]                              // 8 values
action  = softmax_sample(logits)
h_prev  = h_blend                                                          // stored in memory[0..9]
```
Previous hidden state stored in entity memory slots 0–9. This makes each entity an Elman
recurrent network — it can condition decisions on temporal context (recent threats, resource
changes, signals, social encounters). `memoryPersistence` is an evolvable world law.

**Kin recognition**: cosine similarity of W2 SIGNAL+EAT columns (same 20 weights used for
species hue visualization). Entities with similarity ≥ `kinThreshold` are considered kin.

**Initialization**: Xavier normal — W1 ~ N(0, √(2/10)), W2 ~ N(0, √(2/10))
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
- **DEPOSIT** (6) — write compressed hidden state into glyph grid at current cell (stigmergic memory)
- **ABSORB** (7) — read glyph at current cell, blend into hidden state (cultural transmission)

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

### Scoring (12 metrics → `scores.total`)

Anti-gaming measures prevent degenerate "monoculture soup" strategies from scoring well.
Scoring heavily favors ecological richness (interactions 3.5, speciation 3.0) and social complexity (socialDifferentiation 3.0, stigmergicUse 2.5) over mere survival (persistence 0.5).

| Metric | Weight | What it rewards | Anti-gaming |
|---|---|---|---|
| persistence | 0.5 | Fraction of ticks alive + population dynamism | Low weight — survival is necessary but shouldn't dominate |
| diversity | 1.5 | Mean pairwise genome distance (√GENOME_LENGTH normalized) | ×chaosFactor: `1/(1 + mutRate×mutStrength×10)` |
| complexityGrowth | 1.0 | Sustained diversity growth across 4 quarters + pop stability | ×chaosFactor + requires monotonic multi-quarter increase |
| communication | 2.0 | Lagged signal→birth correlation (lag 5–15 ticks) | Temporal lag prevents same-tick coincidence gaming |
| envStructure | 0.5 | Variance in resource coverage | Low weight — too easy to max out |
| adaptability | 1.0 | Recovery from population crashes | — |
| speciation | 3.0 | Variance of pairwise genome distances (high = clusters = real species) | Random drift → low variance; real species → high variance |
| interactions | 3.5 | Ecological richness: attacks + signals per entity, balanced (geometric mean) | Rewards predator-prey arms races, penalizes passive grazer monocultures |
| spatialStructure | 1.5 | Birth/death rate variance + poison zone activity | Penalizes populations >2000 (uniform monoculture soup penalty) |
| populationDynamics | 1.5 | Population oscillation (peaks/troughs) + coefficient of variation | Penalizes flat population lines, rewards predator-prey cycles |
| stigmergicUse | 2.5 | Balanced DEPOSIT+ABSORB glyph usage (geometric mean × balance factor) | One-sided usage (deposit-only or absorb-only) penalized via balance factor |
| socialDifferentiation | 3.0 | Kin-selective behavior: attack rate variance + deposit rate variance | Rewards worlds where entities behave differently toward kin vs non-kin |

**Anti-monoculture design**: The scoring system is designed to prevent the "fill the screen with one species" trap:
- spatialStructure penalizes mean population >2000 (subtracts up to 0.5 from score)
- populationDynamics rewards oscillating populations, not static equilibria
- interactions requires BOTH attacks AND signals (geometric mean — one alone scores zero)
- cooperationBonus is capped at 0.02/neighbor to prevent free-energy farming

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
| 200 | Hard Reset | Keep alltime best + starter variants, reseed rest fully random. Resets stagnation counter. |

The hard reset at 200 gens forces the meta-evolution to re-explore the law space from scratch
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
{ "version": 4, "generation": N, "bestScore": X, "bestLaws": {...},
  "lastImprovementGen": N, "generationSummaries": [...], "population": [...] }
```
Loaded on startup (version check). Survives redeploys because the file is outside `server/`.
**STATE_VERSION=8** — incremented when scoring semantics or WorldLaws shape change.

### WorldLaws — Full Parameter Table (38 evolvable parameters)

| Category | Parameter | Range | Starter | What it does |
|---|---|---|---|---|
| Reproduction | reproductionCost | 0.1–1.0 | 0.28 | Energy cost to reproduce |
| | offspringEnergy | 0.05–0.8 | 0.20 | Starting energy of children |
| | mutationRate | 0.01–0.5 | 0.06 | Per-weight mutation probability |
| | mutationStrength | 0.01–0.3 | 0.06 | Gaussian noise std |
| | sexualReproduction | bool | true | Crossover vs asexual |
| | spawnDistance | 1–4 | 1 | Offspring placement range (colonial vs dispersal) |
| Energy | resourceRegenRate | 0.001–0.1 | 0.028 | Resource regrowth per tick |
| | eatGain | 0.1–1.0 | 0.42 | Energy per EAT action |
| | moveCost | 0.001–0.1 | 0.007 | Energy per MOVE (×moveSpeed) |
| | idleCost | 0.001–0.05 | 0.004 | Baseline energy drain |
| | energyCap | 0.5–3.0 | 1.5 | Max energy (tanky vs fragile) |
| | corpseEnergy | 0.1–1.0 | 0.50 | Fraction returned to grid on death |
| | agingRate | 0–0.01 | 0.002 | Extra drain per tick × age |
| Combat | attackTransfer | 0–0.8 | 0.50 | Energy stolen per attack |
| | attackRange | 1–3 | 1 | Attack search radius (ranged vs melee) |
| | moveSpeed | 1–3 | 1 | Cells per MOVE action |
| Communication | signalRange | 1–8 | 4 | Signal broadcast radius |
| | signalChannels | 1–6 | 3 | Frequency channels |
| | signalDecay | 0.1–0.99 | 0.80 | Per-tick signal multiplier |
| | signalCost | 0–0.05 | 0.01 | Energy per SIGNAL action |
| Social | cooperationBonus | 0–0.15 | 0.03 | Energy bonus per neighbor (herding) |
| | crowdingThreshold | 1–6 | 3 | Neighbors before overcrowding penalty |
| Memory | memorySize | 1–16 | 4 | Hidden units preserved |
| | memoryPersistence | 0–1 | 0.65 | Elman recurrent blend factor |
| Environment | resourceDistribution | 0–2 | CLUSTERED | Uniform/Clustered/Gradient |
| | disasterProbability | 0–0.05 | 0.003 | Per-tick disaster chance |
| | terrainVariability | 0–1 | 0.65 | Landscape roughness |
| | driftSpeed | 0–0.4 | 0.05 | Environmental current strength |
| Poison | poisonStrength | 0–0.3 | 0.05 | Damage per tick at full concentration |
| | deathToxin | 0–0.8 | 0.25 | Poison deposited per death |
| Stigmergy | glyphDecay | 0.990–0.999 | 0.996 | Per-tick glyph persistence (half-life 693–6931 ticks) |
| | depositCost | 0–0.03 | 0.01 | Energy cost per DEPOSIT action |
| | absorbCost | 0–0.02 | 0.005 | Energy cost per ABSORB action |
| | absorbRate | 0–0.3 | 0.1 | How much glyph overwrites hidden state on ABSORB |
| Social | kinThreshold | 0.6–0.95 | 0.8 | Cosine similarity cutoff for kin recognition |
| Other | maxPerceptionRadius | 1–6 | 3 | (currently unused) |
| | maxAge | 200–800 | 300 | Hard lifespan limit |
| | carryingCapacity | 0.02–0.30 | 0.10 | Sustainable cell fraction |

### Poison & Server Pressure

**Poison grid** (`Float32Array[gridSize²]`):
- Dying entities deposit `deathToxin` poison at their cell
- Poison damages living entities: `energy -= poison × poisonStrength` per tick
- Poison suppresses local resource regen: effective capacity × `(1 - poison×0.5)`
- Decays at ~0.5%/tick (base), slower under server pressure
- Rendered as pulsing magenta-red glow in the microscopy shader

**Server pressure** (display world only):
- `pressure = clamp((displayStepMs - 33) / 33, 0, 2)`
- Effects: slower regen, persistent poison, more disasters, higher energy costs
- Self-regulating feedback loop: heavy populations → slow server → harsh world → culls population

### Stigmergic Memory (Glyph Grid)

**Glyph grid** (`Float32Array[gridSize² × 4]`) — persistent environmental marks enabling cultural transmission:
- Entities DEPOSIT their compressed hidden state (10→4 channels, sum pairs) into the glyph grid
- Blend: `glyph = glyph × 0.3 + deposit × 0.7` (new info dominates)
- Entities ABSORB glyphs by reading the 4-channel vector, expanding to hidden state, and blending
- Absorb blend: `hidden = hidden × (1 - absorbRate) + expanded_glyph × absorbRate`
- Decay: `glyphs *= glyphDecay` per tick (0.990–0.999, much slower than signal decay)
- Rendered as warm gold/amber glow in the microscopy shader
- **Cultural transmission**: entities deposit learned behavioral representations (hidden state) into the environment. Offspring and strangers can absorb these, creating knowledge transfer across generations without genetic encoding.
- **glyphAffinity input**: dot product of entity's hidden state with local glyph. High affinity = "someone like me was here" = kin trail following.

### Kin Recognition (Social Perception)

Entities perceive neighbors within radius-2 and classify them as kin or threat:
- **Similarity metric**: cosine similarity of W2 SIGNAL+EAT columns (same 20 weights that determine species hue)
- **kinThreshold** (evolvable, 0.6–0.95): cutoff for kin classification
- **Social inputs**: nearestKinEnergy, nearestThreatDist, kinRatio feed into the MLP
- **Cooperation is kin-selective**: cooperationBonus only applies to kin neighbors
- Enables evolved kin-selective behavior: spare kin, attack strangers (or vice versa)

**Drift**: slowly rotating environmental current (`driftSpeed`). Each tick, entities have a probability-based chance of being pushed 1 cell in the current direction. Direction rotates at `tick × 0.003` radians.

---

## Rendering Pipeline

### WorldRenderer (4-pass, adaptive quality — bacteria microscope aesthetic)
1. **Scene pass** → FBO_scene: phase-contrast microscopy shader — agar substrate, squared signal fluorescence, entity presence halos, film grain, circular eyepiece vignette
2. **H-blur** FBO_scene → FBO_blurA (half resolution): 9-tap Gaussian
3. **V-blur** FBO_blurA → FBO_blurB (half resolution): 9-tap Gaussian
4. **Composite** FBO_scene + FBO_blurB → canvas: screen blend + barrel distortion + chromatic aberration + gamma + `u_fade`

**Adaptive quality**: EMA of frame delta. If avg > 22ms, skip blur passes (blit path).

**Cached texture buffers**: CPU-side Uint8Arrays (`_resBuf`, `_entBuf`, `_sigBuf`, `_trailBuf`) are allocated once per grid size change via `ensureTexBufs(n)` instead of per-frame. Eliminates ~1MB/frame GC pressure.

### 10 Distinct Body Plans (genome-driven morphology)
Each entity is splatted onto a W×H RGBA texture. Body plan selected from genome traits:

| # | Plan | Shape | Selection Criteria | Distance Distortion |
|---|---|---|---|---|
| 0 | **Coccus** | Round sphere | low aggression + low complexity | Standard ellipse (aspect ≈ 1.0) |
| 1 | **Bacillus** | Elongated rod | moderate complexity (default) | Standard ellipse + flagella |
| 2 | **Vibrio** | Comma/crescent | aggression > 0.55 | `ldx += curvature * ldy²` (quadratic bend) |
| 3 | **Amoeba** | Star/pseudopods | complexity > 0.5 + motility < 0.4 | `rr = rawR / (1 + amp * cos(angle * N))` |
| 4 | **Dividing** | Hourglass | energy > 0.7 | Gaussian center pinch on ldy |
| 5 | **Spirillum** | Corkscrew wave | motility > 0.7 + complexity > 0.5 | `ldy -= amp * sin(ldx * freq)` |
| 6 | **Diplococcus** | Paired circles | mid energy + complex + sessile | Gaussian center pinch (gentler) |
| 7 | **Filamentous** | Chain of cells | complexity > 0.7 + motility < 0.35 | Periodic `cos()` pinches along long axis |
| 8 | **Fusiform** | Diamond/spindle | moderate aggression + complex | `ldy *= (1 + taper * ldx²/cellR²)` |
| 9 | **Spirochete** | Thin undulating | motility > 0.75 + low complexity | High aspect + sinusoidal wave |

All plans share membrane ring, organelle, and halo rendering. Flagella on bacillus/vibrio/fusiform only.
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

## Emergence Ladder (10 stages)

The left sidebar tracks observable milestones. Each stage lights up when its progress ≥ 60%.

| # | Stage | Metric | Threshold | What to look for on screen |
|---|---|---|---|---|
| 0 | Survival | persistence | ≥ 0.3 | Entities persist, eat, reproduce — population stays above zero |
| 1 | Resource Cycling | envStructure | ≥ 0.15 | Resource coverage visibly fluctuates — entities reshape the agar |
| 2 | Signaling | communication | ≥ 0.12 | Fluorescent dye channels glow; signals predict births (lagged correlation) |
| 3 | Diversity | diversity | ≥ 0.15 | Multiple behavioral strategies coexist; genome divergence visible |
| 4 | Predation | interactions | ≥ 0.12 | Curved vibrio predators hunt round prey; attacks + survival coexist |
| 5 | Cultural Marks | stigmergicUse | ≥ 0.09 | Gold glyph marks appear on grid; entities deposit and absorb knowledge |
| 6 | Kin Selection | socialDifferentiation | ≥ 0.12 | Entities cooperate with kin, attack strangers; tribal behavior emerges |
| 7 | Speciation | speciation | ≥ 0.18 | Distinct genome clusters with gaps; different body shapes visible |
| 8 | Ecology | geo-mean(7 metrics) | ≥ 0.12 | All of the above active simultaneously — complex ecosystem |
| 9 | Meta-Evolution | generation acceleration | score rate increasing | Score trend accelerating across generations; physics improving |

Stages are sequential: a stage only lights up if all previous stages are also achieved.
`detectEmergence()` in `EmergenceLadder.tsx` computes progress[0..9] from WorldScores.

---

## Legal Pages (Austrian law)
Four static HTML files in `public/` served by Express. Contact: Julius Szemelliker, Hainfelder Strasse 19, 3040 Neulengbach, Austria.

---

## Improvement Roadmap

### Completed — Hybrid Social Perception + Stigmergic Memory (v8)

- Expanded neural network: 4→10 inputs, 8→10 hidden, 6→8 outputs (180 weights)
- Social perception: kin recognition via cosine similarity of W2 behavioral columns
- 6 new inputs: nearestKinEnergy, nearestThreatDist, kinRatio, glyphStrength, glyphAffinity, ageNorm
- Stigmergic glyph grid: 4-channel persistent environmental memory (Float32Array)
- DEPOSIT action: write compressed hidden state into glyph grid
- ABSORB action: read glyph and blend into hidden state (cultural transmission)
- 5 new evolvable laws: glyphDecay, depositCost, absorbCost, absorbRate, kinThreshold
- 2 new scoring metrics: stigmergicUse, socialDifferentiation
- Gold/amber glyph visualization in microscopy shader

### Priority 1 — Next steps toward open-ended intelligence

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

- **Elman memory slots**: Memory slots 0–9 store hidden state for the 10-unit recurrent network.
  `memoryPersistence` world law controls carry-over blend. Newborns (age ≤ 1) get pure reactive
  state seeded into memory on first tick to avoid dampening. Slots 0–7 also used by DEPOSIT
  (compressed to 4 glyph channels) and ABSORB (expanded back from 4 channels).

- **Signal saturation**: signals clamped to uint8 at pack time. Very high signal values
  saturate. Not currently a problem.

- **State file versioning**: STATE_VERSION=8. Increment if SavedState shape or scoring semantics change.

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
- `sampleGenome` in meta: 180 floats as JSON ≈ 800 bytes per broadcast. Negligible.
- NeuralNetView: 180 connections × 2 particles = 360 particles in a Canvas2D RAF loop.
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

### Follow-up Observation (Gen 1181, same day)

After running with v3 scoring for ~1200 generations, new local optimum found:

**Best Laws (Passive Grazer Optimum)**:
```
attackTransfer:    0      (ZERO predation — no combat at all!)
mutationRate:      0.132  (moderate — chaos factor 0.74×)
resourceRegenRate: 0.001  (minimum — scarce)
eatGain:           0.932  (near max — eat everything)
moveCost:          0.001  (minimum — free movement)
memoryPersistence: 0.56   (moderate recurrence)
```

**Problem**: Meta-evolution avoids combat entirely. Without predator-prey dynamics, there are no arms races, no defensive specialization, no ecological niches. Result: boring passive grazer monoculture.

**Also observed**: Stagnant 206 gens in aggressive tier. Aggressive injection (50% random, 4× mutation) wasn't breaking through — hard reset threshold was too high at 300.

### Fixes Applied (v4)
1. **Interactions metric (new, weight 1.5)** — `sqrt(attackScore × signalScore)` rewards worlds where entities BOTH attack AND signal. Geometric mean = both must be nonzero. Attack sweet spot at 0.05–0.3/entity/tick prevents massacre.
2. **Hard reset lowered**: 300 → 200 gens. Aggressive injection alone couldn't break through.
3. **5 distinct body plans** — Creatures no longer all ellipses. Shape selected from genome traits: coccus (round), bacillus (rod), vibrio (comma/crescent for predators), amoeba (star/pseudopods for complex sessile), dividing (hourglass for high energy).
4. **STATE_VERSION bumped to 4**.
