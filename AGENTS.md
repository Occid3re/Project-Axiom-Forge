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

The server bundle is built with esbuild (`build-server.mjs`) — no TypeScript runtime on the VPS. The client is built with Vite + React.

---

## Architecture

```
Browser (Socket.IO client)
  └─ App.tsx             React shell, socket wiring
  └─ WorldView.tsx        60fps RAF loop → WorldRenderer
  └─ WorldRenderer        4-pass WebGL: scene→H-blur→V-blur→composite
  └─ EmergenceLadder      8-stage emergence detection from scores

Node.js server (server/index.ts)
  ├─ setInterval 10ms    evalBatch(5) — 500 t/s meta-evolution
  └─ setInterval 33ms    displayStep() → Socket.IO binary broadcast

SimulationController (server/simulation.ts)
  ├─ Eval loop           evolves WorldLaws across generations
  └─ Display loop        runs best-laws World at 30fps, packs frames
```

### Two-loop timing (critical — do not break)
- Eval: `evalBatch(5)` every **10ms** → 500 t/s.
  **Must be small batches** or the 33ms display timer gets starved.
  Previous bug: `evalBatch(50)` every 100ms caused visible jitter.
- Display: `displayStep()` every **33ms** → exactly one world tick per frame.

### Frame wire format (`packFrame` → `decodeFrame`)
```
Bytes 0-3:   magic 0x41584647 ("AXFG")
Bytes 4-7:   gridW (uint32 LE)
Bytes 8-11:  gridH (uint32 LE)
Bytes 12-15: entityCount (uint32 LE)
Bytes 16-19: tick (uint32 LE)
Bytes 20...: W*H resource bytes (uint8, 0-255)
           + W*H*3 signal bytes (3 channels interleaved per cell)
           + entityCount X values (all sequential)
           + entityCount Y values (all sequential)
           + entityCount Energy values (uint8, 0-255 = 0-1.5 scaled)
           + entityCount Action values (0-5)
           + entityCount Aggression values (gene[3] * 255)
```
**IMPORTANT**: entity arrays are written **sequentially** (all X, then all Y, etc.)
not interleaved. The decoder in `protocol.ts` expects this layout. If you ever
change `packFrame`, make sure to match the sequential decoder.
(Previous bug: packer wrote interleaved x,y,e,a,g per entity → all entity
positions were garbled for N > 1.)

---

## File Map

```
server/
  index.ts              Express + Socket.IO, two setInterval loops
  simulation.ts         SimulationController, EVAL_CONFIG, DISPLAY_CONFIG, packFrame
  ecosystem.config.cjs  PM2 config
  package.json          Only express + socket.io (runtime deps)

src/engine/
  constants.ts          GENOME_LENGTH=16, Gene enum, ActionType enum, ResourceDist
  entity-pool.ts        SoA typed arrays for all entities (MAX_ENTITIES=4096)
  world.ts              World.step() — the hot loop (signals, resources, entities)
  world-laws.ts         WorldLaws interface, PRNG (xoshiro128**), randomLaws/mutateLaws/crossoverLaws
  scoring.ts            scoreWorld() — 6 metrics (persistence, diversity, complexityGrowth, communication, envStructure, adaptability)
  protocol.ts           decodeFrame(), DecodedFrame, ServerMeta
  meta.ts               (GenerationResult type used by EmergenceLadder)

src/ui/
  renderer.ts           WorldRenderer — 4-pass WebGL bloom, trail texture, adaptive quality
  components/
    WorldView.tsx        Canvas + RAF loop, consumes frameRef (NOT React state)
    EmergenceLadder.tsx  8-stage emergence detection + sidebar visualization
    PopulationChart.tsx  Population over time chart
    TransmissionLog.tsx  Server log message feed
    AmbientStats.tsx     Background stat display
    LineageTree.tsx       (scaffolded, not wired to live data yet)
    WorldLawsView.tsx    (scaffolded, not wired to live data yet)

src/
  App.tsx               Root: epilepsy gate, socket wiring, layout
  main.tsx              React entry

public/
  impressum.html / .de.html    Legal Notice (EN + DE) — required by Austrian law
  datenschutz.html / .de.html  Privacy Notice (EN + DE) — required by Austrian law
  favicon.svg

ops/nginx/
  lostuplink.conf       nginx: HTTPS, Socket.IO WebSocket upgrade, static serving

build-server.mjs        esbuild script: server/index.ts → server/dist/server.mjs
deploy.sh               Full build + SSH tar upload + PM2 restart
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
| sexualReproduction | bool | If true, offspring get 50/50 crossover with nearby mate |
| resourceRegenRate | 0.001–0.1 | How fast resources grow back |
| eatGain | 0.1–1.0 | Energy gained per EAT action |
| moveCost | 0.001–0.1 | Energy cost per MOVE |
| idleCost | 0.001–0.05 | Metabolic cost per tick |
| signalRange | 1–8 | How far signals propagate |
| signalChannels | 1–6 | Number of independent signal channels |
| signalDecay | 0.1–0.99 | How fast signals fade per tick |
| memorySize | 1–16 | Entity memory slots (floats) |
| memoryPersistence | 0–1 | How fast memory decays |
| disasterProbability | 0–0.05 | Chance of resource wipe per tick |
| maxPerceptionRadius | 1–6 | How far entities see |

### Entity Genome (16 genes, each float32 in [0,1])
| Index | Gene | Effect |
|---|---|---|
| 0 | MOVE_BIAS_X | Directional drift in X |
| 1 | MOVE_BIAS_Y | Directional drift in Y |
| 2 | MOVE_RANDOMNESS | Noise on movement direction |
| 3 | AGGRESSION | Weight toward ATTACK action |
| 4 | REPRO_THRESHOLD | Energy ratio needed before reproducing |
| 5 | EAT_PRIORITY | Weight toward EAT action |
| 6 | SIGNAL_CHANNEL | Which signal channel to use (0–signalChannels) |
| 7 | SIGNAL_STRENGTH | How loud signals are + weight toward SIGNAL |
| 8 | SIGNAL_RESPONSIVENESS | Influence of nearby signals on action choice |
| 9 | PERCEPTION_RANGE | Sensing radius for nearby entities/signals |
| 10 | MEMORY_WRITE_RATE | How fast entity writes experience to memory |
| 11 | MEMORY_READ_WEIGHT | How much memory influences action decisions |
| 12 | COOPERATION | Kin-selection: reduces attack/boosts signaling near similar genomes |
| 13 | EXPLORE_EXPLOIT | Balance between roaming vs. staying at good spots |
| 14 | ENERGY_CONSERVATISM | Preference for IDLE when energy is low |
| 15 | ADAPTATION_RATE | (defined, not yet fully implemented in decideAction) |

### Entity Actions (6 types, chosen via softmax)
- **IDLE** (0) — do nothing, save energy
- **MOVE** (1) — step in genome-biased direction
- **EAT** (2) — consume resource at current cell
- **REPRODUCE** (3) — spawn child with mutated genome (+ mate crossover if sexualReproduction)
- **SIGNAL** (4) — emit signal on chosen channel, broadcast to nearby cells
- **ATTACK** (5) — steal energy from adjacent entity

### Scoring (6 metrics)
| Metric | Weight | What it rewards |
|---|---|---|
| persistence | 1.0 | Fraction of ticks with living population + dynamism |
| diversity | 1.5 | Mean pairwise genome distance over time |
| complexityGrowth | 1.5 | Diversity increasing over time + population stability |
| communication | 2.5 | Signal-birth correlation (signals that correlate with reproduction) |
| envStructure | 1.0 | Variance in resource coverage (entities impact environment) |
| adaptability | 1.8 | Recovery from population crashes |

### Meta-Evolution Config (EVAL_CONFIG)
- `gridSize`: 64×64 grid for fast evaluation
- `worldSteps`: 800 ticks per candidate world
- `worldsPerGeneration`: 10 candidate worlds per generation
- `topK`: 2 survivors (strong selection pressure)
- `mutationStrength`: 0.08 (tight mutations preserve good laws)
- Stagnation escape: after 30 generations with <5% improvement, inject 25% random laws + double mutation
- Each generation also produces crossover children from top-2 survivors

### Display Config (DISPLAY_CONFIG)
- `gridSize`: 80×80 (larger, more dramatic)
- `initialEntities`: 80
- `minLifetimeTicks`: 600 (~20s at 30fps before a new best world can replace it)
- Display world persists until extinction, then restarts with best laws
- Periodic refresh every 9000 ticks (~5 min) to keep up-to-date

---

## Rendering Pipeline

### WorldRenderer (4-pass, adaptive quality)
1. **Scene pass** → FBO_scene (canvas resolution): resources=teal glow, signals=aurora, entities=gaussian splat glow, trails=amber wake
2. **H-blur** FBO_scene → FBO_blurA (half resolution): 9-tap Gaussian
3. **V-blur** FBO_blurA → FBO_blurB (half resolution): 9-tap Gaussian
4. **Composite** FBO_scene + FBO_blurB → canvas: screen blend + chromatic aberration + gamma

**Adaptive quality**: EMA of frame delta. If avg > 22ms (<45fps), skip blur passes and blit scene directly (fast path). Re-enables automatically when device keeps up.

**Trail texture**: CPU-side Float32Array that decays by 8% per frame. Entity positions written as energy values. Gives warm amber wake behind entity movement.

**Entity texture**: 3×3 Gaussian splat per entity (CPU-side). Center weight 1.0, axis-adjacent 0.45, diagonal 0.20. Makes entities visible as glowing dots rather than single pixels.

**Data textures**: All use `LINEAR` filtering — the 80×80 grid is bilinearly upscaled to canvas resolution with no pixel squares.

### Frame → Render path (zero React state)
```
Socket.IO 'frame' event
  → decodeFrame(buf)
  → frameRef.current = decoded    (ref, no re-render)

requestAnimationFrame loop (60fps)
  → read frameRef.current
  → renderer.updateFrame(f)       (upload to GPU)
  → frameRef.current = null
  → renderer.render(ms)           (4 render passes)
```

---

## Legal Pages (Austrian law requirement)
Four static HTML files in `public/` served directly by nginx/Express:
- `/impressum.html` — Legal Notice (EN)
- `/impressum.de.html` — Impressum (DE)
- `/datenschutz.html` — Privacy Notice (EN)
- `/datenschutz.de.html` — Datenschutz (DE)

Contact: Julius Szemelliker, Hainfelder Strasse 19, 3040 Neulengbach, Austria. juliussze@icloud.com

---

## Improvement Roadmap

### Priority 1 — Simulation depth (biggest impact on artificial life quality)

**A. Neural network genome**
Replace the 16 fixed-role genes with a genome that encodes weights of a tiny MLP:
`[4 inputs] → [8 hidden, tanh] → [6 action logits]`
The 4 inputs: localResource, entityDensity, signalStrength, energy.
Genome = 4×8 + 8×6 = 80 weights. Store in a longer genome (GENOME_LENGTH=80).
This allows emergent specialization — hunters, farmers, broadcasters — without
hand-coding the decision function. Currently, the decision function is hardcoded
and evolution can only tune the weights, not the structure.

**B. Species detection and coloring**
Each display frame, cluster entity genomes by first 4 genes (fast O(n) bucketing).
Assign stable color IDs to clusters. Send species color per entity in the binary frame.
Renderer shows different species as different hues — makes speciation visible.

**C. Persist best laws to disk**
On every new best score, write `bestLaws` as JSON to `/opt/axiom-forge/server/best-laws.json`.
On server startup, load this file if it exists and seed the initial population with it.
This means the simulation accumulates progress over restarts instead of resetting.
Implementation: in `SimulationController` constructor, try to load JSON; in `finishEvalWorld`
when new best is found, `fs.writeFileSync(...)`.

**D. Environmental engineering / niche construction**
Add a `DEPOSIT` action (ActionType = 6): entity deposits some energy into the grid,
permanently raising resource capacity at that cell. Entities that build "nests" by
depositing can support larger colonies. This implements Stage 3 (External Memory) and
Stage 4 (Tool Use) on the emergence ladder. Requires a new `resourceDeposit: Float32Array`
in World that persists across regeneration.

**E. Directed signaling**
Add a signal that carries the sender's REPRO_THRESHOLD gene value.
If receivers read this signal and their own threshold is nearby, they feel pressure to
cooperate. This seeds semantic communication — signals that carry meaningful content.

### Priority 2 — Infrastructure

**A. Island model meta-evolution**
Instead of one population of 10 worlds, run 3 independent islands of 4 worlds each.
Every 10 generations, migrate the best laws from each island to the others.
Prevents premature convergence, explores more of the law space simultaneously.
Each island can run in a separate `Worker` thread (Node.js `worker_threads`).

**B. Progressive evaluation (beam search)**
First pass: evaluate each candidate for 200 ticks. Keep top 50%.
Second pass: evaluate survivors for 800 ticks. Final scoring.
This eliminates bad laws 4× faster with the same total compute.

**C. Multi-core worker threads**
The eval loop is CPU-bound. On a multi-core VPS, spawn N workers (one per core)
each running independent eval worlds. Main thread collects results and runs
display loop. Could give 4–8× eval throughput.

**D. Checkpoint/restore**
Every 100 generations, save full simulation state to disk:
`{ generation, bestScore, bestLaws, population, generationSummaries }`
On crash/restart, restore from checkpoint. Requires `JSON.stringify` of WorldLaws[]
(all serializable) and a simple file write.

### Priority 3 — Visualization

**A. Species color bands in renderer**
Pack a 4th byte in the entity texture encoding species ID (0-7).
Fragment shader maps species ID to a hue offset. Makes evolution visible as color
divergence — one population splitting into multiple colored groups.

**B. World laws heatmap panel**
The `WorldLawsView.tsx` component exists but is not wired.
Wire `meta.bestLaws` from the server meta broadcast (add to `MetaBroadcast`).
Show key parameters (signalRange, memorySize, mutationRate) as a mini radar chart.
This lets viewers understand WHY the current world behaves as it does.

**C. Population chart with speciation**
`PopulationChart.tsx` exists but may not show species breakdown.
Wire the species count into the chart as stacked colored bands.
Show total population as a line, species as fills.

**D. Event annotations**
When interesting events happen (new best, extinction, stagnation escape), broadcast
a short string in `logEntry` that appears as a floating tooltip on the canvas.
Currently `logEntry` exists in the meta broadcast but is only shown in the log feed.

**E. Smooth camera (ambient drift)**
Add gentle UV offset in the scene fragment shader that slowly drifts over time:
`v_uv_shifted = v_uv + vec2(sin(u_time * 0.03) * 0.02, cos(u_time * 0.04) * 0.02)`
Makes the simulation feel like you're observing through a lens rather than a static grid.

### Priority 4 — The path to actual intelligence

Current level: **reactive agents** (no generalization, no cumulative learning, no semantics).

**What's needed for each emergence stage:**

| Stage | Currently | What Would Unlock It |
|---|---|---|
| 0 Replicators | ✅ working | — |
| 1 Communication | ✅ signals correlate | Semantic signal content (signals encoding specific states) |
| 2 External Memory | Partial (signals as trail) | Resource deposit action (persistent environmental writing) |
| 3 Tool Use | Not yet | Deposit + signal reading creates "constructed niches" |
| 4 Abstraction | Not yet | NN genome (memory that generalizes across situations) |
| 5 Civilization | Not yet | Species specialization + directed signal exchange |
| 6 World Engineering | Meta-evolving | Entities that modify their own mutation rate |
| 7 Recursive Threshold | Not yet | Entities encoding "instructions" in signal patterns that guide offspring behavior → cultural evolution outpaces genetic evolution |

**The key bottleneck**: memory is too simple. 4–16 floats that just store recent
observations can't form associations. Hebbian learning (connections strengthen when
co-active) within an entity's lifetime would unlock Stage 4. This would require
memory[i] += rate * input[j] * memory[j] (correlation-based update, not just direct
write), allowing entities to learn causal structure within their lifetime.

---

## Known Issues / Gotchas

- **Entity limit**: MAX_ENTITIES=4096. With 80 initial entities on 80×80 grid, practical
  max population is ~200-300 before grid cells fill up and limit reproduction.
- **Signal normalization**: Signals are clamped at packing time (×255 → uint8). Very high
  signal values saturate. Consider increasing the scale factor in `packFrame`.
- **Memory size mismatch**: WorldLaws has `memorySize` up to 16 but `MAX_MEMORY_SIZE=16` in
  constants. `updateMemory` only writes up to 4 slots regardless of memorySize.
  The other 12 slots are never written — wasted allocation.
- **ADAPTATION_RATE gene** (index 15) is defined in the Gene enum and in the genome, but
  `decideAction` in world.ts does not use it. It's dead code waiting to be wired.
- **sexualReproduction mate scan**: the radius-2 scan in `executeReproduce` is O(25) per
  reproduction event. At high entity density this is fine; at 200 entities it's ~200×25=5000
  array lookups per tick, which is still sub-millisecond.
- **Server meta broadcast**: `ServerMeta` in protocol.ts is missing `evalSpeed` field that
  `MetaBroadcast` in simulation.ts includes. The client won't receive evalSpeed.
- **`src/engine/meta.ts`**: contains `GenerationResult` type used by EmergenceLadder but
  the actual generation data comes from `meta.generations` over the wire. Check for drift.

---

## Performance Notes

- Eval loop: 500 t/s is comfortable on a 1-vCPU VPS. With 45 entities on 64×64,
  each tick takes ~0.1ms. With 200 entities, ~0.4ms. The 10ms interval gives plenty of slack.
- Display world: one tick per 33ms = effectively free.
- Socket.IO frame size: 20 + 80×80 + 80×80×3 + N×5 ≈ 26KB per frame at 30fps = ~780KB/s
  outbound per connected client. 10 clients = 7.8MB/s (fine for a VPS with 1Gbps uplink).
- WebGL renderer: 4-pass at 800×800 canvas = ~2.5M fragment samples per frame. Fast on
  desktop GPU, may hit 22ms budget on mobile → adaptive quality kicks in (blit path).
