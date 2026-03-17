# Axiom Forge

**Meta-evolution simulation** — evolves not just agents, but entire worlds with different underlying laws to discover which rule sets create open-ended complexity, adaptation, communication, and intelligence-like behavior from simple self-replicating entities.

## Quick Start

```bash
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. Click **Start Evolution** and watch worlds compete.

## Architecture

```
src/
├── engine/              # Pure simulation logic (no DOM, no UI)
│   ├── constants.ts     # Genome layout, entity limits, action types
│   ├── entity-pool.ts   # Struct-of-Arrays entity storage (zero GC)
│   ├── world-laws.ts    # Evolvable world physics + seeded PRNG
│   ├── world.ts         # Grid simulation engine
│   ├── scoring.ts       # World fitness metrics
│   └── meta.ts          # Meta-evolution outer loop (generator-based)
├── ui/
│   ├── renderer.ts      # WebGL grid renderer
│   └── components/      # React UI panels
└── App.tsx              # Main app shell
```

### Two Levels of Evolution

1. **Entity evolution** (inner loop): Entities with 16-gene genomes live on a grid, eat resources, reproduce with mutation, signal, attack, and use memory. No behavior is hardcoded — genome values are the policy via weighted sums and softmax selection.

2. **World evolution** (outer loop): Each world has ~20 evolvable "physics" parameters (reproduction cost, mutation rate, signal range, memory persistence, disaster probability, etc.). Worlds are scored on persistence, diversity, complexity growth, communication emergence, environmental structure, and adaptability. High-scoring worlds reproduce into mutated descendants.

### Performance Design

- **Struct-of-Arrays**: All entity data lives in `Float32Array` / `Int32Array` — no per-entity object allocation in the hot loop
- **WebGL rendering**: Grid state is uploaded as textures and rendered via fragment shaders
- **Generator-based meta loop**: `runMetaEvolution()` yields after each world, allowing the UI to update without blocking
- **Seeded PRNG**: xoshiro128** for deterministic, fast random number generation

### World Laws (Evolvable Parameters)

| Category | Parameters |
|---|---|
| Reproduction | cost, offspring energy, mutation rate/strength, sexual/asexual |
| Energy | resource regen, eat gain, move cost, idle cost, attack transfer |
| Communication | signal range, channels, decay |
| Memory | size, persistence |
| Environment | resource distribution, disaster probability, terrain variability |
| Perception | max radius |

### Scoring Metrics

| Metric | What it measures |
|---|---|
| Persistence | Survival rate + population dynamism |
| Diversity | Mean pairwise genome distance over time |
| Complexity Growth | Whether diversity increases across the simulation |
| Communication | Correlation between signal activity and population dynamics |
| Env Structure | How much entities reshape their resource landscape |
| Adaptability | Population recovery after downturns |

### Key Design Principle

> Intelligence is not hardcoded. The `decide()` function maps genome + perception + memory → action via weighted sums and stochastic selection. The genome IS the policy. Complexity emerges from interaction, not from a pre-built architecture.

## Future Expansion

- **Web Workers**: Run world simulations in parallel across CPU cores
- **WebGPU compute shaders**: Move entity processing to the GPU for 10-100x throughput
- **Larger genomes**: Add genes for conditional behavior, variable-length programs
- **Cross-world transfer**: Test if entities evolved in one world can survive in another
- **Persistent storage**: Save/load evolution runs, export lineage data
- **3D visualization**: Three.js or WebGPU for richer visual representation
- **Network topology**: Graph-based worlds instead of grids
- **Coevolution**: Multiple species per world with predator/prey dynamics

## Tech Stack

- TypeScript + React + Vite
- WebGL for rendering
- Tailwind CSS for styling
- Zero runtime dependencies beyond React
