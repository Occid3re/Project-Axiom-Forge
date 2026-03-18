# Hybrid Social Perception + Stigmergic Memory — Implementation Plan

## Goal

Give entities the ability to perceive neighbors' social properties AND deposit/absorb
learned behavioral representations from the environment. This creates a channel for
cumulative cultural transmission — knowledge that persists across generations without
genetic encoding.

## Architecture Changes

### Neural Network: 4→10 inputs, 8→10 hidden, 6→8 outputs

```
Inputs (10):
  [0] localResource        — resource at own cell (existing)
  [1] energyNorm           — own energy / energyCap (existing)
  [2] entityDensity        — neighbor count / 24 (existing)
  [3] signalStrength       — mean signal in radius-2 (existing)
  [4] nearestKinEnergy     — energy of closest same-species neighbor (0 if none)
  [5] nearestThreatDist    — 1/distance to closest different-species neighbor (0 if none)
  [6] kinRatio             — same-species / total neighbors (0.5 if alone)
  [7] glyphStrength        — magnitude of glyph vector at current cell
  [8] glyphAffinity        — dot(own hidden state, local glyph) normalized
  [9] ageNorm              — age / 500 capped at 1.0

Outputs (8):
  [0] IDLE       [1] MOVE       [2] EAT         [3] REPRODUCE
  [4] SIGNAL     [5] ATTACK     [6] DEPOSIT      [7] ABSORB

Genome: W1(10×10=100) + W2(10×8=80) = 180 weights
Hidden: 10 units (Elman recurrent, stored in memory[0..9])
```

### Glyph Grid Layer

```
Float32Array glyphs[gridW × gridH × 4]   — 4-channel persistent marks
Decay: laws.glyphDecay (0.990–0.999)      — half-life 693–6931 ticks
Memory cost: 64×64×4×4 = 65KB (eval), 256×256×4×4 = 1MB (display)
```

### New Actions

- **DEPOSIT**: Write compressed hidden state (10→4 channels, sum pairs) into glyph grid.
  Cost: `laws.depositCost` (0–0.03). Blend: `glyph = glyph×0.3 + deposit×0.7`.
- **ABSORB**: Read glyph, expand to 10 floats, blend into hidden state.
  Rate: `laws.absorbRate` (0–0.3). Cost: `laws.absorbCost` (0–0.02).

### Species Similarity (for kin recognition)

Cosine similarity of W2 SIGNAL+EAT columns (16 weights) — same columns used for
species hue visualization. Threshold: `laws.kinThreshold` (0.6–0.95).

### New WorldLaws (5 additions → 38 total)

| Parameter | Range | Starter | Purpose |
|-----------|-------|---------|---------|
| glyphDecay | 0.990–0.999 | 0.996 | Glyph persistence |
| depositCost | 0.0–0.03 | 0.01 | Energy to write glyph |
| absorbCost | 0.0–0.02 | 0.005 | Energy to read glyph |
| absorbRate | 0.0–0.3 | 0.1 | Hidden state blend factor |
| kinThreshold | 0.6–0.95 | 0.8 | Species similarity cutoff |

### New Scoring Metrics (2 additions → 12 total)

- **stigmergicUse**: sqrt(depositRate × absorbRate) × balance. Rewards worlds where
  both deposit and absorb happen regularly and in balance.
- **socialDifferentiation**: |kinAttackRate − strangerAttackRate|. Rewards selective
  behavior toward kin vs non-kin.

### Binary Protocol Update

Add glyph magnitude layer (W×H bytes) between poison and entity data.
Total frame overhead: +4096 bytes (64×64) or +65536 bytes (256×256).

---

## Implementation Phases

### Phase 1: Foundation (constants, genome, entity pool, laws)
- [ ] `constants.ts`: NN_INPUTS=10, NN_HIDDEN=10, NN_OUTPUTS=8, GENOME_LENGTH=180
- [ ] `constants.ts`: ActionType += DEPOSIT=6, ABSORB=7
- [ ] `constants.ts`: Add GLYPH_CHANNELS=4
- [ ] `entity-pool.ts`: Update Xavier init for 10×10 + 10×8 dimensions
- [ ] `world-laws.ts`: Add 5 new laws, update interface + ranges + random/starter/mutate

### Phase 2: World engine (glyph grid, social perception, new actions)
- [ ] `world.ts` constructor: Allocate glyphs Float32Array
- [ ] `world.ts` step(): Add decayGlyphs() call
- [ ] `world.ts` decideAction(): Expand to 10 inputs with social + stigmergic computation
- [ ] `world.ts`: Add genomeSimilarity() helper
- [ ] `world.ts`: Implement executeDeposit() and executeAbsorb()
- [ ] `world.ts`: Update Elman hidden state handling for 10 units
- [ ] `world.ts` snapshot(): Add deposits/absorbs counters
- [ ] `world.ts` getVisualState(): Expose glyphs

### Phase 3: Scoring
- [ ] `scoring.ts`: Add stigmergicUse and socialDifferentiation metrics
- [ ] `scoring.ts`: Update WorldScores interface and scoreWorld()

### Phase 4: Server (protocol, frame packing, eval worker)
- [ ] `simulation.ts`: STATE_VERSION → 8
- [ ] `simulation.ts` packFrame(): Add glyph layer, update genome derivation for new sizes
- [ ] `simulation.ts` EVAL_CONFIG.scoreWeights: Add new metric weights
- [ ] `simulation.ts` MetaBroadcast: Update sampleGenome size comment
- [ ] `eval-worker.ts`: Update DEFAULT_SCORE_WEIGHTS
- [ ] `protocol.ts`: Add glyphs to DecodedFrame, update decoder offsets
- [ ] `protocol.ts`: Update ServerMeta scores interface

### Phase 5: Client visualization
- [ ] `renderer.ts`: Add glyph texture channel, gold/amber shader glow
- [ ] `StatsPanel.tsx`: Add Stigmergy and Social score bars
- [ ] `App.tsx`: Add new scores to desktop sidebar + mobile scroll
- [ ] `WorldLawsView.tsx`: Add Stigmergy section
- [ ] `NeuralNetView.tsx`: Update for 10-10-8 architecture
- [ ] `EmergenceLadder.tsx`: Update for new score thresholds

### Phase 6: Deploy
- [ ] Build, commit, push
- [ ] Deploy to lostuplink-prod
- [ ] pm2 stop → rm state.json → pm2 start
- [ ] Update AGENTS.md with new architecture docs
