/**
 * WebGL renderer — phase-contrast microscopy pipeline.
 *
 * Pass 1  Scene data → FBO_scene  (full canvas resolution)
 * Pass 2  H-blur FBO_scene → FBO_blurA  (half resolution)
 * Pass 3  V-blur FBO_blurA → FBO_blurB  (half resolution)
 * Pass 4  Composite (FBO_scene + FBO_blurB) → canvas
 *
 * Data textures (resource / entity / signal / trail) use LINEAR filtering
 * so the grid is bilinearly interpolated — no pixel squares.
 * Entities are rendered as rod-shaped bacterial cells via CPU splats.
 * Trail texture persists across frames and decays — movement leaves slime wakes.
 *
 * updateFrame(f)  — upload new data (call when a server frame arrives)
 * render(ms)      — run all 4 passes (call every requestAnimationFrame tick)
 */

import type { DecodedFrame } from '../engine/protocol';

// ── Shaders ──────────────────────────────────────────────────────────────────

const QUAD_VERT = `
  attribute vec2 a_pos;
  attribute vec2 a_uv;
  varying vec2 v_uv;
  void main() { gl_Position = vec4(a_pos, 0.0, 1.0); v_uv = a_uv; }
`;

// Pass 1 — phase-contrast microscopy scene
const SCENE_FRAG = `
  precision highp float;
  varying vec2 v_uv;
  uniform sampler2D u_res;
  uniform sampler2D u_ent;
  uniform sampler2D u_sig;
  uniform sampler2D u_trail;
  uniform float u_time;
  uniform vec2 u_pan;
  uniform float u_zoom;

  // Fast hash for film grain
  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }

  void main() {
    // World UV — zoomed/panned view, tiled with REPEAT wrap
    vec2 wuv = u_pan + (v_uv - 0.5) * u_zoom;

    vec4 res   = texture2D(u_res,   wuv);
    vec4 ent   = texture2D(u_ent,   wuv);
    vec4 sig   = texture2D(u_sig,   wuv);
    vec4 trail = texture2D(u_trail, wuv);

    float r = res.r;

    // Dark microscope field
    vec3 color = vec3(0.004, 0.006, 0.010);

    // Agar substrate — faint warm noise
    float substrate = hash(wuv * 180.0) * 0.006;
    color += vec3(substrate * 0.6, substrate * 0.8, substrate * 0.2);

    // Nutrient medium — warm amber-green, slow gentle pulse
    float pulse = 0.92 + 0.08 * sin(u_time * 0.4 + wuv.x * 9.0 + wuv.y * 7.0);
    color += vec3(0.06, 0.13, 0.02) * r * pulse;
    color += vec3(0.10, 0.20, 0.03) * r * r * 1.4 * pulse;

    // Chemical signal fluorescence — three dye channels (subtle, not lightning)
    float sigR = sig.r * sig.r;  // square for softer falloff — only bright when strong
    float sigG = sig.g * sig.g;
    float sigB = sig.b * sig.b;
    color += vec3(0.85, 0.08, 0.10) * sigR * 0.5;
    color += vec3(0.02, 0.65, 0.50) * sigG * 0.45;
    color += vec3(0.60, 0.06, 0.80) * sigB * 0.40;

    // Phase-contrast cell rendering
    // ent.r = membrane ring brightness, ent.g = species hue, ent.b = role, ent.a = presence
    float presence = ent.a;
    if (presence > 0.01) {
      float ringInt  = ent.r;
      float speciesH = ent.g;
      float role     = ent.b;

      // Biological color palette
      vec3 c00 = vec3(0.04, 0.68, 0.88);  // cyan-teal    (herbivore A)
      vec3 c01 = vec3(0.12, 0.88, 0.32);  // lime-green   (herbivore B)
      vec3 c10 = vec3(0.96, 0.42, 0.06);  // orange       (predator A)
      vec3 c11 = vec3(0.72, 0.06, 0.84);  // violet       (predator B)
      vec3 cellCol = mix(mix(c00, c01, speciesH), mix(c10, c11, speciesH), role);

      // Phase contrast: translucent body + bright membrane
      float body     = presence * (1.0 - ringInt) * 0.10;
      float membrane = ringInt * 1.4;
      color += cellCol * (body + membrane);

      // Phase-contrast halo — bright edge glow at presence boundary
      // presence*(1-presence) peaks at 0.5 = cell edge transition
      float halo = presence * (1.0 - presence) * 4.0;
      color += cellCol * halo * 0.35;
    }

    // Trail — faint slime residue
    color += vec3(0.01, 0.16, 0.05) * trail.r * 0.30;

    // Film grain — microscope camera sensor noise
    float grain = (hash(v_uv * 900.0 + u_time * 4.7) - 0.5) * 0.022;
    color += grain;

    // Circular vignette — microscope eyepiece
    float vigR = length(v_uv - 0.5) * 2.0;
    float vig = smoothstep(1.05, 0.60, vigR);
    gl_FragColor = vec4(max(color, vec3(0.0)) * vig, 1.0);
  }
`;

// Pass 2 & 3 — separable 9-tap Gaussian blur
const BLUR_FRAG = `
  precision highp float;
  varying vec2 v_uv;
  uniform sampler2D u_src;
  uniform vec2 u_dir;

  void main() {
    vec4 c = vec4(0.0);
    c += texture2D(u_src, v_uv)                * 0.227027;
    c += texture2D(u_src, v_uv + u_dir * 1.0) * 0.1945946;
    c += texture2D(u_src, v_uv - u_dir * 1.0) * 0.1945946;
    c += texture2D(u_src, v_uv + u_dir * 2.0) * 0.1216216;
    c += texture2D(u_src, v_uv - u_dir * 2.0) * 0.1216216;
    c += texture2D(u_src, v_uv + u_dir * 3.0) * 0.054054;
    c += texture2D(u_src, v_uv - u_dir * 3.0) * 0.054054;
    c += texture2D(u_src, v_uv + u_dir * 4.0) * 0.016216;
    c += texture2D(u_src, v_uv - u_dir * 4.0) * 0.016216;
    gl_FragColor = c;
  }
`;

// Pass 4 — composite with barrel distortion + chromatic aberration
const COMP_FRAG = `
  precision highp float;
  varying vec2 v_uv;
  uniform sampler2D u_scene;
  uniform sampler2D u_bloom;
  uniform float u_fade;

  void main() {
    vec2 uvc = v_uv - 0.5;
    float r2 = dot(uvc, uvc);

    // Barrel distortion — microscope optics
    vec2 d_uv = v_uv + uvc * r2 * 0.035;

    // Chromatic aberration — colour fringing from lens
    vec2 caOff = uvc * length(uvc) * 0.006;

    vec3 scene;
    scene.r = texture2D(u_scene, d_uv + caOff).r;
    scene.g = texture2D(u_scene, d_uv).g;
    scene.b = texture2D(u_scene, d_uv - caOff).b;

    vec3 bloom;
    bloom.r = texture2D(u_bloom, d_uv + caOff * 0.4).r;
    bloom.g = texture2D(u_bloom, d_uv).g;
    bloom.b = texture2D(u_bloom, d_uv - caOff * 0.4).b;

    // Screen blend
    vec3 color = 1.0 - (1.0 - scene) * (1.0 - bloom * 0.65);

    // Gamma
    color = pow(max(color, vec3(0.0)), vec3(0.90));

    gl_FragColor = vec4(color * u_fade, 1.0);
  }
`;

// Direct blit — used as fast-path when skipBloom is true
const BLIT_FRAG = `
  precision mediump float;
  varying vec2 v_uv;
  uniform sampler2D u_src;
  uniform float u_fade;
  void main() {
    vec3 c = texture2D(u_src, v_uv).rgb;
    c = pow(max(c, vec3(0.0)), vec3(0.90));
    gl_FragColor = vec4(c * u_fade, 1.0);
  }
`;

// ── FBO helper type ───────────────────────────────────────────────────────────

interface FBO { fbo: WebGLFramebuffer; tex: WebGLTexture; w: number; h: number; }

// ── Renderer ─────────────────────────────────────────────────────────────────

export class WorldRenderer {
  private gl: WebGLRenderingContext;
  private cw = 0; private ch = 0;

  // Programs
  private sceneP: WebGLProgram;
  private blurP:  WebGLProgram;
  private compP:  WebGLProgram;
  private blitP:  WebGLProgram;

  // Cached uniform locations
  private uTime:       WebGLUniformLocation | null = null;
  private uBlurDir:    WebGLUniformLocation | null = null;
  private uFadeComp:   WebGLUniformLocation | null = null;
  private uFadeBlit:   WebGLUniformLocation | null = null;
  private uPan:        WebGLUniformLocation | null = null;
  private uZoom:       WebGLUniformLocation | null = null;

  // View state — pan is world UV centre (0.5, 0.5 = centred), zoom >1 = zoomed out
  private viewPanX = 0.5;
  private viewPanY = 0.5;
  private viewZoom = 1.0;

  // Data textures — grid resolution, LINEAR filtered
  private resTex:   WebGLTexture;
  private entTex:   WebGLTexture;
  private sigTex:   WebGLTexture;
  private trailTex: WebGLTexture;

  // Trail state — persists across frames, cleared on world reset
  private trailData: Float32Array | null = null;
  private lastGridW = 0; private lastGridH = 0;
  private lastTick  = -1;

  // Cached CPU texture buffers — avoid allocation each frame (eliminates ~1MB/frame GC)
  private _resBuf:   Uint8Array | null = null;
  private _entBuf:   Uint8Array | null = null;
  private _sigBuf:   Uint8Array | null = null;
  private _trailBuf: Uint8Array | null = null;
  private _texBufN = 0;

  // Render targets
  private fboScene: FBO | null = null;
  private fboBlurA: FBO | null = null;
  private fboBlurB: FBO | null = null;

  private hasData = false;
  private lastRenderMs = 0;
  private smoothDelta = 16; // exponential moving average of frame delta ms
  private skipBloom = false;

  // Fade state — smooth transition when display world resets
  private fadeValue = 0.0; // start invisible, fade in on first frame

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl', {
      alpha: false, antialias: false, preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL unavailable');
    this.gl = gl;

    // Build programs (attribute locations pinned before link)
    this.sceneP = this.buildProgram(QUAD_VERT, SCENE_FRAG);
    this.blurP  = this.buildProgram(QUAD_VERT, BLUR_FRAG);
    this.compP  = this.buildProgram(QUAD_VERT, COMP_FRAG);
    this.blitP  = this.buildProgram(QUAD_VERT, BLIT_FRAG);
    gl.useProgram(this.blitP);
    gl.uniform1i(gl.getUniformLocation(this.blitP, 'u_src'), 0);

    // Bind quad once — attribute locations 0 (pos) and 1 (uv) are pinned
    this.bindQuad();

    // Set sampler uniforms (constant — don't change each frame)
    gl.useProgram(this.sceneP);
    gl.uniform1i(gl.getUniformLocation(this.sceneP, 'u_res'),   0);
    gl.uniform1i(gl.getUniformLocation(this.sceneP, 'u_ent'),   1);
    gl.uniform1i(gl.getUniformLocation(this.sceneP, 'u_sig'),   2);
    gl.uniform1i(gl.getUniformLocation(this.sceneP, 'u_trail'), 3);
    this.uTime = gl.getUniformLocation(this.sceneP, 'u_time');
    this.uPan  = gl.getUniformLocation(this.sceneP, 'u_pan');
    this.uZoom = gl.getUniformLocation(this.sceneP, 'u_zoom');
    // Set defaults
    gl.uniform2f(this.uPan, 0.5, 0.5);
    gl.uniform1f(this.uZoom, 1.0);

    gl.useProgram(this.blurP);
    gl.uniform1i(gl.getUniformLocation(this.blurP, 'u_src'), 0);
    this.uBlurDir = gl.getUniformLocation(this.blurP, 'u_dir');

    gl.useProgram(this.compP);
    gl.uniform1i(gl.getUniformLocation(this.compP, 'u_scene'), 0);
    gl.uniform1i(gl.getUniformLocation(this.compP, 'u_bloom'), 1);
    this.uFadeComp = gl.getUniformLocation(this.compP, 'u_fade');
    this.uFadeBlit = gl.getUniformLocation(this.blitP,  'u_fade');

    // Create data textures (empty until first updateFrame)
    this.resTex   = this.makeLinearTex();
    this.entTex   = this.makeLinearTex();
    this.sigTex   = this.makeLinearTex();
    this.trailTex = this.makeLinearTex();
  }

  /** Update viewport — called from WorldView on wheel / drag. */
  setView(panX: number, panY: number, zoom: number) {
    this.viewPanX = panX;
    this.viewPanY = panY;
    this.viewZoom = zoom;
  }

  resize(w: number, h: number) {
    const c = this.gl.canvas as HTMLCanvasElement;
    c.width = w; c.height = h;
    this.gl.viewport(0, 0, w, h);
    this.cw = w; this.ch = h;
    this.rebuildFBOs();
  }

  /** Ensure CPU texture buffers exist for current grid size. */
  private ensureTexBufs(n: number) {
    if (this._texBufN === n) return;
    this._resBuf   = new Uint8Array(n);
    this._entBuf   = new Uint8Array(n);
    this._sigBuf   = new Uint8Array(n);
    this._trailBuf = new Uint8Array(n);
    this._texBufN  = n;
  }

  /** Upload new simulation data to GPU textures. Call on each server frame. */
  updateFrame(f: DecodedFrame) {
    const { gl } = this;
    const { gridW: W, gridH: H, entityCount } = f;
    const cells = W * H;

    // Allocate trail if grid changed; clear it on world reset (tick goes back to 0)
    if (W !== this.lastGridW || H !== this.lastGridH) {
      this.trailData = new Float32Array(cells);
      this.lastGridW = W; this.lastGridH = H;
    }
    if (f.tick < this.lastTick) {
      // Display world restarted — wipe trail and trigger fade-in
      this.trailData!.fill(0);
      this.fadeValue = 0.0;
    }
    this.lastTick = f.tick;
    const trail = this.trailData!;

    // Reuse CPU buffers — eliminates ~1MB/frame GC pressure
    this.ensureTexBufs(cells * 4);
    const resData    = this._resBuf!;
    const entData    = this._entBuf!;
    const sigData    = this._sigBuf!;
    const trailData8 = this._trailBuf!;

    // ── Trail decay ──────────────────────────────────────────────────────────
    for (let i = 0; i < trail.length; i++) trail[i] *= 0.92;

    // ── Resource texture ─────────────────────────────────────────────────────
    for (let i = 0; i < cells; i++) {
      const v = f.resources[i];
      resData[i*4] = resData[i*4+1] = resData[i*4+2] = v;
      resData[i*4+3] = 255;
    }

    // ── Entity texture — evolving bacterial morphology ─────────────────────
    // R = membrane ring intensity, G = species hue, B = role hue, A = presence
    // Morphology evolves with genome complexity:
    //   Low complexity (early): round coccus, thin membrane, no internal structure
    //   High complexity (evolved): elongated rod, thick ruffled membrane,
    //     visible organelles, flagella-like extensions
    entData.fill(0); // clear — entities are max-blended onto clean slate

    for (let e = 0; e < entityCount; e++) {
      const cx = f.entityX[e];
      const cy = f.entityY[e];
      const energy = f.entityEnergy[e] / 255;
      const baseRole = f.entityAggression[e] / 255;
      const act = f.entityAction[e];
      const actionShift = act === 5 ? 0.28 : act === 3 ? -0.22 : act === 4 ? 0.10 : 0;
      const role = Math.max(0, Math.min(1, baseRole + actionShift));
      const roleU8 = (role * 255) | 0;
      const speciesHue = f.entitySpeciesHue[e]; // 0-255
      const complexity = (f.entityComplexity?.[e] ?? 80) / 255; // 0-1, default ~early
      const motility   = (f.entityMotility?.[e] ?? 128) / 255;  // 0-1

      // Write trail — motile entities leave stronger trails
      const ti = cy * W + cx;
      const trailStr = energy * (0.5 + motility * 0.5);
      if (trail[ti] < trailStr) trail[ti] = trailStr;

      // ── Body plan classification ─────────────────────────────────────
      // 5 distinct shapes from genome traits: coccus, bacillus, vibrio, amoeba, dividing
      const aggression = f.entityAggression[e] / 255;

      let plan: number;       // 0=coccus, 1=bacillus, 2=vibrio, 3=amoeba, 4=dividing
      let planAspect: number;
      let curvature = 0;      // vibrio bend strength
      let lobeCount = 0;      // amoeba pseudopod count
      let lobeAmp = 0;        // amoeba pseudopod depth
      let pinch = 0;          // dividing constriction

      if (energy > 0.7) {
        plan = 4;             // DIVIDING — high energy, about to split
        planAspect = 1.4 + complexity * 0.6;
        pinch = 0.35 + (energy - 0.7) * 1.5;
      } else if (aggression > 0.55) {
        plan = 2;             // VIBRIO — predators are comma-shaped
        planAspect = 1.3 + complexity * 0.8;
        curvature = 0.25 + aggression * 0.35;
      } else if (complexity > 0.5 && motility < 0.4) {
        plan = 3;             // AMOEBA — evolved sessile → star-shaped pseudopods
        planAspect = 1.0 + complexity * 0.3;
        lobeCount = 3 + Math.floor(complexity * 4); // 3-7 lobes
        lobeAmp = 0.25 + complexity * 0.25;
      } else if (complexity < 0.3 && aggression < 0.35) {
        plan = 0;             // COCCUS — simple round cells
        planAspect = 1.0 + complexity * 0.15;
      } else {
        plan = 1;             // BACILLUS — elongated rod (default)
        planAspect = 1.0 + complexity * 1.2;
      }

      // Orientation: species-dependent angle
      const angle = (speciesHue / 255) * Math.PI;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      // Cell size: slight growth with complexity + energy
      const cellR = 1.5 + energy * 1.2 + complexity * 0.5;
      const shapeExtra = plan === 3 ? (lobeAmp * cellR) : plan === 2 ? (curvature * 2) : 0;
      const scanR = Math.ceil(cellR + shapeExtra) + 2;

      // Membrane thickness: thin (early) → thick ruffled (evolved)
      const membraneWidth = 0.30 + complexity * 0.20;
      const membraneStart = 1.0 - membraneWidth;

      // Internal structure: organelle count and visibility
      const organelleStr = complexity * 0.12;
      const organelleFreq = 1.5 + complexity * 3.0;

      // Flagella: only on motile elongated forms (bacillus/vibrio)
      const flagella = (plan === 1 || plan === 2) ? motility * complexity : 0;

      for (let dy = -scanR; dy <= scanR; dy++) {
        for (let dx = -scanR; dx <= scanR; dx++) {
          const nx = cx + dx;
          const ny = cy + dy;
          if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

          // Rotated + elongated local coordinates
          let ldx = dx * cosA + dy * sinA;
          let ldy = (-dx * sinA + dy * cosA) * planAspect;
          let rr: number;

          if (plan === 2) {
            // VIBRIO: bend x-axis quadratically → comma/crescent shape
            ldx = ldx + curvature * ldy * ldy;
            rr = Math.sqrt(ldx * ldx + ldy * ldy);
          } else if (plan === 3) {
            // AMOEBA: star-shaped via angular radius modulation
            const pixelAngle = Math.atan2(ldy, ldx);
            const rawR = Math.sqrt(ldx * ldx + ldy * ldy);
            const lobePhase = (speciesHue / 255) * Math.PI * 2;
            const modulation = 1.0 + lobeAmp * Math.cos(pixelAngle * lobeCount + lobePhase);
            rr = rawR / modulation; // lobes "pull outward"
          } else if (plan === 4) {
            // DIVIDING: hourglass pinch at center
            const pinchFactor = 1.0 - pinch * Math.exp(-ldx * ldx * 2.0);
            const pinchedLdy = ldy / Math.max(0.3, pinchFactor);
            rr = Math.sqrt(ldx * ldx + pinchedLdy * pinchedLdy);
          } else {
            // COCCUS / BACILLUS: standard ellipse
            rr = Math.sqrt(ldx * ldx + ldy * ldy);
          }

          // Flagella: extend presence along the long axis beyond the cell body
          let flagellaVal = 0;
          if (flagella > 0.15 && Math.abs(ldy) < 0.6) {
            const poleR = Math.abs(ldx) - cellR;
            if (poleR > 0 && poleR < flagella * 3.0) {
              const wave = 0.5 + 0.5 * Math.sin(poleR * 4.0 + ldy * 8.0);
              flagellaVal = (1.0 - poleR / (flagella * 3.0)) * wave * 0.35 * flagella;
            }
          }

          if (rr > cellR + 1.5 && flagellaVal <= 0) continue;

          let ringVal:     number;
          let presenceVal: number;

          if (rr < 0.6) {
            // Nucleus — brighter with complexity
            ringVal     = 0.85 + complexity * 0.10;
            presenceVal = 1.0;
          } else if (rr < cellR * membraneStart) {
            // Cytoplasm — internal organelles appear with evolution
            const organelles = organelleStr *
              (0.5 + 0.5 * Math.sin(ldx * organelleFreq) * Math.cos(ldy * organelleFreq * 0.7));
            ringVal     = 0.05 + organelles;
            presenceVal = 1.0;
          } else if (rr <= cellR) {
            // Membrane ring — thicker and brighter with complexity
            const t = (rr - cellR * membraneStart) / (cellR * membraneWidth);
            const base = Math.sin(Math.PI * t);
            // Evolved membranes get ruffles (high-frequency wobble)
            const ruffle = complexity > 0.3
              ? 1.0 + complexity * 0.15 * Math.sin(Math.atan2(ldy, ldx) * (6 + complexity * 10))
              : 1.0;
            ringVal     = (0.75 + complexity * 0.20) * base * ruffle;
            presenceVal = 1.0;
          } else if (flagellaVal > 0) {
            // Flagella region
            ringVal     = flagellaVal;
            presenceVal = flagellaVal * 0.8;
          } else {
            // Phase-contrast outer halo
            const fade = 1.0 - (rr - cellR) / 1.5;
            if (fade <= 0) continue;
            ringVal     = (0.10 + complexity * 0.08) * fade;
            presenceVal = fade * 0.55;
          }

          // Energy dims outer glow only
          if (presenceVal < 1.0) ringVal *= 0.35 + energy * 0.65;

          const ci = (ny * W + nx) * 4;
          entData[ci]   = Math.max(entData[ci],   Math.min(255, (ringVal * 255) | 0));
          entData[ci+1] = Math.max(entData[ci+1], speciesHue);
          entData[ci+2] = Math.max(entData[ci+2], roleU8);
          entData[ci+3] = Math.max(entData[ci+3], Math.min(255, (presenceVal * 255) | 0));
        }
      }
    }

    // ── Signal texture ───────────────────────────────────────────────────────
    for (let i = 0; i < cells; i++) {
      sigData[i*4]   = f.signals[i*3];
      sigData[i*4+1] = f.signals[i*3+1];
      sigData[i*4+2] = f.signals[i*3+2];
      sigData[i*4+3] = 255;
    }

    // ── Trail texture ────────────────────────────────────────────────────────
    for (let i = 0; i < cells; i++) {
      const v = Math.min(255, (trail[i] * 255) | 0);
      trailData8[i*4]   = v;
      trailData8[i*4+1] = 0;
      trailData8[i*4+2] = 0;
      trailData8[i*4+3] = 255;
    }

    // Upload all four textures
    this.uploadTex(this.resTex,   W, H, resData);
    this.uploadTex(this.entTex,   W, H, entData);
    this.uploadTex(this.sigTex,   W, H, sigData);
    this.uploadTex(this.trailTex, W, H, trailData8);

    this.hasData = true;
  }

  /** Execute the 4-pass render pipeline. Call every requestAnimationFrame. */
  render(ms: number) {
    const { gl, cw, ch } = this;
    if (!this.hasData || !this.fboScene || !this.fboBlurA || !this.fboBlurB) return;

    // Adaptive quality: skip bloom on slow devices
    const delta = ms - this.lastRenderMs;
    if (this.lastRenderMs > 0 && delta < 200) { // ignore first frame and tab-hidden spikes
      this.smoothDelta = this.smoothDelta * 0.97 + delta * 0.03;
    }
    this.lastRenderMs = ms;
    this.skipBloom = this.smoothDelta > 22; // < ~45fps

    // Advance fade — 0 → 1 over ~60 frames (~1 second)
    if (this.fadeValue < 1.0) {
      this.fadeValue = Math.min(1.0, this.fadeValue + 0.017);
    }

    const t = ms * 0.001;
    const bw = this.fboBlurA.w;
    const bh = this.fboBlurA.h;

    // ── Pass 1: Scene → FBO_scene ────────────────────────────────────────────
    gl.useProgram(this.sceneP);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboScene.fbo);
    gl.viewport(0, 0, cw, ch);

    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.resTex);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.entTex);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.sigTex);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.trailTex);
    gl.uniform1f(this.uTime, t);
    gl.uniform2f(this.uPan, this.viewPanX, this.viewPanY);
    gl.uniform1f(this.uZoom, this.viewZoom);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    if (this.skipBloom) {
      // Fast path: blit scene directly to canvas
      gl.useProgram(this.blitP);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, cw, ch);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.fboScene.tex);
      gl.uniform1f(this.uFadeBlit, this.fadeValue);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      return;
    }

    // ── Pass 2: H-blur FBO_scene → FBO_blurA (half-res) ─────────────────────
    gl.useProgram(this.blurP);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboBlurA.fbo);
    gl.viewport(0, 0, bw, bh);

    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.fboScene.tex);
    gl.uniform2f(this.uBlurDir, 2.0 / cw, 0.0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // ── Pass 3: V-blur FBO_blurA → FBO_blurB (half-res) ─────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboBlurB.fbo);

    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.fboBlurA.tex);
    gl.uniform2f(this.uBlurDir, 0.0, 2.0 / ch);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // ── Pass 4: Composite → canvas ───────────────────────────────────────────
    gl.useProgram(this.compP);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, cw, ch);

    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.fboScene.tex);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.fboBlurB.tex);
    gl.uniform1f(this.uFadeComp, this.fadeValue);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  destroy() {
    const { gl } = this;
    gl.deleteProgram(this.sceneP);
    gl.deleteProgram(this.blurP);
    gl.deleteProgram(this.compP);
    gl.deleteProgram(this.blitP);
    for (const t of [this.resTex, this.entTex, this.sigTex, this.trailTex]) {
      gl.deleteTexture(t);
    }
    this.deleteFBOs();
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  private rebuildFBOs() {
    this.deleteFBOs();
    const { cw: w, ch: h } = this;
    if (w <= 0 || h <= 0) return;
    this.fboScene = this.makeFBO(w, h);
    this.fboBlurA = this.makeFBO(Math.ceil(w / 2), Math.ceil(h / 2));
    this.fboBlurB = this.makeFBO(Math.ceil(w / 2), Math.ceil(h / 2));
  }

  private deleteFBOs() {
    const { gl } = this;
    for (const f of [this.fboScene, this.fboBlurA, this.fboBlurB]) {
      if (f) { gl.deleteFramebuffer(f.fbo); gl.deleteTexture(f.tex); }
    }
    this.fboScene = null; this.fboBlurA = null; this.fboBlurB = null;
  }

  private makeFBO(w: number, h: number): FBO {
    const { gl } = this;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, tex, w, h };
  }

  private makeLinearTex(): WebGLTexture {
    const { gl } = this;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // REPEAT — world is toroidal, seamless wrap when panning past edges
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    return tex;
  }

  private uploadTex(tex: WebGLTexture, w: number, h: number, data: Uint8Array) {
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
  }

  private bindQuad() {
    const { gl } = this;
    const mk = (data: number[]) => {
      const buf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
      return buf;
    };
    const pos = mk([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]);
    const uv  = mk([0,1, 1,1, 0,0, 0,0, 1,1, 1,0]);

    // Attribute locations 0 and 1 are pinned via buildProgram → same for all progs
    gl.bindBuffer(gl.ARRAY_BUFFER, pos);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, uv);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
  }

  private buildProgram(vs: string, fs: string): WebGLProgram {
    const { gl } = this;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        throw new Error(`Shader compile: ${gl.getShaderInfoLog(s)}`);
      }
      return s;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
    // Pin attribute locations so they are consistent across all three programs
    gl.bindAttribLocation(prog, 0, 'a_pos');
    gl.bindAttribLocation(prog, 1, 'a_uv');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`Program link: ${gl.getProgramInfoLog(prog)}`);
    }
    return prog;
  }
}
