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
 * Entities are rendered as continuous evolved morphologies via CPU splats.
 * Trail texture persists across frames and decays — movement leaves slime wakes.
 *
 * updateFrame(f)  — upload new data (call when a server frame arrives)
 * render(ms)      — run all 4 passes (call every requestAnimationFrame tick)
 */

import type { DecodedEntityFrame, DecodedFieldFrame } from '../engine/protocol';

type CombinedFrame = DecodedEntityFrame & DecodedFieldFrame;

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}

const COLOR_GRADING = {
  entityBrightness: 1.14,
  entitySaturation: 0.86,
  backgroundBrightness: 0.96,
  glowStrength: 0.46,
  contrast: 1.04,
  paletteMix: 0.24,
} as const;

const SCENE_TUNING = {
  sceneSaturation: 0.80,
  bloomStrength: 0.40,
  maxBrightness: 0.90,
} as const;

const BIO_PALETTE = {
  backgroundBase: [0.018, 0.024, 0.028],
  backgroundNoise: [0.010, 0.012, 0.008],
  resourceLow: [0.045, 0.090, 0.040],
  resourceHigh: [0.085, 0.140, 0.070],
  poisonPrimary: [0.180, 0.055, 0.085],
  poisonSecondary: [0.080, 0.020, 0.040],
  glyphPrimary: [0.220, 0.170, 0.090],
  glyphSecondary: [0.100, 0.075, 0.040],
  signalA: [0.220, 0.120, 0.090],
  signalB: [0.120, 0.220, 0.170],
  signalC: [0.210, 0.160, 0.240],
  trail: [0.060, 0.110, 0.080],
  herbivoreA: [0.180, 0.420, 0.400],
  herbivoreB: [0.340, 0.470, 0.270],
  predatorA: [0.520, 0.340, 0.180],
  predatorB: [0.420, 0.280, 0.430],
  cellShadow: [0.080, 0.100, 0.120],
} as const;

function glNum(value: number) {
  return value.toFixed(3);
}

function glVec3(value: readonly [number, number, number]) {
  return `vec3(${value.map(glNum).join(', ')})`;
}

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
  uniform float u_specimen;

  // Fast hash for film grain
  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }

  vec3 applySaturation(vec3 color, float saturation) {
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    return mix(vec3(luma), color, saturation);
  }

  vec3 applyContrast(vec3 color, float contrast) {
    return (color - 0.5) * contrast + 0.5;
  }

  vec3 softClamp(vec3 color, float maxValue) {
    return min(color, vec3(maxValue));
  }

  void main() {
    // World UV — zoomed/panned view, tiled with REPEAT wrap
    vec2 wuv = u_pan + (v_uv - 0.5) * u_zoom;
    float deepZoom = 1.0 - smoothstep(0.12, 0.32, u_zoom);

    vec4 res   = texture2D(u_res,   wuv);
    vec4 ent   = texture2D(u_ent,   wuv);
    vec4 sig   = texture2D(u_sig,   wuv);
    vec4 trail = texture2D(u_trail, wuv);

    float r = res.r;       // resource concentration
    float poison = res.g;  // toxin concentration
    float glyph = res.b;   // stigmergic memory magnitude

    vec3 darkBg = ${glVec3(BIO_PALETTE.backgroundBase)};
    float substrate = hash(wuv * 180.0) * 0.006;
    darkBg += ${glVec3(BIO_PALETTE.backgroundNoise)} * substrate * 3.0;

    float pulse = 0.92 + 0.08 * sin(u_time * 0.4 + wuv.x * 9.0 + wuv.y * 7.0);
    darkBg += ${glVec3(BIO_PALETTE.resourceLow)} * r * pulse;
    darkBg += ${glVec3(BIO_PALETTE.resourceHigh)} * r * r * 1.15 * pulse;
    darkBg += vec3(0.010, 0.012, 0.010) * deepZoom;

    float poisonPulse = 0.85 + 0.15 * sin(u_time * 1.2 + wuv.x * 13.0 + wuv.y * 11.0);
    darkBg += ${glVec3(BIO_PALETTE.poisonPrimary)} * poison * poison * poisonPulse * 0.45;
    darkBg += ${glVec3(BIO_PALETTE.poisonSecondary)} * poison * 0.22;

    float glyphPulse = 0.90 + 0.10 * sin(u_time * 0.8 + wuv.x * 7.0 + wuv.y * 5.0);
    darkBg += ${glVec3(BIO_PALETTE.glyphPrimary)} * glyph * glyph * glyphPulse * 0.34;
    darkBg += ${glVec3(BIO_PALETTE.glyphSecondary)} * glyph * 0.18;

    float sigR = sig.r * sig.r;
    float sigG = sig.g * sig.g;
    float sigB = sig.b * sig.b;
    float fieldFade = 1.0 - deepZoom * 0.72;
    darkBg += ${glVec3(BIO_PALETTE.signalA)} * sigR * 0.22 * fieldFade;
    darkBg += ${glVec3(BIO_PALETTE.signalB)} * sigG * 0.24 * fieldFade;
    darkBg += ${glVec3(BIO_PALETTE.signalC)} * sigB * 0.18 * fieldFade;
    darkBg += vec3(0.010, 0.012, 0.014) * deepZoom * (sigR + sigG + sigB) * 0.12;

    float slideMask = smoothstep(0.02, 0.12, v_uv.x) * smoothstep(0.02, 0.12, v_uv.y)
      * smoothstep(0.02, 0.12, 1.0 - v_uv.x) * smoothstep(0.02, 0.12, 1.0 - v_uv.y);
    float glassNoise = hash(v_uv * 900.0 + vec2(7.0, 13.0)) * 0.012;
    float specimenLight = 0.92 + 0.08 * smoothstep(0.95, 0.15, length(v_uv - 0.5) * 1.6);
    vec3 brightBg = vec3(0.78, 0.76, 0.71) * specimenLight;
    brightBg += vec3(0.015, 0.012, 0.008) * sin(u_time * 0.15 + wuv.y * 12.0);
    brightBg += vec3(glassNoise);
    brightBg += vec3(0.025, 0.020, 0.012) * r;
    brightBg -= vec3(0.12, 0.05, 0.07) * poison * 0.45;
    brightBg += vec3(0.09, 0.07, 0.03) * glyph * 0.15;
    brightBg += vec3(0.015, 0.03, 0.025) * (sigR + sigG + sigB) * 0.14;
    float slideShadow = (1.0 - slideMask) * 0.30;
    brightBg -= vec3(slideShadow * 0.85, slideShadow * 0.80, slideShadow * 0.72);
    float glassEdge = (1.0 - slideMask) * 0.45;
    brightBg += vec3(glassEdge * 0.015, glassEdge * 0.02, glassEdge * 0.025);

    darkBg *= ${glNum(COLOR_GRADING.backgroundBrightness)};
    vec3 color = mix(darkBg, brightBg, u_specimen);

    // Phase-contrast cell rendering
    // ent.r = membrane ring brightness, ent.g = species hue, ent.b = role, ent.a = presence
    float presence = ent.a;
    if (presence > 0.01) {
      float ringInt  = ent.r;
      float speciesH = ent.g;
      float role     = ent.b;

      // Biological color palette
      vec3 c00 = ${glVec3(BIO_PALETTE.herbivoreA)};
      vec3 c01 = ${glVec3(BIO_PALETTE.herbivoreB)};
      vec3 c10 = ${glVec3(BIO_PALETTE.predatorA)};
      vec3 c11 = ${glVec3(BIO_PALETTE.predatorB)};
      vec3 baseCellCol = mix(mix(c00, c01, speciesH), mix(c10, c11, speciesH), role);
      vec3 richerCellCol = mix(baseCellCol, baseCellCol * vec3(1.08, 1.03, 1.10) + vec3(0.015, 0.012, 0.014), ${glNum(COLOR_GRADING.paletteMix)});
      vec3 cellCol = applySaturation(richerCellCol, ${glNum(COLOR_GRADING.entitySaturation)});

      vec3 brightCell = mix(${glVec3(BIO_PALETTE.cellShadow)}, cellCol * 0.92 + vec3(0.024), 0.72);
      float body = presence * (1.0 - ringInt) * mix(0.14, 0.30, u_specimen) * (1.0 + deepZoom * 0.10);
      float membrane = ringInt * mix(1.34, 1.04, u_specimen) * (1.0 + deepZoom * 0.14);
      body *= ${glNum(COLOR_GRADING.entityBrightness)};
      membrane *= ${glNum(COLOR_GRADING.entityBrightness)};
      color += mix(cellCol, brightCell, u_specimen) * (body + membrane);

      float specimenShadow = presence * (0.18 + ringInt * 0.12);
      color -= vec3(specimenShadow * 0.28 * u_specimen);

      float halo = presence * (1.0 - presence) * 4.0;
      color += mix(cellCol * halo * (0.20 + deepZoom * 0.07), brightCell * halo * 0.040, u_specimen) * ${glNum(COLOR_GRADING.glowStrength)};
    }

    color += ${glVec3(BIO_PALETTE.trail)} * trail.r * 0.13 * (1.0 - u_specimen) * fieldFade;

    float grain = (hash(v_uv * 900.0 + u_time * 4.7) - 0.5) * 0.022;
    color += grain * mix(1.0, 0.35, u_specimen);
    color = applyContrast(color, ${glNum(COLOR_GRADING.contrast)});
    color = applySaturation(color, ${glNum(SCENE_TUNING.sceneSaturation)});
    color = softClamp(color, ${glNum(SCENE_TUNING.maxBrightness)});

    float scaleBarMask =
      step(0.12, v_uv.x) * step(v_uv.x, 0.34) *
      step(0.90, v_uv.y) * step(v_uv.y, 0.92);
    color = mix(color, vec3(0.08, 0.09, 0.10), scaleBarMask * u_specimen);

    float vigR = length(v_uv - 0.5) * 2.0;
    float dishVig = smoothstep(1.05, 0.60, vigR);
    float specimenVig = 0.96 + slideMask * 0.04;
    float vig = mix(dishVig, specimenVig, u_specimen);
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
  uniform float u_specimen;
  uniform vec2 u_sceneTexel;

  void main() {
    vec2 uvc = v_uv - 0.5;
    float r2 = dot(uvc, uvc);

    vec2 d_uv = v_uv + uvc * r2 * mix(0.035, 0.006, u_specimen);
    vec2 caOff = uvc * length(uvc) * mix(0.006, 0.001, u_specimen);

    vec3 scene;
    scene.r = texture2D(u_scene, d_uv + caOff).r;
    scene.g = texture2D(u_scene, d_uv).g;
    scene.b = texture2D(u_scene, d_uv - caOff).b;

    vec3 sceneN = texture2D(u_scene, d_uv + vec2(0.0, u_sceneTexel.y)).rgb;
    vec3 sceneS = texture2D(u_scene, d_uv - vec2(0.0, u_sceneTexel.y)).rgb;
    vec3 sceneE = texture2D(u_scene, d_uv + vec2(u_sceneTexel.x, 0.0)).rgb;
    vec3 sceneW = texture2D(u_scene, d_uv - vec2(u_sceneTexel.x, 0.0)).rgb;
    vec3 sceneBlur = (sceneN + sceneS + sceneE + sceneW) * 0.25;
    vec3 sharpenedScene = max(scene + (scene - sceneBlur) * 0.22, vec3(0.0));

    vec3 bloom;
    bloom.r = texture2D(u_bloom, d_uv + caOff * 0.4).r;
    bloom.g = texture2D(u_bloom, d_uv).g;
    bloom.b = texture2D(u_bloom, d_uv - caOff * 0.4).b;

    vec3 screenColor = 1.0 - (1.0 - sharpenedScene) * (1.0 - bloom * ${glNum(SCENE_TUNING.bloomStrength)});
    vec3 specimenColor = sharpenedScene + bloom * 0.12;
    vec3 color = mix(screenColor, specimenColor, u_specimen);
    color = pow(max(color, vec3(0.0)), mix(vec3(${glNum(1 / COLOR_GRADING.contrast)}), vec3(1.00), u_specimen));
    color = min(color, vec3(${glNum(SCENE_TUNING.maxBrightness)}));

    gl_FragColor = vec4(color * u_fade, 1.0);
  }
`;

// Direct blit — used as fast-path when skipBloom is true
const BLIT_FRAG = `
  precision mediump float;
  varying vec2 v_uv;
  uniform sampler2D u_src;
  uniform float u_fade;
  uniform float u_specimen;
  void main() {
    vec3 c = texture2D(u_src, v_uv).rgb;
    c = pow(max(c, vec3(0.0)), mix(vec3(${glNum(1 / COLOR_GRADING.contrast)}), vec3(1.00), u_specimen));
    c = min(c, vec3(${glNum(SCENE_TUNING.maxBrightness)}));
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
  private uSpecimenScene: WebGLUniformLocation | null = null;
  private uSpecimenComp:  WebGLUniformLocation | null = null;
  private uSpecimenBlit:  WebGLUniformLocation | null = null;
  private uSceneTexel:    WebGLUniformLocation | null = null;

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
  private _fieldBufN = 0;
  private _entBufN = 0;

  // Render targets
  private fboScene: FBO | null = null;
  private fboBlurA: FBO | null = null;
  private fboBlurB: FBO | null = null;

  private hasData = false;
  private latestFieldFrame: DecodedFieldFrame | null = null;
  private latestEntityFrame: DecodedEntityFrame | null = null;
  private lastRenderMs = 0;
  private smoothDelta = 16; // exponential moving average of frame delta ms
  private skipBloom = false;
  private specimenBlend = 0;

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
    this.uSpecimenScene = gl.getUniformLocation(this.sceneP, 'u_specimen');
    // Set defaults
    gl.uniform2f(this.uPan, 0.5, 0.5);
    gl.uniform1f(this.uZoom, 1.0);
    gl.uniform1f(this.uSpecimenScene, 0.0);

    gl.useProgram(this.blurP);
    gl.uniform1i(gl.getUniformLocation(this.blurP, 'u_src'), 0);
    this.uBlurDir = gl.getUniformLocation(this.blurP, 'u_dir');

    gl.useProgram(this.compP);
    gl.uniform1i(gl.getUniformLocation(this.compP, 'u_scene'), 0);
    gl.uniform1i(gl.getUniformLocation(this.compP, 'u_bloom'), 1);
    this.uFadeComp = gl.getUniformLocation(this.compP, 'u_fade');
    this.uSpecimenComp = gl.getUniformLocation(this.compP, 'u_specimen');
    this.uSceneTexel = gl.getUniformLocation(this.compP, 'u_sceneTexel');
    this.uFadeBlit = gl.getUniformLocation(this.blitP,  'u_fade');
    this.uSpecimenBlit = gl.getUniformLocation(this.blitP, 'u_specimen');
    gl.uniform1f(this.uSpecimenComp, 0.0);
    gl.uniform2f(this.uSceneTexel, 1 / 1024, 1 / 1024);
    gl.useProgram(this.blitP);
    gl.uniform1f(this.uSpecimenBlit, 0.0);

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

  /** Ensure CPU texture buffers exist for current field/entity sizes. */
  private ensureTexBufs(fieldBytes: number, entityBytes: number) {
    if (this._fieldBufN !== fieldBytes) {
      this._resBuf   = new Uint8Array(fieldBytes);
      this._sigBuf   = new Uint8Array(fieldBytes);
      this._trailBuf = new Uint8Array(fieldBytes);
      this._fieldBufN = fieldBytes;
    }
    if (this._entBufN !== entityBytes) {
      this._entBuf = new Uint8Array(entityBytes);
      this._entBufN = entityBytes;
    }
  }

  updateFieldFrame(f: DecodedFieldFrame) {
    this.latestFieldFrame = f;
    this.flushCombinedFrame();
  }

  updateEntityFrame(f: DecodedEntityFrame) {
    this.latestEntityFrame = f;
    this.flushCombinedFrame();
  }

  private flushCombinedFrame() {
    if (!this.latestFieldFrame || !this.latestEntityFrame) return;
    this.updateFrame({
      ...this.latestFieldFrame,
      ...this.latestEntityFrame,
    });
  }

  /** Upload new simulation data to GPU textures. */
  private updateFrame(f: CombinedFrame) {
    const { gridW: W, gridH: H, entityCount } = f;
    const cells = W * H;
    const detailBoost = (f as CombinedFrame & { renderScale?: number }).renderScale ?? 1;
    const veryCrowded = entityCount >= 2800;
    const crowded = entityCount >= 1800;
    const morphologyQuality = detailBoost >= 2.2 && !crowded ? 2 : detailBoost >= 1.3 && !veryCrowded ? 1 : 0;
    const atlasScale = morphologyQuality >= 2 ? 4 : morphologyQuality >= 1 ? 2 : 1;
    const renderScale = 1 + ((morphologyQuality === 0 ? Math.min(detailBoost, 1.4) : detailBoost) - 1) * 0.42;
    this.specimenBlend = 0;
    const entW = W * atlasScale;
    const entH = H * atlasScale;
    const entCells = entW * entH;

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
    this.ensureTexBufs(cells * 4, entCells * 4);
    const resData    = this._resBuf!;
    const entData    = this._entBuf!;
    const sigData    = this._sigBuf!;
    const trailData8 = this._trailBuf!;

    // ── Trail decay ──────────────────────────────────────────────────────────
    for (let i = 0; i < trail.length; i++) trail[i] *= 0.92;

    // ── Resource texture ─────────────────────────────────────────────────────
    // R = resource, G = poison, B = glyph magnitude (stigmergic memory)
    for (let i = 0; i < cells; i++) {
      const v = f.resources[i];
      const p = f.poison[i];
      const g = f.glyphs?.[i] ?? 0;
      resData[i*4]   = v;            // R: resource
      resData[i*4+1] = p;            // G: poison concentration
      resData[i*4+2] = g;            // B: glyph magnitude
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
      const cx = Math.round((f.entityX[e] / Math.max(1, W - 1)) * Math.max(1, entW - 1));
      const cy = Math.round((f.entityY[e] / Math.max(1, H - 1)) * Math.max(1, entH - 1));
      const energy = f.entityEnergy[e] / 255;
      const baseRole = f.entityAggression[e] / 255;
      const act = f.entityAction[e];
      const actionShift = act === 5 ? 0.28 : act === 3 ? -0.22 : act === 4 ? 0.10 : 0;
      const role = Math.max(0, Math.min(1, baseRole + actionShift));
      const roleU8 = (role * 255) | 0;
      const speciesHue = f.entitySpeciesHue[e]; // 0-255
      const complexity = (f.entityComplexity?.[e] ?? 80) / 255; // 0-1, default ~early
      const motility   = (f.entityMotility?.[e] ?? 128) / 255;  // 0-1
      const sizeMul    = Math.max(0.7, Math.min(1.95, ((f.entitySize?.[e] ?? 102) / 255) * 2));

      // Write trail — motile entities leave stronger trails
      const ti = f.entityY[e] * W + f.entityX[e];
      const trailStr = energy * (0.5 + motility * 0.5);
      if (trail[ti] < trailStr) trail[ti] = trailStr;

      // ── Body plan classification ─────────────────────────────────────
      // Shape is synthesized from evolved behavioural traits instead of fixed plans.
      const aggression = f.entityAggression[e] / 255;

      const hue01 = speciesHue / 255;
      const phase = hue01 * Math.PI * 2 + role * Math.PI;
      const aspect = morphologyQuality === 0
        ? 0.85 + 1.15 * clamp01(0.55 * motility + 0.25 * complexity + 0.20 * energy)
        : 0.70 + 2.05 * clamp01(0.45 * motility + 0.35 * complexity + 0.20 * energy);
      const curvature = (aggression * 2 - 1) * (morphologyQuality === 0 ? 0.08 + 0.18 * complexity : 0.10 + 0.42 * complexity);
      const waveAmp = morphologyQuality === 0 ? 0 : motility * (0.08 + 0.34 * (1 - energy * 0.4));
      const waveFreq = 1.4 + hue01 * 3.8 + aggression * 1.3;
      const lobeAmp = morphologyQuality >= 2 ? complexity * (0.05 + 0.18 * (1 - motility)) : 0;
      const lobeFreq = 2.0 + hue01 * 4.0;
      const taper = morphologyQuality === 0
        ? 0.06 + 0.20 * (0.55 * aggression + 0.45 * complexity)
        : 0.08 + 0.40 * (0.55 * aggression + 0.45 * complexity);
      const pinch = clamp01((energy - 0.48) * (morphologyQuality === 0 ? 0.7 : 1.1) + complexity * (morphologyQuality === 0 ? 0.08 : 0.18));
      const skew = (hue01 * 2 - 1) * (morphologyQuality === 0 ? 0.04 + 0.08 * motility : 0.08 + 0.16 * motility);
      const contourRipple = morphologyQuality >= 2 ? 0.03 + complexity * 0.10 + motility * 0.04 : 0;
      const membraneRuffle = 4 + complexity * 8 + motility * 3;
      /*

      if (energy > 0.7) {
        plan = 4;             // DIVIDING — high energy, about to split
        planAspect = 1.4 + complexity * 0.6;
        pinch = 0.35 + (energy - 0.7) * 1.5;
      } else if (motility > 0.7 && complexity > 0.5) {
        plan = 5;             // SPIRILLUM — fast evolved swimmers, corkscrew body
        planAspect = 1.8 + complexity * 0.8;
        waveAmp = 0.4 + motility * 0.4;
        waveFreq = 2.5 + complexity * 2.0;
      } else if (motility > 0.75 && complexity < 0.4) {
        plan = 9;             // SPIROCHETE — thin undulating wave, primitive fast movers
        planAspect = 2.5 + motility * 1.0;
        waveAmp = 0.6 + motility * 0.3;
        waveFreq = 3.0;
      } else if (aggression > 0.55) {
        plan = 2;             // VIBRIO — predators are comma-shaped
        planAspect = 1.3 + complexity * 0.8;
        curvature = 0.25 + aggression * 0.35;
      } else if (complexity > 0.7 && motility < 0.35) {
        plan = 7;             // FILAMENTOUS — very evolved sessile, chain of linked cells
        planAspect = 2.5 + complexity * 1.5;
        chainPinches = 2 + Math.floor(complexity * 3); // 2-5 segments
      } else if (complexity > 0.5 && motility < 0.4) {
        plan = 3;             // AMOEBA — evolved sessile → star-shaped pseudopods
        planAspect = 1.0 + complexity * 0.3;
        lobeCount = 3 + Math.floor(complexity * 4);
        lobeAmp = 0.25 + complexity * 0.25;
      } else if (aggression > 0.35 && aggression <= 0.55 && complexity > 0.3) {
        plan = 8;             // FUSIFORM — spindle/diamond, moderate predators
        planAspect = 1.5 + complexity * 0.7;
        taper = 0.4 + aggression * 0.4;
      } else if (energy > 0.45 && energy <= 0.7 && complexity > 0.3 && motility < 0.5) {
        plan = 6;             // DIPLOCOCCUS — paired cells, cooperative/about to divide
        planAspect = 1.0 + complexity * 0.2;
        pinch = 0.5 + complexity * 0.3;
      } else if (complexity < 0.3 && aggression < 0.35) {
        plan = 0;             // COCCUS — simple round cells
        planAspect = 1.0 + complexity * 0.15;
      } else {
        plan = 1;             // BACILLUS — elongated rod (default)
        planAspect = 1.0 + complexity * 1.2;
      }

      */
      // Orientation: species-dependent angle
      const angle = (speciesHue / 255) * Math.PI;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      // Cell size: slight growth with complexity + energy
      const cellR = (1.5 + energy * 1.2 + complexity * 0.5) * renderScale * sizeMul;
      const shapeExtra = cellR * ((morphologyQuality === 0 ? 0.25 : 0.45 + waveAmp * 0.9 + lobeAmp * 0.8 + Math.abs(curvature) * 0.6) + Math.max(0, sizeMul - 1) * 0.22);
      const scanR = Math.ceil(cellR + shapeExtra) + 3;

      // Membrane thickness: thin (early) → thick ruffled (evolved)
      const membraneWidth = 0.28 + complexity * 0.20 + Math.max(0, sizeMul - 1) * 0.05;
      const membraneStart = 1.0 - membraneWidth;

      // Internal structure: organelle count and visibility
      const organelleStr = morphologyQuality >= 2 ? complexity * 0.12 : 0;
      const organelleFreq = 1.5 + complexity * 3.0;

      // Flagella emerge continuously from motility and elongation, not a fixed plan.
      const flagella = morphologyQuality >= 2
        ? clamp01((motility - 0.22) * 1.25) * clamp01((aspect - 1.0) / 1.8) * (0.4 + complexity * 0.6)
        : 0;

      for (let dy = -scanR; dy <= scanR; dy++) {
        for (let dx = -scanR; dx <= scanR; dx++) {
          const nx = ((cx + dx) % entW + entW) % entW;
          const ny = ((cy + dy) % entH + entH) % entH;

          // Continuous local-coordinate distortion synthesized from evolved traits.
          let ldx = dx * cosA + dy * sinA;
          let ldy = (-dx * sinA + dy * cosA) * aspect;
          ldx += curvature * ldy * ldy / Math.max(1.0, cellR * 1.4);
          if (morphologyQuality >= 1) {
            ldy -= waveAmp * Math.sin(ldx * waveFreq * 0.55 + phase);
          }
          const skewedLdx = ldx + skew * ldy;
          const taperedLdy = ldy * (1.0 + taper * (skewedLdx * skewedLdx) / Math.max(1.0, cellR * cellR));
          const rawR = Math.sqrt(skewedLdx * skewedLdx + taperedLdy * taperedLdy);
          let pixelAngle = 0;
          let rr = rawR;
          if (morphologyQuality >= 1) {
            const pinchScale = 1.0 - pinch * Math.exp(-skewedLdx * skewedLdx * (1.2 / (cellR * cellR + 0.01)));
            rr = rawR / Math.max(0.45, pinchScale);
          }
          if (morphologyQuality >= 2) {
            pixelAngle = Math.atan2(taperedLdy, skewedLdx);
            const lobeMod = 1.0 + lobeAmp * Math.cos(pixelAngle * lobeFreq + phase);
            const pinchScale = 1.0 - pinch * Math.exp(-skewedLdx * skewedLdx * (1.2 / (cellR * cellR + 0.01)));
            const ripple = 1.0 + contourRipple * Math.sin(pixelAngle * membraneRuffle + phase * 0.7);
            rr = rawR / Math.max(0.45, lobeMod * pinchScale * ripple);
          }

          /*
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
            rr = rawR / modulation;
          } else if (plan === 4 || plan === 6) {
            // DIVIDING / DIPLOCOCCUS: hourglass pinch at center
            const pinchFactor = 1.0 - pinch * Math.exp(-ldx * ldx * 2.0);
            const pinchedLdy = ldy / Math.max(0.3, pinchFactor);
            rr = Math.sqrt(ldx * ldx + pinchedLdy * pinchedLdy);
          } else if (plan === 5 || plan === 9) {
            // SPIRILLUM / SPIROCHETE: sinusoidal wave body
            // Displace perpendicular to long axis with sine wave
            const waveLdy = ldy - waveAmp * Math.sin(ldx * waveFreq);
            rr = Math.sqrt(ldx * ldx + waveLdy * waveLdy);
          } else if (plan === 7) {
            // FILAMENTOUS: chain of linked cells — periodic radius pinches
            const chainFreq = Math.PI * chainPinches / (cellR * 2);
            const chainMod = 1.0 - 0.35 * (0.5 + 0.5 * Math.cos(ldx * chainFreq));
            const modLdy = ldy / Math.max(0.4, chainMod);
            rr = Math.sqrt(ldx * ldx + modLdy * modLdy);
          } else if (plan === 8) {
            // FUSIFORM: diamond/spindle — tapered ends
            const taperFactor = 1.0 + taper * (ldx * ldx) / (cellR * cellR);
            const taperedLdy = ldy * taperFactor;
            rr = Math.sqrt(ldx * ldx + taperedLdy * taperedLdy);
          } else {
            // COCCUS / BACILLUS: standard ellipse
            rr = Math.sqrt(ldx * ldx + ldy * ldy);
          }

          */
          // Flagella: extend presence along the long axis beyond the cell body
          let flagellaVal = 0;
          if (flagella > 0.08 && Math.abs(taperedLdy) < 0.85) {
            const poleR = Math.abs(skewedLdx) - cellR;
            const flagellaReach = cellR * (0.8 + flagella * 1.8);
            if (poleR > 0 && poleR < flagellaReach) {
              const wave = 0.5 + 0.5 * Math.sin(poleR * (3.4 + waveFreq * 0.25) + taperedLdy * (5.0 + motility * 4.0));
              flagellaVal = (1.0 - poleR / flagellaReach) * wave * (0.18 + flagella * 0.32);
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
            const organelles = organelleStr > 0
              ? organelleStr * (0.5 + 0.5 * Math.sin(skewedLdx * organelleFreq + phase) * Math.cos(taperedLdy * organelleFreq * 0.7))
              : 0;
            ringVal     = 0.05 + organelles;
            presenceVal = 1.0;
          } else if (rr <= cellR) {
            // Membrane ring — thicker and brighter with complexity
            const t = (rr - cellR * membraneStart) / (cellR * membraneWidth);
            const base = Math.sin(Math.PI * t);
            // Evolved membranes get ruffles (high-frequency wobble)
            const ruffle = morphologyQuality >= 2 && complexity > 0.3
              ? 1.0 + complexity * 0.15 * Math.sin(pixelAngle * (6 + complexity * 10) + phase * 0.5)
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

          const ci = (ny * entW + nx) * 4;
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
    this.uploadTex(this.entTex,   entW, entH, entData);
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
    gl.uniform1f(this.uSpecimenScene, this.specimenBlend);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    if (this.skipBloom) {
      // Fast path: blit scene directly to canvas
      gl.useProgram(this.blitP);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, cw, ch);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.fboScene.tex);
      gl.uniform1f(this.uFadeBlit, this.fadeValue);
      gl.uniform1f(this.uSpecimenBlit, this.specimenBlend);
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
    gl.uniform1f(this.uSpecimenComp, this.specimenBlend);
    gl.uniform2f(this.uSceneTexel, 1 / Math.max(1, cw), 1 / Math.max(1, ch));
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
