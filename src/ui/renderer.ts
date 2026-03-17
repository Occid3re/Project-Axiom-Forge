/**
 * WebGL renderer — cinematic 4-pass bloom pipeline.
 *
 * Pass 1  Scene data → FBO_scene  (full canvas resolution)
 * Pass 2  H-blur FBO_scene → FBO_blurA  (half resolution)
 * Pass 3  V-blur FBO_blurA → FBO_blurB  (half resolution)
 * Pass 4  Composite (FBO_scene + FBO_blurB) → canvas
 *
 * Data textures (resource / entity / signal / trail) use LINEAR filtering
 * so the 80×80 grid is bilinearly interpolated — no pixel squares.
 * Entities are rendered as 3×3 Gaussian splats on the CPU for visible dots.
 * Trail texture persists across frames and decays — movement leaves amber wakes.
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

// Pass 1 — world scene, renders to FBO at canvas resolution
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

  void main() {
    // World UV — zoomed/panned view, tiled with REPEAT wrap
    vec2 wuv = u_pan + (v_uv - 0.5) * u_zoom;

    vec4 res   = texture2D(u_res,   wuv);
    vec4 ent   = texture2D(u_ent,   wuv);
    vec4 sig   = texture2D(u_sig,   wuv);
    vec4 trail = texture2D(u_trail, wuv);

    float r = res.r;

    // Dark microscope field — near-black with faint cool tint
    vec3 color = vec3(0.003, 0.005, 0.009);

    // Nutrient medium — warm amber-green, slow gentle pulse
    float pulse = 0.88 + 0.12 * sin(u_time * 0.7 + wuv.x * 11.0 + wuv.y * 8.0);
    color += vec3(0.10, 0.18, 0.03) * r * pulse;
    color += vec3(0.18, 0.28, 0.04) * r * r * 1.8 * pulse;  // bright nutrient hotspot

    // Chemical signal fluorescence — three dye channels
    color += vec3(0.88, 0.06, 0.10) * sig.r * 1.1;   // red: danger / alarm pheromone
    color += vec3(0.00, 0.65, 0.50) * sig.g * 0.9;   // teal-green: vitality signal
    color += vec3(0.70, 0.05, 0.88) * sig.b * 0.8;   // magenta: marker / kin signal

    // Cell rendering — 4-color 2D palette (role × species)
    // ent.r = ring brightness, ent.g = species hue 0-1, ent.b = predator/role hue, ent.a = presence
    float presence = ent.a;
    if (presence > 0.01) {
      float ringInt  = ent.r;
      float speciesH = ent.g;  // species hue 0-1
      float role     = ent.b;  // predator/herbivore 0-1

      // 2D species color space:
      // role axis: herbivore (teal/lime) ↔ predator (orange/purple)
      // species axis: type A ↔ type B within each role
      vec3 c00 = vec3(0.02, 0.74, 0.92);  // blue-teal   (herbivore A)
      vec3 c01 = vec3(0.10, 0.92, 0.28);  // lime-green  (herbivore B)
      vec3 c10 = vec3(0.98, 0.38, 0.02);  // orange-red  (predator A)
      vec3 c11 = vec3(0.76, 0.04, 0.88);  // violet      (predator B)
      vec3 cellCol = mix(mix(c00, c01, speciesH), mix(c10, c11, speciesH), role);
      color += cellCol * ringInt * 1.5;
    }

    // Trail — faint green cytoplasm residue
    color += vec3(0.01, 0.22, 0.09) * trail.r * 0.38;

    // Vignette — always canvas-relative, independent of zoom/pan
    vec2 uvc = v_uv - 0.5;
    float vig = clamp(1.0 - dot(uvc, uvc) * 1.8, 0.0, 1.0);
    gl_FragColor = vec4(color * vig, 1.0);
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

// Pass 4 — composite scene + bloom with chromatic aberration
const COMP_FRAG = `
  precision highp float;
  varying vec2 v_uv;
  uniform sampler2D u_scene;
  uniform sampler2D u_bloom;
  uniform float u_fade;

  void main() {
    // Chromatic aberration — subtle radial shift of R and B
    vec2 uvc  = v_uv - 0.5;
    float dist = length(uvc);
    vec2 rOff  = uvc * dist * 0.008;
    vec2 bOff  = uvc * dist * 0.008;

    vec3 scene;
    scene.r = texture2D(u_scene, v_uv + rOff).r;
    scene.g = texture2D(u_scene, v_uv).g;
    scene.b = texture2D(u_scene, v_uv - bOff).b;

    vec3 bloom;
    bloom.r = texture2D(u_bloom, v_uv + rOff * 0.5).r;
    bloom.g = texture2D(u_bloom, v_uv).g;
    bloom.b = texture2D(u_bloom, v_uv - bOff * 0.5).b;

    // Screen blend: 1 - (1 - scene)(1 - bloom * strength)
    vec3 color = 1.0 - (1.0 - scene) * (1.0 - bloom * 0.7);

    // Subtle gamma lift to preserve shadow detail
    color = pow(max(color, vec3(0.0)), vec3(0.88));

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
    c = pow(max(c, vec3(0.0)), vec3(0.88));
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

  /** Upload new simulation data to GPU textures. Call on each server frame. */
  updateFrame(f: DecodedFrame) {
    const { gl } = this;
    const { gridW: W, gridH: H, entityCount } = f;

    // Allocate trail if grid changed; clear it on world reset (tick goes back to 0)
    if (W !== this.lastGridW || H !== this.lastGridH) {
      this.trailData = new Float32Array(W * H);
      this.lastGridW = W; this.lastGridH = H;
    }
    if (f.tick < this.lastTick) {
      // Display world restarted — wipe trail and trigger fade-in
      this.trailData!.fill(0);
      this.fadeValue = 0.0;
    }
    this.lastTick = f.tick;
    const trail = this.trailData!;

    // ── Trail decay ──────────────────────────────────────────────────────────
    for (let i = 0; i < trail.length; i++) trail[i] *= 0.92;

    // ── Resource texture ─────────────────────────────────────────────────────
    const resData = new Uint8Array(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      const v = f.resources[i];
      resData[i*4] = resData[i*4+1] = resData[i*4+2] = v;
      resData[i*4+3] = 255;
    }

    // ── Entity texture — biological cell ring pattern ────────────────────────
    // R = membrane ring intensity
    // G = species hue (0-255, same for all pixels of this entity)
    // B = predator/role hue
    // A = presence (non-zero anywhere the cell contributes, gates shader)
    const entData = new Uint8Array(W * H * 4);
    for (let e = 0; e < entityCount; e++) {
      const cx = f.entityX[e];
      const cy = f.entityY[e];
      const energy = f.entityEnergy[e] / 255;
      // entityAggression now carries combined predatorDrive (packed in simulation.ts)
      // Boost role toward red when currently attacking, toward lime when reproducing
      const baseRole = f.entityAggression[e] / 255;
      const act = f.entityAction[e];
      const actionShift = act === 5 ? 0.28 : act === 3 ? -0.22 : act === 4 ? 0.10 : 0;
      const role = Math.max(0, Math.min(1, baseRole + actionShift));
      const aggr = (role * 255) | 0; // reuse variable name for the packing below
      const speciesHue = f.entitySpeciesHue[e]; // 0-255

      // Write trail at entity center
      const ti = cy * W + cx;
      if (trail[ti] < energy) trail[ti] = energy;

      // Cell radius: 1.8–3.5 grid cells — small enough not to carpet the grid
      const cellR = 1.8 + energy * 1.7;
      const scanR = Math.ceil(cellR) + 1;

      for (let dy = -scanR; dy <= scanR; dy++) {
        for (let dx = -scanR; dx <= scanR; dx++) {
          // Toroidal wrap — matches REPEAT texture tiling when zoomed out
          const nx = ((cx + dx) % W + W) % W;
          const ny = ((cy + dy) % H + H) % H;

          const rr = Math.sqrt(dx * dx + dy * dy);
          if (rr > scanR) continue;

          let ringVal:     number; // brightness of ring/nucleus
          let presenceVal: number; // gate for shader

          if (rr < 0.7) {
            // Bright nucleus
            ringVal     = 0.95;
            presenceVal = 1.0;
          } else if (rr < cellR * 0.60) {
            // Dark cytoplasm
            ringVal     = 0.0;
            presenceVal = 1.0;
          } else if (rr <= cellR) {
            // Membrane ring — sin bump
            const t = (rr - cellR * 0.60) / (cellR * 0.40);
            ringVal     = 0.95 * Math.sin(Math.PI * t);
            presenceVal = 1.0;
          } else {
            // Outer glow only
            const fade = 1.0 - (rr - cellR) / 1.2;
            if (fade <= 0) continue;
            ringVal     = 0.12 * fade;
            presenceVal = fade * 0.5;
          }

          // Energy only dims outer glow; nucleus+ring stay visible so new worlds look correct
          if (presenceVal < 1.0) ringVal *= 0.40 + energy * 0.60;  // outer glow scales with energy

          const ci = (ny * W + nx) * 4;
          entData[ci]   = Math.max(entData[ci],   Math.min(255, (ringVal * 255) | 0));
          entData[ci+1] = Math.max(entData[ci+1], speciesHue);   // species hue — same for all pixels of this entity
          entData[ci+2] = Math.max(entData[ci+2], aggr);
          entData[ci+3] = Math.max(entData[ci+3], Math.min(255, (presenceVal * 255) | 0));
        }
      }
    }

    // ── Signal texture ───────────────────────────────────────────────────────
    const sigData = new Uint8Array(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      sigData[i*4]   = f.signals[i*3];
      sigData[i*4+1] = f.signals[i*3+1];
      sigData[i*4+2] = f.signals[i*3+2];
      sigData[i*4+3] = 255;
    }

    // ── Trail texture ────────────────────────────────────────────────────────
    const trailData8 = new Uint8Array(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      const v = Math.min(255, (trail[i] * 255) | 0);
      trailData8[i*4] = v;
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
    // REPEAT so zoomed-out views tile the world seamlessly (world is toroidal)
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
