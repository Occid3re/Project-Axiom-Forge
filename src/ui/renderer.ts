/**
 * WebGL grid renderer — cinematic world visualization.
 * Agents glow by genome traits. Signals bloom. Resources pulse.
 */

const VERT_SRC = `
  attribute vec2 a_pos;
  attribute vec2 a_uv;
  varying vec2 v_uv;
  void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
  }
`;

const FRAG_SRC = `
  precision highp float;
  varying vec2 v_uv;

  uniform sampler2D u_res;
  uniform sampler2D u_ent;
  uniform sampler2D u_sig;
  uniform float u_time;
  uniform vec2 u_resolution;

  // Smooth noise
  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }
  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f*f*(3.0-2.0*f);
    return mix(mix(hash(i), hash(i+vec2(1,0)), f.x),
               mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y);
  }

  void main() {
    vec4 res = texture2D(u_res, v_uv);
    vec4 ent = texture2D(u_ent, v_uv);
    vec4 sig = texture2D(u_sig, v_uv);

    // Deep space background
    float n = noise(v_uv * 80.0 + u_time * 0.02);
    vec3 bg = vec3(0.006, 0.008, 0.014) + vec3(0.008, 0.006, 0.012) * n;

    // Resource field — bioluminescent ground
    float r = res.r;
    vec3 resGlow = vec3(0.02, 0.09, 0.06) * r * r;
    resGlow += vec3(0.0, 0.04, 0.03) * r * (0.5 + 0.5 * sin(u_time * 0.3 + v_uv.x * 20.0));

    // Signal bloom — multi-channel aura
    float s0 = sig.r;
    float s1 = sig.g;
    float s2 = sig.b;
    vec3 sigColor = vec3(0.15, 0.05, 0.5) * s0
                  + vec3(0.0, 0.3, 0.4) * s1
                  + vec3(0.4, 0.1, 0.2) * s2;
    sigColor *= 1.8;

    // Entity rendering
    vec3 entColor = vec3(0.0);
    float entPresent = ent.a;
    if (entPresent > 0.5) {
      float energy    = ent.r;
      float action    = ent.g;   // 0-1 mapped from ActionType/5
      float aggression = ent.b;  // genome trait

      // Base color by trait — cooperative=cyan, aggressive=red, balanced=purple
      vec3 lowTrait  = vec3(0.0, 0.7, 0.9);
      vec3 highTrait = vec3(1.0, 0.15, 0.1);
      vec3 baseColor = mix(lowTrait, highTrait, aggression);

      // Brightness from energy
      float bright = 0.3 + energy * 0.9;
      entColor = baseColor * bright;

      // Action flash: signaling=violet bloom, attacking=red spike
      float signaling = step(0.75, action) * (1.0 - step(0.99, action));
      float attacking = step(0.99, action);
      entColor += vec3(0.3, 0.0, 0.8) * signaling * 0.6;
      entColor += vec3(1.0, 0.1, 0.0) * attacking * 0.8;

      // Core hotspot
      entColor += baseColor * 0.5 * energy;
    }

    // Bloom: diffuse entity glow to neighbours (approximated by signal texture bleed)
    vec3 bloom = entColor * 0.25;

    // Vignette
    vec2 uvc = v_uv - 0.5;
    float vignette = 1.0 - dot(uvc, uvc) * 1.6;
    vignette = max(0.0, vignette);

    // Scanline (subtle)
    float scan = 0.97 + 0.03 * sin(v_uv.y * u_resolution.y * 2.0);

    vec3 color = (bg + resGlow + sigColor + entColor + bloom) * vignette * scan;

    // Subtle chromatic shift at edges
    float edgeDist = length(uvc);
    color.r *= 1.0 + edgeDist * 0.04;
    color.b *= 1.0 - edgeDist * 0.02;

    gl_FragColor = vec4(color, 1.0);
  }
`;

export interface VisualState {
  resources: Float32Array;
  signals: Float32Array;
  entityX: Int32Array;
  entityY: Int32Array;
  entityEnergy: Float32Array;
  entityAction: Uint8Array;
  entityGenomes: Float32Array;
  entityCount: number;
  gridW: number;
  gridH: number;
  signalChannels: number;
}

export class WorldRenderer {
  private gl: WebGLRenderingContext;
  private program: WebGLProgram;
  private resTex: WebGLTexture;
  private entTex: WebGLTexture;
  private sigTex: WebGLTexture;
  private timeLoc: WebGLUniformLocation | null;
  private resLoc: WebGLUniformLocation | null;
  private frame = 0;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl', {
      alpha: false,
      antialias: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL not supported');
    this.gl = gl;

    this.program = this.buildProgram(VERT_SRC, FRAG_SRC);
    gl.useProgram(this.program);

    // Full-screen quad
    this.bindBuffer(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1,  1, 1, -1,  1, 1,
    ]), 'a_pos', 2);
    this.bindBuffer(gl.ARRAY_BUFFER, new Float32Array([
      0, 1, 1, 1, 0, 0,
      0, 0, 1, 1, 1, 0,
    ]), 'a_uv', 2);

    this.resTex = this.makeTex(0, 'u_res');
    this.entTex = this.makeTex(1, 'u_ent');
    this.sigTex = this.makeTex(2, 'u_sig');

    this.timeLoc = gl.getUniformLocation(this.program, 'u_time');
    this.resLoc  = gl.getUniformLocation(this.program, 'u_resolution');
  }

  resize(w: number, h: number) {
    const canvas = this.gl.canvas as HTMLCanvasElement;
    canvas.width = w;
    canvas.height = h;
    this.gl.viewport(0, 0, w, h);
  }

  update(state: VisualState) {
    const { gl } = this;
    const { gridW: W, gridH: H, signalChannels: CH } = state;
    this.frame++;

    // Resource texture
    const resData = new Uint8Array(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      const v = (state.resources[i] * 255) | 0;
      resData[i * 4] = v;
      resData[i * 4 + 1] = v;
      resData[i * 4 + 2] = v;
      resData[i * 4 + 3] = 255;
    }
    this.upload(this.resTex, 0, W, H, resData);

    // Entity texture — each cell: R=energy G=action/5 B=aggression A=present
    const entData = new Uint8Array(W * H * 4);
    for (let e = 0; e < state.entityCount; e++) {
      const ci = (state.entityY[e] * W + state.entityX[e]) * 4;
      const g0 = e * 16; // GENOME_LENGTH = 16
      entData[ci]     = (state.entityEnergy[e] * 255) | 0;
      entData[ci + 1] = ((state.entityAction[e] / 5) * 255) | 0;
      entData[ci + 2] = (state.entityGenomes[g0 + 3] * 255) | 0; // aggression
      entData[ci + 3] = 255;
    }
    this.upload(this.entTex, 1, W, H, entData);

    // Signal texture — up to 3 channels → RGB
    const sigData = new Uint8Array(W * H * 4);
    const ch = Math.min(CH, 3);
    for (let i = 0; i < W * H; i++) {
      for (let c = 0; c < ch; c++) {
        sigData[i * 4 + c] = Math.min(255, (state.signals[i * CH + c] * 255) | 0);
      }
      sigData[i * 4 + 3] = 255;
    }
    this.upload(this.sigTex, 2, W, H, sigData);

    if (this.timeLoc) gl.uniform1f(this.timeLoc, this.frame * 0.016);
    const canvas = gl.canvas as HTMLCanvasElement;
    if (this.resLoc) gl.uniform2f(this.resLoc, canvas.width, canvas.height);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  destroy() {
    const { gl } = this;
    gl.deleteTexture(this.resTex);
    gl.deleteTexture(this.entTex);
    gl.deleteTexture(this.sigTex);
    gl.deleteProgram(this.program);
  }

  private bindBuffer(target: number, data: Float32Array, attr: string, size: number) {
    const { gl } = this;
    const buf = gl.createBuffer()!;
    gl.bindBuffer(target, buf);
    gl.bufferData(target, data, gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(this.program, attr);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }

  private makeTex(unit: number, uniform: string): WebGLTexture {
    const { gl } = this;
    const tex = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.uniform1i(gl.getUniformLocation(this.program, uniform), unit);
    return tex;
  }

  private upload(tex: WebGLTexture, unit: number, w: number, h: number, data: Uint8Array) {
    const { gl } = this;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
  }

  private buildProgram(vs: string, fs: string): WebGLProgram {
    const { gl } = this;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(s) ?? 'shader error');
      return s;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(prog) ?? 'link error');
    return prog;
  }
}
