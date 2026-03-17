/**
 * WebGL grid renderer — cinematic world visualization.
 * Now accepts decoded binary frames from the server.
 */

import type { DecodedFrame } from '../engine/protocol';

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

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }
  float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    f = f*f*(3.0-2.0*f);
    return mix(mix(hash(i), hash(i+vec2(1,0)), f.x),
               mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y);
  }

  void main() {
    vec4 res = texture2D(u_res, v_uv);
    vec4 ent = texture2D(u_ent, v_uv);
    vec4 sig = texture2D(u_sig, v_uv);

    float n = noise(v_uv * 80.0 + u_time * 0.02);
    vec3 bg = vec3(0.006, 0.008, 0.014) + vec3(0.008,0.006,0.012)*n;

    float r = res.r;
    vec3 resGlow = vec3(0.02, 0.09, 0.06) * r * r;
    resGlow += vec3(0.0, 0.04, 0.03) * r * (0.5 + 0.5*sin(u_time*0.3 + v_uv.x*20.0));

    vec3 sigColor = vec3(0.15,0.05,0.5)*sig.r + vec3(0.0,0.3,0.4)*sig.g + vec3(0.4,0.1,0.2)*sig.b;
    sigColor *= 1.8;

    vec3 entColor = vec3(0.0);
    if (ent.a > 0.5) {
      float energy = ent.r; float action = ent.g; float aggr = ent.b;
      vec3 base = mix(vec3(0.0,0.7,0.9), vec3(1.0,0.15,0.1), aggr) * (0.3 + energy*0.9);
      entColor = base;
      entColor += vec3(0.3,0.0,0.8) * step(0.75,action) * (1.0-step(0.99,action)) * 0.6;
      entColor += vec3(1.0,0.1,0.0) * step(0.99,action) * 0.8;
      entColor += base * 0.5 * energy;
    }

    vec2 uvc = v_uv - 0.5;
    float vignette = max(0.0, 1.0 - dot(uvc,uvc)*1.6);
    float scan = 0.97 + 0.03*sin(v_uv.y * u_resolution.y * 2.0);
    vec3 color = (bg + resGlow + sigColor + entColor + entColor*0.25) * vignette * scan;
    float ed = length(uvc);
    color.r *= 1.0 + ed*0.04; color.b *= 1.0 - ed*0.02;
    gl_FragColor = vec4(color, 1.0);
  }
`;

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
    const gl = canvas.getContext('webgl', { alpha: false, antialias: false, powerPreference: 'high-performance' });
    if (!gl) throw new Error('WebGL not supported');
    this.gl = gl;
    this.program = this.buildProgram(VERT_SRC, FRAG_SRC);
    gl.useProgram(this.program);
    this.bindQuad();
    this.resTex = this.makeTex(0, 'u_res');
    this.entTex = this.makeTex(1, 'u_ent');
    this.sigTex = this.makeTex(2, 'u_sig');
    this.timeLoc = gl.getUniformLocation(this.program, 'u_time');
    this.resLoc  = gl.getUniformLocation(this.program, 'u_resolution');
  }

  resize(w: number, h: number) {
    const c = this.gl.canvas as HTMLCanvasElement;
    c.width = w; c.height = h;
    this.gl.viewport(0, 0, w, h);
  }

  updateFromFrame(f: DecodedFrame) {
    const { gl } = this;
    const { gridW: W, gridH: H, entityCount } = f;
    this.frame++;

    // Resource RGBA
    const resData = new Uint8Array(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      resData[i*4] = resData[i*4+1] = resData[i*4+2] = f.resources[i];
      resData[i*4+3] = 255;
    }
    this.upload(this.resTex, 0, W, H, resData);

    // Entity RGBA
    const entData = new Uint8Array(W * H * 4);
    for (let e = 0; e < entityCount; e++) {
      const ci = (f.entityY[e] * W + f.entityX[e]) * 4;
      entData[ci]   = f.entityEnergy[e];
      entData[ci+1] = Math.min(255, (f.entityAction[e] / 5 * 255) | 0);
      entData[ci+2] = f.entityAggression[e];
      entData[ci+3] = 255;
    }
    this.upload(this.entTex, 1, W, H, entData);

    // Signal RGBA (3 channels packed)
    const sigData = new Uint8Array(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      sigData[i*4]   = f.signals[i*3];
      sigData[i*4+1] = f.signals[i*3+1];
      sigData[i*4+2] = f.signals[i*3+2];
      sigData[i*4+3] = 255;
    }
    this.upload(this.sigTex, 2, W, H, sigData);

    if (this.timeLoc) gl.uniform1f(this.timeLoc, this.frame * 0.016);
    const c = gl.canvas as HTMLCanvasElement;
    if (this.resLoc) gl.uniform2f(this.resLoc, c.width, c.height);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  destroy() {
    const { gl } = this;
    [this.resTex, this.entTex, this.sigTex].forEach(t => gl.deleteTexture(t));
    gl.deleteProgram(this.program);
  }

  private bindQuad() {
    const { gl } = this;
    const bind = (data: Float32Array, attr: string, size: number) => {
      const buf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(this.program, attr);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    };
    bind(new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), 'a_pos', 2);
    bind(new Float32Array([0,1, 1,1, 0,0, 0,0, 1,1, 1,0]), 'a_uv', 2);
  }

  private makeTex(unit: number, uniform: string): WebGLTexture {
    const { gl } = this;
    const tex = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T].forEach(p => gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE));
    [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER].forEach(p => gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST));
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
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)!);
      return s;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog)!);
    return prog;
  }
}
