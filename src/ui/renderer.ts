/**
 * WebGL grid renderer.
 * Draws resources, entities, and signals on a canvas using GPU shaders.
 */

const VERT_SRC = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

const FRAG_SRC = `
  precision mediump float;
  varying vec2 v_texCoord;
  uniform sampler2D u_resources;
  uniform sampler2D u_entities;
  uniform sampler2D u_signals;
  uniform float u_time;

  void main() {
    vec4 res = texture2D(u_resources, v_texCoord);
    vec4 ent = texture2D(u_entities, v_texCoord);
    vec4 sig = texture2D(u_signals, v_texCoord);

    // Background: dark with resource tint
    vec3 bgColor = vec3(0.02, 0.03, 0.05);
    vec3 resColor = vec3(0.05, 0.25, 0.12) * res.r;

    // Signal glow
    vec3 sigColor = vec3(sig.r * 0.3, sig.g * 0.15, sig.b * 0.5);

    // Entity color based on genome data packed in the texture
    // r = energy, g = action type / 5, b = genome trait (aggression)
    vec3 entColor = vec3(0.0);
    if (ent.a > 0.0) {
      float energy = ent.r;
      float actionType = ent.g;
      float trait = ent.b;

      // Color by dominant trait with energy as brightness
      entColor = mix(
        vec3(0.2, 0.8, 0.4),   // cooperative (green)
        vec3(0.9, 0.2, 0.15),  // aggressive (red)
        trait
      ) * (0.5 + energy * 0.8);

      // Action highlight
      if (actionType > 0.55) { // SIGNAL or ATTACK
        entColor += vec3(0.15, 0.1, 0.3);
      }
    }

    vec3 color = bgColor + resColor + sigColor + entColor;
    gl_FragColor = vec4(color, 1.0);
  }
`;

export class WorldRenderer {
  private gl: WebGLRenderingContext;
  private program: WebGLProgram;
  private resourceTex: WebGLTexture;
  private entityTex: WebGLTexture;
  private signalTex: WebGLTexture;
  private gridW: number = 64;
  private gridH: number = 64;
  private timeLocation: WebGLUniformLocation | null;
  private time: number = 0;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl', { alpha: false, antialias: false });
    if (!gl) throw new Error('WebGL not supported');
    this.gl = gl;

    this.program = this.createProgram(VERT_SRC, FRAG_SRC);
    gl.useProgram(this.program);

    // Full-screen quad
    const posBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1, 1, 1, -1, 1, 1,
    ]), gl.STATIC_DRAW);

    const posLoc = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    const texBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, texBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      0, 1, 1, 1, 0, 0,
      0, 0, 1, 1, 1, 0,
    ]), gl.STATIC_DRAW);

    const texLoc = gl.getAttribLocation(this.program, 'a_texCoord');
    gl.enableVertexAttribArray(texLoc);
    gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);

    // Textures
    this.resourceTex = this.createTexture(0);
    this.entityTex = this.createTexture(1);
    this.signalTex = this.createTexture(2);

    gl.uniform1i(gl.getUniformLocation(this.program, 'u_resources'), 0);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_entities'), 1);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_signals'), 2);

    this.timeLocation = gl.getUniformLocation(this.program, 'u_time');
  }

  resize(width: number, height: number): void {
    const canvas = this.gl.canvas as HTMLCanvasElement;
    canvas.width = width;
    canvas.height = height;
    this.gl.viewport(0, 0, width, height);
  }

  update(state: {
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
  }): void {
    const { gl } = this;
    this.gridW = state.gridW;
    this.gridH = state.gridH;
    const w = state.gridW;
    const h = state.gridH;
    this.time++;

    // Resource texture (R channel = resource level)
    const resData = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      const v = Math.floor(state.resources[i] * 255);
      resData[i * 4] = v;
      resData[i * 4 + 1] = v;
      resData[i * 4 + 2] = v;
      resData[i * 4 + 3] = 255;
    }
    this.uploadTexture(this.resourceTex, 0, w, h, resData);

    // Entity texture
    const entData = new Uint8Array(w * h * 4);
    for (let e = 0; e < state.entityCount; e++) {
      const idx = (state.entityY[e] * w + state.entityX[e]) * 4;
      const genomeOffset = e * 16; // GENOME_LENGTH
      entData[idx] = Math.floor(state.entityEnergy[e] * 255);       // R = energy
      entData[idx + 1] = Math.floor((state.entityAction[e] / 5) * 255); // G = action
      entData[idx + 2] = Math.floor(state.entityGenomes[genomeOffset + 3] * 255); // B = aggression gene
      entData[idx + 3] = 255; // A = entity present
    }
    this.uploadTexture(this.entityTex, 1, w, h, entData);

    // Signal texture (RGB = up to 3 channels)
    const sigData = new Uint8Array(w * h * 4);
    const channels = Math.min(state.signalChannels, 3);
    for (let i = 0; i < w * h; i++) {
      for (let c = 0; c < channels; c++) {
        const val = Math.min(255, Math.floor(state.signals[i * state.signalChannels + c] * 255));
        sigData[i * 4 + c] = val;
      }
      sigData[i * 4 + 3] = 255;
    }
    this.uploadTexture(this.signalTex, 2, w, h, sigData);

    // Draw
    if (this.timeLocation) {
      gl.uniform1f(this.timeLocation, this.time * 0.01);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  private createProgram(vertSrc: string, fragSrc: string): WebGLProgram {
    const { gl } = this;
    const vert = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vert, vertSrc);
    gl.compileShader(vert);
    if (!gl.getShaderParameter(vert, gl.COMPILE_STATUS)) {
      throw new Error('Vertex shader: ' + gl.getShaderInfoLog(vert));
    }

    const frag = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(frag, fragSrc);
    gl.compileShader(frag);
    if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS)) {
      throw new Error('Fragment shader: ' + gl.getShaderInfoLog(frag));
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('Program link: ' + gl.getProgramInfoLog(prog));
    }
    return prog;
  }

  private createTexture(unit: number): WebGLTexture {
    const { gl } = this;
    const tex = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return tex;
  }

  private uploadTexture(tex: WebGLTexture, unit: number, w: number, h: number, data: Uint8Array): void {
    const { gl } = this;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
  }

  destroy(): void {
    const { gl } = this;
    gl.deleteTexture(this.resourceTex);
    gl.deleteTexture(this.entityTex);
    gl.deleteTexture(this.signalTex);
    gl.deleteProgram(this.program);
  }
}
