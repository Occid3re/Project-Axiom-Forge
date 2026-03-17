/**
 * Build the server into a single bundled JS file.
 * Bundles server/index.ts + server/simulation.ts + src/engine/* into server/dist/server.cjs
 * No tsx or TypeScript runtime needed on the production server.
 */

import * as esbuild from 'esbuild';
import { mkdir } from 'fs/promises';

await mkdir('server/dist', { recursive: true });

const result = await esbuild.build({
  entryPoints: ['server/index.ts'],
  bundle: true,
  platform: 'node',
  target: 'node18',
  format: 'esm',
  outfile: 'server/dist/server.mjs',
  external: ['express', 'socket.io'], // keep npm deps external
  sourcemap: false,
  minify: false,
  logLevel: 'info',
});

if (result.errors.length > 0) {
  console.error('Server build failed:', result.errors);
  process.exit(1);
}

console.log('Server bundled → server/dist/server.mjs');
