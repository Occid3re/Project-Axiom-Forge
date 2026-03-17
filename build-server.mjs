/**
 * Build the server into a single bundled JS file.
 * Bundles server/index.ts + server/simulation.ts + src/engine/* into server/dist/server.cjs
 * No tsx or TypeScript runtime needed on the production server.
 */

import * as esbuild from 'esbuild';
import { mkdir } from 'fs/promises';

await mkdir('server/dist', { recursive: true });

const shared = {
  bundle: true,
  platform: 'node',
  target: 'node18',
  format: 'esm',
  external: ['express', 'socket.io'],
  sourcemap: false,
  minify: false,
  logLevel: 'info',
};

const [r1, r2] = await Promise.all([
  esbuild.build({ ...shared, entryPoints: ['server/index.ts'],      outfile: 'server/dist/server.mjs' }),
  esbuild.build({ ...shared, entryPoints: ['server/eval-worker.ts'], outfile: 'server/dist/eval-worker.mjs' }),
]);

const errors = [...r1.errors, ...r2.errors];
if (errors.length > 0) {
  console.error('Server build failed:', errors);
  process.exit(1);
}

console.log('Server bundled → server/dist/server.mjs + eval-worker.mjs');
