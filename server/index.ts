/**
 * LostUplink: Axiom Forge — server entry point
 * - Eval loop:    500 ticks/sec via setInterval (finds best laws fast)
 * - Display loop: 30fps via setInterval (slow-motion best world for viewers)
 * - Socket.IO:    broadcasts display state to all clients
 */

import express from 'express';
import { createServer } from 'http';
import { Server as SocketIO } from 'socket.io';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { SimulationController } from './simulation.ts';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PORT = parseInt(process.env.PORT ?? '3001', 10);

// ── HTTP + Socket.IO ────────────────────────────────────────────────────────

const app = express();
const httpServer = createServer(app);
const io = new SocketIO(httpServer, {
  cors: { origin: '*' },
  maxHttpBufferSize: 512 * 1024, // 512KB max frame
});

// In the bundle (server/dist/server.mjs), __dirname = server/dist/ so we need ../../dist.
// In source (server/index.ts), __dirname = server/ — but dev uses vite, not this route.
app.use(express.static(join(__dirname, '../../dist')));
app.get('*', (_req, res) => res.sendFile(join(__dirname, '../../dist/index.html')));

// ── Simulation ──────────────────────────────────────────────────────────────

const sim = new SimulationController();
let connectedClients = 0;

// Eval loop — async, runs on worker threads in parallel.
sim.startEvalLoop().catch(e => {
  console.error('[eval] loop crashed:', e);
  process.exit(1);
});

// Display loop — fixed 30fps, never drops frames to the client
const DISPLAY_MS = Math.round(1000 / 30);
setInterval(() => {
  try {
    const result = sim.displayStep();
    if (!result || connectedClients === 0) return;
    io.emit('entities', result.entities);
    if (result.fields) io.emit('fields', result.fields);
    io.emit('meta', result.meta);
  } catch (e) {
    console.error('[display] step error (recovering):', e);
  }
}, DISPLAY_MS);

// ── Socket.IO ───────────────────────────────────────────────────────────────

io.on('connection', (socket) => {
  connectedClients++;
  console.log(`[io] +client (${connectedClients} total)`);
  socket.on('disconnect', () => {
    connectedClients--;
    console.log(`[io] -client (${connectedClients} total)`);
  });
});

// ── Start ───────────────────────────────────────────────────────────────────

httpServer.listen(PORT, () => {
  console.log(`[server] LostUplink: Axiom Forge on :${PORT}`);
  console.log(`[server] Eval: worker threads  |  Display: 30 fps`);
});
