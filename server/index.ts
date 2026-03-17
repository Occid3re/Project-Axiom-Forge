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

app.use(express.static(join(__dirname, '../dist')));
app.get('*', (_req, res) => res.sendFile(join(__dirname, '../dist/index.html')));

// ── Simulation ──────────────────────────────────────────────────────────────

const sim = new SimulationController();
let connectedClients = 0;

// Eval loop — 5 ticks every 10ms = 500 ticks/sec
// Tiny batches keep the event loop free for Socket.IO and the 30fps display timer.
setInterval(() => {
  try { sim.evalBatch(5); } catch (e) {
    console.error('[eval] tick error (recovering):', e);
  }
}, 10);

// Display loop — fixed 30fps, never drops frames to the client
const DISPLAY_MS = Math.round(1000 / 30);
setInterval(() => {
  try {
    const result = sim.displayStep();
    if (!result || connectedClients === 0) return;
    io.emit('frame', result.frame);
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
  console.log(`[server] Eval: 500 t/s  |  Display: 30 fps`);
});
