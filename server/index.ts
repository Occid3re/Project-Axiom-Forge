/**
 * LostUplink: Axiom Forge — server
 * Runs the simulation continuously, streams state to all connected clients.
 * Express serves static files. Socket.IO broadcasts binary frames + JSON meta.
 */

import express from 'express';
import { createServer } from 'http';
import { Server as SocketIO } from 'socket.io';
import { EventEmitter } from 'events';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { SimulationController } from './simulation.ts';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PORT = parseInt(process.env.PORT ?? '3001', 10);

// ---- HTTP + Socket.IO setup ------------------------------------------------

const app = express();
const httpServer = createServer(app);
const io = new SocketIO(httpServer, {
  cors: { origin: '*' },
  maxHttpBufferSize: 1e6,
});

// Serve static client files
app.use(express.static(join(__dirname, '../dist')));
app.get('*', (_req, res) => res.sendFile(join(__dirname, '../dist/index.html')));

// ---- Simulation loop -------------------------------------------------------

const emitter = new EventEmitter();
const sim = new SimulationController(emitter);

let connectedClients = 0;
let lastBroadcast = 0;
const BROADCAST_INTERVAL_MS = 66; // ~15fps

function simLoop() {
  const result = sim.tick();

  if (result && connectedClients > 0) {
    const now = Date.now();
    if (now - lastBroadcast >= BROADCAST_INTERVAL_MS) {
      lastBroadcast = now;
      io.emit('frame', result.frame);
      io.emit('meta', result.meta);
    }
  }

  // Yield to event loop — lets Socket.IO flush, keeps CPU reasonable
  setImmediate(simLoop);
}

// ---- Socket.IO events ------------------------------------------------------

io.on('connection', (socket) => {
  connectedClients++;
  console.log(`[io] client connected — ${connectedClients} total`);

  socket.on('disconnect', () => {
    connectedClients--;
    console.log(`[io] client disconnected — ${connectedClients} total`);
  });
});

// ---- Start -----------------------------------------------------------------

httpServer.listen(PORT, () => {
  console.log(`[server] LostUplink: Axiom Forge`);
  console.log(`[server] Listening on :${PORT}`);
  console.log(`[server] Starting simulation loop...`);
  setImmediate(simLoop);
});
