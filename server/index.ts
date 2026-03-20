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
const ADMIN_TOKEN = (process.env.ADMIN_TOKEN ?? '').trim();
const ADMIN_ENABLED = ADMIN_TOKEN.length >= 24;

// ── HTTP + Socket.IO ────────────────────────────────────────────────────────

const app = express();
const httpServer = createServer(app);
const io = new SocketIO(httpServer, {
  cors: { origin: '*' },
  maxHttpBufferSize: 512 * 1024,
});

const sim = new SimulationController();

app.use(express.json({ limit: '256kb' }));

// ── Admin API (must be before static catch-all) ─────────────────────────────

function requireAdmin(req: express.Request, res: express.Response, next: express.NextFunction) {
  if (!ADMIN_ENABLED) {
    res.status(503).json({ error: 'Admin disabled' });
    return;
  }

  const token = (req.headers['x-admin-token'] as string) ?? '';
  if (token !== ADMIN_TOKEN) {
    const forwarded = req.headers['cf-connecting-ip'] as string | undefined;
    const remote = forwarded || req.ip || req.socket.remoteAddress || 'unknown';
    console.warn(`[admin] unauthorized ${req.method} ${req.path} from ${remote}`);
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }
  next();
}

app.get('/api/admin/config', requireAdmin, (_req, res) => {
  res.json(sim.getAdminConfig());
});

app.post('/api/admin/config', requireAdmin, (req, res) => {
  try {
    sim.applyAdminConfig(req.body);
    console.log('[admin] config updated');
    res.json({ ok: true });
  } catch (e: any) {
    res.status(400).json({ error: e?.message ?? 'Unknown error' });
  }
});

app.post('/api/admin/reset', requireAdmin, (_req, res) => {
  sim.resetSimulation();
  console.log('[admin] reset accepted');
  res.json({ ok: true });
});

// ── Static / SPA ─────────────────────────────────────────────────────────────

// In the bundle (server/dist/server.mjs), __dirname = server/dist/ so we need ../../dist.
// In source (server/index.ts), __dirname = server/ — but dev uses vite, not this route.
app.use(express.static(join(__dirname, '../../dist')));
app.get('*', (_req, res) => res.sendFile(join(__dirname, '../../dist/index.html')));

// ── Simulation ──────────────────────────────────────────────────────────────

let connectedClients = 0;

// Eval loop — async, runs on worker threads in parallel.
sim.startEvalLoop().catch(e => {
  console.error('[eval] loop crashed:', e);
  process.exit(1);
});

// Display loop — fixed 30fps, paused when nobody is watching
const DISPLAY_MS = Math.round(1000 / 30);
setInterval(() => {
  if (connectedClients === 0) return;
  try {
    const result = sim.displayStep();
    if (!result) return;
    io.emit('entities', result.entities);
    if (result.fields) io.emit('fields', result.fields);
    io.emit('meta', result.meta);
  } catch (e) {
    console.error('[display] step error (recovering):', e);
  }
}, DISPLAY_MS);

// ── Socket.IO ───────────────────────────────────────────────────────────────

io.on('connection', (socket) => {
  const wasEmpty = connectedClients === 0;
  connectedClients++;
  console.log(`[io] +client (${connectedClients} total)`);
  // First viewer after idle period — restart display world with latest best laws
  if (wasEmpty) sim.restartDisplay();
  const bootstrap = sim.getBootstrapFrames();
  if (bootstrap) {
    socket.emit('fields', bootstrap.fields);
    socket.emit('entities', bootstrap.entities);
  }
  socket.on('disconnect', () => {
    connectedClients--;
    console.log(`[io] -client (${connectedClients} total)`);
  });
});

// ── Start ───────────────────────────────────────────────────────────────────

httpServer.listen(PORT, () => {
  console.log(`[server] LostUplink: Axiom Forge on :${PORT}`);
  console.log(`[server] Eval: worker threads  |  Display: 30 fps`);
  console.log(`[server] Admin API: ${ADMIN_ENABLED ? 'enabled' : 'disabled'}`);
});
