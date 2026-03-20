const fs = require('fs');

const ADMIN_TOKEN_PATH = process.env.ADMIN_TOKEN_PATH || '/opt/axiom-forge/admin_token';
let adminToken = process.env.ADMIN_TOKEN || '';
if (!adminToken && fs.existsSync(ADMIN_TOKEN_PATH)) {
  adminToken = fs.readFileSync(ADMIN_TOKEN_PATH, 'utf8').trim();
}

module.exports = {
  apps: [
    {
      name: 'axiom-forge',
      script: 'dist/server.mjs',
      cwd: '/opt/axiom-forge/server',
      env: {
        NODE_ENV: 'production',
        PORT: '3001',
        STATE_PATH: '/opt/axiom-forge/state.json',
        ADMIN_TOKEN_PATH,
        ...(adminToken ? { ADMIN_TOKEN: adminToken } : {}),
      },
      restart_delay: 3000,
      max_restarts: 20,
      watch: false,
    },
  ],
};
