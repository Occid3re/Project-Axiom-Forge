module.exports = {
  apps: [
    {
      name: 'axiom-forge',
      script: 'dist/server.mjs',
      cwd: '/opt/axiom-forge/server',
      env: { NODE_ENV: 'production', PORT: '3001' },
      restart_delay: 3000,
      max_restarts: 20,
      watch: false,
    },
  ],
};
