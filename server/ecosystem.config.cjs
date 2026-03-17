module.exports = {
  apps: [
    {
      name: 'axiom-forge',
      script: 'index.ts',
      interpreter: 'node',
      interpreter_args: '--import tsx/esm',
      cwd: '/opt/axiom-forge/server',
      env: { NODE_ENV: 'production', PORT: '3001' },
      restart_delay: 3000,
      max_restarts: 20,
      watch: false,
    },
  ],
};
