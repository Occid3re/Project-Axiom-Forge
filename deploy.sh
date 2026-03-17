#!/usr/bin/env bash
# deploy.sh — Deploy LostUplink: Axiom Forge
# Pre-bundles server to plain JS — no TypeScript runtime on server.

set -euo pipefail

DEPLOY_HOST="${DEPLOY_HOST:-lostuplink-prod}"
DEPLOY_DIR="/opt/axiom-forge"

echo "==> Building client..."
npm install --silent
npm run build

echo "==> Bundling server..."
npm run build:server

echo ""
echo "==> Preparing remote directories..."
ssh "${DEPLOY_HOST}" "mkdir -p ${DEPLOY_DIR}/dist ${DEPLOY_DIR}/server/dist"

echo "==> Uploading client dist..."
tar -czf - -C dist . | ssh "${DEPLOY_HOST}" "rm -rf ${DEPLOY_DIR}/dist/* && tar -xzf - -C ${DEPLOY_DIR}/dist/"

echo "==> Uploading server bundle..."
tar -czf - -C server dist/server.mjs dist/eval-worker.mjs ecosystem.config.cjs package.json \
  | ssh "${DEPLOY_HOST}" "tar -xzf - -C ${DEPLOY_DIR}/server/"

echo "==> Installing server runtime dependencies (express, socket.io)..."
ssh "${DEPLOY_HOST}" "cd ${DEPLOY_DIR}/server && npm install --omit=dev --silent"

echo "==> Updating nginx config..."
cat ops/nginx/lostuplink.conf | ssh "${DEPLOY_HOST}" "cat > /etc/nginx/sites-available/lostuplink.conf"
ssh "${DEPLOY_HOST}" "ln -sfn /etc/nginx/sites-available/lostuplink.conf /etc/nginx/sites-enabled/lostuplink.conf && nginx -t"

echo "==> Reloading nginx..."
ssh "${DEPLOY_HOST}" "systemctl reload nginx"

echo "==> Restarting simulation server (PM2)..."
ssh "${DEPLOY_HOST}" "cd ${DEPLOY_DIR}/server && pm2 delete axiom-forge 2>/dev/null || true && pm2 start ecosystem.config.cjs && pm2 save"

echo ""
echo "Deploy complete! Live at https://lostuplink.com"
