#!/usr/bin/env bash
# deploy.sh — Deploy LostUplink: Axiom Forge to production
# Replaces the old Signal Lost game entirely.
#
# Usage:
#   ./deploy.sh
#
# Prerequisites:
#   - SSH access to lostuplink-prod (configured in ~/.ssh/config)
#   - npm installed locally

set -euo pipefail

DEPLOY_HOST="${DEPLOY_HOST:-lostuplink-prod}"
DEPLOY_DIR="/opt/axiom-forge"

echo "==> Building Axiom Forge..."
npm install --silent
npm run build

echo ""
echo "==> Preparing remote directory..."
ssh "${DEPLOY_HOST}" "mkdir -p ${DEPLOY_DIR}"

echo "==> Uploading dist..."
rsync -az --delete dist/ "${DEPLOY_HOST}:${DEPLOY_DIR}/dist/"

echo "==> Updating nginx config..."
rsync -az ops/nginx/lostuplink.conf "${DEPLOY_HOST}:/etc/nginx/sites-available/lostuplink.conf"
ssh "${DEPLOY_HOST}" "ln -sfn /etc/nginx/sites-available/lostuplink.conf /etc/nginx/sites-enabled/lostuplink.conf"

echo "==> Testing nginx config..."
ssh "${DEPLOY_HOST}" "nginx -t"

echo "==> Reloading nginx..."
ssh "${DEPLOY_HOST}" "systemctl reload nginx"

echo "==> Stopping old Signal Lost PM2 process (if running)..."
ssh "${DEPLOY_HOST}" "pm2 delete signal-lost 2>/dev/null && pm2 save || echo 'No signal-lost process found, skipping'"

echo ""
echo "Deploy complete! Site live at https://lostuplink.com"
