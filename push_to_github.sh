#!/bin/bash
# Push model_server to https://github.com/aykumar21/Drone_LLM
set -euo pipefail

cd "$(dirname "$0")"

REMOTE="https://github.com/aykumar21/Drone_LLM.git"

if ! git remote get-url origin &>/dev/null; then
  git remote add origin "$REMOTE"
else
  git remote set-url origin "$REMOTE"
fi

# Prefer GitHub CLI if installed and authenticated
if command -v gh &>/dev/null && gh auth status &>/dev/null 2>&1; then
  if ! gh repo view aykumar21/Drone_LLM &>/dev/null 2>&1; then
    echo "Creating GitHub repo Drone_LLM..."
    gh repo create Drone_LLM --public --source=. --remote=origin
  fi
  git push -u origin main
  echo "Done: https://github.com/aykumar21/Drone_LLM"
  exit 0
fi

echo "GitHub CLI not authenticated. Create the repo first:"
echo "  https://github.com/new  → name: Drone_LLM → Public → Create (no README)"
echo ""
echo "Then push (use Personal Access Token as password):"
echo "  git push -u origin main"
git push -u origin main
