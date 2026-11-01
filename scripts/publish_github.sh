#!/usr/bin/env bash
# Publish Curio to GitHub for teammates. Run in Terminal:
#   bash scripts/publish_github.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

echo "==> Working directory: $ROOT"

if ! command -v gh >/dev/null 2>&1; then
  if [[ -x "$ROOT/.tools/gh_2.67.0_macOS_arm64/bin/gh" ]]; then
    export PATH="$ROOT/.tools/gh_2.67.0_macOS_arm64/bin:$PATH"
  else
    echo "Installing GitHub CLI via Homebrew..."
    brew install gh
  fi
fi

echo "==> GitHub auth"
if ! gh auth status >/dev/null 2>&1; then
  echo "Browser login required (one time)."
  gh auth login -h github.com -p https -w
fi

if [[ ! -d .git ]]; then
  git init -b main
fi

# Ensure ignore rules are present
git add -A
# Drop accidental tool binaries if staged
git rm -r --cached .tools 2>/dev/null || true

if git rev-parse HEAD >/dev/null 2>&1; then
  echo "==> Creating update commit if needed"
  if ! git diff --cached --quiet || ! git diff --quiet; then
    git add -A
    git commit -m "Update Curio museum retrieval for group sharing"
  else
    echo "No new changes to commit."
  fi
else
  echo "==> Initial commit"
  git commit -m "$(cat <<'EOF'
Initial Curio attribute-compositional museum retrieval system.

Include gallery images, CLIP index, UI, and docs for COMP5523 group sharing.
CLIP weights are gitignored and download from Hugging Face on first run.
EOF
)"
fi

REPO_NAME="${REPO_NAME:-curio-retrieval}"
VISIBILITY="${VISIBILITY:-public}"

if git remote get-url origin >/dev/null 2>&1; then
  echo "==> Remote origin already set: $(git remote get-url origin)"
else
  echo "==> Creating GitHub repo: $REPO_NAME ($VISIBILITY)"
  gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --description "COMP5523 Curio: attribute-compositional Chinese museum text-to-image retrieval"
fi

echo "==> Pushing main"
git push -u origin main

URL=$(gh repo view --json url -q .url)
echo ""
echo "Done. Share this URL with teammates:"
echo "  $URL"
echo ""
echo "Teammate setup:"
echo "  git clone $URL"
echo "  cd $REPO_NAME && python3 -m venv .venv && source .venv/bin/activate"
echo "  pip install -r requirements.txt && python app.py --port 8765 --preload"
