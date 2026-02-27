#!/usr/bin/env bash
# =============================================================================
# AI Inventor — Start Services
# =============================================================================
# Starts all backend servers required by the pipeline.
# Idempotent: skips servers that are already running.
#
# Usage:
#   bash scripts/start.sh
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

YELLOW='\033[1;33m'
ok()   { echo -e "  ${GREEN}[OK]${NC}   $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "  ${CYAN}[INFO]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Auto-unset CLAUDECODE (prevents "nested session" error)
unset CLAUDECODE 2>/dev/null || true

# ---------------------------------------------------------------------------
# GitHub CLI auth (from GH_TOKEN in .env or environment)
# ---------------------------------------------------------------------------
if ! gh auth status &>/dev/null; then
    # Try loading from .env if not already in environment
    if [[ -z "${GH_TOKEN:-}" && -f "$PROJECT_ROOT/.env" ]]; then
        GH_TOKEN=$(grep -E '^GH_TOKEN=' "$PROJECT_ROOT/.env" 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" || true)
        export GH_TOKEN
    fi

    if [[ -n "${GH_TOKEN:-}" ]]; then
        info "Authenticating GitHub CLI with GH_TOKEN..."
        echo "$GH_TOKEN" | gh auth login --with-token 2>/dev/null && ok "GitHub CLI authenticated" || warn "GitHub CLI auth failed"
    else
        warn "GH_TOKEN not found — gh CLI will not be authenticated (gen_paper_repo will skip repo creation)"
    fi
else
    ok "GitHub CLI already authenticated"
fi

echo "--- Starting Servers ---"

info "Starting servers (uses zellij if available, direct background otherwise)..."

python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/aii_lib/src')
from aii_lib.utils.start_servers import ensure_server_running
ok = ensure_server_running(log_func=lambda msg: print(f'  {msg}'))
sys.exit(0 if ok else 1)
" 2>&1

if [[ $? -eq 0 ]]; then
    ok "All servers started"
else
    fail "Some servers failed to start"
    exit 1
fi
