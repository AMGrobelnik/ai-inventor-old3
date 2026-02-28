#!/usr/bin/env bash
# =============================================================================
# AI Inventor — Setup Script
# =============================================================================
# Usage: bash scripts/setup.sh
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

info()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()     { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "  AI Inventor — Setup"
echo "========================================="
echo ""

# -------------------------------------------------------------------------
# 0. Unset CLAUDECODE (prevents "nested session" error in pipeline)
# -------------------------------------------------------------------------
if [[ -n "${CLAUDECODE:-}" ]]; then
    unset CLAUDECODE
    info "Unset CLAUDECODE (pipeline can now spawn Claude Code subprocesses)"
else
    info "CLAUDECODE not set (OK)"
fi

# -------------------------------------------------------------------------
# 1. Check uv (needed first — it can install Python)
# -------------------------------------------------------------------------
echo "--- Checking uv ---"

if command -v uv &>/dev/null; then
    info "uv found: $(uv --version)"
else
    warn "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        info "uv installed: $(uv --version)"
    else
        err "Failed to install uv. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi

# -------------------------------------------------------------------------
# 2. Check Claude Code CLI
# -------------------------------------------------------------------------
echo ""
echo "--- Checking Claude Code CLI ---"

if command -v claude &>/dev/null; then
    info "Claude Code CLI found: $(claude --version 2>/dev/null || echo 'installed')"
else
    warn "Claude Code CLI not found. Installing..."
    curl -fsSL https://claude.ai/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
    if command -v claude &>/dev/null; then
        info "Claude Code CLI installed."
        echo ""
        warn "You need to authenticate Claude Code before running the pipeline."
        warn "Run: claude login"
    else
        err "Failed to install Claude Code CLI."
        err "Install manually: https://docs.anthropic.com/en/docs/claude-code"
        exit 1
    fi
fi

# -------------------------------------------------------------------------
# 3. Check GitHub CLI
# -------------------------------------------------------------------------
echo ""
echo "--- Checking GitHub CLI ---"

if command -v gh &>/dev/null; then
    if gh auth status &>/dev/null; then
        info "GitHub CLI authenticated"
    else
        warn "GitHub CLI installed but not authenticated."
        warn "Run: gh auth login"
    fi
else
    warn "GitHub CLI (gh) not installed — gen_paper_repo will skip repo creation."
    warn "Install: https://cli.github.com/"
fi

# -------------------------------------------------------------------------
# 4. Create virtual environment (uv auto-downloads Python 3.10 if needed)
# -------------------------------------------------------------------------
echo ""
echo "--- Setting up virtual environment ---"

if [[ -d ".venv" ]]; then
    info "Virtual environment already exists at .venv/"
else
    # uv will auto-download Python 3.10 if not on system
    uv venv .venv --python=3.10
    info "Created virtual environment with Python 3.10"
fi

# Activate
source .venv/bin/activate
info "Activated .venv ($(python --version))"

# Verify Python version is 3.10 or 3.11
PY_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+')
if [[ "$PY_VERSION" != "3.10" && "$PY_VERSION" != "3.11" ]]; then
    err "Virtual environment has Python $PY_VERSION but 3.10 or 3.11 is required."
    err "Delete .venv and re-run: rm -rf .venv && bash scripts/setup.sh"
    exit 1
fi

# -------------------------------------------------------------------------
# 5. Install packages
# -------------------------------------------------------------------------
echo ""
echo "--- Installing packages ---"

# Install CPU-only PyTorch first (avoids 2.5GB GPU download)
echo "  Installing PyTorch (CPU-only)..."
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
info "PyTorch CPU installed"

echo "  Installing aii_lib..."
uv pip install -e aii_lib/
info "aii_lib installed"

echo "  Installing aii_pipeline..."
uv pip install -e aii_pipeline/
info "aii_pipeline installed"

# -------------------------------------------------------------------------
# 6. Link skill venvs to main .venv
# -------------------------------------------------------------------------
echo ""
echo "--- Linking skill venvs ---"

SKILL_COUNT=0
for skill_scripts in .claude/skills/*/scripts; do
    if [[ -d "$skill_scripts" ]]; then
        target="$skill_scripts/.venv"
        if [[ -L "$target" ]]; then
            : # Already a symlink, skip
        elif [[ -d "$target" ]]; then
            rm -rf "$target"
            ln -s "$PROJECT_ROOT/.venv" "$target"
            SKILL_COUNT=$((SKILL_COUNT + 1))
        else
            ln -s "$PROJECT_ROOT/.venv" "$target"
            SKILL_COUNT=$((SKILL_COUNT + 1))
        fi
    fi
done
if [[ $SKILL_COUNT -gt 0 ]]; then
    info "Linked $SKILL_COUNT skill venvs to .venv"
else
    info "All skill venvs already linked"
fi

# -------------------------------------------------------------------------
# 7. Set up .env
# -------------------------------------------------------------------------
echo ""
echo "--- Checking .env ---"

if [[ -f ".env" ]]; then
    info ".env file exists"
else
    cp .env.template .env
    warn "Created .env from template — edit it with your API keys:"
    warn "  GEMINI_API_KEY, HF_TOKEN, SERPER_API_KEY, LEANEXPLORE_API_KEY"
fi

# -------------------------------------------------------------------------
# 7. Verify installation
# -------------------------------------------------------------------------
echo ""
echo "--- Verifying installation ---"

if python -c "from aii_pipeline.cli import cli_main; print('OK')" 2>/dev/null | grep -q "OK"; then
    info "aii_pipeline imports successfully"
else
    err "aii_pipeline import failed. Check error output above."
    exit 1
fi

if command -v aii_pipeline &>/dev/null; then
    info "aii_pipeline command is available"
else
    warn "aii_pipeline command not on PATH. Run: source .venv/bin/activate"
fi

# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------
echo ""
echo "========================================="
echo -e "  ${GREEN}Setup complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Authenticate: claude login && gh auth login"
echo "  3. Edit aii_pipeline/config.yaml (research_direction, pipeline steps)"
echo "  4. source .venv/bin/activate && unset CLAUDECODE"
echo "  5. aii_pipeline"
echo ""
