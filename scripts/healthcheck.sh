#!/usr/bin/env bash
# =============================================================================
# AI Inventor — Health Check
# =============================================================================
# Verifies all services and APIs are working. Does NOT start anything.
# Run after start.sh to confirm everything is healthy.
#
# Usage:
#   bash scripts/healthcheck.sh
#
# Exit codes:
#   0 = all checks passed
#   1 = one or more checks failed
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC}   $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "  ${CYAN}[INFO]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ERRORS=0

# Auto-unset CLAUDECODE (prevents "nested session" error)
unset CLAUDECODE 2>/dev/null || true

echo "========================================="
echo "  AI Inventor — Health Check"
echo "========================================="
echo ""

# -------------------------------------------------------------------------
# 1. Environment checks
# -------------------------------------------------------------------------
echo "--- Environment ---"

# Python
if command -v python &>/dev/null; then
    PY_VERSION=$(python --version 2>&1)
    ok "Python: $PY_VERSION"
else
    fail "Python not found"
    ERRORS=$((ERRORS + 1))
fi

# aii_pipeline importable
if python -c "from aii_pipeline.cli import cli_main" 2>/dev/null; then
    ok "aii_pipeline importable"
else
    fail "aii_pipeline import failed"
    ERRORS=$((ERRORS + 1))
fi

# Claude Code CLI
if command -v claude &>/dev/null; then
    ok "Claude Code CLI: $(claude --version 2>/dev/null || echo 'installed')"
else
    fail "Claude Code CLI not found"
    ERRORS=$((ERRORS + 1))
fi

# GitHub CLI
if command -v gh &>/dev/null; then
    if gh auth status &>/dev/null; then
        GH_USER=$(gh api user -q '.login' 2>/dev/null || echo "unknown")
        ok "GitHub CLI authenticated as $GH_USER"
    else
        warn "GitHub CLI installed but not authenticated (gen_paper_repo will skip repo creation)"
    fi
else
    warn "GitHub CLI (gh) not installed"
fi

# .env file
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    ok ".env file exists"
else
    warn ".env file not found (API keys may be missing)"
fi

# CLAUDECODE env var (should be unset for pipeline)
if [[ -n "${CLAUDECODE:-}" ]]; then
    warn "CLAUDECODE env var is set — pipeline will fail with 'nested session' error"
else
    ok "CLAUDECODE not set (pipeline can spawn Claude Code)"
fi

# -------------------------------------------------------------------------
# 2. Server health checks
# -------------------------------------------------------------------------
echo ""
echo "--- Server Health Checks ---"

ABILITY_PORT=8100
AII_TU_PORT=8101
FULL_TU_PORT=8102

# Helper: MCP Streamable HTTP test (initialize + tools/list)
test_mcp_server() {
    local port=$1
    local name=$2

    local init_response
    init_response=$(curl -sf -D /tmp/_healthcheck_mcp_headers.txt -X POST "http://localhost:$port/mcp" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "healthcheck", "version": "1.0"}}}' \
        --max-time 10 2>/dev/null)

    if [[ -z "$init_response" ]]; then
        fail "$name MCP initialize failed (empty response)"
        ERRORS=$((ERRORS + 1))
        return 1
    fi

    local session_id
    session_id=$(grep -i "mcp-session-id" /tmp/_healthcheck_mcp_headers.txt 2>/dev/null | tr -d '\r' | awk '{print $2}')

    if [[ -z "$session_id" ]]; then
        fail "$name MCP initialize: no session ID"
        ERRORS=$((ERRORS + 1))
        return 1
    fi

    local tools_response
    tools_response=$(curl -sf -X POST "http://localhost:$port/mcp" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -H "Mcp-Session-Id: $session_id" \
        -d '{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}' \
        --max-time 10 2>/dev/null)

    local tool_count
    tool_count=$(echo "$tools_response" | python3 -c "
import sys
text = sys.stdin.read()
for line in text.split('\n'):
    if line.startswith('data:'):
        import json
        d = json.loads(line[5:])
        tools = d.get('result',{}).get('tools',[])
        print(len(tools))
        break
" 2>/dev/null)

    if [[ -n "$tool_count" && "$tool_count" -gt 0 ]]; then
        ok "$name MCP: $tool_count tools available"
        return 0
    else
        fail "$name MCP tools/list: no tools found"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

# --- Ability Server (port 8100) ---
if curl -sf "http://localhost:$ABILITY_PORT/health" > /dev/null 2>&1; then
    ok "Ability Server listening on port $ABILITY_PORT"
    HEALTH=$(curl -sf "http://localhost:$ABILITY_PORT/health" 2>/dev/null)
    if [[ -n "$HEALTH" ]]; then
        ENDPOINT_COUNT=$(echo "$HEALTH" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('endpoints',[])))" 2>/dev/null || echo "?")
        ok "Ability Server /health → $ENDPOINT_COUNT endpoints"
    else
        fail "Ability Server /health returned empty"
        ERRORS=$((ERRORS + 1))
    fi
else
    fail "Ability Server not responding on port $ABILITY_PORT"
    ERRORS=$((ERRORS + 1))
fi

# --- AII ToolUniverse (port 8101) ---
if (echo > /dev/tcp/localhost/$AII_TU_PORT) 2>/dev/null; then
    ok "AII ToolUniverse listening on port $AII_TU_PORT"
    test_mcp_server $AII_TU_PORT "AII ToolUniverse"
else
    fail "AII ToolUniverse not responding on port $AII_TU_PORT"
    ERRORS=$((ERRORS + 1))
fi

# --- Full ToolUniverse (port 8102) ---
if (echo > /dev/tcp/localhost/$FULL_TU_PORT) 2>/dev/null; then
    ok "Full ToolUniverse listening on port $FULL_TU_PORT"
    test_mcp_server $FULL_TU_PORT "Full ToolUniverse"
else
    fail "Full ToolUniverse not responding on port $FULL_TU_PORT"
    ERRORS=$((ERRORS + 1))
fi

# -------------------------------------------------------------------------
# 3. Claude Agent SDK test (minimal — 1 turn, haiku)
# -------------------------------------------------------------------------
echo ""
echo "--- Claude Agent SDK Test ---"

info "Running minimal Claude agent test (1 turn, haiku)..."
CLAUDE_TEST_RESULT=$(python3 -c "
import asyncio, json, os, sys
sys.path.insert(0, '$PROJECT_ROOT/aii_lib/src')
os.chdir('/tmp')

from loguru import logger
logger.remove()

from aii_lib.agents.claude.utils.monitor import UsageMonitor
monitor = UsageMonitor()
monitor._config['usage_tracking']['enabled'] = False

async def test():
    from aii_lib.agent_backend.claude import Agent, AgentOptions
    from aii_lib.telemetry import AIITelemetry
    tel = AIITelemetry()
    agent = Agent(AgentOptions(
        model='claude-haiku-4-5',
        max_turns=1,
        system_prompt='You are a calculator. Reply with ONLY the number.',
        agent_timeout=30,
        agent_retries=1,
        seq_prompt_retries=1,
        message_timeout=30,
        message_retries=1,
        cwd='/tmp',
        log_mode='none',
        telemetry=tel,
        run_id='healthcheck',
    ))
    result = await agent.run('What is 2+2?')
    return {
        'response': (result.final_response or '')[:100],
        'cost': result.total_cost,
        'failed': result.failed,
    }

try:
    r = asyncio.run(test())
    print(json.dumps(r))
except Exception as e:
    print(json.dumps({'error': str(e)[:200]}))
" 2>/dev/null)

CLAUDE_TEST_JSON=$(echo "$CLAUDE_TEST_RESULT" | tail -1)

if echo "$CLAUDE_TEST_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'error' not in d
resp = d['response']
cost = d['cost']
failed = d['failed']
print(f'response={resp!r}, cost=\${cost:.4f}, failed={failed}')
" 2>/dev/null; then
    ok "Claude Agent SDK works"
else
    ERROR_MSG=$(echo "$CLAUDE_TEST_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error','unknown error'))" 2>/dev/null || echo "unknown error")
    fail "Claude Agent SDK test failed: $ERROR_MSG"
    ERRORS=$((ERRORS + 1))
fi

# -------------------------------------------------------------------------
# 4. OpenRouter test (minimal — 1 request, cheapest model)
# -------------------------------------------------------------------------
echo ""
echo "--- OpenRouter API Test ---"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    OPENROUTER_KEY=$(grep -E '^OPENROUTER_API_KEY=' "$PROJECT_ROOT/.env" 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'")
fi
OPENROUTER_KEY="${OPENROUTER_KEY:-${OPENROUTER_API_KEY:-}}"

if [[ -z "$OPENROUTER_KEY" ]]; then
    warn "OPENROUTER_API_KEY not found — skipping OpenRouter test"
else
    info "Testing OpenRouter API (openai/gpt-oss-20b)..."
    OR_RESULT=$(curl -sf -X POST "https://openrouter.ai/api/v1/chat/completions" \
        -H "Authorization: Bearer $OPENROUTER_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "Reply with only the number: 2+2"}],
            "max_tokens": 10
        }' \
        --max-time 30 2>/dev/null)

    if echo "$OR_RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
content = d['choices'][0]['message']['content']
model = d.get('model', 'unknown')
print(f'response={content!r}, model={model}')
" 2>/dev/null; then
        ok "OpenRouter API works"
    else
        OR_ERROR=$(echo "$OR_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',{}).get('message','unknown'))" 2>/dev/null || echo "connection failed")
        fail "OpenRouter API test failed: $OR_ERROR"
        ERRORS=$((ERRORS + 1))
    fi
fi

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo ""
echo "========================================="
if [[ $ERRORS -eq 0 ]]; then
    echo -e "  ${GREEN}All checks passed!${NC}"
    echo "========================================="
    exit 0
else
    echo -e "  ${RED}$ERRORS check(s) failed${NC}"
    echo "========================================="
    exit 1
fi
