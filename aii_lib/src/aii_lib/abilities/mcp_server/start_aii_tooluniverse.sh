#!/bin/bash
# Start AII ToolUniverse MCP server (aii_web_search_fast, aii_web_fetch_direct, aii_web_fetch_grep, dblp_bib_search)
# Uses MCPServer class with colorful logging

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
SESSION_NAME="aii_tooluniverse"
LOG="$PROJECT_ROOT/logs/aii_tooluniverse.log"
PORT=8101

# Check if server already running
if nc -z localhost $PORT 2>/dev/null; then
    echo "aii_tooluniverse already running on port $PORT"
    exit 0
fi

# Clean up dead session
zellij delete-session "$SESSION_NAME" 2>/dev/null

mkdir -p "$(dirname "$LOG")"

echo "Starting aii_tooluniverse in zellij session '$SESSION_NAME'..."

zellij --session "$SESSION_NAME" --new-session-with-layout <(cat <<EOF
layout {
    pane command="bash" {
        args "-c" "cd $PROJECT_ROOT && [ -f .venv/bin/activate ] && source .venv/bin/activate; while true; do echo '=== aii_tooluniverse starting at \$(date) ===' | tee -a $LOG; python -m aii_lib.abilities.mcp_server.server aii_tooluniverse 2>&1 | tee -a $LOG; echo '=== Server exited, restarting in 3s ===' | tee -a $LOG; sleep 3; done"
    }
}
EOF
) 2>/dev/null &

echo "Waiting for aii_tooluniverse..."
for i in {1..30}; do
    if nc -z localhost $PORT 2>/dev/null; then
        echo "aii_tooluniverse ready on port $PORT (took ${i}s)"
        exit 0
    fi
    sleep 1
done

echo "Failed to start within 30s. Check: zellij a $SESSION_NAME"
exit 1
