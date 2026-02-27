#!/bin/bash
# Start ability server and both tooluniverse servers in separate zellij sessions
# Usage: ./start_ability_server.sh
#
# Starts:
#   - Ability Server on port 8100 (REST endpoints) in session 'aii_ability_server'
#   - AII ToolUniverse on port 8001 (aii_web_search_fast, aii_web_fetch_direct only) in session 'aii_tooluniverse'
#   - Full ToolUniverse on port 8002 (all tools via execute_tool) in session 'full_tooluniverse'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# === Ability Server ===
SESSION_NAME="aii_ability_server"
LOG="$PROJECT_ROOT/logs/ability_server.log"
PORT=8100

# Check if ability server already running
if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "Ability Server already running on port $PORT"
else
    # Clean up dead session if exists
    zellij delete-session "$SESSION_NAME" 2>/dev/null

    # Create log dir
    mkdir -p "$(dirname "$LOG")"

    # Start zellij session with server
    echo "Starting Ability Server in zellij session '$SESSION_NAME'..."
    zellij --session "$SESSION_NAME" --new-session-with-layout <(cat <<EOF
layout {
    pane command="bash" {
        args "-c" "cd $PROJECT_ROOT && [ -f .venv/bin/activate ] && source .venv/bin/activate; export AII_SKIP_MCP_SUBPROCESS=1 && while true; do echo \"=== Ability Server starting at \$(date) ===\" | tee -a $LOG; python -m aii_lib.abilities.ability_server.endpoints 2>&1 | tee -a $LOG; echo \"=== Server exited, restarting in 3s ===\" | tee -a $LOG; sleep 3; done"
    }
}
EOF
) 2>/dev/null &

    # Wait for ability server to be ready
    echo "Waiting for Ability Server..."
    for i in {1..30}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Ability Server ready (took ${i}s)"
            break
        fi
        sleep 1
    done

    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Ability Server failed to start within 30s"
        echo "Check: zellij a $SESSION_NAME"
    fi
fi

# === ToolUniverse Servers ===
# Start both in their own zellij sessions (independent processes)
MCP_DIR="$SCRIPT_DIR/../abilities/mcp_server"
echo ""
"$MCP_DIR/start_aii_tooluniverse.sh"
echo ""
"$MCP_DIR/start_full_tooluniverse.sh"

# Summary
echo ""
echo "Sessions:"
echo "  Ability Server:    zellij a aii_ability_server"
echo "  AII ToolUniverse:  zellij a aii_tooluniverse   (port 8001, aii_web_search_fast + aii_web_fetch_direct)"
echo "  Full ToolUniverse: zellij a full_tooluniverse  (port 8002, all tools via execute_tool)"
