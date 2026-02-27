#!/bin/bash
# Start the graph visualization web server
# Usage: ./viz_graphs.sh [port]

PORT=${1:-9020}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIZ_DIR="$SCRIPT_DIR/_10_viz_graphs"

echo "Starting Graph Visualization on http://localhost:$PORT"
cd "$VIZ_DIR" && ./dev.sh "$PORT"
