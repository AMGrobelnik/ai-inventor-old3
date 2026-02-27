#!/bin/bash
# Simple HTTP server to view the visualization

PORT=${1:-9020}
SCRIPT_DIR="$(dirname "$0")"
GRAPHS_DIR="$SCRIPT_DIR/../../data/_9_graphs"

cd "$SCRIPT_DIR"

# Auto-detect most recent run and create symlink
if [ -d "$GRAPHS_DIR" ]; then
    LATEST_RUN=$(ls -t "$GRAPHS_DIR" | head -1)
    if [ -n "$LATEST_RUN" ]; then
        rm -f data  # Remove old symlink
        ln -sf "$(realpath "$GRAPHS_DIR/$LATEST_RUN")" data
        echo "Using run: $LATEST_RUN"
    fi
fi

echo "Starting web server on port $PORT..."
echo "Open http://localhost:$PORT in your browser"
echo "Press Ctrl+C to stop"

python3 -m http.server $PORT
