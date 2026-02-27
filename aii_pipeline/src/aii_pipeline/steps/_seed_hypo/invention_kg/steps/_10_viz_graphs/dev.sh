#!/bin/bash
# Development server with auto-compilation

PORT=${1:-9020}

echo "========================================"
echo "Building Blocks Graph - Dev Server"
echo "========================================"
echo ""
echo "Starting TypeScript compiler in watch mode..."
echo "Starting HTTP server on port $PORT..."
echo ""
echo "Open http://localhost:$PORT in your browser"
echo ""
echo "Press Ctrl+C to stop"
echo "========================================"

cd "$(dirname "$0")"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    echo ""
fi

# Start both processes
export PORT=$PORT
npm run dev
