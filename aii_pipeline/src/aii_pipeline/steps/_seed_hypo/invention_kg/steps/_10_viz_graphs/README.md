# Building Blocks Knowledge Graph Visualization

Interactive visualization of building blocks knowledge graph using ECharts and TypeScript.

## Quick Start

### Development Mode (with auto-compilation)

```bash
# Install dependencies and start dev server with auto-recompilation
./dev.sh 8000

# Opens on http://localhost:8000
# TypeScript compiler watches for changes and recompiles automatically
```

### Production Mode

```bash
# Build once
npm run build

# Start HTTP server
./serve.sh 8000

# Then open http://localhost:8000
```

## Development

The project uses TypeScript for type safety:

- **Source**: `graph.ts` (TypeScript source with types)
- **Output**: `graph.js` (compiled JavaScript)
- **Auto-compile**: `npm run watch` or `npm run dev`

### Scripts

- `npm run build` - Compile TypeScript once
- `npm run watch` - Watch and recompile on changes
- `npm run serve` - Start HTTP server
- `npm run dev` - Watch + serve (development mode)
- `npm start` - Alias for dev

## Features

- **Interactive Graph**: Drag to pan, scroll to zoom
- **Year Slider**: View graphs for specific years or all years combined
- **Play Mode**: Automatic slideshow through years
- **Hover Details**: Rich tooltips showing building block information
- **Emphasis**: Click nodes to highlight connections
- **Type Safety**: Full TypeScript support with proper interfaces

## Styling

The visualization uses styling inspired by ECharts Les Miserables example:

- System sans-serif fonts for clean appearance
- Normalized node sizes (8-50 range)
- Source-colored edges (`color: 'source'`)
- Label overlap prevention
- Zoom limits (0.4x - 2x)

## Data Source

The visualization reads from `data/_5_bblocks_graph/*.json` (symlinked from `../../data/_5_bblocks_graph/`).

Run `_5_gen_graph.py` to generate the graph data first.

## File Structure

```
_6_viz_graph/
├── index.html          # Main HTML page
├── graph.ts            # TypeScript source (with type annotations)
├── graph.js            # Compiled JavaScript (generated from graph.ts)
├── graph.js.map        # Source map for debugging
├── package.json        # npm dependencies and scripts
├── tsconfig.json       # TypeScript configuration
├── dev.sh              # Development server (watch + serve)
├── serve.sh            # Production server (serve only)
└── README.md           # This file

../data/_5_bblocks_graph/
├── bblocks_graph_all.json    # Combined graph (all years)
├── bblocks_graph_1980.json   # Per-year graphs
├── bblocks_graph_1981.json
└── ...
```

## TypeScript Interfaces

```typescript
interface RawNode {
    x: number;
    y: number;
    id: string;
    label: string;
    size: number;
    color: string;
    count: number;
    pagerank: number;
    total_citations: number;
}

interface RawEdge {
    sourceID: string;
    targetID: string;
    width: number;
    count: number;
}

interface GraphData {
    nodes: RawNode[];
    edges: RawEdge[];
}
```

## Dependencies

- **ECharts 5.4.3**: Loaded via CDN
- **jQuery 3.7.1**: Loaded via CDN (for JSON loading)
- **TypeScript 5.5.4**: Dev dependency (for compilation)
- **concurrently 8.2.2**: Dev dependency (for running watch + serve)
