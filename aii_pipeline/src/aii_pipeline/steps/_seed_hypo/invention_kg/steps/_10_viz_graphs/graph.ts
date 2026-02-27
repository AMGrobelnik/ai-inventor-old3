// TypeScript interfaces
interface RawNode {
    x: number;
    y: number;
    id: string;
    label: string;
    size: number;
    color: string;
    count?: number;
    pagerank?: number;
    total_citations?: number;
    topic?: string;
    topics?: string[];
    type?: string;
    entity_type?: string;
    from_triple?: boolean;
    edge_count?: number;
    year?: number;
    normalized_score?: number;
    labelText?: string;
    [key: string]: any;
}

interface RawEdge {
    sourceID?: string;
    targetID?: string;
    source: string;
    target: string;
    width?: number;
    count?: number;
    relation?: string;
    type?: string;
    [key: string]: any;
}

interface GraphData {
    nodes: RawNode[];
    edges: RawEdge[];
    topic_edges?: RawEdge[];
}

interface YearInfo {
    value: string;
    label: string;
    file?: string;
}

interface GraphTypeConfig {
    name: string;
    file: string;
    hasYears: boolean;
    yearSource?: 'files' | 'data';
    hasTopics: boolean;
}

// ECharts types
declare const echarts: any;

// Initialize chart
const chartElement = document.getElementById('main');
if (!chartElement) {
    throw new Error('Chart element not found');
}
const myChart = echarts.init(chartElement, 'dark');

// Graph type configuration
// hasYears: true means the graph supports year filtering
// yearSource: 'files' for separate year files (concepts), 'data' for in-memory filtering (paper_concepts)
const GRAPH_TYPES: Record<string, GraphTypeConfig> = {
    'concepts': { name: 'Concepts', file: 'data/cooccurrence/all.json', hasYears: true, yearSource: 'files', hasTopics: true },
    'paper_concepts': { name: 'Paper to Concepts', file: 'data/semantic/full.json', hasYears: true, yearSource: 'data', hasTopics: true },
    'ontology': { name: 'Concept Ontology', file: 'data/ontology/full.json', hasYears: false, hasTopics: true },
    'blind_spots': { name: 'Blind Spots', file: 'data/derived/blind_spots.json', hasYears: false, hasTopics: true },
};

// State management
let currentGraphType: string = 'concepts';
let currentTopic: string = 'all';
let currentRelation: string = 'all';
let ontologyLayout: 'force' | 'circular' = 'force';
let ontologyFilter: 'overlap' | 'full' = 'overlap';
let crossTopicLayout: 'force' | 'fixed' = 'fixed';
let showGaps: boolean = true;
let showBestOnly: boolean = true;
let availableYears: YearInfo[] = [];
let availableTopics: string[] = [];
let currentYearIndex: number = 0;
let currentYear: string = 'all';
let dataYears: number[] = [];
let isPlaying: boolean = false;
let playInterval: number | null = null;
let currentGraphData: GraphData | null = null;
let originalGraphData: GraphData | null = null;
let isUpdatingTopicFilter: boolean = false;
const PLAY_INTERVAL_MS: number = 5000;

// Type colors for nodes
const TYPE_COLORS: Record<string, string> = {
    'paper': '#4a90d9',
    'concept': '#34d399',
    'wikidata_class': '#f59e0b',
    'task': '#ef4444',
    'method': '#8b5cf6',
    'artifact': '#ec4899',
};

// Topic colors
const TOPIC_COLORS = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

/**
 * Show/hide loading indicator
 */
function setLoading(show: boolean, text: string = 'Loading...'): void {
    const loading = document.getElementById('loading');
    if (loading) {
        if (show) {
            loading.style.display = 'flex';
            const span = loading.querySelector('span');
            if (span) span.textContent = text;
        } else {
            loading.style.display = 'none';
        }
    }
}

/**
 * Legend content for each graph type
 */
const GRAPH_LEGENDS: Record<string, { title: string; html: string }> = {
    'concepts': {
        title: 'Concept Co-occurrence',
        html: `
            <p><span class="font-medium text-foreground">Node size:</span> citation count (log-scaled)</p>
            <p><span class="font-medium text-foreground">Node color:</span> PageRank centrality</p>
            <p class="flex items-center gap-1">
                <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
                <span class="w-2 h-2 rounded-full bg-yellow-500"></span>
                <span class="w-2 h-2 rounded-full bg-orange-500"></span>
                <span class="w-2 h-2 rounded-full bg-red-500"></span>
                <span class="text-muted-foreground ml-1">peripheral → hub</span>
            </p>
            <p><span class="font-medium text-foreground">Edge:</span> co-occurrence count</p>
        `
    },
    'paper_concepts': {
        title: 'Paper to Concepts',
        html: `
            <p><span class="font-medium text-foreground">■ Square nodes:</span> Papers</p>
            <p><span class="font-medium text-foreground">● Circle nodes:</span> Concepts</p>
            <p class="flex items-center gap-1.5">
                <span class="w-3 h-2 rounded" style="background:#27ae60"></span>
                <span>Paper uses concept</span>
            </p>
            <p class="flex items-center gap-1.5">
                <span class="w-3 h-2 rounded" style="background:#e74c3c"></span>
                <span>Paper proposes concept</span>
            </p>
        `
    },
    'ontology': {
        title: 'Concept Ontology',
        html: `
            <p><span class="font-medium text-foreground">Node size:</span> connection count</p>
            <p class="flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full" style="background:#34d399"></span>
                <span>Paper concepts</span>
            </p>
            <p class="flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full" style="background:#f59e0b"></span>
                <span>Wikidata classes</span>
            </p>
            <p><span class="font-medium text-foreground">Edge:</span> instance_of / subclass_of</p>
        `
    },
    'blind_spots': {
        title: 'Topic Blind Spots',
        html: `
            <p><span class="font-medium text-foreground">Square nodes:</span> Research topics</p>
            <p><span class="font-medium text-foreground">Circle nodes:</span> Blind spot concepts</p>
            <p class="flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full" style="background:#e74c3c"></span>
                <span>High opportunity score</span>
            </p>
            <p class="flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full" style="background:#3b82f6"></span>
                <span>Lower opportunity score</span>
            </p>
            <p><span class="font-medium text-foreground">Edge:</span> topic is missing concept</p>
        `
    }
};

/**
 * Update legend based on graph type
 */
function updateLegend(graphType: string): void {
    const titleEl = document.getElementById('graph-title');
    const legendEl = document.getElementById('graph-legend');
    const legend = GRAPH_LEGENDS[graphType] || GRAPH_LEGENDS['concepts'];
    if (titleEl) titleEl.textContent = legend.title;
    if (legendEl) legendEl.innerHTML = legend.html;
}

/**
 * Discover available graph files
 */
async function discoverGraphFiles(): Promise<YearInfo[]> {
    const baseFiles: YearInfo[] = [
        { value: 'all', label: 'All Years', file: 'data/cooccurrence/all.json' }
    ];

    const yearPromises = [];
    for (let year = 2015; year <= 2025; year++) {
        yearPromises.push(
            fetch(`data/cooccurrence/by_year/${year}.json`)
                .then(response => {
                    if (response.ok) {
                        return { value: year.toString(), label: year.toString(), file: `data/cooccurrence/by_year/${year}.json` };
                    }
                    return null;
                })
                .catch(() => null)
        );
    }

    const yearResults = await Promise.all(yearPromises);
    const validYears = yearResults.filter((y): y is YearInfo => y !== null);
    return [...baseFiles, ...validYears];
}

/**
 * Load graph JSON
 */
async function loadGraph(graphFile: string): Promise<GraphData> {
    const response = await fetch(graphFile);
    if (!response.ok) {
        throw new Error(`Failed to load ${graphFile}`);
    }
    return await response.json();
}

/**
 * Extract unique topics from graph data
 */
function extractTopics(data: GraphData): string[] {
    const topics = new Set<string>();
    data.nodes.forEach(node => {
        if (node.topic) {
            topics.add(node.topic);
        }
        const nodeTopics = node.topics;
        if (Array.isArray(nodeTopics)) {
            nodeTopics.forEach((t: string) => topics.add(t));
        }
    });
    return Array.from(topics).sort();
}

/**
 * Extract unique years from graph data
 */
function extractYearsFromData(data: GraphData): number[] {
    const years = new Set<number>();
    data.nodes.forEach(node => {
        if (node.year !== undefined && node.year !== null) {
            years.add(node.year);
        }
    });
    return Array.from(years).sort((a, b) => a - b);
}

/**
 * Filter graph data by year (in-memory filtering)
 */
function filterByYear(data: GraphData, year: string): GraphData {
    if (year === 'all') return data;

    const yearNum = parseInt(year);
    const yearNodeIds = new Set(data.nodes.filter(node => {
        return node.year === yearNum || node.year === parseInt(year);
    }).map(n => n.id));

    if (yearNodeIds.size === 0) {
        return { nodes: [], edges: [] };
    }

    const connectedNodeIds = new Set(yearNodeIds);
    data.edges.forEach(edge => {
        const sourceId = edge.sourceID || edge.source;
        const targetId = edge.targetID || edge.target;
        if (yearNodeIds.has(sourceId)) {
            connectedNodeIds.add(targetId);
        }
        if (yearNodeIds.has(targetId)) {
            connectedNodeIds.add(sourceId);
        }
    });

    const filteredNodes = data.nodes.filter(node => connectedNodeIds.has(node.id));
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = data.edges.filter(edge => {
        const sourceId = edge.sourceID || edge.source;
        const targetId = edge.targetID || edge.target;
        return nodeIds.has(sourceId) && nodeIds.has(targetId);
    });

    return { nodes: filteredNodes, edges: filteredEdges };
}

/**
 * Filter graph data by topic
 */
function filterByTopic(data: GraphData, topic: string): GraphData {
    if (topic === 'all') return data;

    const topicNodeIds = new Set(
        data.nodes.filter(node => {
            if (node.topic === topic) return true;
            if (Array.isArray(node.topics) && node.topics.includes(topic)) return true;
            return false;
        }).map(n => n.id)
    );

    if (topicNodeIds.size === 0) return { nodes: [], edges: [] };

    const connectedNodeIds = new Set(topicNodeIds);
    data.edges.forEach(edge => {
        const sourceId = edge.sourceID || edge.source;
        const targetId = edge.targetID || edge.target;
        if (topicNodeIds.has(sourceId)) {
            connectedNodeIds.add(targetId);
        }
        if (topicNodeIds.has(targetId)) {
            connectedNodeIds.add(sourceId);
        }
    });

    const filteredNodes = data.nodes.filter(node => connectedNodeIds.has(node.id));
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = data.edges.filter(edge => {
        const sourceId = edge.sourceID || edge.source;
        const targetId = edge.targetID || edge.target;
        return nodeIds.has(sourceId) && nodeIds.has(targetId);
    });

    return { nodes: filteredNodes, edges: filteredEdges };
}

/**
 * Filter by relation type (for semantic graph)
 */
function filterByRelation(data: GraphData, relation: string): GraphData {
    if (relation === 'all') return data;
    const filteredEdges = data.edges.filter(edge => edge.relation === relation);
    const connectedNodeIds = new Set<string>();
    filteredEdges.forEach(edge => {
        const sourceId = edge.sourceID || edge.source;
        const targetId = edge.targetID || edge.target;
        connectedNodeIds.add(sourceId);
        connectedNodeIds.add(targetId);
    });
    const filteredNodes = data.nodes.filter(node => connectedNodeIds.has(node.id));
    return { nodes: filteredNodes, edges: filteredEdges };
}

/**
 * Filter ontology to only show nodes connected to multiple topics
 */
function filterOntologyOverlap(data: GraphData): GraphData {
    const multiTopicNodeIds = new Set(
        data.nodes.filter(node => {
            const nodeTopics = node.topics;
            return Array.isArray(nodeTopics) && nodeTopics.length > 1;
        }).map(n => n.id)
    );

    if (multiTopicNodeIds.size === 0) return data;

    const connectedNodeIds = new Set(multiTopicNodeIds);
    data.edges.forEach(edge => {
        const sourceId = edge.source;
        const targetId = edge.target;
        if (multiTopicNodeIds.has(sourceId)) {
            connectedNodeIds.add(targetId);
        }
        if (multiTopicNodeIds.has(targetId)) {
            connectedNodeIds.add(sourceId);
        }
    });

    const filteredNodes = data.nodes.filter(node => connectedNodeIds.has(node.id));
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = data.edges.filter(edge => {
        return nodeIds.has(edge.source) && nodeIds.has(edge.target);
    });

    return { nodes: filteredNodes, edges: filteredEdges };
}

/**
 * Filter to show gaps for selected topic
 * In blind_spots.json, edges FROM topic TO concept mean "concept is a gap for topic"
 * So we just show the selected topic + all concepts directly connected to it
 */
function filterCrossTopicGaps(data: GraphData, selectedTopic: string): GraphData {
    if (selectedTopic === 'all') return data;

    const selectedTopicNode = data.nodes.find(n => n.type === 'topic' && n.label === selectedTopic);
    if (!selectedTopicNode) return data;

    // Find all concepts connected to this topic (these ARE the gaps)
    const gapConceptIds = new Set<string>();
    const gapEdges: RawEdge[] = [];

    data.edges.forEach(edge => {
        if (edge.source === selectedTopicNode.id) {
            gapConceptIds.add(edge.target);
            gapEdges.push(edge);
        }
    });

    // Only include selected topic + its gap concepts
    const filteredNodes = data.nodes.filter(node =>
        node.id === selectedTopicNode.id || gapConceptIds.has(node.id)
    );

    return { nodes: filteredNodes, edges: gapEdges };
}

/**
 * Filter blind spots to only include top N% of concepts by score
 */
function filterBlindSpotsTop(data: GraphData, topPercent: number = 0.05): GraphData {
    const topics = data.nodes.filter(n => n.type === 'topic');
    const concepts = data.nodes.filter(n => n.type === 'bridging_concept' || n.type === 'blind_spot_concept');

    if (concepts.length === 0) return data;

    // Sort by max_opportunity_score (primary) or avg_opportunity_score (fallback) or percentile
    const sortedConcepts = [...concepts].sort((a, b) => {
        const scoreA = a.max_opportunity_score ?? a.avg_opportunity_score ?? a.percentile ?? a.normalized_score ?? 0;
        const scoreB = b.max_opportunity_score ?? b.avg_opportunity_score ?? b.percentile ?? b.normalized_score ?? 0;
        return scoreB - scoreA;
    });

    const keepCount = Math.max(1, Math.ceil(concepts.length * topPercent));
    const topConcepts = sortedConcepts.slice(0, keepCount);
    const topConceptIds = new Set(topConcepts.map(c => c.id));

    const filteredNodes = [...topics, ...topConcepts];
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = data.edges.filter(edge =>
        nodeIds.has(edge.source) && nodeIds.has(edge.target)
    );

    return { ...data, nodes: filteredNodes, edges: filteredEdges };
}

/**
 * Set up year slider from extracted data years
 */
function setupYearSliderFromData(years: number[]): void {
    const yearSlider = document.getElementById('year-slider') as HTMLInputElement | null;
    const yearDisplay = document.getElementById('year-display');

    dataYears = years;
    const yearEntries: YearInfo[] = [
        { value: 'all', label: 'All Years' },
        ...years.map(y => ({ value: y.toString(), label: y.toString() }))
    ];

    availableYears = yearEntries;
    currentYearIndex = 0;
    currentYear = 'all';

    if (yearSlider) {
        yearSlider.max = (yearEntries.length - 1).toString();
        yearSlider.value = '0';
    }
    if (yearDisplay) {
        yearDisplay.textContent = 'All Years';
    }
}

/**
 * Update topic filter dropdown
 */
function updateTopicFilter(topics: string[]): void {
    const select = document.getElementById('topic-filter') as HTMLSelectElement | null;
    if (!select) return;

    isUpdatingTopicFilter = true;
    const previousSelection = currentTopic;

    while (select.options.length > 1) {
        select.remove(1);
    }

    const allTopics = new Set(topics);
    if (previousSelection !== 'all') {
        allTopics.add(previousSelection);
    }

    Array.from(allTopics).sort().forEach(topic => {
        const option = document.createElement('option');
        option.value = topic;
        option.textContent = topic;
        select.appendChild(option);
    });

    availableTopics = Array.from(allTopics);

    if (previousSelection !== 'all') {
        select.value = previousSelection;
    }

    setTimeout(() => {
        isUpdatingTopicFilter = false;
    }, 0);
}

/**
 * Shared tooltip configuration
 */
function getTooltipConfig() {
    return {
        show: true,
        trigger: 'item',
        backgroundColor: 'rgba(15, 23, 42, 0.95)',
        borderColor: '#3b82f6',
        borderWidth: 1,
        textStyle: { color: '#e2e8f0', fontSize: 13 },
        formatter: function(params: any) {
            if (params.dataType === 'node') {
                const d = params.data;
                let html = `<strong>${d.labelText || d.name}</strong>`;
                if (d.type) html += `<br/>Type: <span style="color:#94a3b8">${d.type}</span>`;
                if (d.topic) html += `<br/>Topic: <span style="color:#94a3b8">${d.topic}</span>`;
                if (d.count !== undefined) html += `<br/>Count: <span style="color:#3b82f6">${d.count}</span>`;
                if (d.citations !== undefined) html += `<br/>Citations: <span style="color:#34d399">${d.citations.toLocaleString()}</span>`;
                if (d.total_citations !== undefined) html += `<br/>Citations: <span style="color:#34d399">${d.total_citations.toLocaleString()}</span>`;
                if (d.pagerank !== undefined) html += `<br/>PageRank: <span style="color:#fbbf24">${d.pagerank.toFixed(2)}</span>`;
                if (d.normalized_score !== undefined) html += `<br/>Score: <span style="color:#f59e0b">${(d.normalized_score * 100).toFixed(0)}%</span>`;
                return html;
            } else if (params.dataType === 'edge') {
                const d = params.data;
                let html = `<strong>${d.relation || d.type || 'Connection'}</strong><br/>`;
                html += `${d.source} → ${d.target}`;
                if (d.count !== undefined) html += `<br/><span style="color:#3b82f6">${d.count}</span> occurrences`;
                return html;
            }
            return '';
        }
    };
}

/**
 * Blind spots tooltip configuration - shows percentile prominently
 */
function getBlindSpotsTooltipConfig(selectedTopic: string) {
    return {
        show: true,
        trigger: 'item',
        backgroundColor: 'rgba(15, 23, 42, 0.95)',
        borderColor: '#3b82f6',
        borderWidth: 1,
        padding: [10, 14],
        textStyle: { color: '#e2e8f0', fontSize: 12 },
        extraCssText: 'max-width: 400px; white-space: normal;',
        formatter: function(params: any) {
            if (params.dataType === 'node') {
                const d = params.data;

                // Topic node
                if (d.type === 'topic') {
                    let html = `<strong style="font-size:14px">${d.name || d.labelText || d.topic}</strong>`;
                    if (d.blind_spot_count !== undefined) {
                        html += `<br/><span style="color:#ef4444">Gap concepts: ${d.blind_spot_count}</span>`;
                    }
                    return html;
                }

                // Blind spot concept node
                if (d.type === 'blind_spot_concept' || d.type === 'bridging_concept') {
                    let html = `<strong style="font-size:14px">${d.name || d.labelText}</strong>`;

                    // Percentile at top - most important
                    const pct = typeof d.percentile === 'number' ? d.percentile : 0;
                    html += `<br/><span style="color:#ef4444;font-size:16px;font-weight:bold">${pct.toFixed(1)}%</span> <span style="color:#94a3b8">overall</span>`;

                    // Per-topic score breakdowns (sorted by percentile, highest first)
                    const topicScores = Array.isArray(d.topic_scores) ? d.topic_scores : [];
                    if (topicScores.length > 0) {
                        // Top 2 topics: full breakdown
                        const fullBreakdown = topicScores.slice(0, 2);
                        for (const ts of fullBreakdown) {
                            const topicName = (ts.topic || '').length > 30 ? ts.topic.substring(0, 27) + '...' : ts.topic;
                            html += `<br/><br/><span style="color:#f59e0b;font-weight:bold">${topicName}</span>`;
                            html += `<br/><span style="color:#ef4444;font-weight:bold">${(ts.percentile || 0).toFixed(0)}%</span> <span style="color:#64748b">overall</span>`;
                            html += `<br/><span style="color:#3b82f6">${(ts.importance_pct || 0).toFixed(0)}%</span> <span style="color:#64748b">importance</span>`;
                            html += `<br/><span style="color:#22c55e">${(ts.transferability_pct || 0).toFixed(0)}%</span> <span style="color:#64748b">transferability</span>`;
                            html += `<br/><span style="color:#f59e0b">${(ts.novelty_pct || 0).toFixed(0)}%</span> <span style="color:#64748b">novelty</span>`;
                            html += `<br/><span style="color:#8b5cf6">${(ts.topic_pair_pct || 0).toFixed(0)}%</span> <span style="color:#64748b">topic pair</span>`;
                        }
                        // Remaining topics: just name + percentile
                        const remaining = topicScores.slice(2, 6);
                        if (remaining.length > 0) {
                            html += `<br/>`;
                            for (const ts of remaining) {
                                const topicName = (ts.topic || '').length > 25 ? ts.topic.substring(0, 22) + '...' : ts.topic;
                                html += `<br/><span style="color:#94a3b8">${topicName}:</span> <span style="color:#ef4444">${(ts.percentile || 0).toFixed(0)}%</span>`;
                            }
                        }
                        if (topicScores.length > 6) {
                            html += `<br/><span style="color:#64748b">+${topicScores.length - 6} more</span>`;
                        }
                    }

                    return html;
                }

                // Default node
                return `<strong>${d.labelText || d.name}</strong>`;

            } else if (params.dataType === 'edge') {
                const d = params.data;

                // Get concept name from target (remove "concept:" prefix)
                const conceptName = (d.target || '').replace('concept:', '');
                // Get topic name from source (remove "topic:" prefix)
                const topicName = (d.source || '').replace('topic:', '');

                let html = `<strong style="font-size:14px">${conceptName}</strong>`;

                // Overall percentile
                const pct = typeof d.percentile === 'number' ? d.percentile : 0;
                html += `<br/><span style="color:#ef4444;font-size:16px;font-weight:bold">${pct.toFixed(0)}%</span> <span style="color:#94a3b8">overall</span>`;

                // Topic name
                html += `<br/><br/><span style="color:#f59e0b;font-weight:bold">${topicName}</span>`;

                // Score breakdown
                html += `<br/><span style="color:#ef4444;font-weight:bold">${pct.toFixed(0)}%</span> <span style="color:#64748b">overall</span>`;
                html += `<br/><span style="color:#3b82f6">${(d.importance_pct || 0).toFixed(0)}%</span> <span style="color:#64748b">importance</span>`;
                html += `<br/><span style="color:#22c55e">${(d.transferability_pct || 0).toFixed(0)}%</span> <span style="color:#64748b">transferability</span>`;
                html += `<br/><span style="color:#f59e0b">${(d.novelty_pct || 0).toFixed(0)}%</span> <span style="color:#64748b">novelty</span>`;
                html += `<br/><span style="color:#8b5cf6">${(d.topic_pair_pct || 0).toFixed(0)}%</span> <span style="color:#64748b">topic pair</span>`;

                return html;
            }
            return '';
        }
    };
}

/**
 * Shared label configuration
 */
function getLabelConfig() {
    return {
        show: true,
        position: 'right',
        formatter: '{b}',
        fontSize: 10,
        color: '#e2e8f0',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    };
}

/**
 * Shared emphasis configuration
 */
function getEmphasisConfig() {
    return {
        focus: 'adjacency',
        label: { show: true, fontSize: 12, fontWeight: 'bold' },
        itemStyle: {
            borderWidth: 3,
            borderColor: '#3b82f6',
            shadowBlur: 10,
            shadowColor: '#3b82f6'
        }
    };
}

/**
 * Render graph - dispatches to type-specific renderers
 */
function renderGraph(json: GraphData, yearLabel: string): void {
    setLoading(false);
    currentGraphData = json;

    const nodeCountEl = document.getElementById('node-count');
    const edgeCountEl = document.getElementById('edge-count');
    if (nodeCountEl) nodeCountEl.textContent = json.nodes.length.toString();
    if (edgeCountEl) edgeCountEl.textContent = json.edges.length.toString();

    switch (currentGraphType) {
        case 'concepts':
            renderCooccurrenceGraph(json, yearLabel);
            break;
        case 'paper_concepts':
            renderSemanticGraph(json);
            break;
        case 'ontology':
            if (ontologyLayout === 'circular') {
                renderOntologyCircular(json);
            } else {
                renderOntologyForce(json);
            }
            break;
        case 'blind_spots':
            if (crossTopicLayout === 'force') {
                renderBlindSpotsForce(json);
            } else {
                renderBlindSpotsBipartite(json);
            }
            break;
        default:
            renderForceGraph(json, yearLabel);
    }

    console.log(`Graph loaded: ${yearLabel}`);
    console.log(`Nodes: ${json.nodes.length}, Edges: ${json.edges.length}`);
}

/**
 * Render co-occurrence graph with pre-computed UMAP coordinates
 */
function renderCooccurrenceGraph(json: GraphData, yearLabel: string): void {
    const sizes = json.nodes.map((n: any) => n.size || 10);
    const minSize = Math.min(...sizes);
    const maxSize = Math.max(...sizes);

    function normalizeSize(size: number): number {
        if (maxSize === minSize) return 15;
        return 6 + ((size - minSize) / (maxSize - minSize)) * 22;
    }

    const option = {
        tooltip: getTooltipConfig(),
        animationDurationUpdate: 1500,
        animationEasingUpdate: 'quinticInOut',
        series: [{
            type: 'graph',
            layout: 'none',
            roam: true,
            scaleLimit: { min: 0.4, max: 2 },
            label: getLabelConfig(),
            labelLayout: { hideOverlap: true },
            data: json.nodes.map((node: any) => ({
                x: node.x,
                y: node.y,
                id: node.id,
                name: node.label || node.id,
                labelText: node.label || node.id,
                symbolSize: normalizeSize(node.size || 10),
                itemStyle: {
                    color: node.color || TYPE_COLORS[node.entity_type] || '#3b82f6',
                    borderColor: '#1e293b',
                    borderWidth: 2
                },
                ...node
            })),
            edges: json.edges.map((edge: any) => ({
                source: edge.sourceID || edge.source,
                target: edge.targetID || edge.target,
                lineStyle: {
                    width: Math.max(0.5, Math.min(edge.width || 1, 4)),
                    opacity: 0.5,
                    curveness: 0.3,
                    color: 'source'
                },
                ...edge
            })),
            emphasis: getEmphasisConfig(),
            lineStyle: { color: 'source', curveness: 0.3, opacity: 0.5 }
        }]
    };

    myChart.setOption(option, true);
}

/**
 * Render semantic (paper-concepts) graph
 */
function renderSemanticGraph(json: GraphData): void {
    const option = {
        tooltip: getTooltipConfig(),
        animationDurationUpdate: 1500,
        series: [{
            type: 'graph',
            layout: 'none',
            roam: true,
            scaleLimit: { min: 0.2, max: 4 },
            label: {
                show: true,
                position: 'right',
                formatter: (params: any) => {
                    const name = params.data.labelText || params.data.name || '';
                    return name.length > 25 ? name.substring(0, 25) + '...' : name;
                },
                fontSize: 9,
                color: '#e2e8f0'
            },
            labelLayout: { hideOverlap: true },
            data: json.nodes.map((node: any) => {
                const isPaper = node.type === 'paper';
                return {
                    x: node.x,
                    y: node.y,
                    id: node.id,
                    name: node.label || node.id,
                    labelText: node.label || node.id,
                    symbol: isPaper ? 'rect' : 'circle',
                    symbolSize: isPaper ? Math.max(10, Math.min(node.size || 12, 25)) : Math.max(4, Math.min(node.size || 6, 15)),
                    itemStyle: {
                        color: node.color || (isPaper ? '#27ae60' : '#3498db'),
                        borderColor: '#1e293b',
                        borderWidth: isPaper ? 2 : 1
                    },
                    ...node
                };
            }),
            edges: json.edges.map((edge: any) => ({
                source: edge.sourceID || edge.source,
                target: edge.targetID || edge.target,
                lineStyle: {
                    width: Math.max(0.3, Math.min(edge.width || 0.5, 2)),
                    opacity: 0.4,
                    curveness: 0.2,
                    color: edge.relation === 'proposes' ? '#e74c3c' : '#27ae60'
                },
                ...edge
            })),
            emphasis: getEmphasisConfig(),
            lineStyle: { curveness: 0.2, opacity: 0.4 }
        }]
    };

    myChart.setOption(option, true);
}

/**
 * Render ontology graph - force layout
 */
function renderOntologyForce(json: GraphData): void {
    const option = {
        tooltip: getTooltipConfig(),
        animationDurationUpdate: 1500,
        series: [{
            type: 'graph',
            layout: 'force',
            roam: true,
            scaleLimit: { min: 0.1, max: 4 },
            force: {
                repulsion: 80,
                gravity: 0.05,
                edgeLength: [30, 150],
                layoutAnimation: true
            },
            label: {
                show: true,
                position: 'right',
                formatter: '{b}',
                fontSize: 8,
                color: '#e2e8f0'
            },
            labelLayout: { hideOverlap: true },
            data: json.nodes.map((node: any) => ({
                id: node.id,
                name: node.label || node.id,
                labelText: node.label || node.id,
                symbolSize: Math.max(6, Math.min((node.edge_count || 1) * 1.5, 30)),
                itemStyle: {
                    color: node.from_triple ? '#34d399' : '#f59e0b',
                    borderColor: '#1e293b',
                    borderWidth: 1
                },
                ...node
            })),
            edges: json.edges.map((edge: any) => ({
                source: edge.source,
                target: edge.target,
                lineStyle: {
                    width: 0.5,
                    opacity: 0.4,
                    curveness: 0.2,
                    color: '#64748b'
                },
                ...edge
            })),
            emphasis: getEmphasisConfig(),
            lineStyle: { curveness: 0.2, opacity: 0.4 }
        }]
    };

    myChart.setOption(option, true);
}

/**
 * Render ontology with circular layout
 */
function renderOntologyCircular(json: GraphData): void {
    const option = {
        tooltip: getTooltipConfig(),
        animationDurationUpdate: 1500,
        series: [{
            type: 'graph',
            layout: 'circular',
            roam: true,
            scaleLimit: { min: 0.2, max: 4 },
            circular: { rotateLabel: true },
            label: {
                show: true,
                position: 'right',
                formatter: '{b}',
                fontSize: 8,
                color: '#e2e8f0'
            },
            labelLayout: { hideOverlap: true },
            data: json.nodes.map((node: any) => ({
                id: node.id,
                name: node.label || node.id,
                labelText: node.label || node.id,
                symbolSize: Math.max(5, Math.min((node.edge_count || 1), 25)),
                itemStyle: {
                    color: node.from_triple ? '#34d399' : '#f59e0b',
                    borderColor: '#1e293b',
                    borderWidth: 1
                },
                ...node
            })),
            edges: json.edges.map((edge: any) => ({
                source: edge.source,
                target: edge.target,
                lineStyle: {
                    width: 0.3,
                    opacity: 0.3,
                    curveness: 0.3,
                    color: '#64748b'
                },
                ...edge
            })),
            emphasis: getEmphasisConfig(),
            lineStyle: { curveness: 0.3, opacity: 0.3 }
        }]
    };

    myChart.setOption(option, true);
}

/**
 * Render blind spots with force layout (using UMAP positions)
 */
function renderBlindSpotsForce(json: GraphData): void {
    const topics = json.nodes.filter((n: any) => n.type === 'topic');
    const concepts = json.nodes.filter((n: any) => n.type === 'bridging_concept' || n.type === 'blind_spot_concept');
    const topicColors = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

    const scores = concepts.map(c => c.normalized_score ?? 0.5);
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);

    function getPercentileColor(score: number): string {
        if (maxScore === minScore) return '#3b82f6';
        const percentile = (score - minScore) / (maxScore - minScore);
        if (percentile >= 0.8) return '#ef4444';
        if (percentile >= 0.6) return '#f59e0b';
        if (percentile >= 0.4) return '#eab308';
        if (percentile >= 0.2) return '#22c55e';
        return '#3b82f6';
    }

    const positionedNodes = [
        ...topics.map((node: any, i: number) => {
            const labelText = node.label || node.id;
            return {
                ...node,
                labelText: labelText,
                symbolSize: 30,
                topicColor: topicColors[i % topicColors.length],
                itemStyle: {
                    color: topicColors[i % topicColors.length],
                    borderColor: '#fff',
                    borderWidth: 2
                },
                z: 10
            };
        }),
        ...concepts.map((node: any) => {
            const score = node.normalized_score ?? 0.5;
            return {
                ...node,
                labelText: node.label || node.id,
                symbolSize: 8 + Math.min((node.total_usage || 1) * 2, 20),
                itemStyle: {
                    color: getPercentileColor(score),
                    borderColor: '#1e293b',
                    borderWidth: 1
                }
            };
        })
    ];

    const option = {
        tooltip: getBlindSpotsTooltipConfig(currentTopic),
        animationDurationUpdate: 1500,
        series: [{
            type: 'graph',
            layout: 'none',
            roam: true,
            scaleLimit: { min: 0.2, max: 4 },
            label: {
                show: true,
                position: 'right',
                formatter: (params: any) => {
                    const name = params.data.labelText || params.data.name || '';
                    return name.length > 20 ? name.substring(0, 20) + '...' : name;
                },
                fontSize: 9,
                color: '#e2e8f0'
            },
            labelLayout: { hideOverlap: true },
            data: positionedNodes.map((node: any) => {
                const isTopicNode = node.type === 'topic';
                return {
                    ...node,
                    id: node.id,
                    name: node.labelText || node.id,
                    x: node.x,
                    y: node.y,
                    symbol: isTopicNode ? 'roundRect' : 'circle',
                    label: {
                        show: true,
                        position: 'right',
                        fontSize: isTopicNode ? 13 : 9,
                        fontWeight: isTopicNode ? 'bold' : 'normal',
                        color: isTopicNode ? '#fff' : '#e2e8f0',
                        textBorderColor: isTopicNode ? '#000' : 'transparent',
                        textBorderWidth: isTopicNode ? 2 : 0
                    },
                    z: isTopicNode ? 10 : 1
                };
            }),
            edges: json.edges.map((edge: any) => ({
                source: edge.source,
                target: edge.target,
                lineStyle: {
                    width: Math.max(0.5, Math.min((edge.count || 1) * 0.3, 3)),
                    opacity: 0.4,
                    curveness: 0.2,
                    color: '#64748b'
                },
                ...edge
            })),
            emphasis: getEmphasisConfig()
        }]
    };

    myChart.setOption(option, true);
}

/**
 * Render blind spots with bipartite layout
 */
function renderBlindSpotsBipartite(json: GraphData): void {
    const topics = json.nodes.filter((n: any) => n.type === 'topic');
    const concepts = json.nodes.filter((n: any) => n.type === 'bridging_concept' || n.type === 'blind_spot_concept');

    const chartHeight = 600;
    const topicSpacing = chartHeight / (topics.length + 1);
    const conceptSpacing = chartHeight / (concepts.length + 1);
    const topicColors = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

    const scores = concepts.map(c => c.normalized_score ?? 0.5);
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);

    function getPercentileColor(score: number): string {
        if (maxScore === minScore) return '#3b82f6';
        const percentile = (score - minScore) / (maxScore - minScore);
        if (percentile >= 0.8) return '#ef4444';
        if (percentile >= 0.6) return '#f59e0b';
        if (percentile >= 0.4) return '#eab308';
        if (percentile >= 0.2) return '#22c55e';
        return '#3b82f6';
    }

    const positionedNodes = [
        ...topics.map((node: any, i: number) => {
            const labelText = node.label || node.id;
            return {
                ...node,
                labelText: labelText,
                x: 150,
                y: (i + 1) * topicSpacing,
                symbolSize: 30,
                topicColor: topicColors[i % topicColors.length],
                itemStyle: {
                    color: topicColors[i % topicColors.length],
                    borderColor: '#fff',
                    borderWidth: 2
                },
                z: 10
            };
        }),
        ...concepts.map((node: any, i: number) => {
            const score = node.normalized_score ?? 0.5;
            return {
                ...node,
                labelText: node.label || node.id,
                x: 650,
                y: (i + 1) * conceptSpacing,
                symbolSize: 8 + Math.min((node.total_usage || 1) * 2, 20),
                itemStyle: {
                    color: getPercentileColor(score),
                    borderColor: '#1e293b',
                    borderWidth: 1
                }
            };
        })
    ];

    const topicColorMap = new Map(topics.map((t: any, i: number) => [t.id, topicColors[i % topicColors.length]]));

    const option = {
        tooltip: getBlindSpotsTooltipConfig(currentTopic),
        animationDurationUpdate: 1500,
        series: [{
            type: 'graph',
            layout: 'none',
            roam: true,
            scaleLimit: { min: 0.3, max: 3 },
            label: {
                show: true,
                position: 'right',
                formatter: (params: any) => {
                    const name = params.data.labelText || params.data.name || '';
                    if (params.data.type === 'topic') {
                        return name.split(' ').slice(0, 3).join(' ');
                    }
                    return name.length > 20 ? name.substring(0, 20) + '...' : name;
                },
                fontSize: 9,
                color: '#e2e8f0'
            },
            data: positionedNodes.map((node: any) => {
                const isTopicNode = node.type === 'topic';
                return {
                    ...node,
                    id: node.id,
                    name: node.labelText || node.id,
                    fixed: true,
                    symbol: isTopicNode ? 'roundRect' : 'circle',
                    label: {
                        show: true,
                        position: 'right',
                        fontSize: isTopicNode ? 13 : 9,
                        fontWeight: isTopicNode ? 'bold' : 'normal',
                        color: isTopicNode ? '#fff' : '#e2e8f0',
                        textBorderColor: isTopicNode ? '#000' : 'transparent',
                        textBorderWidth: isTopicNode ? 2 : 0
                    },
                    z: isTopicNode ? 10 : 1
                };
            }),
            edges: json.edges.map((edge: any) => ({
                source: edge.source,
                target: edge.target,
                lineStyle: {
                    width: Math.max(1, Math.min((edge.count || 1) * 0.8, 6)),
                    opacity: 0.6,
                    curveness: 0.2,
                    color: topicColorMap.get(edge.source) || '#64748b'
                },
                ...edge
            })),
            emphasis: {
                focus: 'adjacency',
                itemStyle: {
                    borderWidth: 4,
                    shadowBlur: 15,
                    shadowColor: 'rgba(59, 130, 246, 0.5)'
                }
            }
        }]
    };

    myChart.setOption(option, true);
}

/**
 * Generic force layout fallback
 */
function renderForceGraph(json: GraphData, label: string): void {
    const option = {
        tooltip: getTooltipConfig(),
        animationDurationUpdate: 1500,
        series: [{
            type: 'graph',
            layout: 'force',
            roam: true,
            force: { repulsion: 100, gravity: 0.1, edgeLength: [50, 200] },
            label: getLabelConfig(),
            data: json.nodes.map((node: any) => ({
                id: node.id,
                name: node.label || node.id,
                labelText: node.label || node.id,
                symbolSize: 15,
                itemStyle: { color: '#3b82f6', borderColor: '#1e293b', borderWidth: 2 },
                ...node
            })),
            edges: json.edges.map((edge: any) => ({
                source: edge.sourceID || edge.source,
                target: edge.targetID || edge.target,
                lineStyle: { width: 1, opacity: 0.5, color: '#64748b' },
                ...edge
            })),
            emphasis: getEmphasisConfig()
        }]
    };
    myChart.setOption(option, true);
}

/**
 * Load and display graph for selected year
 */
async function loadSelectedYear(): Promise<void> {
    const yearValue = availableYears[currentYearIndex].value;
    const yearLabel = availableYears[currentYearIndex].label;
    const graphFile = availableYears[currentYearIndex].file;

    if (!graphFile) return;

    setLoading(true, `Loading ${yearLabel}...`);

    try {
        const json = await loadGraph(graphFile);
        originalGraphData = json;
        const topics = extractTopics(json);
        updateTopicFilter(topics);
        const filteredData = filterByTopic(json, currentTopic);
        renderGraph(filteredData, yearLabel);

        const yearSlider = document.getElementById('year-slider') as HTMLInputElement | null;
        const yearDisplay = document.getElementById('year-display');
        if (yearSlider) yearSlider.value = currentYearIndex.toString();
        if (yearDisplay) yearDisplay.textContent = yearLabel;
    } catch (error) {
        setLoading(false);
        console.error(`Failed to load graph for ${yearLabel}:`, error);
        alert(`Failed to load graph data for ${yearLabel}.`);
    }
}

/**
 * Go to next year (file-based)
 */
function nextYear(): void {
    currentYearIndex = (currentYearIndex + 1) % availableYears.length;
    loadSelectedYear();
}

/**
 * Apply all active filters to data based on graph type
 */
function applyAllFilters(data: GraphData, graphType: string): GraphData {
    const config = GRAPH_TYPES[graphType];
    let filteredData = data;

    if (config.hasYears && config.yearSource === 'data' && currentYear !== 'all') {
        filteredData = filterByYear(filteredData, currentYear);
    }

    if (config.hasTopics) {
        if (graphType === 'blind_spots' && showGaps && currentTopic !== 'all') {
            filteredData = filterCrossTopicGaps(filteredData, currentTopic);
        } else {
            filteredData = filterByTopic(filteredData, currentTopic);
        }
    }

    if (graphType === 'paper_concepts') {
        filteredData = filterByRelation(filteredData, currentRelation);
    }

    if (graphType === 'ontology' && ontologyFilter === 'overlap') {
        filteredData = filterOntologyOverlap(filteredData);
    }

    if (graphType === 'blind_spots' && showBestOnly) {
        filteredData = filterBlindSpotsTop(filteredData, 0.05);
    }

    return filteredData;
}

/**
 * Get current year label for display
 */
function getYearLabel(): string {
    if (availableYears.length > 0 && currentYearIndex < availableYears.length) {
        return availableYears[currentYearIndex].label;
    }
    return GRAPH_TYPES[currentGraphType]?.name || 'Graph';
}

/**
 * Start slideshow
 */
function startPlay(): void {
    if (isPlaying) return;

    isPlaying = true;
    const button = document.getElementById('play-button');
    if (!button) return;

    const icon = button.querySelector('i');
    if (icon) {
        icon.setAttribute('data-lucide', 'pause');
        (window as any).lucide?.createIcons();
    }
    button.classList.add('active');

    nextYearData();
    playInterval = setInterval(nextYearData, PLAY_INTERVAL_MS) as unknown as number;
}

/**
 * Stop slideshow
 */
function stopPlay(): void {
    if (!isPlaying) return;

    isPlaying = false;
    const button = document.getElementById('play-button');
    if (!button) return;

    const icon = button.querySelector('i');
    if (icon) {
        icon.setAttribute('data-lucide', 'play');
        (window as any).lucide?.createIcons();
    }
    button.classList.remove('active');

    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
    }
}

/**
 * Toggle play/pause
 */
function togglePlay(): void {
    if (isPlaying) {
        stopPlay();
    } else {
        startPlay();
    }
}

/**
 * Load graph by type
 */
async function loadGraphByType(graphType: string): Promise<void> {
    const config = GRAPH_TYPES[graphType];
    if (!config) return;

    setLoading(true, `Loading ${config.name}...`);

    try {
        const json = await loadGraph(config.file);
        originalGraphData = json;

        if (config.hasTopics) {
            const topics = extractTopics(json);
            updateTopicFilter(topics);
        }

        if (config.hasYears && config.yearSource === 'data') {
            const years = extractYearsFromData(json);
            setupYearSliderFromData(years);
        }

        const filteredData = applyAllFilters(json, graphType);
        renderGraph(filteredData, getYearLabel());
    } catch (error) {
        setLoading(false);
        console.error(`Failed to load ${config.name}:`, error);
        alert(`Failed to load ${config.name} graph.`);
    }
}

/**
 * Switch graph type
 */
async function switchGraphType(graphType: string): Promise<void> {
    stopPlay();
    currentGraphType = graphType;
    currentTopic = 'all';
    currentRelation = 'all';
    currentYear = 'all';
    currentYearIndex = 0;
    showGaps = (graphType === 'blind_spots');  // Default to gaps for blind spots view

    updateLegend(graphType);

    const config = GRAPH_TYPES[graphType];
    const yearControls = document.getElementById('year-controls');
    const topicSelect = document.getElementById('topic-filter') as HTMLSelectElement | null;
    const relationSelect = document.getElementById('relation-filter') as HTMLSelectElement | null;
    const layoutToggle = document.getElementById('layout-toggle-container');
    const ontologyFilterToggle = document.getElementById('ontology-filter-container');
    const gapsToggle = document.getElementById('gaps-toggle-container');
    const bestToggle = document.getElementById('best-toggle-container');

    if (yearControls) {
        yearControls.style.display = config.hasYears ? 'flex' : 'none';
    }

    if (topicSelect) {
        topicSelect.style.display = config.hasTopics ? 'block' : 'none';
        topicSelect.value = 'all';
    }

    if (relationSelect) {
        relationSelect.style.display = graphType === 'paper_concepts' ? 'block' : 'none';
        relationSelect.value = 'all';
    }

    if (layoutToggle) {
        const hasLayoutToggle = graphType === 'ontology' || graphType === 'blind_spots';
        layoutToggle.style.display = hasLayoutToggle ? 'flex' : 'none';
        if (hasLayoutToggle) {
            updateLayoutToggleLabel(graphType);
            if (graphType === 'ontology') {
                ontologyLayout = 'force';
            } else if (graphType === 'blind_spots') {
                crossTopicLayout = 'force';
            }
            updateLayoutToggleStyle(true);
        }
    }

    if (ontologyFilterToggle) {
        ontologyFilterToggle.style.display = graphType === 'ontology' ? 'flex' : 'none';
        if (graphType === 'ontology') {
            ontologyFilter = 'overlap';
            updateOntologyFilterStyle(true);
        }
    }

    if (gapsToggle) {
        gapsToggle.style.display = 'none';
    }

    if (bestToggle) {
        bestToggle.style.display = graphType === 'blind_spots' ? 'flex' : 'none';
        if (graphType === 'blind_spots') {
            showBestOnly = true;
            updateBestToggleStyle(true);
        }
    }

    if (config.hasYears && config.yearSource === 'files') {
        availableYears = await discoverGraphFiles();
        const yearSlider = document.getElementById('year-slider') as HTMLInputElement | null;
        if (yearSlider) {
            yearSlider.max = (availableYears.length - 1).toString();
            yearSlider.value = '0';
        }
        currentYearIndex = 0;
        await loadSelectedYear();
    } else {
        await loadGraphByType(graphType);
    }
}

/**
 * Handle topic filter change
 */
async function handleTopicChange(topic: string): Promise<void> {
    if (isUpdatingTopicFilter) return;

    currentTopic = topic;

    const gapsToggle = document.getElementById('gaps-toggle-container');
    if (gapsToggle && currentGraphType === 'blind_spots') {
        if (topic === 'all') {
            gapsToggle.style.display = 'none';
            showGaps = true;  // Keep gaps as default
            updateGapsToggleStyle(true);
        } else {
            gapsToggle.style.display = 'flex';
            updateGapsToggleStyle(true);  // Show gaps toggle as active
        }
    }

    if (originalGraphData) {
        const filteredData = applyAllFilters(originalGraphData, currentGraphType);
        renderGraph(filteredData, getYearLabel());
    }
}

/**
 * Handle relation filter change
 */
async function handleRelationChange(relation: string): Promise<void> {
    currentRelation = relation;
    if (originalGraphData) {
        const filteredData = applyAllFilters(originalGraphData, currentGraphType);
        renderGraph(filteredData, getYearLabel());
    }
}

/**
 * Update layout toggle button styles
 */
function updateLayoutToggleStyle(isForce: boolean): void {
    const forceBtn = document.getElementById('layout-force');
    const altBtn = document.getElementById('layout-alt');
    if (forceBtn && altBtn) {
        if (isForce) {
            forceBtn.classList.add('active');
            altBtn.classList.remove('active');
        } else {
            forceBtn.classList.remove('active');
            altBtn.classList.add('active');
        }
    }
}

/**
 * Update layout toggle label based on graph type
 */
function updateLayoutToggleLabel(graphType: string): void {
    const label = document.getElementById('layout-alt-label');
    const forceLabel = document.getElementById('layout-force');
    if (label) {
        if (graphType === 'ontology') {
            label.textContent = 'Circular';
            if (forceLabel) forceLabel.textContent = 'Force';
        } else if (graphType === 'blind_spots') {
            label.textContent = 'Bipartite';
            if (forceLabel) forceLabel.textContent = 'Graph';
        }
    }
}

/**
 * Update ontology filter toggle button styles
 */
function updateOntologyFilterStyle(isOverlap: boolean): void {
    const overlapBtn = document.getElementById('filter-overlap');
    const fullBtn = document.getElementById('filter-full');
    if (overlapBtn && fullBtn) {
        if (isOverlap) {
            overlapBtn.classList.add('active');
            fullBtn.classList.remove('active');
        } else {
            overlapBtn.classList.remove('active');
            fullBtn.classList.add('active');
        }
    }
}

/**
 * Handle ontology filter change to overlap
 */
function handleOntologyOverlap(): void {
    ontologyFilter = 'overlap';
    updateOntologyFilterStyle(true);
    if (originalGraphData) {
        const filteredData = applyAllFilters(originalGraphData, currentGraphType);
        renderGraph(filteredData, getYearLabel());
    }
}

/**
 * Handle ontology filter change to full
 */
function handleOntologyFull(): void {
    ontologyFilter = 'full';
    updateOntologyFilterStyle(false);
    if (originalGraphData) {
        const filteredData = applyAllFilters(originalGraphData, currentGraphType);
        renderGraph(filteredData, getYearLabel());
    }
}

/**
 * Update gaps toggle button styles
 */
function updateGapsToggleStyle(showGapsActive: boolean): void {
    const gapsOffBtn = document.getElementById('gaps-off');
    const gapsOnBtn = document.getElementById('gaps-on');
    if (gapsOffBtn && gapsOnBtn) {
        if (showGapsActive) {
            gapsOffBtn.classList.remove('active');
            gapsOnBtn.classList.add('active');
        } else {
            gapsOffBtn.classList.add('active');
            gapsOnBtn.classList.remove('active');
        }
    }
}

/**
 * Handle gaps toggle - show shared concepts
 */
function handleGapsOff(): void {
    showGaps = false;
    updateGapsToggleStyle(false);
    if (originalGraphData) {
        const filteredData = applyAllFilters(originalGraphData, currentGraphType);
        renderGraph(filteredData, getYearLabel());
    }
}

/**
 * Handle gaps toggle - show gap concepts
 */
function handleGapsOn(): void {
    showGaps = true;
    updateGapsToggleStyle(true);
    if (originalGraphData) {
        const filteredData = applyAllFilters(originalGraphData, currentGraphType);
        renderGraph(filteredData, getYearLabel());
    }
}

/**
 * Update best toggle button styles
 */
function updateBestToggleStyle(isBest: boolean): void {
    const bestBtn = document.getElementById('filter-best');
    const allBtn = document.getElementById('filter-all');
    if (bestBtn && allBtn) {
        if (isBest) {
            bestBtn.classList.add('active');
            allBtn.classList.remove('active');
        } else {
            bestBtn.classList.remove('active');
            allBtn.classList.add('active');
        }
    }
}

/**
 * Handle best filter - show only top 5% concepts
 */
function handleBestFilter(): void {
    showBestOnly = true;
    updateBestToggleStyle(true);
    if (originalGraphData) {
        const filteredData = applyAllFilters(originalGraphData, currentGraphType);
        renderGraph(filteredData, getYearLabel());
    }
}

/**
 * Handle all filter - show all concepts
 */
function handleAllFilter(): void {
    showBestOnly = false;
    updateBestToggleStyle(false);
    if (originalGraphData) {
        const filteredData = applyAllFilters(originalGraphData, currentGraphType);
        renderGraph(filteredData, getYearLabel());
    }
}

/**
 * Handle layout toggle (force button)
 */
function handleForceLayout(): void {
    if (currentGraphType === 'ontology') {
        ontologyLayout = 'force';
    } else if (currentGraphType === 'blind_spots') {
        crossTopicLayout = 'force';
    }
    updateLayoutToggleStyle(true);
    if (originalGraphData) {
        const filteredData = applyAllFilters(originalGraphData, currentGraphType);
        renderGraph(filteredData, getYearLabel());
    }
}

/**
 * Handle layout toggle (alt button)
 */
function handleAltLayout(): void {
    if (currentGraphType === 'ontology') {
        ontologyLayout = 'circular';
    } else if (currentGraphType === 'blind_spots') {
        crossTopicLayout = 'fixed';
    }
    updateLayoutToggleStyle(false);
    if (originalGraphData) {
        const filteredData = applyAllFilters(originalGraphData, currentGraphType);
        renderGraph(filteredData, getYearLabel());
    }
}

/**
 * Handle year slider change for data-based graphs
 */
function handleYearSliderChange(index: number): void {
    const config = GRAPH_TYPES[currentGraphType];
    if (!config) return;

    currentYearIndex = index;

    if (config.yearSource === 'data') {
        currentYear = availableYears[index]?.value || 'all';
        const yearDisplay = document.getElementById('year-display');
        if (yearDisplay) {
            yearDisplay.textContent = availableYears[index]?.label || 'All Years';
        }
        if (originalGraphData) {
            const filteredData = applyAllFilters(originalGraphData, currentGraphType);
            renderGraph(filteredData, getYearLabel());
        }
    } else {
        loadSelectedYear();
    }
}

/**
 * Handle year play for data-based graphs
 */
function nextYearData(): void {
    const config = GRAPH_TYPES[currentGraphType];
    currentYearIndex = (currentYearIndex + 1) % availableYears.length;

    const yearSlider = document.getElementById('year-slider') as HTMLInputElement | null;
    if (yearSlider) {
        yearSlider.value = currentYearIndex.toString();
    }

    if (config?.yearSource === 'data') {
        handleYearSliderChange(currentYearIndex);
    } else {
        loadSelectedYear();
    }
}

// Initialize the app
(async function init() {
    try {
        availableYears = await discoverGraphFiles();

        if (availableYears.length === 0) {
            setLoading(false);
            alert('No graph files found. Please run _9_gen_graphs.py first.');
            return;
        }

        const yearSlider = document.getElementById('year-slider') as HTMLInputElement | null;
        const yearDisplay = document.getElementById('year-display');
        const playButton = document.getElementById('play-button');
        const graphTypeSelect = document.getElementById('graph-type') as HTMLSelectElement | null;
        const topicSelect = document.getElementById('topic-filter') as HTMLSelectElement | null;
        const layoutToggle = document.getElementById('layout-toggle-container');

        if (!yearSlider || !yearDisplay || !playButton) {
            throw new Error('Required elements not found');
        }

        yearSlider.max = (availableYears.length - 1).toString();
        yearSlider.value = '0';

        if (layoutToggle) {
            layoutToggle.style.display = 'none';
        }

        function updateYearDisplay() {
            if (yearDisplay) {
                yearDisplay.textContent = availableYears[currentYearIndex].label;
            }
        }

        updateLegend('concepts');
        await loadSelectedYear();
        updateYearDisplay();

        yearSlider.addEventListener('input', function(this: HTMLInputElement) {
            stopPlay();
            currentYearIndex = parseInt(this.value);
            updateYearDisplay();
        });

        yearSlider.addEventListener('change', function(this: HTMLInputElement) {
            handleYearSliderChange(parseInt(this.value));
        });

        playButton.addEventListener('click', togglePlay);

        if (graphTypeSelect) {
            graphTypeSelect.addEventListener('change', function(this: HTMLSelectElement) {
                switchGraphType(this.value);
            });
        }

        if (topicSelect) {
            topicSelect.addEventListener('change', function(this: HTMLSelectElement) {
                handleTopicChange(this.value);
            });
        }

        const relationSelect = document.getElementById('relation-filter') as HTMLSelectElement | null;
        if (relationSelect) {
            relationSelect.addEventListener('change', function(this: HTMLSelectElement) {
                handleRelationChange(this.value);
            });
        }

        const layoutForceBtn = document.getElementById('layout-force');
        const layoutAltBtn = document.getElementById('layout-alt');
        if (layoutForceBtn) {
            layoutForceBtn.addEventListener('click', handleForceLayout);
        }
        if (layoutAltBtn) {
            layoutAltBtn.addEventListener('click', handleAltLayout);
        }

        const filterOverlapBtn = document.getElementById('filter-overlap');
        const filterFullBtn = document.getElementById('filter-full');
        if (filterOverlapBtn) {
            filterOverlapBtn.addEventListener('click', handleOntologyOverlap);
        }
        if (filterFullBtn) {
            filterFullBtn.addEventListener('click', handleOntologyFull);
        }

        const gapsOffBtn = document.getElementById('gaps-off');
        const gapsOnBtn = document.getElementById('gaps-on');
        if (gapsOffBtn) {
            gapsOffBtn.addEventListener('click', handleGapsOff);
        }
        if (gapsOnBtn) {
            gapsOnBtn.addEventListener('click', handleGapsOn);
        }

        const filterBestBtn = document.getElementById('filter-best');
        const filterAllBtn = document.getElementById('filter-all');
        if (filterBestBtn) {
            filterBestBtn.addEventListener('click', handleBestFilter);
        }
        if (filterAllBtn) {
            filterAllBtn.addEventListener('click', handleAllFilter);
        }

        window.addEventListener('resize', function() {
            myChart.resize();
        });

        console.log(`Initialized with ${availableYears.length} graphs`);

    } catch (error) {
        setLoading(false);
        console.error('Initialization error:', error);
        alert('Failed to initialize graph viewer.');
    }
})();
