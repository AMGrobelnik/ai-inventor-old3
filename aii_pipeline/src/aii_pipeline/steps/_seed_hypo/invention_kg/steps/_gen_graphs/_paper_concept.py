#!/usr/bin/env python3
"""
Paper-Concept bipartite graph generator.

Creates a bipartite graph:
- Paper nodes: Research papers with metadata
- Concept nodes: Concepts from triples
- Edges: Paper mentions/uses concept with relation type
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from aii_lib import AIITelemetry, MessageType

# ============================================================================
# Module-level telemetry (set by caller via set_telemetry)
# ============================================================================
_telemetry: Optional[AIITelemetry] = None


def set_telemetry(telemetry: Optional[AIITelemetry]):
    """Set module telemetry from calling code."""
    global _telemetry
    _telemetry = telemetry


def _emit(msg_type: MessageType, msg: str):
    """Emit to telemetry if available, otherwise print."""
    if _telemetry:
        _telemetry.emit(msg_type, msg)
    else:
        print(f"[{msg_type.name}] {msg}")


def extract_paper_concept_data(papers: List[Dict]) -> Dict[str, Any]:
    """
    Extract paper-concept relationships.

    Returns:
        Dictionary with paper_nodes, concept_nodes, edges
    """
    paper_nodes: Dict[str, Dict[str, Any]] = {}
    concept_nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    for paper in papers:
        paper_idx = paper.get("index", -1)
        paper_data = paper.get("paper", {}) or {}

        paper_id = f"paper:{paper_idx}"
        title = paper_data.get("title", f"Paper {paper_idx}")

        # Add paper node
        if paper_id not in paper_nodes:
            paper_nodes[paper_id] = {
                "id": paper_id,
                "label": title[:50] + "..." if len(title) > 50 else title,
                "full_title": title,
                "type": "paper",
                "year": paper_data.get("publication_year"),
                "citations": paper_data.get("cited_by_count", 0),
                "topic": paper_data.get("topic_name"),
                "doi": paper_data.get("doi"),
            }

        # Extract triples
        triples_section = paper.get("triples", {}) or {}
        triples = triples_section.get("triples", [])

        for triple in triples:
            name = triple.get("name")
            if not name:
                continue

            concept_id = f"concept:{name}"

            # Add concept node
            if concept_id not in concept_nodes:
                concept_nodes[concept_id] = {
                    "id": concept_id,
                    "label": name,
                    "type": "concept",
                    "entity_type": triple.get("entity_type"),
                    "wikipedia_url": triple.get("wikipedia_url"),
                    "wikidata_id": triple.get("wikidata_id"),
                    "paper_count": 0,
                    "total_citations": 0,
                }

            concept_nodes[concept_id]["paper_count"] += 1
            concept_nodes[concept_id]["total_citations"] += paper_data.get("cited_by_count", 0)

            # Add edge with relation
            edges.append({
                "source": paper_id,
                "target": concept_id,
                "relation": triple.get("relation", "uses"),
                "relevance": triple.get("relevance"),
            })

    return {
        "paper_nodes": paper_nodes,
        "concept_nodes": concept_nodes,
        "edges": edges,
    }


def build_paper_concept_graph(data: Dict[str, Any]) -> Dict[str, Any]:
    """Build the paper-concept graph JSON."""
    paper_nodes = data["paper_nodes"]
    concept_nodes = data["concept_nodes"]
    edges = data["edges"]

    # Build nodes list
    nodes = []

    # Add paper nodes (sized by citations)
    max_citations = max((n["citations"] for n in paper_nodes.values()), default=1)
    for paper in paper_nodes.values():
        citation_ratio = paper["citations"] / max_citations if max_citations > 0 else 0
        nodes.append({
            **paper,
            "size": 6 + citation_ratio * 20,
        })

    # Add concept nodes (sized by paper count)
    max_papers = max((n["paper_count"] for n in concept_nodes.values()), default=1)
    for concept in concept_nodes.values():
        paper_ratio = concept["paper_count"] / max_papers if max_papers > 0 else 0
        nodes.append({
            **concept,
            "size": 8 + paper_ratio * 25,
        })

    # Count relations
    relation_counts = defaultdict(int)
    for edge in edges:
        relation_counts[edge["relation"]] += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "paper_count": len(paper_nodes),
            "concept_count": len(concept_nodes),
            "edge_count": len(edges),
            "relation_counts": dict(relation_counts),
        }
    }


def generate_paper_concept_graph(papers: List[Dict], output_file: Path) -> bool:
    """Generate and save paper-concept bipartite graph."""
    _emit(MessageType.INFO, f"Generating paper-concept graph ({len(papers)} papers)")

    data = extract_paper_concept_data(papers)

    if not data["concept_nodes"]:
        _emit(MessageType.WARNING, "No concepts found")
        return False

    graph_json = build_paper_concept_graph(data)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_json, f, indent=2, ensure_ascii=False)
        _emit(
            MessageType.SUCCESS,
            f"Saved paper-concept graph: {output_file.name} "
            f"({graph_json['metadata']['paper_count']} papers, "
            f"{graph_json['metadata']['concept_count']} concepts, "
            f"{graph_json['metadata']['edge_count']} edges)"
        )
        return True
    except Exception as e:
        _emit(MessageType.ERROR, f"Failed to save graph: {e}")
        return False
