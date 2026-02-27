#!/usr/bin/env python3
"""
Relation-typed knowledge graph generator.

Creates the "classic" KG with:
- Paper nodes
- Concept nodes
- Typed edges: uses, proposes, extends, etc.

This is the raw triple data as a graph, preserving all relation types.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from aii_lib import AIITelemetry, MessageType

from ._common import generate_umap_layout

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


def extract_relation_typed_data(papers: List[Dict]) -> Dict[str, Any]:
    """
    Extract relation-typed graph data.

    Returns:
        Dictionary with paper_nodes, concept_nodes, edges
    """
    paper_nodes: Dict[str, Dict[str, Any]] = {}
    concept_nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    # Track concept-to-concept relations via shared papers
    concept_cooccurrence: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "papers": [], "relations": defaultdict(int)}
    )

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
                "concept_count": 0,
            }

        # Extract triples
        triples_section = paper.get("triples", {}) or {}
        triples = triples_section.get("triples", [])

        paper_concepts = []  # Track concepts in this paper

        for triple in triples:
            name = triple.get("name")
            if not name:
                continue

            paper_concepts.append(name)
            paper_nodes[paper_id]["concept_count"] += 1

            # Add concept node
            if name not in concept_nodes:
                concept_nodes[name] = {
                    "id": name,
                    "label": name,
                    "type": "concept",
                    "entity_type": triple.get("entity_type"),
                    "wikipedia_url": triple.get("wikipedia_url"),
                    "wikidata_id": triple.get("wikidata_id"),
                    "paper_count": 0,
                    "total_citations": 0,
                    "relation_counts": defaultdict(int),
                }

            concept_nodes[name]["paper_count"] += 1
            concept_nodes[name]["total_citations"] += paper_data.get("cited_by_count", 0)

            relation = triple.get("relation", "uses")
            concept_nodes[name]["relation_counts"][relation] += 1

            # Add paper -> concept edge
            edges.append({
                "source": paper_id,
                "target": name,
                "type": "paper_concept",
                "relation": relation,
                "relevance": triple.get("relevance"),
            })

        # Track concept co-occurrences with relation context
        for i, c1 in enumerate(paper_concepts):
            for c2 in paper_concepts[i+1:]:
                key = tuple(sorted([c1, c2]))
                concept_cooccurrence[key]["count"] += 1
                concept_cooccurrence[key]["papers"].append(paper_idx)

    # Convert concept relation_counts from defaultdict
    for concept in concept_nodes.values():
        concept["relation_counts"] = dict(concept["relation_counts"])

    # Add concept-concept edges from co-occurrence
    concept_edges = []
    for (c1, c2), data in concept_cooccurrence.items():
        concept_edges.append({
            "source": c1,
            "target": c2,
            "type": "concept_cooccurrence",
            "count": data["count"],
            "papers": data["papers"],
        })

    return {
        "paper_nodes": paper_nodes,
        "concept_nodes": concept_nodes,
        "paper_concept_edges": edges,
        "concept_concept_edges": concept_edges,
    }


def build_relation_typed_graph(data: Dict[str, Any]) -> Dict[str, Any]:
    """Build the relation-typed graph JSON."""
    paper_nodes = data["paper_nodes"]
    concept_nodes = data["concept_nodes"]
    paper_concept_edges = data["paper_concept_edges"]
    concept_concept_edges = data["concept_concept_edges"]

    nodes = []

    # Add paper nodes
    max_citations = max((n["citations"] for n in paper_nodes.values()), default=1)
    for paper in paper_nodes.values():
        citation_ratio = paper["citations"] / max_citations if max_citations > 0 else 0
        nodes.append({
            **paper,
            "size": 6 + citation_ratio * 15,
            "color": "#4a90d9",  # Blue for papers
        })

    # Add concept nodes
    max_papers = max((n["paper_count"] for n in concept_nodes.values()), default=1)
    for concept in concept_nodes.values():
        paper_ratio = concept["paper_count"] / max_papers if max_papers > 0 else 0

        # Color by dominant relation
        relations = concept.get("relation_counts", {})
        if relations.get("proposes", 0) > relations.get("uses", 0):
            color = "#2ecc71"  # Green for proposed concepts
        elif relations.get("extends", 0) > relations.get("uses", 0):
            color = "#f39c12"  # Orange for extended concepts
        else:
            color = "#e74c3c"  # Red for used concepts

        nodes.append({
            **concept,
            "size": 8 + paper_ratio * 25,
            "color": color,
        })

    # Combine edges
    all_edges = paper_concept_edges + concept_concept_edges

    # Count relations
    relation_counts = defaultdict(int)
    for edge in paper_concept_edges:
        relation_counts[edge["relation"]] += 1

    return {
        "nodes": nodes,
        "edges": all_edges,
        "metadata": {
            "paper_count": len(paper_nodes),
            "concept_count": len(concept_nodes),
            "paper_concept_edges": len(paper_concept_edges),
            "concept_concept_edges": len(concept_concept_edges),
            "relation_counts": dict(relation_counts),
        }
    }


def generate_relation_typed_graph(papers: List[Dict], output_file: Path) -> bool:
    """Generate and save relation-typed knowledge graph."""
    _emit(MessageType.INFO, f"Generating relation-typed KG ({len(papers)} papers)")

    data = extract_relation_typed_data(papers)

    if not data["concept_nodes"]:
        _emit(MessageType.WARNING, "No concepts found")
        return False

    graph_json = build_relation_typed_graph(data)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_json, f, indent=2, ensure_ascii=False)
        _emit(
            MessageType.SUCCESS,
            f"Saved relation-typed KG: {output_file.name} "
            f"({graph_json['metadata']['paper_count']} papers, "
            f"{graph_json['metadata']['concept_count']} concepts)"
        )
        return True
    except Exception as e:
        _emit(MessageType.ERROR, f"Failed to save graph: {e}")
        return False
