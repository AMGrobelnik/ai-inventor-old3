#!/usr/bin/env python3
"""
Topic Blind Spots Opportunity Finder (Concept-Centric).

Finds concepts that a "blind topic" is missing by comparing it to a
semantically dissimilar "reference topic" that shares some concepts.

Each blind spot concept is its own entry with rich metrics for ranking.

Logic:
1. Find topic pairs that are semantically dissimilar but share some concepts
2. For each pair, identify concepts the ref_topic uses that blind_topic doesn't
3. Each concept becomes an individual "blind spot" opportunity with metrics
"""

import re
import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict
from aii_lib import AIITelemetry, MessageType
import numpy as np

# Optional: sentence-transformers for semantic distance
try:
    from sentence_transformers import SentenceTransformer
    _semantic_model = None  # Lazy load
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    _semantic_model = None
    HAS_SENTENCE_TRANSFORMERS = False


# Module-level telemetry (set by caller via set_telemetry)
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

def slugify(text: str) -> str:
    """Convert text to URL/ID-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)  # remove special chars
    text = re.sub(r'[\s-]+', '_', text)   # spaces/hyphens → underscore
    return text.strip('_')


def extract_topic_data(papers: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Extract comprehensive topic data including concept details.

    Returns:
        {topic_name: {
            "field": str,
            "subfield": str,
            "paper_count": int,
            "concepts": {concept_name: {
                "count": int,
                "uses": int,
                "proposes": int,
                "entity_type": str,
                "papers": [{paper_id, citations, year}, ...]
            }},
            "concept_cooccurrence": {concept: {other_concept: count}}
        }}
    """
    topic_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "field": None,
        "subfield": None,
        "paper_count": 0,
        "concepts": defaultdict(lambda: {
            "count": 0,
            "uses": 0,
            "proposes": 0,
            "entity_type": None,
            "papers": []
        }),
        "concept_cooccurrence": defaultdict(lambda: defaultdict(int))
    })

    for paper in papers:
        paper_data = paper.get("paper", {}) or {}
        topic_name = paper_data.get("topic_name")
        if not topic_name:
            continue

        paper_id = paper_data.get("id", "")
        citations = paper_data.get("cited_by_count", 0)
        year = paper_data.get("publication_year", 0)

        topic_data[topic_name]["paper_count"] += 1
        if topic_data[topic_name]["field"] is None:
            topic_data[topic_name]["field"] = paper_data.get("field")
            topic_data[topic_name]["subfield"] = paper_data.get("subfield")

        # Extract concepts from triples
        triples_section = paper.get("triples", {}) or {}
        triples = triples_section.get("triples", [])

        # Track concepts in this paper for co-occurrence
        paper_concepts = set()

        for triple in triples:
            name = triple.get("name")
            if not name:
                continue

            paper_concepts.add(name)
            concept_data = topic_data[topic_name]["concepts"][name]
            concept_data["count"] += 1

            relation = triple.get("relation", "uses")
            if relation == "uses":
                concept_data["uses"] += 1
            elif relation == "proposes":
                concept_data["proposes"] += 1

            # Keep first entity_type found (they should be consistent)
            if concept_data["entity_type"] is None:
                concept_data["entity_type"] = triple.get("entity_type")

            # Track paper info for this concept
            concept_data["papers"].append({
                "paper_id": paper_id,
                "citations": citations,
                "year": year
            })

        # Build co-occurrence (concepts appearing in same paper)
        for c1 in paper_concepts:
            for c2 in paper_concepts:
                if c1 != c2:
                    topic_data[topic_name]["concept_cooccurrence"][c1][c2] += 1

    # Convert defaultdicts to regular dicts
    result = {}
    for topic, data in topic_data.items():
        result[topic] = {
            "field": data["field"],
            "subfield": data["subfield"],
            "paper_count": data["paper_count"],
            "concepts": {
                c: {
                    "count": d["count"],
                    "uses": d["uses"],
                    "proposes": d["proposes"],
                    "entity_type": d["entity_type"],
                    "papers": d["papers"]
                }
                for c, d in data["concepts"].items()
            },
            "concept_cooccurrence": {
                c: dict(others) for c, others in data["concept_cooccurrence"].items()
            }
        }

    return result


def calculate_topic_similarity(
    topic_a_concepts: Set[str],
    topic_b_concepts: Set[str]
) -> Tuple[float, Set[str]]:
    """
    Calculate Jaccard similarity and shared concepts between two topics.

    Returns:
        (similarity_score, shared_concepts)
    """
    shared = topic_a_concepts & topic_b_concepts
    union = topic_a_concepts | topic_b_concepts

    if not union:
        return 0.0, set()

    similarity = len(shared) / len(union)
    return similarity, shared


def compute_topic_centroid_distance(
    topic_a_concepts: Set[str],
    topic_b_concepts: Set[str],
    embeddings_cache: Dict[str, np.ndarray],
    model
) -> float:
    """
    Compute semantic distance between two topics using concept centroids.

    Instead of comparing topic names, we compare the centroid (average embedding)
    of all concepts in each topic. This better reflects actual topic content.

    Returns:
        Distance score (0-1), higher = more semantically different
    """
    if model is None or not topic_a_concepts or not topic_b_concepts:
        return 0.5  # Fallback if no model or empty topics

    # Get embeddings for all concepts in topic A
    topic_a_embeddings = []
    for concept in topic_a_concepts:
        if concept in embeddings_cache:
            topic_a_embeddings.append(embeddings_cache[concept])

    # Get embeddings for all concepts in topic B
    topic_b_embeddings = []
    for concept in topic_b_concepts:
        if concept in embeddings_cache:
            topic_b_embeddings.append(embeddings_cache[concept])

    if not topic_a_embeddings or not topic_b_embeddings:
        return 0.5  # Fallback if no embeddings

    # Compute centroids
    centroid_a = np.mean(topic_a_embeddings, axis=0)
    centroid_b = np.mean(topic_b_embeddings, axis=0)

    # Cosine similarity → distance
    similarity = np.dot(centroid_a, centroid_b) / (
        np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b) + 1e-8
    )

    # Convert to distance (0 = identical, 1 = orthogonal)
    distance = 1 - max(0, min(1, float(similarity)))
    return round(float(distance), 4)


def compute_idf(
    concept: str,
    all_topic_concepts: Dict[str, Set[str]]
) -> float:
    """
    Compute standard IDF (inverse document frequency) for a concept.

    Returns:
        log(total_topics / topics_with_concept)
        Higher = more rare/specialized
    """
    total_topics = len(all_topic_concepts)
    if total_topics == 0:
        return 0.0

    topics_with_concept = sum(
        1 for concepts in all_topic_concepts.values() if concept in concepts
    )

    if topics_with_concept == 0:
        return 0.0

    return math.log(total_topics / topics_with_concept)


def compute_tf_idf(
    concept_count: int,
    total_concepts_in_topic: int,
    idf: float
) -> float:
    """
    Compute TF-IDF score.

    TF = concept_count / total_concepts_in_topic
    TF-IDF = TF * IDF

    Returns:
        TF-IDF score (raw, will be z-score + sigmoid normalized later)
    """
    if total_concepts_in_topic == 0:
        return 0.0

    tf = concept_count / total_concepts_in_topic
    return tf * idf


def compute_citation_weight(papers: List[Dict]) -> float:
    """
    Compute average citations for papers using this concept.
    """
    if not papers:
        return 0.0
    total_citations = sum(p.get("citations", 0) for p in papers)
    return total_citations / len(papers)


def compute_recency_score(
    papers: List[Dict],
    global_min_year: int,
    global_max_year: int
) -> Tuple[float, float]:
    """
    Compute recency score and average publication year.

    Returns:
        (avg_year, recency_score)
        recency_score: 0-1, higher = more recent
    """
    if not papers:
        return 0.0, 0.0

    years = [p.get("year", 0) for p in papers if p.get("year", 0) > 0]
    if not years:
        return 0.0, 0.0

    avg_year = sum(years) / len(years)

    year_range = global_max_year - global_min_year
    if year_range <= 0:
        return avg_year, 0.5

    recency = (avg_year - global_min_year) / year_range
    return round(avg_year, 1), round(min(max(recency, 0), 1), 3)


def compute_bridge_potential(
    concept: str,
    shared_concepts: Set[str],
    cooccurrence: Dict[str, Dict[str, int]]
) -> int:
    """
    Count how many shared concepts co-occur with this blind spot concept.
    Higher = easier transfer path.
    """
    if concept not in cooccurrence:
        return 0

    concept_cooccurs = set(cooccurrence[concept].keys())
    return len(concept_cooccurs & shared_concepts)


def zscore_sigmoid_normalize(value: float, all_values: List[float]) -> float:
    """
    Z-score standardization followed by sigmoid squashing to 0-1.

    This preserves magnitude differences better than percentile:
    - Z-score: "how many standard deviations from mean"
    - Sigmoid: smoothly compresses to (0, 1) range

    Properties:
    - Mean value → 0.5
    - 1 std above mean → ~0.73
    - 2 std above mean → ~0.88
    - Outliers compressed but not collapsed (3 std still > 2 std)
    """
    if not all_values or len(all_values) < 2:
        return 0.5

    mean = np.mean(all_values)
    std = np.std(all_values)

    if std < 1e-8:  # All values identical
        return 0.5

    # Z-score: how many standard deviations from mean
    z = (value - mean) / std

    # Sigmoid: squash to (0, 1) range
    # 1 / (1 + e^(-z))
    sigmoid = 1 / (1 + np.exp(-z))

    return float(sigmoid)


def compute_topic_pair_score(
    topic_sem_dist: float,
    topic_shared_ratio: float
) -> float:
    """
    Compute how good this topic pairing is for knowledge transfer.

    We want topics that are:
    - Semantically different (high distance = novel transfer)
    - But share some concepts (bridge exists for feasibility)

    Inputs are z-score + sigmoid normalized.
    """
    score = (
        topic_sem_dist * 0.6 +       # want different topics
        topic_shared_ratio * 0.4     # but with bridge
    )
    return round(score, 4)


def compute_concept_ref_importance_score(
    concept_ref_tfidf: float,
    concept_citation: float,
    paper_recency: float
) -> float:
    """
    Compute how important/validated this concept is in ref_topic.

    High score = high TF-IDF (frequent locally + rare globally), highly-cited, and recent.

    All inputs are z-score + sigmoid normalized.

    TF-IDF replaces count_percentile to capture both:
    - TF: How important is concept in this ref_topic
    - IDF: How specialized/rare is concept globally
    """
    score = (
        concept_ref_tfidf * 0.5 +
        concept_citation * 0.3 +
        paper_recency * 0.2
    )
    return round(score, 4)


def compute_concept_transferability_score(
    blind_shared_concept_cooccur: float,
) -> float:
    """
    Compute how transferable this concept is to another domain.

    High score = bridges exist (co-occurring shared concepts make transfer easier).

    Input is z-score + sigmoid normalized.
    """
    return round(blind_shared_concept_cooccur, 4)


def compute_concept_novelty_score(
    min_blind_concept_dist: float,
) -> float:
    """
    Compute how novel this concept would be to the blind_topic.

    High score = semantically distant from blind_topic's existing concepts.
    Uses min pairwise distance - if close to ANY existing concept, not novel.

    Input is z-score + sigmoid normalized.
    """
    return round(min_blind_concept_dist, 4)


def compute_seed_score(
    topic_pair_score: float,
    concept_ref_importance_score: float,
    concept_transferability_score: float,
    concept_novelty_score: float
) -> float:
    """
    Final seed score combining all hierarchical scores.

    Weights: Equal 25% each
    - topic_pair: 25% - good topic pairing is foundation
    - importance: 25% - concept should be validated/important
    - transferability: 25% - co-occurrence with shared concepts
    - novelty: 25% - should be new to blind_topic
    """
    score = (
        topic_pair_score * 0.25 +
        concept_ref_importance_score * 0.25 +
        concept_transferability_score * 0.25 +
        concept_novelty_score * 0.25
    )
    return round(score, 4)


def get_semantic_model(model_name: str = 'all-MiniLM-L6-v2'):
    """Lazy load semantic model with HF token authentication."""
    global _semantic_model
    if _semantic_model is None and HAS_SENTENCE_TRANSFORMERS:
        import os

        # Set HF token from config or environment
        if 'HF_TOKEN' not in os.environ:
            try:
                from aii_pipeline.utils import PipelineConfig
                cfg = PipelineConfig.from_yaml()
                hf_token = cfg.raw.get('api_keys', {}).get('huggingface', '')
                if hf_token:
                    os.environ['HF_TOKEN'] = hf_token
                    _emit(MessageType.DEBUG, "Set HF_TOKEN from config")
            except Exception:
                pass  # Config not available, continue without token

        _emit(MessageType.INFO, f"Loading semantic model: {model_name}")
        try:
            # Try local cache first (fast, no network)
            _semantic_model = SentenceTransformer(model_name, local_files_only=True)
            _emit(MessageType.INFO, "Loaded model from local cache")
        except Exception:
            # Not in cache, download it
            _emit(MessageType.INFO, "Model not cached, downloading...")
            _semantic_model = SentenceTransformer(model_name)
            _emit(MessageType.INFO, "Model downloaded and cached")
    return _semantic_model


def compute_semantic_distance_to_topic(
    concept: str,
    topic_concepts: Set[str],
    embeddings_cache: Dict[str, np.ndarray],
    model
) -> Optional[float]:
    """
    Compute semantic distance between a concept and the closest concept in a topic.

    Uses pairwise MIN distance instead of centroid - if the concept is close to
    ANY existing concept in the topic, it's not novel (centroid would miss outlier matches).

    Returns:
        Distance score (0-1), higher = more foreign/distant from topic
        None if embeddings unavailable
    """
    if model is None or not topic_concepts:
        return None

    # Get concept embedding
    if concept not in embeddings_cache:
        embeddings_cache[concept] = model.encode([concept], show_progress_bar=False)[0]

    concept_emb = embeddings_cache[concept]
    concept_norm = np.linalg.norm(concept_emb)

    # Compute distance to each topic concept, keep minimum
    min_distance = 1.0
    for tc in topic_concepts:
        if tc not in embeddings_cache:
            embeddings_cache[tc] = model.encode([tc], show_progress_bar=False)[0]
        tc_emb = embeddings_cache[tc]
        tc_norm = np.linalg.norm(tc_emb)

        # Cosine similarity → distance
        similarity = np.dot(concept_emb, tc_emb) / (concept_norm * tc_norm + 1e-8)
        distance = 1 - max(0, min(1, float(similarity)))

        if distance < min_distance:
            min_distance = distance

    return round(float(min_distance), 3)


def batch_encode_concepts(
    concepts: List[str],
    embeddings_cache: Dict[str, np.ndarray],
    model,
) -> None:
    """
    Encode concepts that aren't in cache.

    Args:
        concepts: List of concept strings to encode
        embeddings_cache: Cache dict to store embeddings
        model: SentenceTransformer model
    """
    if model is None:
        raise ValueError("Semantic model is required but not available")

    # Find concepts not yet encoded
    to_encode = [c for c in concepts if c not in embeddings_cache]
    if not to_encode:
        return

    # Encode with progress bar and batch size 256
    embeddings = model.encode(to_encode, show_progress_bar=True, batch_size=256)

    for concept, emb in zip(to_encode, embeddings):
        embeddings_cache[concept] = emb


def find_blind_spots(
    topic_data: Dict[str, Dict[str, Any]],
    min_shared_concepts: int = 1,
    max_similarity: float = 1.0,
    min_blind_spot_count: int = 1,
) -> List[Dict[str, Any]]:
    """
    Find concept-level blind spots with rich metrics.

    Args:
        topic_data: Topic data from extract_topic_data
        min_shared_concepts: Minimum shared concepts to consider pair (default 1)
        max_similarity: Maximum similarity (default 1.0 = no filter)
        min_blind_spot_count: Minimum times concept used in ref_topic (default 1)

    Returns:
        List of individual blind spot concepts with metrics
    """
    topics = list(topic_data.keys())
    blind_spots = []

    # Pre-compute: all topic concepts for IDF calculation
    all_topic_concepts = {
        topic: set(data["concepts"].keys())
        for topic, data in topic_data.items()
    }

    # Load semantic model (required)
    embeddings_cache: Dict[str, np.ndarray] = {}
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

    semantic_model = get_semantic_model()
    if semantic_model is None:
        raise RuntimeError("Failed to load semantic model")

    # Batch encode topic names for topic pair distance
    _emit(MessageType.INFO, f"Encoding {len(topics)} topic names...")
    batch_encode_concepts(topics, embeddings_cache, semantic_model)

    # Batch encode all concepts for novelty score
    all_concepts = set()
    for concepts in all_topic_concepts.values():
        all_concepts.update(concepts)
    _emit(MessageType.INFO, f"Encoding {len(all_concepts)} concepts...")
    batch_encode_concepts(list(all_concepts), embeddings_cache, semantic_model)

    # Pre-compute: global year range for recency
    all_years = []
    for data in topic_data.values():
        for concept_info in data["concepts"].values():
            for paper in concept_info["papers"]:
                if paper.get("year", 0) > 0:
                    all_years.append(paper["year"])

    global_min_year = min(all_years) if all_years else 2000
    global_max_year = max(all_years) if all_years else 2024

    total_pairs = len(topics) * (len(topics) - 1) // 2
    pair_count = 0
    _emit(MessageType.INFO, f"Processing {total_pairs} topic pairs...")

    for i, blind_topic in enumerate(topics):
        blind_concepts = set(topic_data[blind_topic]["concepts"].keys())

        for ref_topic in topics[i+1:]:
            ref_concepts = set(topic_data[ref_topic]["concepts"].keys())
            pair_count += 1

            similarity, shared = calculate_topic_similarity(blind_concepts, ref_concepts)

            # Skip if not enough shared concepts
            if len(shared) < min_shared_concepts:
                continue

            # Skip if too similar
            if similarity > max_similarity:
                continue

            # Topic-level metrics
            # Use concept centroid distance if available, else fall back to Jaccard
            if semantic_model is not None:
                blind_ref_topic_sem_dist = compute_topic_centroid_distance(
                    blind_concepts, ref_concepts, embeddings_cache, semantic_model
                )
            else:
                blind_ref_topic_sem_dist = round(1 - similarity, 4)
            min_size = min(len(blind_concepts), len(ref_concepts))
            topic_pair_shared_ratio = round(len(shared) / min_size, 4) if min_size > 0 else 0

            # Process blind spots in both directions
            for direction_blind, direction_ref, direction_ref_concepts in [
                (blind_topic, ref_topic, ref_concepts - blind_concepts),
                (ref_topic, blind_topic, blind_concepts - ref_concepts)
            ]:
                ref_data = topic_data[direction_ref]

                # Compute total concepts in ref_topic for TF calculation
                total_concepts_in_ref_topic = sum(
                    c["count"] for c in ref_data["concepts"].values()
                )

                for concept in direction_ref_concepts:
                    concept_info = ref_data["concepts"][concept]

                    if concept_info["count"] < min_blind_spot_count:
                        continue

                    count = concept_info["count"]

                    # Compute IDF and TF-IDF
                    idf = compute_idf(concept, all_topic_concepts)
                    tf_idf = compute_tf_idf(count, total_concepts_in_ref_topic, idf)

                    # Compute citation weight
                    citation_weight = compute_citation_weight(concept_info["papers"])

                    # Compute recency
                    avg_year, recency_score = compute_recency_score(
                        concept_info["papers"], global_min_year, global_max_year
                    )

                    # Compute bridge potential
                    bridge_potential = compute_bridge_potential(
                        concept, shared, ref_data["concept_cooccurrence"]
                    )

                    # Compute semantic distance to blind topic
                    semantic_dist = None
                    if semantic_model is not None:
                        blind_topic_concepts = all_topic_concepts.get(direction_blind, set())
                        semantic_dist = compute_semantic_distance_to_topic(
                            concept,
                            blind_topic_concepts,
                            embeddings_cache,
                            semantic_model
                        )

                    # Create blind spot entry with hierarchical structure
                    blind_spot_id = f"{slugify(concept)}__{slugify(direction_blind)}__{slugify(direction_ref)}"

                    blind_spot_entry = {
                        "id": blind_spot_id,
                        "concept": concept,
                        "entity_type": concept_info["entity_type"],
                        "blind_topic": direction_blind,
                        "ref_topic": direction_ref,

                        # Hierarchical metrics (scores added in second pass)
                        "topic_pair": {
                            "blind_ref_topic_sem_dist": blind_ref_topic_sem_dist,
                            "topic_pair_shared_ratio": topic_pair_shared_ratio,
                            "shared_concepts": list(shared),
                        },
                        "importance": {
                            "count": count,
                            "tf_idf": round(tf_idf, 6),
                            "idf": round(idf, 4),
                            "concept_citation": round(citation_weight, 1),
                            "avg_year": avg_year,
                            "paper_recency": recency_score,
                        },
                        "transferability": {
                            "blind_shared_concept_cooccur": bridge_potential,
                        },
                        "novelty": {
                            "min_concept_to_blind_dist": semantic_dist,
                        },
                        "relation_breakdown": {
                            "uses": concept_info["uses"],
                            "proposes": concept_info["proposes"]
                        },
                    }

                    blind_spots.append(blind_spot_entry)

            # Log progress every 5 pairs
            if pair_count % 5 == 0 or pair_count == total_pairs:
                _emit(MessageType.INFO, f"Processed {pair_count}/{total_pairs} topic pairs ({len(blind_spots)} blind spots so far)")

    # Second pass: compute hierarchical scores with normalization
    _emit(MessageType.INFO, f"Computing scores for {len(blind_spots)} blind spots...")
    if blind_spots:
        # =====================================================================
        # COLLECT VALUES FOR NORMALIZATION
        # =====================================================================

        # Per-ref_topic normalization (different topics have different patterns)
        tf_idf_by_ref_topic: Dict[str, List[float]] = defaultdict(list)
        citation_weights_by_ref_topic: Dict[str, List[float]] = defaultdict(list)
        for bs in blind_spots:
            tf_idf_by_ref_topic[bs["ref_topic"]].append(bs["importance"]["tf_idf"])
            citation_weights_by_ref_topic[bs["ref_topic"]].append(bs["importance"]["concept_citation"])

        # Per-topic_pair normalization (max cooccurrence depends on shared concepts in pair)
        blind_shared_concept_cooccurs_by_pair: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for bs in blind_spots:
            pair_key = tuple(sorted([bs["blind_topic"], bs["ref_topic"]]))
            blind_shared_concept_cooccurs_by_pair[pair_key].append(bs["transferability"]["blind_shared_concept_cooccur"])

        # Global normalization for remaining raw 0-1 scores
        # Even though they're already 0-1, their distributions may cluster differently
        all_paper_recency: List[float] = []
        all_topic_sem_dist: List[float] = []
        all_topic_shared_ratio: List[float] = []
        all_min_blind_concept_dist: List[float] = []

        for bs in blind_spots:
            all_paper_recency.append(bs["importance"]["paper_recency"])
            all_topic_sem_dist.append(bs["topic_pair"]["blind_ref_topic_sem_dist"])
            all_topic_shared_ratio.append(bs["topic_pair"]["topic_pair_shared_ratio"])
            if bs["novelty"]["min_concept_to_blind_dist"] is not None:
                all_min_blind_concept_dist.append(bs["novelty"]["min_concept_to_blind_dist"])

        # =====================================================================
        # COMPUTE HIERARCHICAL SCORES WITH Z-SCORE + SIGMOID NORMALIZATION
        # =====================================================================
        for bs in blind_spots:
            # --- Per-ref_topic normalization ---
            ref_topic_tf_idfs = tf_idf_by_ref_topic[bs["ref_topic"]]
            concept_ref_tfidf = zscore_sigmoid_normalize(
                bs["importance"]["tf_idf"], ref_topic_tf_idfs
            )

            ref_topic_citations = citation_weights_by_ref_topic[bs["ref_topic"]]
            concept_citation = zscore_sigmoid_normalize(
                bs["importance"]["concept_citation"], ref_topic_citations
            )

            # --- Per-topic_pair normalization ---
            pair_key = tuple(sorted([bs["blind_topic"], bs["ref_topic"]]))
            pair_blind_shared_concept_cooccurs = blind_shared_concept_cooccurs_by_pair[pair_key]
            blind_shared_concept_cooccur = zscore_sigmoid_normalize(
                bs["transferability"]["blind_shared_concept_cooccur"], pair_blind_shared_concept_cooccurs
            )

            # --- Global normalization ---
            paper_recency = zscore_sigmoid_normalize(
                bs["importance"]["paper_recency"], all_paper_recency
            )
            topic_sem_dist = zscore_sigmoid_normalize(
                bs["topic_pair"]["blind_ref_topic_sem_dist"], all_topic_sem_dist
            )
            topic_shared_ratio = zscore_sigmoid_normalize(
                bs["topic_pair"]["topic_pair_shared_ratio"], all_topic_shared_ratio
            )

            # Novelty: handle None case (no embeddings available)
            raw_min_dist = bs["novelty"]["min_concept_to_blind_dist"]
            if raw_min_dist is not None and all_min_blind_concept_dist:
                min_blind_concept_dist = zscore_sigmoid_normalize(raw_min_dist, all_min_blind_concept_dist)
            else:
                min_blind_concept_dist = 0.5  # Fallback

            # Level 1: Topic Pair Score
            topic_pair_score = compute_topic_pair_score(
                topic_sem_dist,
                topic_shared_ratio
            )
            bs["topic_pair"]["score"] = topic_pair_score

            # Level 2: Concept Ref Importance Score
            concept_ref_importance_score = compute_concept_ref_importance_score(
                concept_ref_tfidf,
                concept_citation,
                paper_recency
            )
            bs["importance"]["score"] = concept_ref_importance_score

            # Level 3: Concept Transferability Score
            concept_transferability_score = compute_concept_transferability_score(
                blind_shared_concept_cooccur,
            )
            bs["transferability"]["score"] = concept_transferability_score

            # Level 4: Concept Novelty Score
            concept_novelty_score = compute_concept_novelty_score(
                min_blind_concept_dist,
            )
            bs["novelty"]["score"] = concept_novelty_score

            # Final: Seed Score (combines all component scores)
            seed_score = compute_seed_score(
                topic_pair_score,
                concept_ref_importance_score,
                concept_transferability_score,
                concept_novelty_score
            )
            bs["seed_score"] = seed_score

        # Sort by seed_score descending
        blind_spots.sort(key=lambda x: x["seed_score"], reverse=True)

        # Add percentiles for each score type using binary search (O(n log n) each)
        total = len(blind_spots)

        # Compute percentile for each score dimension
        score_types = [
            ("topic_pair", "score", "percentile"),
            ("importance", "score", "percentile"),
            ("transferability", "score", "percentile"),
            ("novelty", "score", "percentile"),
        ]

        for section, score_key, percentile_key in score_types:
            all_scores_asc = sorted(bs[section][score_key] for bs in blind_spots)
            for bs in blind_spots:
                below = bisect_left(all_scores_asc, bs[section][score_key])
                bs[section][percentile_key] = round(below / total * 100, 1)

        # Add percentile for final seed_score
        all_opp_scores_asc = sorted(bs["seed_score"] for bs in blind_spots)
        for bs in blind_spots:
            below = bisect_left(all_opp_scores_asc, bs["seed_score"])
            bs["score_percentile"] = round(below / total * 100, 1)

    return blind_spots


def generate_topic_blind_spots(
    papers: List[Dict],
    output_file: Path,
    min_shared_concepts: int = 1,
    max_similarity: float = 1.0,
    min_count: int = 1,
    entity_types: List[str] = None,
) -> bool:
    """
    Generate concept-level blind spot opportunities and save to file.

    Each concept is its own entry with rich metrics for ranking.

    Args:
        papers: List of paper dictionaries
        output_file: Path to save opportunities JSON
        min_shared_concepts: Minimum shared concepts between topics (default 1)
        max_similarity: Maximum Jaccard similarity (default 1.0 = no filter)
        min_count: Minimum concept count in ref_topic (default 1)
        entity_types: List of entity types to include (e.g., ["method", "concept"]).
                      Empty list or None = include all types.
                      Valid types: method, concept, task, tool, artifact, data, other

    Returns:
        True if successful
    """
    _emit(MessageType.INFO, "Finding concept-level blind spots...")

    # Extract comprehensive topic data
    topic_data = extract_topic_data(papers)
    _emit(MessageType.INFO, f"Found {len(topic_data)} topics")

    if len(topic_data) < 2:
        _emit(MessageType.WARNING, "Need at least 2 topics to find blind spots")
        return False

    # Find blind spots
    blind_spots = find_blind_spots(
        topic_data,
        min_shared_concepts=min_shared_concepts,
        max_similarity=max_similarity,
        min_blind_spot_count=min_count,
    )

    if not blind_spots:
        _emit(MessageType.WARNING, "No blind spot opportunities found")
        return False

    _emit(MessageType.INFO, f"Found {len(blind_spots)} concept-level blind spots")

    # Filter by entity types if specified
    if entity_types:
        pre_filter_count = len(blind_spots)
        blind_spots = [bs for bs in blind_spots if bs.get("entity_type") in entity_types]
        _emit(MessageType.INFO,
            f"Filtered to {len(blind_spots)} blind spots "
            f"(entity_types: {entity_types}, removed {pre_filter_count - len(blind_spots)})"
        )

    # Log top 5 for visibility
    _emit(MessageType.INFO, "Top 5 blind spots:")
    for i, bs in enumerate(blind_spots[:5]):
        _emit(MessageType.INFO,
            f"  {i+1}. {bs['concept']} "
            f"({bs['entity_type']}) "
            f"score={bs['seed_score']:.3f}"
        )

    # Save
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(blind_spots, f, indent=2, ensure_ascii=False)
        _emit(MessageType.SUCCESS, f"Saved {len(blind_spots)} blind spots to {output_file.name}")
        return True
    except Exception as e:
        _emit(MessageType.ERROR, f"Failed to save: {e}")
        raise
