#!/usr/bin/env python3
"""
Wikidata enrichment for triples.

Two-phase enrichment:
1. Wikipedia REST API -> Get QID and description (fast)
2. Wikidata Entity API -> Get ALL properties (claims)

Stores everything under a single `wikidata` key so we can later
analyze which properties are common/useful and discard others.

No API key needed - both APIs are free and public.
"""

import re
import time
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Set
from urllib.parse import unquote, quote
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


# ============================================================================
# Property ID to human-readable name mapping
# We'll resolve unknown ones via API, but cache common ones
# ============================================================================
KNOWN_PROPERTIES = {
    # Ontological
    "P31": "instance_of",
    "P279": "subclass_of",
    "P361": "part_of",
    "P527": "has_parts",
    "P1889": "different_from",
    "P460": "same_as",
    "P461": "opposite_of",
    "P2283": "uses",
    "P366": "use",
    "P155": "follows",
    "P156": "followed_by",
    "P1382": "coincident_with",

    # People/orgs
    "P178": "developer",
    "P170": "creator",
    "P287": "designed_by",
    "P112": "founded_by",
    "P3095": "practiced_by",

    # Topics
    "P921": "main_subject",
    "P910": "topic_main_category",
    "P1424": "topic_maintained_by",
    "P2578": "study_of",

    # External IDs
    "P10283": "openalex_id",
    "P6366": "mag_id",
    "P646": "freebase_id",
    "P227": "gnd_id",
    "P244": "loc_id",
    "P2581": "babelnet_id",
    "P10565": "openalex_topic_id",
    "P373": "commons_category",
    "P18": "image",
    "P5555": "schematic",
    "P1813": "short_name",
    "P2671": "google_kg_id",
    "P3417": "quora_topic_id",
    "P1417": "britannica_id",
    "P486": "mesh_id",
    "P672": "mesh_tree_code",

    # Other useful
    "P1343": "described_by_source",
    "P1482": "stack_exchange_tag",
}


def extract_title_from_wikipedia_url(url: str) -> Optional[str]:
    """Extract Wikipedia article title from URL."""
    if not url:
        return None
    match = re.match(r"https?://en\.wikipedia\.org/wiki/(.+)", url)
    if not match:
        return None
    title = unquote(match.group(1)).replace("_", " ")
    return title


def parse_claim_value(datavalue: Dict[str, Any]) -> Any:
    """Parse a Wikidata claim datavalue into a simple value."""
    if not datavalue:
        return None

    dtype = datavalue.get("type")
    value = datavalue.get("value")

    if dtype == "wikibase-entityid":
        # Return Q-ID (we'll resolve labels in batch later)
        return value.get("id")
    elif dtype == "string":
        return value
    elif dtype == "monolingualtext":
        return value.get("text")
    elif dtype == "time":
        # Extract just the year or date
        time_str = value.get("time", "")
        if time_str.startswith("+"):
            time_str = time_str[1:]
        return time_str[:10]  # YYYY-MM-DD
    elif dtype == "quantity":
        return value.get("amount")
    elif dtype == "globecoordinate":
        return {"lat": value.get("latitude"), "lon": value.get("longitude")}
    else:
        return str(value) if value else None


def parse_wikidata_entity_full(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse ALL data from a Wikidata entity.

    Returns dict with:
    - id: Q-ID
    - label: English label
    - description: English description
    - aliases: List of alternative names
    - claims: Dict of property_name -> list of values
    """
    result = {
        "id": entity.get("id"),
        "label": entity.get("labels", {}).get("en", {}).get("value"),
        "description": entity.get("descriptions", {}).get("en", {}).get("value"),
        "aliases": [a["value"] for a in entity.get("aliases", {}).get("en", [])],
        "claims": {},
    }

    claims = entity.get("claims", {})

    for prop_id, values in claims.items():
        # Get human-readable property name
        prop_name = KNOWN_PROPERTIES.get(prop_id, prop_id)

        # Parse all values for this property
        parsed_values = []
        for v in values:
            mainsnak = v.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue")
            if datavalue:
                parsed = parse_claim_value(datavalue)
                if parsed is not None:
                    parsed_values.append(parsed)

        if parsed_values:
            # Single value -> store directly, multiple -> store as list
            if len(parsed_values) == 1:
                result["claims"][prop_name] = parsed_values[0]
            else:
                result["claims"][prop_name] = parsed_values

    return result


async def get_wikipedia_qid_async(
    session: aiohttp.ClientSession,
    title: str,
    semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """Phase 1: Get QID from Wikipedia REST API."""
    async with semaphore:
        try:
            encoded_title = quote(title.replace(" ", "_"))
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 404:
                    return None
                response.raise_for_status()
                data = await response.json()

                qid = data.get("wikibase_item")
                if qid:
                    return {
                        "id": qid,
                        "description": data.get("description", "")
                    }
                return None
        except Exception:
            raise


async def get_wikidata_entity_full_async(
    session: aiohttp.ClientSession,
    qid: str,
    semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """Phase 2: Get full Wikidata entity with ALL claims."""
    async with semaphore:
        try:
            url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 404:
                    return None
                response.raise_for_status()
                data = await response.json()

                entity = data.get("entities", {}).get(qid)
                if entity:
                    return parse_wikidata_entity_full(entity)
                return None
        except Exception:
            raise


async def resolve_qids_batch(
    session: aiohttp.ClientSession,
    qids: Set[str],
    semaphore: asyncio.Semaphore
) -> Dict[str, str]:
    """Resolve Q-IDs to labels in batch (max 50 at a time)."""
    if not qids:
        return {}

    qid_labels = {}
    qid_list = list(qids)

    # Process in batches of 50 (Wikidata API limit)
    for i in range(0, len(qid_list), 50):
        batch = qid_list[i:i+50]
        async with semaphore:
            try:
                url = "https://www.wikidata.org/w/api.php"
                params = {
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "labels",
                    "languages": "en",
                    "format": "json"
                }
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        for qid, ent in data.get("entities", {}).items():
                            label = ent.get("labels", {}).get("en", {}).get("value")
                            if label:
                                qid_labels[qid] = label
            except Exception:
                raise

    return qid_labels


def resolve_qids_in_claims(claims: Dict[str, Any], qid_labels: Dict[str, str]) -> Dict[str, Any]:
    """Replace Q-IDs with {id, label} objects in claims."""
    resolved = {}

    for prop_name, value in claims.items():
        if isinstance(value, str) and value.startswith("Q"):
            # Single Q-ID
            label = qid_labels.get(value)
            if label:
                resolved[prop_name] = {"id": value, "label": label}
            else:
                resolved[prop_name] = value
        elif isinstance(value, list):
            # List of values (might contain Q-IDs)
            resolved_list = []
            for v in value:
                if isinstance(v, str) and v.startswith("Q"):
                    label = qid_labels.get(v)
                    if label:
                        resolved_list.append({"id": v, "label": label})
                    else:
                        resolved_list.append(v)
                else:
                    resolved_list.append(v)
            resolved[prop_name] = resolved_list
        else:
            resolved[prop_name] = value

    return resolved


async def enrich_triples_async(
    triples: List[Dict[str, Any]],
    max_concurrent: int = 10,
    resolve_labels: bool = True
) -> List[Dict[str, Any]]:
    """
    Enrich all triples with full Wikidata data.

    Phase 1: Wikipedia REST API -> QID
    Phase 2: Wikidata Entity API -> ALL claims
    Phase 3: Resolve Q-IDs to labels (optional)

    Stores everything under triple["wikidata"] = {...}
    """
    # Filter triples that need enrichment
    to_enrich = [(i, t) for i, t in enumerate(triples)
                 if not t.get("wikidata") and t.get("wikipedia_url")]

    if not to_enrich:
        already_done = sum(1 for t in triples if t.get("wikidata"))
        _emit(MessageType.INFO, f"Wikidata: {already_done}/{len(triples)} already enriched")
        return triples

    _emit(MessageType.INFO, f"Phase 1: Getting QIDs for {len(to_enrich)} triples...")

    semaphore = asyncio.Semaphore(max_concurrent)
    headers = {"User-Agent": "InventionKG/1.0 (research project)"}

    async with aiohttp.ClientSession(headers=headers) as session:
        # Phase 1: Get QIDs
        phase1_tasks = []
        for idx, triple in to_enrich:
            title = extract_title_from_wikipedia_url(triple.get("wikipedia_url", ""))
            if title:
                phase1_tasks.append((idx, title, get_wikipedia_qid_async(session, title, semaphore)))

        phase1_results = await asyncio.gather(*[t[2] for t in phase1_tasks], return_exceptions=True)

        # Collect QIDs for Phase 2
        qids_to_fetch = []
        for (idx, title, _), result in zip(phase1_tasks, phase1_results):
            if isinstance(result, Exception) or not result:
                continue
            qids_to_fetch.append((idx, result["id"], result.get("description", "")))

        _emit(MessageType.INFO, f"Phase 1: {len(qids_to_fetch)}/{len(to_enrich)} QIDs found")

        if not qids_to_fetch:
            return triples

        # Phase 2: Get full entity data
        _emit(MessageType.INFO, "Phase 2: Fetching full entity data...")

        phase2_tasks = []
        for idx, qid, desc in qids_to_fetch:
            phase2_tasks.append((idx, qid, desc, get_wikidata_entity_full_async(session, qid, semaphore)))

        phase2_results = await asyncio.gather(*[t[3] for t in phase2_tasks], return_exceptions=True)

        # Collect all Q-IDs that need label resolution
        all_qids_to_resolve: Set[str] = set()
        entities_fetched = []

        for (idx, qid, desc, _), result in zip(phase2_tasks, phase2_results):
            if isinstance(result, Exception) or not result:
                # Still store basic info even if full fetch failed
                triples[idx]["wikidata"] = {"id": qid, "description": desc}
                continue

            entities_fetched.append((idx, result))

            # Collect Q-IDs from claims for resolution
            if resolve_labels:
                for value in result.get("claims", {}).values():
                    if isinstance(value, str) and value.startswith("Q"):
                        all_qids_to_resolve.add(value)
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, str) and v.startswith("Q"):
                                all_qids_to_resolve.add(v)

        _emit(MessageType.INFO, f"Phase 2: {len(entities_fetched)} entities fetched")

        # Phase 3: Resolve Q-ID labels
        qid_labels = {}
        if resolve_labels and all_qids_to_resolve:
            _emit(MessageType.INFO, f"Phase 3: Resolving {len(all_qids_to_resolve)} Q-ID labels...")
            qid_labels = await resolve_qids_batch(session, all_qids_to_resolve, semaphore)
            _emit(MessageType.INFO, f"Phase 3: {len(qid_labels)} labels resolved")

        # Store results in triples
        for idx, entity_data in entities_fetched:
            # Resolve Q-IDs to labels in claims
            if qid_labels and entity_data.get("claims"):
                entity_data["claims"] = resolve_qids_in_claims(entity_data["claims"], qid_labels)

            triples[idx]["wikidata"] = entity_data

    enriched = sum(1 for t in triples if t.get("wikidata"))
    _emit(MessageType.SUCCESS, f"Done: {enriched}/{len(triples)} triples enriched")

    return triples


def enrich_triples_sync(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Synchronous fallback (basic enrichment only)."""
    import requests

    headers = {"User-Agent": "InventionKG/1.0 (research project)"}
    enriched = 0

    for triple in triples:
        if triple.get("wikidata"):
            enriched += 1
            continue

        url = triple.get("wikipedia_url")
        if not url:
            continue

        title = extract_title_from_wikipedia_url(url)
        if not title:
            continue

        try:
            # Get QID
            encoded = quote(title.replace(" ", "_"))
            resp = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}",
                headers=headers, timeout=10
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()

            qid = data.get("wikibase_item")
            if not qid:
                continue

            # Get full entity
            time.sleep(0.1)
            resp2 = requests.get(
                f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
                headers=headers, timeout=15
            )
            if resp2.status_code == 200:
                entity = resp2.json().get("entities", {}).get(qid)
                if entity:
                    triple["wikidata"] = parse_wikidata_entity_full(entity)
                    enriched += 1

            time.sleep(0.1)

        except Exception:
            raise

    _emit(MessageType.INFO, f"Wikidata: {enriched}/{len(triples)} enriched")
    return triples


# ============================================================================
# Main enrichment function (used by step 5)
# ============================================================================

def enrich_triples_with_wikidata(
    triples: List[Dict[str, Any]],
    use_async: bool = True,
    resolve_labels: bool = True
) -> List[Dict[str, Any]]:
    """
    Enrich triples with full Wikidata data.

    Stores ALL Wikidata info under triple["wikidata"] = {
        "id": "Q2539",
        "label": "machine learning",
        "description": "...",
        "aliases": ["ML", ...],
        "claims": {
            "instance_of": [{"id": "Q11862829", "label": "academic discipline"}, ...],
            "subclass_of": [...],
            "openalex_id": "C2982736386",
            ...all other properties...
        }
    }
    """
    if not triples:
        return triples

    if use_async:
        return asyncio.run(enrich_triples_async(triples, resolve_labels=resolve_labels))
    else:
        return enrich_triples_sync(triples)


# Convenience function
def get_wikidata_from_wikipedia_url(url: str) -> Optional[Dict[str, Any]]:
    """Get full Wikidata info from Wikipedia URL."""
    title = extract_title_from_wikipedia_url(url)
    if not title:
        return None

    import requests
    headers = {"User-Agent": "InventionKG/1.0"}

    try:
        encoded = quote(title.replace(" ", "_"))
        resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}",
            headers=headers, timeout=10
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        qid = resp.json().get("wikibase_item")

        if qid:
            resp2 = requests.get(
                f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
                headers=headers, timeout=15
            )
            if resp2.status_code == 200:
                entity = resp2.json().get("entities", {}).get(qid)
                if entity:
                    return parse_wikidata_entity_full(entity)
    except Exception:
        raise
