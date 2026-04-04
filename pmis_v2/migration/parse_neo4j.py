"""
Neo4j Graph Parser for PMIS v2 Migration.

Supports two modes:
  1. Direct Bolt connection to a running Neo4j instance
  2. JSON/CSV export files from Neo4j

Extracts:
  - Nodes → MemoryNode candidates (with properties as content)
  - Relationships → Relation edges (mapped to PMIS relation types)
  - Graph structure → Tree detection (finds hierarchical patterns)

Neo4j nodes typically have labels and properties:
  (:Memory {content: "...", level: "...", created_at: "..."})
  (:Context {name: "...", summary: "..."})

Relationships:
  (a)-[:CHILD_OF]->(b)
  (a)-[:RELATED_TO]->(b)
  (a)-[:PRECEDED_BY]->(b)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def parse_neo4j(
    source: str,
    mode: str = "auto",
    bolt_uri: str = "bolt://localhost:7687",
    username: str = "neo4j",
    password: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse Neo4j data.

    Args:
        source: Path to JSON/CSV export, or "bolt" for live connection
        mode: "json", "csv", "bolt", or "auto" (detect from source)

    Returns:
        (nodes, relationships) — both as lists of dicts
    """
    if mode == "auto":
        if source == "bolt" or source.startswith("bolt://"):
            mode = "bolt"
        elif source.endswith(".json"):
            mode = "json"
        elif source.endswith(".csv"):
            mode = "csv"
        else:
            mode = "json"

    if mode == "bolt":
        uri = source if source.startswith("bolt://") else bolt_uri
        return _parse_bolt(uri, username, password)
    elif mode == "json":
        return _parse_json_export(source)
    elif mode == "csv":
        return _parse_csv_export(source)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _parse_bolt(
    uri: str, username: str, password: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Connect to Neo4j via Bolt and extract all nodes and relationships."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError(
            "neo4j driver not installed. Run: pip install neo4j"
        )

    driver = GraphDatabase.driver(uri, auth=(username, password))
    nodes = []
    relationships = []

    with driver.session() as session:
        # Extract all nodes
        result = session.run("""
            MATCH (n)
            RETURN id(n) AS neo4j_id,
                   labels(n) AS labels,
                   properties(n) AS props
        """)
        for record in result:
            node = _convert_neo4j_node(
                neo4j_id=record["neo4j_id"],
                labels=record["labels"],
                props=dict(record["props"]),
            )
            if node:
                nodes.append(node)

        # Extract all relationships
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN id(a) AS source_neo4j_id,
                   id(b) AS target_neo4j_id,
                   type(r) AS rel_type,
                   properties(r) AS props
        """)
        for record in result:
            rel = _convert_neo4j_relationship(
                source_id=str(record["source_neo4j_id"]),
                target_id=str(record["target_neo4j_id"]),
                rel_type=record["rel_type"],
                props=dict(record["props"]) if record["props"] else {},
            )
            if rel:
                relationships.append(rel)

    driver.close()
    return nodes, relationships


def _parse_json_export(file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse Neo4j JSON export.

    Supports multiple formats:
      - APOC export: {nodes: [...], relationships: [...]}
      - Neo4j Desktop export: [{type: "node", ...}, {type: "relationship", ...}]
      - Custom export: any JSON with identifiable node/edge structures
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    nodes = []
    relationships = []

    if isinstance(raw, dict):
        # APOC-style: {nodes: [...], relationships: [...]}
        raw_nodes = raw.get("nodes", raw.get("results", []))
        raw_rels = raw.get("relationships", raw.get("edges", []))

        for n in raw_nodes:
            node = _convert_json_node(n)
            if node:
                nodes.append(node)

        for r in raw_rels:
            rel = _convert_json_relationship(r)
            if rel:
                relationships.append(rel)

    elif isinstance(raw, list):
        # Mixed list of nodes and relationships
        for item in raw:
            item_type = item.get("type", "").lower()
            if item_type == "node" or "labels" in item:
                node = _convert_json_node(item)
                if node:
                    nodes.append(node)
            elif item_type == "relationship" or "rel_type" in item or "startNode" in item:
                rel = _convert_json_relationship(item)
                if rel:
                    relationships.append(rel)
            elif "content" in item or "text" in item:
                # Flat memory list with no explicit type
                node = _convert_flat_node(item)
                if node:
                    nodes.append(node)

    return nodes, relationships


def _parse_csv_export(file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse Neo4j CSV export (nodes and relationships in separate files or combined)."""
    import csv

    nodes = []
    relationships = []
    path = Path(file_path)

    # Check if it's a directory with multiple CSVs
    if path.is_dir():
        for csv_file in path.glob("*.csv"):
            fname = csv_file.stem.lower()
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "node" in fname or "content" in row:
                        node = _convert_flat_node(row)
                        if node:
                            nodes.append(node)
                    elif "rel" in fname or "edge" in fname:
                        rel = _convert_csv_relationship(row)
                        if rel:
                            relationships.append(rel)
    else:
        # Single CSV file — guess structure from headers
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "source" in row and "target" in row:
                    rel = _convert_csv_relationship(row)
                    if rel:
                        relationships.append(rel)
                else:
                    node = _convert_flat_node(row)
                    if node:
                        nodes.append(node)

    return nodes, relationships


# ---------------------------------------------------------------
# Node Converters
# ---------------------------------------------------------------

def _convert_neo4j_node(
    neo4j_id: int, labels: List[str], props: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Convert a Neo4j node to a memory candidate dict."""
    content = (
        props.get("content") or
        props.get("text") or
        props.get("summary") or
        props.get("name") or
        props.get("description") or
        ""
    )
    if not content or len(str(content).strip()) < 5:
        return None

    # Infer level from labels
    level = "ANC"
    labels_lower = [l.lower() for l in labels]
    if any(l in labels_lower for l in ["supercontext", "super_context", "sc", "domain"]):
        level = "SC"
    elif any(l in labels_lower for l in ["context", "ctx", "topic", "theme", "category"]):
        level = "CTX"
    elif any(l in labels_lower for l in ["anchor", "anc", "memory", "fact", "note"]):
        level = "ANC"

    created_at = _parse_ts(props.get("created_at") or props.get("timestamp"))

    return {
        "neo4j_id": str(neo4j_id),
        "content": str(content)[:2000],
        "source": "neo4j",
        "level_hint": level,
        "created_at": created_at,
        "labels": labels,
        "properties": props,
        "conversation_id": props.get("conversation_id", "neo4j_import"),
    }


def _convert_json_node(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a JSON-exported Neo4j node."""
    props = item.get("properties", item.get("props", item))
    labels = item.get("labels", [])
    neo4j_id = item.get("id", item.get("neo4j_id", item.get("_id", "")))
    return _convert_neo4j_node(neo4j_id, labels, props)


def _convert_flat_node(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a flat dict (from CSV or simple JSON) to a memory candidate."""
    content = (
        item.get("content") or
        item.get("text") or
        item.get("summary") or
        item.get("name") or
        ""
    )
    if not content or len(str(content).strip()) < 5:
        return None

    return {
        "neo4j_id": str(item.get("id", item.get("neo4j_id", ""))),
        "content": str(content)[:2000],
        "source": "neo4j",
        "level_hint": item.get("level", item.get("type", "ANC")).upper()[:3],
        "created_at": _parse_ts(item.get("created_at") or item.get("timestamp")),
        "labels": [],
        "properties": item,
        "conversation_id": item.get("conversation_id", "neo4j_import"),
    }


# ---------------------------------------------------------------
# Relationship Converters
# ---------------------------------------------------------------

# Map Neo4j relationship types to PMIS RelationType values
_REL_TYPE_MAP = {
    "CHILD_OF": "child_of",
    "IS_CHILD_OF": "child_of",
    "PARENT_OF": "child_of",         # Flip direction during import
    "HAS_CHILD": "child_of",         # Flip direction during import
    "BELONGS_TO": "child_of",
    "RELATED_TO": "related_to",
    "SIMILAR_TO": "similar_to",
    "PRECEDED_BY": "preceded_by",
    "PRECEDES": "followed_by",
    "FOLLOWED_BY": "followed_by",
    "FOLLOWS": "preceded_by",
    "CO_OCCURRED": "co_occurred",
    "SAME_SESSION": "co_occurred",
    "LINKS_TO": "related_to",
    "REFERENCES": "related_to",
}


def _convert_neo4j_relationship(
    source_id: str, target_id: str, rel_type: str, props: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Convert a Neo4j relationship to a PMIS relation dict."""
    mapped_type = _REL_TYPE_MAP.get(rel_type.upper(), "related_to")

    # Handle PARENT_OF / HAS_CHILD by flipping direction
    if rel_type.upper() in ("PARENT_OF", "HAS_CHILD"):
        source_id, target_id = target_id, source_id

    return {
        "source_neo4j_id": source_id,
        "target_neo4j_id": target_id,
        "relation_type": mapped_type,
        "original_type": rel_type,
        "weight": props.get("weight", 1.0),
        "tree_id": props.get("tree_id", "neo4j_import"),
        "created_at": _parse_ts(props.get("created_at")),
    }


def _convert_json_relationship(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a JSON-exported Neo4j relationship."""
    source = str(item.get("startNode", item.get("source", item.get("source_id", ""))))
    target = str(item.get("endNode", item.get("target", item.get("target_id", ""))))
    rel_type = item.get("type", item.get("rel_type", item.get("relationship", "RELATED_TO")))
    props = item.get("properties", item.get("props", {}))
    return _convert_neo4j_relationship(source, target, rel_type, props)


def _convert_csv_relationship(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a CSV row to a relationship."""
    source = str(row.get("source", row.get("source_id", row.get("from", ""))))
    target = str(row.get("target", row.get("target_id", row.get("to", ""))))
    rel_type = row.get("type", row.get("rel_type", row.get("relationship", "RELATED_TO")))
    return _convert_neo4j_relationship(source, target, rel_type, row)


def _parse_ts(ts) -> Optional[datetime]:
    """Parse various timestamp formats."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts)
        except (ValueError, OSError):
            return None
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", ""))
        except ValueError:
            return None
    return None
