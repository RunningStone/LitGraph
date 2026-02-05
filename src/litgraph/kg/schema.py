"""KG schema: entity normalization, type validation, relation validation."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from litgraph.settings import get_settings

logger = logging.getLogger(__name__)

_schema_cache: dict | None = None


def _load_schema(schema_dict: dict | None = None) -> dict:
    """Load schema from YAML or use provided dict."""
    global _schema_cache
    if schema_dict is not None:
        return schema_dict
    if _schema_cache is not None:
        return _schema_cache

    settings = get_settings()
    schema_path = settings.schema_path
    if schema_path.exists():
        with open(schema_path) as f:
            _schema_cache = yaml.safe_load(f)
    else:
        _schema_cache = {"node_types": {}, "relation_types": {}, "aliases": {}}
    return _schema_cache


def reset_schema_cache() -> None:
    """Clear the schema cache (for tests)."""
    global _schema_cache
    _schema_cache = None


def normalize_entity(name: str, schema_dict: dict | None = None) -> str:
    """Normalize entity name via case-insensitive alias lookup.

    Args:
        name: Raw entity name.
        schema_dict: Optional schema dict for testing.

    Returns:
        Canonical name if alias found, otherwise original name.
    """
    schema = _load_schema(schema_dict)
    aliases = schema.get("aliases", {})

    # Case-insensitive alias lookup
    name_lower = name.lower()
    for alias, canonical in aliases.items():
        if alias.lower() == name_lower:
            return canonical

    return name


def validate_entity(name: str, entity_type: str, schema_dict: dict | None = None) -> bool:
    """Check if entity_type is defined in the schema.

    Args:
        name: Entity name (unused, but kept for API consistency).
        entity_type: The type to validate.
        schema_dict: Optional schema dict for testing.

    Returns:
        True if entity_type is in schema's node_types.
    """
    schema = _load_schema(schema_dict)
    node_types = schema.get("node_types", {})
    return entity_type in node_types


def validate_relation(
    from_type: str, rel_type: str, to_type: str, schema_dict: dict | None = None
) -> bool:
    """Check if a relation type with given endpoint types is valid per schema.

    Args:
        from_type: Source node type.
        rel_type: Relation type name.
        to_type: Target node type.
        schema_dict: Optional schema dict for testing.

    Returns:
        True if the relation is valid.
    """
    schema = _load_schema(schema_dict)
    relation_types = schema.get("relation_types", {})

    if rel_type not in relation_types:
        return False

    spec = relation_types[rel_type]
    return spec.get("from") == from_type and spec.get("to") == to_type


def normalize_relation_type(
    from_type: str, rel_type: str, to_type: str, schema_dict: dict | None = None
) -> str:
    """Normalize a relation type: valid → keep, invalid → 'related_to'.

    Args:
        from_type: Source node type.
        rel_type: Relation type name.
        to_type: Target node type.
        schema_dict: Optional schema dict for testing.

    Returns:
        The relation type if valid, otherwise 'related_to'.
    """
    if validate_relation(from_type, rel_type, to_type, schema_dict):
        return rel_type

    logger.debug(
        "Invalid relation %s(%s→%s), degrading to 'related_to'",
        rel_type, from_type, to_type,
    )
    return "related_to"
