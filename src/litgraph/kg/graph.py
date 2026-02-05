"""nano-graphrag integration with schema-aware storage."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from litgraph.kg.schema import normalize_entity
from litgraph.settings import get_settings

logger = logging.getLogger(__name__)


class SchemaAwareStorage:
    """Wraps NetworkXStorage to normalize entities before write.

    Instead of subclassing (which is fragile against nano-graphrag updates),
    we monkey-patch the storage instance's upsert methods after GraphRAG init.
    """

    @staticmethod
    def patch_storage(storage):
        """Patch a NetworkXStorage instance to normalize entities on upsert."""
        original_upsert_node = storage.upsert_node
        original_upsert_edge = storage.upsert_edge

        async def patched_upsert_node(node_id, node_data=None):
            node_id = normalize_entity(str(node_id))
            if node_data and "name" in node_data:
                node_data["name"] = normalize_entity(str(node_data["name"]))
            return await original_upsert_node(node_id, node_data)

        async def patched_upsert_edge(src_id, tgt_id, edge_data=None):
            src_id = normalize_entity(str(src_id))
            tgt_id = normalize_entity(str(tgt_id))
            return await original_upsert_edge(src_id, tgt_id, edge_data)

        storage.upsert_node = patched_upsert_node
        storage.upsert_edge = patched_upsert_edge

        return storage


def _make_embedding_func():
    """Create a local SentenceTransformer embedding function for nano-graphrag."""
    from nano_graphrag._utils import wrap_embedding_func_with_attrs
    settings = get_settings()
    model_name = settings.embedding_model

    _model = None

    def _get_model():
        nonlocal _model
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(model_name)
        return _model

    @wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=512)
    async def local_embedding(texts: list[str]) -> np.ndarray:
        model = _get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)

    return local_embedding


def get_rag(working_dir: str | Path | None = None):
    """Create a configured GraphRAG instance.

    Injects custom LLM functions, embedding, schema-aware storage patching,
    and overrides entity extraction prompts.
    """
    from nano_graphrag import GraphRAG
    from nano_graphrag.prompt import PROMPTS

    from litgraph.llm.client import best_model_complete, cheap_model_complete

    settings = get_settings()
    if working_dir is None:
        working_dir = str(settings.data_dir / "kg_store")

    # Override entity types in prompts
    PROMPTS["DEFAULT_ENTITY_TYPES"] = [
        "Paper", "Concept", "Method", "Dataset", "Finding", "Task"
    ]

    rag = GraphRAG(
        working_dir=str(working_dir),
        best_model_func=best_model_complete,
        cheap_model_func=cheap_model_complete,
        best_model_max_async=1,
        cheap_model_max_async=1,
        embedding_func=_make_embedding_func(),
        enable_llm_cache=True,
    )

    # Patch the graph storage after init for schema normalization
    if hasattr(rag, "_graph_storage"):
        SchemaAwareStorage.patch_storage(rag._graph_storage)

    return rag


def insert_texts(texts: list[str], working_dir: str | Path | None = None) -> None:
    """Insert texts into the knowledge graph.

    Args:
        texts: List of text strings to process and insert.
        working_dir: Optional override for KG storage directory.
    """
    rag = get_rag(working_dir)
    rag.insert(texts)


def query_graph(
    question: str,
    mode: str = "global",
    working_dir: str | Path | None = None,
) -> str:
    """Query the knowledge graph.

    Args:
        question: Natural language question.
        mode: Query mode — "local", "global", or "naive".
        working_dir: Optional override for KG storage directory.

    Returns:
        LLM-generated answer based on KG context.
    """
    from nano_graphrag.base import QueryParam

    rag = get_rag(working_dir)
    return rag.query(question, param=QueryParam(mode=mode))
