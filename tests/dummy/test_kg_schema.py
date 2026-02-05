"""Tests for KG schema: entity normalization, type validation, relation validation."""

from __future__ import annotations

import pytest

from litgraph.kg.schema import (
    normalize_entity,
    normalize_relation_type,
    reset_schema_cache,
    validate_entity,
    validate_relation,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_schema_cache()
    yield
    reset_schema_cache()


class TestNormalizeEntity:
    def test_alias_match(self, sample_schema):
        assert normalize_entity("scRNA-seq", sample_schema) == "single-cell RNA sequencing"

    def test_alias_case_insensitive(self, sample_schema):
        assert normalize_entity("scrna-seq", sample_schema) == "single-cell RNA sequencing"
        assert normalize_entity("SCRNA-SEQ", sample_schema) == "single-cell RNA sequencing"

    def test_alias_variant(self, sample_schema):
        assert normalize_entity("scRNAseq", sample_schema) == "single-cell RNA sequencing"

    def test_alias_vae(self, sample_schema):
        assert normalize_entity("VAE", sample_schema) == "variational autoencoder"

    def test_alias_gan(self, sample_schema):
        assert normalize_entity("GAN", sample_schema) == "generative adversarial network"

    def test_unknown_passthrough(self, sample_schema):
        """Unknown entities should be returned as-is."""
        assert normalize_entity("transformer", sample_schema) == "transformer"
        assert normalize_entity("BERT", sample_schema) == "BERT"

    def test_empty_string(self, sample_schema):
        assert normalize_entity("", sample_schema) == ""


class TestValidateEntity:
    def test_valid_types(self, sample_schema):
        assert validate_entity("test", "Paper", sample_schema) is True
        assert validate_entity("test", "Concept", sample_schema) is True
        assert validate_entity("test", "Method", sample_schema) is True
        assert validate_entity("test", "Dataset", sample_schema) is True
        assert validate_entity("test", "Finding", sample_schema) is True
        assert validate_entity("test", "Task", sample_schema) is True

    def test_invalid_type(self, sample_schema):
        assert validate_entity("test", "Unknown", sample_schema) is False
        assert validate_entity("test", "Person", sample_schema) is False


class TestValidateRelation:
    def test_valid_relations(self, sample_schema):
        assert validate_relation("Paper", "uses_method", "Method", sample_schema) is True
        assert validate_relation("Paper", "studies_topic", "Concept", sample_schema) is True
        assert validate_relation("Paper", "proposes", "Method", sample_schema) is True
        assert validate_relation("Method", "evaluated_on", "Dataset", sample_schema) is True

    def test_invalid_relation_type(self, sample_schema):
        assert validate_relation("Paper", "nonexistent", "Method", sample_schema) is False

    def test_wrong_endpoints(self, sample_schema):
        """Valid relation type but wrong endpoint types."""
        assert validate_relation("Method", "uses_method", "Paper", sample_schema) is False
        assert validate_relation("Concept", "proposes", "Method", sample_schema) is False


class TestNormalizeRelationType:
    def test_valid_stays(self, sample_schema):
        assert normalize_relation_type("Paper", "uses_method", "Method", sample_schema) == "uses_method"

    def test_invalid_degrades(self, sample_schema):
        """Invalid relation should degrade to 'related_to'."""
        assert normalize_relation_type("Paper", "nonexistent", "Method", sample_schema) == "related_to"

    def test_wrong_endpoints_degrades(self, sample_schema):
        """Wrong endpoints should degrade to 'related_to'."""
        assert normalize_relation_type("Method", "uses_method", "Paper", sample_schema) == "related_to"
