"""
Tests for the RelationshipNormalizer class.

Tests threshold resolution, entity/type counting, mapping construction,
normalization-and-merge logic, canonical label selection, and save/load.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from qualitative_analysis.relationships.normalizer import (
    RelationshipNormalizer,
    THRESHOLD_PRESETS,
)
from qualitative_analysis.relationships.models import (
    Relationship,
    EntityCluster,
    NormalizedRelationship,
    NormalizationResult,
)


class TestThresholdResolution:
    """Tests for _resolve_threshold."""

    @pytest.fixture
    def normalizer(self):
        return RelationshipNormalizer.__new__(RelationshipNormalizer)

    def test_float_passthrough(self, normalizer):
        assert normalizer._resolve_threshold(0.9) == 0.9

    def test_preset_conservative(self, normalizer):
        assert normalizer._resolve_threshold("conservative") == 0.85

    def test_preset_moderate(self, normalizer):
        assert normalizer._resolve_threshold("moderate") == 0.75

    def test_preset_aggressive(self, normalizer):
        assert normalizer._resolve_threshold("aggressive") == 0.60

    def test_string_float(self, normalizer):
        assert normalizer._resolve_threshold("0.8") == 0.8

    def test_unknown_preset_defaults_moderate(self, normalizer):
        assert normalizer._resolve_threshold("nonexistent") == 0.75


class TestEntityAndTypeCounting:
    """Tests for _count_entities_and_types."""

    @pytest.fixture
    def normalizer(self):
        return RelationshipNormalizer.__new__(RelationshipNormalizer)

    def test_basic_counts(self, normalizer):
        rels = [
            Relationship(source="A", target="B", type="causes"),
            Relationship(source="A", target="C", type="causes"),
            Relationship(source="B", target="C", type="correlates"),
        ]
        entity_counts, type_counts = normalizer._count_entities_and_types(rels)
        assert entity_counts["A"] == 2
        assert entity_counts["B"] == 2  # once as source, once as target
        assert entity_counts["C"] == 2  # twice as target
        assert type_counts["causes"] == 2
        assert type_counts["correlates"] == 1

    def test_empty_relationships(self, normalizer):
        entity_counts, type_counts = normalizer._count_entities_and_types([])
        assert entity_counts == {}
        assert type_counts == {}


class TestMappingConstruction:
    """Tests for _build_mapping."""

    @pytest.fixture
    def normalizer(self):
        return RelationshipNormalizer.__new__(RelationshipNormalizer)

    def test_builds_mapping(self, normalizer):
        clusters = [
            EntityCluster(canonical="water", members=["water", "H2O", "agua"]),
            EntityCluster(canonical="soil", members=["soil", "earth", "ground"]),
        ]
        mapping = normalizer._build_mapping(clusters)
        assert mapping["water"] == "water"
        assert mapping["H2O"] == "water"
        assert mapping["agua"] == "water"
        assert mapping["soil"] == "soil"
        assert mapping["ground"] == "soil"

    def test_single_member_cluster(self, normalizer):
        clusters = [EntityCluster(canonical="only", members=["only"])]
        mapping = normalizer._build_mapping(clusters)
        assert mapping["only"] == "only"


class TestNormalizationAndMerge:
    """Tests for _apply_normalization_and_merge."""

    @pytest.fixture
    def normalizer(self):
        return RelationshipNormalizer.__new__(RelationshipNormalizer)

    def test_merge_identical_tuples(self, normalizer):
        rels = [
            Relationship(source="A", target="B", type="causes",
                         description="d1", window_index=0, text_id="t1"),
            Relationship(source="A", target="B", type="causes",
                         description="d2", window_index=1, text_id="t2"),
        ]
        entity_mapping = {"A": "A", "B": "B"}
        type_mapping = {"causes": "causes"}
        result = normalizer._apply_normalization_and_merge(rels, entity_mapping, type_mapping)
        assert len(result) == 1
        assert result[0].count == 2
        assert len(result[0].descriptions) == 2
        assert set(result[0].text_ids) == {"t1", "t2"}
        assert 0 in result[0].window_indices
        assert 1 in result[0].window_indices

    def test_merge_with_entity_mapping(self, normalizer):
        rels = [
            Relationship(source="H2O", target="Plant Growth", type="promotes",
                         description="d1", text_id="t1"),
            Relationship(source="water", target="plant growth", type="enhances",
                         description="d2", text_id="t1"),
        ]
        entity_mapping = {"H2O": "water", "water": "water",
                          "Plant Growth": "plant growth", "plant growth": "plant growth"}
        type_mapping = {"promotes": "promotes", "enhances": "promotes"}

        result = normalizer._apply_normalization_and_merge(rels, entity_mapping, type_mapping)
        assert len(result) == 1
        assert result[0].source == "water"
        assert result[0].type == "promotes"
        assert result[0].count == 2

    def test_no_merge_different_tuples(self, normalizer):
        rels = [
            Relationship(source="A", target="B", type="causes"),
            Relationship(source="A", target="C", type="causes"),
        ]
        entity_mapping = {"A": "A", "B": "B", "C": "C"}
        type_mapping = {"causes": "causes"}
        result = normalizer._apply_normalization_and_merge(rels, entity_mapping, type_mapping)
        assert len(result) == 2

    def test_sorted_by_count_descending(self, normalizer):
        rels = [
            Relationship(source="A", target="B", type="t1"),
            Relationship(source="C", target="D", type="t2"),
            Relationship(source="C", target="D", type="t2"),
            Relationship(source="C", target="D", type="t2"),
        ]
        entity_mapping = {"A": "A", "B": "B", "C": "C", "D": "D"}
        type_mapping = {"t1": "t1", "t2": "t2"}
        result = normalizer._apply_normalization_and_merge(rels, entity_mapping, type_mapping)
        assert result[0].count == 3  # C->D->t2
        assert result[1].count == 1  # A->B->t1

    def test_dedup_descriptions(self, normalizer):
        rels = [
            Relationship(source="A", target="B", type="t", description="same desc"),
            Relationship(source="A", target="B", type="t", description="same desc"),
        ]
        entity_mapping = {"A": "A", "B": "B"}
        type_mapping = {"t": "t"}
        result = normalizer._apply_normalization_and_merge(rels, entity_mapping, type_mapping)
        assert len(result[0].descriptions) == 1

    def test_empty_text_id_excluded(self, normalizer):
        rels = [
            Relationship(source="A", target="B", type="t", text_id=""),
            Relationship(source="A", target="B", type="t", text_id="t1"),
        ]
        entity_mapping = {"A": "A", "B": "B"}
        type_mapping = {"t": "t"}
        result = normalizer._apply_normalization_and_merge(rels, entity_mapping, type_mapping)
        assert "" not in result[0].text_ids
        assert "t1" in result[0].text_ids


class TestCanonicalLabelSelection:
    """Tests for _assign_canonical_labels."""

    @pytest.fixture
    def normalizer(self):
        return RelationshipNormalizer.__new__(RelationshipNormalizer)

    def test_shortest_method(self, normalizer):
        clusters = [
            EntityCluster(canonical="", members=["water", "H2O", "liquid water"]),
        ]
        counts = {"water": 5, "H2O": 3, "liquid water": 1}
        result = normalizer._assign_canonical_labels(
            clusters, counts, "shortest", MagicMock(), None
        )
        assert result[0].canonical == "H2O"  # shortest
        assert result[0].count == 9  # sum of all

    def test_frequent_method(self, normalizer):
        clusters = [
            EntityCluster(canonical="", members=["water", "H2O", "liquid water"]),
        ]
        counts = {"water": 10, "H2O": 3, "liquid water": 1}
        result = normalizer._assign_canonical_labels(
            clusters, counts, "frequent", MagicMock(), None
        )
        assert result[0].canonical == "water"  # most frequent

    def test_single_member_cluster(self, normalizer):
        clusters = [
            EntityCluster(canonical="", members=["only_member"]),
        ]
        counts = {"only_member": 5}
        result = normalizer._assign_canonical_labels(
            clusters, counts, "shortest", MagicMock(), None
        )
        assert result[0].canonical == "only_member"
        assert result[0].count == 5


class TestNormalizeEmpty:
    """Test normalize() with edge cases."""

    def test_empty_relationships(self):
        normalizer = RelationshipNormalizer.__new__(RelationshipNormalizer)
        normalizer.embedding_model_name = "test"
        normalizer.device = None
        normalizer._embedding_service = None
        result = normalizer.normalize([])
        assert result.entity_mapping == {}
        assert result.normalized_relationships == []
        assert "error" in result.config


class TestNormalizeCausalRelationships:
    """Test that normalizer can accept CausalRelationship objects (post-causal pipeline)."""

    @pytest.fixture
    def normalizer(self):
        return RelationshipNormalizer.__new__(RelationshipNormalizer)

    def test_merge_causal_relationships(self, normalizer):
        from qualitative_analysis.relationships.models import (
            CausalRelationship,
            CausalAttributes,
        )
        rels = [
            CausalRelationship(
                source="A", target="B", type="causes",
                description="A causes B",
                causal=CausalAttributes(is_causal=True, polarity="positive"),
                text_ids=["t1"], window_indices=[0],
            ),
            CausalRelationship(
                source="A", target="B", type="causes",
                description="A leads to B",
                causal=CausalAttributes(is_causal=True, polarity="negative"),
                text_ids=["t2"], window_indices=[1],
            ),
        ]
        entity_mapping = {"A": "A", "B": "B"}
        type_mapping = {"causes": "causes"}
        result = normalizer._apply_normalization_and_merge(rels, entity_mapping, type_mapping)
        assert len(result) == 1
        assert result[0].count == 2
        assert set(result[0].text_ids) == {"t1", "t2"}
        assert set(result[0].window_indices) == {0, 1}
        assert len(result[0].descriptions) == 2


class TestSaveLoad:
    """Test save/load roundtrip."""

    def test_save_and_load(self):
        normalizer = RelationshipNormalizer.__new__(RelationshipNormalizer)
        normalizer.embedding_model_name = "test"

        result = NormalizationResult(
            entity_mapping={"A": "a", "B": "b"},
            type_mapping={"causes": "causes"},
            entity_clusters=[EntityCluster(canonical="a", members=["A", "a"], count=2)],
            type_clusters=[EntityCluster(canonical="causes", members=["causes"], count=1)],
            normalized_relationships=[
                NormalizedRelationship(
                    source="a", target="b", type="causes",
                    count=2, text_ids=["t1"],
                )
            ],
            config={"entity_threshold": 0.75},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            normalizer.save(result, temp_path)
            loaded = normalizer.load(temp_path)
            assert loaded.entity_mapping == result.entity_mapping
            assert len(loaded.entity_clusters) == 1
            assert loaded.entity_clusters[0].canonical == "a"
            assert len(loaded.normalized_relationships) == 1
            assert loaded.normalized_relationships[0].count == 2
        finally:
            temp_path.unlink(missing_ok=True)
