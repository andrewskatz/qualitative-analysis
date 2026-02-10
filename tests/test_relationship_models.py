"""
Tests for relationship pipeline data models.

Tests serialization (to_dict), deserialization (from_dict), roundtrips,
and edge cases for all model classes in the relationships module.
"""

import pytest

from qualitative_analysis.relationships.models import (
    Summary,
    Relationship,
    RelationshipResult,
    EntityCluster,
    NormalizedEntity,
    NormalizedRelationship,
    NormalizationResult,
    VerifiedRelationship,
    CausalAttributes,
    CausalRelationship,
    CausalAnalysisResult,
    RelationshipGraphNode,
    RelationshipGraphEdge,
    RelationshipGraphData,
    Checkpoint,
)


class TestSummary:
    def test_creation(self):
        s = Summary(text="A summary", window_index=3)
        assert s.text == "A summary"
        assert s.window_index == 3

    def test_defaults(self):
        s = Summary(text="x")
        assert s.window_index == 0


class TestRelationship:
    def test_creation(self):
        rel = Relationship(
            source="A",
            target="B",
            type="causes",
            description="A causes B",
            window_index=1,
            text_id="t1",
        )
        assert rel.source == "A"
        assert rel.target == "B"
        assert rel.type == "causes"
        assert rel.description == "A causes B"
        assert rel.window_index == 1
        assert rel.text_id == "t1"

    def test_defaults(self):
        rel = Relationship(source="A", target="B", type="causes")
        assert rel.description == ""
        assert rel.window_index == 0
        assert rel.text_id == ""

    def test_to_dict(self):
        rel = Relationship(source="A", target="B", type="causes", text_id="t1")
        d = rel.to_dict()
        assert d["source"] == "A"
        assert d["target"] == "B"
        assert d["type"] == "causes"
        assert d["text_id"] == "t1"

    def test_to_tuple(self):
        rel = Relationship(source="  Entity A ", target="Entity B", type=" CAUSES ")
        assert rel.to_tuple() == ("entity a", "causes", "entity b")

    def test_to_tuple_case_insensitive(self):
        r1 = Relationship(source="Water", target="Plants", type="Nourishes")
        r2 = Relationship(source="water", target="plants", type="nourishes")
        assert r1.to_tuple() == r2.to_tuple()


class TestRelationshipResult:
    def test_creation(self):
        rels = [Relationship(source="A", target="B", type="causes")]
        result = RelationshipResult(
            entities=["A", "B"],
            relationships=rels,
            metadata={"window_count": 3},
        )
        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert result.metadata["window_count"] == 3

    def test_defaults(self):
        result = RelationshipResult()
        assert result.entities == []
        assert result.relationships == []
        assert result.metadata == {}

    def test_to_dict(self):
        rels = [Relationship(source="A", target="B", type="causes")]
        result = RelationshipResult(entities=["A", "B"], relationships=rels)
        d = result.to_dict()
        assert d["entities"] == ["A", "B"]
        assert len(d["relationships"]) == 1
        assert d["relationships"][0]["source"] == "A"


class TestEntityCluster:
    def test_creation(self):
        cluster = EntityCluster(
            canonical="water",
            members=["water", "H2O", "agua"],
            count=5,
            avg_similarity=0.92,
        )
        assert cluster.canonical == "water"
        assert len(cluster.members) == 3
        assert cluster.count == 5
        assert cluster.avg_similarity == 0.92

    def test_to_dict(self):
        cluster = EntityCluster(canonical="water", members=["water", "H2O"])
        d = cluster.to_dict()
        assert d["canonical"] == "water"
        assert d["members"] == ["water", "H2O"]


class TestNormalizedEntity:
    def test_creation(self):
        entity = NormalizedEntity(
            canonical="water",
            original="H2O",
            variants=["water", "H2O"],
            frequency=3,
            confidence=0.95,
        )
        assert entity.canonical == "water"
        assert entity.original == "H2O"
        assert len(entity.variants) == 2

    def test_defaults(self):
        entity = NormalizedEntity(canonical="water")
        assert entity.original == ""
        assert entity.variants == []
        assert entity.frequency == 1
        assert entity.confidence == 1.0

    def test_to_dict(self):
        entity = NormalizedEntity(canonical="water", original="H2O")
        d = entity.to_dict()
        assert d["canonical"] == "water"
        assert d["original"] == "H2O"


class TestNormalizedRelationship:
    def test_creation(self):
        rel = NormalizedRelationship(
            source="water",
            target="plant growth",
            type="promotes",
            original_source="H2O",
            original_target="Plant Growth",
            original_type="enhances",
            description="Water promotes growth",
            descriptions=["Water promotes growth", "H2O enhances plants"],
            confidence=0.9,
            count=2,
            window_indices=[0, 1],
            text_ids=["t1", "t2"],
        )
        assert rel.source == "water"
        assert rel.original_source == "H2O"
        assert rel.count == 2
        assert len(rel.descriptions) == 2

    def test_defaults(self):
        rel = NormalizedRelationship(source="A", target="B", type="causes")
        assert rel.original_source == ""
        assert rel.descriptions == []
        assert rel.confidence == 1.0
        assert rel.count == 1
        assert rel.window_indices == []
        assert rel.text_ids == []

    def test_to_dict(self):
        rel = NormalizedRelationship(
            source="A", target="B", type="causes",
            count=3, text_ids=["t1"],
        )
        d = rel.to_dict()
        assert d["count"] == 3
        assert d["text_ids"] == ["t1"]

    def test_to_tuple(self):
        rel = NormalizedRelationship(source="  Water ", target="Plants", type=" CAUSES ")
        assert rel.to_tuple() == ("water", "causes", "plants")


class TestNormalizationResult:
    @pytest.fixture
    def sample_result(self):
        return NormalizationResult(
            entity_mapping={"H2O": "water", "agua": "water"},
            type_mapping={"enhances": "promotes"},
            entity_clusters=[
                EntityCluster(canonical="water", members=["water", "H2O", "agua"], count=3)
            ],
            type_clusters=[
                EntityCluster(canonical="promotes", members=["promotes", "enhances"], count=2)
            ],
            normalized_relationships=[
                NormalizedRelationship(
                    source="water", target="plants", type="promotes",
                    original_source="H2O", original_target="Plants",
                    original_type="enhances", count=2,
                )
            ],
            config={"entity_threshold": 0.75},
        )

    def test_to_dict(self, sample_result):
        d = sample_result.to_dict()
        assert d["entity_mapping"]["H2O"] == "water"
        assert len(d["entity_clusters"]) == 1
        assert len(d["normalized_relationships"]) == 1
        assert d["config"]["entity_threshold"] == 0.75

    def test_from_dict_roundtrip(self, sample_result):
        d = sample_result.to_dict()
        restored = NormalizationResult.from_dict(d)
        assert restored.entity_mapping == sample_result.entity_mapping
        assert restored.type_mapping == sample_result.type_mapping
        assert len(restored.entity_clusters) == 1
        assert restored.entity_clusters[0].canonical == "water"
        assert len(restored.normalized_relationships) == 1
        assert restored.normalized_relationships[0].count == 2

    def test_from_dict_empty(self):
        result = NormalizationResult.from_dict({})
        assert result.entity_mapping == {}
        assert result.entity_clusters == []
        assert result.normalized_relationships == []


class TestVerifiedRelationship:
    def test_creation(self):
        vr = VerifiedRelationship(
            source="A",
            target="B",
            type="causes",
            description="A causes B",
            verification_confidence=0.85,
            verification_note="Strongly supported",
            verified=True,
        )
        assert vr.verification_confidence == 0.85
        assert vr.verified is True

    def test_defaults(self):
        vr = VerifiedRelationship(source="A", target="B", type="causes")
        assert vr.verification_confidence == 1.0
        assert vr.verification_note == ""
        assert vr.verified is True
        assert vr.original_relationship is None

    def test_to_dict(self):
        vr = VerifiedRelationship(
            source="A", target="B", type="causes",
            verification_confidence=0.3, verified=False,
        )
        d = vr.to_dict()
        assert d["source"] == "A"
        assert d["verification_confidence"] == 0.3
        assert d["verified"] is False
        # original_relationship should not appear in to_dict
        assert "original_relationship" not in d

    def test_with_original_relationship(self):
        orig = Relationship(source="A", target="B", type="causes")
        vr = VerifiedRelationship(
            source="A", target="B", type="causes",
            original_relationship=orig,
        )
        assert vr.original_relationship is orig


class TestCausalAttributes:
    def test_causal(self):
        attrs = CausalAttributes(
            is_causal=True,
            polarity="positive",
            certainty="certain",
            explicit_vs_implicit="explicit",
            reasoning="Clear cause-effect",
        )
        assert attrs.is_causal is True
        assert attrs.polarity == "positive"
        assert attrs.certainty == "certain"

    def test_non_causal_defaults(self):
        attrs = CausalAttributes()
        assert attrs.is_causal is False
        assert attrs.polarity is None
        assert attrs.certainty is None
        assert attrs.explicit_vs_implicit is None

    def test_to_dict(self):
        attrs = CausalAttributes(
            is_causal=True, polarity="negative",
            certainty="likely", explicit_vs_implicit="implicit",
        )
        d = attrs.to_dict()
        assert d["is_causal"] is True
        assert d["polarity"] == "negative"
        assert d["certainty"] == "likely"
        assert d["explicit_vs_implicit"] == "implicit"

    def test_from_dict(self):
        data = {
            "is_causal": True,
            "polarity": "positive",
            "certainty": "certain",
            "explicit_vs_implicit": "explicit",
            "reasoning": "Clear pattern",
        }
        attrs = CausalAttributes.from_dict(data)
        assert attrs.is_causal is True
        assert attrs.polarity == "positive"
        assert attrs.reasoning == "Clear pattern"

    def test_from_dict_with_explanation_key(self):
        """from_dict should accept 'explanation' as fallback for 'reasoning'."""
        data = {"is_causal": True, "explanation": "Some reason"}
        attrs = CausalAttributes.from_dict(data)
        assert attrs.reasoning == "Some reason"

    def test_from_dict_empty(self):
        attrs = CausalAttributes.from_dict({})
        assert attrs.is_causal is False
        assert attrs.polarity is None


class TestCausalRelationship:
    def test_creation(self):
        causal = CausalAttributes(is_causal=True, polarity="positive")
        rel = CausalRelationship(
            source="A", target="B", type="causes",
            causal=causal,
            original_source="a", original_target="b",
            count=3, text_ids=["t1", "t2"],
        )
        assert rel.causal.is_causal is True
        assert rel.count == 3

    def test_to_dict(self):
        causal = CausalAttributes(
            is_causal=True, polarity="negative",
            certainty="likely", reasoning="evident",
        )
        rel = CausalRelationship(
            source="A", target="B", type="causes",
            causal=causal, count=2, text_ids=["t1"],
        )
        d = rel.to_dict()
        assert d["is_causal"] is True
        assert d["polarity"] == "negative"
        assert d["count"] == 2

    def test_from_dict_roundtrip(self):
        causal = CausalAttributes(
            is_causal=True, polarity="positive",
            certainty="certain", explicit_vs_implicit="explicit",
            reasoning="obvious cause",
        )
        rel = CausalRelationship(
            source="A", target="B", type="causes",
            description="A causes B", causal=causal,
            original_source="a", count=2,
            window_indices=[0, 1], text_ids=["t1"],
        )
        d = rel.to_dict()
        restored = CausalRelationship.from_dict(d)
        assert restored.source == "A"
        assert restored.causal.is_causal is True
        assert restored.causal.polarity == "positive"
        assert restored.count == 2
        assert restored.window_indices == [0, 1]

    def test_from_relationship(self):
        base = Relationship(
            source="X", target="Y", type="influences",
            description="X influences Y", window_index=3, text_id="t5",
        )
        causal = CausalAttributes(is_causal=True, polarity="neutral")
        cr = CausalRelationship.from_relationship(base, causal)
        assert cr.source == "X"
        assert cr.target == "Y"
        assert cr.description == "X influences Y"
        assert cr.causal.polarity == "neutral"
        assert cr.window_indices == [3]
        assert cr.text_ids == ["t5"]

    def test_from_relationship_no_text_id(self):
        base = Relationship(source="X", target="Y", type="t")
        cr = CausalRelationship.from_relationship(base)
        assert cr.text_ids == []
        assert cr.causal.is_causal is False


class TestCausalAnalysisResult:
    @pytest.fixture
    def sample_result(self):
        rels = [
            CausalRelationship(
                source="A", target="B", type="causes",
                causal=CausalAttributes(
                    is_causal=True, polarity="positive",
                    certainty="certain", explicit_vs_implicit="explicit",
                ),
            ),
            CausalRelationship(
                source="C", target="D", type="correlates",
                causal=CausalAttributes(is_causal=False),
            ),
            CausalRelationship(
                source="E", target="F", type="reduces",
                causal=CausalAttributes(
                    is_causal=True, polarity="negative",
                    certainty="likely", explicit_vs_implicit="implicit",
                ),
            ),
        ]
        return CausalAnalysisResult(causal_relationships=rels)

    def test_compute_statistics(self, sample_result):
        stats = sample_result.compute_statistics()
        assert stats["total"] == 3
        assert stats["causal_count"] == 2
        assert stats["non_causal_count"] == 1
        assert stats["positive_count"] == 1
        assert stats["negative_count"] == 1
        assert stats["certain_count"] == 1
        assert stats["likely_count"] == 1
        assert stats["explicit_count"] == 1
        assert stats["implicit_count"] == 1

    def test_compute_statistics_empty(self):
        result = CausalAnalysisResult()
        stats = result.compute_statistics()
        assert stats["total"] == 0

    def test_to_dict_calls_compute_statistics(self):
        result = CausalAnalysisResult(causal_relationships=[
            CausalRelationship(
                source="A", target="B", type="causes",
                causal=CausalAttributes(is_causal=True, polarity="positive"),
            ),
        ])
        d = result.to_dict()
        assert "stats" in d
        assert d["stats"]["causal_count"] == 1

    def test_from_dict_roundtrip(self, sample_result):
        d = sample_result.to_dict()
        restored = CausalAnalysisResult.from_dict(d)
        assert len(restored.causal_relationships) == 3
        assert restored.causal_relationships[0].causal.is_causal is True
        assert restored.causal_relationships[1].causal.is_causal is False

    def test_from_dict_with_relationships_key(self):
        """Should accept 'relationships' as alternative to 'causal_relationships'."""
        data = {
            "relationships": [
                {"source": "A", "target": "B", "type": "c", "is_causal": True}
            ],
        }
        result = CausalAnalysisResult.from_dict(data)
        assert len(result.causal_relationships) == 1


class TestRelationshipGraphNode:
    def test_creation(self):
        node = RelationshipGraphNode(
            id="water", label="Water",
            frequency=5, degree=3,
            betweenness=0.5, pagerank=0.1,
        )
        assert node.id == "water"
        assert node.label == "Water"
        assert node.frequency == 5

    def test_defaults(self):
        node = RelationshipGraphNode(id="x", label="X")
        assert node.frequency == 1
        assert node.degree == 0
        assert node.betweenness == 0.0
        assert node.raw_entities == []

    def test_to_dict(self):
        node = RelationshipGraphNode(
            id="x", label="X", frequency=2,
            source_text_ids=["t1"], raw_entities=["x1", "x2"],
        )
        d = node.to_dict()
        assert d["id"] == "x"
        assert d["raw_entities"] == ["x1", "x2"]


class TestRelationshipGraphEdge:
    def test_creation(self):
        edge = RelationshipGraphEdge(
            id="e0", source="a", target="b", type="causes",
            weight=3, evidence=["ev1", "ev2"],
        )
        assert edge.weight == 3
        assert len(edge.evidence) == 2

    def test_to_dict_without_causal(self):
        edge = RelationshipGraphEdge(id="e0", source="a", target="b", type="t")
        d = edge.to_dict()
        assert "causal_attributes" not in d

    def test_to_dict_with_causal(self):
        edge = RelationshipGraphEdge(
            id="e0", source="a", target="b", type="t",
            causal_attributes={"is_causal": True, "polarity": "positive"},
        )
        d = edge.to_dict()
        assert d["causal_attributes"]["is_causal"] is True


class TestRelationshipGraphData:
    @pytest.fixture
    def sample_graph_data(self):
        nodes = [
            RelationshipGraphNode(id="a", label="A", frequency=3, degree=2),
            RelationshipGraphNode(id="b", label="B", frequency=1, degree=1),
        ]
        edges = [
            RelationshipGraphEdge(id="e0", source="a", target="b", type="causes", weight=2),
        ]
        return RelationshipGraphData(
            nodes=nodes, edges=edges,
            metrics_summary={"node_count": 2, "edge_count": 1},
            is_normalized=True, has_causal=False,
        )

    def test_to_dict(self, sample_graph_data):
        d = sample_graph_data.to_dict()
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert d["is_normalized"] is True
        assert d["has_causal"] is False

    def test_from_dict_roundtrip(self, sample_graph_data):
        d = sample_graph_data.to_dict()
        restored = RelationshipGraphData.from_dict(d)
        assert len(restored.nodes) == 2
        assert restored.nodes[0].id == "a"
        assert len(restored.edges) == 1
        assert restored.edges[0].weight == 2
        assert restored.is_normalized is True

    def test_from_dict_empty(self):
        data = RelationshipGraphData.from_dict({})
        assert data.nodes == []
        assert data.edges == []
        assert data.is_normalized is False


class TestCheckpoint:
    def test_creation(self):
        cp = Checkpoint(
            processed_indices=[0, 1, 2],
            partial_results=[{"source": "A"}],
            config={"model": "test"},
            timestamp="2026-02-08T10:00:00",
            input_file="input.csv",
            total_items=10,
            step="detect",
        )
        assert len(cp.processed_indices) == 3
        assert cp.step == "detect"

    def test_to_dict_from_dict_roundtrip(self):
        cp = Checkpoint(
            processed_indices=[0, 1],
            partial_results=[{"key": "val"}],
            config={"temperature": 0.3},
            timestamp="2026-02-08",
            input_file="test.csv",
            total_items=5,
            step="causal",
        )
        d = cp.to_dict()
        restored = Checkpoint.from_dict(d)
        assert restored.processed_indices == [0, 1]
        assert restored.step == "causal"
        assert restored.total_items == 5

    def test_from_dict_empty(self):
        cp = Checkpoint.from_dict({})
        assert cp.processed_indices == []
        assert cp.step == ""
