"""
Integration tests for the relationship analysis pipeline.

Tests the causal analyzer, graph builder, and end-to-end flows
through the pipeline stages (detect → normalize → causal → graph).
"""

import asyncio
import json
import tempfile
import pytest
from pathlib import Path

from qualitative_analysis.relationships.models import (
    Relationship,
    NormalizedRelationship,
    CausalAttributes,
    CausalRelationship,
    CausalAnalysisResult,
    RelationshipGraphData,
)
from qualitative_analysis.relationships.causal import CausalAnalyzer
from qualitative_analysis.relationships.graph import RelationshipGraph


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# =============================================================================
# CAUSAL ANALYZER TESTS
# =============================================================================


class MockCausalLLM:
    """Mock LLM that returns causal analysis JSON."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    async def generate(self, prompt: str, temperature: float = 0.3, **kwargs) -> str:
        idx = self.call_count % max(len(self.responses), 1)
        self.call_count += 1
        if self.responses:
            return self.responses[idx]
        return json.dumps({
            "reasoning": "Default mock reasoning",
            "is_causal": True,
            "polarity": "positive",
            "certainty": "likely",
            "explicit_vs_implicit": "explicit",
        })


class TestCausalAnalyzer:
    def test_analyze_empty(self):
        analyzer = CausalAnalyzer()
        result = run_async(analyzer.analyze([], MockCausalLLM()))
        assert result.stats["total"] == 0
        assert result.causal_relationships == []

    def test_analyze_single_causal(self):
        rels = [Relationship(source="A", target="B", type="causes", description="A causes B")]
        llm = MockCausalLLM([json.dumps({
            "reasoning": "Clear cause-effect",
            "is_causal": True,
            "polarity": "positive",
            "certainty": "certain",
            "explicit_vs_implicit": "explicit",
        })])
        analyzer = CausalAnalyzer()
        result = run_async(analyzer.analyze(rels, llm))
        assert result.stats["total"] == 1
        assert result.stats["causal_count"] == 1
        assert result.stats["positive_count"] == 1
        assert result.stats["certain_count"] == 1

    def test_analyze_non_causal(self):
        rels = [Relationship(source="A", target="B", type="near", description="A is near B")]
        llm = MockCausalLLM([json.dumps({
            "reasoning": "Spatial, not causal",
            "is_causal": False,
        })])
        analyzer = CausalAnalyzer()
        result = run_async(analyzer.analyze(rels, llm))
        assert result.stats["causal_count"] == 0
        assert result.stats["non_causal_count"] == 1

    def test_analyze_mixed(self):
        rels = [
            Relationship(source="A", target="B", type="causes"),
            Relationship(source="C", target="D", type="near"),
        ]
        llm = MockCausalLLM([
            json.dumps({"is_causal": True, "polarity": "negative",
                        "certainty": "likely", "explicit_vs_implicit": "implicit"}),
            json.dumps({"is_causal": False, "reasoning": "not causal"}),
        ])
        analyzer = CausalAnalyzer()
        result = run_async(analyzer.analyze(rels, llm))
        assert result.stats["causal_count"] == 1
        assert result.stats["non_causal_count"] == 1
        assert result.stats["negative_count"] == 1

    def test_error_not_counted_as_non_causal(self):
        """Errors during analysis should NOT inflate non_causal_count."""
        rels = [Relationship(source="A", target="B", type="causes")]

        class FailingLLM:
            async def generate(self, **kwargs):
                raise RuntimeError("Connection refused")

        analyzer = CausalAnalyzer()
        result = run_async(analyzer.analyze(rels, FailingLLM()))
        assert result.stats["errors"] == 1
        assert result.stats["non_causal_count"] == 0

    def test_analyze_normalized_relationships(self):
        rels = [NormalizedRelationship(
            source="water", target="growth", type="promotes",
            original_source="H2O", original_target="Plant Growth",
            original_type="enhances",
            description="Water promotes growth",
            descriptions=["Water promotes growth", "H2O enhances plants"],
            count=2, window_indices=[0, 1], text_ids=["t1"],
        )]
        llm = MockCausalLLM([json.dumps({
            "is_causal": True, "polarity": "positive",
            "certainty": "certain", "explicit_vs_implicit": "explicit",
        })])
        analyzer = CausalAnalyzer()
        result = run_async(analyzer.analyze(rels, llm))
        cr = result.causal_relationships[0]
        assert cr.original_source == "H2O"
        assert cr.count == 2
        assert cr.text_ids == ["t1"]

    def test_progress_callback(self):
        rels = [
            Relationship(source="A", target="B", type="t"),
            Relationship(source="C", target="D", type="t"),
        ]
        llm = MockCausalLLM()
        progress_calls = []

        def on_progress(current, total):
            progress_calls.append((current, total))

        analyzer = CausalAnalyzer()
        run_async(analyzer.analyze(rels, llm, batch_size=1, on_progress=on_progress))
        # Should get progress calls
        assert len(progress_calls) > 0
        # Final call should be (total, total)
        assert progress_calls[-1] == (2, 2)


class TestCausalResponseParsing:
    def test_parse_clean_json(self):
        analyzer = CausalAnalyzer()
        attrs = analyzer._parse_response(json.dumps({
            "is_causal": True, "polarity": "positive",
            "certainty": "certain", "explicit_vs_implicit": "explicit",
            "reasoning": "clear",
        }))
        assert attrs.is_causal is True
        assert attrs.polarity == "positive"

    def test_parse_markdown_wrapped(self):
        analyzer = CausalAnalyzer()
        response = '```json\n{"is_causal": false, "reasoning": "not causal"}\n```'
        attrs = analyzer._parse_response(response)
        assert attrs.is_causal is False

    def test_parse_invalid_polarity_defaults(self):
        analyzer = CausalAnalyzer()
        attrs = analyzer._parse_response(json.dumps({
            "is_causal": True, "polarity": "INVALID",
            "certainty": "certain", "explicit_vs_implicit": "explicit",
        }))
        assert attrs.polarity == "neutral"  # default on invalid

    def test_parse_invalid_certainty_defaults(self):
        analyzer = CausalAnalyzer()
        attrs = analyzer._parse_response(json.dumps({
            "is_causal": True, "polarity": "positive",
            "certainty": "INVALID", "explicit_vs_implicit": "explicit",
        }))
        assert attrs.certainty == "possible"  # default on invalid

    def test_parse_non_causal_nulls_attributes(self):
        analyzer = CausalAnalyzer()
        attrs = analyzer._parse_response(json.dumps({
            "is_causal": False, "reasoning": "not causal",
        }))
        assert attrs.polarity is None
        assert attrs.certainty is None
        assert attrs.explicit_vs_implicit is None

    def test_parse_no_json_raises(self):
        analyzer = CausalAnalyzer()
        with pytest.raises(ValueError, match="No valid JSON"):
            analyzer._parse_response("This is not JSON at all")


class TestCausalSaveLoad:
    def test_save_and_load(self):
        result = CausalAnalysisResult(
            causal_relationships=[
                CausalRelationship(
                    source="A", target="B", type="causes",
                    causal=CausalAttributes(
                        is_causal=True, polarity="positive",
                        certainty="certain",
                    ),
                    count=2,
                ),
            ],
            stats={"total": 1, "causal_count": 1},
            config={"temperature": 0.3},
        )
        analyzer = CausalAnalyzer()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            analyzer.save(result, temp_path)
            loaded = analyzer.load(temp_path)
            assert len(loaded.causal_relationships) == 1
            assert loaded.causal_relationships[0].causal.is_causal is True
            assert loaded.stats["causal_count"] == 1
        finally:
            temp_path.unlink(missing_ok=True)


# =============================================================================
# GRAPH BUILDER TESTS
# =============================================================================


class TestGraphBuilder:
    @pytest.fixture
    def sample_relationships(self):
        return [
            Relationship(source="A", target="B", type="causes",
                         description="A causes B", text_id="t1", window_index=0),
            Relationship(source="A", target="B", type="causes",
                         description="A leads to B", text_id="t2", window_index=1),
            Relationship(source="B", target="C", type="influences",
                         description="B influences C", text_id="t1", window_index=0),
        ]

    def test_build_empty(self):
        graph = RelationshipGraph()
        data = graph.build([])
        assert data.metrics_summary["node_count"] == 0
        assert data.metrics_summary["edge_count"] == 0

    def test_build_basic(self, sample_relationships):
        graph = RelationshipGraph()
        data = graph.build(sample_relationships)
        assert len(data.nodes) == 3  # A, B, C
        assert len(data.edges) == 2  # A->B (causes), B->C (influences)

    def test_edge_weight_aggregation(self, sample_relationships):
        graph = RelationshipGraph()
        data = graph.build(sample_relationships)
        ab_edge = next(e for e in data.edges if e.source == "a" and e.target == "b")
        assert ab_edge.weight == 2  # Two A->B causes relationships

    def test_edge_evidence_aggregation(self, sample_relationships):
        graph = RelationshipGraph()
        data = graph.build(sample_relationships)
        ab_edge = next(e for e in data.edges if e.source == "a" and e.target == "b")
        assert "A causes B" in ab_edge.evidence
        assert "A leads to B" in ab_edge.evidence

    def test_node_metrics(self, sample_relationships):
        graph = RelationshipGraph()
        data = graph.build(sample_relationships)
        # B should have highest degree (2 edges: incoming from A, outgoing to C)
        b_node = next(n for n in data.nodes if n.id == "b")
        assert b_node.degree == 2
        assert b_node.in_degree == 1
        assert b_node.out_degree == 1

    def test_self_loops_skipped(self):
        rels = [
            Relationship(source="A", target="A", type="relates"),
            Relationship(source="A", target="B", type="causes"),
        ]
        graph = RelationshipGraph()
        data = graph.build(rels)
        assert len(data.edges) == 1  # Self-loop excluded

    def test_min_edge_weight_filter(self, sample_relationships):
        graph = RelationshipGraph()
        data = graph.build(sample_relationships, min_edge_weight=2)
        # Only A->B has weight 2, B->C has weight 1
        assert len(data.edges) == 1
        assert data.edges[0].source == "a"
        assert data.edges[0].target == "b"

    def test_text_id_aggregation(self, sample_relationships):
        graph = RelationshipGraph()
        data = graph.build(sample_relationships)
        ab_edge = next(e for e in data.edges if e.source == "a" and e.target == "b")
        assert "t1" in ab_edge.source_text_ids
        assert "t2" in ab_edge.source_text_ids

    def test_node_frequency(self, sample_relationships):
        graph = RelationshipGraph()
        data = graph.build(sample_relationships)
        a_node = next(n for n in data.nodes if n.id == "a")
        assert a_node.frequency == 2  # A appears in 2 relationships as source

    def test_nodes_sorted_by_frequency(self, sample_relationships):
        graph = RelationshipGraph()
        data = graph.build(sample_relationships)
        freqs = [n.frequency for n in data.nodes]
        assert freqs == sorted(freqs, reverse=True)


class TestGraphBuilderCausal:
    def test_causal_attributes_in_edges(self):
        rels = [
            CausalRelationship(
                source="A", target="B", type="causes",
                causal=CausalAttributes(
                    is_causal=True, polarity="positive",
                    certainty="certain", explicit_vs_implicit="explicit",
                ),
                text_ids=["t1"],
            ),
        ]
        graph = RelationshipGraph()
        data = graph.build(rels, include_causal=True)
        edge = data.edges[0]
        assert edge.causal_attributes is not None
        assert edge.causal_attributes["is_causal"] is True
        assert edge.causal_attributes["polarity"] == "positive"

    def test_causal_majority_vote(self):
        """Multiple causal relationships should be resolved by majority vote."""
        rels = [
            CausalRelationship(
                source="A", target="B", type="causes",
                causal=CausalAttributes(
                    is_causal=True, polarity="positive",
                    certainty="certain", explicit_vs_implicit="explicit",
                ),
                text_ids=["t1"],
            ),
            CausalRelationship(
                source="A", target="B", type="causes",
                causal=CausalAttributes(
                    is_causal=True, polarity="positive",
                    certainty="likely", explicit_vs_implicit="implicit",
                ),
                text_ids=["t2"],
            ),
            CausalRelationship(
                source="A", target="B", type="causes",
                causal=CausalAttributes(
                    is_causal=True, polarity="negative",
                    certainty="likely", explicit_vs_implicit="implicit",
                ),
                text_ids=["t3"],
            ),
        ]
        graph = RelationshipGraph()
        data = graph.build(rels, include_causal=True)
        edge = data.edges[0]
        assert edge.causal_attributes["polarity"] == "positive"  # 2 positive vs 1 negative
        assert edge.causal_attributes["certainty"] == "likely"  # 2 likely vs 1 certain
        assert edge.causal_attributes["explicit_vs_implicit"] == "implicit"  # 2 implicit vs 1 explicit

    def test_causal_excluded_when_disabled(self):
        rels = [
            CausalRelationship(
                source="A", target="B", type="causes",
                causal=CausalAttributes(is_causal=True, polarity="positive"),
            ),
        ]
        graph = RelationshipGraph()
        data = graph.build(rels, include_causal=False)
        assert data.edges[0].causal_attributes is None


class TestGraphFilter:
    def test_filter_by_weight(self):
        rels = [
            Relationship(source="A", target="B", type="t1"),
            Relationship(source="A", target="B", type="t1"),
            Relationship(source="C", target="D", type="t2"),
        ]
        graph = RelationshipGraph()
        graph.build(rels)
        filtered = graph.filter(min_edge_weight=2)
        assert len(filtered.edges) == 1

    def test_filter_by_type(self):
        rels = [
            Relationship(source="A", target="B", type="causes"),
            Relationship(source="C", target="D", type="correlates"),
        ]
        graph = RelationshipGraph()
        graph.build(rels)
        filtered = graph.filter(relationship_types=["causes"])
        assert len(filtered.edges) == 1
        assert filtered.edges[0].type == "causes"

    def test_filter_no_build_raises(self):
        graph = RelationshipGraph()
        with pytest.raises(ValueError, match="No graph data"):
            graph.filter()


class TestGraphExport:
    @pytest.fixture
    def built_graph(self):
        rels = [
            CausalRelationship(
                source="A", target="B", type="causes",
                description="A causes B",
                causal=CausalAttributes(
                    is_causal=True, polarity="positive",
                    certainty="certain", explicit_vs_implicit="explicit",
                ),
                text_ids=["t1"],
            ),
        ]
        graph = RelationshipGraph()
        graph.build(rels, include_causal=True)
        return graph

    def test_to_json(self, built_graph):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            data = built_graph.to_json(path)
            assert path.exists()
            assert "nodes" in data
            assert "edges" in data

            loaded = json.loads(path.read_text())
            assert len(loaded["nodes"]) == 2
            assert len(loaded["edges"]) == 1

    def test_to_csv(self, built_graph):
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes_path, edges_path = built_graph.to_csv(tmpdir, "test")
            assert Path(nodes_path).exists()
            assert Path(edges_path).exists()

            # Verify edge CSV has causal columns
            import csv
            with open(edges_path, "r") as f:
                reader = csv.DictReader(f)
                row = next(reader)
                assert "is_causal" in row
                assert "polarity" in row
                assert "certainty" in row
                assert "explicit_vs_implicit" in row

    def test_to_gexf(self, built_graph):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.gexf"
            result = built_graph.to_gexf(path)
            assert Path(result).exists()

            # Verify GEXF contains causal attributes
            content = path.read_text()
            assert "is_causal" in content
            assert "polarity" in content
            assert "certainty" in content
            assert "explicit_vs_implicit" in content

    def test_to_json_no_build_raises(self):
        graph = RelationshipGraph()
        with pytest.raises(ValueError, match="No graph data"):
            graph.to_json()


class TestGraphNodeRoles:
    def test_get_node_roles(self):
        rels = [
            Relationship(source="A", target="B", type="causes"),
            Relationship(source="B", target="C", type="leads_to"),
        ]
        graph = RelationshipGraph()
        graph.build(rels)
        roles = graph._get_node_roles()
        assert roles["a"] == "source"  # A is only a source
        assert roles["b"] == "both"    # B is both source and target
        assert roles["c"] == "target"  # C is only a target


class TestGraphMetrics:
    def test_metrics_summary(self):
        rels = [
            Relationship(source="A", target="B", type="causes"),
            Relationship(source="B", target="C", type="causes"),
            Relationship(source="A", target="C", type="influences"),
        ]
        graph = RelationshipGraph()
        data = graph.build(rels)
        metrics = data.metrics_summary
        assert metrics["node_count"] == 3
        assert metrics["edge_count"] == 3
        assert "density" in metrics
        assert "average_degree" in metrics
        assert "top_nodes_by_pagerank" in metrics

    def test_causal_edge_count(self):
        rels = [
            CausalRelationship(
                source="A", target="B", type="causes",
                causal=CausalAttributes(is_causal=True, polarity="positive"),
            ),
            CausalRelationship(
                source="C", target="D", type="near",
                causal=CausalAttributes(is_causal=False),
            ),
        ]
        graph = RelationshipGraph()
        data = graph.build(rels, include_causal=True)
        assert data.metrics_summary["causal_edge_count"] == 1


# =============================================================================
# END-TO-END PIPELINE TESTS
# =============================================================================


class TestEndToEndPipeline:
    """Test that data flows correctly across pipeline stages."""

    def test_relationships_to_graph(self):
        """Raw relationships → graph should work without intermediate steps."""
        rels = [
            Relationship(source="Water", target="Plant Growth", type="promotes",
                         description="Water promotes plant growth", text_id="p1"),
            Relationship(source="Sunlight", target="Plant Growth", type="enables",
                         description="Sunlight enables growth", text_id="p1"),
            Relationship(source="Water", target="Plant Growth", type="promotes",
                         description="Water drives growth", text_id="p2"),
        ]
        graph = RelationshipGraph()
        data = graph.build(rels)
        assert len(data.nodes) == 3
        # Water->Plant Growth has weight 2
        wp_edge = next(e for e in data.edges
                       if e.source == "water" and e.target == "plant growth")
        assert wp_edge.weight == 2

    def test_causal_to_graph_preserves_attributes(self):
        """Causal relationships → graph should preserve causal attributes."""
        rels = [
            CausalRelationship(
                source="Deforestation", target="Flooding", type="causes",
                description="Deforestation causes flooding",
                causal=CausalAttributes(
                    is_causal=True, polarity="negative",
                    certainty="likely", explicit_vs_implicit="explicit",
                ),
                text_ids=["p1"],
            ),
        ]
        graph = RelationshipGraph()
        data = graph.build(rels, include_causal=True)
        edge = data.edges[0]
        assert edge.causal_attributes["polarity"] == "negative"
        assert edge.causal_attributes["certainty"] == "likely"

    def test_normalized_relationships_to_graph(self):
        """Normalized relationships → graph should use normalized labels."""
        rels = [
            NormalizedRelationship(
                source="climate change", target="flooding", type="causes",
                original_source="Climate Change; global warming",
                original_target="Flooding; floods",
                count=3, text_ids=["p1", "p2"],
            ),
        ]
        graph = RelationshipGraph()
        data = graph.build(rels)
        assert data.is_normalized is True
        node = next(n for n in data.nodes if n.id == "climate change")
        assert len(node.raw_entities) > 0  # Should have original entities
