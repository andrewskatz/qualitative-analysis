"""
Tests for the domain mapping and normalization module.
"""

import unittest
import asyncio
from unittest.mock import Mock, AsyncMock

from qualitative_analysis.figurative.domains import (
    DomainMappedInstance,
    DomainCluster,
    NormalizationResult,
    DomainGraphNode,
    DomainGraphEdge,
    DomainGraphData,
    Checkpoint,
    NormalizeCheckpoint,
    DomainExtractor,
    DomainGraph,
)


class TestDomainModels(unittest.TestCase):
    """Test data model classes."""
    
    def test_domain_mapped_instance_creation(self):
        """Test creating a DomainMappedInstance."""
        instance = DomainMappedInstance(
            text="Time is a thief",
            type="metaphor",
            confidence=0.9,
            source_domain="theft",
            target_domain="time",
            source_domain_levels={"specific": "criminal theft", "moderate": "theft", "abstract": "possession loss"},
            target_domain_levels={"specific": "passing moments", "moderate": "time", "abstract": "abstract concept"},
        )
        
        self.assertEqual(instance.text, "Time is a thief")
        self.assertEqual(instance.type, "metaphor")
        self.assertEqual(instance.source_domain, "theft")
        self.assertEqual(instance.source_domain_levels["specific"], "criminal theft")
    
    def test_domain_cluster_creation(self):
        """Test creating a DomainCluster."""
        cluster = DomainCluster(
            canonical="JOURNEY",
            members=["journey", "path", "road", "trip"],
            count=15,
            avg_similarity=0.82,
        )
        
        self.assertEqual(cluster.canonical, "JOURNEY")
        self.assertEqual(len(cluster.members), 4)
        self.assertEqual(cluster.count, 15)
    
    def test_normalization_result_serialization(self):
        """Test NormalizationResult to_dict and from_dict."""
        result = NormalizationResult(
            source_mapping={"journey": "JOURNEY", "path": "JOURNEY"},
            target_mapping={"life": "EXISTENCE"},
            source_clusters=[
                DomainCluster(canonical="JOURNEY", members=["journey", "path"], count=5, avg_similarity=0.9)
            ],
            target_clusters=[
                DomainCluster(canonical="EXISTENCE", members=["life"], count=3, avg_similarity=1.0)
            ],
            config={"conservativeness": "moderate"},
        )
        
        # Serialize
        data = result.to_dict()
        self.assertIn("source_mapping", data)
        self.assertIn("source_clusters", data)
        
        # Deserialize
        restored = NormalizationResult.from_dict(data)
        self.assertEqual(restored.source_mapping["journey"], "JOURNEY")
        self.assertEqual(len(restored.source_clusters), 1)
        self.assertEqual(restored.source_clusters[0].canonical, "JOURNEY")
    
    def test_checkpoint_serialization(self):
        """Test Checkpoint to_dict and from_dict."""
        checkpoint = Checkpoint(
            processed_indices=[0, 1, 2],
            partial_results=[{"text": "test", "source_domain": "x"}],
            config={"model": "test"},
            timestamp="2026-01-06T12:00:00",
            total_items=10,
        )
        
        data = checkpoint.to_dict()
        restored = Checkpoint.from_dict(data)
        
        self.assertEqual(restored.processed_indices, [0, 1, 2])
        self.assertEqual(len(restored.partial_results), 1)
    
    def test_normalize_checkpoint_serialization(self):
        """Test NormalizeCheckpoint to_dict and from_dict."""
        checkpoint = NormalizeCheckpoint(
            phase="source",
            processed_source_indices=[0, 1, 2],
            processed_target_indices=[],
            source_clusters=[
                {"canonical": "JOURNEY", "members": ["journey", "path"], "avg_similarity": 0.85}
            ],
            target_clusters=[],
            config={"embedding_model": "test-model"},
            timestamp="2026-02-03T09:00:00",
            total_source_clusters=5,
            total_target_clusters=3,
        )
        
        data = checkpoint.to_dict()
        self.assertEqual(data["phase"], "source")
        self.assertEqual(data["processed_source_indices"], [0, 1, 2])
        self.assertEqual(len(data["source_clusters"]), 1)
        
        restored = NormalizeCheckpoint.from_dict(data)
        self.assertEqual(restored.phase, "source")
        self.assertEqual(restored.processed_source_indices, [0, 1, 2])
        self.assertEqual(restored.total_source_clusters, 5)
        self.assertEqual(restored.source_clusters[0]["canonical"], "JOURNEY")


class TestDomainExtractorParsing(unittest.TestCase):
    """Test domain extractor response parsing."""
    
    def test_parse_v1_response(self):
        """Test parsing basic v1 format response."""
        extractor = DomainExtractor.__new__(DomainExtractor)
        extractor.prompt_version = "v1"
        extractor.multi_level = False
        
        response = '''
        {
            "source_domain": "theft",
            "target_domain": "time",
            "mapping_explanation": "Time steals our moments",
            "confidence": 0.85
        }
        '''
        
        result = extractor._parse_response(response)
        
        self.assertEqual(result["source_domain"], "theft")
        self.assertEqual(result["target_domain"], "time")
        self.assertEqual(result["confidence"], 0.85)
    
    def test_parse_multilevel_response(self):
        """Test parsing v3 multilevel format response."""
        extractor = DomainExtractor.__new__(DomainExtractor)
        extractor.prompt_version = "v3-multilevel"
        extractor.multi_level = True
        
        response = '''
        {
            "reasoning": "This metaphor maps theft to time passing",
            "source_domain": {
                "specific": "criminal theft",
                "moderate": "theft",
                "abstract": "possession loss"
            },
            "target_domain": {
                "specific": "passing moments",
                "moderate": "time",
                "abstract": "abstract concept"
            },
            "mapping_explanation": "Time steals moments",
            "confidence": 0.9
        }
        '''
        
        result = extractor._parse_response(response)
        
        # Check multilevel parsing
        self.assertIn("source_domain_levels", result)
        self.assertIn("target_domain_levels", result)
        self.assertEqual(result["source_domain_levels"]["specific"], "criminal theft")
        self.assertEqual(result["target_domain_levels"]["moderate"], "time")
        
        # Check that primary domain is set to moderate
        self.assertEqual(result["source_domain"], "theft")
        self.assertEqual(result["target_domain"], "time")
    
    def test_parse_markdown_wrapped_json(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        extractor = DomainExtractor.__new__(DomainExtractor)
        extractor.prompt_version = "v1"
        extractor.multi_level = False
        
        response = '''```json
        {
            "source_domain": "journey",
            "target_domain": "life",
            "confidence": 0.8
        }
        ```'''
        
        result = extractor._parse_response(response)
        
        self.assertEqual(result["source_domain"], "journey")
        self.assertEqual(result["target_domain"], "life")


class TestDomainGraph(unittest.TestCase):
    """Test domain graph generation."""
    
    def test_graph_generation_basic(self):
        """Test basic graph generation from instances."""
        instances = [
            DomainMappedInstance(
                text="Time is a thief",
                type="metaphor",
                source_domain="theft",
                target_domain="time",
            ),
            DomainMappedInstance(
                text="Time steals our youth",
                type="metaphor",
                source_domain="theft",
                target_domain="time",
            ),
            DomainMappedInstance(
                text="Life is a journey",
                type="metaphor",
                source_domain="journey",
                target_domain="life",
            ),
        ]
        
        graph = DomainGraph(instances)
        data = graph.generate()
        
        self.assertEqual(len(data.nodes), 4)  # theft, time, journey, life
        self.assertEqual(len(data.edges), 2)  # theft->time, journey->life
        
        # Check edge weights
        theft_time_edge = next(e for e in data.edges if e.source == "theft" and e.target == "time")
        self.assertEqual(theft_time_edge.weight, 2)
        
        journey_life_edge = next(e for e in data.edges if e.source == "journey" and e.target == "life")
        self.assertEqual(journey_life_edge.weight, 1)
    
    def test_graph_with_normalization(self):
        """Test graph generation with normalization applied."""
        instances = [
            DomainMappedInstance(
                text="Life is a journey",
                type="metaphor",
                source_domain="journey",
                target_domain="life",
            ),
            DomainMappedInstance(
                text="Life is a path",
                type="metaphor",
                source_domain="path",
                target_domain="existence",
            ),
        ]
        
        normalization = NormalizationResult(
            source_mapping={"journey": "JOURNEY", "path": "JOURNEY"},
            target_mapping={"life": "EXISTENCE", "existence": "EXISTENCE"},
            source_clusters=[],
            target_clusters=[],
            config={},
        )
        
        graph = DomainGraph(instances, normalization)
        data = graph.generate()
        
        # Should have only 2 nodes after normalization
        self.assertEqual(len(data.nodes), 2)  # JOURNEY, EXISTENCE
        self.assertEqual(len(data.edges), 1)  # JOURNEY->EXISTENCE
        
        # Check edge weight is combined
        edge = data.edges[0]
        self.assertEqual(edge.weight, 2)
    
    def test_graph_to_dict(self):
        """Test graph serialization to dict."""
        instances = [
            DomainMappedInstance(
                text="Time is a thief",
                type="metaphor",
                source_domain="theft",
                target_domain="time",
            ),
        ]
        
        graph = DomainGraph(instances)
        graph.generate()
        data = graph.to_json()
        
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("stats", data)
        self.assertEqual(len(data["nodes"]), 2)
        self.assertEqual(len(data["edges"]), 1)


if __name__ == "__main__":
    unittest.main()
