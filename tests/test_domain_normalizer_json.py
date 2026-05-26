import unittest
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from qualitative_analysis.figurative.domains.normalizer import DomainNormalizer
from qualitative_analysis.figurative.domains.models import DomainCluster, DomainMappedInstance

class TestDomainNormalizerJSON(unittest.IsolatedAsyncioTestCase):
    async def test_generate_canonical_labels_llm_json_success(self):
        """Test successful JSON parsing from LLM response."""
        normalizer = DomainNormalizer()
        
        # Create a cluster
        cluster = DomainCluster(
            canonical="",
            members=["valuable container", "valuable objects"],
            avg_similarity=0.9
        )
        
        # Mock OllamaProvider
        with patch("qualitative_analysis.core.providers.OllamaProvider") as MockProvider:
            mock_llm = MockProvider.return_value
            mock_llm.generate = AsyncMock(return_value="""
            ```json
            {
                "reasoning": "Both terms refer to objects of value.",
                "canonical_label": "VALUABLES"
            }
            ```
            """)
            
            # Run method
            clusters = await normalizer._generate_canonical_labels_llm(
                [cluster], 
                model="test-model", 
                provider="ollama", 
                config={}
            )
            
            self.assertEqual(clusters[0].canonical, "VALUABLES")
            
    async def test_generate_canonical_labels_llm_json_malformed(self):
        """Test fallback when JSON is malformed."""
        normalizer = DomainNormalizer()
        
        # Create a cluster
        cluster = DomainCluster(
            canonical="",
            members=["foo", "bar"],
            avg_similarity=0.9
        )
        
        # Mock OllamaProvider to return garbage
        with patch("qualitative_analysis.core.providers.OllamaProvider") as MockProvider:
            mock_llm = MockProvider.return_value
            mock_llm.generate = AsyncMock(return_value="Not a JSON object")
            
            # Run method
            clusters = await normalizer._generate_canonical_labels_llm(
                [cluster], 
                model="test-model", 
                provider="ollama", 
                config={}
            )
            
            # Should fallback to first member
            self.assertEqual(clusters[0].canonical, "FOO")

    async def test_normalize_async_llm_is_safe_in_active_event_loop(self):
        """LLM normalization should work via the async entry point inside a running loop."""
        normalizer = DomainNormalizer()
        instances = [
            DomainMappedInstance(text="Life is a journey", type="metaphor", source_domain="journey", target_domain="life")
        ]
        source_cluster = DomainCluster(canonical="JOURNEY", members=["journey"], avg_similarity=1.0)
        target_cluster = DomainCluster(canonical="LIFE", members=["life"], avg_similarity=1.0)

        with patch.object(normalizer, "_get_embedding_model", return_value=MagicMock()), \
             patch.object(normalizer, "_cluster_domains", side_effect=[
                 [DomainCluster(canonical="", members=["journey"], avg_similarity=1.0)],
                 [DomainCluster(canonical="", members=["life"], avg_similarity=1.0)],
             ]), \
             patch.object(
                 normalizer,
                 "_generate_canonical_labels_llm_with_checkpoint",
                 new=AsyncMock(return_value=([source_cluster], [target_cluster])),
             ) as mock_generate:
            result = await normalizer.normalize_async(
                instances,
                conservativeness="custom",
                similarity_threshold=0.8,
                canonical_method="llm",
                llm_model="test-model",
            )

        mock_generate.assert_awaited_once()
        self.assertEqual(result.source_mapping["journey"], "JOURNEY")
        self.assertEqual(result.target_mapping["life"], "LIFE")
        self.assertEqual(result.source_clusters[0].count, 1)
        self.assertEqual(result.target_clusters[0].count, 1)

    async def test_sync_normalize_llm_raises_clear_error_in_active_event_loop(self):
        """Sync LLM normalization should direct async callers to normalize_async."""
        normalizer = DomainNormalizer()
        instances = [
            DomainMappedInstance(text="Life is a journey", type="metaphor", source_domain="journey", target_domain="life")
        ]

        with self.assertRaisesRegex(RuntimeError, "normalize_async"):
            normalizer.normalize(
                instances,
                canonical_method="llm",
                llm_model="test-model",
            )
