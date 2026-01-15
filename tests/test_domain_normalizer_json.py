import unittest
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from qualitative_analysis.figurative.domains.normalizer import DomainNormalizer
from qualitative_analysis.figurative.domains.models import DomainCluster

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
