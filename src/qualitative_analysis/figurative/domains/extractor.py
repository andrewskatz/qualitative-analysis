"""
Domain extractor for figurative language analysis.

Extracts source and target conceptual domains from figurative instances.
"""

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from ...core.llm import BaseLLMProvider
from ...core.providers import OllamaProvider
from .models import DomainMappedInstance, Checkpoint

logger = logging.getLogger(__name__)


def _prompts_dir() -> Path:
    """Get the prompts directory."""
    return Path(__file__).parent / "prompts"


def _load_prompt(version: str) -> tuple:
    """
    Load prompt templates by version.
    
    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    prompts_dir = _prompts_dir()
    
    if version == "v1":
        # v1 uses combined prompt (legacy)
        filepath = prompts_dir / "domain_extraction_v1.txt"
        if not filepath.exists():
            raise FileNotFoundError(f"Prompt file not found: {filepath}")
        return None, filepath.read_text(encoding="utf-8")
    
    elif version in ("v3", "v3-multilevel"):
        # v3 uses separate system and user prompts
        system_path = prompts_dir / "domain_extraction_v3_system.txt"
        user_path = prompts_dir / "domain_extraction_v3_user.txt"
        
        if not system_path.exists() or not user_path.exists():
            # Fall back to combined prompt if split files don't exist
            combined_path = prompts_dir / "domain_extraction_v3_multilevel.txt"
            if combined_path.exists():
                logger.warning("Using legacy combined prompt, split prompts not found")
                return None, combined_path.read_text(encoding="utf-8")
            raise FileNotFoundError(f"Prompt files not found: {system_path} or {user_path}")
        
        system_prompt = system_path.read_text(encoding="utf-8")
        user_prompt = user_path.read_text(encoding="utf-8")
        return system_prompt, user_prompt
    
    else:
        # Default to v3 multilevel
        logger.warning(f"Unknown prompt version '{version}', using v3-multilevel")
        return _load_prompt("v3-multilevel")


class DomainExtractor:
    """
    Extracts source and target domains from figurative language instances.
    
    Supports both single-level (v1) and multi-level (v3) domain extraction.
    """
    
    def __init__(
        self,
        model_name: str = "qwen3:30b-a3b-instruct-2507-q4_K_M",
        provider: str = "ollama",
        provider_config: Optional[Dict[str, Any]] = None,
        prompt_version: str = "v3-multilevel",
        multi_level: bool = True,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        """
        Initialize the domain extractor.
        
        Args:
            model_name: LLM model to use
            provider: LLM provider ("ollama")
            provider_config: Optional provider configuration
            prompt_version: Prompt version ("v1", "v3-multilevel")
            multi_level: Whether to extract domains at multiple abstraction levels
            llm_provider: Optional pre-configured LLM provider (for testing)
        """
        provider_config = provider_config or {}
        
        # Setup LLM
        if llm_provider:
            self.llm = llm_provider
        elif provider == "ollama":
            self.llm = OllamaProvider(model_name, **provider_config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.prompt_version = prompt_version
        self.multi_level = multi_level
        
        # Load prompt templates (system + user)
        if multi_level:
            self.system_prompt, self.user_prompt_template = _load_prompt("v3-multilevel")
        else:
            self.system_prompt, self.user_prompt_template = _load_prompt(prompt_version)
    
    async def extract(
        self,
        instances: List[Dict[str, Any]],
        window_texts: Optional[Dict[int, str]] = None,
        checkpoint_path: Optional[Path] = None,
        checkpoint_interval: int = 50,
        on_progress: Optional[callable] = None,
    ) -> List[DomainMappedInstance]:
        """
        Extract domains for a list of figurative instances.
        
        Args:
            instances: List of instance dicts with at minimum:
                - text: str (the figurative phrase)
                - type: str (metaphor, analogy, etc.)
                Optional: confidence, explanation, window_index, window_text, text_id
            window_texts: Optional map from window_index to window text
            checkpoint_path: Optional path to save checkpoints
            checkpoint_interval: How often to save checkpoints
            on_progress: Optional callback(index, total) for progress reporting
            
        Returns:
            List of DomainMappedInstance objects
        """
        window_texts = window_texts or {}
        results: List[DomainMappedInstance] = []
        
        # Load checkpoint if exists
        start_index = 0
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = self._load_checkpoint(checkpoint_path)
            if checkpoint:
                start_index = len(checkpoint.processed_indices)
                results = [self._dict_to_instance(r) for r in checkpoint.partial_results]
                logger.info(f"Resuming from checkpoint at index {start_index}")
        
        total = len(instances)
        for i, instance_data in enumerate(instances):
            if i < start_index:
                continue
            
            try:
                result = await self._analyze_single(instance_data, window_texts)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing instance {i}: {e}")
                # Create a partial result with error
                results.append(DomainMappedInstance(
                    text=instance_data.get("text", ""),
                    type=instance_data.get("type", "unknown"),
                    confidence=instance_data.get("confidence", 0.0),
                    explanation=instance_data.get("explanation", ""),
                    window_index=instance_data.get("window_index", 0),
                    text_id=instance_data.get("text_id", ""),
                    raw_response={"error": str(e)},
                ))
            
            # Progress callback
            if on_progress:
                on_progress(i + 1, total)
            
            # Save checkpoint
            if checkpoint_path and (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(
                    checkpoint_path,
                    results,
                    instances,
                    i + 1,
                )
        
        return results
    
    async def extract_from_csv(
        self,
        input_path: Path,
        text_col: str = "instance_text",
        type_col: str = "type",
        window_col: Optional[str] = "window_text",
        text_id_col: Optional[str] = "text_id",
        checkpoint_path: Optional[Path] = None,
        checkpoint_interval: int = 50,
        on_progress: Optional[callable] = None,
    ) -> List[DomainMappedInstance]:
        """
        Extract domains from a CSV file of figurative instances.
        
        Args:
            input_path: Path to input CSV
            text_col: Column name for figurative text
            type_col: Column name for figurative type
            window_col: Optional column for window context
            text_id_col: Optional column for text ID
            checkpoint_path: Optional path to save checkpoints
            checkpoint_interval: How often to save checkpoints
            on_progress: Optional callback(index, total) for progress
            
        Returns:
            List of DomainMappedInstance objects
        """
        instances = []
        
        with open(input_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                instance = {
                    "text": row.get(text_col, ""),
                    "type": row.get(type_col, "unknown"),
                }
                if window_col and window_col in row:
                    instance["window_text"] = row.get(window_col, "")
                if text_id_col and text_id_col in row:
                    instance["text_id"] = row.get(text_id_col, "")
                if "confidence" in row:
                    try:
                        instance["confidence"] = float(row["confidence"])
                    except ValueError:
                        pass
                if "explanation" in row:
                    instance["explanation"] = row["explanation"]
                if "window_index" in row:
                    try:
                        instance["window_index"] = int(row["window_index"])
                    except ValueError:
                        pass
                
                instances.append(instance)
        
        return await self.extract(
            instances,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            on_progress=on_progress,
        )
    
    async def _analyze_single(
        self,
        instance_data: Dict[str, Any],
        window_texts: Dict[int, str],
    ) -> DomainMappedInstance:
        """Analyze a single figurative instance."""
        figurative_text = instance_data.get("text", "")
        figurative_type = instance_data.get("type", "metaphor")
        window_index = instance_data.get("window_index", 0)
        
        # Get window text from either the instance or the lookup
        window_text = instance_data.get("window_text", "")
        if not window_text and window_index in window_texts:
            window_text = window_texts[window_index]
        if not window_text:
            window_text = "No additional context available."
        
        # Format user prompt
        user_prompt = self.user_prompt_template.format(
            figurative_text=figurative_text,
            figurative_type=figurative_type,
            window_text=window_text,
        )
        
        logger.debug(f"Analyzing: {figurative_text[:50]}...")
        
        # Call LLM with separate system and user prompts
        response = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
            temperature=0.3,
        )
        
        # Parse response
        parsed = self._parse_response(response)
        
        # Build result
        result = DomainMappedInstance(
            text=figurative_text,
            type=figurative_type,
            confidence=instance_data.get("confidence", 0.0),
            explanation=instance_data.get("explanation", ""),
            window_index=window_index,
            text_id=instance_data.get("text_id", ""),
            window_text=window_text if window_text != "No additional context available." else "",
            source_domain=parsed.get("source_domain", ""),
            target_domain=parsed.get("target_domain", ""),
            mapping_explanation=parsed.get("mapping_explanation", ""),
            domain_confidence=parsed.get("confidence", 0.0),
            source_domain_levels=parsed.get("source_domain_levels", {}),
            target_domain_levels=parsed.get("target_domain_levels", {}),
            raw_response=parsed,
        )
        
        return result
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        clean_response = response.strip()
        
        # Strip markdown code blocks
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        
        clean_response = clean_response.strip()
        
        try:
            result = json.loads(clean_response)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError(f"No valid JSON found in response: {clean_response[:200]}")
        
        # Normalize confidence
        if "confidence" in result:
            conf = result["confidence"]
            if isinstance(conf, str):
                conf_mapping = {"high": 0.9, "medium": 0.6, "low": 0.3}
                result["confidence"] = conf_mapping.get(conf.lower(), 0.5)
            else:
                try:
                    result["confidence"] = max(0.0, min(1.0, float(conf)))
                except (ValueError, TypeError):
                    result["confidence"] = 0.5
        
        # Handle multi-level format
        if isinstance(result.get("source_domain"), dict):
            source_levels = result["source_domain"]
            result["source_domain_levels"] = source_levels
            # Use moderate as primary for backward compatibility
            result["source_domain"] = source_levels.get("moderate", "")
        
        if isinstance(result.get("target_domain"), dict):
            target_levels = result["target_domain"]
            result["target_domain_levels"] = target_levels
            result["target_domain"] = target_levels.get("moderate", "")
        
        return result
    
    def _save_checkpoint(
        self,
        path: Path,
        results: List[DomainMappedInstance],
        instances: List[Dict[str, Any]],
        processed_count: int,
    ) -> None:
        """Save checkpoint to file."""
        checkpoint = Checkpoint(
            processed_indices=list(range(processed_count)),
            partial_results=[self._instance_to_dict(r) for r in results],
            config={
                "prompt_version": self.prompt_version,
                "multi_level": self.multi_level,
            },
            timestamp=datetime.now().isoformat(),
            total_items=len(instances),
        )
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        
        logger.info(f"Saved checkpoint at {processed_count}/{len(instances)}")
    
    def _load_checkpoint(self, path: Path) -> Optional[Checkpoint]:
        """Load checkpoint from file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Checkpoint.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def _instance_to_dict(self, instance: DomainMappedInstance) -> Dict[str, Any]:
        """Convert instance to serializable dict."""
        return {
            "text": instance.text,
            "type": instance.type,
            "confidence": instance.confidence,
            "explanation": instance.explanation,
            "window_index": instance.window_index,
            "text_id": instance.text_id,
            "window_text": instance.window_text,
            "source_domain": instance.source_domain,
            "target_domain": instance.target_domain,
            "mapping_explanation": instance.mapping_explanation,
            "domain_confidence": instance.domain_confidence,
            "source_domain_levels": instance.source_domain_levels,
            "target_domain_levels": instance.target_domain_levels,
        }
    
    def _dict_to_instance(self, data: Dict[str, Any]) -> DomainMappedInstance:
        """Convert dict back to instance."""
        return DomainMappedInstance(
            text=data.get("text", ""),
            type=data.get("type", "unknown"),
            confidence=data.get("confidence", 0.0),
            explanation=data.get("explanation", ""),
            window_index=data.get("window_index", 0),
            text_id=data.get("text_id", ""),
            window_text=data.get("window_text", ""),
            source_domain=data.get("source_domain", ""),
            target_domain=data.get("target_domain", ""),
            mapping_explanation=data.get("mapping_explanation", ""),
            domain_confidence=data.get("domain_confidence", 0.0),
            source_domain_levels=data.get("source_domain_levels", {}),
            target_domain_levels=data.get("target_domain_levels", {}),
        )
