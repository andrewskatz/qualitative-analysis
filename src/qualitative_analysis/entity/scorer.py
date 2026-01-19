"""
Entity Scorer for multi-dimensional entity classification.

Scores entities along custom dimensions using LLM-based analysis with
uncertainty quantification through multiple runs.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from .models import (
    DimensionDefinition,
    DimensionSet,
    DimensionScore,
    SingleRunScore,
    EntityScore,
    EntityScoreResult,
)

logger = logging.getLogger(__name__)

# Path to prompt templates
PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT_FILE = PROMPT_DIR / "entity_scoring_v2.txt"


class EntityScorer:
    """
    Score entities along multiple dimensions using LLM analysis.

    Supports uncertainty quantification through multiple scoring runs,
    with aggregated statistics (mean, median, std_dev, CI, CV).

    Example:
        >>> from qualitative_analysis.entity.scorer import EntityScorer
        >>> from qualitative_analysis.entity.models import DimensionSet
        >>> from qualitative_analysis.core.providers import OllamaProvider
        >>>
        >>> scorer = EntityScorer()
        >>> llm = OllamaProvider(model_name="gpt-oss:120b")
        >>> dimensions = DimensionSet.sets_framework()
        >>>
        >>> result = await scorer.score_entity(
        ...     entity="climate change",
        ...     context="The community discussed climate change impacts on agriculture.",
        ...     dimensions=dimensions.dimensions,
        ...     llm_provider=llm,
        ...     num_runs=3
        ... )
        >>> print(f"Social score: {result.dimension_scores['social'].mean}")
    """

    def __init__(
        self,
        prompt_template: Optional[str] = None,
        temperature: float = 0.3,
    ):
        """
        Initialize the entity scorer.

        Args:
            prompt_template: Custom prompt template string. If None, uses default.
            temperature: LLM temperature for generation (default 0.3 for stability).
        """
        self.temperature = temperature

        if prompt_template:
            self._prompt_template = prompt_template
        else:
            self._prompt_template = self._load_default_prompt()

    def _load_default_prompt(self) -> str:
        """Load the default prompt template from file."""
        if DEFAULT_PROMPT_FILE.exists():
            return DEFAULT_PROMPT_FILE.read_text(encoding="utf-8")
        else:
            logger.warning("Default scoring prompt not found, using fallback")
            return self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """Return a minimal fallback prompt."""
        return """You are an expert analyst. Score the entity "{entity}" on the following dimensions.

Context: {context}

Dimensions:
{dimension_definitions}

Return JSON with this structure:
{{
  "entity": "{entity}",
  "initial_observations": "Your reasoning about the entity",
  "dimension_scores": [
    {{"dimension": "dimension_name", "score": 75, "justification": "Brief explanation"}}
  ]
}}

Score each dimension on {scale_range_description}. Return ONLY valid JSON."""

    async def score_entity(
        self,
        entity: str,
        context: str,
        dimensions: List[DimensionDefinition],
        llm_provider: Any,
        num_runs: int = 3,
        research_context: Optional[Dict[str, str]] = None,
        text_id: str = "",
    ) -> EntityScore:
        """
        Score an entity with multiple runs for uncertainty estimation.

        Args:
            entity: Entity to score.
            context: Context where entity appears.
            dimensions: List of dimension definitions.
            llm_provider: LLM provider instance with generate() method.
            num_runs: Number of scoring runs (default 3).
            research_context: Optional research context dict with:
                data_type, data_collection_context, research_question.
            text_id: Identifier for the source text.

        Returns:
            EntityScore with aggregated statistics across runs.

        Raises:
            ValueError: If all scoring runs fail.
        """
        start_time = time.time()

        logger.info(f"Scoring entity '{entity}' with {num_runs} runs")

        # Run scoring multiple times
        runs: List[SingleRunScore] = []
        for run_num in range(1, num_runs + 1):
            try:
                run_result = await self._score_single_run(
                    entity=entity,
                    context=context,
                    dimensions=dimensions,
                    llm_provider=llm_provider,
                    research_context=research_context,
                    run_number=run_num,
                )
                runs.append(run_result)
            except Exception as e:
                logger.error(f"Run {run_num} failed for entity '{entity}': {e}")
                # Continue with other runs

        if not runs:
            raise ValueError(f"All {num_runs} scoring runs failed for entity: {entity}")

        # Aggregate results
        dimension_scores = self._aggregate_runs(runs, dimensions)

        processing_time_ms = (time.time() - start_time) * 1000

        return EntityScore(
            entity=entity,
            text_id=text_id,
            context=context,
            dimension_scores=dimension_scores,
            runs=runs,
            num_runs=len(runs),
            processing_time_ms=round(processing_time_ms, 2),
        )

    async def score_entities(
        self,
        entities: List[Dict[str, Any]],
        dimensions: List[DimensionDefinition],
        llm_provider: Any,
        num_runs: int = 3,
        research_context: Optional[Dict[str, str]] = None,
        on_progress: Optional[callable] = None,
    ) -> EntityScoreResult:
        """
        Score multiple entities.

        Args:
            entities: List of dicts with 'entity', 'context', and optionally 'text_id'.
            dimensions: List of dimension definitions.
            llm_provider: LLM provider instance.
            num_runs: Number of scoring runs per entity.
            research_context: Optional research context.
            on_progress: Optional callback(current, total) for progress.

        Returns:
            EntityScoreResult with all scores and statistics.
        """
        if not entities:
            return EntityScoreResult(
                scores=[],
                dimensions=dimensions,
                config={"num_runs": num_runs},
                statistics={},
            )

        logger.info(f"Scoring {len(entities)} entities")

        scores = []
        errors = 0

        for i, ent_data in enumerate(entities):
            entity = ent_data.get("entity", "")
            context = ent_data.get("context", "")
            text_id = ent_data.get("text_id", "")

            if not entity:
                logger.warning(f"Skipping empty entity at index {i}")
                continue

            try:
                score = await self.score_entity(
                    entity=entity,
                    context=context,
                    dimensions=dimensions,
                    llm_provider=llm_provider,
                    num_runs=num_runs,
                    research_context=research_context,
                    text_id=text_id,
                )
                scores.append(score)
            except Exception as e:
                logger.error(f"Failed to score entity '{entity}': {e}")
                errors += 1

            if on_progress:
                on_progress(i + 1, len(entities))

        result = EntityScoreResult(
            scores=scores,
            dimensions=dimensions,
            config={
                "num_runs": num_runs,
                "temperature": self.temperature,
                "errors": errors,
            },
        )
        result.compute_statistics()

        logger.info(f"Scored {len(scores)} entities ({errors} errors)")

        return result

    async def _score_single_run(
        self,
        entity: str,
        context: str,
        dimensions: List[DimensionDefinition],
        llm_provider: Any,
        research_context: Optional[Dict[str, str]] = None,
        run_number: int = 1,
    ) -> SingleRunScore:
        """
        Score an entity with a single LLM run.

        Args:
            entity: Entity to score.
            context: Context where entity appears.
            dimensions: List of dimension definitions.
            llm_provider: LLM provider instance.
            research_context: Optional research context.
            run_number: Run number (for logging).

        Returns:
            SingleRunScore with dimension scores.
        """
        start_time = time.time()

        # Build prompt
        prompt = self._build_prompt(
            entity=entity,
            context=context,
            dimensions=dimensions,
            research_context=research_context,
        )

        logger.debug(f"Scoring '{entity}' (run {run_number})")

        # Call LLM
        response = await llm_provider.generate(
            prompt=prompt,
            temperature=self.temperature,
        )

        # Parse response
        parsed = self._parse_response(response, dimensions)

        processing_time_ms = (time.time() - start_time) * 1000

        return SingleRunScore(
            run_number=run_number,
            dimension_scores=parsed["dimension_scores"],
            initial_observations=parsed.get("initial_observations", ""),
            raw_response=response,
            processing_time_ms=round(processing_time_ms, 2),
        )

    def _build_prompt(
        self,
        entity: str,
        context: str,
        dimensions: List[DimensionDefinition],
        research_context: Optional[Dict[str, str]] = None,
    ) -> str:
        """Build the scoring prompt."""
        # Format dimension definitions
        dim_text = self._format_dimensions(dimensions)

        # Get scale description
        scale_desc = self._get_scale_description(dimensions)

        # Format research context
        research_text = self._format_research_context(research_context)

        return self._prompt_template.format(
            entity=entity,
            context=context,
            dimension_definitions=dim_text,
            scale_range_description=scale_desc,
            research_context=research_text,
        )

    def _format_dimensions(self, dimensions: List[DimensionDefinition]) -> str:
        """Format dimension definitions for the prompt."""
        lines = []

        for dim in dimensions:
            lines.append(f"**{dim.name.upper()}**")
            lines.append(f"Definition: {dim.description}")
            lines.append(f"Scale: {dim.scale_min}-{dim.scale_max}")

            if dim.min_anchor:
                lines.append(f"Low anchor ({dim.scale_min}): {dim.min_anchor}")
            if dim.max_anchor:
                lines.append(f"High anchor ({dim.scale_max}): {dim.max_anchor}")

            lines.append("")

        return "\n".join(lines)

    def _get_scale_description(self, dimensions: List[DimensionDefinition]) -> str:
        """Get scale range description."""
        scales = set((d.scale_min, d.scale_max) for d in dimensions)

        if len(scales) == 1:
            scale_min, scale_max = list(scales)[0]
            return f"a {scale_min}-{scale_max} scale"
        else:
            return "the specified scale for each dimension"

    def _format_research_context(
        self, research_context: Optional[Dict[str, str]]
    ) -> str:
        """Format research context for the prompt."""
        if not research_context or not isinstance(research_context, dict):
            return ""

        parts = []

        if research_context.get("data_type"):
            parts.append(f"Data Type: {research_context['data_type']}")
        if research_context.get("data_collection_context"):
            parts.append(f"Data Collection Context: {research_context['data_collection_context']}")
        if research_context.get("research_question"):
            parts.append(f"Research Question: {research_context['research_question']}")

        if not parts:
            return ""

        return (
            "\n\nRESEARCH CONTEXT\n\n"
            "The entity was extracted from data with the following research context:\n\n"
            + "\n".join(parts)
            + "\n\nConsider this context when scoring the entity."
        )

    def _parse_response(
        self, response: str, dimensions: List[DimensionDefinition]
    ) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        clean_response = response.strip()

        # Strip markdown code blocks
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()

        # Try to parse JSON
        try:
            data = json.loads(clean_response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            json_match = re.search(r'\{.*"dimension_scores".*\}', clean_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                raise ValueError(f"No valid JSON found in response: {clean_response[:200]}")

        # Extract dimension scores
        if "dimension_scores" not in data:
            raise ValueError(f"Response missing 'dimension_scores'. Keys: {list(data.keys())}")

        raw_scores = data["dimension_scores"]
        if not isinstance(raw_scores, list):
            raise ValueError(f"'dimension_scores' must be a list, got {type(raw_scores).__name__}")

        expected_dims = {d.name.lower() for d in dimensions}
        dimension_scores = {}

        for score_data in raw_scores:
            if not isinstance(score_data, dict):
                continue

            dim_name = score_data.get("dimension", "").lower()
            score = score_data.get("score")
            justification = score_data.get("justification", "")

            if dim_name not in expected_dims:
                logger.warning(f"Unexpected dimension: {dim_name}")
                continue

            if not isinstance(score, (int, float)):
                logger.warning(f"Invalid score for {dim_name}: {score}")
                continue

            dimension_scores[dim_name] = {
                "score": int(score),
                "justification": justification,
            }

        # Check for missing dimensions
        missing = expected_dims - set(dimension_scores.keys())
        if missing:
            raise ValueError(f"Missing dimensions: {missing}")

        return {
            "dimension_scores": dimension_scores,
            "initial_observations": data.get("initial_observations", ""),
        }

    def _aggregate_runs(
        self,
        runs: List[SingleRunScore],
        dimensions: List[DimensionDefinition],
    ) -> Dict[str, DimensionScore]:
        """Aggregate scores from multiple runs."""
        aggregated = {}

        for dim in dimensions:
            dim_name = dim.name.lower()

            # Collect scores and justifications
            scores = []
            justifications = []

            for run in runs:
                if dim_name in run.dimension_scores:
                    scores.append(run.dimension_scores[dim_name]["score"])
                    justifications.append(run.dimension_scores[dim_name].get("justification", ""))

            if not scores:
                continue

            # Select justification closest to mean
            mean_score = sum(scores) / len(scores)
            if justifications:
                non_empty = [(i, j) for i, j in enumerate(justifications) if j.strip()]
                if non_empty:
                    closest_idx = min(non_empty, key=lambda x: abs(scores[x[0]] - mean_score))[0]
                    combined_justification = justifications[closest_idx]
                else:
                    combined_justification = ""
            else:
                combined_justification = ""

            aggregated[dim_name] = DimensionScore.from_scores(
                dimension=dim_name,
                scores=scores,
                justification=combined_justification,
            )

        return aggregated

    def save(self, result: EntityScoreResult, path: Union[str, Path]) -> None:
        """Save entity score result to JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved entity scores to {path}")

    def load(self, path: Union[str, Path]) -> EntityScoreResult:
        """Load entity score result from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct the result
        scores = []
        for s_data in data.get("scores", []):
            dim_scores = {}
            for dim_name in data.get("dimensions", []):
                dim_key = dim_name.get("name", "").lower() if isinstance(dim_name, dict) else dim_name.lower()
                if f"{dim_key}_mean" in s_data:
                    dim_scores[dim_key] = DimensionScore(
                        dimension=dim_key,
                        mean=s_data.get(f"{dim_key}_mean", 0),
                        median=s_data.get(f"{dim_key}_median", 0),
                        mode=s_data.get(f"{dim_key}_mode", 0),
                        std_dev=s_data.get(f"{dim_key}_std", 0),
                        confidence_interval_95=(
                            s_data.get(f"{dim_key}_ci_low", 0),
                            s_data.get(f"{dim_key}_ci_high", 0),
                        ),
                        coefficient_of_variation=s_data.get(f"{dim_key}_cv", 0),
                        scores=[],
                        justification=s_data.get(f"{dim_key}_justification", ""),
                    )

            scores.append(EntityScore(
                entity=s_data.get("entity", ""),
                text_id=s_data.get("text_id", ""),
                context=s_data.get("context", ""),
                dimension_scores=dim_scores,
                num_runs=s_data.get("num_runs", 1),
                processing_time_ms=s_data.get("processing_time_ms", 0),
            ))

        dimensions = [
            DimensionDefinition.from_dict(d) for d in data.get("dimensions", [])
        ]

        return EntityScoreResult(
            scores=scores,
            dimensions=dimensions,
            config=data.get("config", {}),
            statistics=data.get("statistics", {}),
        )


async def score_entities(
    entities: List[Dict[str, Any]],
    dimensions: Union[List[DimensionDefinition], DimensionSet],
    llm_provider: Any,
    num_runs: int = 3,
    temperature: float = 0.3,
    research_context: Optional[Dict[str, str]] = None,
) -> EntityScoreResult:
    """
    Convenience function to score multiple entities.

    Args:
        entities: List of dicts with 'entity', 'context', and optionally 'text_id'.
        dimensions: Dimension definitions (list or DimensionSet).
        llm_provider: LLM provider instance.
        num_runs: Number of scoring runs per entity.
        temperature: LLM temperature.
        research_context: Optional research context.

    Returns:
        EntityScoreResult with all scores and statistics.
    """
    if isinstance(dimensions, DimensionSet):
        dim_list = dimensions.dimensions
    else:
        dim_list = dimensions

    scorer = EntityScorer(temperature=temperature)
    return await scorer.score_entities(
        entities=entities,
        dimensions=dim_list,
        llm_provider=llm_provider,
        num_runs=num_runs,
        research_context=research_context,
    )
