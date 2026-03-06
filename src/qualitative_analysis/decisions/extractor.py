"""
Decision/factor extraction using LLM analysis.

Supports three strategies:
- one_pass: Combined decisions+factors in a single LLM call per window
- semantic_two_pass: RAG-based extraction (decisions first, then semantic search
  to find relevant windows, then factors from relevant windows only)
- context_aware: Summarize-then-extract with running context accumulator,
  followed by per-decision factor probing across all windows
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.llm import BaseLLMProvider
from ..core.text import SlidingWindowProcessor
from .models import (
    Decision,
    DecisionConfig,
    DecisionExtractionResult,
    Factor,
    WindowSummary,
)
from .prompts.loader import load_prompt
from .validator import DecisionValidator

logger = logging.getLogger(__name__)


class DecisionExtractor:
    """
    High-level interface for extracting decisions and factors from text.

    Strategies:
    - one_pass: Extract decisions+factors in a single LLM call per window.
    - semantic_two_pass: Extract decisions from all windows, use embeddings
      to find relevant windows, then extract factors per decision.
    - context_aware: Summarize-then-extract with running context, followed
      by per-decision factor probing across all windows.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        strategy: str = "semantic_two_pass",
        config: Optional[DecisionConfig] = None,
        # Windowing
        window_size: int = 6,
        stride: int = 4,
        chunk_unit: str = "sentences",
        tokenizer_name: str = "cl100k_base",
        # Prompt versions
        decisions_prompt_version: int = 5,
        factors_prompt_version: int = 5,
        combined_prompt_version: int = 2,
        summary_prompt_version: int = 1,
        # Custom prompt file overrides
        decisions_prompt_path: Optional[str] = None,
        factors_prompt_path: Optional[str] = None,
        combined_prompt_path: Optional[str] = None,
        summary_prompt_path: Optional[str] = None,
        # Embedding model (for semantic_two_pass)
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        if strategy not in {"one_pass", "semantic_two_pass", "context_aware"}:
            raise ValueError(f"Unsupported strategy: {strategy}")

        self.llm = llm_provider
        self.strategy = strategy
        self.config = config or DecisionConfig(strategy=strategy)

        self.text_processor = SlidingWindowProcessor(
            window_size=window_size,
            stride=stride,
            chunk_unit=chunk_unit,
            tokenizer_name=tokenizer_name,
        )

        # Validator
        self.validator = (
            DecisionValidator() if self.config.validate_decisions else None
        )

        # Load prompts (custom path overrides version-based loading)
        self.decisions_prompt = load_prompt(
            "decisions_only",
            decisions_prompt_version,
            custom_path=decisions_prompt_path,
        )
        self.factors_prompt = load_prompt(
            "factors_for_decision",
            factors_prompt_version,
            custom_path=factors_prompt_path,
        )
        self.combined_prompt = load_prompt(
            "decision_extraction",
            combined_prompt_version,
            custom_path=combined_prompt_path,
        )
        self.summary_prompt = load_prompt(
            "window_summary",
            summary_prompt_version,
            custom_path=summary_prompt_path,
        )

        # Embedding service (lazy init)
        self._embedding_service = None
        self._embedding_model_name = embedding_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(
        self,
        text: str,
        text_id: str = "",
    ) -> DecisionExtractionResult:
        """
        Extract decisions and factors from text.

        Args:
            text: Input text to analyze.
            text_id: Identifier for the source text.

        Returns:
            DecisionExtractionResult with decisions, factors, and metadata.
        """
        if not text or not text.strip():
            return DecisionExtractionResult(
                text_id=text_id,
                metadata={"strategy": self.strategy, "window_count": 0},
            )

        if self.strategy == "one_pass":
            return await self._extract_one_pass(text, text_id)
        elif self.strategy == "semantic_two_pass":
            return await self._extract_semantic_two_pass(text, text_id)
        elif self.strategy == "context_aware":
            return await self._extract_context_aware(text, text_id)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    # ------------------------------------------------------------------
    # Strategy: one_pass
    # ------------------------------------------------------------------

    async def _extract_one_pass(
        self, text: str, text_id: str
    ) -> DecisionExtractionResult:
        """Single-pass: combined decisions+factors prompt per window."""
        start_time = time.time()
        windows = self.text_processor.process(text)

        all_decisions: List[Decision] = []
        all_factors: List[Factor] = []
        seen_decisions: Dict[str, Decision] = {}

        for i, window_text in enumerate(windows):
            prompt = self.combined_prompt.replace("{text}", window_text)

            try:
                response = await self.llm.generate(
                    prompt=prompt,
                    temperature=self.config.temperature,
                )
                data = self._parse_json_response(response)

                # Parse decisions
                raw_decisions = data.get("decisions", [])
                for rd in raw_decisions:
                    d_text = rd if isinstance(rd, str) else rd.get("decision", "")
                    d_text = d_text.strip()
                    if not d_text:
                        continue
                    if self.config.normalize_decisions:
                        d_text = DecisionValidator.strip_causal_clauses(d_text)
                    key = d_text.lower()
                    if key not in seen_decisions:
                        dec = Decision(
                            text=d_text,
                            window_indices=[i],
                            text_ids=[text_id] if text_id else [],
                        )
                        seen_decisions[key] = dec
                        all_decisions.append(dec)
                    else:
                        if i not in seen_decisions[key].window_indices:
                            seen_decisions[key].window_indices.append(i)

                # Parse factors
                raw_factors = data.get("factors", [])
                for rf in raw_factors:
                    if isinstance(rf, dict):
                        f_text = rf.get("factor", "")
                        f_decision = rf.get("decision", "")
                        f_polarity = rf.get("polarity", "neutral")
                    else:
                        continue
                    f_text = f_text.strip()
                    f_decision = f_decision.strip()
                    if not f_text:
                        continue
                    factor = Factor(
                        text=f_text,
                        decision_text=f_decision,
                        polarity=f_polarity,
                        window_index=i,
                        text_id=text_id,
                    )
                    all_factors.append(factor)

            except Exception as e:
                logger.warning(f"Failed to extract from window {i}: {e}")
                continue

        # Validate decisions
        if self.validator:
            all_decisions = [
                d
                for d in all_decisions
                if self.validator.is_valid_decision(d.text)
            ]

        # Link factors to decisions
        for d in all_decisions:
            d.factors = [
                f for f in all_factors if f.decision_text.lower() == d.text.lower()
            ]

        return DecisionExtractionResult(
            text_id=text_id,
            decisions=all_decisions,
            factors=all_factors,
            window_count=len(windows),
            metadata={
                "strategy": "one_pass",
                "window_count": len(windows),
                "decisions_found": len(all_decisions),
                "factors_found": len(all_factors),
                "extraction_time": round(time.time() - start_time, 2),
            },
        )

    # ------------------------------------------------------------------
    # Strategy: semantic_two_pass
    # ------------------------------------------------------------------

    async def _extract_semantic_two_pass(
        self, text: str, text_id: str
    ) -> DecisionExtractionResult:
        """
        RAG-based two-pass extraction.

        1. Chunk into windows
        2. Generate window summaries
        3. Extract decisions from ALL windows (pass 1)
        4. Embed decisions + summaries, find top-K relevant windows
        5. Extract factors from relevant windows only (pass 2)
        6. Deduplicate
        """
        start_time = time.time()
        windows = self.text_processor.process(text)

        if not windows:
            return DecisionExtractionResult(
                text_id=text_id,
                metadata={"strategy": "semantic_two_pass", "window_count": 0},
            )

        # Step 1: Generate window summaries
        logger.info(f"Generating summaries for {len(windows)} windows...")
        summaries = await self._generate_window_summaries(windows)

        # Step 2: Extract decisions from all windows (pass 1)
        logger.info("Pass 1: Extracting decisions from all windows...")
        raw_decisions: List[str] = []
        decision_window_map: Dict[str, List[int]] = {}

        for i, window_text in enumerate(windows):
            window_decisions = await self._extract_decisions_from_window(
                window_text
            )
            for d_text in window_decisions:
                if self.config.normalize_decisions:
                    d_text = DecisionValidator.strip_causal_clauses(d_text)
                raw_decisions.append(d_text)
                decision_window_map.setdefault(d_text.lower(), []).append(i)

        if not raw_decisions:
            return DecisionExtractionResult(
                text_id=text_id,
                window_count=len(windows),
                metadata={
                    "strategy": "semantic_two_pass",
                    "window_count": len(windows),
                    "decisions_found": 0,
                },
            )

        # Deduplicate decisions
        unique_decisions = self._deduplicate_decisions(raw_decisions)

        # Validate decisions
        if self.validator:
            unique_decisions = self.validator.validate_decisions(unique_decisions)

        if not unique_decisions:
            return DecisionExtractionResult(
                text_id=text_id,
                window_count=len(windows),
                metadata={
                    "strategy": "semantic_two_pass",
                    "window_count": len(windows),
                    "decisions_found": 0,
                },
            )

        logger.info(f"Found {len(unique_decisions)} unique decisions")

        # Step 3: Semantic search — find relevant windows per decision
        logger.info("Semantic search: finding relevant windows...")
        semantic_start = time.time()
        relevant_map = self._find_relevant_windows(unique_decisions, summaries)
        semantic_time = time.time() - semantic_start

        # Step 4: Extract factors from relevant windows (pass 2)
        logger.info("Pass 2: Extracting factors from relevant windows...")
        all_factors: List[Factor] = []
        all_decisions: List[Decision] = []

        for d_text in unique_decisions:
            window_data = relevant_map.get(d_text, {})
            window_indices = window_data.get("window_indices", [])

            decision_factors: List[Factor] = []

            for w_idx in window_indices:
                if w_idx >= len(windows):
                    continue
                factors = await self._extract_factors_for_decision(
                    d_text, windows[w_idx], window_index=w_idx, text_id=text_id
                )
                decision_factors.extend(factors)

            # Deduplicate factors per decision
            decision_factors = self._deduplicate_factors(decision_factors)

            dec = Decision(
                text=d_text,
                factors=decision_factors,
                window_indices=decision_window_map.get(d_text.lower(), []),
                text_ids=[text_id] if text_id else [],
            )
            all_decisions.append(dec)
            all_factors.extend(decision_factors)

        # Compute stats
        total_windows_used = sum(
            len(relevant_map.get(d, {}).get("window_indices", []))
            for d in unique_decisions
        )
        avg_windows = (
            total_windows_used / len(unique_decisions)
            if unique_decisions
            else 0.0
        )

        return DecisionExtractionResult(
            text_id=text_id,
            decisions=all_decisions,
            factors=all_factors,
            window_count=len(windows),
            metadata={
                "strategy": "semantic_two_pass",
                "window_count": len(windows),
                "decisions_found": len(all_decisions),
                "factors_found": len(all_factors),
                "avg_windows_per_decision": round(avg_windows, 2),
                "semantic_search_time": round(semantic_time, 2),
                "extraction_time": round(time.time() - start_time, 2),
                "embedding_model": self._embedding_model_name,
                "top_k_windows": self.config.top_k_windows,
                "min_similarity": self.config.min_similarity,
            },
        )

    # ------------------------------------------------------------------
    # Strategy: context_aware
    # ------------------------------------------------------------------

    async def _extract_context_aware(
        self, text: str, text_id: str
    ) -> DecisionExtractionResult:
        """
        Context-aware extraction with summarization and per-decision probing.

        1. For each window: summarize, then extract decisions with context
        2. Collect + validate + normalize decisions
        3. Per-decision factor probing across all windows
        4. Validate + deduplicate factors
        """
        start_time = time.time()
        windows = self.text_processor.process(text)

        if not windows:
            return DecisionExtractionResult(
                text_id=text_id,
                metadata={"strategy": "context_aware", "window_count": 0},
            )

        # Stage 1: Summarize + extract decisions from each window
        logger.info(f"Stage 1: Processing {len(windows)} windows...")
        summaries: List[WindowSummary] = []
        raw_decisions: List[str] = []
        decision_window_map: Dict[str, List[int]] = {}
        context_points: List[str] = []

        for i, window_text in enumerate(windows):
            # Summarize window
            summary = await self._generate_single_summary(window_text, i)
            summaries.append(summary)

            # Extract decisions with summary context
            context_str = ""
            if context_points:
                recent = context_points[-10:]
                context_str = "\n".join(f"- {p}" for p in recent)

            prompt = self.decisions_prompt.replace("{text}", window_text)
            if context_str:
                prompt = (
                    f"Context from previous passages:\n{context_str}\n\n{prompt}"
                )

            try:
                response = await self.llm.generate(
                    prompt=prompt,
                    temperature=self.config.temperature,
                )
                data = self._parse_json_response(response)
                decisions = data.get("decisions", [])

                for d in decisions:
                    d_text = d if isinstance(d, str) else d.get("decision", "")
                    d_text = d_text.strip()
                    if d_text:
                        raw_decisions.append(d_text)
                        decision_window_map.setdefault(
                            d_text.lower(), []
                        ).append(i)
            except Exception as e:
                logger.warning(f"Failed to extract from window {i}: {e}")

            # Update context points from summary
            if summary.bullet_points:
                context_points.extend(summary.bullet_points[:3])

        # Stage 2: Validate + normalize decisions
        if self.config.normalize_decisions:
            raw_decisions = [
                DecisionValidator.strip_causal_clauses(d) for d in raw_decisions
            ]

        # Deduplicate (case-insensitive exact match for context_aware)
        seen: Dict[str, str] = {}
        unique_decisions: List[str] = []
        for d in raw_decisions:
            key = d.lower()
            if key not in seen:
                seen[key] = d
                unique_decisions.append(d)

        if self.validator:
            unique_decisions = self.validator.validate_decisions(unique_decisions)

        if not unique_decisions:
            return DecisionExtractionResult(
                text_id=text_id,
                window_count=len(windows),
                metadata={
                    "strategy": "context_aware",
                    "window_count": len(windows),
                    "decisions_found": 0,
                },
            )

        logger.info(
            f"Stage 2: Probing factors for {len(unique_decisions)} decisions "
            f"across {len(windows)} windows..."
        )

        # Stage 3: Per-decision factor probing
        all_factors: List[Factor] = []
        all_decision_objs: List[Decision] = []

        for d_text in unique_decisions:
            decision_factors: List[Factor] = []

            for i, window_text in enumerate(windows):
                # Build context-enriched prompt
                summary = summaries[i] if i < len(summaries) else None
                context_str = ""
                if summary and summary.bullet_points:
                    context_str = "\n".join(
                        f"- {p}" for p in summary.bullet_points
                    )

                factors = await self._extract_factors_for_decision(
                    d_text,
                    window_text,
                    window_index=i,
                    text_id=text_id,
                    context=context_str,
                )
                decision_factors.extend(factors)

            # Deduplicate factors for this decision
            decision_factors = self._deduplicate_factors(decision_factors)

            dec = Decision(
                text=d_text,
                factors=decision_factors,
                window_indices=decision_window_map.get(d_text.lower(), []),
                text_ids=[text_id] if text_id else [],
            )
            all_decision_objs.append(dec)
            all_factors.extend(decision_factors)

        return DecisionExtractionResult(
            text_id=text_id,
            decisions=all_decision_objs,
            factors=all_factors,
            window_count=len(windows),
            metadata={
                "strategy": "context_aware",
                "window_count": len(windows),
                "decisions_found": len(all_decision_objs),
                "factors_found": len(all_factors),
                "extraction_time": round(time.time() - start_time, 2),
            },
        )

    # ------------------------------------------------------------------
    # Shared: Window summarization
    # ------------------------------------------------------------------

    async def _generate_window_summaries(
        self, windows: List[str]
    ) -> List[WindowSummary]:
        """Generate summaries for all windows."""
        summaries = []
        for i, window_text in enumerate(windows):
            summary = await self._generate_single_summary(window_text, i)
            summaries.append(summary)
        return summaries

    async def _generate_single_summary(
        self, window_text: str, window_index: int
    ) -> WindowSummary:
        """Generate a summary for a single window."""
        prompt = self.summary_prompt.replace("{text}", window_text)

        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
            )
            data = self._parse_json_response(response)
            return WindowSummary(
                window_index=window_index,
                text=window_text,
                summary=data.get("summary", ""),
                bullet_points=data.get("bullet_points", []),
            )
        except Exception as e:
            label = type(e).__name__
            detail = str(e) or "(no details — likely a timeout)"
            logger.warning(f"Failed to summarize window {window_index}: {label}: {detail}")
            # Extractive fallback
            sentences = window_text.split(". ")
            return WindowSummary(
                window_index=window_index,
                text=window_text,
                summary=sentences[0] if sentences else window_text[:100],
                bullet_points=[s[:80] for s in sentences[:3] if s],
            )

    # ------------------------------------------------------------------
    # Shared: Decision extraction from a single window
    # ------------------------------------------------------------------

    async def _extract_decisions_from_window(
        self, window_text: str
    ) -> List[str]:
        """Extract decisions from a single window using the decisions-only prompt."""
        prompt = self.decisions_prompt.replace("{text}", window_text)

        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
            )
            data = self._parse_json_response(response)
            decisions = data.get("decisions", [])

            result = []
            for d in decisions:
                d_text = d if isinstance(d, str) else d.get("decision", "")
                d_text = d_text.strip()
                if d_text:
                    result.append(d_text)
            return result

        except Exception as e:
            logger.warning(f"Failed to extract decisions from window: {e}")
            return []

    # ------------------------------------------------------------------
    # Shared: Factor extraction for a specific decision
    # ------------------------------------------------------------------

    async def _extract_factors_for_decision(
        self,
        decision: str,
        window_text: str,
        window_index: int = 0,
        text_id: str = "",
        context: str = "",
    ) -> List[Factor]:
        """Extract factors for one decision from one window."""
        prompt = self.factors_prompt.replace("{decision}", decision)
        prompt = prompt.replace("{text}", window_text)
        if context:
            prompt = f"Context:\n{context}\n\n{prompt}"

        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
            )
            data = self._parse_json_response(response)
            raw_factors = data.get("factors", [])

            factors = []
            for rf in raw_factors:
                if isinstance(rf, dict):
                    f_text = rf.get("factor", "").strip()
                    f_polarity = rf.get("polarity", "neutral")
                else:
                    f_text = str(rf).strip()
                    f_polarity = "neutral"

                if not self._validate_factor(f_text, decision):
                    continue

                factors.append(
                    Factor(
                        text=f_text,
                        decision_text=decision,
                        polarity=f_polarity,
                        window_index=window_index,
                        text_id=text_id,
                    )
                )
            return factors

        except Exception as e:
            label = type(e).__name__
            detail = str(e) or "(no details — likely a timeout)"
            logger.warning(
                f"Failed to extract factors for '{decision[:30]}...' "
                f"from window {window_index}: {label}: {detail}"
            )
            return []

    # ------------------------------------------------------------------
    # Shared: Semantic search for relevant windows
    # ------------------------------------------------------------------

    def _find_relevant_windows(
        self,
        decisions: List[str],
        summaries: List[WindowSummary],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Find top-K most relevant windows per decision using embeddings.

        Returns dict mapping decision text -> {window_indices, similarities}.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        embedding_service = self._get_embedding_service()

        # Embed decisions
        decision_embeddings = embedding_service.embed(decisions)

        # Embed window summaries (combine summary + bullet points)
        summary_texts = []
        for ws in summaries:
            combined = ws.summary + " " + " ".join(ws.bullet_points)
            summary_texts.append(combined)

        summary_embeddings = embedding_service.embed(summary_texts)

        # Cosine similarity matrix: [num_decisions, num_windows]
        similarities = cosine_similarity(decision_embeddings, summary_embeddings)

        relevant_map: Dict[str, Dict[str, Any]] = {}

        for i, decision in enumerate(decisions):
            scores = similarities[i]
            top_k_indices = np.argsort(scores)[::-1][: self.config.top_k_windows]

            filtered_indices = []
            filtered_scores = []

            for idx in top_k_indices:
                if scores[idx] >= self.config.min_similarity:
                    filtered_indices.append(int(idx))
                    filtered_scores.append(float(scores[idx]))

            # Fallback: use top-1 if nothing meets threshold
            if not filtered_indices and len(top_k_indices) > 0:
                idx = top_k_indices[0]
                filtered_indices = [int(idx)]
                filtered_scores = [float(scores[idx])]

            relevant_map[decision] = {
                "window_indices": filtered_indices,
                "similarities": filtered_scores,
            }

        return relevant_map

    # ------------------------------------------------------------------
    # Shared: Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_decisions(self, decisions: List[str]) -> List[str]:
        """Deduplicate decisions using exact or semantic matching."""
        if not decisions or len(decisions) <= 1:
            return decisions

        if self.config.dedup_method == "exact":
            seen: Dict[str, str] = {}
            unique = []
            for d in decisions:
                key = d.lower().strip()
                if key not in seen:
                    seen[key] = d
                    unique.append(d)
            return unique

        # Semantic deduplication
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            embedding_service = self._get_embedding_service()
            embeddings = embedding_service.embed(decisions)
            sims = cosine_similarity(embeddings)

            unique = []
            used = set()

            for i in range(len(decisions)):
                if i in used:
                    continue
                unique.append(decisions[i])
                used.add(i)
                for j in range(i + 1, len(decisions)):
                    if j not in used and sims[i][j] >= self.config.decision_dedup_threshold:
                        used.add(j)

            return unique
        except Exception as e:
            logger.warning(f"Semantic dedup failed, falling back to exact: {e}")
            seen_lower: Dict[str, str] = {}
            unique_fallback = []
            for d in decisions:
                key = d.lower().strip()
                if key not in seen_lower:
                    seen_lower[key] = d
                    unique_fallback.append(d)
            return unique_fallback

    def _deduplicate_factors(self, factors: List[Factor]) -> List[Factor]:
        """Deduplicate factors using exact or semantic matching."""
        if not factors or len(factors) <= 1:
            return factors

        if self.config.dedup_method == "exact":
            seen: set = set()
            unique = []
            for f in factors:
                key = (f.text.lower(), f.polarity)
                if key not in seen:
                    seen.add(key)
                    unique.append(f)
            return unique

        # Semantic deduplication
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            embedding_service = self._get_embedding_service()
            texts = [f.text for f in factors]
            embeddings = embedding_service.embed(texts)
            sims = cosine_similarity(embeddings)

            unique = []
            used = set()

            for i in range(len(factors)):
                if i in used:
                    continue
                unique.append(factors[i])
                used.add(i)
                for j in range(i + 1, len(factors)):
                    if j not in used and sims[i][j] >= self.config.factor_dedup_threshold:
                        used.add(j)

            return unique
        except Exception as e:
            logger.warning(f"Semantic dedup failed, falling back to exact: {e}")
            seen_keys: set = set()
            unique_fallback = []
            for f in factors:
                key = (f.text.lower(), f.polarity)
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_fallback.append(f)
            return unique_fallback

    # ------------------------------------------------------------------
    # Shared: Factor validation
    # ------------------------------------------------------------------

    def _validate_factor(self, factor_text: str, decision_text: str) -> bool:
        """
        Validate a factor against quality criteria.

        Rejects:
        - Empty factors
        - Factors identical to the decision
        - Factors with fewer than min_words words
        - Factors with > max_decision_overlap word overlap with the decision
        """
        if not factor_text:
            return False

        # Identical to decision
        if factor_text.lower().strip() == decision_text.lower().strip():
            return False

        # Too short
        words = factor_text.split()
        if len(words) < self.config.factor_min_words:
            return False

        # High word overlap with decision
        factor_words = set(w.lower() for w in words)
        decision_words = set(w.lower() for w in decision_text.split())
        if factor_words and decision_words:
            overlap = len(factor_words & decision_words) / len(factor_words)
            if overlap > self.config.factor_max_decision_overlap:
                return False

        return True

    # ------------------------------------------------------------------
    # Shared: JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _find_balanced_json(text: str, open_ch: str, close_ch: str) -> Optional[str]:
        """Find the first balanced JSON object/array using brace counting.

        Returns the substring from the first *open_ch* to its matching
        *close_ch*, respecting nesting and JSON string literals.
        Returns ``None`` when no balanced pair is found.
        """
        start = text.find(open_ch)
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\" and in_string:
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """
        Extract JSON from LLM response, handling markdown fences,
        nested JSON strings, and garbage text after valid output.
        """
        content = response.strip()

        # Strip <think>…</think> reasoning blocks (thinking models like Qwen3.5)
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        content = content.strip()

        # Strip markdown code fences
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        # Try balanced JSON object first
        obj = DecisionExtractor._find_balanced_json(content, "{", "}")
        if obj is not None:
            data = json.loads(obj)
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict):
                return {"decisions": data if isinstance(data, list) else []}
            return data

        # Fall back to JSON array
        arr = DecisionExtractor._find_balanced_json(content, "[", "]")
        if arr is not None:
            parsed = json.loads(arr)
            if isinstance(parsed, list):
                return {"decisions": parsed}

        # Last resort — try the whole content
        data = json.loads(content)
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            return {"decisions": data if isinstance(data, list) else []}
        return data

    # ------------------------------------------------------------------
    # Shared: Embedding service
    # ------------------------------------------------------------------

    def _get_embedding_service(self):
        """Lazy-load embedding service from core/embeddings.py."""
        if self._embedding_service is None:
            from ..core.embeddings import EmbeddingService

            self._embedding_service = EmbeddingService(
                model_name=self._embedding_model_name,
            )
        return self._embedding_service
