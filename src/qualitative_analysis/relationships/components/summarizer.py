"""
Window summarizer component for relationship extraction.
"""

import json
import logging
from typing import List

from ...core.llm import BaseLLMProvider
from ..models import Summary
from ..prompts.loader import load_prompt

logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(self, llm: BaseLLMProvider, prompt_version: int = 1):
        self.llm = llm
        self.system_prompt = load_prompt("system_prompt", version=1)
        self.user_template = load_prompt("window_summary", version=prompt_version)

    async def summarize(
        self,
        text: str,
        prior_summaries: List[Summary],
        window_index: int,
    ) -> Summary:
        """
        Generate a summary for a specific window of text.
        """
        context_text = self._format_prior_context(prior_summaries)
        prompt = self._render_prompt(text, context_text)

        summary_text = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.3,
        )

        parsed = self._parse_summary(summary_text)
        return Summary(text=parsed, window_index=window_index)

    def _format_prior_context(self, prior_summaries: List[Summary]) -> str:
        if not prior_summaries:
            return "No prior context."

        formatted = "PRIOR CONTEXT:\n"
        for summary in prior_summaries:
            formatted += f"\nWindow {summary.window_index}:\n{summary.text}\n"
        return formatted.rstrip()

    def _render_prompt(self, text: str, prior_summaries: str) -> str:
        prompt = self.user_template
        if "{prior_summaries}" in prompt:
            prompt = prompt.replace("{prior_summaries}", prior_summaries)
        if "{text}" in prompt:
            prompt = prompt.replace("{text}", text)
        return prompt

    def _parse_summary(self, response: str) -> str:
        try:
            data = json.loads(response.strip())
            if "summary_points" in data:
                points = data["summary_points"]
                if isinstance(points, list) and points:
                    return "\n".join(f"- {point}" for point in points).strip()
            if "summary" in data and data["summary"].strip():
                return data["summary"].strip()
        except json.JSONDecodeError:
            logger.warning("Summarizer returned non-JSON response; using raw text.")
        except Exception as exc:
            logger.warning(f"Summarizer response parse failed; using raw text: {exc}")

        return response.strip()
