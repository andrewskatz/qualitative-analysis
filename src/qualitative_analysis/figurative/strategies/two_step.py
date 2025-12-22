"""
Two-step strategy with summaries.
"""

import time
from typing import Dict, Any, List

from .base import BaseStrategy
from ..models import DetectionResult
from ..components.summarizer import Summarizer
from ..components.scanner import Scanner
from ..components.buffer import SummaryBuffer

class TwoStepWithSummariesStrategy(BaseStrategy):
    """
    Detects figurative language by:
    1. Summarizing windows to maintain context.
    2. Scanning windows with context awareness.
    """
    
    def __init__(
        self,
        llm,
        text_processor,
        summary_buffer_size: int = 5,
        threshold: float = 0.5,
        prompt_version: int = 1,
        return_windows: bool = False,
    ):
        super().__init__(llm, text_processor)
        self.summary_buffer_size = summary_buffer_size
        self.threshold = threshold
        self.return_windows = return_windows
        
        # Initialize components
        self.summarizer = Summarizer(llm, prompt_version=prompt_version)
        self.scanner = Scanner(llm, prompt_version=prompt_version)

    async def detect(self, text: str) -> DetectionResult:
        # 1. Chunk text
        windows = self.text_processor.process(text)
        
        if not windows:
             return DetectionResult(False, 0.0, [], {})

        # 2. Process windows
        summary_buffer = SummaryBuffer(self.summary_buffer_size)
        all_instances = []
        window_results = []
        
        for i, window in enumerate(windows):
            # Step 1: Summarize
            summary = await self.summarizer.summarize(
                window, 
                summary_buffer.get_context(), 
                window_index=i
            )
            summary_buffer.add(summary)
            
            # Step 2: Detect & Extract
            context_str = summary_buffer.get_formatted_context()
            
            has_fig, confidence = await self.scanner.detect(window, context_str)
            
            if has_fig and confidence >= self.threshold:
                instances = await self.scanner.extract(window, context_str, window_index=i)
                all_instances.extend(instances)
            else:
                instances = []

            if self.return_windows:
                window_results.append({
                    "window_index": i,
                    "window_text": window,
                    "summary": summary.text,
                    "has_figurative": has_fig,
                    "confidence": confidence,
                    "instances_count": len(instances),
                })
        
        # 3. Aggregate
        return self._aggregate_instances(all_instances, len(windows), window_results)

    def _aggregate_instances(
        self,
        instances: List[Any],
        window_count: int,
        window_results: List[Dict[str, Any]],
    ) -> DetectionResult:
        has_figurative = len(instances) > 0
        
        confidence = 0.0
        if instances:
            confidence = sum(i.confidence for i in instances) / len(instances)
            
        return DetectionResult(
            contains_figurative=has_figurative,
            confidence=confidence,
            instances=instances,
            metadata={
                "strategy": "two_step_with_summaries",
                "window_count": window_count,
                "instance_count": len(instances),
                "windows": window_results if self.return_windows else [],
            }
        )
