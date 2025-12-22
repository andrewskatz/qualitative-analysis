"""
Summary buffer for maintaining context.
"""

from typing import List
from ..models import Summary

class SummaryBuffer:
    def __init__(self, max_size: int = 5):
        self.summaries: List[Summary] = []
        self.max_size = max_size
    
    def add(self, summary: Summary) -> None:
        """Add summary to buffer (FIFO behavior)."""
        self.summaries.append(summary)
        if len(self.summaries) > self.max_size:
            self.summaries.pop(0)

    def get_context(self) -> List[Summary]:
        """Get current summaries."""
        return list(self.summaries)

    def get_formatted_context(self) -> str:
        """Format for prompt injection."""
        if not self.summaries:
            return "No prior context."

        formatted = "PRIOR CONTEXT:\n"
        for summary in self.summaries:
            formatted += f"\nWindow {summary.window_index}:\n{summary.text}\n"

        return formatted.rstrip()
