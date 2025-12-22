"""
Text processing utilities.
"""

import re
from typing import List, Optional

class TextProcessor:
    """Base class for text processors."""
    
    def process(self, text: str) -> List[str]:
        raise NotImplementedError


class SlidingWindowProcessor(TextProcessor):
    """
    Sliding window processor that creates overlapping windows of text.
    """
    
    def __init__(
        self,
        window_size: int = 3,
        stride: int = 2,
        chunk_unit: str = 'sentences'
    ):
        """
        Initialize sliding window processor.

        Args:
            window_size: Number of units per window (default: 3)
            stride: Number of units to move forward (default: 2)
            chunk_unit: Unit type - 'sentences' or 'tokens' (default: 'sentences')
        """
        self.window_size = window_size
        self.stride = stride
        self.chunk_unit = chunk_unit
    
    def process(self, text: str) -> List[str]:
        """
        Process text using sliding window with overlap.

        Args:
            text: The text to process

        Returns:
            List of overlapping text windows
        """
        if not text or not text.strip():
            return []

        # TODO: Implement token-based if needed. For now, assume sentences.
        if self.chunk_unit == 'tokens':
             # Placeholder for token logic
             raise NotImplementedError("Token-based chunking not yet implemented in generic module.")

        # Default: sentence-based
        sentences = self._split_sentences(text)

        if len(sentences) <= self.window_size:
            return [text]

        windows = []
        for i in range(0, len(sentences) - self.window_size + 1, self.stride):
            window_sentences = sentences[i:i + self.window_size]
            window_text = ' '.join(window_sentences)
            windows.append(window_text)

        # Handle last chunk
        last_window_start = (len(windows) - 1) * self.stride if windows else 0
        last_window_end = last_window_start + self.window_size

        if last_window_end < len(sentences):
            final_window_sentences = sentences[-self.window_size:]
            final_window_text = ' '.join(final_window_sentences)
            if not windows or final_window_text != windows[-1]:
                windows.append(final_window_text)

        return windows
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter."""
        # This is a naive implementation. In a real module, use nltk or spacy.
        # For portability, we'll strive for regex-based for now.
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
