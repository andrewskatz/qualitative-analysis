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
        chunk_unit: str = 'sentences',
        tokenizer_name: str = "cl100k_base",
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
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
    
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

        if self.chunk_unit == 'tokens':
            units = self._encode_tokens(text)
            joiner = self._decode_tokens
        elif self.chunk_unit == 'sentences':
            units = self._split_sentences(text)
            joiner = lambda items: ' '.join(items)
        else:
            raise ValueError(f"Unsupported chunk_unit: {self.chunk_unit}")

        if len(units) <= self.window_size:
            return [text]

        windows = []
        for i in range(0, len(units) - self.window_size + 1, self.stride):
            window_units = units[i:i + self.window_size]
            window_text = joiner(window_units)
            windows.append(window_text)

        # Handle last chunk
        last_window_start = (len(windows) - 1) * self.stride if windows else 0
        last_window_end = last_window_start + self.window_size

        if last_window_end < len(units):
            final_window_units = units[-self.window_size:]
            final_window_text = joiner(final_window_units)
            if not windows or final_window_text != windows[-1]:
                windows.append(final_window_text)

        return windows
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter."""
        # This is a naive implementation. In a real module, use nltk or spacy.
        # For portability, we'll strive for regex-based for now.
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                import tiktoken
            except ImportError as exc:
                raise ImportError(
                    "Token-based chunking requires tiktoken. "
                    "Install it with `pip install tiktoken`."
                ) from exc
            self._tokenizer = tiktoken.get_encoding(self.tokenizer_name)
        return self._tokenizer

    def _encode_tokens(self, text: str) -> List[int]:
        """Tokenize text into model token IDs."""
        tokenizer = self._get_tokenizer()
        return tokenizer.encode(text)

    def _decode_tokens(self, tokens: List[int]) -> str:
        """Decode token IDs into text."""
        tokenizer = self._get_tokenizer()
        return tokenizer.decode(tokens)
