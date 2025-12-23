"""
Context buffer for relationship extraction.
"""

from typing import List


class EntityBuffer:
    """
    FIFO buffer for maintaining recent entities across windows.
    """

    def __init__(self, max_size: int = 0):
        self.max_size = max_size
        self._buffers: List[List[str]] = []

    def add(self, entities: List[str]) -> None:
        if not self.max_size or self.max_size < 1:
            return
        if entities:
            self._buffers.append(list(entities))
            if len(self._buffers) > self.max_size:
                self._buffers.pop(0)

    def get_entities(self) -> List[str]:
        if not self._buffers:
            return []
        merged = []
        seen = set()
        for chunk in self._buffers:
            for entity in chunk:
                if entity and entity not in seen:
                    merged.append(entity)
                    seen.add(entity)
        return merged
