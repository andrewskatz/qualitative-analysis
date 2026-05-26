"""
Two-step strategy with summaries.
"""

from difflib import SequenceMatcher
import string
from typing import Dict, Any, List, Optional, Tuple

from .base import BaseStrategy
from ..models import DetectionResult, Instance
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
        figurative_types: Optional[List[str]] = None,
    ):
        super().__init__(llm, text_processor)
        self.summary_buffer_size = summary_buffer_size
        self.threshold = threshold
        self.return_windows = return_windows
        self.figurative_types = figurative_types
        
        # Initialize components
        self.summarizer = Summarizer(llm, prompt_version=prompt_version)
        self.scanner = Scanner(
            llm,
            prompt_version=prompt_version,
            figurative_types=figurative_types,
        )

    @staticmethod
    def _empty_scanner_stats() -> Dict[str, int]:
        """Create the default aggregated scanner-metadata structure."""
        return {
            "detection_structured_output_fallbacks": 0,
            "detection_parse_failures": 0,
            "extraction_structured_output_fallbacks": 0,
            "extraction_parse_failures": 0,
            "invalid_extraction_items_skipped": 0,
        }

    @staticmethod
    def _empty_canonicalization_stats() -> Dict[str, Any]:
        """Create default metadata for canonicalization and alignment."""
        return {
            "raw_instance_count": 0,
            "canonical_instance_count": 0,
            "deduplicated_instance_count": 0,
            "alignment_status_counts": {
                "exact": 0,
                "normalized_exact": 0,
                "ambiguous": 0,
                "missing": 0,
                "approximate": 0,
            },
            "supporting_windows": [],
        }

    @staticmethod
    def _normalize_type(value: str) -> str:
        return value.lower().strip().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _translate_char(value: str) -> str:
        quote_map = {
            "“": '"',
            "”": '"',
            "„": '"',
            "‟": '"',
            "«": '"',
            "»": '"',
            "‘": "'",
            "’": "'",
            "`": "'",
            "´": "'",
        }
        dash_map = {
            "‐": "-",
            "‑": "-",
            "‒": "-",
            "–": "-",
            "—": "-",
            "−": "-",
        }
        return dash_map.get(quote_map.get(value, value), quote_map.get(value, value))

    @classmethod
    def _trim_edge_indices(cls, text: str) -> Tuple[int, int]:
        start = 0
        end = len(text)
        while start < end and (text[start].isspace() or text[start] in string.punctuation + "“”‘’«»„‟"):
            start += 1
        while end > start and (text[end - 1].isspace() or text[end - 1] in string.punctuation + "“”‘’«»„‟"):
            end -= 1
        return start, end

    @classmethod
    def _normalize_for_matching(
        cls,
        text: str,
        *,
        trim_outer_punctuation: bool,
    ) -> Tuple[str, List[int]]:
        start = 0
        end = len(text)
        if trim_outer_punctuation:
            start, end = cls._trim_edge_indices(text)

        normalized_chars: List[str] = []
        original_positions: List[int] = []

        for index in range(start, end):
            translated = cls._translate_char(text[index]).lower()
            if translated.isspace():
                if normalized_chars and normalized_chars[-1] != " ":
                    normalized_chars.append(" ")
                    original_positions.append(index)
                continue

            normalized_chars.append(translated)
            original_positions.append(index)

        if normalized_chars and normalized_chars[-1] == " ":
            normalized_chars.pop()
            original_positions.pop()

        return "".join(normalized_chars), original_positions

    @staticmethod
    def _find_all_occurrences(text: str, needle: str) -> List[int]:
        if not text or not needle or len(needle) > len(text):
            return []

        matches = []
        start = 0
        while True:
            index = text.find(needle, start)
            if index == -1:
                return matches
            matches.append(index)
            start = index + 1

    @classmethod
    def _align_instance_to_window(
        cls,
        instance_text: str,
        window_text: str,
    ) -> Tuple[str, Optional[int], Optional[int]]:
        raw_matches = cls._find_all_occurrences(window_text, instance_text)
        if len(raw_matches) == 1:
            start = raw_matches[0]
            return "exact", start, start + len(instance_text)
        if len(raw_matches) > 1:
            return "ambiguous", None, None

        normalized_window, window_positions = cls._normalize_for_matching(
            window_text,
            trim_outer_punctuation=False,
        )
        normalized_instance, _ = cls._normalize_for_matching(
            instance_text,
            trim_outer_punctuation=True,
        )

        if not normalized_window or not normalized_instance:
            return "missing", None, None

        normalized_matches = cls._find_all_occurrences(normalized_window, normalized_instance)
        if len(normalized_matches) == 1:
            start_index = normalized_matches[0]
            end_index = start_index + len(normalized_instance) - 1
            return (
                "normalized_exact",
                window_positions[start_index],
                window_positions[end_index] + 1,
            )
        if len(normalized_matches) > 1:
            return "ambiguous", None, None

        if cls._has_approximate_candidate(normalized_instance, normalized_window):
            return "approximate", None, None
        return "missing", None, None

    @staticmethod
    def _has_approximate_candidate(instance_text: str, window_text: str) -> bool:
        if len(instance_text) < 4 or len(window_text) < 4:
            return False

        min_len = max(1, len(instance_text) - 2)
        max_len = min(len(window_text), len(instance_text) + 2)
        best_ratio = 0.0

        for candidate_length in range(min_len, max_len + 1):
            for start in range(0, len(window_text) - candidate_length + 1):
                candidate = window_text[start:start + candidate_length]
                ratio = SequenceMatcher(None, instance_text, candidate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio

        return best_ratio >= 0.88

    @staticmethod
    def _alignment_rank(status: str) -> int:
        if status == "exact":
            return 2
        if status == "normalized_exact":
            return 1
        return 0

    @classmethod
    def _canonical_text_key(cls, text: str) -> str:
        normalized, _ = cls._normalize_for_matching(text, trim_outer_punctuation=True)
        return normalized

    def _window_overlap_delta(self) -> int:
        window_size = getattr(self.text_processor, "window_size", None)
        stride = getattr(self.text_processor, "stride", None)
        if not isinstance(window_size, int) or not isinstance(stride, int):
            return 0
        if window_size <= 0 or stride <= 0 or stride >= window_size:
            return 0
        return max(1, (window_size - 1) // stride)

    @staticmethod
    def _shared_window_overlap(left_window: str, right_window: str) -> int:
        max_length = min(len(left_window), len(right_window))
        for size in range(max_length, 0, -1):
            if left_window[-size:] == right_window[:size]:
                return size
        return 0

    def _instances_share_occurrence(
        self,
        left: Instance,
        right: Instance,
        windows: List[str],
        overlap_delta: int,
    ) -> bool:
        if left.window_index == right.window_index:
            return (
                left.start_char is not None
                and left.end_char is not None
                and right.start_char is not None
                and right.end_char is not None
                and left.start_char == right.start_char
                and left.end_char == right.end_char
            )

        if self._alignment_rank(left.alignment_status) == 0 or self._alignment_rank(right.alignment_status) == 0:
            return False

        if abs(left.window_index - right.window_index) > overlap_delta:
            return False

        earlier, later = (left, right) if left.window_index < right.window_index else (right, left)
        earlier_window = windows[earlier.window_index]
        later_window = windows[later.window_index]
        overlap_length = self._shared_window_overlap(earlier_window, later_window)
        if overlap_length <= 0:
            return False

        earlier_overlap_start = len(earlier_window) - overlap_length
        return (
            earlier.start_char is not None
            and earlier.end_char is not None
            and later.start_char is not None
            and later.end_char is not None
            and earlier.start_char >= earlier_overlap_start
            and earlier.end_char <= len(earlier_window)
            and later.start_char >= 0
            and later.end_char <= overlap_length
        )

    def _canonicalize_instances(
        self,
        instances: List[Instance],
        windows: List[str],
    ) -> Tuple[List[Instance], Dict[str, Any]]:
        metadata = self._empty_canonicalization_stats()
        metadata["raw_instance_count"] = len(instances)

        if not instances:
            return [], metadata

        for instance in instances:
            window_text = windows[instance.window_index] if 0 <= instance.window_index < len(windows) else ""
            status, start_char, end_char = self._align_instance_to_window(instance.text, window_text)
            instance.alignment_status = status
            instance.start_char = start_char
            instance.end_char = end_char
            instance.support_count = 1
            instance.supporting_window_indices = [instance.window_index]

        overlap_delta = self._window_overlap_delta()
        grouped: List[Dict[str, Any]] = []

        ordered_instances = sorted(
            instances,
            key=lambda item: (
                self._canonical_text_key(item.text),
                self._normalize_type(item.type),
                item.window_index,
            ),
        )

        for instance in ordered_instances:
            key = (self._canonical_text_key(instance.text), self._normalize_type(instance.type))
            placed = False
            for group in grouped:
                if group["key"] != key:
                    continue
                if any(
                    self._instances_share_occurrence(instance, existing, windows, overlap_delta)
                    for existing in group["instances"]
                ):
                    group["instances"].append(instance)
                    placed = True
                    break
            if not placed:
                grouped.append({"key": key, "instances": [instance]})

        canonical_instances: List[Instance] = []
        for group in grouped:
            candidates = group["instances"]
            representative = sorted(
                candidates,
                key=lambda item: (
                    -item.confidence,
                    -self._alignment_rank(item.alignment_status),
                    item.window_index,
                ),
            )[0]
            support_windows = sorted({item.window_index for item in candidates})
            representative.supporting_window_indices = support_windows
            representative.support_count = len(support_windows)
            canonical_instances.append(representative)
            metadata["supporting_windows"].append(
                {
                    "instance_text": representative.text,
                    "type": representative.type,
                    "window_index": representative.window_index,
                    "supporting_window_indices": support_windows,
                    "support_count": len(support_windows),
                    "alignment_status": representative.alignment_status,
                }
            )

        canonical_instances.sort(
            key=lambda item: (
                item.window_index,
                item.start_char if item.start_char is not None else 10**9,
                self._canonical_text_key(item.text),
                self._normalize_type(item.type),
            )
        )

        metadata["canonical_instance_count"] = len(canonical_instances)
        metadata["deduplicated_instance_count"] = len(instances) - len(canonical_instances)
        for instance in canonical_instances:
            metadata["alignment_status_counts"][instance.alignment_status] += 1

        return canonical_instances, metadata

    async def detect(self, text: str) -> DetectionResult:
        # 1. Chunk text
        windows = self.text_processor.process(text)
        
        if not windows:
             return DetectionResult(
                 False,
                 0.0,
                 [],
                 {
                     "strategy": "two_step_with_summaries",
                     "window_count": 0,
                     "instance_count": 0,
                     "scanner_stats": self._empty_scanner_stats(),
                     "windows": [],
                 },
             )

        # 2. Process windows
        summary_buffer = SummaryBuffer(self.summary_buffer_size)
        all_instances = []
        window_results = []
        scanner_stats = self._empty_scanner_stats()
        
        for i, window in enumerate(windows):
            prior_summaries = summary_buffer.get_context()
            context_str = summary_buffer.get_formatted_context()

            # Step 1: Summarize
            summary = await self.summarizer.summarize(
                window, 
                prior_summaries,
                window_index=i
            )
            
            # Step 2: Detect & Extract
            has_fig, confidence, detect_meta = await self.scanner.detect(window, context_str)
            scanner_stats["detection_structured_output_fallbacks"] += int(
                detect_meta.get("structured_output_fallback", False)
            )
            scanner_stats["detection_parse_failures"] += int(
                detect_meta.get("parse_failed", False)
            )
            extract_meta = {
                "structured_output_fallback": False,
                "parse_failed": False,
                "invalid_items_skipped": 0,
            }
            
            if has_fig and confidence >= self.threshold:
                instances, extract_meta = await self.scanner.extract(
                    window,
                    context_str,
                    window_index=i,
                )
                scanner_stats["extraction_structured_output_fallbacks"] += int(
                    extract_meta.get("structured_output_fallback", False)
                )
                scanner_stats["extraction_parse_failures"] += int(
                    extract_meta.get("parse_failed", False)
                )
                scanner_stats["invalid_extraction_items_skipped"] += int(
                    extract_meta.get("invalid_items_skipped", 0)
                )
                all_instances.extend(instances)
            else:
                instances = []

            summary_buffer.add(summary)

            if self.return_windows:
                window_results.append({
                    "window_index": i,
                    "window_text": window,
                    "summary": summary.text,
                    "has_figurative": has_fig,
                    "confidence": confidence,
                    "instances_count": len(instances),
                    "detection_parse_failed": detect_meta.get("parse_failed", False),
                    "extraction_parse_failed": extract_meta.get("parse_failed", False),
                    "invalid_items_skipped": extract_meta.get("invalid_items_skipped", 0),
                })
        
        # 3. Aggregate
        return self._aggregate_instances(
            all_instances,
            windows,
            len(windows),
            window_results,
            scanner_stats,
        )

    def _aggregate_instances(
        self,
        instances: List[Instance],
        windows: List[str],
        window_count: int,
        window_results: List[Dict[str, Any]],
        scanner_stats: Dict[str, int],
    ) -> DetectionResult:
        canonical_instances, canonicalization = self._canonicalize_instances(instances, windows)
        has_figurative = len(canonical_instances) > 0
        
        confidence = 0.0
        if canonical_instances:
            confidence = sum(i.confidence for i in canonical_instances) / len(canonical_instances)
            
        return DetectionResult(
            contains_figurative=has_figurative,
            confidence=confidence,
            instances=canonical_instances,
            metadata={
                "strategy": "two_step_with_summaries",
                "window_count": window_count,
                "instance_count": len(canonical_instances),
                **canonicalization,
                "scanner_stats": scanner_stats,
                "windows": window_results if self.return_windows else [],
            }
        )
