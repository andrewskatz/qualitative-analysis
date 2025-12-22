"""
Legacy prompt accessors (kept for compatibility).
"""

from .loader import load_prompt

SUMMARIZER_SYSTEM = load_prompt("system_prompt", version=1)
SUMMARIZER_USER = load_prompt("two_step_summarization", version=1)

DETECTOR_SYSTEM = load_prompt("system_prompt", version=1)
DETECTOR_USER = load_prompt("two_step_binary_detection", version=1)

EXTRACTOR_SYSTEM = load_prompt("system_prompt", version=1)
EXTRACTOR_USER = load_prompt("two_step_instance_extraction", version=1)
