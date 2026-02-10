"""
Prompt loading utilities for versioned entity prompt files.
"""

from pathlib import Path
from typing import Optional

DEFAULT_VERSIONS = {
    "system_prompt": 1,
    "entity_extraction": 1,
    "entity_scoring": 2,
    "scoring_system_prompt": 1,
}


def _prompts_dir() -> Path:
    return Path(__file__).parent


def load_entity_prompt(prompt_type: str, version: Optional[int] = None) -> str:
    """
    Load a versioned prompt file from the entity prompts directory.

    Args:
        prompt_type: One of the keys in DEFAULT_VERSIONS
            (e.g., "system_prompt", "entity_extraction", "entity_scoring").
        version: Prompt version number. If None, uses the default version.

    Returns:
        The prompt text content.

    Raises:
        ValueError: If prompt_type is not recognized.
        FileNotFoundError: If the versioned prompt file doesn't exist.
    """
    if prompt_type not in DEFAULT_VERSIONS:
        raise ValueError(
            f"Invalid prompt type: {prompt_type}. "
            f"Valid types: {', '.join(DEFAULT_VERSIONS.keys())}"
        )

    if version is None:
        version = DEFAULT_VERSIONS[prompt_type]

    filename = f"{prompt_type}_v{version}.txt"
    filepath = _prompts_dir() / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    return filepath.read_text(encoding="utf-8")
