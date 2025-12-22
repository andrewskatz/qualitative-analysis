"""
Prompt loading utilities for versioned prompt files.
"""

from pathlib import Path
from typing import Optional, Dict, Any

DEFAULT_VERSIONS = {
    "system_prompt": 1,
    "two_step_summarization": 1,
    "two_step_binary_detection": 1,
    "two_step_instance_extraction": 1,
}


def _prompts_dir() -> Path:
    return Path(__file__).parent


def load_prompt(prompt_type: str, version: Optional[int] = None) -> str:
    """
    Load a prompt template by type and version.
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


def list_versions(prompt_type: str) -> list[int]:
    """
    List all available versions for a prompt type.
    """
    if prompt_type not in DEFAULT_VERSIONS:
        raise ValueError(
            f"Invalid prompt type: {prompt_type}. "
            f"Valid types: {', '.join(DEFAULT_VERSIONS.keys())}"
        )

    pattern = f"{prompt_type}_v*.txt"
    versions = []
    for filepath in _prompts_dir().glob(pattern):
        version_str = filepath.stem.split("_v")[-1]
        try:
            versions.append(int(version_str))
        except ValueError:
            continue
    return sorted(versions)


def get_prompt_info(prompt_type: str, version: Optional[int] = None) -> Dict[str, Any]:
    """
    Get prompt metadata for inspection/debugging.
    """
    if version is None:
        version = DEFAULT_VERSIONS[prompt_type]

    filename = f"{prompt_type}_v{version}.txt"
    filepath = _prompts_dir() / filename
    info: Dict[str, Any] = {
        "prompt_type": prompt_type,
        "version": version,
        "filename": filename,
        "filepath": str(filepath),
        "exists": filepath.exists(),
        "available_versions": list_versions(prompt_type),
    }

    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        info["size_bytes"] = filepath.stat().st_size
        info["length_chars"] = len(content)
        info["has_xml_tags"] = "<text_to_analyze>" in content or "<text_to_summarize>" in content

    return info
