"""
Prompt loading utilities for versioned decision/factor prompt files.

Supports both built-in versioned prompts and custom user-provided prompt files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_VERSIONS = {
    "decisions_only": 5,
    "factors_for_decision": 6,
    "decision_extraction": 2,
    "window_summary": 1,
}


def _prompts_dir() -> Path:
    return Path(__file__).parent


def load_prompt(
    prompt_type: str,
    version: Optional[int] = None,
    custom_path: Optional[str] = None,
) -> str:
    """
    Load a prompt template.

    If custom_path is provided, reads from that file and ignores
    prompt_type/version. Otherwise loads from the built-in versioned
    files.

    Args:
        prompt_type: Type of prompt (e.g. "decisions_only", "factors_for_decision").
        version: Version number. If None, uses the default version.
        custom_path: Path to a custom prompt file. Overrides prompt_type/version.

    Returns:
        The prompt template text.
    """
    if custom_path:
        path = Path(custom_path)
        if not path.exists():
            raise FileNotFoundError(f"Custom prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

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


def list_versions(prompt_type: str) -> List[int]:
    """
    List available versions for a prompt type.

    Args:
        prompt_type: Type of prompt.

    Returns:
        Sorted list of available version numbers.
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


def get_prompt_info(
    prompt_type: str, version: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get metadata about a prompt file.

    Args:
        prompt_type: Type of prompt.
        version: Version number. If None, uses the default.

    Returns:
        Dictionary with prompt metadata.
    """
    if version is None:
        version = DEFAULT_VERSIONS.get(prompt_type, 1)

    filename = f"{prompt_type}_v{version}.txt"
    filepath = _prompts_dir() / filename
    info: Dict[str, Any] = {
        "prompt_type": prompt_type,
        "version": version,
        "filename": filename,
        "filepath": str(filepath),
        "exists": filepath.exists(),
        "available_versions": list_versions(prompt_type)
        if prompt_type in DEFAULT_VERSIONS
        else [],
    }

    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        info["size_bytes"] = filepath.stat().st_size
        info["length_chars"] = len(content)

    return info
