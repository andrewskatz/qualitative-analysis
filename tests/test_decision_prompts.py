"""Tests for decision prompt loader."""

import tempfile
import pytest
from pathlib import Path

from qualitative_analysis.decisions.prompts.loader import (
    DEFAULT_VERSIONS,
    load_prompt,
    list_versions,
    get_prompt_info,
)


class TestLoadPrompt:
    def test_load_default_decisions_only(self):
        prompt = load_prompt("decisions_only")
        assert len(prompt) > 0
        assert "{text}" in prompt

    def test_load_default_factors_for_decision(self):
        prompt = load_prompt("factors_for_decision")
        assert len(prompt) > 0
        assert "{decision}" in prompt
        assert "{text}" in prompt

    def test_load_default_decision_extraction(self):
        prompt = load_prompt("decision_extraction")
        assert len(prompt) > 0

    def test_load_default_window_summary(self):
        prompt = load_prompt("window_summary")
        assert len(prompt) > 0

    def test_load_specific_version(self):
        prompt_v2 = load_prompt("decisions_only", version=2)
        prompt_v5 = load_prompt("decisions_only", version=5)
        assert len(prompt_v2) > 0
        assert len(prompt_v5) > 0
        # v5 should generally be longer/more detailed than v2
        assert prompt_v2 != prompt_v5

    def test_invalid_prompt_type(self):
        with pytest.raises(ValueError, match="Invalid prompt type"):
            load_prompt("nonexistent_type")

    def test_missing_version(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("decisions_only", version=999)

    def test_custom_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Custom prompt: {text}")
            f.flush()

            prompt = load_prompt("decisions_only", custom_path=f.name)
            assert prompt == "Custom prompt: {text}"

    def test_custom_path_missing(self):
        with pytest.raises(FileNotFoundError, match="Custom prompt file not found"):
            load_prompt("decisions_only", custom_path="/nonexistent/path.txt")

    def test_custom_path_overrides_type(self):
        """When custom_path is provided, prompt_type is ignored."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("My custom prompt")
            f.flush()

            # Even with an invalid prompt type, custom_path should work
            prompt = load_prompt("decisions_only", version=999, custom_path=f.name)
            assert prompt == "My custom prompt"


class TestListVersions:
    def test_decisions_only_versions(self):
        versions = list_versions("decisions_only")
        assert 2 in versions
        assert 5 in versions
        assert versions == sorted(versions)

    def test_factors_for_decision_versions(self):
        versions = list_versions("factors_for_decision")
        assert 1 in versions
        assert 5 in versions

    def test_decision_extraction_versions(self):
        versions = list_versions("decision_extraction")
        assert 1 in versions
        assert 2 in versions

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            list_versions("nonexistent")


class TestGetPromptInfo:
    def test_basic_info(self):
        info = get_prompt_info("decisions_only")
        assert info["prompt_type"] == "decisions_only"
        assert info["version"] == 5  # default
        assert info["exists"] is True
        assert "size_bytes" in info
        assert "length_chars" in info

    def test_specific_version(self):
        info = get_prompt_info("decisions_only", version=2)
        assert info["version"] == 2
        assert info["exists"] is True

    def test_available_versions(self):
        info = get_prompt_info("decisions_only")
        assert len(info["available_versions"]) >= 4


class TestDefaultVersions:
    def test_all_defaults_exist(self):
        """All default versions should have corresponding files."""
        for prompt_type, version in DEFAULT_VERSIONS.items():
            prompt = load_prompt(prompt_type, version=version)
            assert len(prompt) > 0, f"Default prompt {prompt_type} v{version} is empty"
