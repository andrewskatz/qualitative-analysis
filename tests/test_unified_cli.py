"""
CLI integration tests for the unified qa command.

These tests verify the CLI structure, help output, and aliases work correctly.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _run_qa_command(*args: str) -> subprocess.CompletedProcess:
    """Run the qa module with the package src directory on PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{SRC_DIR}{os.pathsep}{existing}" if existing else str(SRC_DIR)
    )
    return subprocess.run(
        [sys.executable, "-m", "qualitative_analysis.unified_cli", *args],
        capture_output=True,
        text=True,
        env=env,
    )


class TestQAHelpOutput(unittest.TestCase):
    """Test help output for the unified qa command."""

    def _run_qa(self, *args: str) -> subprocess.CompletedProcess:
        """Run the qa command with given arguments."""
        return _run_qa_command(*args)

    def test_qa_help(self):
        """Test that qa --help shows analysis types."""
        result = self._run_qa("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("figurative", result.stdout)
        self.assertIn("relationships", result.stdout)
        self.assertIn("fig", result.stdout)  # alias shown
        self.assertIn("rel", result.stdout)  # alias shown

    def test_qa_version(self):
        """Test that qa --version shows package version."""
        from qualitative_analysis import __version__

        result = self._run_qa("--version")
        self.assertEqual(result.returncode, 0)
        self.assertIn(__version__, result.stdout)

    def test_qa_figurative_help(self):
        """Test that qa figurative --help shows subcommands."""
        result = self._run_qa("figurative", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("detect", result.stdout)
        self.assertIn("map", result.stdout)
        self.assertIn("normalize", result.stdout)
        self.assertIn("graph", result.stdout)
        self.assertIn("pipeline", result.stdout)

    def test_qa_fig_alias(self):
        """Test that 'fig' alias works for 'figurative'."""
        result = self._run_qa("fig", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("detect", result.stdout)
        self.assertIn("map", result.stdout)

    def test_qa_relationships_help(self):
        """Test that qa relationships --help shows subcommands."""
        result = self._run_qa("relationships", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("detect", result.stdout)

    def test_qa_rel_alias(self):
        """Test that 'rel' alias works for 'relationships'."""
        result = self._run_qa("rel", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("detect", result.stdout)

    def test_qa_figurative_detect_help(self):
        """Test that qa figurative detect --help shows arguments."""
        result = self._run_qa("figurative", "detect", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("input_csv", result.stdout)
        self.assertIn("--model", result.stdout)
        self.assertIn("--log-llm", result.stdout)
        self.assertIn("--window-size", result.stdout)
        self.assertIn("alignment_status", result.stdout)
        self.assertIn("supporting_window_indices", result.stdout)
        self.assertIn("window-local", result.stdout)

    def test_qa_figurative_detect_checkpoint_args(self):
        """Test that qa figurative detect --help shows checkpoint arguments."""
        result = self._run_qa("figurative", "detect", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--checkpoint", result.stdout)
        self.assertIn("--checkpoint-interval", result.stdout)

    def test_qa_figurative_map_help(self):
        """Test that qa figurative map --help shows arguments."""
        result = self._run_qa("figurative", "map", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("input_csv", result.stdout)
        self.assertIn("--multi-level", result.stdout)
        self.assertIn("--model", result.stdout)
        self.assertIn("does not require", result.stdout)
        self.assertIn("instance_text", result.stdout)

    def test_qa_figurative_normalize_help(self):
        """Test that qa figurative normalize --help shows arguments."""
        result = self._run_qa("figurative", "normalize", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("input_csv", result.stdout)
        self.assertIn("--merge-threshold", result.stdout)
        self.assertIn("--canonical-method", result.stdout)

    def test_qa_figurative_graph_help(self):
        """Test that qa figurative graph --help shows arguments."""
        result = self._run_qa("figurative", "graph", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("input_csv", result.stdout)
        self.assertIn("--visualize", result.stdout)
        self.assertIn("--format", result.stdout)

    def test_qa_relationships_detect_help(self):
        """Test that qa relationships detect --help shows arguments."""
        result = self._run_qa("relationships", "detect", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("input_csv", result.stdout)
        self.assertIn("--model", result.stdout)
        self.assertIn("--strategy", result.stdout)
        self.assertIn("--coref", result.stdout)

    def test_qa_entity_detect_help(self):
        """Test that qa entity detect --help shows key arguments."""
        result = self._run_qa("entity", "detect", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("input_csv", result.stdout)
        self.assertIn("--text-col", result.stdout)
        self.assertIn("--group-col", result.stdout)
        self.assertIn("--enable-thinking", result.stdout)
        self.assertIn("--response-format", result.stdout)


class TestCommonArguments(unittest.TestCase):
    """Test that common arguments are consistent across commands."""

    def _run_qa(self, *args: str) -> subprocess.CompletedProcess:
        """Run the qa command with given arguments."""
        return _run_qa_command(*args)

    def test_model_arg_in_detect_commands(self):
        """Test that --model is present in all detect commands."""
        for cmd in [
            ["figurative", "detect", "--help"],
            ["relationships", "detect", "--help"],
        ]:
            result = self._run_qa(*cmd)
            self.assertIn("--model", result.stdout, f"--model not in {cmd}")

    def test_log_llm_arg_in_llm_commands(self):
        """Test that --log-llm is present in LLM-using commands."""
        for cmd in [
            ["figurative", "detect", "--help"],
            ["figurative", "map", "--help"],
            ["relationships", "detect", "--help"],
        ]:
            result = self._run_qa(*cmd)
            self.assertIn("--log-llm", result.stdout, f"--log-llm not in {cmd}")

    def test_base_url_arg_in_llm_commands(self):
        """Test that --base-url is present in LLM-using commands."""
        for cmd in [
            ["figurative", "detect", "--help"],
            ["figurative", "map", "--help"],
            ["relationships", "detect", "--help"],
        ]:
            result = self._run_qa(*cmd)
            self.assertIn("--base-url", result.stdout, f"--base-url not in {cmd}")


if __name__ == "__main__":
    unittest.main()
