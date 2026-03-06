"""Tests for decisions CLI utilities: checkpointing, auto-detection, provider factory."""

import json
import tempfile
import pytest
from pathlib import Path

from qualitative_analysis.decisions.models import (
    DecisionExtractionResult,
    Decision,
    Factor,
)
from qualitative_analysis.core.cli_utils import (
    auto_detect_csv_columns,
    resolve_verbose_flag,
)


# =====================================================================
# Checkpoint helpers
# =====================================================================

class TestCheckpointRoundtrip:
    def test_append_and_load(self, tmp_path):
        from qualitative_analysis.decisions_cli import (
            _append_to_checkpoint,
            _load_checkpoint,
        )

        cp_path = tmp_path / "test_checkpoint.jsonl"

        # Create sample results
        r1 = DecisionExtractionResult(
            text_id="doc1",
            decisions=[Decision(text="reduce budget", factors=[
                Factor(text="cost overruns", polarity="supporting"),
            ])],
            factors=[Factor(text="cost overruns", decision_text="reduce budget", polarity="supporting")],
            window_count=3,
            metadata={"strategy": "one_pass"},
        )
        r2 = DecisionExtractionResult(
            text_id="doc2",
            decisions=[Decision(text="expand team")],
            factors=[],
            window_count=2,
        )

        # Append both
        _append_to_checkpoint(cp_path, r1)
        _append_to_checkpoint(cp_path, r2)

        # Load back
        results, scored_keys = _load_checkpoint(cp_path)

        assert len(results) == 2
        assert scored_keys == {"doc1", "doc2"}
        assert results[0].text_id == "doc1"
        assert results[1].text_id == "doc2"
        assert len(results[0].decisions) == 1
        assert results[0].decisions[0].text == "reduce budget"

    def test_load_empty_file(self, tmp_path):
        from qualitative_analysis.decisions_cli import _load_checkpoint

        cp_path = tmp_path / "empty.jsonl"
        cp_path.write_text("")

        results, scored_keys = _load_checkpoint(cp_path)
        assert results == []
        assert scored_keys == set()

    def test_load_with_malformed_lines(self, tmp_path):
        from qualitative_analysis.decisions_cli import _load_checkpoint

        cp_path = tmp_path / "mixed.jsonl"
        good = json.dumps({"text_id": "doc1", "decisions": [], "factors": []})
        cp_path.write_text(f"{good}\nnot valid json\n{good.replace('doc1', 'doc2')}\n")

        results, scored_keys = _load_checkpoint(cp_path)
        assert len(results) == 2
        assert scored_keys == {"doc1", "doc2"}


class TestCheckpointKey:
    def test_with_id_col(self):
        from qualitative_analysis.decisions_cli import _text_checkpoint_key

        row = {"text_id": "abc", "text": "some text"}
        key = _text_checkpoint_key(row, "text", "text_id", 1)
        assert key == "abc"

    def test_without_id_col(self):
        from qualitative_analysis.decisions_cli import _text_checkpoint_key

        row = {"text": "some text"}
        key = _text_checkpoint_key(row, "text", None, 5)
        assert key == "5"

    def test_id_col_not_in_row(self):
        from qualitative_analysis.decisions_cli import _text_checkpoint_key

        row = {"text": "some text"}
        key = _text_checkpoint_key(row, "text", "missing_col", 3)
        assert key == "3"


# =====================================================================
# Auto-detect CSV columns
# =====================================================================

class TestAutoDetectCSVColumns:
    def test_exact_match(self):
        text_col, id_col = auto_detect_csv_columns(
            ["text", "id", "group"],
            text_col="text",
            id_col=None,
        )
        assert text_col == "text"
        assert id_col == "id"

    def test_text_fallback_instance_text(self, capsys):
        text_col, id_col = auto_detect_csv_columns(
            ["instance_text", "type", "window_text"],
            text_col="text",
            id_col=None,
        )
        assert text_col == "instance_text"
        captured = capsys.readouterr()
        assert "Auto-detected text column" in captured.out

    def test_id_fallback_text_id(self, capsys):
        text_col, id_col = auto_detect_csv_columns(
            ["text", "text_id", "group"],
            text_col="text",
            id_col=None,
        )
        assert id_col == "text_id"
        captured = capsys.readouterr()
        assert "Auto-detected ID column" in captured.out

    def test_explicit_id_not_overridden(self):
        text_col, id_col = auto_detect_csv_columns(
            ["text", "text_id", "my_id"],
            text_col="text",
            id_col="my_id",
        )
        assert id_col == "my_id"

    def test_no_fallback_available(self):
        text_col, id_col = auto_detect_csv_columns(
            ["content_blob", "meta"],
            text_col="text",
            id_col=None,
        )
        # Should keep original since no alternatives match
        assert text_col == "text"
        assert id_col is None

    def test_content_fallback(self, capsys):
        text_col, id_col = auto_detect_csv_columns(
            ["content", "doc_id"],
            text_col="text",
            id_col=None,
        )
        assert text_col == "content"
        assert id_col == "doc_id"


# =====================================================================
# Resolve verbose flag
# =====================================================================

class TestResolveVerboseFlag:
    def test_verbose_true(self):
        import argparse
        args = argparse.Namespace(verbose=True, log_llm=False)
        assert resolve_verbose_flag(args) is True

    def test_log_llm_true(self):
        import argparse
        args = argparse.Namespace(verbose=False, log_llm=True)
        assert resolve_verbose_flag(args) is True

    def test_both_false(self):
        import argparse
        args = argparse.Namespace(verbose=False, log_llm=False)
        assert resolve_verbose_flag(args) is False

    def test_both_true(self):
        import argparse
        args = argparse.Namespace(verbose=True, log_llm=True)
        assert resolve_verbose_flag(args) is True

    def test_missing_attrs(self):
        import argparse
        args = argparse.Namespace()
        assert resolve_verbose_flag(args) is False
