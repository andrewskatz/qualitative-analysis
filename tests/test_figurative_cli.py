import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qualitative_analysis.cli import run_figurative_detect
from qualitative_analysis.figurative.models import DetectionResult, Instance


class TestFigurativeCLI(unittest.IsolatedAsyncioTestCase):
    async def test_instances_csv_includes_provenance_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_csv = tmp_path / "input.csv"
            output_dir = tmp_path / "output"

            with input_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["text_id", "text"])
                writer.writeheader()
                writer.writerow({"text_id": "row-1", "text": "Before time is a thief after."})

            captured_kwargs = {}

            class StubDetector:
                def __init__(self, *args, **kwargs):
                    captured_kwargs.update(kwargs)

                async def detect(self, text: str) -> DetectionResult:
                    return DetectionResult(
                        contains_figurative=True,
                        confidence=0.91,
                        instances=[
                            Instance(
                                text="time is a thief",
                                type="metaphor",
                                confidence=0.91,
                                explanation="Time is framed as stealing.",
                                context_dependent=False,
                                window_index=0,
                                start_char=7,
                                end_char=23,
                                alignment_status="normalized_exact",
                                support_count=2,
                                supporting_window_indices=[0, 1],
                            )
                        ],
                        metadata={
                            "window_count": 2,
                            "strategy": "two_step_with_summaries",
                            "windows": [
                                {
                                    "window_index": 0,
                                    "window_text": "Before time is a thief after.",
                                }
                            ],
                        },
                    )

            args = argparse.Namespace(
                input_csv=str(input_csv),
                id_col="text_id",
                text_col="text",
                output="instances",
                output_dir=str(output_dir),
                model="mock-model",
                base_url="http://localhost:11434",
                log_llm=False,
                timeout=60.0,
                window_size=3,
                stride=2,
                chunk_unit="sentences",
                tokenizer="cl100k_base",
                no_windowing=False,
                threshold=0.5,
                context_window=5,
                prompt_version=1,
                types=None,
                checkpoint=None,
                checkpoint_interval=10,
            )

            with patch("qualitative_analysis.cli.FigurativeDetector", StubDetector):
                exit_code = await run_figurative_detect(args)

            self.assertEqual(exit_code, 0)
            self.assertTrue(captured_kwargs["return_windows"])

            run_dirs = list(output_dir.glob("run_*"))
            self.assertEqual(len(run_dirs), 1)
            instance_files = list(run_dirs[0].glob("*_instances_*.csv"))
            self.assertEqual(len(instance_files), 1)

            with instance_files[0].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    [
                        "text_id",
                        "window_index",
                        "window_text",
                        "instance_text",
                        "type",
                        "confidence",
                        "explanation",
                        "context_dependent",
                        "start_char",
                        "end_char",
                        "alignment_status",
                        "support_count",
                        "supporting_window_indices",
                    ],
                )
                row = next(reader)

            self.assertEqual(row["text_id"], "row-1")
            self.assertEqual(row["window_text"], "Before time is a thief after.")
            self.assertEqual(row["alignment_status"], "normalized_exact")
            self.assertEqual(row["support_count"], "2")
            self.assertEqual(row["supporting_window_indices"], json.dumps([0, 1]))