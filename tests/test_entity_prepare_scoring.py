import csv

from qualitative_analysis.entity_cli import _extract_entity_contexts_from_windows


def _write_windows_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text_id", "window_text", "entities_json"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_scoring_preserves_first_seen_entity_case(tmp_path):
    windows_path = tmp_path / "windows.csv"
    _write_windows_csv(
        windows_path,
        [
            {
                "text_id": "p1",
                "window_text": "Climate Change affects farming.",
                "entities_json": '["Climate Change"]',
            },
            {
                "text_id": "p1",
                "window_text": "climate change also affects flooding.",
                "entities_json": '["climate change"]',
            },
        ],
    )

    rows = _extract_entity_contexts_from_windows(windows_path)

    assert rows == [
        {
            "entity": "Climate Change",
            "context": "Climate Change affects farming.",
            "text_id": "p1",
        }
    ]
