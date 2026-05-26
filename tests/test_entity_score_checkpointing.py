import argparse
import json

from qualitative_analysis.entity.models import DimensionDefinition, DimensionScore, EntityScore
from qualitative_analysis.entity_cli import (
    _apply_scoring_scale_overrides,
    _apply_participant_filter,
    _append_to_checkpoint,
    _entity_checkpoint_key,
    _load_checkpoint,
    _read_scored_entities_csv,
    _resolve_scale,
)
from qualitative_analysis.entity.comparison import ParticipantComparison


def test_checkpoint_round_trip_preserves_metadata_and_run_scores(tmp_path):
    dimensions = [DimensionDefinition(name="Social", description="Social factors")]
    score = EntityScore(
        entity="river",
        text_id="p01",
        context="river management discussion",
        dimension_scores={
            "social": DimensionScore.from_scores("social", [60, 70], justification="balanced")
        },
        num_runs=2,
        processing_time_ms=14.2,
        window_index=3,
        group="delta",
    )
    checkpoint_path = tmp_path / "scores_checkpoint.jsonl"

    _append_to_checkpoint(checkpoint_path, score)
    loaded_scores, scored_keys = _load_checkpoint(checkpoint_path, dimensions)

    assert len(loaded_scores) == 1
    loaded = loaded_scores[0]
    assert loaded.window_index == 3
    assert loaded.group == "delta"
    assert loaded.dimension_scores["social"].scores == [60, 70]
    assert loaded.dimension_scores["social"].mode == 65.0
    assert _entity_checkpoint_key(score.to_flat_dict()) in scored_keys


def test_load_checkpoint_skips_malformed_lines(tmp_path):
    dimensions = [DimensionDefinition(name="Social", description="Social factors")]
    checkpoint_path = tmp_path / "scores_checkpoint.jsonl"
    valid_line = json.dumps(
        {
            "entity": "forest",
            "text_id": "p02",
            "context": "forest governance",
            "num_runs": 1,
            "social_run1": 55,
            "social_justification": "present",
        }
    )
    checkpoint_path.write_text("not-json\n" + valid_line + "\n", encoding="utf-8")

    loaded_scores, scored_keys = _load_checkpoint(checkpoint_path, dimensions)

    assert len(loaded_scores) == 1
    assert loaded_scores[0].entity == "forest"
    assert _entity_checkpoint_key({"entity": "forest", "text_id": "p02", "window_index": None}) in scored_keys


def test_checkpoint_keys_support_resume_filtering():
    entities = [
        {"entity": "river", "text_id": "p01", "window_index": 1},
        {"entity": "forest", "text_id": "p01", "window_index": 2},
    ]
    scored_keys = {_entity_checkpoint_key(entities[0])}

    remaining = [e for e in entities if _entity_checkpoint_key(e) not in scored_keys]

    assert remaining == [entities[1]]
    assert _entity_checkpoint_key(entities[0]) == "river|p01|1"


def test_scoring_scale_keeps_dimension_defaults_without_cli_override(tmp_path):
    (tmp_path / "score_metadata.json").write_text(
        json.dumps({"scale_min": 10, "scale_max": 20}),
        encoding="utf-8",
    )
    dimensions = [
        DimensionDefinition(name="Social", description="Social factors", scale_min=1, scale_max=7)
    ]

    resolved = _apply_scoring_scale_overrides(
        dimensions,
        argparse.Namespace(scale_min=None, scale_max=None),
    )

    assert resolved[0].scale_min == 1
    assert resolved[0].scale_max == 7


def test_scoring_scale_cli_overrides_replace_dimension_bounds():
    dimensions = [
        DimensionDefinition(name="Social", description="Social factors", scale_min=0, scale_max=100),
        DimensionDefinition(name="Ecological", description="Ecological factors", scale_min=0, scale_max=100),
    ]

    resolved = _apply_scoring_scale_overrides(
        dimensions,
        argparse.Namespace(scale_min=1, scale_max=10),
    )

    assert [(dim.scale_min, dim.scale_max) for dim in resolved] == [(1, 10), (1, 10)]


def test_downstream_scale_resolution_still_reads_metadata_and_respects_cli_override(tmp_path):
    (tmp_path / "score_metadata.json").write_text(
        json.dumps({"scale_min": 10, "scale_max": 20}),
        encoding="utf-8",
    )

    scale = _resolve_scale(
        argparse.Namespace(scale_min=5, scale_max=None),
        tmp_path,
    )

    assert scale.scale_min == 5
    assert scale.scale_max == 20


def test_read_scored_entities_csv_raises_on_missing_dimension_values(tmp_path):
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text(
        "entity,text_id,social_mean,ecological_mean\nriver,p01,,55\n",
        encoding="utf-8",
    )

    try:
        _read_scored_entities_csv(csv_path, ["social", "ecological"])
    except ValueError as exc:
        assert "Invalid score data" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing score data")


def test_apply_participant_filter_returns_subset_in_requested_order():
    comparison = ParticipantComparison()
    comparison.scores_by_participant = {
        "p1": [1],
        "p2": [2],
        "p3": [3],
    }
    comparison.entity_names_by_participant = {
        "p1": ["e1"],
        "p2": ["e1"],
        "p3": ["e1"],
    }
    comparison.dimension_names = ["social"]
    comparison.entity_names = ["e1"]

    filtered, participants = _apply_participant_filter(comparison, "p3,p1")

    assert participants == ["p3", "p1"]
    assert list(filtered.scores_by_participant.keys()) == ["p3", "p1"]
