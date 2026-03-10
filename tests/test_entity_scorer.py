import json

import pytest

from qualitative_analysis.entity.models import (
    DimensionDefinition,
    DimensionScore,
    EntityScore,
    EntityScoreResult,
    SingleRunScore,
)
from qualitative_analysis.entity.scorer import EntityScorer


class MockLLMProvider:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    async def generate(self, prompt="", system_prompt="", temperature=0.3):
        response = self.responses[self.call_count]
        self.call_count += 1
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def dimensions():
    return [
        DimensionDefinition(name="Social", description="Social factors"),
        DimensionDefinition(name="Ecological", description="Ecological factors"),
    ]


def test_parse_response_accepts_fenced_json(dimensions):
    scorer = EntityScorer()
    response = """```json
    {
      "initial_observations": "Observed social emphasis",
      "dimension_scores": [
        {"dimension": "Social", "score": 75, "justification": "Mostly social"},
        {"dimension": "Ecological", "score": 40, "justification": "Some eco relevance"}
      ]
    }
    ```"""

    parsed = scorer._parse_response(response, dimensions)

    assert parsed["initial_observations"] == "Observed social emphasis"
    assert parsed["dimension_scores"]["social"]["score"] == 75
    assert parsed["dimension_scores"]["ecological"]["justification"] == "Some eco relevance"


def test_parse_response_accepts_embedded_json(dimensions):
    scorer = EntityScorer()
    response = (
        'Here is the analysis: {"dimension_scores": ['
        '{"dimension": "Social", "score": 80, "justification": "clear"}, '
        '{"dimension": "Ecological", "score": 35, "justification": "limited"}'
        ']} Thanks.'
    )

    parsed = scorer._parse_response(response, dimensions)

    assert parsed["dimension_scores"]["social"]["score"] == 80
    assert parsed["dimension_scores"]["ecological"]["score"] == 35


def test_parse_response_rejects_missing_expected_dimensions(dimensions):
    scorer = EntityScorer()
    response = json.dumps(
        {
            "dimension_scores": [
                {"dimension": "Social", "score": 70, "justification": "present"},
            ]
        }
    )

    with pytest.raises(ValueError, match="Missing dimensions") as exc:
        scorer._parse_response(response, dimensions)

    assert "ecological" in str(exc.value).lower()
    assert "expected=" in str(exc.value)


def test_parse_response_accepts_v4_style_output_without_reasoning(dimensions):
    scorer = EntityScorer(prompt_version="v4")
    response = json.dumps(
        {
            "dimension_scores": [
                {"dimension": "Social", "score": 88},
                {"dimension": "Ecological", "score": 12},
            ]
        }
    )

    parsed = scorer._parse_response(response, dimensions)

    assert parsed["initial_observations"] == ""
    assert parsed["dimension_scores"]["social"]["justification"] == ""
    assert parsed["dimension_scores"]["ecological"]["score"] == 12


def test_parse_response_rejects_non_integer_scores(dimensions):
    scorer = EntityScorer()
    response = json.dumps(
        {
            "dimension_scores": [
                {"dimension": "Social", "score": 74.9, "justification": "too precise"},
                {"dimension": "Ecological", "score": 20, "justification": "ok"},
            ]
        }
    )

    with pytest.raises(ValueError, match="non-integer score"):
        scorer._parse_response(response, dimensions)


def test_aggregate_runs_selects_justification_closest_to_mean(dimensions):
    scorer = EntityScorer()
    runs = [
        SingleRunScore(1, {"social": {"score": 10, "justification": "low"}}),
        SingleRunScore(2, {"social": {"score": 20, "justification": "middle"}}),
        SingleRunScore(3, {"social": {"score": 30, "justification": "high"}}),
    ]

    aggregated = scorer._aggregate_runs(runs, dimensions[:1])

    assert aggregated["social"].mean == 20
    assert aggregated["social"].justification == "middle"


@pytest.mark.asyncio
async def test_score_entity_preserves_partial_success_and_metadata(dimensions):
    scorer = EntityScorer()
    provider = MockLLMProvider(
        responses=[
            json.dumps(
                {
                    "dimension_scores": [
                        {"dimension": "Social", "score": 80, "justification": "first"},
                        {"dimension": "Ecological", "score": 30, "justification": "first eco"},
                    ]
                }
            ),
            RuntimeError("transient failure"),
            json.dumps(
                {
                    "dimension_scores": [
                        {"dimension": "Social", "score": 70, "justification": "second"},
                        {"dimension": "Ecological", "score": 40, "justification": "second eco"},
                    ]
                }
            ),
        ]
    )

    score = await scorer.score_entity(
        entity="river",
        context="river supports nearby farming",
        dimensions=dimensions,
        llm_provider=provider,
        num_runs=3,
        text_id="doc-1",
        window_index=4,
        group="A",
    )

    assert score.num_runs == 2
    assert len(score.runs) == 2
    assert score.text_id == "doc-1"
    assert score.window_index == 4
    assert score.group == "A"
    assert score.dimension_scores["social"].scores == [80, 70]


@pytest.mark.asyncio
async def test_score_entities_counts_errors_and_invokes_callbacks(dimensions):
    scorer = EntityScorer()
    provider = MockLLMProvider(
        responses=[
            json.dumps(
                {
                    "dimension_scores": [
                        {"dimension": "Social", "score": 65, "justification": "ok"},
                        {"dimension": "Ecological", "score": 45, "justification": "ok"},
                    ]
                }
            ),
            RuntimeError("complete failure"),
        ]
    )
    progress_calls = []
    scored_entities = []
    run_calls = []

    result = await scorer.score_entities(
        entities=[
            {"entity": "bridge", "context": "bridge over river", "text_id": "t1", "window_index": 1, "group": "G1"},
            {"entity": "levee", "context": "levee near river", "text_id": "t2", "window_index": 2, "group": "G2"},
        ],
        dimensions=dimensions,
        llm_provider=provider,
        num_runs=1,
        on_progress=lambda current, total, entity: progress_calls.append((current, total, entity)),
        on_entity_scored=lambda score: scored_entities.append(score.entity),
        on_run_progress=lambda entity_idx, total_entities, entity_name, run_num, total_runs: run_calls.append(
            (entity_idx, total_entities, entity_name, run_num, total_runs)
        ),
    )

    assert result.config["errors"] == 1
    assert len(result.scores) == 1
    assert result.scores[0].entity == "bridge"
    assert result.scores[0].text_id == "t1"
    assert scored_entities == ["bridge"]
    assert progress_calls[-1] == (2, 2, "levee")
    assert run_calls == [(1, 2, "bridge", 1, 1), (2, 2, "levee", 1, 1)]


def test_save_load_round_trip_preserves_mode_metadata_and_runs(tmp_path, dimensions):
    scorer = EntityScorer()
    runs = [
        SingleRunScore(
            run_number=1,
            dimension_scores={
                "social": {"score": 75, "justification": "first social"},
                "ecological": {"score": 35, "justification": "first eco"},
            },
            initial_observations="obs 1",
            raw_response="raw 1",
            processing_time_ms=12.5,
        ),
        SingleRunScore(
            run_number=2,
            dimension_scores={
                "social": {"score": 80, "justification": "second social"},
                "ecological": {"score": 45, "justification": "second eco"},
            },
            initial_observations="obs 2",
            raw_response="raw 2",
            processing_time_ms=13.5,
        ),
    ]
    entity_score = EntityScore(
        entity="wetland",
        text_id="doc-9",
        context="wetland restoration discussion",
        dimension_scores={
            "social": DimensionScore.from_scores("social", [75, 80], justification="second social"),
            "ecological": DimensionScore.from_scores("ecological", [35, 45], justification="second eco"),
        },
        runs=runs,
        num_runs=2,
        processing_time_ms=55.0,
        window_index=7,
        group="coastal",
    )
    result = EntityScoreResult(scores=[entity_score], dimensions=dimensions)

    path = tmp_path / "scores.json"
    scorer.save(result, path)
    loaded = scorer.load(path)
    loaded_score = loaded.scores[0]

    assert loaded_score.window_index == 7
    assert loaded_score.group == "coastal"
    assert loaded_score.dimension_scores["social"].mode == entity_score.dimension_scores["social"].mode
    assert len(loaded_score.runs) == 2
    assert loaded_score.runs[0].initial_observations == "obs 1"
    assert loaded_score.runs[1].dimension_scores["ecological"]["score"] == 45